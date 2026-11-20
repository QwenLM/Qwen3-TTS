import argparse
import json
import tempfile
from pathlib import Path

import torch
from torch import nn


def causal_mask(attention_mask, dtype, window=None):
    positions = torch.arange(attention_mask.shape[1], device=attention_mask.device)
    allowed = positions.unsqueeze(0) <= positions.unsqueeze(1)
    if window is not None:
        allowed = allowed & (positions.unsqueeze(0) > positions.unsqueeze(1) - window)
    allowed = allowed.unsqueeze(0).unsqueeze(0) & attention_mask[:, None, None, :].bool()
    return torch.where(allowed, 0.0, torch.finfo(dtype).min).to(dtype)


class TalkerGraph(nn.Module):
    def __init__(self, talker):
        super().__init__()
        self.decoder = talker.model
        self.head = talker.codec_head

    def forward(self, inputs_embeds, attention_mask):
        position_ids = (attention_mask.cumsum(-1) - 1).clamp(min=0)
        hidden = self.decoder(
            inputs_embeds=inputs_embeds,
            attention_mask=causal_mask(attention_mask, inputs_embeds.dtype, self.decoder.config.sliding_window),
            position_ids=position_ids, use_cache=False, return_dict=False,
        )[0]
        return self.head(hidden[:, -1]), hidden[:, -1]


class TextEmbeddingGraph(nn.Module):
    def __init__(self, talker):
        super().__init__()
        self.embedding = talker.get_text_embeddings()
        self.projection = talker.text_projection

    def forward(self, token_ids):
        return self.projection(self.embedding(token_ids))


class CodecEmbeddingGraph(nn.Module):
    def __init__(self, talker):
        super().__init__()
        self.embeddings = nn.ModuleList(
            [talker.get_input_embeddings(), *talker.code_predictor.get_input_embeddings()]
        )

    def forward(self, codec_ids):
        return torch.stack([embedding(codec_ids[:, :, group])
                            for group, embedding in enumerate(self.embeddings)], dim=2)


class CodePredictorGraph(nn.Module):
    def __init__(self, predictor):
        super().__init__()
        self.decoder = predictor.model
        self.projection = predictor.small_to_mtp_projection
        self.register_buffer("head_weights", torch.stack([head.weight.detach() for head in predictor.lm_head]))

    def forward(self, inputs_embeds, head_index):
        projected = self.projection(inputs_embeds)
        mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=inputs_embeds.device)
        masks = {"full_attention": causal_mask(mask, projected.dtype),
                 "sliding_attention": causal_mask(mask, projected.dtype, self.decoder.config.sliding_window)}
        hidden = self.decoder(inputs_embeds=projected, attention_mask=masks,
                              use_cache=False, return_dict=False)[0]
        weight = torch.index_select(self.head_weights, 0, head_index.reshape(1)).squeeze(0)
        return torch.nn.functional.linear(hidden[:, -1], weight)


class AudioDecoderGraph(nn.Module):
    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder

    def forward(self, audio_codes):
        hidden = self.decoder.quantizer.decode(audio_codes.transpose(1, 2))
        hidden = self.decoder.pre_conv(hidden).transpose(1, 2)
        mask = torch.ones(hidden.shape[:2], dtype=torch.long, device=hidden.device)
        masks = {"full_attention": causal_mask(mask, hidden.dtype),
                 "sliding_attention": causal_mask(mask, hidden.dtype, self.decoder.config.sliding_window)}
        hidden = self.decoder.pre_transformer(inputs_embeds=hidden, attention_mask=masks,
                                             use_cache=False).last_hidden_state
        hidden = hidden.transpose(1, 2)
        for blocks in self.decoder.upsample:
            for block in blocks:
                hidden = block(hidden)
        for block in self.decoder.decoder:
            hidden = block(hidden)
        return hidden.clamp(min=-1, max=1).squeeze(1)


def export_graph(module, inputs, path, input_names, output_names, dynamic_axes, verify=False):
    import onnx

    path = Path(path)
    module.eval()
    with tempfile.TemporaryDirectory(prefix=f".{path.stem}-", dir=path.parent) as temporary:
        intermediate = Path(temporary) / path.name
        with torch.inference_mode():
            torch.onnx.export(
                module, inputs, str(intermediate), input_names=input_names, output_names=output_names,
                dynamic_axes=dynamic_axes, opset_version=17, dynamo=False, external_data=True,
            )
        graph = onnx.load(str(intermediate))
        onnx.save_model(graph, str(path), save_as_external_data=True, all_tensors_to_one_file=True,
                        location=path.name + ".data", size_threshold=1024)
        del graph
    onnx.checker.check_model(str(path))
    if verify:
        verify_graph(module, inputs, path, input_names)


def verify_graph(module, inputs, path, input_names):
    import numpy as np
    import onnxruntime as ort

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    feed = {name: value.detach().cpu().numpy() for name, value in zip(input_names, inputs)}
    actual = session.run(None, feed)
    with torch.inference_mode():
        expected = module(*inputs)
    if isinstance(expected, torch.Tensor):
        expected = (expected,)
    if len(expected) != len(actual):
        raise ValueError("ONNX graph output count differs from PyTorch")
    for reference, result in zip(expected, actual):
        np.testing.assert_allclose(result, reference.detach().cpu().numpy(), rtol=2e-3, atol=2e-4)


def graph_signature(path):
    import onnx

    graph = onnx.load(str(path), load_external_data=False).graph

    def describe(value):
        tensor = value.type.tensor_type
        return {"name": value.name, "dtype": onnx.TensorProto.DataType.Name(tensor.elem_type),
                "shape": [dimension.dim_param or dimension.dim_value for dimension in tensor.shape.dim]}

    return {"file": Path(path).name, "inputs": [describe(value) for value in graph.input],
            "outputs": [describe(value) for value in graph.output]}


def export_bundle(wrapper, output_dir, verify=False):
    output = Path(output_dir).resolve()
    if output.exists():
        raise ValueError("Output already exists; choose a new directory")
    model = wrapper.model
    if model.config.tts_model_type != "custom_voice":
        raise ValueError("ONNX export supports CustomVoice only; first export/merge your fine-tune to CustomVoice")
    tokenizer = model.speech_tokenizer.model
    if tokenizer.config.model_type != "qwen3_tts_tokenizer_12hz":
        raise ValueError("ONNX export requires the 12Hz speech tokenizer")
    model.cpu().float().eval()
    tokenizer.cpu().float().eval()
    groups = model.talker.config.num_code_groups
    if groups != tokenizer.decoder.config.num_quantizers:
        raise ValueError("Talker and audio decoder code-group counts differ")
    hidden_size = model.talker.config.hidden_size
    sequence_axes = {0: "batch", 1: "sequence"}
    mask = torch.ones(2, 7, dtype=torch.long)
    mask[0, :2] = 0
    specs = [
        ("text_embeddings", TextEmbeddingGraph(model.talker),
         (torch.zeros(1, 4, dtype=torch.long),), (torch.ones(2, 7, dtype=torch.long),),
         ["token_ids"], ["embeddings"], {"token_ids": sequence_axes, "embeddings": sequence_axes}),
        ("codec_embeddings", CodecEmbeddingGraph(model.talker),
         (torch.zeros(1, 4, groups, dtype=torch.long),), (torch.ones(2, 7, groups, dtype=torch.long),),
         ["codec_ids"], ["embeddings"], {"codec_ids": sequence_axes, "embeddings": sequence_axes}),
        ("talker", TalkerGraph(model.talker),
         (torch.randn(1, 4, hidden_size), torch.ones(1, 4, dtype=torch.long)),
         (torch.randn(2, 7, hidden_size), mask),
         ["inputs_embeds", "attention_mask"], ["logits", "hidden_state"],
         {"inputs_embeds": sequence_axes, "attention_mask": sequence_axes,
          "logits": {0: "batch"}, "hidden_state": {0: "batch"}}),
        ("code_predictor", CodePredictorGraph(model.talker.code_predictor),
         (torch.randn(1, 2, hidden_size), torch.tensor(0)),
         (torch.randn(2, 4, hidden_size), torch.tensor(2)),
         ["inputs_embeds", "head_index"], ["logits"],
         {"inputs_embeds": {0: "batch", 1: "prefix"}, "logits": {0: "batch"}}),
        ("audio_decoder", AudioDecoderGraph(tokenizer.decoder),
         (torch.zeros(1, 4, groups, dtype=torch.long),),
         (torch.randint(tokenizer.decoder.config.codebook_size, (1, 9, groups)),),
         ["audio_codes"], ["waveform"],
         {"audio_codes": {0: "batch", 1: "frames"}, "waveform": {0: "batch", 1: "samples"}}),
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temporary:
        staging = Path(temporary) / "bundle"
        staging.mkdir()
        signatures = {}
        for name, graph, inputs, alternate, input_names, output_names, axes in specs:
            print(f"Exporting {name}...", flush=True)
            path = staging / f"{name}.onnx"
            export_graph(graph, inputs, path, input_names, output_names, axes, verify=verify)
            if verify:
                verify_graph(graph, alternate, path, input_names)
            signatures[name] = graph_signature(path)
        model.config.save_pretrained(staging)
        wrapper.processor.save_pretrained(staging)
        (staging / "generation_config.json").write_text(
            json.dumps(wrapper.generate_defaults, indent=2) + "\n", encoding="utf-8"
        )
        manifest = {
            "format_version": 1, "model_type": "qwen3_tts_12hz_custom_voice",
            "opset": 17, "precision": "float32", "kv_cache": False,
            "num_code_groups": groups, "sample_rate": tokenizer.config.output_sample_rate,
            "samples_per_frame": tokenizer.config.decode_upsample_rate,
            "onnxruntime_verified": verify, "graphs": signatures,
            "host_operations": ["text tokenization", "prompt/embedding assembly", "autoregressive loop",
                                "sampling and EOS handling", "audio chunking with left context"],
        }
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        staging.rename(output)
    return output


def main():
    parser = argparse.ArgumentParser(description="Export a 12Hz CustomVoice checkpoint as ONNX inference components.")
    parser.add_argument("--model_path", required=True, help="Local CustomVoice checkpoint; merge LoRA first.")
    parser.add_argument("--output_dir", required=True, help="New ONNX bundle directory.")
    parser.add_argument("--verify", action="store_true", help="Check CPU ONNX Runtime parity at two shapes per graph.")
    args = parser.parse_args()
    source = Path(args.model_path).resolve()
    output = Path(args.output_dir).resolve()
    if not source.is_dir():
        parser.error("--model_path must be a local checkpoint directory")
    if output.exists() or output.is_relative_to(source) or source.is_relative_to(output):
        parser.error("Output must be new and must not contain, or be inside, the source directory")
    from qwen_tts import Qwen3TTSModel

    wrapper = Qwen3TTSModel.from_pretrained(str(source), device_map="cpu", dtype=torch.float32,
                                          attn_implementation="eager", local_files_only=True)
    exported = export_bundle(wrapper, output, verify=args.verify)
    print(f"Exported ONNX components to {exported}; see manifest.json for graph signatures and host operations.")


if __name__ == "__main__":
    main()