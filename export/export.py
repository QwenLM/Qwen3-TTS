import argparse
import gc
import shutil
import tempfile
from pathlib import Path


def export_checkpoint(model_path, output_dir, adapter_path=None, dtype="float32",
                      device="cpu", max_shard_size="5GB", verify=False):
    import torch
    from qwen_tts import Qwen3TTSModel

    source = Path(model_path).resolve()
    output = Path(output_dir).resolve()
    adapter = Path(adapter_path).resolve() if adapter_path else None
    if not source.is_dir():
        raise ValueError("--model_path must be a local model directory")
    if output.exists():
        raise ValueError("Output already exists; choose a new directory")
    for input_dir in (source, adapter):
        if input_dir is not None and (output.is_relative_to(input_dir) or input_dir.is_relative_to(output)):
            raise ValueError("Output and input directories must not contain one another")
    for asset in ("speech_tokenizer/config.json", "generation_config.json"):
        if not (source / asset).is_file():
            raise ValueError(f"Missing serving asset: {source / asset}")
    if adapter is not None and not (adapter / "adapter_config.json").is_file():
        raise ValueError("--adapter_path must contain a PEFT adapter checkpoint")
    if dtype not in ("float32", "float16", "bfloat16"):
        raise ValueError("Unsupported export dtype")

    wrapper = Qwen3TTSModel.from_pretrained(
        str(source), device_map=device, dtype=getattr(torch, dtype),
        attn_implementation="eager",
    )
    model = wrapper.model
    if adapter is not None:
        from peft import PeftModel
        if __package__:
            from .lora import apply_speaker, load_speaker
        else:
            from lora import apply_speaker, load_speaker

        if model.config.tts_model_type != "base":
            raise ValueError("Adapter export requires the original Base checkpoint")
        speaker_name, embedding = load_speaker(adapter)
        model = PeftModel.from_pretrained(model, str(adapter)).merge_and_unload(safe_merge=True)
        apply_speaker(model, speaker_name, embedding)

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temporary:
        staging = Path(temporary) / "model"
        model.save_pretrained(staging, safe_serialization=True, max_shard_size=max_shard_size)
        wrapper.processor.save_pretrained(staging)
        shutil.copytree(source / "speech_tokenizer", staging / "speech_tokenizer",
                        ignore=shutil.ignore_patterns(".cache", "__pycache__"))
        shutil.copy2(source / "generation_config.json", staging / "generation_config.json")
        if verify:
            del model, wrapper
            gc.collect()
            reloaded = Qwen3TTSModel.from_pretrained(
                str(staging), device_map="cpu", dtype=getattr(torch, dtype),
                attn_implementation="eager", local_files_only=True,
            )
            del reloaded
        staging.rename(output)
    return output


def main():
    parser = argparse.ArgumentParser(description="Export a standalone toolkit checkpoint, optionally merging LoRA.")
    parser.add_argument("--model_path", required=True, help="Full checkpoint, or the exact Base weights used for LoRA.")
    parser.add_argument("--adapter_path", help="LoRA checkpoint including speaker.json and speaker_embedding.safetensors.")
    parser.add_argument("--output_dir", required=True, help="New output directory; never overwritten.")
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max_shard_size", default="5GB")
    parser.add_argument("--verify", action="store_true", help="Reload the export locally before publishing it.")
    args = parser.parse_args()
    output = export_checkpoint(**vars(args))
    print(f"Exported standalone model to {output}")


if __name__ == "__main__":
    main()