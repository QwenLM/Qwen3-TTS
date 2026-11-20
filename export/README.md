## Exporting Fine-tuned Qwen3-TTS-12Hz Checkpoints

Two scripts for checkpoints produced by the [fine-tuning workflow](../finetuning/README.md). Please run `pip install qwen-tts` first, then run the commands below from this folder:

```
git clone https://github.com/QwenLM/Qwen3-TTS.git
cd Qwen3-TTS/export
```

### 1) Export a standalone checkpoint

Repackages a full fine-tuning checkpoint as a standalone checkpoint (sharded Safetensors, tokenizer files, generation config and speech tokenizer):

```bash
python export.py \
  --model_path ../finetuning/output/checkpoint-epoch-2 \
  --output_dir serving_model \
  --dtype bfloat16 \
  --verify
```

- `--output_dir` must not exist yet; it is never overwritten.
- `--dtype` is `float32` (default), `float16` or `bfloat16`. The export runs on CPU unless you pass `--device cuda:0`.
- `--verify` reloads the exported model before publishing it. It checks that the model loads, not the speech quality.

The result loads like any other checkpoint with `Qwen3TTSModel.from_pretrained("serving_model")`.

### 2) Export ONNX components

Exports a 12Hz CustomVoice checkpoint (for example the output of step 1) as five ONNX graphs: text embeddings, codec embeddings, talker, code predictor and audio decoder.

```bash
pip install onnx==1.17.0 "torch>=2.6,<2.8"

python export_onnx.py \
  --model_path serving_model \
  --output_dir onnx_model \
  --verify
```

- The export runs on CPU in FP32 (opset 17). No GPU is needed.
- Keep each `.onnx.data` file next to its `.onnx` file. `manifest.json` lists the graph inputs and outputs.
- `--verify` compares each graph against PyTorch with ONNX Runtime.
- This is not an end-to-end text-to-waveform graph: sampling and the autoregressive loop have to be implemented by the caller, following `Qwen3TTSForConditionalGeneration.generate`. There is no KV cache, so generation is slower than PyTorch.
- Base, VoiceDesign, 25Hz and quantized models are not supported, and only tiny test models have been validated so far.
