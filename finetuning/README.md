## Fine Tuning Qwen3-TTS-12Hz-1.7B/0.6B-Base

The Qwen3-TTS-12Hz-1.7B/0.6B-Base model series currently supports single-speaker fine-tuning. Please run `pip install qwen-tts` first, then run the command below:

```
git clone https://github.com/QwenLM/Qwen3-TTS.git
cd Qwen3-TTS/finetuning
```

Then follow the steps below to complete the entire fine-tuning workflow. Multi-speaker fine-tuning and other advanced fine-tuning features will be supported in future releases.

### 1) Input JSONL format

Prepare your training file as a JSONL (one JSON object per line). Each line must contain:

- `audio`: path to the target training audio (wav)
- `text`: transcript corresponding to `audio`
- `ref_audio`: path to the reference speaker audio (wav)

Example:
```jsonl
{"audio":"./data/utt0001.wav","text":"其实我真的有发现，我是一个特别善于观察别人情绪的人。","ref_audio":"./data/ref.wav"}
{"audio":"./data/utt0002.wav","text":"She said she would be here by noon.","ref_audio":"./data/ref.wav"}
```

`ref_audio` recommendation:
- Strongly recommended: use the same `ref_audio` for all samples.
- Keeping `ref_audio` identical across the dataset usually improves speaker consistency and stability during generation.


### 2) Prepare data (extract `audio_codes`)

Convert `train_raw.jsonl` into a training JSONL that includes `audio_codes`:

```bash
python prepare_data.py \
  --device cuda:0 \
  --tokenizer_model_path Qwen/Qwen3-TTS-Tokenizer-12Hz \
  --input_jsonl train_raw.jsonl \
  --output_jsonl train_with_codes.jsonl
```


### 3) Fine-tune

Run SFT using the prepared JSONL:

```bash
python sft_12hz.py \
  --init_model_path Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --output_model_path output \
  --train_jsonl train_with_codes.jsonl \
  --batch_size 32 \
  --lr 2e-6 \
  --num_epochs 10 \
  --speaker_name speaker_test
```

Checkpoints will be written to:
- `output/checkpoint-epoch-0`
- `output/checkpoint-epoch-1`
- `output/checkpoint-epoch-2`
- ...


### 4) Quick inference test

```python
import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

device = "cuda:0"
tts = Qwen3TTSModel.from_pretrained(
    "output/checkpoint-epoch-2",
    device_map=device,
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)

wavs, sr = tts.generate_custom_voice(
    text="She said she would be here by noon.",
    speaker="speaker_test",
)
sf.write("output.wav", wavs[0], sr)
```

### One-click shell script example

```bash
#!/usr/bin/env bash
set -e

DEVICE="cuda:0"
TOKENIZER_MODEL_PATH="Qwen/Qwen3-TTS-Tokenizer-12Hz"
INIT_MODEL_PATH="Qwen/Qwen3-TTS-12Hz-1.7B-Base"

RAW_JSONL="train_raw.jsonl"
TRAIN_JSONL="train_with_codes.jsonl"
OUTPUT_DIR="output"

BATCH_SIZE=2
LR=2e-5
EPOCHS=3
SPEAKER_NAME="speaker_1"

python prepare_data.py \
  --device ${DEVICE} \
  --tokenizer_model_path ${TOKENIZER_MODEL_PATH} \
  --input_jsonl ${RAW_JSONL} \
  --output_jsonl ${TRAIN_JSONL}

python sft_12hz.py \
  --init_model_path ${INIT_MODEL_PATH} \
  --output_model_path ${OUTPUT_DIR} \
  --train_jsonl ${TRAIN_JSONL} \
  --batch_size ${BATCH_SIZE} \
  --lr ${LR} \
  --num_epochs ${EPOCHS} \
  --speaker_name ${SPEAKER_NAME}
```