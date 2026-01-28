# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import io
import requests
import soundfile as sf

from qwen_tts import Qwen3TTSTokenizer
from qwen_tts.core.device_utils import get_optimal_device, get_device_info, get_model_path

audio_1 = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-TTS-Repo/tokenizer_demo_1.wav"
audio_2 = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-TTS-Repo/tokenizer_demo_2.wav"

# Auto-detect device
device = get_optimal_device()
print(f"Using device: {get_device_info(device)}\n")

# Use local model if available, otherwise download from HuggingFace
tokenizer_path = get_model_path("Qwen/Qwen3-TTS-Tokenizer-12Hz")

# -------- Single input: wav path --------
tokenizer_12hz = Qwen3TTSTokenizer.from_pretrained(
    tokenizer_path,
    device_map=device,
)

enc1 = tokenizer_12hz.encode(audio_1)
wavs1, out_sr1 = tokenizer_12hz.decode(enc1)
sf.write("decoded_single_12hz.wav", wavs1[0], out_sr1)

# -------- Batch input: wav path list --------
enc2 = tokenizer_12hz.encode([audio_1, audio_2])
wavs2, out_sr2 = tokenizer_12hz.decode(enc2)
for i, w in enumerate(wavs2):
    sf.write(f"decoded_batch_12hz_{i}.wav", w, out_sr2)

# -------- Decode input as dict (12hz) --------
# Take the first sample codes and pass as a dict.
dict_input_12hz = {"audio_codes": enc2.audio_codes[0]}  # torch.Tensor
wavs_d1, out_sr_d1 = tokenizer_12hz.decode(dict_input_12hz)
sf.write("decoded_dict_12hz.wav", wavs_d1[0], out_sr_d1)

# -------- Decode input as list[dict] (12hz) --------
list_dict_input_12hz = [{"audio_codes": c} for c in enc2.audio_codes]  # list of torch.Tensor
wavs_d2, out_sr_d2 = tokenizer_12hz.decode(list_dict_input_12hz)
for i, w in enumerate(wavs_d2):
    sf.write(f"decoded_listdict_12hz_{i}.wav", w, out_sr_d2)

# -------- Decode input as list[dict] with numpy (12hz) --------
# Convert codes to numpy to simulate "serialized" payload.
list_dict_numpy_12hz = [{"audio_codes": c.cpu().numpy()} for c in enc2.audio_codes]
wavs_d3, out_sr_d3 = tokenizer_12hz.decode(list_dict_numpy_12hz)
for i, w in enumerate(wavs_d3):
    sf.write(f"decoded_listdict_numpy_12hz_{i}.wav", w, out_sr_d3)

# -------- Numpy input (must pass sr) --------
data = requests.get(audio_2, timeout=30).content
y, sr = sf.read(io.BytesIO(data))
enc3 = tokenizer_12hz.encode(y, sr=sr)
wavs3, out_sr3 = tokenizer_12hz.decode(enc3)
sf.write("decoded_numpy_12hz.wav", wavs3[0], out_sr3)