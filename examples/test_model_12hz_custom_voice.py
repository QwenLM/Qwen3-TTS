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
import time
import torch
import soundfile as sf

from qwen_tts import Qwen3TTSModel


def main():
    device = "cuda:0"
    MODEL_PATH = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"

    tts = Qwen3TTSModel.from_pretrained(
        MODEL_PATH,
        device_map=device,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )

    # -------- Single (with instruct) --------
    torch.cuda.synchronize()
    t0 = time.time()

    wavs, sr = tts.generate_custom_voice(
        text="其实我真的有发现，我是一个特别善于观察别人情绪的人。",
        language="Chinese",
        speaker="Vivian",
        instruct="用特别愤怒的语气说",
    )

    torch.cuda.synchronize()
    t1 = time.time()
    print(f"[CustomVoice Single] time: {t1 - t0:.3f}s")

    sf.write("qwen3_tts_test_custom_single.wav", wavs[0], sr)

    # -------- Batch (some empty instruct) --------
    texts = ["其实我真的有发现，我是一个特别善于观察别人情绪的人。", "She said she would be here by noon."]
    languages = ["Chinese", "English"]
    speakers = ["Vivian", "Ryan"]
    instructs = ["", "Very happy."]

    torch.cuda.synchronize()
    t0 = time.time()

    wavs, sr = tts.generate_custom_voice(
        text=texts,
        language=languages,
        speaker=speakers,
        instruct=instructs,
        max_new_tokens=2048,
    )

    torch.cuda.synchronize()
    t1 = time.time()
    print(f"[CustomVoice Batch] time: {t1 - t0:.3f}s")

    for i, w in enumerate(wavs):
        sf.write(f"qwen3_tts_test_custom_batch_{i}.wav", w, sr)


if __name__ == "__main__":
    main()
