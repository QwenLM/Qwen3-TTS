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
import os
import time
import torch
import soundfile as sf

from qwen_tts import Qwen3TTSModel
from qwen_tts.core.device_utils import (
    get_optimal_device,
    get_attention_implementation,
    device_synchronize,
    get_device_info,
)


def ensure_dir(d: str):
    os.makedirs(d, exist_ok=True)


def run_case(tts: Qwen3TTSModel, out_dir: str, case_name: str, call_fn, device: str):
    device_synchronize(device)
    t0 = time.time()

    wavs, sr = call_fn()

    device_synchronize(device)
    t1 = time.time()
    print(f"[{case_name}] time: {t1 - t0:.3f}s, n_wavs={len(wavs)}, sr={sr}")

    for i, w in enumerate(wavs):
        sf.write(os.path.join(out_dir, f"{case_name}_{i}.wav"), w, sr)


def main():
    device = get_optimal_device()
    print(f"Using device: {get_device_info(device)}\n")
    MODEL_PATH = "Qwen/Qwen3-TTS-12Hz-1.7B-Base/"
    OUT_DIR = "qwen3_tts_test_voice_clone_output_wav"
    ensure_dir(OUT_DIR)

    attn_impl = get_attention_implementation(device)

    tts = Qwen3TTSModel.from_pretrained(
        MODEL_PATH,
        device_map=device,
        dtype=torch.bfloat16,
        attn_implementation=attn_impl,
    )

    # Reference audio(s)
    ref_audio_path_1 = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-TTS-Repo/clone_2.wav"
    ref_audio_path_2 = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-TTS-Repo/clone_1.wav"

    ref_audio_single = ref_audio_path_1
    ref_audio_batch = [ref_audio_path_1, ref_audio_path_2]

    ref_text_single = "Okay. Yeah. I resent you. I love you. I respect you. But you know what? You blew it! And thanks to you."
    ref_text_batch = [
        "Okay. Yeah. I resent you. I love you. I respect you. But you know what? You blew it! And thanks to you.",
        "甚至出现交易几乎停滞的情况。",
    ]

    # Synthesis targets
    syn_text_single = "Good one. Okay, fine, I'm just gonna leave this sock monkey here. Goodbye."
    syn_lang_single = "Auto"

    syn_text_batch = [
        "Good one. Okay, fine, I'm just gonna leave this sock monkey here. Goodbye.",
        "其实我真的有发现，我是一个特别善于观察别人情绪的人。",
    ]
    syn_lang_batch = ["Chinese", "English"]

    common_gen_kwargs = dict(
        max_new_tokens=2048,
        do_sample=True,
        top_k=50,
        top_p=1.0,
        temperature=0.9,
        repetition_penalty=1.05,
        subtalker_dosample=True,
        subtalker_top_k=50,
        subtalker_top_p=1.0,
        subtalker_temperature=0.9,
    )

    for xvec_only in [False, True]:
        mode_tag = "xvec_only" if xvec_only else "icl"

        # Case 1: prompt single + synth single, direct
        run_case(
            tts, OUT_DIR, f"case1_promptSingle_synSingle_direct_{mode_tag}",
            lambda: tts.generate_voice_clone(
                text=syn_text_single,
                language=syn_lang_single,
                ref_audio=ref_audio_single,
                ref_text=ref_text_single,
                x_vector_only_mode=xvec_only,
                **common_gen_kwargs,
            ),
            device,
        )

        # Case 1b: prompt single + synth single, via create_voice_clone_prompt
        def _case1b():
            prompt_items = tts.create_voice_clone_prompt(
                ref_audio=ref_audio_single,
                ref_text=ref_text_single,
                x_vector_only_mode=xvec_only,
            )
            return tts.generate_voice_clone(
                text=syn_text_single,
                language=syn_lang_single,
                voice_clone_prompt=prompt_items,
                **common_gen_kwargs,
            )

        run_case(
            tts, OUT_DIR, f"case1_promptSingle_synSingle_promptThenGen_{mode_tag}",
            _case1b,
            device,
        )

        # Case 2: prompt single + synth batch, direct
        run_case(
            tts, OUT_DIR, f"case2_promptSingle_synBatch_direct_{mode_tag}",
            lambda: tts.generate_voice_clone(
                text=syn_text_batch,
                language=syn_lang_batch,
                ref_audio=ref_audio_single,
                ref_text=ref_text_single,
                x_vector_only_mode=xvec_only,
                **common_gen_kwargs,
            ),
            device,
        )

        # Case 2b: prompt single + synth batch, via create_voice_clone_prompt
        def _case2b():
            prompt_items = tts.create_voice_clone_prompt(
                ref_audio=ref_audio_single,
                ref_text=ref_text_single,
                x_vector_only_mode=xvec_only,
            )
            return tts.generate_voice_clone(
                text=syn_text_batch,
                language=syn_lang_batch,
                voice_clone_prompt=prompt_items,
                **common_gen_kwargs,
            )

        run_case(
            tts, OUT_DIR, f"case2_promptSingle_synBatch_promptThenGen_{mode_tag}",
            _case2b,
            device,
        )

        # Case 3: prompt batch + synth batch, direct
        run_case(
            tts, OUT_DIR, f"case3_promptBatch_synBatch_direct_{mode_tag}",
            lambda: tts.generate_voice_clone(
                text=syn_text_batch,
                language=syn_lang_batch,
                ref_audio=ref_audio_batch,
                ref_text=ref_text_batch,
                x_vector_only_mode=[xvec_only, xvec_only],
                **common_gen_kwargs,
            ),
            device,
        )

        # Case 3b: prompt batch + synth batch, via create_voice_clone_prompt
        def _case3b():
            prompt_items = tts.create_voice_clone_prompt(
                ref_audio=ref_audio_batch,
                ref_text=ref_text_batch,
                x_vector_only_mode=[xvec_only, xvec_only],
            )
            return tts.generate_voice_clone(
                text=syn_text_batch,
                language=syn_lang_batch,
                voice_clone_prompt=prompt_items,
                **common_gen_kwargs,
            )

        run_case(
            tts, OUT_DIR, f"case3_promptBatch_synBatch_promptThenGen_{mode_tag}",
            _case3b,
            device,
        )


if __name__ == "__main__":
    main()
