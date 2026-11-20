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

import argparse
import gc
import shutil
import tempfile
from pathlib import Path


def export_checkpoint(model_path, output_dir, dtype="float32",
                      device="cpu", max_shard_size="5GB", verify=False):
    import torch
    from qwen_tts import Qwen3TTSModel

    source = Path(model_path).resolve()
    output = Path(output_dir).resolve()
    if not source.is_dir():
        raise ValueError("--model_path must be a local model directory")
    if output.exists():
        raise ValueError("Output already exists; choose a new directory")
    if output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError("Output and input directories must not contain one another")
    for asset in ("speech_tokenizer/config.json", "generation_config.json"):
        if not (source / asset).is_file():
            raise ValueError(f"Missing serving asset: {source / asset}")
    if dtype not in ("float32", "float16", "bfloat16"):
        raise ValueError("Unsupported export dtype")

    wrapper = Qwen3TTSModel.from_pretrained(
        str(source), device_map=device, dtype=getattr(torch, dtype),
        attn_implementation="eager",
    )
    model = wrapper.model

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
    parser = argparse.ArgumentParser(description="Repackage a full fine-tuning checkpoint as a standalone toolkit checkpoint.")
    parser.add_argument("--model_path", required=True, help="Local full fine-tuning checkpoint.")
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