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
import json
import os

from tqdm import tqdm

from qwen_tts import Qwen3TTSTokenizer


def count_lines(filepath):
    """Count lines in a file efficiently."""
    if not os.path.exists(filepath):
        return 0
    with open(filepath, 'r') as f:
        return sum(1 for _ in f)


def get_shard_output_path(output_jsonl, shard_id, num_shards):
    """Generate output path with shard suffix if sharding is enabled."""
    if num_shards <= 1:
        return output_jsonl
    base, ext = os.path.splitext(output_jsonl)
    return f"{base}.shard{shard_id:04d}_of_{num_shards:04d}{ext}"


def process_batch(tokenizer, batch_lines, batch_audios, output_file):
    """Process a batch and write results to output file."""
    if not batch_audios:
        return
    enc_res = tokenizer.encode(batch_audios)
    for code, line in zip(enc_res.audio_codes, batch_lines):
        line['audio_codes'] = code.cpu().tolist()
        output_file.write(json.dumps(line, ensure_ascii=False) + '\n')
    output_file.flush()  # Ensure data is written to disk


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--tokenizer_model_path", type=str, default="Qwen/Qwen3-TTS-Tokenizer-12Hz")
    parser.add_argument("--input_jsonl", type=str, required=True)
    parser.add_argument("--output_jsonl", type=str, required=True)
    parser.add_argument("--shard_id", type=int, default=0,
                        help="Shard index for parallel processing (0-indexed). Use with --num_shards.")
    parser.add_argument("--num_shards", type=int, default=1,
                        help="Total number of shards for parallel processing.")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Size of the processor's input batches.")
    args = parser.parse_args()

    # Validate sharding arguments
    if args.shard_id < 0 or args.shard_id >= args.num_shards:
        raise ValueError(f"shard_id must be in range [0, {args.num_shards}), got {args.shard_id}")

    # Count total lines for progress tracking
    total_lines = count_lines(args.input_jsonl)
    
    # Calculate lines for this shard
    shard_lines = (total_lines + args.num_shards - 1) // args.num_shards  # ceil division
    shard_start = args.shard_id * shard_lines
    shard_end = min(shard_start + shard_lines, total_lines)
    shard_total = shard_end - shard_start
    
    # Get output path (with shard suffix if sharding)
    output_path = get_shard_output_path(args.output_jsonl, args.shard_id, args.num_shards)
    
    # Check how many lines are already processed (for resumability)
    already_processed = count_lines(output_path)
    
    if args.num_shards > 1:
        print(f"Shard {args.shard_id}/{args.num_shards}: processing lines {shard_start}-{shard_end-1} "
              f"({shard_total} lines) -> {output_path}")
    
    if already_processed > 0:
        print(f"Resuming: already processed {already_processed} lines")
    
    if already_processed >= shard_total:
        print("All lines for this shard already processed. Nothing to do.")
        return

    tokenizer_12hz = Qwen3TTSTokenizer.from_pretrained(
        args.tokenizer_model_path,
        device_map=args.device,
    )

    batch_lines = []
    batch_audios = []
    lines_in_shard = 0  # Counter for lines belonging to this shard
    
    # Open output file in append mode for resumability
    with open(output_path, 'a') as output_file:
        with open(args.input_jsonl, 'r') as input_file:
            # Create progress bar for this shard
            pbar = tqdm(
                enumerate(input_file),
                total=shard_end,
                initial=shard_start + already_processed,
                desc=f"Shard {args.shard_id}" if args.num_shards > 1 else "Processing",
                unit="lines"
            )
            
            for idx, line_str in pbar:
                # Skip lines not in this shard's range
                if idx < shard_start:
                    continue
                if idx >= shard_end:
                    break
                
                lines_in_shard += 1
                
                # Skip already processed lines (for resumability)
                if lines_in_shard <= already_processed:
                    continue
                
                line = json.loads(line_str.strip())
                batch_lines.append(line)
                batch_audios.append(line['audio'])

                if len(batch_lines) >= args.batch_size:
                    process_batch(tokenizer_12hz, batch_lines, batch_audios, output_file)
                    batch_lines.clear()
                    batch_audios.clear()

            # Process remaining batch
            process_batch(tokenizer_12hz, batch_lines, batch_audios, output_file)

    print(f"Done! Processed {shard_total - already_processed} new lines for shard {args.shard_id}.")


if __name__ == "__main__":
    main()
