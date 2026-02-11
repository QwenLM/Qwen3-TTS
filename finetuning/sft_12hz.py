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
import shutil
from pathlib import Path

import torch
import wandb
from accelerate import Accelerator
from dataset import TTSDataset
from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
from qwen_tts.core.models.modeling_qwen3_tts import mel_spectrogram
from safetensors.torch import save_file, load_file
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoConfig, get_cosine_schedule_with_warmup
from huggingface_hub import snapshot_download

from tqdm import tqdm



def load_checkpoint(model, optimizer, scheduler, save_dir: Path, accelerator, train_subtalker: bool = False):
    """
    Load the latest checkpoint if it exists.
    Returns the step number to resume from, or 0 if no checkpoint found.
    """
    latest_folder = save_dir / "latest"
    if not latest_folder.exists():
        return 0

    # Load optimizer state
    optimizer_path = latest_folder / "optimizer.pth"
    if optimizer_path.exists():
        optimizer.load_state_dict(torch.load(optimizer_path, map_location="cpu"))
        accelerator.print(f"Loaded optimizer state from {optimizer_path}")

    # Load scheduler state
    scheduler_path = latest_folder / "scheduler.pth"
    if scheduler_path.exists():
        scheduler.load_state_dict(torch.load(scheduler_path, map_location="cpu"))
        accelerator.print(f"Loaded scheduler state from {scheduler_path}")

    unwrapped_model = accelerator.unwrap_model(model)

    # Load full model weights
    model_path = latest_folder / "model.safetensors"
    if model_path.exists():
        state_dict = load_file(str(model_path))
        unwrapped_model.load_state_dict(state_dict, strict=False)
        accelerator.print(f"Loaded model weights from {model_path}")

    # Try to infer step from training_state.json
    state_path = latest_folder / "training_state.json"
    if state_path.exists():
        with open(state_path, "r") as f:
            state = json.load(f)
        resume_step = state.get("step", 0)
        accelerator.print(f"Resuming from step {resume_step}")
        return resume_step

    # Fallback: infer step from checkpoint folders
    step_folders = [d for d in save_dir.iterdir() if d.is_dir() and d.name.startswith("step_")]
    if step_folders:
        steps = [int(d.name.split("_")[1]) for d in step_folders]
        resume_step = max(steps)
        accelerator.print(f"Resuming from step {resume_step}")
        return resume_step

    return 0


def save_checkpoint(model, optimizer, scheduler, save_dir: Path, step: int, epoch: int,
                    init_model_path: str, accelerator, is_latest: bool = False,
                    train_subtalker: bool = False):
    """
    Save checkpoint including model weights, optimizer, scheduler, and training state.
    """

    save_dir.mkdir(parents=True, exist_ok=True)

    if is_latest:
        folder = save_dir / "latest"
    else:
        tag = f"step_{step:07d}"
        folder = save_dir / tag

    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True, exist_ok=True)

    # Copy config and other files from init model
    shutil.copytree(init_model_path, folder, dirs_exist_ok=True)

    # Update config.json
    input_config_file = os.path.join(init_model_path, "config.json")
    output_config_file = folder / "config.json"
    with open(input_config_file, 'r', encoding='utf-8') as f:
        config_dict = json.load(f)
    config_dict["tts_model_type"] = "new_language"
    with open(output_config_file, 'w', encoding='utf-8') as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)

    unwrapped_model = accelerator.unwrap_model(model)

    # Save full model weights
    state_dict = {k: v.detach().to("cpu") for k, v in unwrapped_model.state_dict().items()}
    save_path = folder / "model.safetensors"
    save_file(state_dict, str(save_path))

    # Save optimizer and scheduler
    torch.save(optimizer.state_dict(), folder / "optimizer.pth")
    torch.save(scheduler.state_dict(), folder / "scheduler.pth")

    # Save training state
    training_state = {
        "step": step,
        "epoch": epoch,
        "train_subtalker": train_subtalker,
    }
    with open(folder / "training_state.json", "w") as f:
        json.dump(training_state, f, indent=2)

    accelerator.print(f"Checkpoint saved to {folder}")


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--init_model_path", type=str, default="Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    parser.add_argument("--output_model_path", type=str, default="/mnt/data_write/meaning-ckpts/")
    parser.add_argument("--train_jsonl", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=2e-6)
    parser.add_argument("--text_embedding_lr", type=float, default=None, 
                        help="Learning rate for text embedding layers. If not set, uses --lr")
    parser.add_argument("--num_epochs", type=int, default=5)
    # Learning rate schedule
    parser.add_argument("--warmup_steps", type=int, default=1000, help="Number of warmup steps for LR scheduler")
    # Logging
    parser.add_argument("--log_interval", type=int, default=10, help="Log metrics every N steps")
    # Checkpointing
    parser.add_argument("--save_interval", type=int, default=5000, help="Save checkpoint every N steps")
    parser.add_argument("--save_total_limit", type=int, default=1, help="Max number of checkpoints to keep")
    parser.add_argument("--resume_from_checkpoint", action="store_true", help="Resume from latest checkpoint")
    # Wandb
    parser.add_argument("--wandb_project", type=str, default="qwen3-tts-finetune", help="Wandb project name (empty to disable)")
    parser.add_argument("--wandb_run_name", type=str, default="", help="Wandb run name")
    parser.add_argument("--wandb_entity", type=str, default="", help="Wandb entity/team name")
    # Other
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm for clipping")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for AdamW")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of dataloader workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    # Sub-talker training
    parser.add_argument("--train_subtalker", action="store_true", 
                        help="Enable sub-talker (code_predictor) training for better timing control")
    parser.add_argument("--subtalker_loss_weight", type=float, default=1.0,
                        help="Final weight for sub-talker loss (relative to main talker loss)")
    parser.add_argument("--subtalker_warmup_steps", type=int, default=0,
                        help="Number of steps to linearly warm up sub-talker loss weight from 0 to subtalker_loss_weight")
    args = parser.parse_args()

    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Initialize accelerator with wandb logging
    accelerator = Accelerator(
        mixed_precision="bf16",
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        log_with="wandb",
    )

    # Initialize wandb (before save directory setup to get run ID)
    if accelerator.is_main_process:
        accelerator.init_trackers(
            project_name=args.wandb_project,
            config=vars(args),
            init_kwargs={
                "wandb": {
                    "name": args.wandb_run_name or None,
                    "entity": args.wandb_entity or None,
                    "resume": "allow",
                }
            }
        )

    # Get wandb run ID for checkpoint directory
    wandb_id = None
    if accelerator.is_main_process:
        wandb_id = accelerator.get_tracker("wandb").run.id
    # Broadcast wandb_id to all processes in multi-GPU setup
    if accelerator.num_processes > 1:
        wandb_id_list = [wandb_id]
        torch.distributed.broadcast_object_list(wandb_id_list, src=0)
        wandb_id = wandb_id_list[0]

    # Setup save directory: <output_model_path>/<wandb_id>
    save_dir = Path(args.output_model_path) / wandb_id
    if accelerator.is_main_process:
        save_dir.mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()

    MODEL_PATH = args.init_model_path

    # Resolve HuggingFace model ID to local cache path if needed
    if not os.path.isdir(MODEL_PATH):
        accelerator.print(f"Downloading model from HuggingFace: {MODEL_PATH}")
        MODEL_PATH = snapshot_download(repo_id=MODEL_PATH)
        accelerator.print(f"Model cached at: {MODEL_PATH}")

    qwen3tts = Qwen3TTSModel.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        # attn_implementation="flash_attention_2",
    )
    config = AutoConfig.from_pretrained(MODEL_PATH)

    dataset = TTSDataset(args.train_jsonl, qwen3tts.processor, config)
    train_dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=dataset.collate_fn, num_workers=args.num_workers, drop_last=True)

    def filter_no_grad_params(module):
        module_params = []
        for n, p in module.named_parameters():
            if not p.requires_grad:
                print(f"Gradient calculation for: {n} is disabled, excluding from optimization")
                continue
            module_params.append((n, p))
        return module_params

    # Freeze speaker encoder (used for embedding extraction only, always frozen)
    for p in qwen3tts.model.speaker_encoder.parameters():
        p.requires_grad = False

    # Freeze code_predictor (sub-talker) when not training it
    if not args.train_subtalker:
        for p in qwen3tts.model.talker.code_predictor.parameters():
            p.requires_grad = False
    else:
        # If training subtalker, ensure the model config outputs hidden states
        qwen3tts.model.talker.config.output_hidden_states = True
        qwen3tts.model.talker.model.config.output_hidden_states = True

    # Separate trainable parameters into text embedding and other for different learning rates
    text_embedding_params = []
    other_params = []
    
    for name, param in filter_no_grad_params(qwen3tts.model):
        if "text_embedding" in name:
            text_embedding_params.append(param)
            print(f"Text embedding parameter: {name}")
        else:
            other_params.append(param)
    
    # Use separate learning rate for text embedding if specified
    text_embedding_lr = args.text_embedding_lr if args.text_embedding_lr is not None else args.lr
    
    param_groups = [
        {"params": text_embedding_params, "lr": text_embedding_lr},
        {"params": other_params, "lr": args.lr},
    ]
    
    optimizer = AdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay)

    trainable_params = sum(p.numel() for p in qwen3tts.model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in qwen3tts.model.parameters())
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)", flush=True)

    model, optimizer, train_dataloader = accelerator.prepare(
        qwen3tts.model, optimizer, train_dataloader
    )

    # Calculate total training steps AFTER prepare() so len(train_dataloader) reflects sharding
    num_update_steps_per_epoch = len(train_dataloader) // args.gradient_accumulation_steps
    num_epochs = args.num_epochs
    total_training_steps = num_update_steps_per_epoch * num_epochs

    # Create learning rate scheduler (cosine with warmup)
    # NOTE: Do NOT use accelerator.prepare(scheduler) — AcceleratedScheduler calls
    # scheduler.step() num_processes times per optimizer step (assuming unsharded total_steps),
    # but we already computed total_training_steps from the sharded dataloader length.
    # We step the scheduler manually on gradient sync instead.
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_training_steps,
    )

    # Resume from checkpoint if requested
    global_step = 0
    start_epoch = 0
    if args.resume_from_checkpoint:
        global_step = load_checkpoint(model, optimizer, scheduler, save_dir, accelerator, 
                                       train_subtalker=args.train_subtalker)
        start_epoch = global_step // num_update_steps_per_epoch

    model.train()
    accelerator.print(f"Saving checkpoints to: {save_dir}")
    accelerator.print(f"Starting training for {total_training_steps} steps ({num_epochs} epochs)")
    accelerator.print(f"  Batch size: {args.batch_size}")
    accelerator.print(f"  Gradient accumulation steps: {args.gradient_accumulation_steps}")
    accelerator.print(f"  Effective batch size: {args.batch_size * args.gradient_accumulation_steps * accelerator.num_processes}")
    accelerator.print(f"  Learning rate: {args.lr}")
    accelerator.print(f"  Text embedding learning rate: {text_embedding_lr}")
    accelerator.print(f"  Warmup steps: {args.warmup_steps}")
    if args.train_subtalker:
        accelerator.print(f"  Sub-talker training: enabled (loss_weight={args.subtalker_loss_weight}, warmup_steps={args.subtalker_warmup_steps})")

    for epoch in range(start_epoch, num_epochs):
        progress_bar = tqdm(
            train_dataloader,
            desc=f"Epoch {epoch}",
            disable=not accelerator.is_local_main_process
        )
        epoch_loss = 0.0
        num_steps = 0

        # Get the unwrapped model for accessing sub-modules (model is DDP-wrapped after prepare())
        unwrapped = accelerator.unwrap_model(model)

        for step, batch in enumerate(progress_bar):

            # Skip steps if resuming (accounting for gradient accumulation)
            if global_step > 0 and step < (global_step % num_update_steps_per_epoch) * args.gradient_accumulation_steps:
                continue

            with accelerator.accumulate(model):

                input_ids = batch['input_ids']
                codec_ids = batch['codec_ids']
                ref_audios = batch['ref_audios']  # List of variable-length numpy arrays
                text_embedding_mask = batch['text_embedding_mask']
                codec_embedding_mask = batch['codec_embedding_mask']
                attention_mask = batch['attention_mask']
                codec_0_labels = batch['codec_0_labels']
                codec_mask = batch['codec_mask']

                # Extract speaker embeddings from variable-length audio
                # Process each audio individually through mel spectrogram and speaker encoder
                speaker_embeddings_list = []
                for audio in ref_audios:
                    # Compute mel spectrogram for this audio (variable length)
                    mel = mel_spectrogram(
                        torch.from_numpy(audio).unsqueeze(0),
                        n_fft=1024,
                        num_mels=128,
                        sampling_rate=24000,
                        hop_size=256,
                        win_size=1024,
                        fmin=0,
                        fmax=12000
                    ).transpose(1, 2).to(device=model.device, dtype=torch.bfloat16)  # [1, time, 128]
                    
                    # Extract speaker embedding (handles variable length via attentive pooling)
                    spk_emb = unwrapped.speaker_encoder(mel).detach()  # [1, embed_dim]
                    speaker_embeddings_list.append(spk_emb)
                
                # Stack into batch tensor [batch_size, embed_dim]
                speaker_embedding = torch.cat(speaker_embeddings_list, dim=0)

                input_text_ids = input_ids[:, :, 0]
                input_codec_ids = input_ids[:, :, 1]

                text_emb = unwrapped.talker.model.text_embedding(input_text_ids)
                text_emb = unwrapped.talker.text_projection(text_emb)
                input_text_embedding = text_emb * text_embedding_mask
                input_codec_embedding = unwrapped.talker.model.codec_embedding(input_codec_ids) * codec_embedding_mask
                input_codec_embedding[:, 6, :] = speaker_embedding

                input_embeddings = input_text_embedding + input_codec_embedding

                for i in range(1, 16):
                    codec_i_embedding = unwrapped.talker.code_predictor.get_input_embeddings()[i - 1](codec_ids[:, :, i])
                    codec_i_embedding = codec_i_embedding * codec_mask.unsqueeze(-1)
                    input_embeddings = input_embeddings + codec_i_embedding
                
                outputs = unwrapped.talker(
                    inputs_embeds=input_embeddings,
                    attention_mask=attention_mask,
                    labels=codec_0_labels,
                    output_hidden_states=True
                )

                # Compute sub-talker loss if enabled
                sub_talker_loss = None
                current_subtalker_weight = 0.0
                if args.train_subtalker:
                    
                    # Get the last layer's hidden states from the nested tuple structure
                    hidden_states = outputs.hidden_states[0][-1]

                    target_codec_mask = codec_mask[:, 1:]
                    talker_hidden_states = hidden_states[:, :-1][target_codec_mask]
                    talker_codec_ids = codec_ids[:, 1:][target_codec_mask]
                    
                    # Compute sub-talker loss
                    _, sub_talker_loss = unwrapped.talker.forward_sub_talker_finetune(
                        talker_codec_ids, talker_hidden_states
                    )
                    
                    # Progressive sub-talker loss weight: linear warmup from 0 to subtalker_loss_weight
                    if args.subtalker_warmup_steps > 0:
                        warmup_progress = min(1.0, global_step / args.subtalker_warmup_steps)
                        current_subtalker_weight = args.subtalker_loss_weight * warmup_progress
                    else:
                        current_subtalker_weight = args.subtalker_loss_weight
                    
                    loss = outputs.loss + current_subtalker_weight * sub_talker_loss
                else:
                    loss = outputs.loss

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                else:
                    grad_norm = 0.0

                optimizer.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                scheduler.step()
                global_step += 1

            epoch_loss += loss.item()
            num_steps += 1

            current_lr = scheduler.get_last_lr()[0]
            progress_bar.set_postfix(
                loss=f"{loss.item():.4f}",
                avg_loss=f"{epoch_loss/num_steps:.4f}",
                lr=f"{current_lr:.2e}",
                step=global_step
            )

            # Logging
            if accelerator.sync_gradients and global_step % args.log_interval == 0:
                metrics = {
                    "train/loss": loss.item(),
                    "train/avg_loss": epoch_loss / num_steps,
                    "train/talker_loss": outputs.loss.item(),
                    "train/learning_rate": current_lr,
                    "train/epoch": epoch + step / len(train_dataloader),
                    "train/grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
                }
                if sub_talker_loss is not None:
                    metrics["train/sub_talker_loss"] = sub_talker_loss.item()
                    metrics["train/sub_talker_weight"] = current_subtalker_weight
                accelerator.log(metrics, step=global_step)

            # Save checkpoint
            if accelerator.sync_gradients and global_step % args.save_interval == 0:
                if accelerator.is_main_process:
                    save_checkpoint(
                        model, optimizer, scheduler, save_dir, global_step, epoch,
                        MODEL_PATH, accelerator, is_latest=False,
                        train_subtalker=args.train_subtalker
                    )
                accelerator.wait_for_everyone()

            # step += 1

        accelerator.print(f"Epoch {epoch} completed | Average Loss: {epoch_loss/num_steps:.4f}")

    # Final checkpoint
    if accelerator.is_main_process:
        accelerator.print("Training completed! Saving final checkpoint...")
        save_checkpoint(
            model, optimizer, scheduler, save_dir, global_step, epoch,
            MODEL_PATH, accelerator, is_latest=False,
            train_subtalker=args.train_subtalker
        )

    accelerator.end_training()


if __name__ == "__main__":
    train()
