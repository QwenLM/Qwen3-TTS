"""
VibeVoice Gradio Demo - High-Quality Dialogue Generation Interface with Streaming Support
"""

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import List, Dict, Any, Iterator
from datetime import datetime
import threading
import numpy as np
import gradio as gr
import librosa
import soundfile as sf
import torch
import os
import traceback
import psutil
from transformers import BitsAndBytesConfig
from io import BytesIO
import requests
import re
from urllib.parse import urlparse
try:
    from pydub import AudioSegment
except ImportError:
    print("Warning: pydub is not installed. MP3 export will not be available.")
    print("Please install it using: pip install pydub")
    AudioSegment = None

try:
    from ruaccent import RUAccent
    RUACCENT_AVAILABLE = True
except ImportError:
    print("Warning: ruaccent is not installed. Stress marks functionality will not be available.")
    print("Please install it using: pip install ruaccent")
    RUAccent = None
    RUACCENT_AVAILABLE = False

# НОВОЕ: Импорты для скачивания моделей
from huggingface_hub import hf_hub_download

from vibevoice.modular.configuration_vibevoice import VibeVoiceConfig
from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor
from vibevoice.modular.streamer import AudioStreamer
from transformers.utils import logging
from transformers import set_seed

logging.set_verbosity_info()
logger = logging.get_logger(__name__)

# НОВОЕ: Словарь с доступными моделями
MODEL_REPOS = {
    "VibeVoice-Large": "aoi-ot/VibeVoice-Large",
    "VibeVoice-Large-Q8": "FabioSarracino/VibeVoice-Large-Q8",
    "VibeVoice-7B": "aoi-ot/VibeVoice-7B",
    "VibeVoice-1.5B": "microsoft/VibeVoice-1.5B",
    
    
}


class VibeVoiceDemo:
    def __init__(self, device: str = "cuda", inference_steps: int = 5):
        """Initialize the VibeVoice demo."""
        self.device = device
        self.initial_inference_steps = inference_steps
        self.is_generating = False
        self.stop_generation = False
        self.current_streamer = None

        # НОВОЕ: Атрибуты для ленивой загрузки моделей
        self.model = None
        self.processor = None
        self.current_model_name = None

        self.use_int4 = False

        # Инициализация RUAccent для расстановки ударений
        self.accentizer = None
        if RUACCENT_AVAILABLE:
            try:
                self.accentizer = RUAccent()
                self.accentizer.load(omograph_model_size='turbo3.1', use_dictionary=True, device='CPU')
                print("✅ RUAccent успешно загружен для расстановки ударений")
            except Exception as e:
                print(f"⚠️ Ошибка при загрузке RUAccent: {e}")
                self.accentizer = None

        self.setup_voice_presets()
        self.load_example_scripts()

    def load_model(self, model_name: str, use_int4: bool = True):
        """
        Download (if necessary) and load the selected VibeVoice model and processor.
        This function is called on-the-fly before the first generation.
        """
        if model_name not in MODEL_REPOS:
            raise ValueError(f"Unknown model: {model_name}. Available models are: {list(MODEL_REPOS.keys())}")

        model_path = MODEL_REPOS[model_name]
        print(f"Loading processor & model '{model_name}' from {model_path}")
        
        # Очищаем память GPU перед загрузкой новой модели
        if self.model is not None:
            del self.model
            del self.processor
            torch.cuda.empty_cache()
            self.model = None
            self.processor = None

        if self.device.lower() == "mpx":
            print("Note: device 'mpx' detected, treating it as 'mps'.")
            self.device = "mps"
        if self.device == "mps" and not torch.backends.mps.is_available():
            print("Warning: MPS not available. Falling back to CPU.")
            self.device = "cpu"
        print(f"Using device: {self.device}")
        
        self.processor = VibeVoiceProcessor.from_pretrained(model_path)
        
        if self.device == "mps":
            load_dtype = torch.float16
            attn_impl_primary = "sdpa"
        elif self.device == "cuda":
            load_dtype = torch.float16
            attn_impl_primary = "flash_attention_2"
        else:
            load_dtype = torch.float16
            attn_impl_primary = "sdpa"
        print(f"Using device: {self.device}, torch_dtype: {load_dtype}, attn_implementation: {attn_impl_primary}")
        
        bnb_config = None
        if self.use_int4 and self.device == "cuda":
            try:
                from transformers import BitsAndBytesConfig
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                print("INT4 quantization enabled via bitsandbytes.")
            except Exception as e:
                print(f"[WARNING] Could not enable int4 quantization: {e}")

        try:
            device_map_config = self.device if self.device in ("cuda", "cpu") else None
            if self.device == "mps":
                device_map_config = None # MPS requires explicit .to()

            self.model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                model_path,
                torch_dtype=load_dtype,
                device_map=device_map_config,
                attn_implementation=attn_impl_primary,
                quantization_config=bnb_config,
            )
            if self.device == "mps":
                self.model.to("mps")
        except Exception as e:
            if attn_impl_primary == 'flash_attention_2':
                print(f"[ERROR] : {type(e).__name__}: {e}")
                print(traceback.format_exc())
                fallback_attn = "sdpa"
                print(f"Falling back to attention implementation: {fallback_attn}")
                self.model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                    model_path,
                    torch_dtype=load_dtype,
                    device_map=(self.device if self.device in ("cuda", "cpu") else None),
                    attn_implementation=fallback_attn,
                    quantization_config=bnb_config, 
                )
                if self.device == "mps":
                    self.model.to("mps")
            else:
                raise e
        
        self.model.eval()
        self.model.model.noise_scheduler = self.model.model.noise_scheduler.from_config(
            self.model.model.noise_scheduler.config, 
            algorithm_type='sde-dpmsolver++',
            beta_schedule='squaredcos_cap_v2'
        )
        self.model.set_ddpm_inference_steps(num_steps=self.initial_inference_steps)
        if hasattr(self.model.model, 'language_model'):
            print(f"Language model attention: {self.model.model.language_model.config._attn_implementation}")

        self.current_model_name = model_name
        print(f"✅ Модель '{model_name}' Успешно загружена.")

        # === Мониторинг памяти ===
        import psutil
        import os

        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
            vram_allocated = torch.cuda.memory_allocated() / (1024 ** 2)
            vram_reserved = torch.cuda.memory_reserved() / (1024 ** 2)
            print(f"[VRAM] Allocated: {vram_allocated:.2f} MB | Reserved: {vram_reserved:.2f} MB")

        process = psutil.Process(os.getpid())
        ram_usage = process.memory_info().rss / (1024 ** 2)
        print(f"[RAM] Used by process: {ram_usage:.2f} MB")


    def setup_voice_presets(self):
        """Setup voice presets by scanning the voices directory."""
        voices_dir = os.path.join(os.path.dirname(__file__), "voices")
        
        if not os.path.exists(voices_dir):
            print(f"Warning: Voices directory not found at {voices_dir}")
            self.voice_presets = {}
            self.available_voices = {}
            return
        
        self.voice_presets = {}
        supported_extensions = ('.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac')
        audio_files = [f for f in os.listdir(voices_dir) 
                       if f.lower().endswith(supported_extensions) and os.path.isfile(os.path.join(voices_dir, f))]
        
        for audio_file in audio_files:
            name = os.path.splitext(audio_file)[0]
            full_path = os.path.join(voices_dir, audio_file)
            self.voice_presets[name] = full_path
        
        self.voice_presets = dict(sorted(self.voice_presets.items()))
        
        self.available_voices = {
            name: path for name, path in self.voice_presets.items()
            if os.path.exists(path)
        }
        
        if not self.available_voices:
            raise gr.Error("Не найдены пресеты голосов. Пожалуйста, добавьте аудиофайлы в директорию demo/voices.")
        
        print(f"Found {len(self.available_voices)} voice files in {voices_dir}")
        print(f"Available voices: {', '.join(self.available_voices.keys())}")
    
    def read_audio(self, audio_path: str, target_sr: int = 24000) -> np.ndarray:
        """Read and preprocess audio file using librosa for broad format support."""
        if audio_path is None:
             raise ValueError("Audio path is None, cannot read audio.")
        try:
            wav, sr = librosa.load(audio_path, sr=target_sr, mono=True)
            return wav
        except Exception as e:
            print(f"Error reading audio {audio_path}: {e}")
            raise IOError(f"Не удалось прочитать аудиофайл: {os.path.basename(audio_path)}") from e
    
    def generate_podcast_streaming(self,
                                 model_name: str, # НОВЫЙ ПАРАМЕТР
                                 num_speakers: int,
                                 script: str,
                                 speaker_1_audio: str,
                                 speaker_2_audio: str,
                                 speaker_3_audio: str,
                                 speaker_4_audio: str,
                                 cfg_scale: float,
                                 inference_steps: int,
                                 do_sample: bool,
                                 temperature: float,
                                 top_p: float,
                                 refresh_negative,
                                 use_int4: bool = False,
                                 seed: int = None) -> Iterator[tuple]:
        try:
            # НОВОЕ: Ленивая загрузка модели при необходимости
            if self.model is None or self.current_model_name != model_name:
                log_msg = f"⏳ Загрузка модели '{model_name}'. Это может занять несколько минут при первом запуске..."
                yield None, None, log_msg, gr.update(visible=False)
                self.load_model(model_name)
                log_msg = f"✅ Модель '{model_name}' успешно загружена. Начинаю генерацию..."
                yield None, None, log_msg, gr.update(visible=False)

            self.stop_generation = False
            self.is_generating = True
            
            if not script.strip():
                self.is_generating = False
                raise gr.Error("Ошибка: Пожалуйста, предоставьте сценарий.")

            script = script.replace("’", "'")
            
            if num_speakers < 1 or num_speakers > 4:
                self.is_generating = False
                raise gr.Error("Ошибка: Количество дикторов должно быть от 1 до 4.")
            
            selected_speaker_audios = [speaker_1_audio, speaker_2_audio, speaker_3_audio, speaker_4_audio][:num_speakers]
            
            for i, audio_path in enumerate(selected_speaker_audios):
                if not audio_path:
                    self.is_generating = False
                    raise gr.Error(f"Ошибка: Пожалуйста, предоставьте аудиофайл для Диктора {i+1}.")
            
            self.model.set_ddpm_inference_steps(num_steps=inference_steps)
            
            log = f"🧠 Модель: {model_name}\n"
            log += f"⚙️ Режим: {'INT4' if use_int4 else 'Full model'}\n"
            log += f"🎲 Seed: {seed}\n"
            log += f"🎙️ Генерирую подкаст с {num_speakers} дикторами\n"
            log += f"📊 Параметры: Сила ведения={cfg_scale}, Шаги вывода={inference_steps}, Семплирование={do_sample}\n"
            if do_sample:
                log += f"🌡️ Температура={temperature}, Top-p={top_p}\n"
            
            speaker_names_for_log = [os.path.basename(p) if p else "N/A" for p in selected_speaker_audios]
            log += f"🎭 Дикторы: {', '.join(speaker_names_for_log)}\n"
            
            if self.stop_generation:
                self.is_generating = False
                yield None, "🛑 Генерация остановлена пользователем", gr.update(visible=False)
                return
            
            voice_samples = []
            for i, audio_path in enumerate(selected_speaker_audios):
                try:
                    audio_data = self.read_audio(audio_path)
                    voice_samples.append(audio_data)
                except (ValueError, IOError) as e:
                    self.is_generating = False
                    raise gr.Error(f"Ошибка при загрузке аудио для Диктора {i+1}: {e}")

            if self.stop_generation:
                self.is_generating = False
                yield None, "🛑 Генерация остановлена пользователем", gr.update(visible=False)
                return
            
            lines = script.strip().split('\n')
            formatted_script_lines = []
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                if line.startswith('Speaker ') and ':' in line:
                    formatted_script_lines.append(line)
                else:
                    speaker_id = len(formatted_script_lines) % num_speakers
                    formatted_script_lines.append(f"Speaker {speaker_id}: {line}")
            
            formatted_script = '\n'.join(formatted_script_lines)
            log += f"📝 Отформатированный сценарий с {len(formatted_script_lines)} репликами\n\n"
            log += "🔄 Обработка с помощью VibeVoice (потоковый режим)...\n"
            
            if self.stop_generation:
                self.is_generating = False
                yield None, "🛑 Генерация остановлена пользователем", gr.update(visible=False)
                return
            
            start_time = time.time()
            
            inputs = self.processor(
                text=[formatted_script], voice_samples=[voice_samples],
                padding=True, return_tensors="pt", return_attention_mask=True,
            )
            target_device = self.device if self.device in ("cuda", "mps") else "cpu"
            for k, v in inputs.items():
                if torch.is_tensor(v):
                    inputs[k] = v.to(target_device)
            
            audio_streamer = AudioStreamer(batch_size=1, stop_signal=None, timeout=None)
            self.current_streamer = audio_streamer
            
            generation_thread = threading.Thread(
                target=self._generate_with_streamer,
                args=(inputs, cfg_scale, do_sample, temperature, top_p, refresh_negative, audio_streamer)
            )
            generation_thread.start()
            
            time.sleep(1)

            if self.stop_generation:
                audio_streamer.end()
                generation_thread.join(timeout=5.0)
                self.is_generating = False
                yield None, "🛑 Генерация остановлена пользователем", gr.update(visible=False)
                return

            sample_rate = 24000
            all_audio_chunks, pending_chunks = [], []
            chunk_count = 0
            last_yield_time = time.time()
            min_yield_interval = 15
            min_chunk_size = sample_rate * 30
            
            audio_stream = audio_streamer.get_stream(0)
            has_yielded_audio = False
            has_received_chunks = False
            
            for audio_chunk in audio_stream:
                if self.stop_generation:
                    audio_streamer.end()
                    break
                    
                chunk_count += 1
                has_received_chunks = True
                
                if torch.is_tensor(audio_chunk):
                    if audio_chunk.dtype in [torch.bfloat16, torch.float16]:
                        audio_chunk = audio_chunk.float()
                    audio_np = audio_chunk.cpu().numpy().astype(np.float32)
                    # Очистка тензора после конвертации
                    del audio_chunk
                else:
                    audio_np = np.array(audio_chunk, dtype=np.float32)
                
                if len(audio_np.shape) > 1:
                    audio_np = audio_np.squeeze()
                
                audio_16bit = convert_to_16_bit_wav(audio_np)
                all_audio_chunks.append(audio_16bit)
                pending_chunks.append(audio_16bit)
                
                pending_audio_size = sum(len(chunk) for chunk in pending_chunks)
                current_time = time.time()
                time_since_last_yield = current_time - last_yield_time
                
                should_yield = False
                if not has_yielded_audio and pending_audio_size >= min_chunk_size:
                    should_yield = True
                    has_yielded_audio = True
                elif has_yielded_audio and (pending_audio_size >= min_chunk_size or time_since_last_yield >= min_yield_interval):
                    should_yield = True
                
                if should_yield and pending_chunks:
                    new_audio = np.concatenate(pending_chunks)
                    total_duration = sum(len(chunk) for chunk in all_audio_chunks) / sample_rate
                    log_update = log + f"🎵 Потоковая передача: {total_duration:.1f}с сгенерировано (чанк {chunk_count})\n"
                    yield (sample_rate, new_audio), None, log_update, gr.update(visible=True)
                    pending_chunks = []
                    last_yield_time = current_time
            
            if pending_chunks:
                final_new_audio = np.concatenate(pending_chunks)
                total_duration = sum(len(chunk) for chunk in all_audio_chunks) / sample_rate
                log_update = log + f"🎵 Потоковая передача последнего чанка: {total_duration:.1f}с всего\n"
                yield (sample_rate, final_new_audio), None, log_update, gr.update(visible=True)
                has_yielded_audio = True
            
            generation_thread.join(timeout=5.0)

            if generation_thread.is_alive():
                print("Warning: Generation thread did not complete within timeout")
                audio_streamer.end()
                generation_thread.join(timeout=5.0)

            self.current_streamer = None
            self.is_generating = False
            generation_time = time.time() - start_time
            
            if self.stop_generation:
                yield None, None, "🛑 Генерация остановлена пользователем", gr.update(visible=False)
                return
            
            if not has_received_chunks:
                error_log = log + f"\n❌ Ошибка: От модели не получено ни одного аудио-чанка. Время генерации: {generation_time:.2f}с"
                yield None, None, error_log, gr.update(visible=False)
                return

            if all_audio_chunks:
                complete_audio = np.concatenate(all_audio_chunks)
                final_duration = len(complete_audio) / sample_rate
                
                final_log = log + f"⏱️ Генерация завершена за {generation_time:.2f} секунд\n"
                final_log += f"🎵 Итоговая длительность аудио: {final_duration:.2f} секунд\n"
                final_log += f"📊 Всего чанков: {chunk_count}\n"
                final_log += "✨ Генерация прошла успешно! Полное аудио готово.\n"
                final_log += "💡 Не удовлетворены? Вы можете сгенерировать заново или изменить параметры генерации для других результатов."
                
                yield None, (sample_rate, complete_audio), final_log, gr.update(visible=False)
                
                # Очистка памяти после завершения
                del all_audio_chunks
                del complete_audio
            else:
                final_log = log + "❌ Аудио не было сгенерировано."
                yield None, None, final_log, gr.update(visible=False)
            
            # Очистка тензоров и освобождение памяти
            for k in list(inputs.keys()):
                if torch.is_tensor(inputs[k]):
                    inputs[k] = inputs[k].cpu()
                    del inputs[k]
            del inputs
            del voice_samples
            
            if self.device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()

        except gr.Error as e:
            self.is_generating = False
            self.current_streamer = None
            error_msg = f"❌ Ошибка ввода: {str(e)}"
            print(error_msg)
            # Очистка памяти при ошибке
            if 'inputs' in locals():
                for k in list(inputs.keys()):
                    if torch.is_tensor(inputs[k]):
                        inputs[k] = inputs[k].cpu()
                        del inputs[k]
                del inputs
            if 'voice_samples' in locals():
                del voice_samples
            if self.device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
            yield None, None, error_msg, gr.update(visible=False)
            
        except Exception as e:
            self.is_generating = False
            self.current_streamer = None
            error_msg = f"❌ Произошла непредвиденная ошибка: {str(e)}"
            print(error_msg, traceback.format_exc())
            # Очистка памяти при ошибке
            if 'inputs' in locals():
                for k in list(inputs.keys()):
                    if torch.is_tensor(inputs[k]):
                        inputs[k] = inputs[k].cpu()
                        del inputs[k]
                del inputs
            if 'voice_samples' in locals():
                del voice_samples
            if self.device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
            yield None, None, error_msg, gr.update(visible=False)

    def _generate_with_streamer(self, inputs, cfg_scale, do_sample, temperature, top_p, refresh_negative, audio_streamer):
        """Helper method to run generation with streamer in a separate thread."""
        try:
            if self.stop_generation:
                audio_streamer.end()
                return
                
            def check_stop_generation():
                return self.stop_generation
            
            generation_config = {
                'do_sample': do_sample,
                'temperature': temperature if do_sample else None,
                'top_p': top_p if do_sample else None,
            }
                
            outputs = self.model.generate(
                **inputs, max_new_tokens=None, cfg_scale=cfg_scale,
                tokenizer=self.processor.tokenizer, generation_config=generation_config,
                audio_streamer=audio_streamer, stop_check_fn=check_stop_generation,
                verbose=False, refresh_negative=refresh_negative,
            )
            
        except Exception as e:
            print(f"Error in generation thread: {e}")
            traceback.print_exc()
            audio_streamer.end()
    
    def stop_audio_generation(self):
        """Stop the current audio generation process."""
        self.stop_generation = True
        if self.current_streamer is not None:
            try:
                self.current_streamer.end()
            except Exception as e:
                print(f"Error stopping streamer: {e}")
        print("🛑 Audio generation stop requested")
    
    def load_example_scripts(self):
        """Load example scripts from the text_examples directory."""
        examples_dir = os.path.join(os.path.dirname(__file__), "text_examples")
        self.example_scripts = []
        
        if not os.path.exists(examples_dir):
            print(f"Warning: text_examples directory not found at {examples_dir}")
            return
        
        txt_files = sorted([f for f in os.listdir(examples_dir) 
                          if f.lower().endswith('.txt') and os.path.isfile(os.path.join(examples_dir, f))])
        
        for txt_file in txt_files:
            file_path = os.path.join(examples_dir, txt_file)
            
            import re
            time_pattern = re.search(r'(\d+)min', txt_file.lower())
            if time_pattern:
                minutes = int(time_pattern.group(1))
                if minutes > 15:
                    print(f"Skipping {txt_file}: duration {minutes} minutes exceeds 15-minute limit")
                    continue

            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    script_content = f.read().strip()
                script_content = '\n'.join(line for line in script_content.split('\n') if line.strip())
                if not script_content:
                    continue
                
                num_speakers = self._get_num_speakers_from_script(script_content)
                self.example_scripts.append([num_speakers, script_content])
                print(f"Loaded example: {txt_file} with {num_speakers} speakers")
                
            except Exception as e:
                print(f"Error loading example script {txt_file}: {e}")
        
        if self.example_scripts:
            print(f"Successfully loaded {len(self.example_scripts)} example scripts")
        else:
            print("No example scripts were loaded")
    
    def _get_num_speakers_from_script(self, script: str) -> int:
        """Determine the number of unique speakers in a script."""
        import re
        speakers = set()
        
        lines = script.strip().split('\n')
        for line in lines:
            match = re.match(r'^Speaker\s+(\d+)\s*:', line.strip(), re.IGNORECASE)
            if match:
                speaker_id = int(match.group(1))
                speakers.add(speaker_id)
        
        if not speakers:
            return 1
        
        max_speaker = max(speakers)
        min_speaker = min(speakers)
        
        if min_speaker == 0:
            return max_speaker + 1
        else:
            return len(speakers)
    
    def add_stress_marks(self, script: str) -> str:
        """Add stress marks to Russian text while preserving Speaker labels."""
        if not self.accentizer:
            raise gr.Error("❌ RUAccent не загружен. Установите библиотеку: pip install ruaccent")
        
        def convert_plus_to_accent(text: str) -> str:
            """Convert +vowel notation to vowel with combining acute accent (U+0301)."""
            # Словарь замен: +гласная -> гласная с ударением
            replacements = {
                '+а': 'а́', '+А': 'А́',
                '+е': 'е́', '+Е': 'Е́',
                '+ё': 'ё́', '+Ё': 'Ё́',
                '+и': 'и́', '+И': 'И́',
                '+о': 'о́', '+О': 'О́',
                '+у': 'у́', '+У': 'У́',
                '+ы': 'ы́', '+Ы': 'Ы́',
                '+э': 'э́', '+Э': 'Э́',
                '+ю': 'ю́', '+Ю': 'Ю́',
                '+я': 'я́', '+Я': 'Я́',
            }
            
            for old, new in replacements.items():
                text = text.replace(old, new)
            
            return text
        
        try:
            lines = script.strip().split('\n')
            processed_lines = []
            
            for line in lines:
                line = line.strip()
                if not line:
                    processed_lines.append(line)
                    continue
                
                # Проверяем, начинается ли строка с "Speaker N:"
                match = re.match(r'^(Speaker\s+\d+\s*:\s*)(.*)', line, re.IGNORECASE)
                if match:
                    speaker_label = match.group(1)
                    text_content = match.group(2)
                    
                    # Применяем ударения только к тексту после метки диктора
                    if text_content.strip():
                        accented_text = self.accentizer.process_all(text_content)
                        accented_text = convert_plus_to_accent(accented_text)
                        processed_lines.append(speaker_label + accented_text)
                    else:
                        processed_lines.append(line)
                else:
                    # Если нет метки диктора, обрабатываем всю строку
                    accented_line = self.accentizer.process_all(line)
                    accented_line = convert_plus_to_accent(accented_line)
                    processed_lines.append(accented_line)
            
            return '\n'.join(processed_lines)
        except Exception as e:
            raise gr.Error(f"❌ Ошибка при расстановке ударений: {str(e)}")

def save_audio_formats(sample_rate, audio_data, formats_to_save):
    if not isinstance(audio_data, np.ndarray):
        raise TypeError("audio_data must be a numpy array")
    
    saved_files = {}
    temp_dir = tempfile.mkdtemp()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if audio_data.dtype == np.int16:
        audio_float = audio_data.astype(np.float32) / 32767.0
    else:
        audio_float = audio_data.astype(np.float32)
    
    if np.max(np.abs(audio_float)) > 1.0:
        audio_float /= np.max(np.abs(audio_float))

    for fmt in formats_to_save:
        try:
            filename = f"podcast_{timestamp}.{fmt.lower()}"
            filepath = os.path.join(temp_dir, filename)
            
            if fmt == 'WAV':
                sf.write(filepath, audio_float, sample_rate, subtype='PCM_16')
                saved_files['WAV'] = filepath
            elif fmt == 'FLAC':
                sf.write(filepath, audio_float, sample_rate, format='FLAC')
                saved_files['FLAC'] = filepath
            elif fmt == 'MP3':
                if AudioSegment is None:
                    print("Skipping MP3 export: pydub is not available.")
                    continue
                audio_int16 = (audio_float * 32767).astype(np.int16)
                segment = AudioSegment(
                    audio_int16.tobytes(), frame_rate=sample_rate,
                    sample_width=audio_int16.dtype.itemsize, channels=1
                )
                segment.export(filepath, format="mp3", bitrate="192k")
                saved_files['MP3'] = filepath
        except Exception as e:
            print(f"Error saving to {fmt}: {e}")

    updates = []
    for fmt in ['MP3', 'WAV', 'FLAC']:
        if fmt in saved_files:
            updates.append(gr.update(value=saved_files[fmt], visible=True))
        else:
            updates.append(gr.update(visible=False))
            
    return updates

def create_demo_interface(demo_instance: VibeVoiceDemo):
    """Create the Gradio interface with streaming support."""
    
    custom_css = """
    /* --- LIGHT MODE --- */
    #main-container {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%) !important;
    }
    #main-container .settings-card, #main-container .generation-card {
        background: rgba(255, 255, 255, 0.8) !important;
        border: 1px solid rgba(203, 213, 225, 0.5) !important;
        box-shadow: 0 8px 32px rgba(100, 116, 139, 0.1) !important;
        border-radius: 16px !important;
    }
    #main-container .speaker-block {
        background: #e2e8f0 !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 12px !important;
    }
    #main-container textarea, #main-container input[type=text] {
        background-color: #ffffff !important;
        color: #1e293b !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 8px !important;
    }
    #main-container textarea:disabled {
        background-color: #f1f5f9 !important;
        color: #64748b !important;
    }
    #main-container label {
        color: #1e293b !important;
        font-weight: 600 !important;
    }
    
    /* --- DARK MODE --- */
    .dark #main-container {
        background: #020617 !important;
    }
    .dark #main-container .settings-card, .dark #main-container .generation-card {
        background: rgba(15, 23, 42, 0.8) !important;
        border: 1px solid #1e293b !important;
        box-shadow: 0 8px 40px rgba(0, 0, 0, 0.3) !important;
    }
    .dark #main-container .speaker-block {
        background: #0f172a !important;
        border: 1px solid #1e293b !important;
    }
    .dark #main-container textarea, .dark #main-container input[type=text] {
        background-color: #0f172a !important;
        color: #e2e8f0 !important;
        border: 1px solid #334155 !important;
    }
    .dark #main-container textarea:disabled {
        background-color: #020617 !important;
        color: #94a3b8 !important;
    }
    .dark #main-container label {
        color: #cbd5e1 !important;
    }
    
    /* --- HEADER (общий для обеих тем) --- */
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem; border-radius: 20px; margin-bottom: 2rem; text-align: center;
        box-shadow: 0 10px 40px rgba(102, 126, 234, 0.3);
    }
    .main-header h1 { color: white; font-size: 2.5rem; font-weight: 700; margin: 0; text-shadow: 0 2px 4px rgba(0,0,0,0.3); }
    .main-header p { color: rgba(255,255,255,0.9); font-size: 1.1rem; margin: 0.5rem 0 0 0; }
    
    /* --- VOICE CHECKBOX GROUP --- */
    .voice-checkbox-group {
        max-height: 400px;
        overflow-y: auto;
        padding: 10px;
        border: 1px solid #cbd5e1;
        border-radius: 8px;
    }
    .dark .voice-checkbox-group {
        border-color: #334155;
    }
    """
    
    with gr.Blocks(
        title="VibeVoice Подкастер PRO",
        css=custom_css,
        theme=gr.themes.Soft(primary_hue="blue", secondary_hue="purple", neutral_hue="slate"),
        elem_id="main-container"
    ) as interface:
        
        gr.HTML("""
        <div class="main-header">
            <h1>🎙️ VibeVoice Подкастер PRO v3</h1>
            <p>Портативная версия от <a href="https://t.me/li_aeron" target="_blank" style="color: white;">LeeAeron</a> и <a href="https://t.me/ruweb24" target="_blank" style="color: white;"> Slait</a></p> на базе версии <a href="https://t.me/neuroport" target="_blank" style="color: white;">👾 НЕЙРО-СОФТ</a></p>
        </div>
        """)
        
        with gr.Row():
            with gr.Column(scale=1, elem_classes="settings-card"):
                gr.Markdown("### 🎛️ **Настройки подкаста**")
                
                # НОВОЕ: Выпадающий список для выбора модели
                model_dropdown = gr.Dropdown(
                    choices=list(MODEL_REPOS.keys()),
                    value="VibeVoice-7B",  # Модель по умолчанию
                    label="🧠 Выбор модели",
                    info="Модель будет скачана при первой генерации."
                )

                int4_checkbox = gr.Checkbox(label="INT4-эмуляция", value=True, info="Эмуляция 4bit-квантования для экономии VRAM/RAM")

                num_speakers = gr.Slider(
                    minimum=1, maximum=4, value=2, step=1,
                    label="Количество дикторов",
                )
                
                gr.Markdown("### 🎭 **Выбор дикторов**")
                
                available_speaker_names = list(demo_instance.available_voices.keys())
                default_speakers = available_speaker_names[:4]

                speaker_audio_inputs = []
                speaker_blocks_ui = []
                
                for i in range(4):
                    with gr.Column(visible=(i < 2), elem_classes="speaker-block") as speaker_block:
                        gr.Markdown(f"**Диктор {i+1}**")
                        default_preset_name = default_speakers[i] if i < len(default_speakers) else None
                        default_audio_path = demo_instance.available_voices.get(default_preset_name)
                        
                        preset_dropdown = gr.Dropdown(
                            choices=["- Загрузить свой файл -"] + available_speaker_names,
                            value=default_preset_name,
                            label="Выбрать пресет голоса",
                        )
                        
                        audio_input = gr.Audio(
                            label="Аудиофайл голоса",
                            value=default_audio_path,
                            type="filepath",
                            sources=["upload", "microphone"]
                        )
                        
                        def update_audio_from_preset_factory(demo_ref):
                            def update_audio(preset_name):
                                if preset_name and preset_name != "- Загрузить свой файл -":
                                    return gr.update(value=demo_ref.voice_presets.get(preset_name))
                                return gr.update(value=None)
                            return update_audio
                        
                        preset_dropdown.change(
                            fn=update_audio_from_preset_factory(demo_instance),
                            inputs=preset_dropdown,
                            outputs=audio_input,
                            queue=False
                        )

                        speaker_audio_inputs.append(audio_input)
                        speaker_blocks_ui.append(speaker_block)

                # Voice Download Section
                with gr.Accordion("🎤 Загрузить новые голоса", open=False):
                    download_voices_btn = gr.Button("📥 Загрузить список голосов", variant="primary")
                    voice_list_status = gr.Textbox(label="Статус", interactive=False, lines=2)
                    
                    with gr.Column(visible=False) as voice_selection_box:
                        gr.Markdown("### Выберите голоса для загрузки")
                        with gr.Row():
                            select_all_btn = gr.Button("✅ Выбрать все", size="sm")
                            deselect_all_btn = gr.Button("❌ Убрать все", size="sm")
                        
                        voice_checkboxes = gr.CheckboxGroup(
                            choices=[],
                            label="Доступные голоса",
                            elem_classes="voice-checkbox-group"
                        )
                        
                        download_selected_btn = gr.Button("⬇️ Загрузить выбранные голоса", variant="primary")
                        gr.Markdown("💡 **Подсказка:** Новые голоса появятся после перезагрузки программы.")
                        download_result = gr.Textbox(label="Результат загрузки", interactive=False, lines=4)
                    
                    # Voice download logic
                    voices_data = gr.State([])
                    
                    def handle_download_list():
                        file_path, status = download_voice_list()
                        if file_path:
                            voices, parse_status = parse_voice_list(file_path)
                            if voices:
                                voice_names = [v["name"] for v in voices]
                                return (
                                    status + "\n" + parse_status,
                                    gr.update(visible=True),
                                    gr.update(choices=voice_names, value=[]),
                                    voices
                                )
                        return status, gr.update(visible=False), gr.update(choices=[], value=[]), []
                    
                    def select_all_voices(current_choices):
                        return current_choices
                    
                    def deselect_all_voices():
                        return []
                    
                    def handle_download_selected(selected, all_voices_data):
                        if not selected:
                            return "⚠️ Пожалуйста, выберите хотя бы один голос."
                        return download_selected_voices(selected, all_voices_data)
                    
                    download_voices_btn.click(
                        fn=handle_download_list,
                        outputs=[voice_list_status, voice_selection_box, voice_checkboxes, voices_data],
                        queue=False
                    )
                    
                    select_all_btn.click(
                        fn=select_all_voices,
                        inputs=[voice_checkboxes],
                        outputs=voice_checkboxes,
                        queue=False
                    )
                    
                    deselect_all_btn.click(
                        fn=deselect_all_voices,
                        outputs=voice_checkboxes,
                        queue=False
                    )
                    
                    download_selected_btn.click(
                        fn=handle_download_selected,
                        inputs=[voice_checkboxes, voices_data],
                        outputs=download_result,
                        queue=True
                    )

                with gr.Accordion("⚙️ Параметры генерации", open=True):
                    # НОВОЕ: Чекбокс для включения bnb_config
                    cfg_scale = gr.Slider(minimum=1.0, maximum=3.0, value=1.3, step=0.05, label="Сила ведения (CFG)")
                    inference_steps = gr.Slider(minimum=4, maximum=50, value=demo_instance.initial_inference_steps, step=1, label="Шаги вывода")
                    with gr.Row():
                        seed_input = gr.Number(
                            value=-1,
                            label="Seed",
                            info="Укажите число для фиксированной генерации. -1 = случайный seed"
                        )
                        random_seed_btn = gr.Button("🎲 Случайный Seed", size="sm", variant="secondary")

                    # Подключение кнопки к генератору
                    def generate_random_seed():
                        import random
                        return random.randint(0, 2**32 - 1)

                    random_seed_btn.click(fn=generate_random_seed, outputs=seed_input, queue=False)

                    refresh_negative = gr.Checkbox(label="Обновление 'негатива'", value=True)
                    gr.Markdown("#### **Настройки вариативности**")
                    do_sample = gr.Checkbox(label="Включить семплирование", value=False)
                    with gr.Group(visible=False) as sampling_params:
                        temperature = gr.Slider(minimum=0.1, maximum=1.5, value=0.8, step=0.05, label="Температура")
                        top_p = gr.Slider(minimum=0.1, maximum=1.0, value=0.95, step=0.05, label="Top-p семплирование")
                    do_sample.change(fn=lambda x: gr.update(visible=x), inputs=do_sample, outputs=sampling_params, queue=False)
            
            with gr.Column(scale=2, elem_classes="generation-card"):
                gr.Markdown("### 📝 **Ввод сценария**")
                script_input = gr.Textbox(
                    label="Сценарий разговора",
                    value="Speaker 1: Слушай, а ты уже подписался на телеграм канал Нейро-софт? \nSpeaker 2: Конечно, там не только репаки и портативки полезных нейросетей, но и крутые новостные дайджесты и мемы!",
                    placeholder="Введите сценарий вашего подкаста здесь...",
                    lines=12, max_lines=20,
                )
                with gr.Row():
                    random_example_btn = gr.Button("🎲 Случайный пример", size="lg", variant="secondary", scale=1)
                    add_stress_btn = gr.Button("📍 Расставить ударения", size="lg", variant="secondary", scale=1)
                    generate_btn = gr.Button("🚀 Сгенерировать подкаст", size="lg", variant="primary", scale=2)
                
                stop_btn = gr.Button("🛑 Остановить генерацию", size="lg", variant="stop", visible=False)
                streaming_status = gr.HTML(visible=False)
                
                gr.Markdown("### 🎵 **Сгенерированный подкаст**")
                autoplay_checkbox = gr.Checkbox(
                    label="AutoPlay",
                    value=False,
                    info="Автоматическое воспроизведение аудио после генерации"
                )
                audio_output = gr.Audio(label="Потоковое аудио (в реальном времени)", type="numpy", streaming=True, autoplay=False, show_download_button=False)
                complete_audio_output = gr.Audio(label="Готовый подкаст (для прослушивания)", type="numpy", autoplay=False, show_download_button=True)
                
                with gr.Column(elem_classes="download-section") as download_box:
                    save_formats = gr.CheckboxGroup(["MP3", "WAV", "FLAC"], value=["MP3"], label="Форматы для сохранения")
                    with gr.Row(visible=False) as download_links_row:
                        download_mp3 = gr.File(label="Скачать MP3", visible=False)
                        download_wav = gr.File(label="Скачать WAV", visible=False)
                        download_flac = gr.File(label="Скачать FLAC", visible=False)
                
                log_output = gr.Textbox(label="Лог генерации", lines=8, max_lines=15, interactive=False)

        all_generation_inputs = [
            model_dropdown, # НОВОЕ: Добавляем выбор модели в список входов
            num_speakers, script_input, 
            *speaker_audio_inputs, 
            cfg_scale, inference_steps, 
            seed_input, 
            do_sample, temperature, top_p, refresh_negative,
            int4_checkbox, 
            save_formats,
            autoplay_checkbox
        ]
        all_download_buttons = [download_mp3, download_wav, download_flac]
        
        import gc
        import torch

        def reload_model_on_int4_toggle(use_int4_val):
            # 🧹 Очистка памяти перед перезагрузкой модели
            if demo_instance.model is not None:
                del demo_instance.model
            if demo_instance.processor is not None:
                del demo_instance.processor
            demo_instance.model = None
            demo_instance.processor = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            demo_instance.use_int4 = bool(use_int4_val)
            demo_instance.load_model(demo_instance.current_model_name or "VibeVoice-7B", use_int4=demo_instance.use_int4)
            return f"🔄 Модель перезагружена. INT4: {'ВКЛ' if use_int4_val else 'ВЫКЛ'}"

            # 🧹 Очистка памяти после перезагрузки модели
            if demo_instance.model is not None:
                del demo_instance.model
            if demo_instance.processor is not None:
                del demo_instance.processor
            demo_instance.model = None
            demo_instance.processor = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        int4_checkbox.change(fn=reload_model_on_int4_toggle, inputs=int4_checkbox, outputs=log_output)

        def update_speaker_visibility(num_speakers_val):
            return [gr.update(visible=(i < num_speakers_val)) for i in range(len(speaker_blocks_ui))]
        
        num_speakers.change(fn=update_speaker_visibility, inputs=num_speakers, outputs=speaker_blocks_ui)
        
        def generate_podcast_wrapper(*args):
            try:
                # Распаковываем аргументы, учитывая новый `model_name`
                (model_name_val, num_speakers_val, script_val, speaker_1, speaker_2, speaker_3, speaker_4,
                 cfg_scale_val, inference_steps_val, seed_val, do_sample_val, temperature_val, top_p_val, refresh_negative_val,
                 int4_checkbox_val,
                 formats_to_save, autoplay_checkbox_val) = args

                # 🎲 Установка сидов
                import random, torch
                if seed_val is None or int(seed_val) < 0:
                    seed_val = random.randint(0, 2**32 - 1)
                    print(f"[Seed] Случайный сид: {seed_val}")
                else:
                    print(f"[Seed] Фиксированный сид: {seed_val}")
                torch.manual_seed(seed_val)
                try:
                    import numpy as np
                    np.random.seed(seed_val)
                except ImportError:
                    pass

                yield None, gr.update(value=None), "🎙️ Начинаю генерацию...", gr.update(visible=True), gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), *[gr.update(visible=False)]*3
                
                final_audio_data = None
                
                demo_instance.use_int4 = bool(int4_checkbox_val)

                stream_generator = demo_instance.generate_podcast_streaming(
                    model_name=model_name_val,
                    num_speakers=int(num_speakers_val), script=script_val,
                    speaker_1_audio=speaker_1, speaker_2_audio=speaker_2, speaker_3_audio=speaker_3, speaker_4_audio=speaker_4,
                    cfg_scale=cfg_scale_val, inference_steps=int(inference_steps_val),
                    do_sample=do_sample_val, temperature=temperature_val, top_p=top_p_val, refresh_negative=refresh_negative_val,
                    seed=seed_val, use_int4=bool(int4_checkbox_val)
                )

                for streaming_audio, complete_audio, log, streaming_visible in stream_generator:
                    if complete_audio is not None:
                        final_audio_data = complete_audio
                    
                    if streaming_audio is not None:
                         yield streaming_audio, gr.update(value=None, visible=False), log, streaming_visible, gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), *[gr.update(visible=False)]*3
                    else:
                         yield None, gr.update(value=complete_audio, visible=(complete_audio is not None), autoplay=autoplay_checkbox_val),log, streaming_visible, gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), *[gr.update(visible=False)]*3

                if final_audio_data:
                    sample_rate, audio_np = final_audio_data
                    download_updates = save_audio_formats(sample_rate, audio_np, formats_to_save)
                    yield (None, gr.update(value=final_audio_data, visible=True, autoplay=autoplay_checkbox_val), log, 
                        gr.update(visible=False), gr.update(visible=True), gr.update(visible=False),
                        gr.update(visible=True), *download_updates)

                else:
                    yield None, None, log, gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), *[gr.update(visible=False)]*3

            except Exception as e:
                error_msg = f"❌ Критическая ошибка в оболочке: {e}"
                print(error_msg, traceback.format_exc())
                yield None, None, error_msg, gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), *[gr.update(visible=False)]*3
        
        def stop_generation_handler():
            demo_instance.stop_audio_generation()
            return "🛑 Генерация остановлена.", gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)
        
        def clear_outputs_before_gen():
            return None, gr.update(value=None, visible=False), gr.update(visible=False), *[gr.update(visible=False)]*3

        generate_btn.click(
            fn=clear_outputs_before_gen, outputs=[audio_output, complete_audio_output, download_links_row, *all_download_buttons], queue=False
        ).then(
            fn=lambda: (gr.update(visible=False), gr.update(visible=True)), outputs=[generate_btn, stop_btn], queue=False
        ).then(
            fn=generate_podcast_wrapper,
            inputs=all_generation_inputs,
            outputs=[audio_output, complete_audio_output, log_output, streaming_status, generate_btn, stop_btn, download_links_row, *all_download_buttons],
        )
        
        stop_btn.click(
            fn=stop_generation_handler, outputs=[log_output, streaming_status, generate_btn, stop_btn], queue=False
        ).then(
            fn=lambda: (None, None), outputs=[audio_output, complete_audio_output], queue=False
        )
        
        def load_random_example():
            import random
            examples = (demo_instance.example_scripts if hasattr(demo_instance, 'example_scripts') and demo_instance.example_scripts
                        else [[2, "Speaker 0: Добро пожаловать!\nSpeaker 1: Спасибо!"]])
            if examples:
                num_speakers_value, script_value = random.choice(examples)
                return num_speakers_value, script_value
            return 2, ""
        
        random_example_btn.click(fn=load_random_example, outputs=[num_speakers, script_input], queue=False)
        
        def add_stress_marks_handler(script):
            """Handler for adding stress marks to the script."""
            if not script.strip():
                raise gr.Error("❌ Пожалуйста, введите текст сценария.")
            return demo_instance.add_stress_marks(script)
        
        add_stress_btn.click(fn=add_stress_marks_handler, inputs=script_input, outputs=script_input, queue=False)
        
        gr.Examples(examples=demo_instance.example_scripts or [], inputs=[num_speakers, script_input], label="Попробуйте эти примеры сценариев:")
        
        gr.Markdown("""
        ### 💡 **Советы по использованию**
        - Нажмите **🚀 Сгенерировать подкаст**, чтобы начать.
        - **Потоковое аудио** играет по мере создания.
        - **Готовый подкаст** появится ниже после завершения.
        - **Шаги вывода**: 10-20 для баланса скорости и качества.
        - **Включить семплирование**: Включите для более 'живой', но менее предсказуемой речи.
        """,
        elem_classes="generation-card",
        )
        
        gr.Markdown("""
        ### ℹ️ **Авторы:**
        <a href=@nerual_dreming style="color: blue;"><a href="https://t.me/nerual_dreming" target="_blank" style="color: #9932CC;">👾 Nerual Dreming</a>, основатель онлайн нейросети <a href=https://artgeneration.me style="color: blue;"><a href="https://artgeneration.me" target="_blank" style="color: blue;">ArtGeneration.me</a>, техноблогер и нейро-евангелист, - база, основной код.</p>
        <a href=@li_aeron style="color: blue;"><a href="https://t.me/li_aeron" target="_blank" style="color: #9932CC;"> 😈 LeeAeron</a>, автор <a href=https://github.com/LeeAeron/stable-diffusion-webui-fastforge style="color: blue;"><a href="https://github.com/LeeAeron/stable-diffusion-webui-fastforge" target="_blank" style="color: blue;">FastForge</a>, - код инсталлятора, хранилище hugginface, тесты.</p>
        <a href=@ruweb24 style="color: blue;"><a href="https://t.me/ruweb24" target="_blank" style="color: #9932CC;"> Slait</a></p>
        """,
        elem_classes="generation-card",
        )

    return interface

def convert_to_16_bit_wav(data):
    if torch.is_tensor(data): data = data.detach().cpu().numpy()
    data = np.array(data)
    if np.max(np.abs(data)) > 1.0: data = data / np.max(np.abs(data))
    return (data * 32767).astype(np.int16)

def download_voice_list():
    """Download the voice list file from the server."""
    try:
        url = "http://thebrand.su/Vibevoice/download_voices.txt"
        demo_dir = os.path.join(os.path.dirname(__file__))
        save_path = os.path.join(demo_dir, "download_voices.txt")
        
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(response.text)
        
        return save_path, "✅ Список голосов успешно загружен!"
    except Exception as e:
        return None, f"❌ Ошибка при загрузке списка: {str(e)}"

def parse_voice_list(file_path):
    """Parse the voice list file and extract MP3 and TXT URLs and names."""
    try:
        if not file_path or not os.path.exists(file_path):
            return [], "❌ Файл списка голосов не найден."
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Find all MP3 URLs
        mp3_pattern = r'https?://[^\s]+\.mp3(?:\?[^\s]*)?'
        mp3_urls = re.findall(mp3_pattern, content)
        
        # Find all TXT URLs
        txt_pattern = r'https?://[^\s]+\.txt(?:\?[^\s]*)?'
        txt_urls = re.findall(txt_pattern, content)
        
        # Extract voice names from MP3 URLs and match with TXT URLs
        voices = []
        for mp3_url in mp3_urls:
            # Extract filename from URL
            parsed = urlparse(mp3_url)
            filename = os.path.basename(parsed.path)
            # Remove .mp3 extension
            voice_name = os.path.splitext(filename)[0]
            
            # Try to find corresponding TXT URL
            txt_url = None
            for txt in txt_urls:
                if voice_name in txt:
                    txt_url = txt
                    break
            
            voices.append({"name": voice_name, "mp3_url": mp3_url, "txt_url": txt_url})
        
        if not voices:
            return [], "❌ В файле не найдено ссылок на MP3."
        
        return voices, f"✅ Найдено {len(voices)} голосов."
    except Exception as e:
        return [], f"❌ Ошибка при парсинге файла: {str(e)}"

def download_selected_voices(selected_voices, all_voices):
    """Download selected voice files (MP3 and TXT) to the voices directory."""
    try:
        demo_dir = os.path.dirname(__file__)
        voices_dir = os.path.join(demo_dir, "voices")
        
        # Create voices directory if it doesn't exist
        os.makedirs(voices_dir, exist_ok=True)
        
        downloaded_mp3 = 0
        downloaded_txt = 0
        failed = 0
        skipped = 0
        
        for voice in all_voices:
            if voice["name"] not in selected_voices:
                continue
            
            mp3_path = os.path.join(voices_dir, f"{voice['name']}.mp3")
            txt_path = os.path.join(voices_dir, f"{voice['name']}.txt")
            
            # Download MP3 file
            if os.path.exists(mp3_path):
                skipped += 1
            else:
                try:
                    response = requests.get(voice["mp3_url"], timeout=60, stream=True)
                    response.raise_for_status()
                    
                    with open(mp3_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                    
                    downloaded_mp3 += 1
                except Exception as e:
                    print(f"Failed to download MP3 {voice['name']}: {e}")
                    failed += 1
            
            # Download TXT file if URL exists
            if voice["txt_url"]:
                if not os.path.exists(txt_path):
                    try:
                        response = requests.get(voice["txt_url"], timeout=60, stream=True)
                        response.raise_for_status()
                        
                        with open(txt_path, 'wb') as f:
                            for chunk in response.iter_content(chunk_size=8192):
                                if chunk:
                                    f.write(chunk)
                        
                        downloaded_txt += 1
                    except Exception as e:
                        print(f"Failed to download TXT {voice['name']}: {e}")
        
        result_msg = f"✅ Загрузка завершена!\n"
        result_msg += f"📥 Загружено MP3: {downloaded_mp3}\n"
        if downloaded_txt > 0:
            result_msg += f"📄 Загружено TXT: {downloaded_txt}\n"
        if skipped > 0:
            result_msg += f"⏭️ Пропущено (уже существуют): {skipped}\n"
        if failed > 0:
            result_msg += f"❌ Ошибок: {failed}\n"
        result_msg += f"\n💡 Перезагрузите программу, чтобы увидеть новые голоса."
        
        return result_msg
    except Exception as e:
        return f"❌ Ошибка при загрузке голосов: {str(e)}"

def parse_args():
    parser = argparse.ArgumentParser(description="VibeVoice Gradio Demo")
    # Удален аргумент --model_path
    parser.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")), help="Device: cuda | mps | cpu")
    parser.add_argument("--inference_steps", type=int, default=10, help="Default number of inference steps")
    parser.add_argument("--share", action="store_true", help="Share the demo publicly")
    parser.add_argument("--port", type=int, default=7860, help="Port to run on")
    return parser.parse_args()

def main():
    args = parse_args()
    print("🎙️ Initializing VibeVoice Demo with Streaming and On-Demand Model Loading...")
    # Инициализация без model_path
    demo_instance = VibeVoiceDemo(
        device=args.device, inference_steps=args.inference_steps
    )
    interface = create_demo_interface(demo_instance)
    print(f"🚀 Launching demo on port {args.port}")
    print(f"✅ Application ready. Models will be downloaded on first use.")
    print(f"🎭 Available voices: {len(demo_instance.available_voices)}")
    print(f"🔴 Streaming mode: ENABLED")
    print(f"🔒 Session isolation: ENABLED")
    try:
        interface.queue(max_size=20, default_concurrency_limit=1).launch(
            share=args.share, server_name="0.0.0.0" if args.share else "127.0.0.1",
            show_error=True, show_api=False, inbrowser=True
        )
    except KeyboardInterrupt: print("\n🛑 Shutting down gracefully...")
    except Exception as e: print(f"❌ Server error: {e}"); raise

if __name__ == "__main__":
    main()