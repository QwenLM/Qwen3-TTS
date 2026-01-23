# -*- coding: utf-8 -*-

import argparse
import tempfile
import traceback
import time
from pathlib import Path
import numpy as np

import gradio as gr
import librosa
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoFeatureExtractor
from transformers.utils import logging

logging.set_verbosity_info()
logger = logging.get_logger(__name__)

class BorealisDemo:
    def __init__(self, device: str = "cuda"):
        self.device = self._get_device(device)
        self.model = None
        self.tokenizer = None
        self.extractor = None
        
        self.default_generation_params = {
            "do_sample": True,
            "top_p": 0.9,
            "top_k": 50,
            "temperature": 0.2,
        }
        
        self.load_model()
        self.model_max_chunk_s = 30
        self.sampling_rate = 16000

    def _get_device(self, requested_device: str) -> str:
        if requested_device == "cuda" and not torch.cuda.is_available():
            print("Warning: CUDA not available. Falling back to CPU.")
            return "cpu"
        return requested_device

    def load_model(self):
        model_name = "Vikhrmodels/Borealis"
        print(f"Loading model '{model_name}' to device '{self.device}'...")
        try:
            self.model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.extractor = AutoFeatureExtractor.from_pretrained(model_name)
            self.model.eval()
            print("✅ Model loaded successfully.")
        except Exception as e:
            print(f"❌ Failed to load model. Error: {e}")
            print(traceback.format_exc())
            raise gr.Error("Критическая ошибка: не удалось загрузить модель. Проверьте интернет-соединение и зависимости.")

    def transcribe(self, audio_filepath: str, do_sample: bool, temperature: float, top_p: float, top_k: int):
        if not audio_filepath:
            raise gr.Error("Файл не предоставлен. Загрузите аудио или видео.")

        current_generation_params = {
            "max_new_tokens": 350, 
            "do_sample": do_sample,
            "top_p": top_p,
            "top_k": top_k,
            "temperature": temperature,
        }
        
        model_max_chunk_samples = int(self.model_max_chunk_s * self.sampling_rate)

        print(f"Transcribing file: {audio_filepath} with params: {current_generation_params}")
        
        try:
            start_time = time.time()
            waveform, sr = librosa.load(audio_filepath, sr=self.sampling_rate, mono=True)
            
            # 1. Нормализуем громкость для стабильной работы детектора тишины
            if np.any(waveform):
                waveform = librosa.util.normalize(waveform)
            audio_duration = len(waveform) / self.sampling_rate

            # 2. Находим речевые сегменты
            top_db = 30
            speech_intervals = librosa.effects.split(waveform, top_db=top_db)
            
            # 3. Обработка и принудительное разделение слишком длинных сегментов
            padding_s = 0.25
            padding_samples = int(padding_s * self.sampling_rate)
            
            final_segments = []
            if speech_intervals.size > 0:
                for start, end in speech_intervals:
                    padded_start = max(0, start - padding_samples)
                    padded_end = min(len(waveform), end + padding_samples)
                    duration = padded_end - padded_start

                    # Если сегмент сам по себе длиннее 30 сек, принудительно его режем
                    if duration > model_max_chunk_samples:
                        print(f"  > Found a long speech segment ({duration/self.sampling_rate:.2f}s), splitting it...")
                        offset = padded_start
                        while offset < padded_end:
                            segment_end = min(offset + model_max_chunk_samples, padded_end)
                            final_segments.append(waveform[offset:segment_end])
                            offset += model_max_chunk_samples
                    else:
                        final_segments.append(waveform[padded_start:padded_end])
            
            # 4. Новая, надежная логика сборки чанков
            chunks = []
            if final_segments:
                current_chunk = np.array([], dtype=np.float32)
                for segment_waveform in final_segments:
                    if len(current_chunk) + len(segment_waveform) <= model_max_chunk_samples:
                        current_chunk = np.concatenate([current_chunk, segment_waveform])
                    else:
                        if len(current_chunk) > 0:
                            chunks.append(current_chunk)
                        current_chunk = segment_waveform
                if len(current_chunk) > 0:
                    chunks.append(current_chunk)


            # 5. Fallback: если речь не найдена (например, музыка или шум), просто режем на куски
            if not chunks and len(waveform) > 0:
                print("  > No speech detected, falling back to simple chunking.")
                for i in range(0, len(waveform), model_max_chunk_samples):
                    chunks.append(waveform[i:i + model_max_chunk_samples])

            num_chunks = len(chunks)
            print(f"Audio intelligently split into {num_chunks} chunks.")
            
            full_transcript = []

            for i, audio_chunk in enumerate(chunks):
                progress_log = f"Обработка чанка {i + 1}/{num_chunks}..."
                
                yield {
                    "text": "\n\n".join(full_transcript).strip(),
                    "audio_duration": audio_duration,
                    "processing_time": time.time() - start_time,
                    "progress": progress_log
                }
                print(progress_log)

                final_chunk = audio_chunk
                current_len = len(final_chunk)
                if current_len < model_max_chunk_samples:
                    padding = np.zeros(model_max_chunk_samples - current_len, dtype=np.float32)
                    final_chunk = np.concatenate([final_chunk, padding])

                proc = self.extractor(
                    final_chunk,
                    sampling_rate=self.sampling_rate,
                    padding="max_length",
                    max_length=model_max_chunk_samples,
                    return_attention_mask=True,
                    return_tensors="pt",
                )
            
                mel = proc.input_features.to(self.device)
                att_mask = proc.attention_mask.to(self.device)

                with torch.inference_mode():
                    result_list = self.model.generate(mel=mel, att_mask=att_mask, **current_generation_params)
                    
                    text_chunk = ""
                    if isinstance(result_list, list) and len(result_list) > 0 and isinstance(result_list[0], str):
                        text_chunk = result_list[0].strip()
                    
                    print(f"  > Chunk {i + 1} result: '{text_chunk}'")

                    if text_chunk:
                        full_transcript.append(text_chunk)
            
            final_text = "\n\n".join(full_transcript).strip()
            yield {
                "text": final_text,
                "audio_duration": audio_duration,
                "processing_time": time.time() - start_time,
                "progress": f"Завершено {num_chunks}/{num_chunks} чанков."
            }

        except Exception as e:
            print(f"❌ Error during transcription: {e}")
            print(traceback.format_exc())
            raise gr.Error(f"Произошла ошибка во время обработки файла: {e}")


def create_ui(demo_instance: BorealisDemo):
    custom_css = """
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem; border-radius: 15px; margin-bottom: 1.5rem; text-align: center;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.2);
    }
    .main-header h1 { color: white; font-size: 2.2rem; font-weight: 700; margin: 0; text-shadow: 0 2px 4px rgba(0,0,0,0.3); }
    .main-header p { color: rgba(255,255,255,0.9); font-size: 1.1rem; margin: 0.5rem 0 0 0; }
    """

    with gr.Blocks(theme=gr.themes.Soft(primary_hue="purple", neutral_hue="slate"), css=custom_css) as interface:
        
        gr.HTML("""
        <div class="main-header">
            <h1>🎙️ Vikhr Borealis Speech-to-Text</h1>
            <p>Портативная версия от <a href="https://t.me/neuroport" target="_blank" style="color: white;">👾 НЕЙРО-СОФТ</a></p>
            <p style="font-size: 0.9rem; opacity: 0.8;">Собрал <a href="https://t.me/nerual_dreming" target="_blank" style="color: white;">Nerual Dreming</a> - основатель <a href="https://artgeneration.me/" target="_blank" style="color: white;">ArtGeneration.me</a>, техноблогер и нейро-евангелист.</p>
        </div>
        """)

        with gr.Row():
            with gr.Column(scale=1):
                # Сохраняем состояние активной вкладки
                selected_tab = gr.State("audio")

                with gr.Tabs() as tabs:
                    with gr.TabItem("🎧 Аудио") as audio_tab:
                        audio_input = gr.Audio(label="Аудиовход", type="filepath", sources=["upload", "microphone"])
                    with gr.TabItem("🎬 Видео") as video_tab:
                        video_input = gr.Video(label="Видеовход", sources=["upload"])
                
                # Слушатели для обновления состояния при смене вкладки
                audio_tab.select(lambda: "audio", None, selected_tab, queue=False)
                video_tab.select(lambda: "video", None, selected_tab, queue=False)

                with gr.Accordion("⚙️ Дополнительные настройки", open=False):
                    params = demo_instance.default_generation_params
                    do_sample_checkbox = gr.Checkbox(label="Использовать сэмплирование (do_sample)", value=params["do_sample"])
                    temperature_slider = gr.Slider(minimum=0.0, maximum=2.0, value=params["temperature"], step=0.1, label="Температура (temperature)")
                    top_p_slider = gr.Slider(minimum=0.0, maximum=1.0, value=params["top_p"], step=0.05, label="Top-p")
                    top_k_slider = gr.Slider(minimum=0, maximum=200, value=params["top_k"], step=1, label="Top-k")
                
                transcribe_button = gr.Button("🚀 Распознать речь", variant="primary")

            with gr.Column(scale=2):
                output_textbox = gr.Textbox(label="Результат расшифровки", lines=15, show_copy_button=True)
                log_output = gr.Textbox(label="Лог", lines=4, interactive=False)
                download_file = gr.File(label="Скачать результат (.txt)", visible=False)

        def transcription_handler(active_tab, audio_file, video_file, do_sample, temperature, top_p, top_k):
            # Определяем, какой файл использовать, на основе активной вкладки
            if active_tab == "video":
                filepath = video_file
            else:
                filepath = audio_file

            if filepath is None:
                yield "", "Файл не предоставлен.", gr.update(visible=False), gr.update(interactive=True)
                return

            yield "", "🔄 Инициализация...", gr.update(visible=False), gr.update(interactive=False)
            
            final_result_data = None
            try:
                for progress_data in demo_instance.transcribe(filepath, do_sample, temperature, top_p, top_k):
                    partial_text = progress_data["text"]
                    progress_log = progress_data["progress"]
                    yield partial_text, f"🔄 {progress_log}", gr.update(visible=False), gr.update(interactive=False)
                    final_result_data = progress_data

                result_text = final_result_data["text"]
                log_message = (
                    f"✅ Готово!\n"
                    f"⏱️ Длительность аудио: {final_result_data['audio_duration']:.2f} сек.\n"
                    f"⚙️ Время обработки: {final_result_data['processing_time']:.2f} сек.\n"
                    f"✍️ Символов в результате: {len(result_text)}"
                )

                with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as tmp_file:
                    tmp_file.write(result_text)
                    temp_filepath = tmp_file.name
                
                yield result_text, log_message, gr.update(value=temp_filepath, visible=True), gr.update(interactive=True)
            
            except gr.Error as e:
                yield "", f"❌ Ошибка: {e}", gr.update(visible=False), gr.update(interactive=True)
            except Exception as e:
                yield "", f"❌ Непредвиденная ошибка: {e}", gr.update(visible=False), gr.update(interactive=True)

        
        transcribe_button.click(
            fn=transcription_handler,
            inputs=[selected_tab, audio_input, video_input, do_sample_checkbox, temperature_slider, top_p_slider, top_k_slider],
            outputs=[output_textbox, log_output, download_file, transcribe_button]
        )
        
        gr.Markdown("### 💡 **Как использовать**\n1. Выберите вкладку 'Аудио' или 'Видео'.\n2. Загрузите файл или запишите голос.\n3. (Опционально) Раскройте меню 'Дополнительные настройки' и измените параметры.\n4. Нажмите кнопку **🚀 Распознать речь** (она заблокируется до окончания работы).\n5. Результат появится в текстовом поле справа в виде абзацев.")

    return interface


def main():
    parser = argparse.ArgumentParser(description="Borealis Gradio Demo")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use: 'cuda' or 'cpu'")
    parser.add_argument("--share", action="store_true", help="Create a public link for the demo")
    parser.add_argument("--port", type=int, default=7860, help="Port to run the web server on")
    args = parser.parse_args()

    try:
        demo_instance = BorealisDemo(device=args.device)
        interface = create_ui(demo_instance)
        
        print(f"🚀 Launching Gradio demo on port {args.port}...")
        interface.queue().launch(
            server_name="0.0.0.0" if args.share else "127.0.0.1",
            server_port=args.port,
            share=args.share,
            show_error=True,
            inbrowser=True
        )
    except gr.Error as e:
        print(f"❌ Could not launch the demo due to a critical error: {e}")
    except Exception as e:
        print(f"❌ An unexpected error occurred during startup: {e}")
        print(traceback.format_exc())


if __name__ == "__main__":
    main()

