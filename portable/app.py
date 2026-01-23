# coding=utf-8
"""
Qwen3-TTS Portable - Русскоязычная версия со стримингом
Синтез речи с поддержкой: Дизайн голоса, Клонирование голоса, Пресеты голосов
"""

import os
import sys
import time
import threading
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Iterator
from dataclasses import asdict

import gradio as gr
import numpy as np
import torch
import soundfile as sf
from huggingface_hub import snapshot_download

# Добавляем родительскую директорию для импорта qwen_tts
sys.path.insert(0, str(Path(__file__).parent.parent))

from qwen_tts import Qwen3TTSModel, VoiceClonePromptItem

# =====================================================
# Глобальные переменные
# =====================================================

# Загруженные модели (кэш)
loaded_models: Dict[tuple, Qwen3TTSModel] = {}

# Флаги для стриминга
is_generating = False
stop_generation = False

# Размеры моделей
MODEL_SIZES = ["0.6B", "1.7B"]

# Типы моделей
MODEL_TYPES = {
    "Base": "Клонирование голоса",
    "CustomVoice": "Пресеты голосов",
    "VoiceDesign": "Дизайн голоса"
}

# Спикеры для CustomVoice
SPEAKERS = {
    "Aiden": "Эйден (мужской, английский)",
    "Dylan": "Дилан (мужской, английский)",
    "Eric": "Эрик (мужской, английский)",
    "Ono_anna": "Анна (женский, японский)",
    "Ryan": "Райан (мужской, английский)",
    "Serena": "Серена (женский, английский)",
    "Sohee": "Сохи (женский, корейский)",
    "Uncle_fu": "Дядя Фу (мужской, китайский)",
    "Vivian": "Вивиан (женский, английский)"
}

# Языки
LANGUAGES = {
    "Auto": "Авто (определить автоматически)",
    "Russian": "Русский",
    "English": "Английский",
    "Chinese": "Китайский",
    "Japanese": "Японский",
    "Korean": "Корейский",
    "French": "Французский",
    "German": "Немецкий",
    "Spanish": "Испанский",
    "Portuguese": "Португальский",
    "Italian": "Итальянский"
}

# =====================================================
# Вспомогательные функции
# =====================================================

def get_device():
    """Определение устройства для вычислений."""
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def get_model_path(model_type: str, model_size: str) -> str:
    """Получение пути к модели."""
    return snapshot_download(f"Qwen/Qwen3-TTS-12Hz-{model_size}-{model_type}")


def get_model(model_type: str, model_size: str) -> Qwen3TTSModel:
    """Получение или загрузка модели."""
    global loaded_models
    key = (model_type, model_size)

    if key not in loaded_models:
        model_path = get_model_path(model_type, model_size)
        device = get_device()
        dtype = torch.bfloat16 if device == "cuda" else torch.float32

        # Пробуем использовать Flash Attention, если доступен
        attn_impl = None
        if device == "cuda":
            try:
                import flash_attn
                attn_impl = "flash_attention_2"
                print(f"Flash Attention 2 активирован для {model_type} {model_size}")
            except ImportError:
                attn_impl = "sdpa"
                print(f"Используется SDPA для {model_type} {model_size}")

        loaded_models[key] = Qwen3TTSModel.from_pretrained(
            model_path,
            device_map=device,
            dtype=dtype,
            attn_implementation=attn_impl,
        )
        print(f"Модель {model_type} {model_size} загружена успешно!")

    return loaded_models[key]


def normalize_audio(wav, eps=1e-12, clip=True):
    """Нормализация аудио в диапазон [-1, 1]."""
    x = np.asarray(wav)

    if np.issubdtype(x.dtype, np.integer):
        info = np.iinfo(x.dtype)
        if info.min < 0:
            y = x.astype(np.float32) / max(abs(info.min), info.max)
        else:
            mid = (info.max + 1) / 2.0
            y = (x.astype(np.float32) - mid) / mid
    elif np.issubdtype(x.dtype, np.floating):
        y = x.astype(np.float32)
        m = np.max(np.abs(y)) if y.size else 0.0
        if m > 1.0 + 1e-6:
            y = y / (m + eps)
    else:
        raise TypeError(f"Неподдерживаемый тип данных: {x.dtype}")

    if clip:
        y = np.clip(y, -1.0, 1.0)

    if y.ndim > 1:
        y = np.mean(y, axis=-1).astype(np.float32)

    return y


def audio_to_tuple(audio) -> Optional[Tuple[np.ndarray, int]]:
    """Конвертация аудио Gradio в кортеж (wav, sr)."""
    if audio is None:
        return None

    if isinstance(audio, tuple) and len(audio) == 2 and isinstance(audio[0], int):
        sr, wav = audio
        wav = normalize_audio(wav)
        return wav, int(sr)

    if isinstance(audio, dict) and "sampling_rate" in audio and "data" in audio:
        sr = int(audio["sampling_rate"])
        wav = normalize_audio(audio["data"])
        return wav, sr

    return None


def save_audio_file(audio_data: np.ndarray, sample_rate: int, output_dir: str = "output") -> str:
    """Сохранение аудио в файл."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"qwen3_tts_{timestamp}.wav"
    filepath = os.path.join(output_dir, filename)
    sf.write(filepath, audio_data, sample_rate)
    return filepath


# =====================================================
# Функции генерации
# =====================================================

def generate_voice_design(
    text: str,
    language: str,
    voice_description: str,
    model_size: str,
    max_tokens: int,
    temperature: float,
    top_p: float
) -> Iterator[Tuple[Optional[Tuple[int, np.ndarray]], str]]:
    """Генерация речи с дизайном голоса (стриминг)."""
    global is_generating, stop_generation

    if not text or not text.strip():
        yield None, "Ошибка: Введите текст для синтеза."
        return

    if not voice_description or not voice_description.strip():
        yield None, "Ошибка: Введите описание голоса."
        return

    # Только 1.7B поддерживает VoiceDesign
    if model_size != "1.7B":
        yield None, "Ошибка: Дизайн голоса доступен только для модели 1.7B."
        return

    is_generating = True
    stop_generation = False

    try:
        yield None, "Загрузка модели VoiceDesign..."
        tts = get_model("VoiceDesign", model_size)

        yield None, f"Генерация речи...\nТекст: {text[:50]}...\nОписание: {voice_description[:50]}..."

        # Получаем реальный код языка
        lang_code = language.split()[0] if language != "Auto" else "Auto"
        for code, name in LANGUAGES.items():
            if name == language:
                lang_code = code
                break

        start_time = time.time()

        wavs, sr = tts.generate_voice_design(
            text=text.strip(),
            language=lang_code,
            instruct=voice_description.strip(),
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )

        if stop_generation:
            is_generating = False
            yield None, "Генерация остановлена пользователем."
            return

        generation_time = time.time() - start_time
        audio_duration = len(wavs[0]) / sr

        # Сохраняем файл
        saved_path = save_audio_file(wavs[0], sr)

        status = f"Генерация завершена!\n"
        status += f"Время генерации: {generation_time:.2f} сек\n"
        status += f"Длительность аудио: {audio_duration:.2f} сек\n"
        status += f"Файл сохранен: {saved_path}"

        yield (sr, wavs[0]), status

    except Exception as e:
        yield None, f"Ошибка: {type(e).__name__}: {e}"
    finally:
        is_generating = False


def generate_voice_clone(
    ref_audio,
    ref_text: str,
    target_text: str,
    language: str,
    use_xvector_only: bool,
    model_size: str,
    max_tokens: int,
    temperature: float,
    top_p: float
) -> Iterator[Tuple[Optional[Tuple[int, np.ndarray]], str]]:
    """Клонирование голоса (стриминг)."""
    global is_generating, stop_generation

    if not target_text or not target_text.strip():
        yield None, "Ошибка: Введите текст для синтеза."
        return

    audio_tuple = audio_to_tuple(ref_audio)
    if audio_tuple is None:
        yield None, "Ошибка: Загрузите референсное аудио."
        return

    if not use_xvector_only and (not ref_text or not ref_text.strip()):
        yield None, "Ошибка: Введите текст референсного аудио или включите режим 'Только x-vector'."
        return

    is_generating = True
    stop_generation = False

    try:
        yield None, "Загрузка модели Base..."
        tts = get_model("Base", model_size)

        yield None, f"Клонирование голоса...\nТекст: {target_text[:50]}..."

        # Получаем реальный код языка
        lang_code = "Auto"
        for code, name in LANGUAGES.items():
            if name == language:
                lang_code = code
                break

        start_time = time.time()

        wavs, sr = tts.generate_voice_clone(
            text=target_text.strip(),
            language=lang_code,
            ref_audio=audio_tuple,
            ref_text=ref_text.strip() if ref_text else None,
            x_vector_only_mode=use_xvector_only,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )

        if stop_generation:
            is_generating = False
            yield None, "Генерация остановлена пользователем."
            return

        generation_time = time.time() - start_time
        audio_duration = len(wavs[0]) / sr

        # Сохраняем файл
        saved_path = save_audio_file(wavs[0], sr)

        status = f"Клонирование завершено!\n"
        status += f"Время генерации: {generation_time:.2f} сек\n"
        status += f"Длительность аудио: {audio_duration:.2f} сек\n"
        status += f"Файл сохранен: {saved_path}"

        yield (sr, wavs[0]), status

    except Exception as e:
        yield None, f"Ошибка: {type(e).__name__}: {e}"
    finally:
        is_generating = False


def generate_custom_voice(
    text: str,
    language: str,
    speaker: str,
    instruct: str,
    model_size: str,
    max_tokens: int,
    temperature: float,
    top_p: float
) -> Iterator[Tuple[Optional[Tuple[int, np.ndarray]], str]]:
    """Генерация с пресетами голосов (стриминг)."""
    global is_generating, stop_generation

    if not text or not text.strip():
        yield None, "Ошибка: Введите текст для синтеза."
        return

    if not speaker:
        yield None, "Ошибка: Выберите голос."
        return

    is_generating = True
    stop_generation = False

    try:
        yield None, "Загрузка модели CustomVoice..."
        tts = get_model("CustomVoice", model_size)

        # Получаем реальное имя спикера
        speaker_id = speaker.split()[0] if speaker else "Vivian"
        for sid, sname in SPEAKERS.items():
            if sname == speaker:
                speaker_id = sid
                break

        yield None, f"Генерация речи...\nТекст: {text[:50]}...\nГолос: {speaker}"

        # Получаем реальный код языка
        lang_code = "Auto"
        for code, name in LANGUAGES.items():
            if name == language:
                lang_code = code
                break

        start_time = time.time()

        wavs, sr = tts.generate_custom_voice(
            text=text.strip(),
            language=lang_code,
            speaker=speaker_id.lower(),
            instruct=instruct.strip() if instruct else None,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )

        if stop_generation:
            is_generating = False
            yield None, "Генерация остановлена пользователем."
            return

        generation_time = time.time() - start_time
        audio_duration = len(wavs[0]) / sr

        # Сохраняем файл
        saved_path = save_audio_file(wavs[0], sr)

        status = f"Генерация завершена!\n"
        status += f"Время генерации: {generation_time:.2f} сек\n"
        status += f"Длительность аудио: {audio_duration:.2f} сек\n"
        status += f"Файл сохранен: {saved_path}"

        yield (sr, wavs[0]), status

    except Exception as e:
        yield None, f"Ошибка: {type(e).__name__}: {e}"
    finally:
        is_generating = False


def stop_generation_fn():
    """Остановка генерации."""
    global stop_generation
    stop_generation = True
    return "Остановка генерации..."


# =====================================================
# Построение интерфейса
# =====================================================

def build_ui():
    """Построение интерфейса Gradio."""

    # CSS стили (взято из VibeVoice)
    css = """
    .gradio-container {max-width: none !important;}

    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem 2rem;
        border-radius: 15px;
        margin-bottom: 1rem;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.2);
    }
    .main-header h1 {
        color: white;
        font-size: 2rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    .main-header p {
        color: rgba(255,255,255,0.9);
        margin: 0.5rem 0 0 0;
    }

    .settings-card {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 12px;
        padding: 1rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
    }

    .generation-card {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 12px;
        padding: 1rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
    }

    .tab-nav button {
        font-size: 1rem !important;
        padding: 0.75rem 1.5rem !important;
    }

    /* Dark mode support */
    .dark .settings-card, .dark .generation-card {
        background: rgba(30, 41, 59, 0.95);
    }
    .dark .main-header {
        background: linear-gradient(90deg, #4f46e5 0%, #7c3aed 100%);
    }
    """

    theme = gr.themes.Soft(
        font=[gr.themes.GoogleFont("Inter"), "Arial", "sans-serif"],
        primary_hue="indigo",
        secondary_hue="purple",
    )

    with gr.Blocks(theme=theme, css=css, title="Qwen3-TTS Portable") as demo:
        # Заголовок
        gr.HTML("""
        <div class="main-header">
            <h1>Qwen3-TTS - Синтез речи</h1>
            <p>Портативная русскоязычная версия со стримингом</p>
        </div>
        """)

        with gr.Tabs() as tabs:
            # =====================================================
            # Вкладка 1: Пресеты голосов (CustomVoice)
            # =====================================================
            with gr.Tab("Пресеты голосов", id="custom"):
                gr.Markdown("### Синтез речи с предустановленными голосами")

                with gr.Row():
                    with gr.Column(scale=1, elem_classes="settings-card"):
                        cv_text = gr.Textbox(
                            label="Текст для синтеза",
                            lines=4,
                            placeholder="Введите текст, который нужно озвучить...",
                            value="Привет! Это демонстрация системы синтеза речи Qwen3-TTS. Она поддерживает русский язык и множество других языков."
                        )

                        with gr.Row():
                            cv_language = gr.Dropdown(
                                label="Язык",
                                choices=list(LANGUAGES.values()),
                                value=LANGUAGES["Russian"],
                                interactive=True,
                            )
                            cv_speaker = gr.Dropdown(
                                label="Голос",
                                choices=list(SPEAKERS.values()),
                                value=SPEAKERS["Vivian"],
                                interactive=True,
                            )

                        cv_instruct = gr.Textbox(
                            label="Стиль (опционально)",
                            lines=2,
                            placeholder="Например: Говорить радостно и энергично",
                        )

                        with gr.Row():
                            cv_model_size = gr.Dropdown(
                                label="Размер модели",
                                choices=MODEL_SIZES,
                                value="1.7B",
                                interactive=True,
                            )

                        with gr.Accordion("Параметры генерации", open=False):
                            cv_max_tokens = gr.Slider(
                                label="Макс. токенов",
                                minimum=256, maximum=4096, value=2048, step=256
                            )
                            cv_temperature = gr.Slider(
                                label="Температура",
                                minimum=0.1, maximum=2.0, value=0.7, step=0.1
                            )
                            cv_top_p = gr.Slider(
                                label="Top-P",
                                minimum=0.1, maximum=1.0, value=0.9, step=0.05
                            )

                        with gr.Row():
                            cv_generate_btn = gr.Button("Сгенерировать", variant="primary", scale=2)
                            cv_stop_btn = gr.Button("Стоп", variant="stop", scale=1)

                    with gr.Column(scale=1, elem_classes="generation-card"):
                        cv_audio_out = gr.Audio(
                            label="Результат",
                            type="numpy",
                            interactive=False,
                        )
                        cv_status = gr.Textbox(
                            label="Статус",
                            lines=4,
                            interactive=False,
                        )

                cv_generate_btn.click(
                    generate_custom_voice,
                    inputs=[cv_text, cv_language, cv_speaker, cv_instruct, cv_model_size, cv_max_tokens, cv_temperature, cv_top_p],
                    outputs=[cv_audio_out, cv_status],
                )
                cv_stop_btn.click(stop_generation_fn, outputs=[cv_status])

            # =====================================================
            # Вкладка 2: Клонирование голоса (Base)
            # =====================================================
            with gr.Tab("Клонирование голоса", id="clone"):
                gr.Markdown("### Клонирование голоса из референсного аудио")

                with gr.Row():
                    with gr.Column(scale=1, elem_classes="settings-card"):
                        vc_ref_audio = gr.Audio(
                            label="Референсное аудио (голос для клонирования)",
                            type="numpy",
                            sources=["upload", "microphone"],
                        )
                        vc_ref_text = gr.Textbox(
                            label="Текст референсного аудио",
                            lines=2,
                            placeholder="Введите текст, который произносится в референсном аудио...",
                        )
                        vc_xvector_only = gr.Checkbox(
                            label="Только x-vector (без текста референса, качество ниже)",
                            value=False,
                        )

                        gr.Markdown("---")

                        vc_target_text = gr.Textbox(
                            label="Текст для синтеза",
                            lines=4,
                            placeholder="Введите текст, который нужно озвучить клонированным голосом...",
                        )

                        with gr.Row():
                            vc_language = gr.Dropdown(
                                label="Язык",
                                choices=list(LANGUAGES.values()),
                                value=LANGUAGES["Auto"],
                                interactive=True,
                            )
                            vc_model_size = gr.Dropdown(
                                label="Размер модели",
                                choices=MODEL_SIZES,
                                value="1.7B",
                                interactive=True,
                            )

                        with gr.Accordion("Параметры генерации", open=False):
                            vc_max_tokens = gr.Slider(
                                label="Макс. токенов",
                                minimum=256, maximum=4096, value=2048, step=256
                            )
                            vc_temperature = gr.Slider(
                                label="Температура",
                                minimum=0.1, maximum=2.0, value=0.7, step=0.1
                            )
                            vc_top_p = gr.Slider(
                                label="Top-P",
                                minimum=0.1, maximum=1.0, value=0.9, step=0.05
                            )

                        with gr.Row():
                            vc_generate_btn = gr.Button("Клонировать и озвучить", variant="primary", scale=2)
                            vc_stop_btn = gr.Button("Стоп", variant="stop", scale=1)

                    with gr.Column(scale=1, elem_classes="generation-card"):
                        vc_audio_out = gr.Audio(
                            label="Результат",
                            type="numpy",
                            interactive=False,
                        )
                        vc_status = gr.Textbox(
                            label="Статус",
                            lines=4,
                            interactive=False,
                        )

                vc_generate_btn.click(
                    generate_voice_clone,
                    inputs=[vc_ref_audio, vc_ref_text, vc_target_text, vc_language, vc_xvector_only, vc_model_size, vc_max_tokens, vc_temperature, vc_top_p],
                    outputs=[vc_audio_out, vc_status],
                )
                vc_stop_btn.click(stop_generation_fn, outputs=[vc_status])

            # =====================================================
            # Вкладка 3: Дизайн голоса (VoiceDesign)
            # =====================================================
            with gr.Tab("Дизайн голоса", id="design"):
                gr.Markdown("### Создание голоса по текстовому описанию")
                gr.Markdown("*Доступно только для модели 1.7B*")

                with gr.Row():
                    with gr.Column(scale=1, elem_classes="settings-card"):
                        vd_text = gr.Textbox(
                            label="Текст для синтеза",
                            lines=4,
                            placeholder="Введите текст, который нужно озвучить...",
                            value="Это невероятно! Я не могу поверить, что это действительно работает!"
                        )

                        vd_language = gr.Dropdown(
                            label="Язык",
                            choices=list(LANGUAGES.values()),
                            value=LANGUAGES["Russian"],
                            interactive=True,
                        )

                        vd_description = gr.Textbox(
                            label="Описание голоса",
                            lines=3,
                            placeholder="Опишите желаемый голос и стиль речи...",
                            value="Говорить с удивлением и восторгом, голос молодой женщины, энергичный и выразительный."
                        )

                        vd_model_size = gr.Dropdown(
                            label="Размер модели",
                            choices=["1.7B"],
                            value="1.7B",
                            interactive=False,
                        )

                        with gr.Accordion("Параметры генерации", open=False):
                            vd_max_tokens = gr.Slider(
                                label="Макс. токенов",
                                minimum=256, maximum=4096, value=2048, step=256
                            )
                            vd_temperature = gr.Slider(
                                label="Температура",
                                minimum=0.1, maximum=2.0, value=0.7, step=0.1
                            )
                            vd_top_p = gr.Slider(
                                label="Top-P",
                                minimum=0.1, maximum=1.0, value=0.9, step=0.05
                            )

                        with gr.Row():
                            vd_generate_btn = gr.Button("Сгенерировать", variant="primary", scale=2)
                            vd_stop_btn = gr.Button("Стоп", variant="stop", scale=1)

                    with gr.Column(scale=1, elem_classes="generation-card"):
                        vd_audio_out = gr.Audio(
                            label="Результат",
                            type="numpy",
                            interactive=False,
                        )
                        vd_status = gr.Textbox(
                            label="Статус",
                            lines=4,
                            interactive=False,
                        )

                vd_generate_btn.click(
                    generate_voice_design,
                    inputs=[vd_text, vd_language, vd_description, vd_model_size, vd_max_tokens, vd_temperature, vd_top_p],
                    outputs=[vd_audio_out, vd_status],
                )
                vd_stop_btn.click(stop_generation_fn, outputs=[vd_status])

        # Нижний колонтитул
        gr.Markdown("""
---

**Qwen3-TTS Portable** - Русскоязычная версия со стримингом

Создано на основе [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) от Alibaba Qwen Team.

**Поддерживаемые языки:** Русский, Английский, Китайский, Японский, Корейский, Французский, Немецкий, Испанский, Португальский, Итальянский

**Системные требования:**
- GPU: минимум 8GB VRAM для модели 1.7B, 4GB для 0.6B
- RAM: минимум 16GB
- При первом запуске модели загружаются из интернета (~4-8GB)
        """)

    return demo


# =====================================================
# Точка входа
# =====================================================

if __name__ == "__main__":
    print("=" * 50)
    print("Qwen3-TTS Portable - Русскоязычная версия")
    print("=" * 50)
    print()

    # Определяем устройство
    device = get_device()
    print(f"Устройство: {device.upper()}")

    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    print()
    print("Запуск веб-интерфейса...")
    print()

    # Строим и запускаем интерфейс
    demo = build_ui()
    demo.queue(default_concurrency_limit=4).launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True,
    )
