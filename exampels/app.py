import gradio as gr
import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig, TextIteratorStreamer
from qwen_vl_utils import process_vision_info
from PIL import Image, ImageDraw, ImageFont
import random
import os
import warnings
from typing import List, Tuple, Optional, Generator
import gc
import json
import re
import ast
import csv
import shutil
from datetime import datetime
import time
import tempfile
from threading import Thread
import io
import sys
import logging

# Optional: psutil for memory monitoring
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("Note: psutil not installed. RAM monitoring will be limited.")


# Suppress meta device warning (not useful)
warnings.filterwarnings('ignore', message='.*meta device.*')

# Global flag for stopping generation
stop_generation_flag = False

# ==========================================
# Console Log Capture for UI display
# ==========================================
class LogCapture:
    """Captures stdout and stderr for real-time console output in UI"""
    def __init__(self):
        self.log_buffer = io.StringIO()
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        self.is_capturing = False

    def start_capture(self):
        """Start capturing stdout and stderr"""
        if not self.is_capturing:
            sys.stdout = self
            sys.stderr = self
            self.is_capturing = True

    def stop_capture(self):
        """Stop capturing stdout and stderr"""
        if self.is_capturing:
            sys.stdout = self.original_stdout
            sys.stderr = self.original_stderr
            self.is_capturing = False

    def write(self, message):
        """Write to both original stdout/stderr and buffer"""
        self.original_stdout.write(message)
        self.log_buffer.write(message)

    def flush(self):
        """Flush stdout and stderr"""
        self.original_stdout.flush()
        self.original_stderr.flush()

    def get_logs(self):
        """Get captured logs"""
        return self.log_buffer.getvalue()

    def clear_logs(self):
        """Clear log buffer"""
        self.log_buffer = io.StringIO()

# Global log capture instance
log_capture = LogCapture()

# ==========================================
# Thinking Models Utilities
# ==========================================
def parse_thinking_output(text: str) -> Tuple[str, str]:
    """
    Parse thinking model output to separate reasoning from final answer

    Returns:
        Tuple of (reasoning, final_answer)
        If no </think> tag found, returns ("", full_text)
    """
    # Check for </think> tag
    if '</think>' in text:
        parts = text.split('</think>', 1)
        if len(parts) == 2:
            reasoning = parts[0].strip()
            # Remove <think> tag if present at start
            if reasoning.startswith('<think>'):
                reasoning = reasoning[7:].strip()
            final_answer = parts[1].strip()
            return reasoning, final_answer

    # No thinking tag found
    return "", text

# ==========================================
# Visual Grounding Utilities
# ==========================================
def parse_bboxes_from_text(text: str) -> list:
    """
    Parse bounding boxes from model output text
    Supports multiple formats:
    1. JSON format: [{"bbox_2d": [x1, y1, x2, y2], "label": "..."}]
    2. Inline format: {"bbox_2d": [x1, y1, x2, y2], "label": "..."}
    """
    bboxes = []

    # Try JSON format first
    try:
        # Extract JSON from markdown code block if present
        if '```json' in text:
            json_str = text.split('```json')[1].split('```')[0].strip()
        elif '```' in text:
            json_str = text.split('```')[1].split('```')[0].strip()
        else:
            # Try to find JSON array directly
            start = text.find('[')
            end = text.rfind(']') + 1
            if start >= 0 and end > start:
                json_str = text[start:end]
            else:
                # Try single object
                start = text.find('{')
                end = text.rfind('}') + 1
                if start >= 0 and end > start:
                    json_str = '[' + text[start:end] + ']'
                else:
                    return []

        parsed = json.loads(json_str)
        if isinstance(parsed, list):
            bboxes = parsed
        elif isinstance(parsed, dict):
            bboxes = [parsed]
    except (json.JSONDecodeError, ValueError, IndexError):
        # Try ast.literal_eval as fallback
        try:
            # Find all dict-like patterns
            pattern = r'\{[^{}]*"bbox_2d"[^{}]*\}'
            matches = re.findall(pattern, text, re.DOTALL)
            for match in matches:
                try:
                    bbox_dict = ast.literal_eval(match)
                    if 'bbox_2d' in bbox_dict:
                        bboxes.append(bbox_dict)
                except:
                    pass
        except:
            pass

    return bboxes

def draw_bboxes_on_image(image_path: str, bboxes: list, output_path: str = None, normalized: bool = True) -> str:
    """
    Draw bounding boxes on image

    Args:
        image_path: path to image file
        bboxes: list of bbox dicts with 'bbox_2d' and 'label' keys
        output_path: optional path to save the result (if None, creates temp file)
        normalized: if True, bboxes are in [0,1000] range; if False, pixel coords

    Returns:
        Path to output image with drawn boxes
    """
    if not bboxes:
        return image_path

    # Load image
    img = Image.open(image_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')

    draw = ImageDraw.Draw(img)
    img_width, img_height = img.size

    # Try to load a font, fallback to default if not available
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
        except:
            font = ImageFont.load_default()

    # Draw each bounding box
    for bbox_item in bboxes:
        bbox = bbox_item.get('bbox_2d', [])
        label = bbox_item.get('label', 'unknown')

        if len(bbox) != 4:
            continue

        x1, y1, x2, y2 = bbox

        # Convert from normalized to pixel coordinates if needed
        if normalized:
            x1 = int(x1 / 1000 * img_width)
            y1 = int(y1 / 1000 * img_height)
            x2 = int(x2 / 1000 * img_width)
            y2 = int(y2 / 1000 * img_height)
        else:
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        # Draw rectangle
        draw.rectangle([x1, y1, x2, y2], outline='red', width=3)

        # Draw label background
        try:
            bbox_text = draw.textbbox((x1, y1 - 20), label, font=font)
            draw.rectangle(bbox_text, fill='red')
            draw.text((x1, y1 - 20), label, fill='white', font=font)
        except:
            # Fallback without textbbox if not available
            draw.text((x1, y1 - 20), label, fill='red', font=font)

    # Save result
    if output_path is None:
        output_path = os.path.join(TEMP_DIR, f"bbox_{os.path.basename(image_path)}")

    img.save(output_path)
    return output_path

# Base directory for portable app
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TEMP_DIR = os.path.join(SCRIPT_DIR, "temp")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
DATASETS_DIR = os.path.join(SCRIPT_DIR, "datasets")
PROMPTS_DIR = os.path.join(SCRIPT_DIR, "prompts")

# Create directories if they don't exist
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DATASETS_DIR, exist_ok=True)
os.makedirs(PROMPTS_DIR, exist_ok=True)

# Huggingface cache directory - check env variables first, then platform-specific defaults
def get_hf_cache_dir():
    """Get HuggingFace cache directory, respecting environment variables and platform defaults"""
    # Check HuggingFace environment variables first
    if os.environ.get("HUGGINGFACE_HUB_CACHE"):
        return os.environ["HUGGINGFACE_HUB_CACHE"]
    if os.environ.get("HF_HOME"):
        return os.path.join(os.environ["HF_HOME"], "hub")
    if os.environ.get("HF_HUB_CACHE"):
        return os.environ["HF_HUB_CACHE"]

    # Platform-specific defaults
    if os.name == 'nt':  # Windows
        # Windows: typically in USERPROFILE\.cache\huggingface\hub
        local_app_data = os.environ.get("LOCALAPPDATA")
        if local_app_data:
            cache_in_localappdata = os.path.join(local_app_data, "huggingface", "hub")
            if os.path.exists(cache_in_localappdata):
                return cache_in_localappdata
        # Fall back to home directory
        return os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
    else:
        # Linux/Mac
        xdg_cache = os.environ.get("XDG_CACHE_HOME", os.path.join(os.path.expanduser("~"), ".cache"))
        return os.path.join(xdg_cache, "huggingface", "hub")

HF_CACHE_DIR = get_hf_cache_dir()

def get_model_cache_size(model_id: str) -> Optional[str]:
    """Check if model is downloaded and return its size on disk"""
    try:
        # Convert model_id to cache folder name format
        # huihui-ai/Huihui-Qwen3-VL-2B-Instruct-abliterated -> models--huihui-ai--Huihui-Qwen3-VL-2B-Instruct-abliterated
        cache_name = "models--" + model_id.replace("/", "--")
        cache_path = os.path.join(HF_CACHE_DIR, cache_name)

        if os.path.exists(cache_path):
            # Calculate total size
            total_size = 0
            for dirpath, dirnames, filenames in os.walk(cache_path):
                for f in filenames:
                    fp = os.path.join(dirpath, f)
                    if os.path.exists(fp):
                        total_size += os.path.getsize(fp)

            # Convert to human-readable format
            if total_size >= 1024 ** 3:
                return f"{total_size / (1024 ** 3):.1f}GB"
            elif total_size >= 1024 ** 2:
                return f"{total_size / (1024 ** 2):.0f}MB"
            else:
                return f"{total_size / 1024:.0f}KB"
        return None
    except:
        return None

# Available models (without MOE models which have different architecture)
AVAILABLE_MODELS = [
    # Abliterated models (без цензуры) - рекомендуемые
    ("2B Instruct Abliterated", "huihui-ai/Huihui-Qwen3-VL-2B-Instruct-abliterated"),
    ("2B Thinking Abliterated", "huihui-ai/Huihui-Qwen3-VL-2B-Thinking-abliterated"),
    ("4B Instruct Abliterated", "huihui-ai/Huihui-Qwen3-VL-4B-Instruct-abliterated"),
    ("4B Thinking Abliterated", "huihui-ai/Huihui-Qwen3-VL-4B-Thinking-abliterated"),
    ("8B Instruct Abliterated", "huihui-ai/Huihui-Qwen3-VL-8B-Instruct-abliterated"),
    ("8B Thinking Abliterated", "huihui-ai/Huihui-Qwen3-VL-8B-Thinking-abliterated"),
    ("32B Instruct Abliterated", "huihui-ai/Huihui-Qwen3-VL-32B-Instruct-abliterated"),
    ("32B Thinking Abliterated", "huihui-ai/Huihui-Qwen3-VL-32B-Thinking-abliterated"),
    # Original Qwen models (с цензурой)
    ("Qwen 2B Instruct", "Qwen/Qwen3-VL-2B-Instruct"),
    ("Qwen 4B Instruct", "Qwen/Qwen3-VL-4B-Instruct"),
    ("Qwen 8B Instruct", "Qwen/Qwen3-VL-8B-Instruct"),
]

def get_model_choices():
    """Get model choices with download status indicator"""
    choices = []
    for name, model_id in AVAILABLE_MODELS:
        cached_size = get_model_cache_size(model_id)
        if cached_size:
            # Model is downloaded - show size
            display_name = f"✅ {name} [{cached_size}]"
        else:
            # Model not downloaded - will be downloaded on first use
            display_name = f"⬇️ {name} [не скачана]"
        choices.append((display_name, model_id))
    return choices

# Extra options for description enhancement (IMAGE)
EXTRA_OPTIONS = {
    "en": {
        "Include lighting info": "Include information about lighting.",
        "Include camera angle": "Include information about camera angle.",
        "Include watermark info": "Include information about whether there is a watermark or not.",
        "Include JPEG artifacts info": "Include information about whether there are JPEG artifacts or not.",
        "Include camera/photo details": "If it is a photo, include information about what camera was likely used and details such as aperture, shutter speed, ISO, etc.",
        "Keep it SFW/PG": "Do NOT include anything sexual; keep it PG.",
        "Don't mention resolution": "Do NOT mention the image's resolution.",
        "Include aesthetic quality": "Include information about the subjective aesthetic quality of the image from low to very high.",
        "Include composition style": "Include information on the image's composition style, such as leading lines, rule of thirds, or symmetry.",
        "Don't mention text in image": "Do NOT mention any text that is in the image.",
        "Include depth of field": "Specify the depth of field and whether the background is in focus or blurred.",
        "Describe only key elements": "ONLY describe the most important elements of the image."
    },
    "ru": {
        "Добавить информацию об освещении": "Добавь информацию об освещении.",
        "Добавить ракурс камеры": "Добавь информацию о ракурсе камеры.",
        "Упомянуть водяной знак": "Укажи, есть ли водяной знак на изображении.",
        "Упомянуть JPEG артефакты": "Укажи, есть ли JPEG артефакты на изображении.",
        "Добавить детали камеры/фото": "Если это фотография, укажи информацию о камере, апертуре, выдержке, ISO и т.д.",
        "Сохранить SFW/PG рейтинг": "НЕ включай сексуальный контент, сохраняй рейтинг PG.",
        "Не упоминать разрешение": "НЕ упоминай разрешение изображения.",
        "Добавить эстетическую оценку": "Добавь информацию о субъективном эстетическом качестве от низкого до очень высокого.",
        "Добавить стиль композиции": "Добавь информацию о стиле композиции (ведущие линии, правило третей, симметрия).",
        "Не упоминать текст на изображении": "НЕ упоминай текст, который есть на изображении.",
        "Добавить глубину резкости": "Укажи глубину резкости и размытость фона.",
        "Описывать только ключевые элементы": "Описывай ТОЛЬКО самые важные элементы изображения."
    },
    "zh": {
        "包含光照信息": "包含光照信息。",
        "包含相机角度": "包含相机角度信息。",
        "提及水印": "说明图片是否有水印。",
        "提及JPEG伪影": "说明图片是否有JPEG伪影。",
        "包含相机/照片详情": "如果是照片，包含可能使用的相机信息，如光圈、快门速度、ISO等。",
        "保持SFW/PG级别": "不要包含任何性相关内容，保持PG级别。",
        "不要提及分辨率": "不要提及图片的分辨率。",
        "包含美学质量评价": "包含从低到非常高的主观美学质量评价。",
        "包含构图风格": "包含构图风格信息，如引导线、三分法则或对称性。",
        "不要提及图片中的文字": "不要提及图片中的任何文字。",
        "包含景深信息": "说明景深以及背景是否模糊。",
        "只描述关键元素": "只描述图片中最重要的元素。"
    }
}

# Extra options for VIDEO description enhancement
EXTRA_OPTIONS_VIDEO = {
    "en": {
        "Describe camera movement": "Describe camera movements (panning, zooming, static, etc.).",
        "Include audio description": "If the video has audio, describe it (music, speech, sound effects).",
        "Describe plot/story": "Describe the plot or story progression in the video.",
        "Include timestamps": "Provide timestamps in HH:MM:SS format for key events (e.g., 'person started running at 00:01:23').",
        "Include lighting info": "Include information about lighting changes throughout the video.",
        "Include editing style": "Describe the editing style (cuts, transitions, effects).",
        "Keep it SFW/PG": "Do NOT include anything sexual; keep it PG.",
        "Describe only key moments": "ONLY describe the most important moments in the video.",
        "Include aesthetic quality": "Include information about the subjective aesthetic quality from low to very high."
    },
    "ru": {
        "Описать движение камеры": "Опиши движения камеры (панорамирование, зум, статичная и т.д.).",
        "Добавить описание звука": "Если в видео есть звук, опиши его (музыка, речь, звуковые эффекты).",
        "Описать сюжет/историю": "Опиши сюжет или развитие истории в видео.",
        "Указать таймстампы": "Укажи таймстампы в формате ЧЧ:ММ:СС для ключевых событий (например, 'человек начал бежать в 00:01:23').",
        "Добавить информацию об освещении": "Добавь информацию об изменениях освещения в течение видео.",
        "Добавить стиль монтажа": "Опиши стиль монтажа (переходы, эффекты).",
        "Сохранить SFW/PG рейтинг": "НЕ включай сексуальный контент, сохраняй рейтинг PG.",
        "Описывать только ключевые моменты": "Описывай ТОЛЬКО самые важные моменты видео.",
        "Добавить эстетическую оценку": "Добавь информацию о субъективном эстетическом качестве от низкого до очень высокого."
    },
    "zh": {
        "描述镜头运动": "描述镜头运动（平移、缩放、静止等）。",
        "包含音频描述": "如果视频有音频，描述它（音乐、语音、音效）。",
        "描述情节/故事": "描述视频中的情节或故事发展。",
        "包含时间戳": "以HH:MM:SS格式提供关键事件的时间戳（例如，'人物在00:01:23开始奔跑'）。",
        "包含光照信息": "包含视频中光照变化的信息。",
        "包含剪辑风格": "描述剪辑风格（切换、过渡、效果）。",
        "保持SFW/PG级别": "不要包含任何性相关内容，保持PG级别。",
        "只描述关键时刻": "只描述视频中最重要的时刻。",
        "包含美学质量评价": "包含从低到非常高的主观美学质量评价。"
    }
}

def get_memory_info() -> str:
    """Get current memory usage information"""
    info_parts = []

    # GPU memory
    if torch.cuda.is_available():
        try:
            allocated = torch.cuda.memory_allocated() / (1024 ** 3)
            reserved = torch.cuda.memory_reserved() / (1024 ** 3)
            total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            info_parts.append(f"GPU: {allocated:.1f}/{total:.1f} GB (reserved: {reserved:.1f} GB)")
        except:
            pass

    # RAM
    if PSUTIL_AVAILABLE:
        try:
            process = psutil.Process(os.getpid())
            ram_usage = process.memory_info().rss / (1024 ** 3)
            info_parts.append(f"RAM: {ram_usage:.1f} GB")
        except:
            pass

    return " | ".join(info_parts) if info_parts else "Memory info unavailable"

def get_video_duration(video_path: str) -> float:
    """Get video duration in seconds using ffprobe (lightweight, no extra dependencies)"""
    if not video_path or not os.path.exists(video_path):
        return 7200.0  # Default 2 hours

    # Use ffprobe - fast, lightweight, no loading video into memory
    try:
        import subprocess
        result = subprocess.run(
            ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
             '-of', 'default=noprint_wrappers=1:nokey=1', video_path],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            duration = float(result.stdout.strip())
            if duration > 0:
                return duration
    except:
        pass

    # Fallback to 2 hours if ffprobe not available
    return 7200.0

def load_prompt_presets() -> dict:
    """Load prompt presets from the prompts directory"""
    presets = {"None": ""}

    if not os.path.exists(PROMPTS_DIR):
        return presets

    for filename in os.listdir(PROMPTS_DIR):
        if filename.endswith('.txt'):
            preset_name = os.path.splitext(filename)[0]
            try:
                with open(os.path.join(PROMPTS_DIR, filename), 'r', encoding='utf-8') as f:
                    presets[preset_name] = f.read().strip()
            except:
                pass

    return presets

def save_prompt_preset(name: str, prompt: str) -> str:
    """Save a prompt preset to the prompts directory"""
    if not name or not name.strip():
        if current_language == "ru":
            return "❌ Введите имя пресета"
        return "❌ Please provide a preset name"

    if not prompt or not prompt.strip():
        if current_language == "ru":
            return "❌ Введите промпт для сохранения"
        return "❌ Please provide a prompt to save"

    # Sanitize filename
    safe_name = "".join(c for c in name if c.isalnum() or c in "_ -").strip()
    if not safe_name:
        if current_language == "ru":
            return "❌ Недопустимое имя пресета"
        return "❌ Invalid preset name"

    try:
        os.makedirs(PROMPTS_DIR, exist_ok=True)
        filepath = os.path.join(PROMPTS_DIR, f"{safe_name}.txt")
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(prompt.strip())
        if current_language == "ru":
            return f"✅ Пресет '{safe_name}' сохранён!"
        return f"✅ Preset '{safe_name}' saved successfully!"
    except Exception as e:
        if current_language == "ru":
            return f"❌ Ошибка сохранения: {str(e)}"
        return f"❌ Error saving preset: {str(e)}"

def save_text_to_file(text: str, filename: str = "result.txt") -> str:
    """Save text to a temporary file and return the path"""
    temp_file = os.path.join(TEMP_DIR, filename)
    with open(temp_file, 'w', encoding='utf-8') as f:
        f.write(text)
    return temp_file

def stop_generation():
    """Set the stop flag to True"""
    global stop_generation_flag
    stop_generation_flag = True
    return "🛑 Stopping generation..."

def reset_stop_flag():
    """Reset the stop flag"""
    global stop_generation_flag
    stop_generation_flag = False

# Custom CSS for beautiful UI
CUSTOM_CSS = """
/* Main container styling */
.gradio-container {
    background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%) !important;
}

/* Card style blocks */
.card-style {
    background: rgba(255, 255, 255, 0.9) !important;
    border: 1px solid rgba(203, 213, 225, 0.5) !important;
    box-shadow: 0 8px 32px rgba(100, 116, 139, 0.1) !important;
    border-radius: 16px !important;
    padding: 1.5rem !important;
    margin-bottom: 1rem !important;
}

/* Main header with gradient */
.main-header {
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    padding: 2rem;
    border-radius: 20px;
    margin-bottom: 1.5rem;
    text-align: center;
    box-shadow: 0 10px 40px rgba(102, 126, 234, 0.3);
}

.main-header h1 {
    color: white !important;
    font-size: 2.5rem !important;
    font-weight: 700 !important;
    margin: 0 !important;
    text-shadow: 0 2px 4px rgba(0,0,0,0.3);
}

.main-header p {
    color: rgba(255,255,255,0.9) !important;
    font-size: 1.1rem !important;
    margin: 0.5rem 0 0 0 !important;
}

.main-header a {
    color: white !important;
    text-decoration: underline;
}

/* Generate button styling */
.generate-btn {
    min-height: 36px !important;
    font-size: 0.95rem !important;
    font-weight: 500 !important;
    padding: 8px 16px !important;
}

/* Status box */
.status-box {
    background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%) !important;
    border: 1px solid #7dd3fc !important;
    border-radius: 12px !important;
    padding: 1rem !important;
}

/* Progress info */
.progress-info {
    background: #f0fdf4 !important;
    border: 1px solid #86efac !important;
    border-radius: 8px !important;
    padding: 0.75rem !important;
}

/* Dark mode support */
.dark .gradio-container {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%) !important;
}

.dark .card-style {
    background: rgba(30, 41, 59, 0.9) !important;
    border: 1px solid rgba(51, 65, 85, 0.5) !important;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3) !important;
}

.dark .status-box {
    background: linear-gradient(135deg, #1e3a5f 0%, #1e293b 100%) !important;
    border: 1px solid #3b82f6 !important;
}

.dark .progress-info {
    background: #14532d !important;
    border: 1px solid #22c55e !important;
}

/* Accordion styling */
.settings-accordion {
    border-radius: 12px !important;
    overflow: hidden !important;
}

/* Result variants */
.variant-box {
    border: 2px solid #e2e8f0;
    border-radius: 12px;
    padding: 1rem;
    margin-bottom: 0.5rem;
    transition: all 0.2s ease;
}

.variant-box:hover {
    border-color: #667eea;
    box-shadow: 0 4px 12px rgba(102, 126, 234, 0.15);
}

/* Export buttons */
.export-btn {
    margin: 0.25rem !important;
}
"""

# Description types with prompts
DESCRIPTION_TYPES = {
    "en": {
        "Descriptive (Formal)": "Write a detailed and formal description of this image.",
        "Descriptive (Informal)": "Write a casual, friendly description of this image.",
        "Product Description": "Write a compelling product description for e-commerce based on this image.",
        "SEO Description": "Write an SEO-optimized description for this image, maximum 160 characters.",
        "Stable Diffusion Prompt": "Write a detailed Stable Diffusion prompt to recreate this image.",
        "MidJourney Prompt": "Write a MidJourney prompt to recreate this image.",
        "Booru Tags": "Write a list of Booru-style tags for this image, separated by commas.",
        "Art Critic Analysis": "Analyze this image like an art critic, discussing composition, style, color, lighting, and artistic elements.",
        "Social Media Caption": "Write an engaging social media caption for this image.",
        "OCR: Extract all text": "Extract ALL text from the image. Read every word, number, and symbol visible.",
        "OCR: Text with coordinates": "Extract all text and provide position coordinates [x1, y1, x2, y2] for each text region.",
        "OCR: Table to HTML": "If there is a table, convert it to HTML format using <table>, <tr>, and <td> tags.",
        "OCR: Structured JSON": "Extract all information in structured JSON format with key-value pairs.",
        "🔀 Compare products": "Compare these product images side by side. List: 1) design differences, 2) color variations, 3) feature changes, 4) quality assessment, 5) which one to recommend and why.",
        "🔀 Before/After comparison": "Analyze the before (first image) and after (last image) states: 1) What changed? 2) Quantify improvements if measurable, 3) Rate the transformation quality 1-10, 4) What could be improved further?",
        "🔀 Time-series analysis": "These images show a sequence over time. Describe: 1) progression and trends, 2) identify causality between frames, 3) predict what happens next, 4) rate of change (fast/slow/accelerating), 5) any anomalies.",
        "🔀 Quality control": "Review these quality control images: 1) identify defects in each, 2) classify defect types, 3) rate pass/fail for each, 4) percentage meeting standards, 5) recommend corrective actions, 6) any systematic issues?",
        "📍 Detect objects with locations": "Detect all objects in the image and return their locations in the format: {\"bbox_2d\": [x1, y1, x2, y2], \"label\": \"object_name\"}",
        "📍 Visual grounding": "Describe the image in detail with grounding. For each important object, provide bounding box coordinates.",
        "📍 Find and locate": "Find all instances of specific objects and provide their precise locations with bounding boxes in JSON format.",
        "🧠 Math step-by-step": "Solve the mathematical problem in the image. Think carefully step by step. Step 1: Identify the data. Step 2: Calculate. Step 3: Verify the answer.",
        "🧠 Logical analysis": "Analyze the image step by step: 1) Identify the main subject, 2) Describe background and context, 3) Note important details, 4) Explain the overall scene and atmosphere.",
        "🧠 Causal analysis": "Examine the sequence and explain: 1) What happened first? 2) What caused the change? 3) What was the effect? 4) What will likely happen next? Provide reasoning for each step.",
        "📊 Chart analysis": "You are an expert in visual analytics. Analyze the chart: 1) Chart type, 2) Title and axes, 3) Key trends, 4) Conclusions and insights.",
        "📊 Data visualization": "Analyze this data visualization: identify the type, extract data points, describe trends, and provide key insights with specific numbers.",
        "🩺 Medical image analysis": "Analyze this medical image. Identify visible structures and note any abnormalities. Use medical terminology where appropriate.",
        "⚙️ Technical diagram": "This is a technical diagram. Identify all components, explain their functions, and describe how they interact with each other.",
        "📄 Document extraction": "Extract structured data from the document without omissions. Return JSON with keys: document_type, date, number, items (array), total.",
        "🔬 Scientific image": "Analyze this scientific image. Describe observed phenomena, structures, and processes. Use scientific terminology.",
        "🎬 Timeline of events": "Describe the chronological sequence of events in the video with timestamps. What happens at each key moment?",
        "🎬 Action detection": "Identify all instances where [action] occurs in the video and provide approximate timestamps.",
        "🎬 Long video summary": "This is a [X]-minute video. Summarize the main topics and key points covered.",
        "🎬 Editing analysis": "Analyze the video editing: shot types, pacing, transitions, background music (if instruments/actions are visible).",
        "📚 Explain concept": "Explain the concept shown in the image in simple language. How does it work and why is it important?",
        "📚 Solve textbook problem": "Solve this textbook problem step by step. Show all calculations and explain the logic.",
        "📚 Historical analysis": "Analyze this historical photo/document: period, context, significance, visible details of the era.",
        "📚 Lab setup": "Describe the laboratory setup: equipment, procedure, safety measures.",
        "🎨 Color analysis": "Detailed color palette analysis: dominant colors, contrasts, harmony, color mood, application in design.",
        "🏗️ Architectural analysis": "Architecture analysis: style, period, structural elements, materials, functionality, cultural significance.",
        "🍽️ Dish analysis": "As a chef, describe the dish: ingredients, cooking technique, presentation, flavor combination, serving recommendations.",
        "💼 Presentation slide": "Describe the presentation slide content: title, key points, visual elements, main message.",
        "🏭 Industrial safety": "Analyze the image for safety: identified risks, violations, recommendations.",
        "🎯 Layered composition": "Analyze the scene in layers: Layer 1: Background, Layer 2: Middle ground, Layer 3: Foreground, Layer 4: Overall composition and visual flow.",
        "🎯 Spatial analysis": "Describe spatial arrangement: overall layout, foreground/middle/background, object relationships, perspective.",
        "🤔 Careful analysis": "Before answering, first carefully observe the image. Then, organize your thoughts about what's important. Finally, provide a structured response covering: subject, context, details, and interpretation.",
        "🤔 Problem-finding": "Examine the image critically. What works well? What could be improved? Provide specific recommendations.",
        "Custom": ""
    },
    "ru": {
        "Описательный (формальный)": "Напиши детальное и формальное описание этого изображения на русском языке.",
        "Описательный (неформальный)": "Напиши непринуждённое, дружелюбное описание этого изображения на русском языке.",
        "Описание товара": "Напиши привлекательное описание товара для интернет-магазина на основе этого изображения на русском языке.",
        "SEO описание": "Напиши SEO-оптимизированное описание для этого изображения на русском языке, максимум 160 символов.",
        "Промпт Stable Diffusion": "Напиши детальный промпт для Stable Diffusion, чтобы воссоздать это изображение.",
        "Промпт MidJourney": "Напиши промпт для MidJourney, чтобы воссоздать это изображение.",
        "Теги Booru": "Напиши список тегов в стиле Booru для этого изображения, разделённых запятыми.",
        "Анализ искусствоведа": "Проанализируй это изображение как искусствовед, обсуждая композицию, стиль, цвет, освещение и художественные элементы на русском языке.",
        "Пост для соцсетей": "Напиши привлекательную подпись для социальных сетей к этому изображению на русском языке.",
        "OCR: Извлечь весь текст": "Извлеки ВЕСЬ текст с изображения. Прочитай каждое слово, цифру и символ.",
        "OCR: Текст с координатами": "Извлеки весь текст с изображения. Для каждой текстовой области верни результат в JSON формате: [{'bbox_2d': [x1, y1, x2, y2], 'label': 'обнаруженный текст'}]. Координаты нормализованы в диапазоне [0, 1000], где (0,0) это верхний левый угол, а (1000,1000) - нижний правый.",
        "OCR: Таблица в HTML": "Если есть таблица, преобразуй её в HTML формат с тегами <table>, <tr> и <td>.",
        "OCR: Структурированный JSON": "Извлеки всю информацию в структурированном JSON формате с ключами и значениями.",
        "🔀 Сравнить товары": "Сравни эти изображения товаров. Перечисли: 1) различия в дизайне, 2) варианты цветов, 3) изменения функций, 4) оценка качества, 5) какой рекомендуешь и почему.",
        "🔀 Сравнение до/после": "Проанализируй состояния до (первое изображение) и после (последнее): 1) Что изменилось? 2) Количественно оцени улучшения, 3) Оцени качество преобразования 1-10, 4) Что можно улучшить?",
        "🔀 Анализ временного ряда": "Эти изображения показывают последовательность во времени. Опиши: 1) развитие и тренды, 2) причинность между кадрами, 3) предскажи что будет дальше, 4) скорость изменений, 5) аномалии.",
        "🔀 Контроль качества": "Проверь эти изображения контроля качества: 1) дефекты в каждом, 2) типы дефектов, 3) сдача/отказ для каждого, 4) процент соответствия стандартам, 5) корректирующие действия, 6) системные проблемы?",
        "📍 Обнаружить объекты с местоположением": "Обнаружь все объекты на изображении и верни их местоположение в формате: {\"bbox_2d\": [x1, y1, x2, y2], \"label\": \"название_объекта\"}",
        "📍 Визуальная привязка": "Опиши изображение подробно с привязкой. Для каждого важного объекта укажи координаты ограничивающей рамки.",
        "📍 Найти и указать местоположение": "Найди все экземпляры конкретных объектов и укажи их точные местоположения с ограничивающими рамками в JSON формате.",
        "🧠 Математика пошагово": "Реши математическую задачу на изображении. Думай последовательно шаг за шагом. Шаг 1: Определи данные. Шаг 2: Вычисли. Шаг 3: Проверь ответ.",
        "🧠 Логический анализ": "Проанализируй изображение пошагово: 1) Определи главный объект, 2) Опиши фон и контекст, 3) Отметь важные детали, 4) Объясни общую сцену и атмосферу.",
        "🧠 Причинно-следственный анализ": "Изучи последовательность и объясни: 1) Что произошло сначала? 2) Что вызвало изменение? 3) Каков эффект? 4) Что произойдет дальше? Обоснуй каждый шаг.",
        "📊 Анализ графиков": "Ты эксперт по визуальной аналитике. Проанализируй график: 1) Тип графика, 2) Заголовок и оси, 3) Ключевые тренды, 4) Выводы и инсайты.",
        "📊 Визуализация данных": "Проанализируй эту визуализацию данных: определи тип, извлеки точки данных, опиши тренды и дай ключевые инсайты с конкретными числами.",
        "🩺 Медицинское изображение": "Проанализируй медицинское изображение. Определи видимые структуры и отметь любые аномалии. Используй медицинскую терминологию.",
        "⚙️ Техническая диаграмма": "Это техническая диаграмма. Определи все компоненты, объясни их функции и опиши как они взаимодействуют.",
        "📄 Извлечение из документа": "Извлеки структурированные данные из документа без пропусков. Верни JSON с ключами: тип_документа, дата, номер, позиции (массив), итого.",
        "🔬 Научное изображение": "Проанализируй научное изображение. Опиши наблюдаемые явления, структуры и процессы. Используй научную терминологию.",
        "🎬 Временная шкала событий": "Опиши хронологическую последовательность событий в видео с временными метками. Что происходит в каждый ключевой момент?",
        "🎬 Обнаружение действий": "Определи все моменты, когда [действие] происходит в видео, и укажи примерные временные метки.",
        "🎬 Резюме длинного видео": "Это [X]-минутное видео. Резюмируй основные темы и ключевые моменты.",
        "🎬 Анализ монтажа": "Проанализируй монтаж видео: типы планов, темп, переходы, музыкальное сопровождение (если видно инструменты/действия).",
        "📚 Объясни концепцию": "Объясни концепцию, показанную на изображении, простым языком. Как это работает и почему это важно?",
        "📚 Решение задачи из учебника": "Реши эту задачу из учебника пошагово. Покажи все вычисления и объясни логику.",
        "📚 Исторический анализ": "Проанализируй это историческое фото/документ: период, контекст, значимость, видимые детали эпохи.",
        "📚 Лабораторная работа": "Опиши лабораторную установку: оборудование, процедуру, меры безопасности.",
        "🎨 Цветовой анализ": "Детальный анализ цветовой палитры: доминирующие цвета, контрасты, гармония, настроение цвета, применение в дизайне.",
        "🏗️ Архитектурный анализ": "Анализ архитектуры: стиль, период, конструктивные элементы, материалы, функциональность, культурное значение.",
        "🍽️ Анализ блюда": "Как шеф-повар опиши блюдо: ингредиенты, техника приготовления, подача, сочетание вкусов, рекомендации к подаче.",
        "💼 Презентация/Слайд": "Опиши содержание слайда презентации: заголовок, ключевые пункты, визуальные элементы, основная мысль.",
        "🏭 Промышленная безопасность": "Проанализируй изображение на предмет безопасности: выявленные риски, нарушения, рекомендации.",
        "🎯 Анализ композиции по слоям": "Анализируй сцену послойно: Слой 1: Фон, Слой 2: Средний план, Слой 3: Передний план, Слой 4: Композиция и поток взгляда.",
        "🎯 Пространственный анализ": "Опиши пространственное расположение: общая компоновка, передний/средний/задний план, взаимоотношения объектов, перспектива.",
        "🤔 Внимательный анализ": "Перед ответом сначала внимательно изучи изображение. Затем организуй свои мысли о важном. Наконец, дай структурированный ответ: предмет, контекст, детали, интерпретация.",
        "🤔 Поиск проблем": "Изучи изображение критически. Что работает хорошо? Что можно улучшить? Предложи конкретные рекомендации.",
        "Свой промпт": ""
    },
    "zh": {
        "描述性（正式）": "写一个详细正式的图像描述。",
        "描述性（非正式）": "写一个轻松友好的图像描述。",
        "产品描述": "根据这张图片为电商平台写一个吸引人的产品描述。",
        "SEO描述": "为这张图片写一个SEO优化的描述，最多160个字符。",
        "Stable Diffusion提示词": "写一个详细的Stable Diffusion提示词来重现这张图片。",
        "MidJourney提示词": "写一个MidJourney提示词来重现这张图片。",
        "Booru标签": "为这张图片写一个Booru风格的标签列表，用逗号分隔。",
        "艺术评论分析": "像艺术评论家一样分析这张图片，讨论构图、风格、色彩、光线和艺术元素。",
        "社交媒体文案": "为这张图片写一个吸引人的社交媒体文案。",
        "OCR: 提取所有文字": "提取图片中的所有文字。读取每个单词、数字和符号。",
        "OCR: 文字与坐标": "提取所有文字并提供每个文字区域的位置坐标[x1, y1, x2, y2]。",
        "OCR: 表格转HTML": "如果有表格，将其转换为HTML格式，使用<table>、<tr>和<td>标签。",
        "OCR: 结构化JSON": "以结构化JSON格式提取所有信息，包含键值对。",
        "🔀 比较产品": "并排比较这些产品图像。列出：1）设计差异，2）颜色变化，3）功能变化，4）质量评估，5）推荐哪一个及原因。",
        "🔀 前后对比": "分析之前（第一张图）和之后（最后一张）的状态：1）发生了什么变化？2）量化改进（如可衡量），3）转换质量评分1-10，4）还可以改进什么？",
        "🔀 时间序列分析": "这些图像显示时间序列。描述：1）进度和趋势，2）识别帧之间的因果关系，3）预测接下来会发生什么，4）变化速率（快/慢/加速），5）任何异常。",
        "🔀 质量控制": "审查这些质量控制图像：1）识别每个缺陷，2）分类缺陷类型，3）对每个评分合格/不合格，4）符合标准的百分比，5）推荐纠正措施，6）是否有系统性问题？",
        "📍 检测对象及位置": "检测图像中的所有对象并以格式返回其位置：{\"bbox_2d\": [x1, y1, x2, y2], \"label\": \"对象名称\"}",
        "📍 视觉定位": "详细描述图像并提供定位。对于每个重要对象，提供边界框坐标。",
        "📍 查找并定位": "查找特定对象的所有实例，并以JSON格式提供其精确位置和边界框。",
        "🧠 逐步数学": "解决图像中的数学问题。逐步仔细思考。步骤1：识别数据。步骤2：计算。步骤3：验证答案。",
        "🧠 逻辑分析": "逐步分析图像：1）识别主要对象，2）描述背景和上下文，3）注意重要细节，4）解释整体场景和氛围。",
        "🧠 因果分析": "检查序列并解释：1）首先发生了什么？2）什么导致了变化？3）效果是什么？4）接下来可能发生什么？为每个步骤提供推理。",
        "📊 图表分析": "你是视觉分析专家。分析图表：1）图表类型，2）标题和坐标轴，3）关键趋势，4）结论和见解。",
        "📊 数据可视化": "分析这个数据可视化：识别类型，提取数据点，描述趋势，并提供具体数字的关键见解。",
        "🩺 医学图像分析": "分析这张医学图像。识别可见结构并注意任何异常。适当使用医学术语。",
        "⚙️ 技术图表": "这是技术图表。识别所有组件，解释其功能，并描述它们如何相互作用。",
        "📄 文档提取": "从文档中提取结构化数据而不遗漏。返回JSON，键：document_type、date、number、items（数组）、total。",
        "🔬 科学图像": "分析这张科学图像。描述观察到的现象、结构和过程。使用科学术语。",
        "🎬 事件时间线": "描述视频中事件的时间顺序和时间戳。每个关键时刻发生了什么？",
        "🎬 动作检测": "识别视频中发生[动作]的所有实例，并提供大致时间戳。",
        "🎬 长视频摘要": "这是一个[X]分钟的视频。总结涵盖的主要主题和要点。",
        "🎬 剪辑分析": "分析视频剪辑：镜头类型、节奏、转场、背景音乐（如果可见乐器/动作）。",
        "📚 解释概念": "用简单的语言解释图像中显示的概念。它是如何工作的，为什么重要？",
        "📚 解决教科书问题": "逐步解决这个教科书问题。显示所有计算并解释逻辑。",
        "📚 历史分析": "分析这张历史照片/文件：时期、背景、意义、时代的可见细节。",
        "📚 实验室设置": "描述实验室设置：设备、程序、安全措施。",
        "🎨 色彩分析": "详细的色彩调色板分析：主色、对比、和谐、色彩情绪、设计应用。",
        "🏗️ 建筑分析": "建筑分析：风格、时期、结构元素、材料、功能、文化意义。",
        "🍽️ 菜肴分析": "作为厨师，描述菜肴：配料、烹饪技术、摆盘、风味组合、上菜建议。",
        "💼 演示幻灯片": "描述演示幻灯片内容：标题、要点、视觉元素、主要信息。",
        "🏭 工业安全": "分析图像的安全性：识别的风险、违规、建议。",
        "🎯 分层构图": "分层分析场景：第1层：背景，第2层：中景，第3层：前景，第4层：整体构图和视觉流。",
        "🎯 空间分析": "描述空间布局：整体布局、前景/中景/背景、对象关系、透视。",
        "🤔 仔细分析": "在回答之前，首先仔细观察图像。然后，整理关于重要内容的想法。最后，提供涵盖以下内容的结构化响应：主题、上下文、细节和解释。",
        "🤔 问题发现": "批判性地检查图像。什么效果好？什么可以改进？提供具体建议。",
        "自定义": ""
    }
}

# Description lengths
DESCRIPTION_LENGTHS = {
    "en": {
        "Any": "",
        "Very Short (1-2 sentences)": "Keep it very short, 1-2 sentences maximum.",
        "Short (3-4 sentences)": "Keep it short, 3-4 sentences.",
        "Medium (1 paragraph)": "Write a medium-length description, about one paragraph.",
        "Long (2-3 paragraphs)": "Write a detailed description, 2-3 paragraphs.",
        "Very Long (comprehensive)": "Write a comprehensive and very detailed description."
    },
    "ru": {
        "Любая": "",
        "Очень короткая (1-2 предложения)": "Сделай очень коротко, максимум 1-2 предложения.",
        "Короткая (3-4 предложения)": "Сделай коротко, 3-4 предложения.",
        "Средняя (1 абзац)": "Напиши описание средней длины, около одного абзаца.",
        "Длинная (2-3 абзаца)": "Напиши детальное описание, 2-3 абзаца.",
        "Очень длинная (подробная)": "Напиши исчерпывающее и очень детальное описание."
    },
    "zh": {
        "任意": "",
        "非常短（1-2句）": "保持非常简短，最多1-2句。",
        "短（3-4句）": "保持简短，3-4句。",
        "中等（1段）": "写一个中等长度的描述，大约一段。",
        "长（2-3段）": "写一个详细的描述，2-3段。",
        "非常长（全面）": "写一个全面且非常详细的描述。"
    }
}

# Multi-language support
TRANSLATIONS = {
    "en": {
        "title": "Qwen VL Image Description Generator",
        "header": "🖼️ Image Description Generator based on Qwen Vision Language Models",
        "subtitle": "Upload an image and enter a prompt to generate a description using Qwen VL models.",
        "language": "Language",
        "language_info": "Select language",
        "model_selection": "Model Selection",
        "model_info": "Select a model for generating descriptions",
        "advanced_params": "⚙️ Advanced Parameters",
        "max_tokens": "Max New Tokens",
        "max_tokens_info": "Maximum number of tokens to generate",
        "temperature": "Temperature",
        "temperature_info": "Controls randomness of generation",
        "top_p": "Top-p (nucleus sampling)",
        "top_p_info": "Probability threshold for token sampling",
        "top_k": "Top-k",
        "top_k_info": "Number of most probable tokens to consider",
        "seed": "Seed",
        "seed_info": "Seed for reproducibility (-1 for random)",
        "random_seed_btn": "🎲 Random Seed",
        "single_processing": "📄 Single Processing",
        "batch_processing": "📚 Batch Processing",
        "upload_image": "Upload Image",
        "image_url": "Or enter Image URL",
        "image_url_placeholder": "https://example.com/image.jpg",
        "prompt": "Prompt",
        "prompt_placeholder": "For example: Create a product description for online store",
        "generate_btn": "🚀 Generate Description",
        "result": "Result",
        "upload_images": "Upload Images",
        "prompts_multiline": "Prompts (one per line)",
        "prompts_placeholder": "Create a product description for online store\nCreate SEO Description for product\n...",
        "prompts_info": "Specify one prompt for all images or one prompt per image",
        "process_batch_btn": "🚀 Process Batch",
        "results": "Results",
        "examples_title": "💡 Example Prompts:",
        "example_1": "Create a product description for online store",
        "example_2": "Create an SEO description for a product with a maximum of 160 characters.",
        "example_3": "Create an attractive product description for marketplace",
        "example_4": "Describe image in detail for product catalog",
        "error_no_image": "Please upload an image or provide an image URL",
        "error_no_prompt": "Please enter a prompt",
        "error_no_images": "Please upload images",
        "error_no_prompts": "Please enter prompts (one per line)",
        "error_prompt_mismatch": "Number of prompts ({}) does not match number of images ({}). Specify either one prompt for all images or one prompt per image.",
        "error_generation": "Error generating description: {}",
        "error_url_load": "Error loading image from URL: {}",
        "loading_model": "Loading model: {}",
        "model_loaded": "Model {} successfully loaded on {}",
        "image_label": "=== Image {}: {} ===",
        "prompt_label": "Prompt: {}",
        "result_label": "Result: {}",
        "model_size_warning": "⚠️ Note: Large models (8B+) may use CPU offloading if GPU memory is insufficient, which can slow down generation.",
        "quantization": "Quantization",
        "quantization_info": "Memory optimization (4-bit = ~75% less VRAM)",
        "quant_4bit": "4-bit (Recommended)",
        "quant_8bit": "8-bit (Better quality)",
        "quant_none": "None (Full precision)",
        "description_type": "Description Type",
        "description_type_info": "Select the type/style of description",
        "description_length": "Description Length",
        "description_length_info": "Select desired length",
        "num_variants": "Number of Variants",
        "num_variants_info": "Generate multiple description variants",
        "custom_prompt_override": "Custom Prompt (overrides type selection)",
        "custom_prompt_placeholder": "Enter your custom prompt here...",
        "status": "Status",
        "processing_time": "Processing time",
        "generating": "⏳ Generating...",
        "stop_btn": "🛑 Stop",
        "save_results": "💾 Save Results",
        "output_folder": "Output Folder Name",
        "output_folder_placeholder": "my_dataset",
        "export_format": "Export Format",
        "export_txt": "TXT (one file per image)",
        "export_json": "JSON (all results)",
        "export_csv": "CSV (table format)",
        "variant": "Variant",
        "copy_btn": "📋 Copy",
        "download_btn": "⬇️ Download",
        "generation_complete": "✅ Generation complete!",
        "seconds": "seconds",
        "processing_image": "Processing image",
        "of": "of",
        "extra_options": "Extra Options",
        "extra_options_info": "Additional modifiers for description",
        "character_name": "Character/Person Name",
        "character_name_placeholder": "e.g., John, Alice, or leave empty",
        "character_name_info": "If provided, will use this name instead of generic terms",
        "prompt_preset": "Prompt Preset",
        "prompt_preset_info": "Load a preset from prompts/ folder",
        "refresh_presets": "Refresh",
        "memory_usage": "Memory Usage",
        "download_result": "Download Result",
        "generation_stopped": "Generation stopped by user",
        "stopping": "Stopping..."
    },
    "ru": {
        "title": "Генератор описаний изображений Qwen VL",
        "header": "🖼️ Генератор описаний изображений на основе Qwen Vision Language Models",
        "subtitle": "Загрузите изображение и введите промт для генерации описания с помощью моделей Qwen VL.",
        "language": "Язык",
        "language_info": "Выберите язык",
        "model_selection": "Выбор модели",
        "model_info": "Выберите модель для генерации описаний",
        "advanced_params": "⚙️ Расширенные параметры",
        "max_tokens": "Макс. количество новых токенов",
        "max_tokens_info": "Максимальное количество токенов для генерации",
        "temperature": "Температура",
        "temperature_info": "Контролирует случайность генерации",
        "top_p": "Top-p (nucleus sampling)",
        "top_p_info": "Вероятностный порог для выборки токенов",
        "top_k": "Top-k",
        "top_k_info": "Количество наиболее вероятных токенов для рассмотрения",
        "seed": "Seed",
        "seed_info": "Seed для воспроизводимости (-1 для случайного)",
        "random_seed_btn": "🎲 Случайный seed",
        "single_processing": "📄 Одиночная обработка",
        "batch_processing": "📚 Пакетная обработка",
        "upload_image": "Загрузите изображение",
        "image_url": "Или введите URL изображения",
        "image_url_placeholder": "https://example.com/image.jpg",
        "prompt": "Промт",
        "prompt_placeholder": "Например: Создать описание товара для онлайн магазина",
        "generate_btn": "🚀 Генерировать описание",
        "result": "Результат",
        "upload_images": "Загрузите изображения",
        "prompts_multiline": "Промты (по одному на строку)",
        "prompts_placeholder": "Создать описание товара для онлайн магазина\nСоздать SEO-описание для товара\n...",
        "prompts_info": "Укажите один промт для всех изображений или по одному промту на каждое изображение",
        "process_batch_btn": "🚀 Обработать пакет",
        "results": "Результаты",
        "examples_title": "💡 Примеры промтов:",
        "example_1": "Создать описание товара ''  на русском языке",
        "example_2": "Создать SEO-описание для товара максимум 160 символов на русском языке",
        "example_3": "Создать привлекательное описание продукта для маркетплейса на русском языке",
        "example_4": "Детально описать изображение для каталога товаров на русском языке",
        "error_no_image": "Пожалуйста, загрузите изображение или укажите URL изображения",
        "error_no_prompt": "Пожалуйста, введите промт",
        "error_no_images": "Пожалуйста, загрузите изображения",
        "error_no_prompts": "Пожалуйста, введите промты (по одному на строку)",
        "error_prompt_mismatch": "Количество промтов ({}) не совпадает с количеством изображений ({}). Укажите либо один промт для всех изображений, либо по одному промту на каждое изображение.",
        "error_generation": "Ошибка при генерации описания: {}",
        "error_url_load": "Ошибка загрузки изображения по URL: {}",
        "loading_model": "Загрузка модели: {}",
        "model_loaded": "Модель {} успешно загружена на {}",
        "image_label": "=== Изображение {}: {} ===",
        "prompt_label": "Промт: {}",
        "result_label": "Результат: {}",
        "model_size_warning": "⚠️ Примечание: Большие модели (8B+) могут использовать выгрузку на CPU при недостатке памяти GPU, что может замедлить генерацию.",
        "quantization": "Квантизация",
        "quantization_info": "Оптимизация памяти (4-bit = ~75% меньше видеопамяти)",
        "quant_4bit": "4-bit (Рекомендуется)",
        "quant_8bit": "8-bit (Лучше качество)",
        "quant_none": "Нет (Полная точность)",
        "description_type": "Тип описания",
        "description_type_info": "Выберите тип/стиль описания",
        "description_length": "Длина описания",
        "description_length_info": "Выберите желаемую длину",
        "num_variants": "Количество вариантов",
        "num_variants_info": "Генерировать несколько вариантов описания",
        "custom_prompt_override": "Свой промпт (переопределяет выбор типа)",
        "custom_prompt_placeholder": "Введите свой промпт здесь...",
        "status": "Статус",
        "processing_time": "Время обработки",
        "generating": "⏳ Генерация...",
        "stop_btn": "🛑 Остановить",
        "save_results": "💾 Сохранить результаты",
        "output_folder": "Название папки",
        "output_folder_placeholder": "мой_датасет",
        "export_format": "Формат экспорта",
        "export_txt": "TXT (один файл на изображение)",
        "export_json": "JSON (все результаты)",
        "export_csv": "CSV (табличный формат)",
        "variant": "Вариант",
        "copy_btn": "📋 Копировать",
        "download_btn": "⬇️ Скачать",
        "generation_complete": "✅ Генерация завершена!",
        "seconds": "секунд",
        "processing_image": "Обработка изображения",
        "of": "из",
        "extra_options": "Дополнительные опции",
        "extra_options_info": "Дополнительные модификаторы описания",
        "character_name": "Имя персонажа/человека",
        "character_name_placeholder": "напр., Иван, Алиса, или оставьте пустым",
        "character_name_info": "Если указано, будет использоваться вместо общих терминов",
        "prompt_preset": "Пресет промпта",
        "prompt_preset_info": "Загрузить пресет из папки prompts/",
        "refresh_presets": "Обновить",
        "memory_usage": "Использование памяти",
        "download_result": "Скачать результат",
        "generation_stopped": "Генерация остановлена пользователем",
        "stopping": "Остановка..."
    },
    "zh": {
        "title": "Qwen VL 图像描述生成器",
        "header": "🖼️ 基于 Qwen Vision Language Models 的图像描述生成器",
        "subtitle": "上传图像并输入提示词，使用 Qwen VL 模型生成描述。",
        "language": "语言",
        "language_info": "选择语言",
        "model_selection": "模型选择",
        "model_info": "选择用于生成描述的模型",
        "advanced_params": "⚙️ 高级参数",
        "max_tokens": "最大新令牌数",
        "max_tokens_info": "生成的最大令牌数",
        "temperature": "温度",
        "temperature_info": "控制生成的随机性",
        "top_p": "Top-p（核采样）",
        "top_p_info": "令牌采样的概率阈值",
        "top_k": "Top-k",
        "top_k_info": "考虑的最可能令牌数",
        "seed": "随机种子",
        "seed_info": "用于可重现性的种子（-1 表示随机）",
        "random_seed_btn": "🎲 随机种子",
        "single_processing": "📄 单张处理",
        "batch_processing": "📚 批量处理",
        "upload_image": "上传图像",
        "image_url": "或输入图像URL",
        "image_url_placeholder": "https://example.com/image.jpg",
        "prompt": "提示词",
        "prompt_placeholder": "例如：为在线商店创建产品描述",
        "generate_btn": "🚀 生成描述",
        "result": "结果",
        "upload_images": "上传图像",
        "prompts_multiline": "提示词（每行一个）",
        "prompts_placeholder": "为在线商店创建产品描述\n为产品创建SEO描述\n...",
        "prompts_info": "为所有图像指定一个提示词，或为每个图像指定一个提示词",
        "process_batch_btn": "🚀 处理批次",
        "results": "结果",
        "examples_title": "💡 示例提示词：",
        "example_1": "为在线商店创建产品描述",
        "example_2": "为产品创建SEO描述最多 160 个字符",
        "example_3": "为市场创建有吸引力的产品描述",
        "example_4": "详细描述产品目录的图像",
        "error_no_image": "请上传图像或提供图像URL",
        "error_no_prompt": "请输入提示词",
        "error_no_images": "请上传图像",
        "error_no_prompts": "请输入提示词（每行一个）",
        "error_prompt_mismatch": "提示词数量（{}）与图像数量（{}）不匹配。请为所有图像指定一个提示词，或为每个图像指定一个提示词。",
        "error_generation": "生成描述时出错：{}",
        "error_url_load": "从URL加载图像时出错：{}",
        "loading_model": "正在加载模型：{}",
        "model_loaded": "模型 {} 已成功加载到 {}",
        "image_label": "=== 图像 {}: {} ===",
        "prompt_label": "提示词：{}",
        "result_label": "结果：{}",
        "model_size_warning": "⚠️ 注意：如果 GPU 内存不足，大型模型（8B+）可能会使用 CPU 卸载，这可能会减慢生成速度。",
        "quantization": "量化",
        "quantization_info": "内存优化（4位 = 减少约75%显存）",
        "quant_4bit": "4位（推荐）",
        "quant_8bit": "8位（更高质量）",
        "quant_none": "无（全精度）",
        "description_type": "描述类型",
        "description_type_info": "选择描述类型/风格",
        "description_length": "描述长度",
        "description_length_info": "选择所需长度",
        "num_variants": "变体数量",
        "num_variants_info": "生成多个描述变体",
        "custom_prompt_override": "自定义提示词（覆盖类型选择）",
        "custom_prompt_placeholder": "在此输入您的自定义提示词...",
        "status": "状态",
        "processing_time": "处理时间",
        "generating": "⏳ 生成中...",
        "stop_btn": "🛑 停止",
        "save_results": "💾 保存结果",
        "output_folder": "输出文件夹名称",
        "output_folder_placeholder": "我的数据集",
        "export_format": "导出格式",
        "export_txt": "TXT（每张图片一个文件）",
        "export_json": "JSON（所有结果）",
        "export_csv": "CSV（表格格式）",
        "variant": "变体",
        "copy_btn": "📋 复制",
        "download_btn": "⬇️ 下载",
        "generation_complete": "✅ 生成完成！",
        "seconds": "秒",
        "processing_image": "处理图像",
        "of": "/",
        "extra_options": "额外选项",
        "extra_options_info": "描述的额外修饰符",
        "character_name": "角色/人物名称",
        "character_name_placeholder": "例如：小明、小红，或留空",
        "character_name_info": "如果提供，将使用此名称代替通用术语",
        "prompt_preset": "提示词预设",
        "prompt_preset_info": "从prompts/文件夹加载预设",
        "refresh_presets": "刷新",
        "memory_usage": "内存使用",
        "download_result": "下载结果",
        "generation_stopped": "用户停止了生成",
        "stopping": "停止中..."
    }
}

# Default language
current_language = "ru"

def get_text(key: str) -> str:
    """Get translated text for the current language"""
    return TRANSLATIONS[current_language].get(key, key)

def get_description_types() -> list:
    """Get description types for current language"""
    return list(DESCRIPTION_TYPES.get(current_language, DESCRIPTION_TYPES["en"]).keys())

def get_description_lengths() -> list:
    """Get description lengths for current language"""
    return list(DESCRIPTION_LENGTHS.get(current_language, DESCRIPTION_LENGTHS["en"]).keys())

def get_extra_options(is_video: bool = False) -> list:
    """Get extra options for current language and media type"""
    options_dict = EXTRA_OPTIONS_VIDEO if is_video else EXTRA_OPTIONS
    return list(options_dict.get(current_language, options_dict["en"]).keys())

def get_extra_option_prompt(option: str, is_video: bool = False) -> str:
    """Get the prompt text for an extra option"""
    options_dict = EXTRA_OPTIONS_VIDEO if is_video else EXTRA_OPTIONS
    lang_options = options_dict.get(current_language, options_dict["en"])
    return lang_options.get(option, "")

def build_prompt(
    description_type: str,
    description_length: str,
    custom_prompt: str,
    base_prompt: str = "",
    extra_options: list = None,
    character_name: str = "",
    is_video: bool = False
) -> str:
    """Build the final prompt based on type, length, custom input, extra options and character name.
    Automatically replaces 'image' with 'video' when is_video=True for context-aware prompts."""
    # If custom prompt is provided, use it (but still add character name if present)
    if custom_prompt and custom_prompt.strip():
        final_prompt = custom_prompt.strip()
    else:
        # Get type prompt
        types_dict = DESCRIPTION_TYPES.get(current_language, DESCRIPTION_TYPES["en"])
        type_prompt = types_dict.get(description_type, "")

        # If type is Custom, use base prompt
        if not type_prompt:
            if is_video:
                type_prompt = base_prompt if base_prompt else "Describe this video."
            else:
                type_prompt = base_prompt if base_prompt else "Describe this image."

        # Get length modifier
        lengths_dict = DESCRIPTION_LENGTHS.get(current_language, DESCRIPTION_LENGTHS["en"])
        # Fix if description_length is a list (shouldn't happen but Gradio bug)
        if isinstance(description_length, list):
            description_length = description_length[0] if description_length else ""
        length_modifier = lengths_dict.get(description_length, "")

        # Combine type and length
        if length_modifier:
            final_prompt = f"{type_prompt} {length_modifier}"
        else:
            final_prompt = type_prompt

    # Context-aware: Replace "image" with "video" when processing video
    if is_video:
        # English replacements
        final_prompt = final_prompt.replace(" image.", " video.")
        final_prompt = final_prompt.replace(" image ", " video ")
        final_prompt = final_prompt.replace("this image", "this video")
        final_prompt = final_prompt.replace("the image", "the video")
        final_prompt = final_prompt.replace("an image", "a video")
        # Russian replacements
        final_prompt = final_prompt.replace("изображения", "видео")
        final_prompt = final_prompt.replace("изображении", "видео")
        final_prompt = final_prompt.replace("изображение", "видео")
        # Chinese replacements
        final_prompt = final_prompt.replace("图片", "视频")
        final_prompt = final_prompt.replace("图像", "视频")

    # Add character name instruction if provided
    if character_name and character_name.strip():
        name = character_name.strip()
        media_term_ru = "видео" if is_video else "изображении"
        media_term_zh = "视频" if is_video else "图片"
        media_term_en = "video" if is_video else "image"

        if current_language == "ru":
            final_prompt += f" Если на {media_term_ru} есть человек или персонаж, используй имя '{name}' вместо общих терминов."
        elif current_language == "zh":
            final_prompt += f" 如果{media_term_zh}中有人或角色，请使用名字'{name}'而不是通用术语。"
        else:
            final_prompt += f" If there is a person or character in the {media_term_en}, use the name '{name}' instead of generic terms."

    # Add extra options
    if extra_options:
        for option in extra_options:
            option_prompt = get_extra_option_prompt(option, is_video=is_video)
            if option_prompt:
                final_prompt += f" {option_prompt}"

    return final_prompt

def create_output_folder(folder_name: str = None) -> str:
    """Create and return path to output folder"""
    if folder_name and folder_name.strip():
        # Sanitize folder name
        safe_name = "".join(c for c in folder_name if c.isalnum() or c in "_ -").strip()
        if not safe_name:
            safe_name = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    else:
        safe_name = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    folder_path = os.path.join(DATASETS_DIR, safe_name)
    os.makedirs(folder_path, exist_ok=True)
    return folder_path

def save_results_txt(results: List[dict], output_folder: str) -> List[str]:
    """Save results as individual TXT files"""
    saved_files = []
    for result in results:
        filename = os.path.splitext(os.path.basename(result.get("image_path", "image")))[0]
        txt_path = os.path.join(output_folder, f"{filename}.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            if isinstance(result.get("descriptions"), list):
                for i, desc in enumerate(result["descriptions"], 1):
                    f.write(f"=== Variant {i} ===\n{desc}\n\n")
            else:
                f.write(result.get("description", ""))
        saved_files.append(txt_path)

        # Copy image if exists
        if result.get("image_path") and os.path.exists(result["image_path"]):
            img_dest = os.path.join(output_folder, os.path.basename(result["image_path"]))
            if not os.path.exists(img_dest):
                shutil.copy2(result["image_path"], img_dest)

    return saved_files

def save_results_json(results: List[dict], output_folder: str) -> str:
    """Save all results as a single JSON file"""
    json_path = os.path.join(output_folder, "results.json")
    export_data = []
    for result in results:
        export_data.append({
            "image": os.path.basename(result.get("image_path", "")),
            "prompt": result.get("prompt", ""),
            "descriptions": result.get("descriptions", [result.get("description", "")]),
            "timestamp": datetime.now().isoformat()
        })

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(export_data, f, ensure_ascii=False, indent=2)

    return json_path

def save_results_csv(results: List[dict], output_folder: str) -> str:
    """Save all results as a CSV file"""
    csv_path = os.path.join(output_folder, "results.csv")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Image", "Prompt", "Description", "Variant"])

        for result in results:
            image_name = os.path.basename(result.get("image_path", ""))
            prompt = result.get("prompt", "")
            descriptions = result.get("descriptions", [result.get("description", "")])

            for i, desc in enumerate(descriptions, 1):
                writer.writerow([image_name, prompt, desc, i])

    return csv_path

class ImageDescriptionGenerator:
    def __init__(self):
        self.model = None
        self.processor = None
        self.current_model_name = None
        self.current_quantization = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load_model(self, model_name: str, quantization: str = "4-bit"):
        """Загрузка модели с квантизацией BitsAndBytes"""
        # Проверяем нужна ли перезагрузка
        if (self.current_model_name == model_name and
            self.current_quantization == quantization and
            self.model is not None):
            print(f"✅ Модель {model_name} уже загружена")
            return

        print(f"\n{'='*50}")
        print(f"🧠 Загрузка модели: {model_name}")
        print(f"⚙️ Квантизация: {quantization}")
        print(f"💻 Устройство: {self.device}")
        print(f"{'='*50}")

        # Проверяем, скачана ли модель
        cached_size = get_model_cache_size(model_name)
        if cached_size:
            print(f"✅ Модель найдена в кэше [{cached_size}]")
        else:
            print(f"⬇️ Модель не найдена в кэше - будет скачана с HuggingFace...")
            print(f"⏳ Это может занять несколько минут при первом запуске")

        # Сохраняем старую модель на случай ошибки
        old_model = self.model
        old_processor = self.processor
        old_model_name = self.current_model_name
        old_quantization = self.current_quantization

        try:
            # Освобождаем память от предыдущей модели
            if self.model is not None:
                print(f"🗑️ Выгружаем предыдущую модель: {old_model_name}")
                self.model = None
                self.processor = None
                del old_model
                del old_processor
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                print(f"✅ Память очищена")

            # Настройка квантизации BitsAndBytes
            bnb_config = None
            dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

            if quantization == "4-bit" and torch.cuda.is_available():
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
                print("⚡ 4-bit квантизация (NF4) — экономия ~75% VRAM")
            elif quantization == "8-bit" and torch.cuda.is_available():
                bnb_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                )
                print("⚡ 8-bit квантизация — экономия ~50% VRAM")
            else:
                print("📊 Полная точность (bfloat16/float32)")

            # Загружаем новую модель с подавлением предупреждений
            print(f"🔄 Загрузка модели...")
            load_start_time = time.time()

            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')
                load_kwargs = {
                    "device_map": "auto",
                    "trust_remote_code": True,
                }

                if bnb_config is not None:
                    load_kwargs["quantization_config"] = bnb_config
                else:
                    load_kwargs["torch_dtype"] = dtype

                # Попробуем Flash Attention 2 для максимальной скорости
                if torch.cuda.is_available():
                    try:
                        # Проверяем доступность Flash Attention 2
                        import flash_attn
                        load_kwargs["attn_implementation"] = "flash_attention_2"
                        print("🚀 Flash Attention 2 (максимальная скорость)")
                    except ImportError:
                        # Fallback на SDPA (медленнее, но стабильно)
                        load_kwargs["attn_implementation"] = "sdpa"
                        print("⚡ SDPA attention (PyTorch native - медленнее Flash Attention 2)")
                        print("💡 Установите flash-attn для ускорения: pip install flash-attn --no-build-isolation")

                self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                    model_name,
                    **load_kwargs
                )

                # Оптимизация с torch.compile (только для non-quantized моделей)
                # torch.compile может не работать с BitsAndBytes квантизацией
                if quantization == "Нет" and torch.cuda.is_available():
                    try:
                        print("🔥 Компиляция модели с torch.compile (ускорение ~20-40%)...")
                        # mode="reduce-overhead" лучше для generative tasks
                        self.model.forward = torch.compile(
                            self.model.forward,
                            mode="reduce-overhead",
                            fullgraph=False
                        )
                        print("✅ Компиляция завершена (первая генерация будет медленнее)")
                    except Exception as e:
                        print(f"⚠️ torch.compile недоступен: {e}")

                print(f"🔄 Загрузка процессора...")
                self.processor = AutoProcessor.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    # Оптимизация разрешения для скорости
                    min_pixels=256*28*28,  # Min resolution
                    max_pixels=1280*28*28  # Max resolution (не слишком большое)
                )

            load_time = time.time() - load_start_time
            self.current_model_name = model_name
            self.current_quantization = quantization

            # Показываем информацию о памяти
            memory_info = get_memory_info()
            print(f"\n{'='*50}")
            print(f"✅ Модель загружена за {load_time:.1f} сек")
            print(f"📊 {memory_info}")
            print(f"{'='*50}\n")

        except Exception as e:
            # При ошибке загрузки - очищаем состояние
            print(f"❌ Ошибка загрузки модели: {str(e)}")
            self.model = None
            self.processor = None
            self.current_model_name = None
            self.current_quantization = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise Exception(f"Ошибка загрузки модели {model_name}: {str(e)}")
    
    def _prepare_inputs(
        self,
        media_path: str,
        prompt: str,
        model_name: str,
        quantization: str,
        seed: int,
        is_video: bool = False,
        video_start_time = None,
        video_end_time = None
    ):
        """Prepare inputs for generation"""
        # Загружаем модель если необходимо
        self.load_model(model_name, quantization)

        # Проверяем что модель загружена
        if self.model is None or self.processor is None:
            raise Exception("Модель не загружена. Попробуйте выбрать другую модель или перезапустить приложение.")

        # Устанавливаем seed если указан
        if seed != -1:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)

        # Подготавливаем сообщения для модели
        # Use "video" type for videos, "image" for images
        if is_video:
            content_item = {
                "type": "video",
                "video": media_path,
            }
            # Add video segment parameters if provided
            # Get video duration to validate end time
            video_duration = get_video_duration(media_path) if media_path else 0

            # video_start: only add if > 0
            if video_start_time is not None and video_start_time > 0:
                content_item["video_start"] = float(video_start_time)

            # video_end: only add if > 0 AND < video duration (otherwise means "to end")
            if video_end_time is not None and video_end_time > 0:
                # If video_end is close to duration (within 1 second), don't add (means full video)
                if video_duration > 0 and video_end_time < (video_duration - 1):
                    content_item["video_end"] = float(video_end_time)
        else:
            content_item = {
                "type": "image",
                "image": media_path,
            }

        messages = [
            {
                "role": "user",
                "content": [
                    content_item,
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        # Подготавливаем текст для модели
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Обрабатываем изображение/видео и текст
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.device)
        return inputs

    def _prepare_inputs_multi_image(
        self,
        image_paths: list,
        prompt: str,
        model_name: str,
        quantization: str,
        seed: int
    ):
        """Prepare inputs for multiple images generation"""
        # Загружаем модель если необходимо
        self.load_model(model_name, quantization)

        # Проверяем что модель загружена
        if self.model is None or self.processor is None:
            raise Exception("Модель не загружена. Попробуйте выбрать другую модель или перезапустить приложение.")

        # Устанавливаем seed если указан
        if seed != -1:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)

        # Создаём список контента с множественными изображениями
        content = []
        for img_path in image_paths:
            content.append({
                "type": "image",
                "image": img_path,
            })

        # Добавляем текстовый промпт в конец
        content.append({"type": "text", "text": prompt})

        messages = [
            {
                "role": "user",
                "content": content,
            }
        ]

        # Подготавливаем текст для модели
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Обрабатываем изображения и текст
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.device)
        return inputs

    def generate_description(
        self,
        image_path: str,
        prompt: str,
        model_name: str,
        quantization: str = "4-bit",
        max_new_tokens: int = 1024,
        temperature: float = 0.6,
        top_p: float = 0.9,
        top_k: int = 50,
        seed: int = -1,
        is_video: bool = False,
        video_start_time = None,
        video_end_time = None
    ) -> str:
        """Генерация описания для одного изображения/видео (без streaming)"""
        try:
            inputs = self._prepare_inputs(
                image_path, prompt, model_name, quantization, seed, is_video,
                video_start_time, video_end_time
            )

            # Non-streaming generation (inference_mode быстрее чем no_grad)
            with torch.inference_mode():
                # Оптимизированные параметры генерации
                gen_kwargs = {
                    **inputs,
                    "max_new_tokens": max_new_tokens,
                    "use_cache": True,  # KV кэширование для ускорения
                    "pad_token_id": self.processor.tokenizer.pad_token_id,
                }

                # Greedy decoding быстрее sampling
                if temperature > 0:
                    gen_kwargs.update({
                        "do_sample": True,
                        "temperature": temperature,
                        "top_p": top_p,
                        "top_k": top_k,
                    })
                else:
                    gen_kwargs["do_sample"] = False

                generated_ids = self.model.generate(**gen_kwargs)

            # Декодируем результат
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )

            return output_text[0]

        except Exception as e:
            error_msg = get_text("error_generation").format(str(e))
            return error_msg

    def generate_description_stream(
        self,
        image_path: str,
        prompt: str,
        model_name: str,
        quantization: str = "4-bit",
        max_new_tokens: int = 1024,
        temperature: float = 0.6,
        top_p: float = 0.9,
        top_k: int = 50,
        seed: int = -1,
        is_video: bool = False,
        video_start_time = None,
        video_end_time = None
    ) -> Generator[str, None, None]:
        """Генерация описания для одного изображения/видео со streaming"""
        try:
            inputs = self._prepare_inputs(
                image_path, prompt, model_name, quantization, seed, is_video,
                video_start_time, video_end_time
            )

            streamer = TextIteratorStreamer(
                self.processor.tokenizer,
                skip_special_tokens=True,
                skip_prompt=True
            )

            generation_kwargs = {
                **inputs,
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "do_sample": True if temperature > 0 else False,
                "use_cache": True,
                "streamer": streamer,
            }

            # Run generation in a separate thread
            thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
            thread.start()

            # Yield tokens as they come
            generated_text = ""
            for new_text in streamer:
                generated_text += new_text
                yield generated_text

            thread.join()

        except Exception as e:
            error_msg = get_text("error_generation").format(str(e))
            yield error_msg

    def generate_description_multi_image_stream(
        self,
        image_paths: list,
        prompt: str,
        model_name: str,
        quantization: str = "4-bit",
        max_new_tokens: int = 2048,
        temperature: float = 0.6,
        top_p: float = 0.9,
        top_k: int = 50,
        seed: int = -1
    ) -> Generator[str, None, None]:
        """Генерация описания для множественных изображений со streaming"""
        try:
            inputs = self._prepare_inputs_multi_image(
                image_paths, prompt, model_name, quantization, seed
            )

            streamer = TextIteratorStreamer(
                self.processor.tokenizer,
                skip_special_tokens=True,
                skip_prompt=True
            )

            generation_kwargs = {
                **inputs,
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "do_sample": True if temperature > 0 else False,
                "use_cache": True,
                "streamer": streamer,
            }

            # Run generation in a separate thread
            thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
            thread.start()

            # Yield tokens as they come
            generated_text = ""
            for new_text in streamer:
                generated_text += new_text
                yield generated_text

            thread.join()

        except Exception as e:
            error_msg = get_text("error_generation").format(str(e))
            yield error_msg

# Создаем глобальный экземпляр генератора
generator = ImageDescriptionGenerator()

def process_single_image(
    image,
    video,
    video_start_time,
    video_end_time,
    description_type: str,
    description_length: str,
    custom_prompt: str,
    extra_options: list,
    character_name: str,
    num_variants: int,
    model_name: str,
    quantization: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    seed: int,
    use_streaming: bool = True,
    progress=gr.Progress(track_tqdm=True)
) -> Generator:
    """Обработка одного изображения/видео с генерацией нескольких вариантов и streaming output"""
    global stop_generation_flag
    reset_stop_flag()
    start_time = time.time()

    # Start capturing console output
    log_capture.clear_logs()
    log_capture.start_capture()
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Начало генерации...")

    # Check if we have either uploaded image or video
    if image is None and video is None:
        yield get_text("error_no_image"), "", [], None, log_capture.get_logs(), "", None
        return

    # Determine if processing video
    is_video = video is not None

    # Build prompt from type, length, custom, extra options and character name
    final_prompt = build_prompt(
        description_type, description_length, custom_prompt,
        extra_options=extra_options or [],
        character_name=character_name or "",
        is_video=is_video
    )
    if not final_prompt.strip():
        yield get_text("error_no_prompt"), "", [], None, log_capture.get_logs(), "", None
        return

    temp_path = None
    num_variants = int(num_variants) if num_variants and str(num_variants).strip() else 1

    try:
        # Priority: video > image
        if video is not None:
            # Video uploaded - use it directly
            media_path = video
            is_video = True
        elif hasattr(image, 'shape'):
            # Сохраняем временное изображение если это numpy array
            temp_path = os.path.join(TEMP_DIR, "temp_image.jpg")
            Image.fromarray(image).save(temp_path)
            media_path = temp_path
            is_video = False
        else:
            media_path = image
            is_video = False

        results = []
        full_results = []  # Store full results before parsing
        reasoning_list = []  # Store reasoning from thinking models
        variant_times = []  # Track time for each variant
        image_preview_path = None  # Path to image with bboxes drawn

        for i in range(num_variants):
            # Check stop flag
            if stop_generation_flag:
                elapsed_time = time.time() - start_time
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Генерация остановлена пользователем")
                yield f"🛑 {get_text('generation_stopped')} ({get_text('processing_time')}: {elapsed_time:.1f} {get_text('seconds')})", final_prompt, results, None, log_capture.get_logs(), "", None
                return

            variant_start = time.time()
            variant_seed = seed if seed == -1 else seed + i
            memory_info = get_memory_info()
            status_msg = f"{get_text('generating')} ({get_text('variant')} {i+1}/{num_variants}) | {memory_info}"
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Генерация варианта {i+1}/{num_variants}, seed={variant_seed}")

            if use_streaming:
                # Stream the generation for real-time output
                current_result = ""
                for partial_result in generator.generate_description_stream(
                    image_path=media_path,
                    prompt=final_prompt,
                    model_name=model_name,
                    quantization=quantization,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    seed=variant_seed,
                    is_video=is_video,
                    video_start_time=video_start_time,
                    video_end_time=video_end_time
                ):
                    if stop_generation_flag:
                        break
                    current_result = partial_result
                    # Update results with streaming text
                    temp_results = results + [current_result]
                    yield status_msg, final_prompt, temp_results, None, log_capture.get_logs(), "", None

                result = current_result
            else:
                # Non-streaming generation
                yield status_msg, final_prompt, results, None, log_capture.get_logs(), "", None
                result = generator.generate_description(
                    image_path=media_path,
                    prompt=final_prompt,
                    model_name=model_name,
                    quantization=quantization,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    seed=variant_seed,
                    is_video=is_video,
                    video_start_time=video_start_time,
                    video_end_time=video_end_time
                )

            variant_time = time.time() - variant_start
            variant_times.append(variant_time)

            # Store full result before parsing
            full_results.append(result)

            # Parse thinking output if present
            reasoning, final_answer = parse_thinking_output(result)
            reasoning_list.append(reasoning)  # Store reasoning (empty string if no thinking)
            results.append(final_answer)  # Store only final answer in results

            print(f"[{datetime.now().strftime('%H:%M:%S')}] Вариант {i+1} завершен за {variant_time:.1f}s")

        # Calculate processing time
        elapsed_time = time.time() - start_time
        memory_info = get_memory_info()

        # Check for thinking output and bounding boxes in results
        total_bboxes = 0
        has_thinking = any(r for r in reasoning_list if r)  # Check if any reasoning exists
        first_bbox_result = None

        for i, full_result in enumerate(full_results):
            # Log thinking if present
            if reasoning_list[i]:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] 💭 Variant {i+1}: Thinking model detected")
                print(f"  Reasoning length: {len(reasoning_list[i])} chars")
                print(f"  Answer length: {len(results[i])} chars")

            # Check for bounding boxes in full result
            bboxes = parse_bboxes_from_text(full_result)
            if bboxes:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] 📍 Variant {i+1}: Found {len(bboxes)} bounding boxes")
                for j, bbox in enumerate(bboxes):
                    label = bbox.get('label', 'unknown')
                    coords = bbox.get('bbox_2d', [])
                    print(f"  - Object {j+1}: {label} at {coords}")
                total_bboxes += len(bboxes)

                # Draw bboxes on image for first variant with bboxes
                if first_bbox_result is None and not is_video and media_path:
                    try:
                        image_preview_path = draw_bboxes_on_image(media_path, bboxes, normalized=True)
                        first_bbox_result = i
                        print(f"[{datetime.now().strftime('%H:%M:%S')}] 📍 Drew {len(bboxes)} bboxes on image: {image_preview_path}")
                    except Exception as e:
                        print(f"[{datetime.now().strftime('%H:%M:%S')}] ⚠️ Failed to draw bboxes: {e}")

        # Build detailed status with per-variant timing
        timing_details = " | ".join([f"V{i+1}: {t:.1f}s" for i, t in enumerate(variant_times)])
        thinking_info = " | 💭 Thinking" if has_thinking else ""
        bbox_info = f" | 📍 {total_bboxes} bbox" if total_bboxes > 0 else ""
        final_status = f"{get_text('generation_complete')} | Total: {elapsed_time:.1f}s ({timing_details}){thinking_info}{bbox_info} | {memory_info}"

        # Prepare download file
        download_path = None
        if results:
            all_text = "\n\n".join([f"=== Variant {i+1} (Time: {variant_times[i]:.1f}s) ===\n{r}" for i, r in enumerate(results)])
            download_path = save_text_to_file(all_text, f"result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")

        # Get first reasoning (or empty string)
        first_reasoning = reasoning_list[0] if reasoning_list and reasoning_list[0] else ""

        print(f"[{datetime.now().strftime('%H:%M:%S')}] Генерация завершена успешно за {elapsed_time:.1f}s")
        yield final_status, final_prompt, results, download_path, log_capture.get_logs(), first_reasoning, image_preview_path

    except Exception as e:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Ошибка: {str(e)}")
        yield f"❌ Error: {str(e)}", final_prompt, [], None, log_capture.get_logs(), "", None
    finally:
        # Stop capturing console output
        log_capture.stop_capture()

        # Удаляем временный файл
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass

        # Clean up GPU memory
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def process_multi_image(
    multi_images_list,
    description_type: str,
    description_length: str,
    custom_prompt: str,
    extra_options: list,
    character_name: str,
    num_variants: int,
    model_name: str,
    quantization: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    seed: int,
    use_streaming: bool = True,
    progress=gr.Progress(track_tqdm=True)
) -> Generator:
    """Обработка множественных изображений с генерацией вариантов и streaming output"""
    global stop_generation_flag
    reset_stop_flag()
    start_time = time.time()

    # Start capturing console output
    log_capture.clear_logs()
    log_capture.start_capture()
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Начало генерации для множественных изображений...")

    # Check if we have images
    if not multi_images_list or len(multi_images_list) == 0:
        yield "❌ Ошибка: Загрузите хотя бы одно изображение", "", [], None, log_capture.get_logs()
        return

    # Extract file paths from Gallery format
    if isinstance(multi_images_list, list):
        # Gallery returns list of dicts or tuples or strings
        image_paths = []
        for item in multi_images_list:
            if isinstance(item, dict) and 'name' in item:
                image_paths.append(item['name'])
            elif isinstance(item, tuple):
                image_paths.append(item[0])
            elif isinstance(item, str):
                image_paths.append(item)
            else:
                image_paths.append(str(item))
    else:
        yield "❌ Ошибка: Неверный формат изображений", "", [], None, log_capture.get_logs()
        return

    num_images = len(image_paths)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Загружено {num_images} изображений")

    # Limit to 20 images max
    if num_images > 20:
        yield f"❌ Ошибка: Максимум 20 изображений, загружено {num_images}", "", [], None, log_capture.get_logs()
        return

    # Build prompt from type, length, custom, extra options and character name
    final_prompt = build_prompt(
        description_type, description_length, custom_prompt,
        extra_options=extra_options or [],
        character_name=character_name or "",
        is_video=False  # Always false for multi-image
    )
    if not final_prompt.strip():
        yield "❌ Ошибка: Промпт пустой", "", [], None, log_capture.get_logs()
        return

    num_variants = int(num_variants) if num_variants and str(num_variants).strip() else 1

    try:
        results = []
        variant_times = []

        for i in range(num_variants):
            # Check stop flag
            if stop_generation_flag:
                elapsed_time = time.time() - start_time
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Генерация остановлена пользователем")
                yield f"🛑 {get_text('generation_stopped')} ({get_text('processing_time')}: {elapsed_time:.1f} {get_text('seconds')})", final_prompt, results, None, log_capture.get_logs(), "", None
                return

            variant_start = time.time()
            variant_seed = seed if seed == -1 else seed + i
            memory_info = get_memory_info()
            status_msg = f"{get_text('generating')} ({get_text('variant')} {i+1}/{num_variants}) | {num_images} изображений | {memory_info}"
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Генерация варианта {i+1}/{num_variants}, seed={variant_seed}")

            if use_streaming:
                # Stream the generation for real-time output
                current_result = ""
                for partial_result in generator.generate_description_multi_image_stream(
                    image_paths=image_paths,
                    prompt=final_prompt,
                    model_name=model_name,
                    quantization=quantization,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    seed=variant_seed
                ):
                    if stop_generation_flag:
                        break
                    current_result = partial_result
                    # Update results with streaming text
                    temp_results = results + [current_result]
                    yield status_msg, final_prompt, temp_results, None, log_capture.get_logs()

                result = current_result
            else:
                # Non-streaming generation (not implemented yet, use streaming)
                yield status_msg, final_prompt, results, None, log_capture.get_logs()
                # Fallback to streaming
                for partial_result in generator.generate_description_multi_image_stream(
                    image_paths=image_paths,
                    prompt=final_prompt,
                    model_name=model_name,
                    quantization=quantization,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    seed=variant_seed
                ):
                    result = partial_result

            variant_time = time.time() - variant_start
            variant_times.append(variant_time)
            results.append(result)
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Вариант {i+1} завершен за {variant_time:.1f}s")

        # Calculate processing time
        elapsed_time = time.time() - start_time
        memory_info = get_memory_info()

        # Check for thinking output and bounding boxes in results
        total_bboxes = 0
        has_thinking = False
        for i, result in enumerate(results):
            # Check for thinking output
            reasoning, final_answer = parse_thinking_output(result)
            if reasoning:
                has_thinking = True
                print(f"[{datetime.now().strftime('%H:%M:%S')}] 💭 Variant {i+1}: Thinking model detected")
                print(f"  Reasoning length: {len(reasoning)} chars")
                print(f"  Answer length: {len(final_answer)} chars")

            # Check for bounding boxes
            bboxes = parse_bboxes_from_text(result)
            if bboxes:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] 📍 Variant {i+1}: Found {len(bboxes)} bounding boxes")
                for j, bbox in enumerate(bboxes):
                    label = bbox.get('label', 'unknown')
                    coords = bbox.get('bbox_2d', [])
                    print(f"  - Object {j+1}: {label} at {coords}")
                total_bboxes += len(bboxes)

        # Build detailed status with per-variant timing
        timing_details = " | ".join([f"V{i+1}: {t:.1f}s" for i, t in enumerate(variant_times)])
        thinking_info = " | 💭 Thinking" if has_thinking else ""
        bbox_info = f" | 📍 {total_bboxes} bbox" if total_bboxes > 0 else ""
        final_status = f"{get_text('generation_complete')} | Total: {elapsed_time:.1f}s ({timing_details}){thinking_info}{bbox_info} | {memory_info}"

        # Prepare download file
        download_path = None
        if results:
            all_text = "\n\n".join([f"=== Variant {i+1} (Time: {variant_times[i]:.1f}s) ===\n{r}" for i, r in enumerate(results)])
            download_path = save_text_to_file(all_text, f"multi_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")

        print(f"[{datetime.now().strftime('%H:%M:%S')}] Генерация завершена успешно за {elapsed_time:.1f}s")
        yield final_status, final_prompt, results, download_path, log_capture.get_logs()

    except Exception as e:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Ошибка: {str(e)}")
        yield f"❌ Error: {str(e)}", final_prompt, [], None, log_capture.get_logs()
    finally:
        # Stop capturing console output
        log_capture.stop_capture()

        # Clean up GPU memory
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def process_batch_images(
    files: List,
    description_type: str,
    description_length: str,
    custom_prompt: str,
    extra_options: list,
    character_name: str,
    num_variants: int,
    output_folder_name: str,
    export_formats: List[str],
    model_name: str,
    quantization: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    seed: int,
    is_video: bool = False,
    progress=gr.Progress(track_tqdm=True)
) -> Generator:
    """Обработка пакета изображений/видео с прогресс-баром и экспортом"""
    global stop_generation_flag
    reset_stop_flag()
    start_time = time.time()

    if not files:
        yield get_text("error_no_images"), "", None
        return

    # Build prompt from type, length, custom, extra options and character name
    final_prompt = build_prompt(
        description_type, description_length, custom_prompt,
        extra_options=extra_options or [],
        character_name=character_name or "",
        is_video=is_video
    )
    if not final_prompt.strip():
        yield get_text("error_no_prompts"), "", None
        return

    num_variants = int(num_variants)
    total_files = len(files)
    all_results = []
    output_lines = []

    # Create output folder
    output_folder = create_output_folder(output_folder_name)

    media_type = "video" if is_video else get_text('processing_image')

    for idx, file in enumerate(progress.tqdm(files, desc=f"Processing {media_type}s")):
        # Check stop flag
        if stop_generation_flag:
            elapsed_time = time.time() - start_time
            final_status = f"🛑 {get_text('generation_stopped')}\n"
            final_status += f"📊 {idx} {get_text('of')} {total_files} files processed in {elapsed_time:.1f} {get_text('seconds')}"
            yield final_status, "\n".join(output_lines), None
            return

        image_path = file.name if hasattr(file, 'name') else file
        filename = os.path.basename(image_path)

        # Status update with memory info
        memory_info = get_memory_info()
        status_msg = f"⏳ Processing {idx + 1} {get_text('of')} {total_files}: {filename} | {memory_info}"
        yield status_msg, "\n".join(output_lines), None

        descriptions = []
        variant_times = []

        for v in range(num_variants):
            # Check stop flag between variants
            if stop_generation_flag:
                break

            variant_start = time.time()
            variant_seed = seed if seed == -1 else seed + idx * num_variants + v

            result = generator.generate_description(
                image_path=image_path,
                prompt=final_prompt,
                model_name=model_name,
                quantization=quantization,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                seed=variant_seed,
                is_video=is_video
            )
            variant_time = time.time() - variant_start
            variant_times.append(variant_time)
            descriptions.append(result)

        if not descriptions:
            continue

        # Store result
        all_results.append({
            "image_path": image_path,
            "prompt": final_prompt,
            "descriptions": descriptions,
            "description": descriptions[0] if descriptions else ""
        })

        # Add to output with timing info
        output_lines.append(f"{'='*50}")
        output_lines.append(get_text("image_label").format(idx + 1, filename))
        output_lines.append(get_text("prompt_label").format(final_prompt))

        for v_idx, desc in enumerate(descriptions, 1):
            if num_variants > 1:
                timing_info = f" ({variant_times[v_idx-1]:.1f}s)" if v_idx-1 < len(variant_times) else ""
                output_lines.append(f"\n--- {get_text('variant')} {v_idx}{timing_info} ---")
            output_lines.append(f"{desc}\n")

        yield status_msg, "\n".join(output_lines), None

    # Save results in selected formats
    saved_paths = []
    if export_formats and all_results:
        if "TXT" in export_formats:
            txt_files = save_results_txt(all_results, output_folder)
            saved_paths.extend(txt_files)

        if "JSON" in export_formats:
            json_path = save_results_json(all_results, output_folder)
            saved_paths.append(json_path)

        if "CSV" in export_formats:
            csv_path = save_results_csv(all_results, output_folder)
            saved_paths.append(csv_path)

    # Calculate total time
    elapsed_time = time.time() - start_time
    memory_info = get_memory_info()

    # Final status
    final_status = f"{get_text('generation_complete')}\n"
    final_status += f"📊 {total_files} images processed in {elapsed_time:.1f} {get_text('seconds')} | {memory_info}\n"
    if saved_paths:
        final_status += f"💾 Results saved to: {output_folder}"

    # Prepare download file for batch results
    download_path = None
    if all_results:
        download_path = save_text_to_file("\n".join(output_lines), f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")

    yield final_status, "\n".join(output_lines), download_path

    # Clean up GPU memory
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def random_seed() -> int:
    """Генерация случайного seed"""
    return random.randint(0, 2**32 - 1)

def update_examples():
    return [
        [get_text("example_1")],
        [get_text("example_2")],
        [get_text("example_3")],
        [get_text("example_4")]
    ]

def update_video_sliders(video_path):
    """Update video time sliders based on actual video duration"""
    if not video_path:
        return gr.update(maximum=7200), gr.update(maximum=7200)

    duration = get_video_duration(video_path)
    # Round up to nearest 10 seconds for cleaner UI
    max_duration = ((int(duration) + 9) // 10) * 10

    return (
        gr.update(maximum=max_duration, value=0),
        gr.update(maximum=max_duration, value=max_duration)
    )

def create_interface():
    """Create Gradio interface with current language and beautiful styling"""
    with gr.Blocks(
        title="SuperCaption Qwen3-VL NSFW",
        theme=gr.themes.Soft(primary_hue="indigo", secondary_hue="purple", neutral_hue="slate"),
        css=CUSTOM_CSS
    ) as demo:
        # Beautiful gradient header with credits
        gr.HTML("""
        <div class="main-header">
            <h1>🎬 SuperCaption Qwen3-VL NSFW</h1>
            <p>Генератор описаний и тегов для фото и видео</p>
            <p style="font-size: 0.85rem; margin-top: 0.5rem; opacity: 0.9;">
                🔓 Используются <b>Abliterated</b> модели — работа с любым контентом без цензуры
            </p>
            <p style="font-size: 0.9rem; margin-top: 0.75rem; opacity: 0.85;">
                Портативная версия от <a href="https://t.me/nerual_dreming" target="_blank">Nerual Dreming</a> и
                <a href="https://t.me/ruweb24" target="_blank">Slait</a> |
                Телеграм канал: <a href="https://t.me/neuroport" target="_blank">👾 НЕЙРО-СОФТ</a>
            </p>
        </div>
        """)

        # Header that will be updated for language
        header_md = gr.Markdown(f"""
        {get_text("subtitle")}
        """)
        
        # Общие настройки - модель, квантизация и язык
        with gr.Row():
            model_dropdown = gr.Dropdown(
                choices=get_model_choices(),
                value="huihui-ai/Huihui-Qwen3-VL-2B-Instruct-abliterated",
                label=get_text("model_selection"),
                info=get_text("model_info"),
                scale=3
            )
            refresh_models_btn = gr.Button(
                "🔄",
                size="sm",
                scale=0,
                min_width=40
            )
            quantization_dropdown = gr.Dropdown(
                choices=[
                    ("4-bit (Рекомендуется)", "4-bit"),
                    ("8-bit (Лучше качество)", "8-bit"),
                    ("Без (Полная точность)", "none"),
                ],
                value="4-bit",
                label=get_text("quantization"),
                info=get_text("quantization_info"),
                scale=1
            )
            language_dropdown = gr.Dropdown(
                choices=[("English", "en"), ("Русский", "ru"), ("中文", "zh")],
                value=current_language,
                label=get_text("language"),
                info=get_text("language_info"),
                scale=1
            )

        # Расширенные параметры
        advanced_accordion = gr.Accordion(get_text("advanced_params"), open=False)
        with advanced_accordion:
            with gr.Row():
                max_tokens_slider = gr.Slider(
                    minimum=1,
                    maximum=4096,
                    value=1024,
                    step=1,
                    label=get_text("max_tokens"),
                    info=get_text("max_tokens_info")
                )
                temperature_slider = gr.Slider(
                    minimum=0.1,
                    maximum=2.0,
                    value=0.6,
                    step=0.1,
                    label=get_text("temperature"),
                    info=get_text("temperature_info")
                )
            
            with gr.Row():
                top_p_slider = gr.Slider(
                    minimum=0.05,
                    maximum=1.0,
                    value=0.9,
                    step=0.05,
                    label=get_text("top_p"),
                    info=get_text("top_p_info")
                )
                top_k_slider = gr.Slider(
                    minimum=1,
                    maximum=1000,
                    value=50,
                    step=1,
                    label=get_text("top_k"),
                    info=get_text("top_k_info")
                )
            
            with gr.Row():
                seed_number = gr.Number(
                    value=-1,
                    label=get_text("seed"),
                    info=get_text("seed_info"),
                    precision=0
                )
                random_seed_btn = gr.Button(get_text("random_seed_btn"), size="sm")
        
        # Вкладки для одиночной и пакетной обработки
        tabs = gr.Tabs()
        with tabs:
            # Вкладка одиночной обработки
            single_tab = gr.TabItem(get_text("single_processing"))
            with single_tab:
                with gr.Row():
                    with gr.Column(scale=1, elem_classes="card-style"):
                        gr.Markdown("### 📷 Входные данные")

                        # Track which media tab is active
                        active_media_type = gr.State("image")  # Default to image tab

                        # Tabs for Image and Video
                        media_tabs = gr.Tabs()
                        with media_tabs:
                            image_tab = gr.TabItem("🖼️ Изображение")
                            with image_tab:
                                single_image = gr.Image(
                                    type="numpy",
                                    label=get_text("upload_image"),
                                    height=300
                                )
                                # Duplicate Generate/Stop buttons for quick access
                                with gr.Row():
                                    single_generate_btn_image = gr.Button(
                                        get_text("generate_btn"),
                                        variant="primary",
                                        elem_classes="generate-btn",
                                        scale=3
                                    )
                                    single_stop_btn_image = gr.Button(
                                        get_text("stop_btn"),
                                        variant="stop",
                                        scale=1,
                                        interactive=False
                                    )

                            video_tab = gr.TabItem("🎥 Видео")
                            with video_tab:
                                single_video = gr.Video(
                                    label="Загрузите видео",
                                    height=300
                                )
                                # Video segment controls
                                with gr.Row():
                                    video_start_time = gr.Slider(
                                        label="⏱️ Начало (секунды)",
                                        value=0,
                                        minimum=0,
                                        maximum=7200,
                                        step=0.1,
                                        info="Начало сегмента видео (0 = с начала)"
                                    )
                                    video_end_time = gr.Slider(
                                        label="⏱️ Конец (секунды)",
                                        value=7200,
                                        minimum=0,
                                        maximum=7200,
                                        step=0.1,
                                        info="Конец сегмента видео (максимум = до конца)"
                                    )
                                # Duplicate Generate/Stop buttons for quick access
                                with gr.Row():
                                    single_generate_btn_video = gr.Button(
                                        get_text("generate_btn"),
                                        variant="primary",
                                        elem_classes="generate-btn",
                                        scale=3
                                    )
                                    single_stop_btn_video = gr.Button(
                                        get_text("stop_btn"),
                                        variant="stop",
                                        scale=1,
                                        interactive=False
                                    )

                            multi_image_tab = gr.TabItem("🖼️✕ Несколько изображений")
                            with multi_image_tab:
                                multi_images = gr.Gallery(
                                    label="Загрузите несколько изображений (до 20)",
                                    columns=4,
                                    rows=3,
                                    height=400,
                                    allow_preview=True,
                                    show_label=True,
                                    type="filepath",
                                    object_fit="contain",
                                    show_download_button=False
                                )
                                gr.Markdown("💡 **Подсказка:** Загрузите 2-20 изображений для сравнения, анализа временных рядов или контроля качества")
                                # Duplicate Generate/Stop buttons for quick access
                                with gr.Row():
                                    single_generate_btn_multi = gr.Button(
                                        get_text("generate_btn"),
                                        variant="primary",
                                        elem_classes="generate-btn",
                                        scale=3
                                    )
                                    single_stop_btn_multi = gr.Button(
                                        get_text("stop_btn"),
                                        variant="stop",
                                        scale=1,
                                        interactive=False
                                    )

                        gr.Markdown("### 📝 Настройки описания")

                        single_desc_type = gr.Dropdown(
                            choices=get_description_types(),
                            value=get_description_types()[0],
                            label=get_text("description_type"),
                            info=get_text("description_type_info")
                        )
                        single_desc_length = gr.Dropdown(
                            choices=get_description_lengths(),
                            value=get_description_lengths()[0],
                            label=get_text("description_length"),
                            info=get_text("description_length_info")
                        )
                        single_num_variants = gr.Slider(
                            minimum=1,
                            maximum=5,
                            value=1,
                            step=1,
                            label=get_text("num_variants"),
                            info=get_text("num_variants_info")
                        )

                        # Custom prompt - visible by default
                        single_custom_prompt = gr.Textbox(
                            label=get_text("custom_prompt_override"),
                            placeholder=get_text("custom_prompt_placeholder"),
                            lines=3
                        )

                        # Extra options (will update based on media type)
                        with gr.Accordion(get_text("extra_options"), open=False):
                            single_extra_options = gr.CheckboxGroup(
                                choices=get_extra_options(is_video=False),  # Default to image
                                value=[],
                                label="",
                                info=get_text("extra_options_info"),
                                elem_id="single_extra_options"
                            )
                            # Hidden state to track if video is selected
                            single_is_video = gr.State(value=False)

                            # Character name field - hidden in accordion
                            single_character_name = gr.Textbox(
                                label=get_text("character_name"),
                                placeholder=get_text("character_name_placeholder"),
                                info=get_text("character_name_info"),
                                lines=1
                            )

                        # Prompt presets section - collapsed accordion at the bottom
                        with gr.Accordion("📋 Пресеты промптов", open=False):
                            with gr.Row():
                                single_preset = gr.Dropdown(
                                    choices=list(load_prompt_presets().keys()),
                                    value="None",
                                    label=get_text("prompt_preset"),
                                    info=get_text("prompt_preset_info"),
                                    scale=3
                                )
                                single_refresh_presets = gr.Button(
                                    "🔄",
                                    size="sm",
                                    scale=0,
                                    min_width=40
                                )
                            gr.Markdown("---")
                            gr.Markdown("**Сохранить текущий промпт как пресет:**")
                            with gr.Row():
                                single_save_preset_name = gr.Textbox(
                                    label="Имя пресета",
                                    placeholder="мой_пресет",
                                    scale=2
                                )
                                single_save_preset_btn = gr.Button(
                                    "💾 Сохранить",
                                    size="sm",
                                    scale=1
                                )
                            single_save_preset_status = gr.Markdown("")

                        with gr.Row():
                            single_submit_btn = gr.Button(
                                get_text("generate_btn"),
                                variant="primary",
                                elem_classes="generate-btn",
                                scale=3
                            )
                            single_stop_btn = gr.Button(
                                get_text("stop_btn"),
                                variant="stop",
                                scale=1,
                                interactive=False
                            )

                    with gr.Column(scale=1, elem_classes="card-style"):
                        gr.Markdown("### 📊 Результаты")
                        single_status = gr.Textbox(
                            label=get_text("status"),
                            interactive=False,
                            elem_classes="status-box"
                        )

                        single_prompt_used = gr.Textbox(
                            label="Использованный промпт",
                            interactive=True,
                            lines=4,
                            info="Можно отредактировать и снова сгенерировать",
                            show_copy_button=True
                        )

                        # Button to copy prompt back to custom prompt
                        use_prompt_btn = gr.Button(
                            "⬆️ Использовать этот промпт",
                            size="sm",
                            variant="secondary"
                        )

                        # Thinking Process section (collapsible)
                        with gr.Accordion("💭 Мыслительный процесс", open=True, visible=True) as thinking_accordion:
                            thinking_output = gr.Textbox(
                                label="",
                                lines=6,
                                interactive=False,
                                show_copy_button=True,
                                placeholder="Здесь будет отображаться процесс рассуждения модели (для Thinking моделей)..."
                            )

                        # Image preview with bboxes
                        with gr.Accordion("🖼️ Превью изображения", open=True, visible=True) as image_preview_accordion:
                            image_preview = gr.Image(
                                label="",
                                type="filepath",
                                interactive=False,
                                show_label=False
                            )

                        # Multiple variant outputs
                        single_outputs = []
                        for i in range(5):
                            with gr.Group(visible=(i == 0)) as variant_group:
                                variant_output = gr.Textbox(
                                    label=f"{get_text('variant')} {i+1}" if i > 0 else get_text("result"),
                                    lines=8,
                                    show_copy_button=True
                                )
                                single_outputs.append((variant_group, variant_output))

                        # Download result file
                        single_download = gr.File(
                            label=get_text("download_result"),
                            visible=True
                        )

                        # Console output accordion
                        with gr.Accordion("📟 Консоль", open=False):
                            single_console_output = gr.Textbox(
                                label="",
                                lines=10,
                                max_lines=20,
                                interactive=False,
                                show_copy_button=True,
                                placeholder="Здесь будут отображаться системные сообщения и логи во время генерации..."
                            )

                # Кликабельные примеры промтов
                examples_title = gr.Markdown(f"### {get_text('examples_title')}")

            # Вкладка пакетной обработки - разделена на изображения и видео
            batch_tab = gr.TabItem(get_text("batch_processing"))
            with batch_tab:
                # Sub-tabs for images and videos
                batch_media_tabs = gr.Tabs()
                with batch_media_tabs:
                    # BATCH IMAGES TAB
                    batch_images_tab = gr.TabItem("📚 Пакет изображений")
                    with batch_images_tab:
                        with gr.Row():
                            with gr.Column(scale=1, elem_classes="card-style"):
                                gr.Markdown("### 📁 Входные файлы")
                                batch_images = gr.File(
                                    file_count="multiple",
                                    label=get_text("upload_images"),
                                    file_types=["image"]
                                )

                                gr.Markdown("### 📝 Настройки описания")

                                # Prompt preset dropdown
                                with gr.Row():
                                    batch_preset = gr.Dropdown(
                                        choices=list(load_prompt_presets().keys()),
                                        value="None",
                                        label=get_text("prompt_preset"),
                                        info=get_text("prompt_preset_info"),
                                        scale=4
                                    )
                                    batch_refresh_presets = gr.Button(
                                        "🔄",
                                        size="sm",
                                        scale=0,
                                        min_width=40
                                    )

                                batch_desc_type = gr.Dropdown(
                                    choices=get_description_types(),
                                    value=get_description_types()[0],
                                    label=get_text("description_type"),
                                    info=get_text("description_type_info")
                                )
                                batch_desc_length = gr.Dropdown(
                                    choices=get_description_lengths(),
                                    value=get_description_lengths()[0],
                                    label=get_text("description_length"),
                                    info=get_text("description_length_info")
                                )
                                batch_num_variants = gr.Slider(
                                    minimum=1,
                                    maximum=3,
                                    value=1,
                                    step=1,
                                    label=get_text("num_variants"),
                                    info=get_text("num_variants_info")
                                )

                                # Character name field
                                batch_character_name = gr.Textbox(
                                    label=get_text("character_name"),
                                    placeholder=get_text("character_name_placeholder"),
                                    info=get_text("character_name_info"),
                                    lines=1
                                )

                                # Extra options for images
                                with gr.Accordion(get_text("extra_options"), open=False):
                                    batch_extra_options = gr.CheckboxGroup(
                                        choices=get_extra_options(is_video=False),
                                        value=[],
                                        label="",
                                        info=get_text("extra_options_info")
                                    )

                                with gr.Accordion(get_text("custom_prompt_override"), open=False):
                                    batch_custom_prompt = gr.Textbox(
                                        placeholder=get_text("custom_prompt_placeholder"),
                                        lines=3,
                                        label=""
                                    )

                                gr.Markdown("### 💾 Настройки экспорта")
                                batch_output_folder = gr.Textbox(
                                    label=get_text("output_folder"),
                                    placeholder=get_text("output_folder_placeholder"),
                                    value=""
                                )
                                batch_export_formats = gr.CheckboxGroup(
                                    choices=["TXT", "JSON", "CSV"],
                                    value=["TXT"],
                                    label=get_text("export_format")
                                )

                                with gr.Row():
                                    batch_submit_btn = gr.Button(
                                        get_text("process_batch_btn"),
                                        variant="primary",
                                        elem_classes="generate-btn",
                                        scale=3
                                    )
                                    batch_stop_btn = gr.Button(
                                        get_text("stop_btn"),
                                        variant="stop",
                                        scale=1
                                    )

                            with gr.Column(scale=1, elem_classes="card-style"):
                                gr.Markdown("### 📊 Результаты")
                                batch_status = gr.Textbox(
                                    label=get_text("status"),
                                    interactive=False,
                                    elem_classes="status-box"
                                )
                                batch_output = gr.Textbox(
                                    label=get_text("results"),
                                    lines=20,
                                    show_copy_button=True
                                )

                                # Download result file
                                batch_download = gr.File(
                                    label=get_text("download_result"),
                                    visible=True
                                )

                                # Console output
                                batch_console_output = gr.Textbox(
                                    label="📟 Console Output",
                                    lines=10,
                                    interactive=False,
                                    show_copy_button=True
                                )

                    # BATCH VIDEOS TAB
                    batch_videos_tab = gr.TabItem("🎬 Пакет видео")
                    with batch_videos_tab:
                        with gr.Row():
                            with gr.Column(scale=1, elem_classes="card-style"):
                                gr.Markdown("### 📁 Входные видео файлы")
                                batch_videos = gr.File(
                                    file_count="multiple",
                                    label="Загрузите видео",
                                    file_types=["video"]
                                )

                                gr.Markdown("### 📝 Настройки описания")

                                # Prompt preset dropdown
                                with gr.Row():
                                    batch_video_preset = gr.Dropdown(
                                        choices=list(load_prompt_presets().keys()),
                                        value="None",
                                        label=get_text("prompt_preset"),
                                        info=get_text("prompt_preset_info"),
                                        scale=4
                                    )
                                    batch_video_refresh_presets = gr.Button(
                                        "🔄",
                                        size="sm",
                                        scale=0,
                                        min_width=40
                                    )

                                batch_video_desc_type = gr.Dropdown(
                                    choices=get_description_types(),
                                    value=get_description_types()[0],
                                    label=get_text("description_type"),
                                    info=get_text("description_type_info")
                                )
                                batch_video_desc_length = gr.Dropdown(
                                    choices=get_description_lengths(),
                                    value=get_description_lengths()[0],
                                    label=get_text("description_length"),
                                    info=get_text("description_length_info")
                                )
                                batch_video_num_variants = gr.Slider(
                                    minimum=1,
                                    maximum=3,
                                    value=1,
                                    step=1,
                                    label=get_text("num_variants"),
                                    info=get_text("num_variants_info")
                                )

                                # Character name field
                                batch_video_character_name = gr.Textbox(
                                    label=get_text("character_name"),
                                    placeholder=get_text("character_name_placeholder"),
                                    info=get_text("character_name_info"),
                                    lines=1
                                )

                                # Extra options for videos
                                with gr.Accordion(get_text("extra_options"), open=False):
                                    batch_video_extra_options = gr.CheckboxGroup(
                                        choices=get_extra_options(is_video=True),
                                        value=[],
                                        label="",
                                        info=get_text("extra_options_info")
                                    )

                                with gr.Accordion(get_text("custom_prompt_override"), open=False):
                                    batch_video_custom_prompt = gr.Textbox(
                                        placeholder=get_text("custom_prompt_placeholder"),
                                        lines=3,
                                        label=""
                                    )

                                gr.Markdown("### 💾 Настройки экспорта")
                                batch_video_output_folder = gr.Textbox(
                                    label=get_text("output_folder"),
                                    placeholder=get_text("output_folder_placeholder"),
                                    value=""
                                )
                                batch_video_export_formats = gr.CheckboxGroup(
                                    choices=["TXT", "JSON", "CSV"],
                                    value=["TXT"],
                                    label=get_text("export_format")
                                )

                                with gr.Row():
                                    batch_video_submit_btn = gr.Button(
                                        "🚀 Обработать видео",
                                        variant="primary",
                                        elem_classes="generate-btn",
                                        scale=3
                                    )
                                    batch_video_stop_btn = gr.Button(
                                        get_text("stop_btn"),
                                        variant="stop",
                                        scale=1
                                    )

                            with gr.Column(scale=1, elem_classes="card-style"):
                                gr.Markdown("### 📊 Результаты")
                                batch_video_status = gr.Textbox(
                                    label=get_text("status"),
                                    interactive=False,
                                    elem_classes="status-box"
                                )
                                batch_video_output = gr.Textbox(
                                    label=get_text("results"),
                                    lines=20,
                                    show_copy_button=True
                                )

                                # Download result file
                                batch_video_download = gr.File(
                                    label=get_text("download_result"),
                                    visible=True
                                )

                                # Console output
                                batch_video_console_output = gr.Textbox(
                                    label="📟 Console Output",
                                    lines=10,
                                    interactive=False,
                                    show_copy_button=True
                                )

        # Обработчики событий

        # Function to update extra options based on media type
        def update_extra_options_for_media(is_video):
            return gr.update(choices=get_extra_options(is_video=is_video), value=[])

        # Media tab selection handlers - update extra options when switching tabs
        def on_image_tab_select():
            return gr.update(choices=get_extra_options(is_video=False), value=[]), False, "image"

        def on_video_tab_select():
            return gr.update(choices=get_extra_options(is_video=True), value=[]), True, "video"

        def on_multi_image_tab_select():
            return gr.update(choices=get_extra_options(is_video=False), value=[]), False, "multi"

        # Connect media tab selection to extra options update and active media type
        image_tab.select(fn=on_image_tab_select, outputs=[single_extra_options, single_is_video, active_media_type])
        video_tab.select(fn=on_video_tab_select, outputs=[single_extra_options, single_is_video, active_media_type])
        multi_image_tab.select(fn=on_multi_image_tab_select, outputs=[single_extra_options, single_is_video, active_media_type])

        def change_language(lang):
            global current_language
            current_language = lang

            # Return updated text for key components
            return [
                f"{get_text('subtitle')}",  # header_md
                gr.update(label=get_text("model_selection"), info=get_text("model_info")),  # model_dropdown
                gr.update(label=get_text("quantization"), info=get_text("quantization_info")),  # quantization_dropdown
                gr.update(label=get_text("language"), info=get_text("language_info")),  # language_dropdown
                gr.update(label=get_text("advanced_params")),  # advanced_accordion
                gr.update(value=get_text("generate_btn")),  # single_submit_btn
                gr.update(value=get_text("process_batch_btn")),  # batch_submit_btn
                gr.update(choices=get_description_types(), value=get_description_types()[0]),  # single_desc_type
                gr.update(choices=get_description_lengths(), value=get_description_lengths()[0]),  # single_desc_length
                gr.update(choices=get_description_types(), value=get_description_types()[0]),  # batch_desc_type
                gr.update(choices=get_description_lengths(), value=get_description_lengths()[0]),  # batch_desc_length
                gr.update(choices=get_extra_options(is_video=False), value=[]),  # single_extra_options
                gr.update(choices=get_extra_options(is_video=False), value=[]),  # batch_extra_options
                gr.update(choices=get_extra_options(is_video=True), value=[]),  # batch_video_extra_options
                gr.update(value=get_text("stop_btn")),  # single_stop_btn
                gr.update(value=get_text("stop_btn")),  # batch_stop_btn
                gr.update(value=get_text("stop_btn")),  # batch_video_stop_btn
                gr.update(value=get_text("generate_btn")),  # single_generate_btn_image
                gr.update(value=get_text("stop_btn")),  # single_stop_btn_image
                gr.update(value=get_text("generate_btn")),  # single_generate_btn_video
                gr.update(value=get_text("stop_btn")),  # single_stop_btn_video
            ]

        language_dropdown.change(
            fn=change_language,
            inputs=language_dropdown,
            outputs=[
                header_md,
                model_dropdown,
                quantization_dropdown,
                language_dropdown,
                advanced_accordion,
                single_submit_btn,
                batch_submit_btn,
                single_desc_type,
                single_desc_length,
                batch_desc_type,
                batch_desc_length,
                single_extra_options,
                batch_extra_options,
                batch_video_extra_options,
                single_stop_btn,
                batch_stop_btn,
                batch_video_stop_btn,
                single_generate_btn_image,
                single_stop_btn_image,
                single_generate_btn_video,
                single_stop_btn_video,
            ]
        )

        # Stop button handlers
        single_stop_btn.click(
            fn=stop_generation,
            outputs=single_status
        )

        batch_stop_btn.click(
            fn=stop_generation,
            outputs=batch_status
        )

        batch_video_stop_btn.click(
            fn=stop_generation,
            outputs=batch_video_status
        )

        # Model list refresh handler
        def refresh_model_list():
            """Обновить список моделей с актуальным статусом загрузки"""
            return gr.update(choices=get_model_choices())

        refresh_models_btn.click(
            fn=refresh_model_list,
            outputs=model_dropdown
        )

        # Preset refresh handlers
        def refresh_presets():
            presets = load_prompt_presets()
            return gr.update(choices=list(presets.keys()), value="None")

        single_refresh_presets.click(
            fn=refresh_presets,
            outputs=single_preset
        )

        batch_refresh_presets.click(
            fn=refresh_presets,
            outputs=batch_preset
        )

        batch_video_refresh_presets.click(
            fn=refresh_presets,
            outputs=batch_video_preset
        )

        # Preset selection handlers (load preset text into custom prompt)
        def load_preset(preset_name):
            presets = load_prompt_presets()
            if preset_name and preset_name != "None":
                return presets.get(preset_name, "")
            return ""

        single_preset.change(
            fn=load_preset,
            inputs=single_preset,
            outputs=single_custom_prompt
        )

        batch_preset.change(
            fn=load_preset,
            inputs=batch_preset,
            outputs=batch_custom_prompt
        )

        batch_video_preset.change(
            fn=load_preset,
            inputs=batch_video_preset,
            outputs=batch_video_custom_prompt
        )

        random_seed_btn.click(
            fn=random_seed,
            outputs=seed_number
        )

        # Update variant visibility based on slider
        def update_variant_visibility(num_variants):
            updates = []
            for i in range(5):
                updates.append(gr.update(visible=(i < num_variants)))
            return updates

        single_num_variants.change(
            fn=update_variant_visibility,
            inputs=[single_num_variants],
            outputs=[group for group, _ in single_outputs]
        )

        # Single image processing with button lock
        def process_single_wrapper(image, video, video_start_time, video_end_time, desc_type, desc_length, custom_prompt,
                                   extra_options, character_name, num_variants,
                                   model_name, quantization, max_tokens, temperature, top_p, top_k, seed, active_media_type):
            # Force media type based on active tab
            if active_media_type == "image":
                video = None  # Ignore video input when image tab is active
            elif active_media_type == "video":
                image = None  # Ignore image input when video tab is active
            elif active_media_type == "multi":
                # Multi-image tab should not use this wrapper
                pass

            # Start capturing console output
            log_capture.clear_logs()
            log_capture.start_capture()

            # Disable all generate buttons, enable all stop buttons
            yield (
                gr.update(value=get_text("generating"), interactive=False),  # single_submit_btn
                gr.update(value=get_text("generating"), interactive=False),  # single_generate_btn_image
                gr.update(value=get_text("generating"), interactive=False),  # single_generate_btn_video
                gr.update(value=get_text("generating"), interactive=False),  # single_generate_btn_multi
                gr.update(interactive=True),   # single_stop_btn
                gr.update(interactive=True),   # single_stop_btn_image
                gr.update(interactive=True),   # single_stop_btn_video
                gr.update(interactive=True),   # single_stop_btn_multi
                "",  # single_status
                gr.update(visible=False),  # thinking_accordion
                "",  # thinking_output
                "",  # single_prompt_used
                gr.update(visible=False),  # image_preview_accordion
                None,  # image_preview
                *[gr.update(value="") for _ in range(5)],  # variant outputs
                None,  # single_download
                ""  # single_console_output
            )

            results = []
            download_path = None

            # Process and yield results
            for status, prompt_used, results, download_path, console_logs, reasoning, image_preview in process_single_image(
                image, video, video_start_time, video_end_time, desc_type, desc_length, custom_prompt,
                extra_options, character_name, num_variants,
                model_name, quantization, max_tokens, temperature, top_p, top_k, seed
            ):
                # Prepare outputs for each variant box
                variant_outputs = []
                for i in range(5):
                    if i < len(results):
                        variant_outputs.append(gr.update(value=results[i]))
                    else:
                        variant_outputs.append(gr.update(value=""))

                # Add model info above status
                cached_size = get_model_cache_size(model_name)
                size_str = f" [{cached_size}]" if cached_size else ""
                status_with_model = f"✅ **{model_name}**{size_str} | {quantization}\n\n{status}"

                # Update thinking and image preview visibility
                thinking_visible = bool(reasoning)
                image_preview_visible = bool(image_preview)

                # Keep buttons disabled during generation
                yield (
                    gr.update(value=get_text("generating"), interactive=False),  # single_submit_btn
                    gr.update(value=get_text("generating"), interactive=False),  # single_generate_btn_image
                    gr.update(value=get_text("generating"), interactive=False),  # single_generate_btn_video
                    gr.update(value=get_text("generating"), interactive=False),  # single_generate_btn_multi
                    gr.update(interactive=True),   # single_stop_btn
                    gr.update(interactive=True),   # single_stop_btn_image
                    gr.update(interactive=True),   # single_stop_btn_video
                    gr.update(interactive=True),   # single_stop_btn_multi
                    status_with_model,  # single_status
                    gr.update(visible=thinking_visible),  # thinking_accordion
                    reasoning if reasoning else "",  # thinking_output
                    prompt_used,  # single_prompt_used
                    gr.update(visible=image_preview_visible),  # image_preview_accordion
                    image_preview,  # image_preview
                    *variant_outputs,  # variant outputs
                    download_path,  # single_download
                    console_logs  # single_console_output
                )

            # Stop capturing and get final logs
            log_capture.stop_capture()
            final_logs = log_capture.get_logs()

            # Re-enable button at end
            final_outputs = []
            for i in range(5):
                if i < len(results):
                    final_outputs.append(gr.update(value=results[i]))
                else:
                    final_outputs.append(gr.update(value=""))

            # Add model info above final status
            cached_size = get_model_cache_size(model_name)
            size_str = f" [{cached_size}]" if cached_size else ""
            final_status_with_model = f"✅ **{model_name}**{size_str} | {quantization}\n\n{status}"

            # Update final thinking and image preview visibility
            final_thinking_visible = bool(reasoning)
            final_image_preview_visible = bool(image_preview)

            # Re-enable all generate buttons, disable all stop buttons
            yield (
                gr.update(value=get_text("generate_btn"), interactive=True),  # single_submit_btn
                gr.update(value=get_text("generate_btn"), interactive=True),  # single_generate_btn_image
                gr.update(value=get_text("generate_btn"), interactive=True),  # single_generate_btn_video
                gr.update(value=get_text("generate_btn"), interactive=True),  # single_generate_btn_multi
                gr.update(interactive=False),  # single_stop_btn
                gr.update(interactive=False),  # single_stop_btn_image
                gr.update(interactive=False),  # single_stop_btn_video
                gr.update(interactive=False),  # single_stop_btn_multi
                final_status_with_model,  # single_status
                gr.update(visible=final_thinking_visible),  # thinking_accordion
                reasoning if reasoning else "",  # thinking_output
                prompt_used,  # single_prompt_used
                gr.update(visible=final_image_preview_visible),  # image_preview_accordion
                image_preview,  # image_preview
                *final_outputs,  # variant outputs
                download_path,  # single_download
                final_logs  # single_console_output
            )

        # Image-only wrapper - ignores video input
        def process_image_only_wrapper(image, desc_type, desc_length, custom_prompt,
                                       extra_options, character_name, num_variants,
                                       model_name, quantization, max_tokens, temperature, top_p, top_k, seed):
            # Force video to None and use default video timestamps
            yield from process_single_wrapper(
                image, None, 0, 7200,  # video=None, default timestamps
                desc_type, desc_length, custom_prompt,
                extra_options, character_name, num_variants,
                model_name, quantization, max_tokens, temperature, top_p, top_k, seed,
                "image"  # active_media_type
            )

        # Video-only wrapper - ignores image input
        def process_video_only_wrapper(video, video_start_time, video_end_time, desc_type, desc_length, custom_prompt,
                                       extra_options, character_name, num_variants,
                                       model_name, quantization, max_tokens, temperature, top_p, top_k, seed):
            # Force image to None
            yield from process_single_wrapper(
                None, video, video_start_time, video_end_time,  # image=None
                desc_type, desc_length, custom_prompt,
                extra_options, character_name, num_variants,
                model_name, quantization, max_tokens, temperature, top_p, top_k, seed,
                "video"  # active_media_type
            )

        # Handler for using the prompt from single_prompt_used
        use_prompt_btn.click(
            fn=lambda x: x,  # Just pass through the value
            inputs=[single_prompt_used],
            outputs=[single_custom_prompt]
        )

        single_submit_btn.click(
            fn=process_single_wrapper,
            inputs=[
                single_image,
                single_video,
                video_start_time,
                video_end_time,
                single_desc_type,
                single_desc_length,
                single_custom_prompt,
                single_extra_options,
                single_character_name,
                single_num_variants,
                model_dropdown,
                quantization_dropdown,
                max_tokens_slider,
                temperature_slider,
                top_p_slider,
                top_k_slider,
                seed_number,
                active_media_type
            ],
            outputs=[single_submit_btn, single_generate_btn_image, single_generate_btn_video, single_generate_btn_multi, single_stop_btn, single_stop_btn_image, single_stop_btn_video, single_stop_btn_multi, single_status, thinking_accordion, thinking_output, single_prompt_used, image_preview_accordion, image_preview] + [output for _, output in single_outputs] + [single_download, single_console_output]
        )

        # Duplicate Generate buttons in Image/Video tabs - use specific wrappers
        single_generate_btn_image.click(
            fn=process_image_only_wrapper,
            inputs=[
                single_image,
                # video inputs removed - image tab only uses image
                single_desc_type,
                single_desc_length,
                single_custom_prompt,
                single_extra_options,
                single_character_name,
                single_num_variants,
                model_dropdown,
                quantization_dropdown,
                max_tokens_slider,
                temperature_slider,
                top_p_slider,
                top_k_slider,
                seed_number
            ],
            outputs=[single_submit_btn, single_generate_btn_image, single_generate_btn_video, single_generate_btn_multi, single_stop_btn, single_stop_btn_image, single_stop_btn_video, single_stop_btn_multi, single_status, thinking_accordion, thinking_output, single_prompt_used, image_preview_accordion, image_preview] + [output for _, output in single_outputs] + [single_download, single_console_output]
        )

        single_generate_btn_video.click(
            fn=process_video_only_wrapper,
            inputs=[
                single_video,
                video_start_time,
                video_end_time,
                # image input removed - video tab only uses video
                single_desc_type,
                single_desc_length,
                single_custom_prompt,
                single_extra_options,
                single_character_name,
                single_num_variants,
                model_dropdown,
                quantization_dropdown,
                max_tokens_slider,
                temperature_slider,
                top_p_slider,
                top_k_slider,
                seed_number
            ],
            outputs=[single_submit_btn, single_generate_btn_image, single_generate_btn_video, single_generate_btn_multi, single_stop_btn, single_stop_btn_image, single_stop_btn_video, single_stop_btn_multi, single_status, thinking_accordion, thinking_output, single_prompt_used, image_preview_accordion, image_preview] + [output for _, output in single_outputs] + [single_download, single_console_output]
        )

        # Duplicate Stop buttons in Image/Video tabs
        single_stop_btn_image.click(
            fn=stop_generation,
            outputs=single_status
        )

        single_stop_btn_video.click(
            fn=stop_generation,
            outputs=single_status
        )

        # Multi-image processing wrapper
        def process_multi_image_wrapper(multi_images_list, desc_type, desc_length, custom_prompt,
                                       extra_options, character_name, num_variants,
                                       model_name, quantization, max_tokens, temperature, top_p, top_k, seed):
            # Start capturing console output
            log_capture.clear_logs()
            log_capture.start_capture()

            # Disable all generate buttons, enable all stop buttons
            yield (
                gr.update(value=get_text("generating"), interactive=False),  # single_submit_btn
                gr.update(value=get_text("generating"), interactive=False),  # single_generate_btn_image
                gr.update(value=get_text("generating"), interactive=False),  # single_generate_btn_video
                gr.update(value=get_text("generating"), interactive=False),  # single_generate_btn_multi
                gr.update(interactive=True),   # single_stop_btn
                gr.update(interactive=True),   # single_stop_btn_image
                gr.update(interactive=True),   # single_stop_btn_video
                gr.update(interactive=True),   # single_stop_btn_multi
                "",  # single_status
                gr.update(visible=False),  # thinking_accordion
                "",  # thinking_output
                "",  # single_prompt_used
                gr.update(visible=False),  # image_preview_accordion
                None,  # image_preview
                *[gr.update(value="") for _ in range(5)],  # variant outputs
                None,  # single_download
                ""  # single_console_output
            )

            results = []
            download_path = None

            # Process and yield results
            for status, prompt_used, results, download_path, console_logs in process_multi_image(
                multi_images_list, desc_type, desc_length, custom_prompt,
                extra_options, character_name, num_variants,
                model_name, quantization, max_tokens, temperature, top_p, top_k, seed
            ):
                # Prepare outputs for each variant box
                variant_outputs = []
                for i in range(5):
                    if i < len(results):
                        variant_outputs.append(gr.update(value=results[i]))
                    else:
                        variant_outputs.append(gr.update(value=""))

                # Add model info above status
                cached_size = get_model_cache_size(model_name)
                size_str = f" [{cached_size}]" if cached_size else ""
                status_with_model = f"✅ **{model_name}**{size_str} | {quantization}\n\n{status}"

                # Keep buttons disabled during generation
                yield (
                    gr.update(value=get_text("generating"), interactive=False),
                    gr.update(value=get_text("generating"), interactive=False),
                    gr.update(value=get_text("generating"), interactive=False),
                    gr.update(value=get_text("generating"), interactive=False),
                    gr.update(interactive=True),
                    gr.update(interactive=True),
                    gr.update(interactive=True),
                    gr.update(interactive=True),
                    status_with_model,  # single_status
                    gr.update(visible=False),  # thinking_accordion
                    "",  # thinking_output
                    prompt_used,  # single_prompt_used
                    gr.update(visible=False),  # image_preview_accordion
                    None,  # image_preview
                    *variant_outputs,  # variant outputs
                    download_path,  # single_download
                    console_logs  # single_console_output
                )

            # Stop capturing and get final logs
            log_capture.stop_capture()
            final_logs = log_capture.get_logs()

            # Re-enable button at end
            final_outputs = []
            for i in range(5):
                if i < len(results):
                    final_outputs.append(gr.update(value=results[i]))
                else:
                    final_outputs.append(gr.update(value=""))

            # Add model info above final status
            cached_size = get_model_cache_size(model_name)
            size_str = f" [{cached_size}]" if cached_size else ""
            final_status_with_model = f"✅ **{model_name}**{size_str} | {quantization}\n\n{status}"

            # Re-enable all generate buttons, disable all stop buttons
            yield (
                gr.update(value=get_text("generate_btn"), interactive=True),
                gr.update(value=get_text("generate_btn"), interactive=True),
                gr.update(value=get_text("generate_btn"), interactive=True),
                gr.update(value=get_text("generate_btn"), interactive=True),
                gr.update(interactive=False),
                gr.update(interactive=False),
                gr.update(interactive=False),
                gr.update(interactive=False),
                final_status_with_model,  # single_status
                gr.update(visible=False),  # thinking_accordion
                "",  # thinking_output
                prompt_used,  # single_prompt_used
                gr.update(visible=False),  # image_preview_accordion
                None,  # image_preview
                *final_outputs,  # variant outputs
                download_path,  # single_download
                final_logs  # single_console_output
            )

        # Wire up multi-image generate button
        single_generate_btn_multi.click(
            fn=process_multi_image_wrapper,
            inputs=[
                multi_images,
                single_desc_type,
                single_desc_length,
                single_custom_prompt,
                single_extra_options,
                single_character_name,
                single_num_variants,
                model_dropdown,
                quantization_dropdown,
                max_tokens_slider,
                temperature_slider,
                top_p_slider,
                top_k_slider,
                seed_number
            ],
            outputs=[single_submit_btn, single_generate_btn_image, single_generate_btn_video, single_generate_btn_multi,
                     single_stop_btn, single_stop_btn_image, single_stop_btn_video, single_stop_btn_multi,
                     single_status, thinking_accordion, thinking_output, single_prompt_used, image_preview_accordion, image_preview] + [output for _, output in single_outputs] + [single_download, single_console_output]
        )

        # Wire up multi-image stop button
        single_stop_btn_multi.click(
            fn=stop_generation,
            outputs=single_status
        )

        # Update video sliders when video is loaded
        single_video.change(
            fn=update_video_sliders,
            inputs=[single_video],
            outputs=[video_start_time, video_end_time]
        )

        # Save preset handler
        def save_preset(name, prompt):
            if not name:
                return "❌ Please enter a preset name"
            msg = save_prompt_preset(name, prompt)
            new_presets = list(load_prompt_presets().keys())
            return msg, gr.update(choices=new_presets, value=name if "successfully" in msg else None)

        single_save_preset_btn.click(
            fn=save_preset,
            inputs=[single_save_preset_name, single_custom_prompt],
            outputs=[single_save_preset_status, single_preset]
        )

        # Batch processing with button lock
        def process_batch_wrapper(files, desc_type, desc_length, custom_prompt,
                                  extra_options, character_name, num_variants,
                                  output_folder, export_formats, model_name, quantization,
                                  max_tokens, temperature, top_p, top_k, seed):
            # Start capturing console output
            log_capture.clear_logs()
            log_capture.start_capture()

            # Disable button at start
            yield gr.update(value=get_text("generating"), interactive=False), "", "", None, ""

            download_path = None

            # Process and yield results
            for status, output_text, download_path in process_batch_images(
                files, desc_type, desc_length, custom_prompt,
                extra_options, character_name, num_variants,
                output_folder, export_formats, model_name, quantization,
                max_tokens, temperature, top_p, top_k, seed
            ):
                yield gr.update(value=get_text("generating"), interactive=False), status, output_text, download_path, log_capture.get_logs()

            # Stop capturing and get final logs
            log_capture.stop_capture()
            final_logs = log_capture.get_logs()

            # Re-enable button at end
            yield gr.update(value=get_text("process_batch_btn"), interactive=True), status, output_text, download_path, final_logs

        batch_submit_btn.click(
            fn=process_batch_wrapper,
            inputs=[
                batch_images,
                batch_desc_type,
                batch_desc_length,
                batch_custom_prompt,
                batch_extra_options,
                batch_character_name,
                batch_num_variants,
                batch_output_folder,
                batch_export_formats,
                model_dropdown,
                quantization_dropdown,
                max_tokens_slider,
                temperature_slider,
                top_p_slider,
                top_k_slider,
                seed_number
            ],
            outputs=[batch_submit_btn, batch_status, batch_output, batch_download, batch_console_output]
        )

        # Batch video processing with is_video=True
        def process_batch_video_wrapper(files, desc_type, desc_length, custom_prompt,
                                        extra_options, character_name, num_variants,
                                        output_folder, export_formats, model_name, quantization,
                                        max_tokens, temperature, top_p, top_k, seed):
            # Start capturing console output
            log_capture.clear_logs()
            log_capture.start_capture()

            # Disable button at start
            yield gr.update(value="⏳ Processing...", interactive=False), "", "", None, ""

            download_path = None

            # Process and yield results with is_video=True
            for status, output_text, download_path in process_batch_images(
                files, desc_type, desc_length, custom_prompt,
                extra_options, character_name, num_variants,
                output_folder, export_formats, model_name, quantization,
                max_tokens, temperature, top_p, top_k, seed,
                is_video=True  # KEY DIFFERENCE: process as videos
            ):
                yield gr.update(value="⏳ Processing...", interactive=False), status, output_text, download_path, log_capture.get_logs()

            # Stop capturing and get final logs
            log_capture.stop_capture()
            final_logs = log_capture.get_logs()

            # Re-enable button at end
            yield gr.update(value="🚀 Обработать видео", interactive=True), status, output_text, download_path, final_logs

        batch_video_submit_btn.click(
            fn=process_batch_video_wrapper,
            inputs=[
                batch_videos,
                batch_video_desc_type,
                batch_video_desc_length,
                batch_video_custom_prompt,
                batch_video_extra_options,
                batch_video_character_name,
                batch_video_num_variants,
                batch_video_output_folder,
                batch_video_export_formats,
                model_dropdown,
                quantization_dropdown,
                max_tokens_slider,
                temperature_slider,
                top_p_slider,
                top_k_slider,
                seed_number
            ],
            outputs=[batch_video_submit_btn, batch_video_status, batch_video_output, batch_video_download, batch_video_console_output]
        )

        return demo

# Создаем интерфейс Gradio
demo = create_interface()

if __name__ == "__main__":
    # Enable queue for progress bar support
    demo.queue(max_size=20, default_concurrency_limit=1).launch(
        server_name="127.0.0.1",
        server_port=None,
        share=False,
        show_error=True,
        inbrowser=True
    )
