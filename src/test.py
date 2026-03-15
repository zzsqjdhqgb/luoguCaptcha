# Copyright (C) 2025 Langning Chen
# Copyright (C) 2026 zzsqjdhqgb
#
# This file is part of luoguCaptcha.
#
# luoguCaptcha is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# luoguCaptcha is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with luoguCaptcha.  If not, see <https://www.gnu.org/licenses/>.

"""
Interactive captcha tester that pulls images from Luogu API and shows
the model's top predictions for manual inspection.

Usage:
  python scripts/test.py [--top 10] [--interval 0.0] [--backend auto|jax|torch|tensorflow]

Controls while running:
  Enter: fetch next captcha
  q + Enter: quit
  s + Enter: save current captcha image to ./captchas/<timestamp>.png
"""

# ---- 自动检测并设置 Keras 后端 ----
import os
import sys


def detect_and_set_backend(preferred: str = "auto") -> str:
    """
    检测可用的深度学习框架并设置 Keras 后端。

    torch 和 tensorflow 后端目前都无法正常工作！！
    
    优先级: jax > torch > tensorflow
    
    Args:
        preferred: 'auto', 'jax', 'torch', 或 'tensorflow'
    
    Returns:
        实际使用的后端名称
    """
    if preferred != "auto":
        # 用户指定了后端，直接尝试使用
        os.environ["KERAS_BACKEND"] = preferred
        return preferred
    
    # 自动检测可用后端
    backends_to_try = []
    
    # 检测 JAX
    try:
        import jax
        backends_to_try.append(("jax", jax.__version__))
    except ImportError:
        pass
    
    # 检测 PyTorch
    try:
        import torch # type: ignore
        backends_to_try.append(("torch", torch.__version__))
    except ImportError:
        pass
    
    # 检测 TensorFlow
    try:
        import tensorflow as tf # type: ignore
        backends_to_try.append(("tensorflow", tf.__version__))
    except ImportError:
        pass
    
    if not backends_to_try:
        print("Error: No deep learning backend found!")
        print("Please install one of: jax, torch, or tensorflow")
        print("  pip install jax jaxlib")
        print("  pip install torch")
        print("  pip install tensorflow")
        sys.exit(1)
    
    # 使用第一个可用的后端（按优先级排序）
    backend, version = backends_to_try[0]
    os.environ["KERAS_BACKEND"] = backend
    
    print(f"Detected backends: {[(b, v) for b, v in backends_to_try]}")
    print(f"Selected backend: {backend} (version {version})")
    
    return backend


def get_device_info(backend: str) -> str:
    """获取当前后端的设备信息。"""
    if backend == "jax":
        import jax
        devices = jax.devices()
        gpu_devices = [d for d in devices if d.platform == "gpu"]
        if gpu_devices:
            return f"JAX GPU: {gpu_devices}"
        return f"JAX CPU: {devices}"
    
    elif backend == "torch":
        import torch
        if torch.cuda.is_available():
            return f"PyTorch CUDA: {torch.cuda.get_device_name(0)}"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "PyTorch MPS (Apple Silicon)"
        return "PyTorch CPU"
    
    elif backend == "tensorflow":
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            return f"TensorFlow GPU: {gpus}"
        return "TensorFlow CPU"
    
    return f"Unknown backend: {backend}"


# 解析命令行参数以获取后端选择（需要在 import keras 之前）
def parse_backend_arg() -> str:
    """从命令行参数中提取后端选择。"""
    for i, arg in enumerate(sys.argv):
        if arg == "--backend" and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
        if arg.startswith("--backend="):
            return arg.split("=", 1)[1]
    return "auto"


# 在导入 keras 之前设置后端
_selected_backend = detect_and_set_backend(parse_backend_arg())

# ---- matplotlib 后端配置（必须在 import pyplot 之前）----
import matplotlib

# WSL 兼容的后端优先级列表
_BACKENDS = ["TkAgg", "Qt5Agg", "Qt6Agg", "GTK3Agg", "GTK4Agg", "WXAgg"]
_backend_set = False

for _backend in _BACKENDS:
    try:
        matplotlib.use(_backend)
        _backend_set = True
        print(f"Using matplotlib backend: {_backend}")
        break
    except Exception:
        continue

if not _backend_set:
    matplotlib.use("Agg")
    print("Warning: No interactive backend available, using Agg")

import argparse
import io
import time
from datetime import datetime
from itertools import product
from typing import List, Tuple

import keras
import numpy as np
import requests
from matplotlib import pyplot as plt
from PIL import Image

# 导入自定义层（触发注册）
import model as custom_model_module  # noqa: F401
from config import (
    IMG_HEIGHT,
    IMG_WIDTH,
    CHARS_PER_LABEL,
    CHAR_SIZE,
    MODEL_PATH,
    LUOGU_CAPTCHA_URL,
)

# 检测是否有交互式后端
INTERACTIVE_MODE = matplotlib.get_backend().lower() != "agg"


def setup_device():
    """打印 Keras 后端与设备信息。"""
    print(f"Keras backend: {keras.backend.backend()}")
    print(f"Device: {get_device_info(_selected_backend)}")


def load_model_or_exit(path: str):
    try:
        m = keras.models.load_model(path)
        print("Model loaded successfully")
        return m
    except Exception as e:
        print(f"Error loading model: {e}")
        print(f"Please ensure the model exists at {path}.")
        sys.exit(1)


def fetch_captcha(timeout: float = 10.0) -> Image.Image:
    """Fetch captcha image from Luogu API (returns PIL Image)."""
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Referer": "https://www.luogu.com.cn/lg4/captcha",
        "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
    }
    r = requests.get(LUOGU_CAPTCHA_URL, headers=headers, timeout=timeout)
    r.raise_for_status()
    img = Image.open(io.BytesIO(r.content)).convert("L")
    return img


def preprocess(img: Image.Image) -> np.ndarray:
    """Convert PIL image to model input shape (1, H, W, 1) normalized to [0,1]."""
    if img.size != (IMG_WIDTH, IMG_HEIGHT):
        img = img.resize((IMG_WIDTH, IMG_HEIGHT))
    x = np.array(img, dtype=np.float32) / 255.0
    x = np.expand_dims(x, axis=-1)
    x = np.expand_dims(x, axis=0)
    return x


def topk_per_position(prob: np.ndarray, k: int) -> List[List[Tuple[int, float]]]:
    """Return top-k (index, prob) per of the 4 positions."""
    tops: List[List[Tuple[int, float]]] = []
    for i in range(CHARS_PER_LABEL):
        p = prob[i]
        idxs = np.argsort(p)[-k:][::-1]
        tops.append([(int(idx), float(p[idx])) for idx in idxs])
    return tops


def combine_topk(
    tops: List[List[Tuple[int, float]]], max_results: int
) -> List[Tuple[str, float]]:
    """Combine per-position top-k into global top strings sorted by product prob."""
    candidates: List[Tuple[str, float]] = []
    for combo in product(*tops):
        chars = [chr(i) for i, _ in combo]
        probs = [max(1e-12, p) for _, p in combo]
        logp = float(np.sum(np.log(probs)))
        candidates.append(("".join(chars), logp))
    candidates.sort(key=lambda x: x[1], reverse=True)
    results = [(s, float(np.exp(lp))) for s, lp in candidates[:max_results]]
    return results


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def display_in_terminal(img: Image.Image, width: int = 60):
    """在终端中用 ASCII 字符显示图像。"""
    aspect_ratio = img.height / img.width
    new_height = int(width * aspect_ratio * 0.5)
    img_resized = img.resize((width, new_height))
    
    chars = " .:-=+*#%@"
    pixels = np.array(img_resized)
    
    lines = []
    for row in pixels:
        line = ""
        for pixel in row:
            char_idx = int(pixel / 255 * (len(chars) - 1))
            line += chars[char_idx]
        lines.append(line)
    
    print("\n" + "\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Luogu captcha manual tester")
    parser.add_argument(
        "--top", type=int, default=10, help="Top-N combined predictions to show"
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Top-k per position to consider when combining",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=0.0,
        help="Auto-advance interval seconds; 0 to wait for input",
    )
    parser.add_argument(
        "--save-dir",
        default="captchas",
        help="Directory to save images when pressing 's'",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Disable image display (terminal ASCII preview only)",
    )
    parser.add_argument(
        "--ascii",
        action="store_true",
        help="Use ASCII art display in terminal instead of GUI",
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "jax", "torch", "tensorflow"],
        default="auto",
        help="Keras backend to use (auto-detected by default)",
    )
    args = parser.parse_args()

    setup_device()
    captcha_model = load_model_or_exit(MODEL_PATH)

    use_gui = INTERACTIVE_MODE and not args.no_display and not args.ascii
    
    if use_gui:
        plt.ion()
        fig, ax = plt.subplots(figsize=(4, 2))
        img_artist = None
        ax.axis("off")
        try:
            fig.canvas.manager.set_window_title("Luogu Captcha Tester")
        except Exception:
            pass
    else:
        fig = ax = img_artist = None
        if not args.ascii:
            print("Running in non-interactive mode (ASCII preview enabled)")

    ensure_dir(args.save_dir)

    idx = 0
    try:
        while True:
            idx += 1
            try:
                pil_img = fetch_captcha()
            except Exception as e:
                print(f"[#{idx}] Fetch failed: {e}")
                if args.interval > 0:
                    time.sleep(min(args.interval, 2.0))
                else:
                    input("Press Enter to retry...")
                continue

            if use_gui:
                disp_img = np.array(pil_img)
                if img_artist is None:
                    img_artist = ax.imshow(disp_img, cmap="gray")
                else:
                    img_artist.set_data(disp_img)
                ax.set_title(f"Captcha #{idx}")
                fig.canvas.draw_idle()
                plt.pause(0.001)
            else:
                print(f"\n{'='*60}")
                print(f"Captcha #{idx}")
                display_in_terminal(pil_img)

            x = preprocess(pil_img)
            probs = captcha_model.predict(x, verbose=0)
            probs = np.squeeze(probs, axis=0)

            per_pos = topk_per_position(probs, args.k)
            combined = combine_topk(per_pos, args.top)

            print(f"\n[#{idx}] Top-{args.top} predictions (probability):")
            for rank, (s, p) in enumerate(combined, 1):
                print(f"  {rank:2d}. {s}  (p={p:.8f})")

            if args.interval > 0:
                time.sleep(args.interval)
                continue

            cmd = input("[Enter=next | s=save | q=quit] > ").strip().lower()
            if cmd == "q":
                break
            if cmd == "s":
                ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                path = os.path.join(args.save_dir, f"captcha_{ts}.png")
                pil_img.save(path)
                print(f"Saved to {path}")

    except KeyboardInterrupt:
        print("\nInterrupted. Bye.")
    finally:
        if use_gui:
            plt.ioff()
            try:
                plt.close(fig)
            except Exception:
                pass


if __name__ == "__main__":
    main()