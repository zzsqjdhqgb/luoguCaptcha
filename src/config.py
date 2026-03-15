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
全局配置：路径、图像参数、标签参数等。
训练超参数（如 d_model、num_heads 等）保留在 train.py 和模型定义中，
因为它们与模型架构紧密耦合，不属于"全局"配置。
"""

import os

# ── 图像 & 标签 ──────────────────────────────────────────────────
CHAR_SIZE = 256
CHARS_PER_LABEL = 4
IMG_HEIGHT = 35
IMG_WIDTH = 90

# ── 数据路径 ─────────────────────────────────────────────────────
DATA_DIR = os.path.join("data", "luogu_captcha_dataset")
NUMPY_DIR = os.path.join("data", "luogu_captcha_numpy")
TFRECORD_DIR = os.path.join("data", "luogu_captcha_tfrecord")
SAMPLES_PER_TFRECORD = 5000

# ── 模型路径 ─────────────────────────────────────────────────────
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "luoguCaptcha.keras")
VIT_MODEL_PATH = os.path.join(MODEL_DIR, "luoguCaptcha.ViT-EncoderOnly.keras")

# ── Hugging Face ─────────────────────────────────────────────────
DATASET_REPO_ID = "langningchen/luogu-captcha-dataset"
MODEL_REPO_ID = "langningchen/luogu-captcha-model"

# ── 外部 API ─────────────────────────────────────────────────────
LUOGU_CAPTCHA_URL = "https://www.luogu.com.cn/api/verify/captcha"