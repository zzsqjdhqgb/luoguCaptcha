# Copyright (C) 2025 zzsqjdhqgb
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
Vision Transformer 验证码识别训练脚本。
Keras 3 + PyTorch 后端，数据从 NumPy .npz 加载。

Usage:
  python train.py [--data-dir DIR] [--epochs N] [--batch-size N]
"""

import os
import argparse

# ── 设置后端（必须在 import keras 之前） ──────────────────────────
os.environ["KERAS_BACKEND"] = "torch"

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

import keras

from config import (
    CHAR_SIZE,
    CHARS_PER_LABEL,
    IMG_HEIGHT,
    IMG_WIDTH,
    NUMPY_DIR,
    MODEL_DIR,
    VIT_MODEL_PATH,
)
from model import build_vit_captcha_model, WarmupCosineDecay

# ── 训练超参数 ────────────────────────────────────────────────────
RANDOM_SEED = 484858
PATCH_SIZE = 5
D_MODEL = 128
NUM_HEADS = 4
NUM_LAYERS = 4
DFF = 256
DROPOUT_RATE = 0.1
EPOCHS = 150
BATCH_SIZE = 256

# 计算 patch 数量
NUM_PATCHES_H = IMG_HEIGHT // PATCH_SIZE  # 35 / 5 = 7
NUM_PATCHES_W = IMG_WIDTH // PATCH_SIZE   # 90 / 5 = 18
NUM_PATCHES = NUM_PATCHES_H * NUM_PATCHES_W  # 7 * 18 = 126


# ── 固定随机种子 ──────────────────────────────────────────────────
def set_all_seeds(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    keras.utils.set_random_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


set_all_seeds(RANDOM_SEED)

# ── 设备信息 ──────────────────────────────────────────────────────
if torch.cuda.is_available():
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    print(f"  CUDA version: {torch.version.cuda}")
    print(f"  Device count: {torch.cuda.device_count()}")
else:
    print("Using CPU")

print(f"Keras version : {keras.__version__}")
print(f"Backend       : {keras.backend.backend()}")
print(f"PyTorch       : {torch.__version__}")
print(f"Random seed   : {RANDOM_SEED}")


# ╔══════════════════════════════════════════════════════════════╗
# ║  PyTorch Dataset + DataLoader                               ║
# ╚══════════════════════════════════════════════════════════════╝
class CaptchaDataset(Dataset):
    """
    从 .npz 文件加载验证码数据。
    images: uint8 (N, H, W, 1) → float32 [0, 1]
    labels: int32 (N, 4)
    """

    def __init__(self, npz_path: str):
        data = np.load(npz_path)
        self.images = data["images"].astype(np.float32) / 255.0  # (N, H, W, 1)
        self.labels = data["labels"].astype(np.int64)  # (N, 4)
        print(
            f"  Loaded {len(self.images)} samples from {npz_path} "
            f"(images: {self.images.shape}, labels: {self.labels.shape})"
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]  # (H, W, 1), float32
        label = self.labels[idx]  # (4,), int64
        return image, label


def numpy_collate(batch):
    """
    自定义 collate 函数：保持 NumPy 数组格式。
    Keras 3 的 model.fit() 接受 NumPy 数组。
    """
    images, labels = zip(*batch)
    images = np.stack(images, axis=0)
    labels = np.stack(labels, axis=0)
    return images, labels


def create_data_loaders(data_dir: str, batch_size: int):
    """创建训练和验证的 DataLoader。"""
    train_path = os.path.join(data_dir, "train.npz")
    test_path = os.path.join(data_dir, "test.npz")

    if not os.path.exists(train_path):
        raise FileNotFoundError(
            f"Train data not found: {train_path}\n"
            f"Run pull_data_numpy.py or tfrecord2numpy.py first."
        )
    if not os.path.exists(test_path):
        raise FileNotFoundError(
            f"Test data not found: {test_path}\n"
            f"Run pull_data_numpy.py or tfrecord2numpy.py first."
        )

    print("Loading data...")
    train_dataset = CaptchaDataset(train_path)
    test_dataset = CaptchaDataset(test_path)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=False,
        drop_last=False,
        collate_fn=numpy_collate,
        generator=torch.Generator().manual_seed(RANDOM_SEED),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=False,
        drop_last=False,
        collate_fn=numpy_collate,
    )

    return train_loader, test_loader


# ╔══════════════════════════════════════════════════════════════╗
# ║  主流程                                                      ║
# ╚══════════════════════════════════════════════════════════════╝
def main():
    parser = argparse.ArgumentParser(
        description="Train ViT captcha model (Keras 3 + PyTorch backend)"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=NUMPY_DIR,
        help=f"Directory containing train.npz and test.npz (default: {NUMPY_DIR})",
    )
    parser.add_argument(
        "--epochs", type=int, default=EPOCHS, help=f"Training epochs (default: {EPOCHS})"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help=f"Batch size (default: {BATCH_SIZE})",
    )
    args = parser.parse_args()

    # 加载数据
    try:
        train_loader, val_loader = create_data_loaders(
            args.data_dir, args.batch_size
        )
        print("Data loaded successfully\n")
    except Exception as e:
        print(f"Error loading data: {e}")
        exit(1)

    # 构建模型
    print("=" * 60)
    print("Building Vision Transformer (Encoder-Only) Model")
    print(f"  Backend    : {keras.backend.backend()}")
    print(f"  Patch size : {PATCH_SIZE}x{PATCH_SIZE}")
    print(f"  Num patches: {NUM_PATCHES} ({NUM_PATCHES_H}x{NUM_PATCHES_W})")
    print(f"  D_model    : {D_MODEL}")
    print(f"  Heads      : {NUM_HEADS}")
    print(f"  Layers     : {NUM_LAYERS}")
    print(f"  DFF        : {DFF}")
    print(f"  Batch size : {args.batch_size}")
    print(f"  Epochs     : {args.epochs}")
    print("=" * 60 + "\n")

    model = build_vit_captcha_model(
        patch_size=PATCH_SIZE,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        dff=DFF,
        dropout_rate=DROPOUT_RATE,
    )

    # 估算 total steps 用于学习率调度
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * args.epochs
    print(f"Steps per epoch: {steps_per_epoch}, Total steps: {total_steps}")

    lr_schedule = WarmupCosineDecay(
        warmup_steps=2000, max_lr=6e-4, total_steps=total_steps
    )

    model.compile(
        optimizer=keras.optimizers.AdamW(
            learning_rate=lr_schedule,
            weight_decay=1e-4,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-8,
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    os.makedirs(MODEL_DIR, exist_ok=True)

    # 训练
    history = model.fit(
        train_loader,
        validation_data=val_loader,
        epochs=args.epochs,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=15,
                restore_best_weights=True,
            ),
        ],
    )

    # 保存模型
    model.save(VIT_MODEL_PATH)
    print(f"\nModel saved to {VIT_MODEL_PATH}")
    print(
        f"Run `python -m src.data.huggingface upload_model {VIT_MODEL_PATH}` "
        f"to upload."
    )


if __name__ == "__main__":
    main()