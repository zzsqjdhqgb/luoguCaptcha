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
Keras 3 + JAX 后端，数据从 NumPy .npz 加载（全量装入内存）。

Usage:
  python train.py [--data-dir DIR] [--epochs N] [--batch-size N]
"""

import os
import argparse

# ── 设置后端（必须在 import keras 之前） ──────────────────────────
os.environ["KERAS_BACKEND"] = "jax"

import numpy as np

import jax
import keras
import matplotlib

matplotlib.use("Agg")  # 无头环境下使用非交互后端
import matplotlib.pyplot as plt

from config import (
    CHAR_SIZE,
    CHARS_PER_LABEL,
    IMG_HEIGHT,
    IMG_WIDTH,
    NUMPY_DIR,
    MODEL_DIR,
    VIT_MODEL_PATH,
    CHECKPOINT_MODEL_PATH,
)
from model import build_vit_captcha_model
from data.load import load_captcha_data

# ── 训练超参数 ────────────────────────────────────────────────────
RANDOM_SEED = 484858
PATCH_SIZE = 5
D_MODEL = 128
NUM_HEADS = 4
NUM_LAYERS = 4
DFF = 256
DROPOUT_RATE = 0.1
EPOCHS = 450
BATCH_SIZE = 256

# 计算 patch 数量
NUM_PATCHES_H = IMG_HEIGHT // PATCH_SIZE  # 35 / 5 = 7
NUM_PATCHES_W = IMG_WIDTH // PATCH_SIZE   # 90 / 5 = 18
NUM_PATCHES = NUM_PATCHES_H * NUM_PATCHES_W  # 7 * 18 = 126


# ── 固定随机种子 ──────────────────────────────────────────────────
def set_all_seeds(seed: int):
    np.random.seed(seed)
    keras.utils.set_random_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


set_all_seeds(RANDOM_SEED)

# ── 设备信息 ──────────────────────────────────────────────────────
jax_devices = jax.devices()
print(f"JAX devices   : {jax_devices}")
print(f"JAX backend   : {jax.default_backend()}")
print(f"Keras version : {keras.__version__}")
print(f"Backend       : {keras.backend.backend()}")
print(f"Random seed   : {RANDOM_SEED}")


# ╔══════════════════════════════════════════════════════════════╗
# ║  学习率记录回调                                               ║
# ╚══════════════════════════════════════════════════════════════╝
class LearningRateLogger(keras.callbacks.Callback):
    """每个 epoch 结束时记录当前学习率。"""

    def __init__(self):
        super().__init__()
        self.learning_rates = []

    def on_epoch_end(self, epoch, logs=None):
        optimizer = self.model.optimizer
        # 获取当前学习率
        if hasattr(optimizer, "learning_rate"):
            lr = optimizer.learning_rate
            # 如果是调度器，需要用当前 step 求值
            if callable(lr):
                current_step = optimizer.iterations
                lr_value = float(lr(current_step))
            else:
                lr_value = float(lr)
        else:
            lr_value = float(keras.backend.get_value(optimizer.lr))

        self.learning_rates.append(lr_value)

        if logs is not None:
            logs["lr"] = lr_value
        else:
            print(f"  ↳ Learning rate: {lr_value:.2e}")


# ╔══════════════════════════════════════════════════════════════╗
# ║  绘图                                                        ║
# ╚══════════════════════════════════════════════════════════════╝
def save_training_plots(history, learning_rates, save_dir: str):
    """
    保存训练曲线图，包含 4 个子图：
    1. Loss (log scale)
    2. Accuracy (train & val)
    3. Learning Rate
    4. Val Loss (放大)
    """
    epochs = range(1, len(history.history["loss"]) + 1)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Training History", fontsize=16, fontweight="bold")

    # ── 1. Loss (Log Scale) ──
    ax = axes[0, 0]
    ax.plot(epochs, history.history["loss"], "b-", label="Train Loss", linewidth=1.5)
    ax.plot(epochs, history.history["val_loss"], "r-", label="Val Loss", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (Log Scale)")
    ax.set_title("Loss (Log Scale)")
    
    # 核心修改：设置 Y 轴为对数尺度
    ax.set_yscale("log") 
    
    ax.legend()
    # 对于对数坐标，显示主次网格线会让图表更容易阅读
    ax.grid(True, which='both', linestyle='-', alpha=0.2)

    # ── 2. Accuracy ──
    ax = axes[0, 1]
    ax.plot(
        epochs, history.history["accuracy"], "b-",
        label="Train Accuracy", linewidth=1.5,
    )
    ax.plot(
        epochs, history.history["val_accuracy"], "r-",
        label="Val Accuracy", linewidth=1.5,
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── 3. Learning Rate ──
    ax = axes[1, 0]
    lr_epochs = range(1, len(learning_rates) + 1)
    ax.plot(lr_epochs, learning_rates, "g-", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.set_title("Learning Rate Schedule")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    # ── 4. Val Loss 放大 ──
    ax = axes[1, 1]
    val_loss = history.history["val_loss"]
    ax.plot(epochs, val_loss, "r-", linewidth=1.5)
    # 标注最低点
    best_epoch = int(np.argmin(val_loss)) + 1
    best_val_loss = min(val_loss)
    ax.axvline(x=best_epoch, color="gray", linestyle="--", alpha=0.7)
    ax.annotate(
        f"Best: {best_val_loss:.4f}\n(epoch {best_epoch})",
        xy=(best_epoch, best_val_loss),
        xytext=(best_epoch + len(val_loss) * 0.05, best_val_loss * 1.1),
        arrowprops=dict(arrowstyle="->", color="gray"),
        fontsize=9,
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Val Loss")
    ax.set_title("Validation Loss (detail)")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    plot_path = os.path.join(save_dir, "loss_plot.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Training plot saved to {plot_path}")


# ╔══════════════════════════════════════════════════════════════╗
# ║  主流程                                                      ║
# ╚══════════════════════════════════════════════════════════════╝
def main():
    parser = argparse.ArgumentParser(
        description="Train ViT captcha model (Keras 3 + JAX backend)"
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

    # 加载数据（全量装入内存）
    try:
        (x_train, y_train), (x_test, y_test) = load_captcha_data(args.data_dir)
        print("Data loaded successfully\n")
    except Exception as e:
        print(f"Error loading data: {e}")
        exit(1)

    # 构建模型
    print("=" * 60)
    print("Building Vision Transformer (Encoder-Only) Model")
    print(f"  Backend    : {keras.backend.backend()}")
    print(f"  JAX devices: {jax.devices()}")
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
    steps_per_epoch = int(np.ceil(len(x_train) / args.batch_size))
    total_steps = steps_per_epoch * args.epochs
    print(f"Steps per epoch: {steps_per_epoch}, Total steps: {total_steps}")

    # Keras 3 的 CosineDecay 已经内置 warmup_target 参数！
    lr_schedule = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=0.0,       # warmup 起始
        decay_steps=total_steps,
        alpha=1e-4,                      # 最终学习率（不会降到0）
        warmup_target=6e-4,              # warmup 目标峰值
        warmup_steps=2000,               # warmup 步数
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

    # 回调
    lr_logger = LearningRateLogger()

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=60,
            restore_best_weights=True,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=CHECKPOINT_MODEL_PATH,      # 或者用别的路径如 "best_model.keras"
            monitor="val_loss",
            save_best_only=True,          # ← 只在 val_loss 创新低时保存
            verbose=1,
        ),
        lr_logger,
    ]

    # 训练（直接传入 NumPy 数组，Keras 3 会自动处理 batching 和 shuffling）
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_test, y_test),
        epochs=args.epochs,
        batch_size=args.batch_size,
        shuffle=True,
        callbacks=callbacks,
    )

    # 保存训练曲线
    save_training_plots(history, lr_logger.learning_rates, MODEL_DIR)

    # 保存模型
    model.save(VIT_MODEL_PATH)
    print(f"\nModel saved to {VIT_MODEL_PATH}")
    print(
        f"Run `python -m data.huggingface upload_model {VIT_MODEL_PATH}` "
        f"to upload."
    )


if __name__ == "__main__":
    main()