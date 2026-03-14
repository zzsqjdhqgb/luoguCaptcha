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
import math
import argparse

# ── 设置后端（必须在 import keras 之前） ──────────────────────────
os.environ["KERAS_BACKEND"] = "torch"

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

import keras
from keras import layers, ops

# ── 固定随机种子 ──────────────────────────────────────────────────
RANDOM_SEED = 484858


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

# 参数
CHAR_SIZE = 256
CHARS_PER_LABEL = 4
IMG_HEIGHT, IMG_WIDTH = 35, 90
PATCH_SIZE = 5  # patch大小，要求能整除IMG_HEIGHT和IMG_WIDTH
D_MODEL = 128  # Transformer嵌入维度 (256 -> 128)
NUM_HEADS = 4  # 多头注意力头数 (8 -> 4)
NUM_LAYERS = 4  # Transformer encoder层数 (6 -> 4)
DFF = 256  # 前馈网络中间层维度 (512 -> 256)
DROPOUT_RATE = 0.1
EPOCHS = 150
BATCH_SIZE = 256
DATA_DIR = "data/luogu_captcha_numpy"

# 计算patch数量
NUM_PATCHES_H = IMG_HEIGHT // PATCH_SIZE  # 35 / 5 = 7
NUM_PATCHES_W = IMG_WIDTH // PATCH_SIZE   # 90 / 5 = 18
NUM_PATCHES = NUM_PATCHES_H * NUM_PATCHES_W  # 7 * 18 = 126


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
        # 转为 float32 并归一化
        self.images = data["images"].astype(np.float32) / 255.0  # (N, H, W, 1)
        self.labels = data["labels"].astype(np.int64)  # (N, 4)
        print(
            f"  Loaded {len(self.images)} samples from {npz_path} "
            f"(images: {self.images.shape}, labels: {self.labels.shape})"
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Keras 3 with torch backend 期望 channels-last (H, W, C) 的 numpy/tensor
        # Keras 内部会自动处理转换
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
        pin_memory=False,  # 不用 pin_memory，因为传 numpy 给 Keras
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
# ║  Keras 3 自定义层（后端无关，使用 keras.ops）               ║
# ╚══════════════════════════════════════════════════════════════╝
class PatchEmbedding(layers.Layer):
    """将图像分割为patch并进行线性嵌入。"""

    def __init__(self, patch_size, d_model, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.d_model = d_model
        # 用卷积实现patch提取+线性投影
        self.projection = layers.Conv2D(
            d_model,
            kernel_size=patch_size,
            strides=patch_size,
            padding="valid",
        )

    def call(self, x):
        # x: (batch, H, W, 1)
        x = self.projection(x)  # (batch, num_patches_h, num_patches_w, d_model)
        batch_size = ops.shape(x)[0]
        x = ops.reshape(x, (batch_size, -1, self.d_model))  # (batch, num_patches, d_model)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({"patch_size": self.patch_size, "d_model": self.d_model})
        return config


class LearnedPositionalEncoding(layers.Layer):
    """可学习的位置编码。"""

    def __init__(self, num_positions, d_model, **kwargs):
        super().__init__(**kwargs)
        self.num_positions = num_positions
        self.d_model = d_model

    def build(self, input_shape):
        self.pos_embedding = self.add_weight(
            name="pos_embedding",
            shape=(1, self.num_positions, self.d_model),
            initializer="truncated_normal",
            trainable=True,
        )

    def call(self, x):
        return x + self.pos_embedding

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_positions": self.num_positions,
            "d_model": self.d_model,
        })
        return config


class CLSTokens(layers.Layer):
    """可学习的 CLS tokens，拼接到序列前面。"""

    def __init__(self, num_tokens, d_model, **kwargs):
        super().__init__(**kwargs)
        self.num_tokens = num_tokens
        self.d_model = d_model

    def build(self, input_shape):
        self.cls_tokens = self.add_weight(
            name="cls_tokens",
            shape=(1, self.num_tokens, self.d_model),
            initializer="truncated_normal",
            trainable=True,
        )

    def call(self, x):
        batch_size = ops.shape(x)[0]
        cls_broadcast = ops.broadcast_to(
            self.cls_tokens, (batch_size, self.num_tokens, self.d_model)
        )
        return ops.concatenate([cls_broadcast, x], axis=1)

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_tokens": self.num_tokens,
            "d_model": self.d_model,
        })
        return config


class TransformerEncoderBlock(layers.Layer):
    """Transformer Encoder 块（Pre-Norm 架构）。"""

    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.dropout_rate = dropout_rate

        self.mha = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,
            dropout=dropout_rate,
        )
        self.ffn = keras.Sequential([
            layers.Dense(dff, activation="gelu"),
            layers.Dropout(dropout_rate),
            layers.Dense(d_model),
            layers.Dropout(dropout_rate),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(dropout_rate)

    def call(self, x, training=None):
        x_norm = self.layernorm1(x)
        attn_output = self.mha(x_norm, x_norm, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        x = x + attn_output

        x_norm = self.layernorm2(x)
        ffn_output = self.ffn(x_norm, training=training)
        x = x + ffn_output

        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "dff": self.dff,
            "dropout_rate": self.dropout_rate,
        })
        return config


class ExtractCLSTokens(layers.Layer):
    """提取序列前 N 个 token（替代 Lambda 层）。"""

    def __init__(self, num_tokens, **kwargs):
        super().__init__(**kwargs)
        self.num_tokens = num_tokens

    def call(self, x):
        return x[:, : self.num_tokens, :]

    def get_config(self):
        config = super().get_config()
        config.update({"num_tokens": self.num_tokens})
        return config


# ╔══════════════════════════════════════════════════════════════╗
# ║  学习率调度                                                  ║
# ╚══════════════════════════════════════════════════════════════╝
class WarmupCosineDecay(keras.optimizers.schedules.LearningRateSchedule):
    """Warmup + Cosine Decay。"""

    def __init__(self, warmup_steps=2000, max_lr=6e-4, total_steps=50000):
        super().__init__()
        self.warmup_steps = warmup_steps
        self.max_lr = max_lr
        self.total_steps = total_steps

    def __call__(self, step):
        step = ops.cast(step, "float32")
        warmup_steps = ops.cast(self.warmup_steps, "float32")
        total_steps = ops.cast(self.total_steps, "float32")

        warmup_lr = self.max_lr * (step / warmup_steps)

        progress = (step - warmup_steps) / ops.maximum(
            total_steps - warmup_steps, 1.0
        )
        decay_lr = self.max_lr * 0.5 * (1.0 + ops.cos(math.pi * progress))

        return ops.where(step < warmup_steps, warmup_lr, decay_lr)

    def get_config(self):
        return {
            "warmup_steps": self.warmup_steps,
            "max_lr": self.max_lr,
            "total_steps": self.total_steps,
        }


# ╔══════════════════════════════════════════════════════════════╗
# ║  构建模型                                                    ║
# ╚══════════════════════════════════════════════════════════════╝
def build_vit_captcha_model():
    """构建 Vision Transformer (Encoder-Only) 验证码识别模型。"""

    inputs = keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 1), name="input")

    # Patch Embedding
    x = PatchEmbedding(
        patch_size=PATCH_SIZE, d_model=D_MODEL, name="patch_embedding"
    )(inputs)

    # CLS tokens
    x = CLSTokens(
        num_tokens=CHARS_PER_LABEL, d_model=D_MODEL, name="cls_tokens"
    )(x)

    # 位置编码
    x = LearnedPositionalEncoding(
        num_positions=CHARS_PER_LABEL + NUM_PATCHES,
        d_model=D_MODEL,
        name="positional_encoding",
    )(x)
    x = layers.Dropout(DROPOUT_RATE)(x)

    # Transformer Encoder
    for i in range(NUM_LAYERS):
        x = TransformerEncoderBlock(
            d_model=D_MODEL,
            num_heads=NUM_HEADS,
            dff=DFF,
            dropout_rate=DROPOUT_RATE,
            name=f"transformer_block_{i}",
        )(x)

    # Final LayerNorm
    x = layers.LayerNormalization(epsilon=1e-6, name="final_norm")(x)

    # 提取 CLS tokens
    x = ExtractCLSTokens(num_tokens=CHARS_PER_LABEL, name="extract_cls")(x)

    # 分类头
    x = layers.Dense(DFF, activation="gelu", name="cls_head_dense")(x)
    x = layers.Dropout(DROPOUT_RATE)(x)
    outputs = layers.Dense(
        CHAR_SIZE, activation="softmax", name="cls_head_output"
    )(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="LuoguCaptcha_ViT")
    return model


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
        default=DATA_DIR,
        help=f"Directory containing train.npz and test.npz (default: {DATA_DIR})",
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

    model = build_vit_captcha_model()

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

    os.makedirs("models", exist_ok=True)

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
    final_model_path = "models/luoguCaptcha.ViT-EncoderOnly.keras"
    model.save(final_model_path)
    print(f"\nModel saved to {final_model_path}")
    print(
        f"Run `python scripts/huggingface.py upload_model {final_model_path}` "
        f"to upload."
    )


if __name__ == "__main__":
    main()