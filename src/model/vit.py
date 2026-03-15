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
Vision Transformer (Encoder-Only) 验证码识别模型构建函数。

使用 keras.Sequential 风格组装模型。
"""

import keras
from keras import layers

from config import CHAR_SIZE, CHARS_PER_LABEL, IMG_HEIGHT, IMG_WIDTH
from model.layers import (
    PatchEmbedding,
    LearnedPositionalEncoding,
    CLSTokens,
    TransformerEncoder,
    ExtractCLSTokens,
)


def build_vit_captcha_model(
    patch_size: int = 5,
    d_model: int = 128,
    num_heads: int = 4,
    num_layers: int = 4,
    dff: int = 256,
    dropout_rate: float = 0.1,
) -> keras.Sequential:
    """
    构建 Vision Transformer 验证码识别模型（Sequential 风格）。

    Args:
        patch_size: patch 边长，需要能整除 IMG_HEIGHT 和 IMG_WIDTH。
        d_model: Transformer 嵌入维度。
        num_heads: 多头注意力头数。
        num_layers: Transformer Encoder 层数。
        dff: 前馈网络中间层维度。
        dropout_rate: Dropout 比率。

    Returns:
        keras.Sequential 模型。
    """
    assert IMG_HEIGHT % patch_size == 0, (
        f"IMG_HEIGHT ({IMG_HEIGHT}) must be divisible by patch_size ({patch_size})"
    )
    assert IMG_WIDTH % patch_size == 0, (
        f"IMG_WIDTH ({IMG_WIDTH}) must be divisible by patch_size ({patch_size})"
    )

    num_patches = (IMG_HEIGHT // patch_size) * (IMG_WIDTH // patch_size)

    model = keras.Sequential(name="LuoguCaptcha_ViT")

    # 输入形状声明
    model.add(keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 1), name="input"))

    # ---- 编码 ----
    model.add(PatchEmbedding(
        patch_size=patch_size, d_model=d_model, name="patch_embedding",
    ))
    model.add(CLSTokens(
        num_tokens=CHARS_PER_LABEL, d_model=d_model, name="cls_tokens",
    ))
    model.add(LearnedPositionalEncoding(
        num_positions=CHARS_PER_LABEL + num_patches,
        d_model=d_model,
        name="positional_encoding",
    ))
    model.add(layers.Dropout(dropout_rate, name="pos_dropout"))

    # ---- Transformer 编码器 ----
    model.add(TransformerEncoder(
        d_model=d_model,
        num_heads=num_heads,
        dff=dff,
        num_layers=num_layers,
        dropout_rate=dropout_rate,
        name="transformer_encoder",
    ))

    # ---- 输出 ----
    model.add(layers.LayerNormalization(epsilon=1e-6, name="final_norm"))
    model.add(ExtractCLSTokens(
        num_tokens=CHARS_PER_LABEL, name="extract_cls",
    ))
    model.add(layers.Dense(dff, activation="gelu", name="cls_head_dense"))
    model.add(layers.Dropout(dropout_rate, name="cls_head_dropout"))
    model.add(layers.Dense(
        CHAR_SIZE, activation="softmax", name="cls_head_output",
    ))

    return model