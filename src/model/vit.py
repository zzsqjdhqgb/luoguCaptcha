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
"""

import keras
from keras import layers

from config import CHAR_SIZE, CHARS_PER_LABEL, IMG_HEIGHT, IMG_WIDTH
from model.layers import (
    PatchEmbedding,
    LearnedPositionalEncoding,
    CLSTokens,
    TransformerEncoderBlock,
    ExtractCLSTokens,
)


def build_vit_captcha_model(
    patch_size: int = 5,
    d_model: int = 128,
    num_heads: int = 4,
    num_layers: int = 4,
    dff: int = 256,
    dropout_rate: float = 0.1,
) -> keras.Model:
    """
    构建 Vision Transformer (Encoder-Only) 验证码识别模型。

    Args:
        patch_size: patch 边长，需要能整除 IMG_HEIGHT 和 IMG_WIDTH。
        d_model: Transformer 嵌入维度。
        num_heads: 多头注意力头数。
        num_layers: Transformer Encoder 层数。
        dff: 前馈网络中间层维度。
        dropout_rate: Dropout 比率。

    Returns:
        编译前的 keras.Model。
    """
    assert IMG_HEIGHT % patch_size == 0, (
        f"IMG_HEIGHT ({IMG_HEIGHT}) must be divisible by patch_size ({patch_size})"
    )
    assert IMG_WIDTH % patch_size == 0, (
        f"IMG_WIDTH ({IMG_WIDTH}) must be divisible by patch_size ({patch_size})"
    )

    num_patches_h = IMG_HEIGHT // patch_size
    num_patches_w = IMG_WIDTH // patch_size
    num_patches = num_patches_h * num_patches_w

    inputs = keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 1), name="input")

    # Patch Embedding
    x = PatchEmbedding(
        patch_size=patch_size, d_model=d_model, name="patch_embedding"
    )(inputs)

    # CLS tokens
    x = CLSTokens(
        num_tokens=CHARS_PER_LABEL, d_model=d_model, name="cls_tokens"
    )(x)

    # 位置编码
    x = LearnedPositionalEncoding(
        num_positions=CHARS_PER_LABEL + num_patches,
        d_model=d_model,
        name="positional_encoding",
    )(x)
    x = layers.Dropout(dropout_rate)(x)

    # Transformer Encoder
    for i in range(num_layers):
        x = TransformerEncoderBlock(
            d_model=d_model,
            num_heads=num_heads,
            dff=dff,
            dropout_rate=dropout_rate,
            name=f"transformer_block_{i}",
        )(x)

    # Final LayerNorm
    x = layers.LayerNormalization(epsilon=1e-6, name="final_norm")(x)

    # 提取 CLS tokens
    x = ExtractCLSTokens(num_tokens=CHARS_PER_LABEL, name="extract_cls")(x)

    # 分类头
    x = layers.Dense(dff, activation="gelu", name="cls_head_dense")(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(
        CHAR_SIZE, activation="softmax", name="cls_head_output"
    )(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="LuoguCaptcha_ViT")
    return model