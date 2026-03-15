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
Vision Transformer 自定义层。

所有层均使用 @keras.saving.register_keras_serializable 注册，
以支持 model.save() / load_model() 的完整序列化。
"""

import keras
from keras import layers, ops


@keras.saving.register_keras_serializable(package="luoguCaptcha")
class PatchEmbedding(layers.Layer):
    """将图像分割为 patch 并进行线性嵌入。"""

    def __init__(self, patch_size, d_model, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.d_model = d_model
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


@keras.saving.register_keras_serializable(package="luoguCaptcha")
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


@keras.saving.register_keras_serializable(package="luoguCaptcha")
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


@keras.saving.register_keras_serializable(package="luoguCaptcha")
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


@keras.saving.register_keras_serializable(package="luoguCaptcha")
class ExtractCLSTokens(layers.Layer):
    """提取序列前 N 个 token。"""

    def __init__(self, num_tokens, **kwargs):
        super().__init__(**kwargs)
        self.num_tokens = num_tokens

    def call(self, x):
        return x[:, : self.num_tokens, :]

    def get_config(self):
        config = super().get_config()
        config.update({"num_tokens": self.num_tokens})
        return config