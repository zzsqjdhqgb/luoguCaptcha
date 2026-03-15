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
luoguCaptcha 模型定义包。

导入此包即可触发所有自定义层的注册（@register_keras_serializable），
使得 keras.models.load_model() 能正确反序列化。
"""

from model.layers import (
    PatchEmbedding,
    LearnedPositionalEncoding,
    CLSTokens,
    TransformerEncoderBlock,
    TransformerEncoder,
    ExtractCLSTokens,
)
from model.lr_schedule import WarmupCosineDecay
from model.vit import build_vit_captcha_model

__all__ = [
    "PatchEmbedding",
    "LearnedPositionalEncoding",
    "CLSTokens",
    "TransformerEncoderBlock",
    "TransformerEncoder",
    "ExtractCLSTokens",
    "WarmupCosineDecay",
    "build_vit_captcha_model",
]