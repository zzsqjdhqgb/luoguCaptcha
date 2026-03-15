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
学习率调度器。

使用 @keras.saving.register_keras_serializable 注册，
以支持 optimizer 的完整序列化。
"""

import math

import keras
from keras import ops


@keras.saving.register_keras_serializable(package="luoguCaptcha")
class WarmupCosineDecay(keras.optimizers.schedules.LearningRateSchedule):
    """Warmup + Cosine Decay 学习率调度。"""

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