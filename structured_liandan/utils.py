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

import tensorflow as tf


def setup_gpu():
    """配置GPU设置，启用动态显存增长"""
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✓ Using GPU: {gpus}")
        except Exception as e:
            print(f"⚠ GPU setup error: {e}")
    else:
        print("ℹ Using CPU")


def print_stage_header(stage_name):
    """
    打印阶段标题
    
    Args:
        stage_name: 阶段名称
    """
    print("\n" + "="*60)
    print(f"{stage_name}")
    print("="*60 + "\n")


def analyze_improvement(stage_name, acc_before, acc_after):
    """
    分析性能改进情况
    
    Args:
        stage_name: 阶段名称
        acc_before: 改进前的准确率
        acc_after: 改进后的准确率
        
    Returns:
        improvement: 改进幅度
    """
    improvement = acc_after - acc_before
    print(f"\n{stage_name}: {acc_before:.4f} → {acc_after:.4f} ({improvement:+.4f})")
    
    if improvement > 0.01:
        print(f"  ✓ Improved (+{improvement:.2%})")
    elif improvement > -0.02:
        print(f"  → Maintained performance")
    else:
        print(f"  ✗ Performance dropped ({improvement:.2%})")
    
    return improvement


def print_model_info(model):
    """
    打印模型信息（可训练/不可训练参数）
    
    Args:
        model: Keras模型
    """
    trainable_count = sum([tf.size(w).numpy() for w in model.trainable_weights])
    non_trainable_count = sum([tf.size(w).numpy() for w in model.non_trainable_weights])
    
    print("\nModel Parameters:")
    print(f"  Trainable:     {trainable_count:,}")
    print(f"  Non-trainable: {non_trainable_count:,}")
    print(f"  Total:         {trainable_count + non_trainable_count:,}")