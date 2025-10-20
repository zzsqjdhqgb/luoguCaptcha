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


class Config:
    """全局配置类"""
    
    # ========== 数据参数 ==========
    CHAR_SIZE = 256
    CHARS_PER_LABEL = 4
    IMG_HEIGHT = 35
    IMG_WIDTH = 90
    BATCH_SIZE = 256
    TFRECORD_DIR = "data/luogu_captcha_tfrecord"
    
    # ========== 训练参数 ==========
    EPOCHS_STAGE1 = 100
    EPOCHS_STAGE2 = 15
    EPOCHS_STAGE3 = 150
    
    LR_STAGE1 = 0.001
    LR_STAGE2 = 0.002
    LR_STAGE3 = 0.0001
    
    # ========== 模型路径 ==========
    MODEL_DIR = "models"
    STAGE1_MODEL_PATH = "models/stage1_cnn_bilstm.keras"
    STAGE2_MODEL_PATH = "models/stage2_with_attention.keras"
    FINAL_MODEL_PATH = "models/luoguCaptcha_final.keras"
    HISTORY_PATH = "models/training_history_all_stages.json"
    
    # ========== 训练控制开关 ==========
    SKIP_STAGE1 = True   # 设为 True 跳过阶段1（需要已有模型）
    SKIP_STAGE2 = False  # 设为 True 跳过阶段2
    SKIP_STAGE3 = False  # 设为 True 跳过阶段3
    
    # ========== EarlyStopping 参数 ==========
    PATIENCE_STAGE1 = 20
    PATIENCE_STAGE2 = 3
    PATIENCE_STAGE3 = 15
    
    STAGE1_TARGET_ACC = 0.80  # 阶段1目标准确率
    
    # ========== 模型超参数 ==========
    CNN_FILTERS = [64, 128]
    LSTM_UNITS = 128
    DROPOUT_RATE = 0.3
    ATTENTION_HEADS = 4
    ATTENTION_KEY_DIM = 64