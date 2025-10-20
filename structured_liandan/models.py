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
from tensorflow import keras
from tensorflow.keras import layers
from config import Config


def build_stage1_model():
    """
    构建阶段1模型：普通CNN + BiLSTM
    
    架构：
        - 2层卷积（64, 128通道）
        - MaxPooling下采样
        - Reshape转为序列
        - 2层双向LSTM
        - Dense输出层
    
    Returns:
        model: Keras模型
    """
    inputs = keras.Input(
        shape=(Config.IMG_HEIGHT, Config.IMG_WIDTH, 1), 
        name="input"
    )

    # ========== CNN部分 ==========
    x = layers.Conv2D(
        Config.CNN_FILTERS[0], 3, 
        padding="same", 
        activation="relu", 
        name="cnn_conv1"
    )(inputs)
    x = layers.BatchNormalization(name="cnn_bn1")(x)
    x = layers.MaxPooling2D((2, 2), name="cnn_pool1")(x)

    x = layers.Conv2D(
        Config.CNN_FILTERS[1], 3, 
        padding="same", 
        activation="relu", 
        name="cnn_conv2"
    )(x)
    x = layers.BatchNormalization(name="cnn_bn2")(x)
    x = layers.MaxPooling2D((2, 2), name="cnn_pool2")(x)

    # ========== 转换为序列 ==========
    cnn_shape = x.shape
    x = layers.Reshape(
        (cnn_shape[1] * cnn_shape[2], cnn_shape[3]), 
        name="reshape_to_seq"
    )(x)

    # ========== BiLSTM部分 ==========
    x = layers.Bidirectional(
        layers.LSTM(Config.LSTM_UNITS, return_sequences=True), 
        name="bilstm_1"
    )(x)
    x = layers.Dropout(Config.DROPOUT_RATE, name="dropout_1")(x)
    
    x = layers.Bidirectional(
        layers.LSTM(Config.LSTM_UNITS, return_sequences=False), 
        name="bilstm_2"
    )(x)
    x = layers.Dropout(Config.DROPOUT_RATE, name="dropout_2")(x)

    # ========== 输出层 ==========
    x = layers.Dense(
        Config.CHARS_PER_LABEL * Config.CHAR_SIZE, 
        activation="softmax", 
        name="dense_output"
    )(x)
    outputs = layers.Reshape(
        (Config.CHARS_PER_LABEL, Config.CHAR_SIZE), 
        name="reshape_output"
    )(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="Stage1_CNN_BiLSTM")
    return model


def build_stage2_model(baseline_model):
    """
    构建阶段2模型：添加Self-Attention，冻结CNN
    
    架构：
        - 复用阶段1的CNN部分（冻结权重）
        - 添加MultiHeadAttention层
        - 重新初始化BiLSTM
        - Dense输出层
    
    Args:
        baseline_model: 阶段1训练好的模型
        
    Returns:
        model: Keras模型
    """
    inputs = keras.Input(
        shape=(Config.IMG_HEIGHT, Config.IMG_WIDTH, 1), 
        name="input"
    )
    
    # ========== 构建CNN部分（结构与stage1相同）==========
    x = inputs
    x = layers.Conv2D(
        Config.CNN_FILTERS[0], 3, 
        padding="same", 
        activation="relu", 
        name="cnn_conv1_frozen"
    )(x)
    x = layers.BatchNormalization(name="cnn_bn1_frozen")(x)
    x = layers.MaxPooling2D((2, 2), name="cnn_pool1_frozen")(x)
    
    x = layers.Conv2D(
        Config.CNN_FILTERS[1], 3, 
        padding="same", 
        activation="relu", 
        name="cnn_conv2_frozen"
    )(x)
    x = layers.BatchNormalization(name="cnn_bn2_frozen")(x)
    x = layers.MaxPooling2D((2, 2), name="cnn_pool2_frozen")(x)
    
    # ========== 创建临时模型用于复制权重 ==========
    temp_model = keras.Model(inputs=inputs, outputs=x, name="temp_cnn")
    
    # 复制CNN权重
    print("Copying CNN weights from Stage 1...")
    for i in range(1, 7):  # 6个CNN层（跳过input层）
        try:
            weights = baseline_model.layers[i].get_weights()
            if weights:
                temp_model.layers[i].set_weights(weights)
                print(f"  ✓ Layer {i}: {baseline_model.layers[i].name}")
        except Exception as e:
            print(f"  ⚠ Layer {i}: {e}")
    
    print("✓ CNN weights copied successfully\n")
    
    # ========== 转换为序列 ==========
    cnn_shape = x.shape
    x = layers.Reshape(
        (cnn_shape[1] * cnn_shape[2], cnn_shape[3]), 
        name="reshape_to_seq"
    )(x)
    
    # ========== Self-Attention层（新增）==========
    attention_output = layers.MultiHeadAttention(
        num_heads=Config.ATTENTION_HEADS,
        key_dim=Config.ATTENTION_KEY_DIM,
        dropout=0.1,
        name="self_attention"
    )(x, x)
    x = layers.LayerNormalization(name="attn_norm")(x + attention_output)
    
    # ========== BiLSTM部分（重新初始化）==========
    x = layers.Bidirectional(
        layers.LSTM(Config.LSTM_UNITS, return_sequences=True), 
        name="bilstm_1_new"
    )(x)
    x = layers.Dropout(Config.DROPOUT_RATE, name="dropout_1_new")(x)
    
    x = layers.Bidirectional(
        layers.LSTM(Config.LSTM_UNITS, return_sequences=False), 
        name="bilstm_2_new"
    )(x)
    x = layers.Dropout(Config.DROPOUT_RATE, name="dropout_2_new")(x)
    
    # ========== 输出层 ==========
    x = layers.Dense(
        Config.CHARS_PER_LABEL * Config.CHAR_SIZE, 
        activation="softmax", 
        name="dense_output_new"
    )(x)
    outputs = layers.Reshape(
        (Config.CHARS_PER_LABEL, Config.CHAR_SIZE), 
        name="reshape_output_new"
    )(x)
    
    model = keras.Model(
        inputs=inputs, 
        outputs=outputs, 
        name="Stage2_CNN_Attention_BiLSTM"
    )
    
    # ========== 冻结CNN层 ==========
    print("Freezing CNN layers...")
    for i in range(1, 7):
        model.layers[i].trainable = False
        print(f"  Frozen: {model.layers[i].name}")
    print()
    
    return model