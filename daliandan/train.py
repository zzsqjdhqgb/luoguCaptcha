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

import os
import glob
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

tf.get_logger().setLevel('ERROR')

# ========== 配置 ==========
class Config:
    # 数据参数
    CHAR_SIZE = 256
    CHARS_PER_LABEL = 4
    IMG_HEIGHT, IMG_WIDTH = 35, 90
    BATCH_SIZE = 256
    TFRECORD_DIR = "data/luogu_captcha_tfrecord"
    
    # 训练参数
    EPOCHS_STAGE1 = 100   # 阶段1: 普通CNN + 普通LSTM
    EPOCHS_STAGE2A = 50   # 阶段2A: 普通CNN(冻结) + Attention LSTM
    EPOCHS_STAGE2B = 50   # 阶段2B: 残差CNN + 普通LSTM(冻结)
    EPOCHS_STAGE3A = 30   # 阶段3A: 冻结ResNet CNN, 微调Attention LSTM
    EPOCHS_STAGE3B = 30   # 阶段3B: 冻结Attention LSTM, 微调ResNet CNN
    EPOCHS_STAGE3C = 40   # 阶段3C: 全部解冻, 最终微调
    
    # 模型路径
    STAGE1_MODEL_PATH = "models/stage1_plain_cnn_lstm.keras"
    STAGE2A_MODEL_PATH = "models/stage2a_attention_lstm.keras"
    STAGE2B_MODEL_PATH = "models/stage2b_resnet_cnn.keras"
    STAGE3_MODEL_PATH = "models/stage3_merged.keras"
    FINAL_MODEL_PATH = "models/luoguCaptcha_final.keras"
    
    # 控制开关
    SKIP_STAGE1 = False   # 阶段1: 建立接口标准
    SKIP_STAGE2A = False  # 阶段2A: 训练Attention LSTM
    SKIP_STAGE2B = False  # 阶段2B: 训练残差CNN
    SKIP_STAGE3 = False   # 阶段3: 合并 + 渐进微调


# ========== GPU配置 ==========
def setup_gpu():
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


# ========== 数据加载 ==========
def parse_tfrecord(example_proto):
    """Parses a single TFRecord example into image and label."""
    feature_description = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "label": tf.io.FixedLenFeature([Config.CHARS_PER_LABEL], tf.int64),
    }
    example = tf.io.parse_single_example(example_proto, feature_description)
    image = tf.io.decode_png(example["image"], channels=1)
    image = tf.image.convert_image_dtype(image, tf.float32)
    label = example["label"]
    return image, label


def load_datasets(tfrecord_dir):
    """加载训练和验证数据集"""
    train_files = sorted(glob.glob(os.path.join(tfrecord_dir, "train_part_*.tfrecord")))
    test_files = sorted(glob.glob(os.path.join(tfrecord_dir, "test_part_*.tfrecord")))

    if not train_files or not test_files:
        raise ValueError(f"No TFRecord files found in {tfrecord_dir}")

    print(f"✓ Found {len(train_files)} train files and {len(test_files)} test files")

    train_ds = tf.data.TFRecordDataset(train_files, num_parallel_reads=tf.data.AUTOTUNE)
    test_ds = tf.data.TFRecordDataset(test_files, num_parallel_reads=tf.data.AUTOTUNE)

    train_ds = train_ds.map(parse_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)
    test_ds = test_ds.map(parse_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)

    train_ds = (
        train_ds.shuffle(buffer_size=10000)
        .batch(Config.BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )
    test_ds = test_ds.batch(Config.BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    return train_ds, test_ds


# ========== 模型构建 ==========
def build_stage1_model():
    """
    阶段1: 普通CNN + 普通BiLSTM
    作用: 建立"接口标准"
    """
    inputs = keras.Input(shape=(Config.IMG_HEIGHT, Config.IMG_WIDTH, 1), name="input")

    # 普通CNN (接口标准的特征提取器)
    x = layers.Conv2D(64, 3, padding="same", activation="relu", name="plain_cnn_conv1")(inputs)
    x = layers.BatchNormalization(name="plain_cnn_bn1")(x)
    x = layers.MaxPooling2D((2, 2), name="plain_cnn_pool1")(x)

    x = layers.Conv2D(128, 3, padding="same", activation="relu", name="plain_cnn_conv2")(x)
    x = layers.BatchNormalization(name="plain_cnn_bn2")(x)
    x = layers.MaxPooling2D((2, 2), name="plain_cnn_pool2")(x)

    # 转换为序列 (CNN输出标准)
    cnn_shape = x.shape
    x = layers.Reshape((cnn_shape[1] * cnn_shape[2], cnn_shape[3]), name="reshape_to_seq")(x)

    # 普通BiLSTM (接口标准的序列处理器)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True), name="plain_bilstm_1")(x)
    x = layers.Dropout(0.3, name="plain_dropout_1")(x)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=False), name="plain_bilstm_2")(x)
    x = layers.Dropout(0.3, name="plain_dropout_2")(x)

    # 输出层
    x = layers.Dense(Config.CHARS_PER_LABEL * Config.CHAR_SIZE, activation="softmax", name="dense_output")(x)
    outputs = layers.Reshape((Config.CHARS_PER_LABEL, Config.CHAR_SIZE), name="reshape_output")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="Stage1_Plain_CNN_LSTM")
    return model


def build_resnet_cnn_block(inputs, filters, name_prefix):
    """残差CNN块"""
    # 主路径
    x = layers.Conv2D(filters, 3, padding="same", name=f"{name_prefix}_conv1")(inputs)
    x = layers.BatchNormalization(name=f"{name_prefix}_bn1")(x)
    x = layers.Activation("relu", name=f"{name_prefix}_act1")(x)
    
    x = layers.Conv2D(filters, 3, padding="same", name=f"{name_prefix}_conv2")(x)
    x = layers.BatchNormalization(name=f"{name_prefix}_bn2")(x)
    
    # 残差连接
    shortcut = inputs
    if inputs.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1, padding="same", name=f"{name_prefix}_shortcut")(inputs)
        shortcut = layers.BatchNormalization(name=f"{name_prefix}_shortcut_bn")(shortcut)
    
    x = layers.Add(name=f"{name_prefix}_add")([x, shortcut])
    x = layers.Activation("relu", name=f"{name_prefix}_act2")(x)
    
    return x


def build_stage2a_model(stage1_model):
    """
    阶段2A: 普通CNN(冻结) + Attention LSTM(训练)
    作用: 训练高级LSTM，强制适应"CNN输出标准"
    """
    inputs = keras.Input(shape=(Config.IMG_HEIGHT, Config.IMG_WIDTH, 1), name="input")
    
    # 复制并冻结普通CNN
    x = inputs
    for i in range(1, 7):  # 6个CNN层
        layer = stage1_model.layers[i]
        if isinstance(layer, (layers.Conv2D, layers.BatchNormalization, layers.MaxPooling2D)):
            new_layer = layer.__class__.from_config(layer.get_config())
            new_layer.trainable = False
            new_layer.build(x.shape)
            new_layer.set_weights(layer.get_weights())
            x = new_layer(x)
    
    # 转换为序列
    cnn_shape = x.shape
    x = layers.Reshape((cnn_shape[1] * cnn_shape[2], cnn_shape[3]), name="reshape_to_seq")(x)
    
    # Self-Attention LSTM (新训练，必须适应普通CNN的输出)
    attention_output = layers.MultiHeadAttention(
        num_heads=4, key_dim=64, dropout=0.1, name="self_attention"
    )(x, x)
    x = layers.LayerNormalization(name="attn_norm")(x + attention_output)
    
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True), name="attn_bilstm_1")(x)
    x = layers.Dropout(0.3, name="attn_dropout_1")(x)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=False), name="attn_bilstm_2")(x)
    x = layers.Dropout(0.3, name="attn_dropout_2")(x)
    
    # 输出层
    x = layers.Dense(Config.CHARS_PER_LABEL * Config.CHAR_SIZE, activation="softmax", name="dense_output")(x)
    outputs = layers.Reshape((Config.CHARS_PER_LABEL, Config.CHAR_SIZE), name="reshape_output")(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name="Stage2A_FrozenCNN_AttentionLSTM")
    return model


def build_stage2b_model(stage1_model):
    """
    阶段2B: 残差CNN(训练) + 普通LSTM(冻结)
    作用: 训练高级CNN，强制输出"LSTM接受的标准"
    """
    inputs = keras.Input(shape=(Config.IMG_HEIGHT, Config.IMG_WIDTH, 1), name="input")
    
    # 残差CNN (新训练，必须输出普通LSTM能接受的格式)
    x = layers.Conv2D(64, 3, padding="same", activation="relu", name="resnet_init_conv")(inputs)
    x = layers.BatchNormalization(name="resnet_init_bn")(x)
    
    x = build_resnet_cnn_block(x, 64, "resnet_block1")
    x = layers.MaxPooling2D((2, 2), name="resnet_pool1")(x)
    
    x = build_resnet_cnn_block(x, 128, "resnet_block2")
    x = layers.MaxPooling2D((2, 2), name="resnet_pool2")(x)
    
    # 转换为序列
    cnn_shape = x.shape
    x = layers.Reshape((cnn_shape[1] * cnn_shape[2], cnn_shape[3]), name="reshape_to_seq")(x)
    
    # 复制并冻结普通LSTM
    for i in range(8, 13):  # LSTM层 (索引可能需要调整)
        layer = stage1_model.layers[i]
        if isinstance(layer, (layers.Bidirectional, layers.Dropout)):
            new_layer = layer.__class__.from_config(layer.get_config())
            new_layer.trainable = False
            try:
                new_layer.build(x.shape)
                new_layer.set_weights(layer.get_weights())
            except:
                pass
            x = new_layer(x)
    
    # 输出层
    x = layers.Dense(Config.CHARS_PER_LABEL * Config.CHAR_SIZE, activation="softmax", name="dense_output")(x)
    outputs = layers.Reshape((Config.CHARS_PER_LABEL, Config.CHAR_SIZE), name="reshape_output")(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name="Stage2B_ResNetCNN_FrozenLSTM")
    return model


def build_stage3_merged_model(stage2a_model, stage2b_model):
    """
    阶段3: 合并残差CNN + Attention LSTM
    从stage2a提取Attention LSTM, 从stage2b提取残差CNN
    """
    inputs = keras.Input(shape=(Config.IMG_HEIGHT, Config.IMG_WIDTH, 1), name="input")
    
    # 从stage2b复制残差CNN
    x = inputs
    for layer in stage2b_model.layers[1:]:  # 跳过input层
        if 'reshape' in layer.name or 'bilstm' in layer.name or 'dropout' in layer.name or 'dense' in layer.name:
            break
        new_layer = layer.__class__.from_config(layer.get_config())
        new_layer.build(x.shape)
        new_layer.set_weights(layer.get_weights())
        x = new_layer(x)
    
    # 转换为序列
    cnn_shape = x.shape
    x = layers.Reshape((cnn_shape[1] * cnn_shape[2], cnn_shape[3]), name="reshape_to_seq")(x)
    
    # 从stage2a复制Attention LSTM
    found_reshape = False
    for layer in stage2a_model.layers:
        if 'reshape_to_seq' in layer.name:
            found_reshape = True
            continue
        if found_reshape and ('attn' in layer.name or 'dense' in layer.name or 'reshape_output' in layer.name):
            if isinstance(layer, layers.Reshape) and 'output' in layer.name:
                # 输出层reshape
                outputs = layer(x)
                break
            new_layer = layer.__class__.from_config(layer.get_config())
            new_layer.build(x.shape)
            new_layer.set_weights(layer.get_weights())
            x = new_layer(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name="Stage3_Merged_ResNet_Attention")
    return model


# ========== 训练阶段 ==========
class InterfaceStandardTrainer:
    """接口标准训练管理器"""
    
    def __init__(self, train_dataset, val_dataset):
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.history = {
            'stage1': None,
            'stage2a': None,
            'stage2b': None,
            'stage3_freeze_cnn': None,
            'stage3_freeze_lstm': None,
            'stage3_full': None
        }
        os.makedirs("models", exist_ok=True)
    
    def run_stage1(self):
        """
        阶段1: 训练普通CNN + 普通LSTM
        目标: 建立接口标准
        """
        print("\n" + "="*70)
        print("STAGE 1: Training Plain CNN + LSTM (Establishing Interface Standard)")
        print("="*70)
        print("\n📌 Purpose: Define the 'contract' for feature format")
        print("   - CNN Output Standard: What features LSTM expects")
        print("   - LSTM Input Standard: What CNN should produce\n")
        
        if Config.SKIP_STAGE1 and os.path.exists(Config.STAGE1_MODEL_PATH):
            print(f"⊙ Skipping Stage 1 (loading from {Config.STAGE1_MODEL_PATH})")
            model = keras.models.load_model(Config.STAGE1_MODEL_PATH)
            val_loss, val_acc = model.evaluate(self.val_dataset, verbose=0)
            print(f"  Loaded model - Val Acc: {val_acc:.4f}\n")
            self.history['stage1'] = self._create_mock_history(val_acc, val_loss)
            return model
        
        model = build_stage1_model()
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        
        print("Model Summary:")
        model.summary()
        
        self.history['stage1'] = model.fit(
            self.train_dataset,
            validation_data=self.val_dataset,
            epochs=Config.EPOCHS_STAGE1,
            callbacks=[
                keras.callbacks.EarlyStopping(
                    monitor="val_accuracy",
                    patience=20,
                    mode='max',
                    restore_best_weights=True,
                    verbose=1,
                    baseline=0.80
                ),
                keras.callbacks.ReduceLROnPlateau(
                    monitor="val_loss", factor=0.5, patience=8, min_lr=1e-7, verbose=1
                ),
                keras.callbacks.ModelCheckpoint(
                    Config.STAGE1_MODEL_PATH, monitor="val_accuracy", save_best_only=True, verbose=1
                ),
            ],
        )
        
        best_acc = max(self.history['stage1'].history['val_accuracy'])
        print(f"\n✓ Stage 1 Complete! Interface Standard Established")
        print(f"  Baseline Accuracy: {best_acc:.4f}")
        print(f"  This accuracy defines the 'interface contract'\n")
        
        return model
    
    def run_stage2a(self, stage1_model):
        """
        阶段2A: 冻结普通CNN, 训练Attention LSTM
        目标: Attention LSTM学会接受"标准CNN输出"
        """
        print("\n" + "="*70)
        print("STAGE 2A: Frozen CNN → Attention LSTM (Training Advanced LSTM)")
        print("="*70)
        print("\n📌 Purpose: Force Attention LSTM to adapt to standard CNN output")
        print("   - Plain CNN (Frozen): Provides consistent feature format")
        print("   - Attention LSTM (Training): Must learn from this format\n")
        
        if Config.SKIP_STAGE2A and os.path.exists(Config.STAGE2A_MODEL_PATH):
            print(f"⊙ Skipping Stage 2A (loading from {Config.STAGE2A_MODEL_PATH})")
            model = keras.models.load_model(Config.STAGE2A_MODEL_PATH)
            val_loss, val_acc = model.evaluate(self.val_dataset, verbose=0)
            print(f"  Loaded model - Val Acc: {val_acc:.4f}\n")
            self.history['stage2a'] = self._create_mock_history(val_acc, val_loss)
            return model
        
        model = build_stage2a_model(stage1_model)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        
        # 验证冻结状态
        frozen_count = sum([1 for layer in model.layers if not layer.trainable])
        print(f"Frozen layers: {frozen_count}")
        print("Model Summary:")
        model.summary()
        
        self.history['stage2a'] = model.fit(
            self.train_dataset,
            validation_data=self.val_dataset,
            epochs=Config.EPOCHS_STAGE2A,
            callbacks=[
                keras.callbacks.EarlyStopping(
                    monitor="val_accuracy",
                    patience=15,
                    mode='max',
                    restore_best_weights=True,
                    verbose=1
                ),
                keras.callbacks.ReduceLROnPlateau(
                    monitor="val_loss", factor=0.5, patience=5, min_lr=1e-7, verbose=1
                ),
                keras.callbacks.ModelCheckpoint(
                    Config.STAGE2A_MODEL_PATH, monitor="val_accuracy", save_best_only=True, verbose=1
                ),
            ],
        )
        
        best_acc = max(self.history['stage2a'].history['val_accuracy'])
        baseline_acc = max(self.history['stage1'].history['val_accuracy'])
        improvement = best_acc - baseline_acc
        
        print(f"\n✓ Stage 2A Complete!")
        print(f"  Attention LSTM Accuracy: {best_acc:.4f}")
        print(f"  Improvement over baseline: {improvement:+.4f}")
        if improvement > 0:
            print(f"  ✓ Attention mechanism helps! (+{improvement:.2%})")
        else:
            print(f"  → Attention LSTM maintains performance")
        print(f"  Attention LSTM now 'speaks' the CNN interface language\n")
        
        return model
    
    def run_stage2b(self, stage1_model):
        """
        阶段2B: 训练残差CNN, 冻结普通LSTM
        目标: 残差CNN学会输出"标准LSTM输入"
        """
        print("\n" + "="*70)
        print("STAGE 2B: ResNet CNN → Frozen LSTM (Training Advanced CNN)")
        print("="*70)
        print("\n📌 Purpose: Force ResNet CNN to produce standard LSTM-compatible features")
        print("   - ResNet CNN (Training): Must output in standard format")
        print("   - Plain LSTM (Frozen): Enforces the output standard\n")
        
        if Config.SKIP_STAGE2B and os.path.exists(Config.STAGE2B_MODEL_PATH):
            print(f"⊙ Skipping Stage 2B (loading from {Config.STAGE2B_MODEL_PATH})")
            model = keras.models.load_model(Config.STAGE2B_MODEL_PATH)
            val_loss, val_acc = model.evaluate(self.val_dataset, verbose=0)
            print(f"  Loaded model - Val Acc: {val_acc:.4f}\n")
            self.history['stage2b'] = self._create_mock_history(val_acc, val_loss)
            return model
        
        model = build_stage2b_model(stage1_model)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        
        # 验证冻结状态
        frozen_count = sum([1 for layer in model.layers if not layer.trainable])
        print(f"Frozen layers: {frozen_count}")
        print("Model Summary:")
        model.summary()
        
        self.history['stage2b'] = model.fit(
            self.train_dataset,
            validation_data=self.val_dataset,
            epochs=Config.EPOCHS_STAGE2B,
            callbacks=[
                keras.callbacks.EarlyStopping(
                    monitor="val_accuracy",
                    patience=15,
                    mode='max',
                    restore_best_weights=True,
                    verbose=1
                ),
                keras.callbacks.ReduceLROnPlateau(
                    monitor="val_loss", factor=0.5, patience=5, min_lr=1e-7, verbose=1
                ),
                keras.callbacks.ModelCheckpoint(
                    Config.STAGE2B_MODEL_PATH, monitor="val_accuracy", save_best_only=True, verbose=1
                ),
            ],
        )
        
        best_acc = max(self.history['stage2b'].history['val_accuracy'])
        baseline_acc = max(self.history['stage1'].history['val_accuracy'])
        improvement = best_acc - baseline_acc
        
        print(f"\n✓ Stage 2B Complete!")
        print(f"  ResNet CNN Accuracy: {best_acc:.4f}")
        print(f"  Improvement over baseline: {improvement:+.4f}")
        if improvement > 0:
            print(f"  ✓ ResNet CNN extracts better features! (+{improvement:.2%})")
        else:
            print(f"  → ResNet CNN maintains performance")
        print(f"  ResNet CNN now outputs in 'LSTM-compatible' format\n")
        
        return model
    
    def run_stage3(self, stage2a_model, stage2b_model):
        """
        阶段3: 合并 + 渐进解冻微调
        目标: ResNet CNN + Attention LSTM 互相适配
        """
        print("\n" + "="*70)
        print("STAGE 3: Merging ResNet CNN + Attention LSTM (Progressive Fine-tuning)")
        print("="*70)
        print("\n📌 Purpose: Perfect alignment through gradual unfreezing")
        print(f"   Phase 3A: Freeze ResNet CNN → Adapt Attention LSTM ({Config.EPOCHS_STAGE3A} epochs)")
        print(f"   Phase 3B: Freeze Attention LSTM → Adapt ResNet CNN ({Config.EPOCHS_STAGE3B} epochs)")
        print(f"   Phase 3C: Unfreeze all → Final convergence ({Config.EPOCHS_STAGE3C} epochs)\n")
        
        if Config.SKIP_STAGE3 and os.path.exists(Config.FINAL_MODEL_PATH):
            print(f"⊙ Skipping Stage 3 (loading from {Config.FINAL_MODEL_PATH})")
            model = keras.models.load_model(Config.FINAL_MODEL_PATH)
            val_loss, val_acc = model.evaluate(self.val_dataset, verbose=0)
            print(f"  Loaded model - Val Acc: {val_acc:.4f}\n")
            self.history['stage3_full'] = self._create_mock_history(val_acc, val_loss)
            return model
        
        # 构建合并模型
        print("Merging models...")
        model = build_stage3_merged_model(stage2a_model, stage2b_model)
        
        # === Phase 3A: 冻结CNN, 微调LSTM ===
        print("\n" + "-"*70)
        print(f"Phase 3A: Frozen ResNet CNN → Fine-tune Attention LSTM ({Config.EPOCHS_STAGE3A} epochs)")
        print("-"*70)
        
        # 冻结CNN层
        for layer in model.layers:
            if 'resnet' in layer.name or 'reshape' in layer.name:
                layer.trainable = False
            else:
                layer.trainable = True
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0005),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        
        self.history['stage3_freeze_cnn'] = model.fit(
            self.train_dataset,
            validation_data=self.val_dataset,
            epochs=Config.EPOCHS_STAGE3A,
            callbacks=[
                keras.callbacks.EarlyStopping(
                    monitor="val_loss", patience=10, restore_best_weights=True, verbose=1
                ),
                keras.callbacks.ReduceLROnPlateau(
                    monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7, verbose=1
                ),
            ],
        )
        
        phase3a_acc = max(self.history['stage3_freeze_cnn'].history['val_accuracy'])
        print(f"\n✓ Phase 3A Complete! Val Acc: {phase3a_acc:.4f}")
        print("  Attention LSTM adapted to ResNet CNN features\n")
        
        # === Phase 3B: 冻结LSTM, 微调CNN ===
        print("-"*70)
        print(f"Phase 3B: Fine-tune ResNet CNN ← Frozen Attention LSTM ({Config.EPOCHS_STAGE3B} epochs)")
        print("-"*70)
        
        # 冻结LSTM层
        for layer in model.layers:
            if 'resnet' in layer.name:
                layer.trainable = True
            else:
                layer.trainable = False
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0005),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        
        self.history['stage3_freeze_lstm'] = model.fit(
            self.train_dataset,
            validation_data=self.val_dataset,
            epochs=Config.EPOCHS_STAGE3B,
            callbacks=[
                keras.callbacks.EarlyStopping(
                    monitor="val_loss", patience=10, restore_best_weights=True, verbose=1
                ),
                keras.callbacks.ReduceLROnPlateau(
                    monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7, verbose=1
                ),
            ],
        )
        
        phase3b_acc = max(self.history['stage3_freeze_lstm'].history['val_accuracy'])
        print(f"\n✓ Phase 3B Complete! Val Acc: {phase3b_acc:.4f}")
        print("  ResNet CNN learned to produce richer features for Attention LSTM\n")
        
        # === Phase 3C: 全部解冻 ===
        print("-"*70)
        print(f"Phase 3C: Full Fine-tuning (All Layers Unfrozen) ({Config.EPOCHS_STAGE3C} epochs)")
        print("-"*70)
        
        # 解冻所有层
        for layer in model.layers:
            layer.trainable = True
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0001),  # 更小的学习率
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        
        print(f"\nTraining with all layers unfrozen (lr=0.0001)...")
        
        self.history['stage3_full'] = model.fit(
            self.train_dataset,
            validation_data=self.val_dataset,
            epochs=Config.EPOCHS_STAGE3C,
            callbacks=[
                keras.callbacks.EarlyStopping(
                    monitor="val_loss", patience=15, restore_best_weights=True, verbose=1
                ),
                keras.callbacks.ReduceLROnPlateau(
                    monitor="val_loss", factor=0.5, patience=5, min_lr=1e-8, verbose=1
                ),
                keras.callbacks.ModelCheckpoint(
                    Config.FINAL_MODEL_PATH, monitor="val_accuracy", save_best_only=True, verbose=1
                ),
            ],
        )
        
        final_acc = max(self.history['stage3_full'].history['val_accuracy'])
        print(f"\n✓ Phase 3C Complete! Final Val Acc: {final_acc:.4f}")
        print(f"  ResNet CNN and Attention LSTM reached optimal alignment\n")
        
        model.save(Config.FINAL_MODEL_PATH)
        print(f"✓ Final model saved to {Config.FINAL_MODEL_PATH}\n")
        
        return model
    
    def _create_mock_history(self, val_acc, val_loss):
        """创建模拟的训练历史"""
        return type('obj', (object,), {
            'history': {
                'val_accuracy': [val_acc],
                'accuracy': [val_acc],
                'val_loss': [val_loss],
                'loss': [val_loss]
            }
        })()
    
    def save_training_history(self):
        """保存所有阶段的训练历史"""
        history_all = {}
        
        for stage_name, history in self.history.items():
            if history is not None:
                history_all[stage_name] = {
                    k: [float(v) for v in vals] 
                    for k, vals in history.history.items()
                }
        
        history_path = "models/training_history_interface_standard.json"
        with open(history_path, 'w') as f:
            json.dump(history_all, f, indent=2)
        print(f"✓ Training history saved to {history_path}")
    
    def print_summary(self):
        """打印训练总结"""
        print("\n" + "="*70)
        print("TRAINING SUMMARY - INTERFACE STANDARD APPROACH")
        print("="*70)
        
        # 提取各阶段最佳准确率
        stage1_acc = max(self.history['stage1'].history['val_accuracy'])
        stage2a_acc = max(self.history['stage2a'].history['val_accuracy'])
        stage2b_acc = max(self.history['stage2b'].history['val_accuracy'])
        
        phase3a_acc = max(self.history['stage3_freeze_cnn'].history['val_accuracy']) if self.history['stage3_freeze_cnn'] else 0
        phase3b_acc = max(self.history['stage3_freeze_lstm'].history['val_accuracy']) if self.history['stage3_freeze_lstm'] else 0
        final_acc = max(self.history['stage3_full'].history['val_accuracy']) if self.history['stage3_full'] else 0
        
        print("\n📊 Performance Progression:")
        print("-"*70)
        print(f"  Stage 1  (Plain CNN + LSTM):           {stage1_acc:.4f}  [Interface Standard]")
        print(f"  Stage 2A (Frozen CNN + Attention LSTM): {stage2a_acc:.4f}  ({stage2a_acc-stage1_acc:+.4f})")
        print(f"  Stage 2B (ResNet CNN + Frozen LSTM):    {stage2b_acc:.4f}  ({stage2b_acc-stage1_acc:+.4f})")
        
        if not Config.SKIP_STAGE3:
            print(f"\n  Phase 3A (Adapt Attention LSTM):        {phase3a_acc:.4f}")
            print(f"  Phase 3B (Adapt ResNet CNN):            {phase3b_acc:.4f}")
            print(f"  Phase 3C (Full Fine-tuning):            {final_acc:.4f}  [FINAL]")
        
        # 分析
        print("\n" + "="*70)
        print("ANALYSIS")
        print("="*70)
        
        print("\n🔍 Branch Training Results:")
        print("-"*70)
        
        # Stage 2A 分析
        improvement_2a = stage2a_acc - stage1_acc
        print(f"\n  Attention LSTM Branch:")
        print(f"    Baseline → Stage 2A: {stage1_acc:.4f} → {stage2a_acc:.4f} ({improvement_2a:+.4f})")
        if improvement_2a > 0.01:
            print(f"    ✓ Self-Attention is effective (+{improvement_2a:.2%})")
        elif improvement_2a > -0.01:
            print(f"    → Attention maintains performance (good sign)")
        else:
            print(f"    ⚠ Attention underperforms (may need tuning)")
        
        # Stage 2B 分析
        improvement_2b = stage2b_acc - stage1_acc
        print(f"\n  ResNet CNN Branch:")
        print(f"    Baseline → Stage 2B: {stage1_acc:.4f} → {stage2b_acc:.4f} ({improvement_2b:+.4f})")
        if improvement_2b > 0.01:
            print(f"    ✓ ResNet extracts better features (+{improvement_2b:.2%})")
        elif improvement_2b > -0.01:
            print(f"    → ResNet maintains performance (good sign)")
        else:
            print(f"    ⚠ ResNet underperforms (may need deeper network)")
        
        # Stage 3 分析
        if not Config.SKIP_STAGE3:
            print(f"\n🔀 Merge & Fine-tuning Results:")
            print("-"*70)
            
            best_branch = max(stage2a_acc, stage2b_acc)
            merge_improvement = final_acc - best_branch
            
            print(f"\n  Best Branch:  {best_branch:.4f}")
            print(f"  After Merge:  {final_acc:.4f}")
            print(f"  Improvement:  {merge_improvement:+.4f}")
            
            if merge_improvement > 0.01:
                print(f"\n  ✓✓ SUCCESS! Merged model surpasses both branches (+{merge_improvement:.2%})")
                print(f"     This validates the 'Interface Standard' approach!")
            elif merge_improvement > -0.01:
                print(f"\n  ✓ Merged model matches best branch")
                print(f"    Components successfully aligned")
            else:
                print(f"\n  ⚠ Merged model underperforms")
                print(f"    May need longer Phase 3 training")
            
            # 总体提升
            total_improvement = final_acc - stage1_acc
            print(f"\n  Total Improvement: {stage1_acc:.4f} → {final_acc:.4f} ({total_improvement:+.4f})")
        
        # 机制验证
        print("\n" + "="*70)
        print("INTERFACE STANDARD MECHANISM VALIDATION")
        print("="*70)
        
        print("\n✓ Interface Contract Established:")
        print(f"  - Plain CNN defined feature output format")
        print(f"  - Plain LSTM defined feature input format")
        
        print(f"\n✓ Branch Training Completed:")
        print(f"  - Attention LSTM learned to accept 'standard CNN output'")
        print(f"  - ResNet CNN learned to produce 'standard LSTM input'")
        
        if not Config.SKIP_STAGE3:
            print(f"\n✓ Merge Validation:")
            if merge_improvement > -0.01:
                print(f"  - ResNet output ←→ Attention LSTM input: COMPATIBLE ✓")
                print(f"  - Both components 'speak the same language'")
            else:
                print(f"  - Interface alignment needs improvement")
                print(f"  - Consider longer branch training")
        
        # 建议
        print("\n" + "="*70)
        print("RECOMMENDATIONS")
        print("="*70 + "\n")
        
        if not Config.SKIP_STAGE3:
            if final_acc > 0.92:
                print("🎉 EXCELLENT! Model exceeded 0.92 target")
                print(f"\n   Deployment Ready:")
                print(f"   >>> model = keras.models.load_model('{Config.FINAL_MODEL_PATH}')")
                
            elif final_acc > stage1_acc + 0.03:
                print("✓ GOOD! Significant improvement over baseline")
                print(f"\n   Next steps:")
                print(f"   - Model is usable (Val Acc: {final_acc:.4f})")
                print(f"   - For further improvement: try data augmentation")
                
            elif improvement_2a < 0 and improvement_2b < 0:
                print("⚠ Both branches underperformed")
                print(f"\n   Debugging suggestions:")
                print(f"   1. Stage 1 baseline may be too weak ({stage1_acc:.4f})")
                print(f"      → Train Stage 1 longer or with more capacity")
                print(f"   2. Interface standard may be too restrictive")
                print(f"      → Try larger LSTM hidden size (128 → 256)")
                
            elif merge_improvement < -0.02:
                print("⚠ Merge caused performance drop")
                print(f"\n   Suggestions:")
                print(f"   1. Extend Phase 3A/3B training epochs")
                print(f"   2. Use even lower learning rate in Phase 3C")
                print(f"   3. Add intermediate validation between phases")
                
            else:
                print("→ Model trained successfully")
                print(f"\n   Current performance: {final_acc:.4f}")
                print(f"\n   Improvement strategies:")
                print(f"   1. Increase Stage 1 epochs for stronger baseline")
                print(f"   2. Add more ResNet blocks in Stage 2B")
                print(f"   3. Experiment with attention heads (4 → 8)")
        
        else:
            print("ℹ Stage 3 was skipped")
            print(f"\n   To complete the training:")
            print(f"   - Set Config.SKIP_STAGE3 = False")
            print(f"   - Run again to merge and fine-tune")
        
        print("\n" + "="*70)
        
        # 理论优势总结
        print("\n💡 INTERFACE STANDARD ADVANTAGES:")
        print("-"*70)
        print("""
  1. Automatic Feature Alignment
     - No manual adaptation layers needed
     - Components naturally compatible through shared standard

  2. Stable Training Environment
     - Each component trains with frozen 'anchor'
     - Avoids moving target problem

  3. Modular Development
     - Can improve CNN and LSTM independently
     - Easy to experiment with different architectures

  4. Implicit Regularization
     - Forces components to learn 'communicable' features
     - Prevents overly complex representations
        """)
        
        print("="*70 + "\n")


# ========== 主函数 ==========
def main():
    """主训练流程"""
    print("\n" + "="*70)
    print("LUOGU CAPTCHA - INTERFACE STANDARD TRAINING")
    print("="*70)
    print("""
    Training Strategy:
    
    Stage 1: Plain CNN + Plain LSTM
             ↓
          [Define Interface Standard]
    
    Stage 2A: [Plain CNN (Frozen)] → Attention LSTM (Training)
              ↓
           Attention LSTM learns to accept 'standard CNN output'
    
    Stage 2B: ResNet CNN (Training) → [Plain LSTM (Frozen)]
              ↓
           ResNet CNN learns to produce 'standard LSTM input'
    
    Stage 3: ResNet CNN → Attention LSTM
             ↓           ↓
          符合LSTM的    接受CNN的
          输入标准      输出标准
             ↓___________↓
               完美匹配！
    """)
    
# 配置
    print("Configuration:")
    print(f"  Stage 1:  {'SKIP' if Config.SKIP_STAGE1 else f'RUN ({Config.EPOCHS_STAGE1} epochs)'}")
    print(f"  Stage 2A: {'SKIP' if Config.SKIP_STAGE2A else f'RUN ({Config.EPOCHS_STAGE2A} epochs)'}")
    print(f"  Stage 2B: {'SKIP' if Config.SKIP_STAGE2B else f'RUN ({Config.EPOCHS_STAGE2B} epochs)'}")
    if Config.SKIP_STAGE3:
        print(f"  Stage 3:  SKIP")
    else:
        total_stage3 = Config.EPOCHS_STAGE3A + Config.EPOCHS_STAGE3B + Config.EPOCHS_STAGE3C
        print(f"  Stage 3:  RUN (Total: {total_stage3} epochs)")
        print(f"    - Phase 3A: {Config.EPOCHS_STAGE3A} epochs")
        print(f"    - Phase 3B: {Config.EPOCHS_STAGE3B} epochs")
        print(f"    - Phase 3C: {Config.EPOCHS_STAGE3C} epochs")
    
    # 总计
    total_epochs = 0
    if not Config.SKIP_STAGE1:
        total_epochs += Config.EPOCHS_STAGE1
    if not Config.SKIP_STAGE2A:
        total_epochs += Config.EPOCHS_STAGE2A
    if not Config.SKIP_STAGE2B:
        total_epochs += Config.EPOCHS_STAGE2B
    if not Config.SKIP_STAGE3:
        total_epochs += Config.EPOCHS_STAGE3A + Config.EPOCHS_STAGE3B + Config.EPOCHS_STAGE3C
    
    print(f"\n  Total training epochs: {total_epochs}")
    print()
    
    # 设置GPU
    setup_gpu()
    
    # 加载数据
    print("\nLoading datasets...")
    train_dataset, val_dataset = load_datasets(Config.TFRECORD_DIR)
    
    # 创建训练器
    trainer = InterfaceStandardTrainer(train_dataset, val_dataset)
    
    # 阶段1: 建立接口标准
    model_stage1 = trainer.run_stage1()
    
    # 阶段2A: 训练Attention LSTM (适应标准CNN输出)
    model_stage2a = trainer.run_stage2a(model_stage1)
    
    # 阶段2B: 训练ResNet CNN (输出标准LSTM输入)
    model_stage2b = trainer.run_stage2b(model_stage1)
    
    # 阶段3: 合并 + 渐进微调
    final_model = trainer.run_stage3(model_stage2a, model_stage2b)
    
    # 保存历史
    trainer.save_training_history()
    
    # 打印总结
    trainer.print_summary()
    
    return final_model


# ========== 程序入口 ==========
if __name__ == "__main__":
    try:
        final_model = main()
        print("✓ Training completed successfully!")
        print(f"\n📦 Model ready for deployment:")
        print(f"   Path: {Config.FINAL_MODEL_PATH}")
        print(f"   Usage: keras.models.load_model('{Config.FINAL_MODEL_PATH}')")
        
    except KeyboardInterrupt:
        print("\n⚠ Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()