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

# ========== 配置 ==========
class Config:
    # 数据参数
    CHAR_SIZE = 256
    CHARS_PER_LABEL = 4
    IMG_HEIGHT, IMG_WIDTH = 35, 90
    BATCH_SIZE = 256
    TFRECORD_DIR = "data/luogu_captcha_tfrecord"
    
    # 训练参数
    EPOCHS_STAGE1 = 100    # 基线模型
    EPOCHS_STAGE2A = 30    # Attention LSTM分支
    EPOCHS_STAGE2B = 30    # 残差CNN分支
    EPOCHS_STAGE3A = 15    # 微调LSTM
    EPOCHS_STAGE3B = 15    # 微调CNN
    EPOCHS_STAGE3C = 50    # 联合微调
    
    # 模型路径
    STAGE1_MODEL_PATH = "models/stage1_simple_cnn_bilstm.keras"
    STAGE2A_MODEL_PATH = "models/stage2a_simple_cnn_attention_lstm.keras"
    STAGE2B_MODEL_PATH = "models/stage2b_residual_cnn_simple_lstm.keras"
    STAGE3A_MODEL_PATH = "models/stage3a_merged_lstm_tuned.keras"
    STAGE3B_MODEL_PATH = "models/stage3b_merged_cnn_tuned.keras"
    FINAL_MODEL_PATH = "models/luoguCaptcha_final_branch_merged.keras"
    
    # 控制开关
    SKIP_STAGE1 = False   # 阶段1: SimpleCNN + SimpleLSTM
    SKIP_STAGE2A = False  # 阶段2A: SimpleCNN(冻结) + AttentionLSTM
    SKIP_STAGE2B = False  # 阶段2B: ResidualCNN + SimpleLSTM(冻结)
    SKIP_STAGE3A = False  # 阶段3A: ResidualCNN(冻结) + AttentionLSTM
    SKIP_STAGE3B = False  # 阶段3B: ResidualCNN + AttentionLSTM(冻结)
    SKIP_STAGE3C = False  # 阶段3C: 全部解冻


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


# ========== 模型构建模块 ==========

class CNNBuilder:
    """CNN构建器：提供SimpleCNN和ResidualCNN"""
    
    @staticmethod
    def build_simple_cnn(inputs, name_prefix="simple_cnn"):
        """构建简单CNN（阶段1使用）"""
        x = layers.Conv2D(64, 3, padding="same", activation="relu", 
                         name=f"{name_prefix}_conv1")(inputs)
        x = layers.BatchNormalization(name=f"{name_prefix}_bn1")(x)
        x = layers.MaxPooling2D((2, 2), name=f"{name_prefix}_pool1")(x)

        x = layers.Conv2D(128, 3, padding="same", activation="relu", 
                         name=f"{name_prefix}_conv2")(x)
        x = layers.BatchNormalization(name=f"{name_prefix}_bn2")(x)
        x = layers.MaxPooling2D((2, 2), name=f"{name_prefix}_pool2")(x)
        
        return x
    
    @staticmethod
    def build_residual_cnn(inputs, name_prefix="residual_cnn"):
        """构建残差CNN（阶段2B使用）"""
        # 初始卷积
        x = layers.Conv2D(64, 3, padding="same", name=f"{name_prefix}_stem_conv")(inputs)
        x = layers.BatchNormalization(name=f"{name_prefix}_stem_bn")(x)
        x = layers.Activation("relu", name=f"{name_prefix}_stem_relu")(x)
        
        # 残差块1
        shortcut = x
        x = layers.Conv2D(64, 3, padding="same", name=f"{name_prefix}_res1_conv1")(x)
        x = layers.BatchNormalization(name=f"{name_prefix}_res1_bn1")(x)
        x = layers.Activation("relu", name=f"{name_prefix}_res1_relu1")(x)
        x = layers.Conv2D(64, 3, padding="same", name=f"{name_prefix}_res1_conv2")(x)
        x = layers.BatchNormalization(name=f"{name_prefix}_res1_bn2")(x)
        x = layers.Add(name=f"{name_prefix}_res1_add")([shortcut, x])
        x = layers.Activation("relu", name=f"{name_prefix}_res1_relu2")(x)
        x = layers.MaxPooling2D((2, 2), name=f"{name_prefix}_res1_pool")(x)
        
        # 残差块2
        shortcut = layers.Conv2D(128, 1, padding="same", 
                                name=f"{name_prefix}_res2_shortcut")(x)
        x = layers.Conv2D(128, 3, padding="same", name=f"{name_prefix}_res2_conv1")(x)
        x = layers.BatchNormalization(name=f"{name_prefix}_res2_bn1")(x)
        x = layers.Activation("relu", name=f"{name_prefix}_res2_relu1")(x)
        x = layers.Conv2D(128, 3, padding="same", name=f"{name_prefix}_res2_conv2")(x)
        x = layers.BatchNormalization(name=f"{name_prefix}_res2_bn2")(x)
        x = layers.Add(name=f"{name_prefix}_res2_add")([shortcut, x])
        x = layers.Activation("relu", name=f"{name_prefix}_res2_relu2")(x)
        x = layers.MaxPooling2D((2, 2), name=f"{name_prefix}_res2_pool")(x)
        
        return x


class LSTMBuilder:
    """LSTM构建器：提供SimpleLSTM和AttentionLSTM"""
    
    @staticmethod
    def build_simple_lstm(sequence_input, name_prefix="simple_lstm"):
        """构建简单BiLSTM（阶段1使用）"""
        x = layers.Bidirectional(
            layers.LSTM(128, return_sequences=True), 
            name=f"{name_prefix}_bilstm1"
        )(sequence_input)
        x = layers.Dropout(0.3, name=f"{name_prefix}_dropout1")(x)
        x = layers.Bidirectional(
            layers.LSTM(128, return_sequences=False), 
            name=f"{name_prefix}_bilstm2"
        )(x)
        x = layers.Dropout(0.3, name=f"{name_prefix}_dropout2")(x)
        return x
    
    @staticmethod
    def build_attention_lstm(sequence_input, name_prefix="attention_lstm"):
        """构建带Self-Attention的BiLSTM（阶段2A使用）"""
        # Self-Attention
        attention_output = layers.MultiHeadAttention(
            num_heads=4, key_dim=64, dropout=0.1, 
            name=f"{name_prefix}_self_attention"
        )(sequence_input, sequence_input)
        x = layers.LayerNormalization(name=f"{name_prefix}_attn_norm")(
            sequence_input + attention_output
        )
        
        # BiLSTM
        x = layers.Bidirectional(
            layers.LSTM(128, return_sequences=True), 
            name=f"{name_prefix}_bilstm1"
        )(x)
        x = layers.Dropout(0.3, name=f"{name_prefix}_dropout1")(x)
        x = layers.Bidirectional(
            layers.LSTM(128, return_sequences=False), 
            name=f"{name_prefix}_bilstm2"
        )(x)
        x = layers.Dropout(0.3, name=f"{name_prefix}_dropout2")(x)
        return x


# ========== 完整模型构建 ==========

def build_stage1_model():
    """阶段1: SimpleCNN + SimpleLSTM"""
    inputs = keras.Input(shape=(Config.IMG_HEIGHT, Config.IMG_WIDTH, 1), name="input")
    
    # CNN部分
    cnn_output = CNNBuilder.build_simple_cnn(inputs, "stage1_cnn")
    
    # 转换为序列
    cnn_shape = cnn_output.shape
    sequence = layers.Reshape(
        (cnn_shape[1] * cnn_shape[2], cnn_shape[3]), 
        name="stage1_reshape"
    )(cnn_output)
    
    # LSTM部分
    lstm_output = LSTMBuilder.build_simple_lstm(sequence, "stage1_lstm")
    
    # 输出层
    x = layers.Dense(
        Config.CHARS_PER_LABEL * Config.CHAR_SIZE, 
        activation="softmax", 
        name="stage1_dense"
    )(lstm_output)
    outputs = layers.Reshape(
        (Config.CHARS_PER_LABEL, Config.CHAR_SIZE), 
        name="stage1_output"
    )(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name="Stage1_SimpleCNN_SimpleLSTM")
    return model


def build_stage2a_model(stage1_model):
    """阶段2A: SimpleCNN(冻结) + AttentionLSTM(训练)"""
    inputs = keras.Input(shape=(Config.IMG_HEIGHT, Config.IMG_WIDTH, 1), name="input")
    
    # 复制CNN部分（冻结）
    cnn_output = CNNBuilder.build_simple_cnn(inputs, "stage2a_cnn_frozen")
    
    # 转换为序列
    cnn_shape = cnn_output.shape
    sequence = layers.Reshape(
        (cnn_shape[1] * cnn_shape[2], cnn_shape[3]), 
        name="stage2a_reshape"
    )(cnn_output)
    
    # 新的Attention LSTM
    lstm_output = LSTMBuilder.build_attention_lstm(sequence, "stage2a_lstm_new")
    
    # 输出层
    x = layers.Dense(
        Config.CHARS_PER_LABEL * Config.CHAR_SIZE, 
        activation="softmax", 
        name="stage2a_dense"
    )(lstm_output)
    outputs = layers.Reshape(
        (Config.CHARS_PER_LABEL, Config.CHAR_SIZE), 
        name="stage2a_output"
    )(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name="Stage2A_SimpleCNN_AttentionLSTM")
    
    # 复制CNN权重
    print("\n📋 Copying CNN weights from Stage 1...")
    _copy_cnn_weights(stage1_model, model, "stage1_cnn", "stage2a_cnn_frozen")
    
    # 冻结CNN层
    print("\n🔒 Freezing CNN layers...")
    _freeze_layers_by_prefix(model, "stage2a_cnn_frozen")
    
    return model


def build_stage2b_model(stage1_model):
    """阶段2B: ResidualCNN(训练) + SimpleLSTM(冻结)"""
    inputs = keras.Input(shape=(Config.IMG_HEIGHT, Config.IMG_WIDTH, 1), name="input")
    
    # 新的残差CNN
    cnn_output = CNNBuilder.build_residual_cnn(inputs, "stage2b_cnn_new")
    
    # 转换为序列
    cnn_shape = cnn_output.shape
    sequence = layers.Reshape(
        (cnn_shape[1] * cnn_shape[2], cnn_shape[3]), 
        name="stage2b_reshape"
    )(cnn_output)
    
    # 复制LSTM部分（冻结）
    lstm_output = LSTMBuilder.build_simple_lstm(sequence, "stage2b_lstm_frozen")
    
    # 输出层
    x = layers.Dense(
        Config.CHARS_PER_LABEL * Config.CHAR_SIZE, 
        activation="softmax", 
        name="stage2b_dense"
    )(lstm_output)
    outputs = layers.Reshape(
        (Config.CHARS_PER_LABEL, Config.CHAR_SIZE), 
        name="stage2b_output"
    )(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name="Stage2B_ResidualCNN_SimpleLSTM")
    
    # 复制LSTM权重
    print("\n📋 Copying LSTM weights from Stage 1...")
    _copy_lstm_weights(stage1_model, model, "stage1_lstm", "stage2b_lstm_frozen")
    
    # 冻结LSTM层
    print("\n🔒 Freezing LSTM layers...")
    _freeze_layers_by_prefix(model, "stage2b_lstm_frozen")
    
    return model


def build_merged_model(stage2a_model, stage2b_model):
    """合并模型: ResidualCNN + AttentionLSTM"""
    inputs = keras.Input(shape=(Config.IMG_HEIGHT, Config.IMG_WIDTH, 1), name="input")
    
    # 从2B提取ResidualCNN
    cnn_output = CNNBuilder.build_residual_cnn(inputs, "merged_cnn")
    
    # 转换为序列
    cnn_shape = cnn_output.shape
    sequence = layers.Reshape(
        (cnn_shape[1] * cnn_shape[2], cnn_shape[3]), 
        name="merged_reshape"
    )(cnn_output)
    
    # 从2A提取AttentionLSTM
    lstm_output = LSTMBuilder.build_attention_lstm(sequence, "merged_lstm")
    
    # 输出层
    x = layers.Dense(
        Config.CHARS_PER_LABEL * Config.CHAR_SIZE, 
        activation="softmax", 
        name="merged_dense"
    )(lstm_output)
    outputs = layers.Reshape(
        (Config.CHARS_PER_LABEL, Config.CHAR_SIZE), 
        name="merged_output"
    )(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name="Merged_ResidualCNN_AttentionLSTM")
    
    # 复制权重
    print("\n🔀 Merging models...")
    print("  CNN:  from Stage 2B (ResidualCNN)")
    _copy_cnn_weights(stage2b_model, model, "stage2b_cnn_new", "merged_cnn")
    
    print("  LSTM: from Stage 2A (AttentionLSTM)")
    _copy_lstm_weights(stage2a_model, model, "stage2a_lstm_new", "merged_lstm")
    
    print("✓ Model merge completed!\n")
    
    return model


# ========== 权重复制工具 ==========

def _copy_cnn_weights(source_model, target_model, source_prefix, target_prefix):
    """复制CNN权重"""
    copied = 0
    for src_layer in source_model.layers:
        if src_layer.name.startswith(source_prefix):
            target_name = src_layer.name.replace(source_prefix, target_prefix)
            try:
                target_layer = target_model.get_layer(target_name)
                if src_layer.get_weights():
                    target_layer.set_weights(src_layer.get_weights())
                    copied += 1
                    print(f"  ✓ {src_layer.name} → {target_name}")
            except ValueError:
                print(f"  ⚠ Layer not found: {target_name}")
    
    print(f"\n  Total: {copied} CNN layers copied")


def _copy_lstm_weights(source_model, target_model, source_prefix, target_prefix):
    """复制LSTM权重"""
    copied = 0
    for src_layer in source_model.layers:
        if src_layer.name.startswith(source_prefix):
            target_name = src_layer.name.replace(source_prefix, target_prefix)
            try:
                target_layer = target_model.get_layer(target_name)
                if src_layer.get_weights():
                    target_layer.set_weights(src_layer.get_weights())
                    copied += 1
                    print(f"  ✓ {src_layer.name} → {target_name}")
            except ValueError:
                print(f"  ⚠ Layer not found: {target_name}")
    
    print(f"\n  Total: {copied} LSTM layers copied")


def _freeze_layers_by_prefix(model, prefix):
    """冻结指定前缀的层"""
    frozen_count = 0
    for layer in model.layers:
        if layer.name.startswith(prefix):
            layer.trainable = False
            frozen_count += 1
            print(f"  🔒 {layer.name}")
    
    print(f"\n  Total: {frozen_count} layers frozen")


def _unfreeze_layers_by_prefix(model, prefix):
    """解冻指定前缀的层"""
    unfrozen_count = 0
    for layer in model.layers:
        if layer.name.startswith(prefix):
            layer.trainable = True
            unfrozen_count += 1
            print(f"  🔓 {layer.name}")
    
    print(f"\n  Total: {unfrozen_count} layers unfrozen")


def _print_trainable_summary(model):
    """打印可训练参数统计"""
    trainable_count = sum([tf.size(w).numpy() for w in model.trainable_weights])
    non_trainable_count = sum([tf.size(w).numpy() for w in model.non_trainable_weights])
    total_count = trainable_count + non_trainable_count
    
    print("\n" + "="*60)
    print("MODEL PARAMETER SUMMARY")
    print("="*60)
    print(f"Trainable:     {trainable_count:>12,} ({trainable_count/total_count*100:>5.1f}%)")
    print(f"Non-trainable: {non_trainable_count:>12,} ({non_trainable_count/total_count*100:>5.1f}%)")
    print(f"Total:         {total_count:>12,}")
    print("="*60 + "\n")


# ========== 训练阶段管理器 ==========

class BranchMergeTrainer:
    """分支合并训练管理器"""
    
    def __init__(self, train_dataset, val_dataset):
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        
        # 训练历史
        self.history = {
            'stage1': None,
            'stage2a': None,
            'stage2b': None,
            'stage3a': None,
            'stage3b': None,
            'stage3c': None,
        }
        
        os.makedirs("models", exist_ok=True)
    
    def run_full_pipeline(self):
        """运行完整训练流程"""
        print("\n" + "="*80)
        print("🚀 BRANCH-MERGE TRAINING PIPELINE")
        print("="*80 + "\n")
        
        # 阶段1
        model_stage1 = self.run_stage1()
        
        # 阶段2A
        model_stage2a = self.run_stage2a(model_stage1)
        
        # 阶段2B
        model_stage2b = self.run_stage2b(model_stage1)
        
        # 合并
        merged_model = self.merge_models(model_stage2a, model_stage2b)
        
        # 阶段3A
        merged_model = self.run_stage3a(merged_model)
        
        # 阶段3B
        merged_model = self.run_stage3b(merged_model)
        
        # 阶段3C
        final_model = self.run_stage3c(merged_model)
        
        # 保存历史和总结
        self.save_training_history()
        self.print_summary()
        
        return final_model
    
    def run_stage1(self):
        """阶段1: SimpleCNN + SimpleLSTM"""
        print("\n" + "🔵"*40)
        print("STAGE 1: Training SimpleCNN + SimpleLSTM (Baseline)")
        print("🔵"*40 + "\n")
        
        if Config.SKIP_STAGE1 and os.path.exists(Config.STAGE1_MODEL_PATH):
            print(f"⊙ Skipping Stage 1 (loading from {Config.STAGE1_MODEL_PATH})")
            model = keras.models.load_model(Config.STAGE1_MODEL_PATH)
            val_loss, val_acc = model.evaluate(self.val_dataset, verbose=0)
            print(f"  Validation accuracy: {val_acc:.4f}")
            self.history['stage1'] = self._create_dummy_history(val_acc, val_loss)
            return model
        
        # 构建并训练
        model = build_stage1_model()
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        
        print("Model Summary:")
        model.summary()
        _print_trainable_summary(model)
        
        history = model.fit(
            self.train_dataset,
            validation_data=self.val_dataset,
            epochs=Config.EPOCHS_STAGE1,
            callbacks=[
                keras.callbacks.EarlyStopping(
                    monitor="val_accuracy", patience=20, mode='max',
                    restore_best_weights=True, verbose=1, baseline=0.80
                ),
                keras.callbacks.ReduceLROnPlateau(
                    monitor="val_loss", factor=0.5, patience=8, 
                    min_lr=1e-7, verbose=1
                ),
                keras.callbacks.ModelCheckpoint(
                    Config.STAGE1_MODEL_PATH, monitor="val_accuracy", 
                    save_best_only=True, verbose=1
                ),
            ],
        )
        
        self.history['stage1'] = history
        best_acc = max(history.history['val_accuracy'])
        print(f"\n✅ Stage 1 completed! Best val_acc: {best_acc:.4f}\n")
        
        return model
    
    def run_stage2a(self, stage1_model):
        """阶段2A: SimpleCNN(冻结) + AttentionLSTM(训练)"""
        print("\n" + "🟢"*40)
        print("STAGE 2A: Training AttentionLSTM (SimpleCNN Frozen)")
        print("🟢"*40 + "\n")
        
        if Config.SKIP_STAGE2A and os.path.exists(Config.STAGE2A_MODEL_PATH):
            print(f"⊙ Skipping Stage 2A (loading from {Config.STAGE2A_MODEL_PATH})")
            model = keras.models.load_model(Config.STAGE2A_MODEL_PATH)
            val_loss, val_acc = model.evaluate(self.val_dataset, verbose=0)
            print(f"  Validation accuracy: {val_acc:.4f}")
            self.history['stage2a'] = self._create_dummy_history(val_acc, val_loss)
            return model
        
        # 构建模型
        model = build_stage2a_model(stage1_model)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        
        _print_trainable_summary(model)
        
        history = model.fit(
            self.train_dataset,
            validation_data=self.val_dataset,
            epochs=Config.EPOCHS_STAGE2A,
            callbacks=[
                keras.callbacks.ReduceLROnPlateau(
                    monitor="val_loss", factor=0.5, patience=5, 
                    min_lr=1e-6, verbose=1
                ),
                keras.callbacks.ModelCheckpoint(
                    Config.STAGE2A_MODEL_PATH, monitor="val_accuracy", 
                    save_best_only=True, verbose=1
                ),
            ],
        )
        
        self.history['stage2a'] = history
        best_acc = max(history.history['val_accuracy'])
        print(f"\n✅ Stage 2A completed! Best val_acc: {best_acc:.4f}\n")
        
        return model
    
    def run_stage2b(self, stage1_model):
        """阶段2B: ResidualCNN(训练) + SimpleLSTM(冻结)"""
        print("\n" + "🟡"*40)
        print("STAGE 2B: Training ResidualCNN (SimpleLSTM Frozen)")
        print("🟡"*40 + "\n")
        
        if Config.SKIP_STAGE2B and os.path.exists(Config.STAGE2B_MODEL_PATH):
            print(f"⊙ Skipping Stage 2B (loading from {Config.STAGE2B_MODEL_PATH})")
            model = keras.models.load_model(Config.STAGE2B_MODEL_PATH)
            val_loss, val_acc = model.evaluate(self.val_dataset, verbose=0)
            print(f"  Validation accuracy: {val_acc:.4f}")
            self.history['stage2b'] = self._create_dummy_history(val_acc, val_loss)
            return model
        
        # 构建模型
        model = build_stage2b_model(stage1_model)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        
        _print_trainable_summary(model)
        
        history = model.fit(
            self.train_dataset,
            validation_data=self.val_dataset,
            epochs=Config.EPOCHS_STAGE2B,
            callbacks=[
                keras.callbacks.ReduceLROnPlateau(
                    monitor="val_loss", factor=0.5, patience=5, 
                    min_lr=1e-6, verbose=1
                ),
                keras.callbacks.ModelCheckpoint(
                    Config.STAGE2B_MODEL_PATH, monitor="val_accuracy", 
                    save_best_only=True, verbose=1
                ),
            ],
        )
        
        self.history['stage2b'] = history
        best_acc = max(history.history['val_accuracy'])
        print(f"\n✅ Stage 2B completed! Best val_acc: {best_acc:.4f}\n")
        
        return model
    
    def merge_models(self, model_2a, model_2b):
        """合并两个分支模型"""
        print("\n" + "🟣"*40)
        print("MERGE STAGE: Combining ResidualCNN + AttentionLSTM")
        print("🟣"*40 + "\n")
        
        merged_model = build_merged_model(model_2a, model_2b)
        
        # 初步评估（不训练）
        merged_model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0001),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        
        print("Evaluating merged model (before fine-tuning)...")
        val_loss, val_acc = merged_model.evaluate(self.val_dataset, verbose=0)
        print(f"  Merged model accuracy: {val_acc:.4f}")
        print(f"  (Expected to be similar to Stage 2A/2B baseline)\n")
        
        return merged_model
    
    def run_stage3a(self, merged_model):
        """阶段3A: ResidualCNN(冻结) + AttentionLSTM(微调)"""
        print("\n" + "🟠"*40)
        print("STAGE 3A: Fine-tuning AttentionLSTM (ResidualCNN Frozen)")
        print("🟠"*40 + "\n")
        
        if Config.SKIP_STAGE3A:
            print("⊙ Skipping Stage 3A")
            return merged_model
        
        # 冻结CNN，解冻LSTM
        print("🔒 Freezing CNN, unfreezing LSTM...")
        _freeze_layers_by_prefix(merged_model, "merged_cnn")
        _unfreeze_layers_by_prefix(merged_model, "merged_lstm")
        
        # 重新编译
        merged_model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0005),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        
        _print_trainable_summary(merged_model)
        
        history = merged_model.fit(
            self.train_dataset,
            validation_data=self.val_dataset,
            epochs=Config.EPOCHS_STAGE3A,
            callbacks=[
                keras.callbacks.ReduceLROnPlateau(
                    monitor="val_loss", factor=0.5, patience=3, 
                    min_lr=1e-6, verbose=1
                ),
                keras.callbacks.ModelCheckpoint(
                    Config.STAGE3A_MODEL_PATH, monitor="val_accuracy", 
                    save_best_only=True, verbose=1
                ),
            ],
        )
        
        self.history['stage3a'] = history
        best_acc = max(history.history['val_accuracy'])
        print(f"\n✅ Stage 3A completed! Best val_acc: {best_acc:.4f}\n")
        
        return merged_model
    
    def run_stage3b(self, merged_model):
        """阶段3B: ResidualCNN(微调) + AttentionLSTM(冻结)"""
        print("\n" + "🔴"*40)
        print("STAGE 3B: Fine-tuning ResidualCNN (AttentionLSTM Frozen)")
        print("🔴"*40 + "\n")
        
        if Config.SKIP_STAGE3B:
            print("⊙ Skipping Stage 3B")
            return merged_model
        
        # 解冻CNN，冻结LSTM
        print("🔓 Unfreezing CNN, freezing LSTM...")
        _unfreeze_layers_by_prefix(merged_model, "merged_cnn")
        _freeze_layers_by_prefix(merged_model, "merged_lstm")
        
        # 重新编译
        merged_model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0005),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        
        _print_trainable_summary(merged_model)
        
        history = merged_model.fit(
            self.train_dataset,
            validation_data=self.val_dataset,
            epochs=Config.EPOCHS_STAGE3B,
            callbacks=[
                keras.callbacks.ReduceLROnPlateau(
                    monitor="val_loss", factor=0.5, patience=3, 
                    min_lr=1e-6, verbose=1
                ),
                keras.callbacks.ModelCheckpoint(
                    Config.STAGE3B_MODEL_PATH, monitor="val_accuracy", 
                    save_best_only=True, verbose=1
                ),
            ],
        )
        
        self.history['stage3b'] = history
        best_acc = max(history.history['val_accuracy'])
        print(f"\n✅ Stage 3B completed! Best val_acc: {best_acc:.4f}\n")
        
        return merged_model
    
    def run_stage3c(self, merged_model):
        """阶段3C: 全部解冻，联合微调"""
        print("\n" + "⚫"*40)
        print("STAGE 3C: Joint Fine-tuning (All Layers Unfrozen)")
        print("⚫"*40 + "\n")
        
        if Config.SKIP_STAGE3C:
            print("⊙ Skipping Stage 3C")
            return merged_model
        
        # 全部解冻
        print("🔓 Unfreezing all layers...")
        for layer in merged_model.layers:
            layer.trainable = True
        
        # 重新编译（非常小的学习率）
        merged_model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0001),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        
        _print_trainable_summary(merged_model)
        
        history = merged_model.fit(
            self.train_dataset,
            validation_data=self.val_dataset,
            epochs=Config.EPOCHS_STAGE3C,
            callbacks=[
                keras.callbacks.EarlyStopping(
                    monitor="val_loss", patience=15, 
                    restore_best_weights=True, verbose=1
                ),
                keras.callbacks.ReduceLROnPlateau(
                    monitor="val_loss", factor=0.5, patience=5, 
                    min_lr=1e-7, verbose=1
                ),
                keras.callbacks.ModelCheckpoint(
                    Config.FINAL_MODEL_PATH, monitor="val_accuracy", 
                    save_best_only=True, verbose=1
                ),
            ],
        )
        
        self.history['stage3c'] = history
        best_acc = max(history.history['val_accuracy'])
        print(f"\n✅ Stage 3C completed! Best val_acc: {best_acc:.4f}\n")
        
        return merged_model
    
    def save_training_history(self):
        """保存所有阶段的训练历史"""
        history_all = {}
        
        for stage_name, hist in self.history.items():
            if hist is not None:
                history_all[stage_name] = {
                    k: [float(v) for v in vals] 
                    for k, vals in hist.history.items()
                }
        
        history_path = "models/training_history_branch_merge.json"
        with open(history_path, 'w') as f:
            json.dump(history_all, f, indent=2)
        print(f"✓ Training history saved to {history_path}")
    
    def print_summary(self):
        """打印训练总结"""
        print("\n" + "="*80)
        print("TRAINING SUMMARY - BRANCH MERGE STRATEGY")
        print("="*80)
        
        # 提取最佳准确率
        results = {}
        for stage_name, hist in self.history.items():
            if hist is not None:
                results[stage_name] = max(hist.history['val_accuracy'])
        
        # 打印结果
        print("\n📊 Validation Accuracy by Stage:\n")
        print(f"  Stage 1  (SimpleCNN + SimpleLSTM):          {results.get('stage1', 0):.4f}")
        print(f"  Stage 2A (SimpleCNN + AttentionLSTM):       {results.get('stage2a', 0):.4f}")
        print(f"  Stage 2B (ResidualCNN + SimpleLSTM):        {results.get('stage2b', 0):.4f}")
        
        if 'stage3a' in results:
            print(f"  Stage 3A (Merged, LSTM tuned):              {results.get('stage3a', 0):.4f}")
        if 'stage3b' in results:
            print(f"  Stage 3B (Merged, CNN tuned):               {results.get('stage3b', 0):.4f}")
        if 'stage3c' in results:
            print(f"  Stage 3C (Merged, joint tuned):             {results.get('stage3c', 0):.4f}")
        
        # 分析改进
        print("\n" + "-"*80)
        print("IMPROVEMENT ANALYSIS")
        print("-"*80 + "\n")
        
        stage1_acc = results.get('stage1', 0)
        final_acc = results.get('stage3c', results.get('stage3b', results.get('stage3a', 0)))
        
        if 'stage2a' in results and 'stage1' in results:
            improvement_2a = results['stage2a'] - stage1_acc
            print(f"Stage 1 → 2A (Adding Attention):    {improvement_2a:+.4f} ({improvement_2a*100:+.2f}%)")
            if improvement_2a > 0.01:
                print("  ✓ Self-Attention mechanism is effective")
            else:
                print("  → Attention had minimal impact on frozen CNN features")
        
        if 'stage2b' in results and 'stage1' in results:
            improvement_2b = results['stage2b'] - stage1_acc
            print(f"Stage 1 → 2B (Adding Residual):     {improvement_2b:+.4f} ({improvement_2b*100:+.2f}%)")
            if improvement_2b > 0.01:
                print("  ✓ Residual CNN extracts better features")
            else:
                print("  → Residual CNN constrained by frozen SimpleLSTM")
        
        if final_acc > 0:
            total_improvement = final_acc - stage1_acc
            print(f"\nOverall (Stage 1 → Final):          {total_improvement:+.4f} ({total_improvement*100:+.2f}%)")
            print(f"  Baseline: {stage1_acc:.4f} → Final: {final_acc:.4f}")
        
        # 建议
        print("\n" + "="*80)
        print("RECOMMENDATIONS")
        print("="*80 + "\n")
        
        if final_acc >= 0.92:
            print("✅✅ SUCCESS! Model exceeded 0.92 target!")
            print(f"   Best model saved to: {Config.FINAL_MODEL_PATH}")
            print("   The branch-merge strategy worked excellently!")
        elif final_acc >= stage1_acc + 0.05:
            print("✅ Good improvement! Branch-merge strategy was beneficial.")
            print(f"   Continue training Stage 3C for potential further gains.")
        elif final_acc >= stage1_acc:
            print("→ Modest improvement. Consider:")
            print("  - Increase EPOCHS_STAGE3C (currently {})".format(Config.EPOCHS_STAGE3C))
            print("  - Try different learning rates in Stage 3")
            print("  - Increase model capacity (more LSTM units)")
        else:
            print("⚠️ Performance degraded. Possible causes:")
            print("  - Learning rates too high in Stage 3")
            print("  - Insufficient training in Stage 2A/2B")
            print("  - Feature mismatch between branches")
            print("\n  Try:")
            print("  - Use Stage 2A or 2B model directly (whichever is better)")
            print("  - Reduce learning rates by 10x")
            print("  - Train Stage 1 longer for better baseline")
        
        # 理论验证
        print("\n" + "-"*80)
        print("THEORETICAL VALIDATION")
        print("-"*80 + "\n")
        
        if 'stage2a' in results and 'stage2b' in results:
            print("🔬 Interface Contract Mechanism:")
            print(f"  Stage 2A learned: AttentionLSTM(SimpleCNN features)")
            print(f"  Stage 2B learned: ResidualCNN(outputs for SimpleLSTM)")
            print(f"  Merge: ResidualCNN → AttentionLSTM")
            print(f"  Compatibility: {'✓ Good' if final_acc >= max(results['stage2a'], results['stage2b']) - 0.02 else '⚠ Mismatch detected'}")
        
        print("\n" + "="*80 + "\n")
    
    def _create_dummy_history(self, val_acc, val_loss):
        """创建虚拟历史对象（用于跳过阶段时）"""
        return type('obj', (object,), {
            'history': {
                'val_accuracy': [val_acc],
                'accuracy': [val_acc],
                'val_loss': [val_loss],
                'loss': [val_loss]
            }
        })()


# ========== 主函数 ==========

def main():
    """主训练流程"""
    print("\n" + "="*80)
    print("LUOGU CAPTCHA - BRANCH-MERGE TRAINING STRATEGY")
    print("="*80 + "\n")
    
    # 打印配置
    print("Configuration:")
    print(f"  Stage 1:  {'SKIP' if Config.SKIP_STAGE1 else 'RUN'} ({Config.EPOCHS_STAGE1} epochs)")
    print(f"  Stage 2A: {'SKIP' if Config.SKIP_STAGE2A else 'RUN'} ({Config.EPOCHS_STAGE2A} epochs)")
    print(f"  Stage 2B: {'SKIP' if Config.SKIP_STAGE2B else 'RUN'} ({Config.EPOCHS_STAGE2B} epochs)")
    print(f"  Stage 3A: {'SKIP' if Config.SKIP_STAGE3A else 'RUN'} ({Config.EPOCHS_STAGE3A} epochs)")
    print(f"  Stage 3B: {'SKIP' if Config.SKIP_STAGE3B else 'RUN'} ({Config.EPOCHS_STAGE3B} epochs)")
    print(f"  Stage 3C: {'SKIP' if Config.SKIP_STAGE3C else 'RUN'} ({Config.EPOCHS_STAGE3C} epochs)")
    print()
    
    # 设置GPU
    setup_gpu()
    
    # 加载数据
    print("\nLoading datasets...")
    train_dataset, val_dataset = load_datasets(Config.TFRECORD_DIR)
    
    # 创建训练器并运行
    trainer = BranchMergeTrainer(train_dataset, val_dataset)
    final_model = trainer.run_full_pipeline()
    
    return final_model


# ========== 辅助工具：模型比较 ==========

def compare_models():
    """比较所有保存的模型"""
    print("\n" + "="*80)
    print("MODEL COMPARISON TOOL")
    print("="*80 + "\n")
    
    # 加载数据集
    _, val_dataset = load_datasets(Config.TFRECORD_DIR)
    
    model_paths = [
        ("Stage 1 (SimpleCNN+SimpleLSTM)", Config.STAGE1_MODEL_PATH),
        ("Stage 2A (SimpleCNN+AttentionLSTM)", Config.STAGE2A_MODEL_PATH),
        ("Stage 2B (ResidualCNN+SimpleLSTM)", Config.STAGE2B_MODEL_PATH),
        ("Stage 3A (Merged, LSTM tuned)", Config.STAGE3A_MODEL_PATH),
        ("Stage 3B (Merged, CNN tuned)", Config.STAGE3B_MODEL_PATH),
        ("Stage 3C (Final)", Config.FINAL_MODEL_PATH),
    ]
    
    results = []
    
    for name, path in model_paths:
        if os.path.exists(path):
            print(f"Evaluating {name}...")
            model = keras.models.load_model(path)
            loss, acc = model.evaluate(val_dataset, verbose=0)
            results.append((name, acc, loss, path))
            print(f"  Accuracy: {acc:.4f}, Loss: {loss:.4f}")
        else:
            print(f"⚠️ {name}: Model not found at {path}")
    
    if results:
        print("\n" + "-"*80)
        print("RANKING (by accuracy)")
        print("-"*80 + "\n")
        
        results.sort(key=lambda x: x[1], reverse=True)
        
        for i, (name, acc, loss, path) in enumerate(results, 1):
            print(f"{i}. {name}")
            print(f"   Accuracy: {acc:.4f} | Loss: {loss:.4f}")
            print(f"   Path: {path}")
            print()
        
        best_name, best_acc, best_loss, best_path = results[0]
        print("="*80)
        print(f"🏆 BEST MODEL: {best_name}")
        print(f"   Accuracy: {best_acc:.4f}")
        print(f"   Use: keras.models.load_model('{best_path}')")
        print("="*80 + "\n")


# ========== 辅助工具：可视化训练历史 ==========

def plot_training_history():
    """绘制训练历史曲线（如果有matplotlib）"""
    history_path = "models/training_history_branch_merge.json"
    
    if not os.path.exists(history_path):
        print(f"⚠️ History file not found: {history_path}")
        return
    
    with open(history_path, 'r') as f:
        history_all = json.load(f)
    
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Branch-Merge Training History', fontsize=16)
        
        stages = ['stage1', 'stage2a', 'stage2b', 'stage3a', 'stage3b', 'stage3c']
        titles = [
            'Stage 1: SimpleCNN+SimpleLSTM',
            'Stage 2A: SimpleCNN+AttentionLSTM',
            'Stage 2B: ResidualCNN+SimpleLSTM',
            'Stage 3A: Merged (LSTM tuned)',
            'Stage 3B: Merged (CNN tuned)',
            'Stage 3C: Merged (Joint tuned)'
        ]
        
        for i, (stage, title) in enumerate(zip(stages, titles)):
            ax = axes[i // 3, i % 3]
            
            if stage in history_all:
                hist = history_all[stage]
                
                if 'accuracy' in hist:
                    ax.plot(hist['accuracy'], label='Train Acc', alpha=0.7)
                if 'val_accuracy' in hist:
                    ax.plot(hist['val_accuracy'], label='Val Acc', alpha=0.7)
                
                ax.set_title(title)
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Accuracy')
                ax.legend()
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, 'No data', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(title + ' (skipped)')
        
        plt.tight_layout()
        plt.savefig('models/training_history_plot.png', dpi=150)
        print("✓ Training history plot saved to models/training_history_plot.png")
        plt.show()
        
    except ImportError:
        print("ℹ️ matplotlib not available, skipping plot")
        print("   Install with: pip install matplotlib")


# ========== 辅助工具：导出最佳模型信息 ==========

def export_best_model_info():
    """导出最佳模型的详细信息"""
    history_path = "models/training_history_branch_merge.json"
    
    if not os.path.exists(history_path):
        print("⚠️ No training history found")
        return
    
    with open(history_path, 'r') as f:
        history_all = json.load(f)
    
    # 找到最佳阶段
    best_stage = None
    best_acc = 0
    
    for stage, hist in history_all.items():
        if 'val_accuracy' in hist:
            max_acc = max(hist['val_accuracy'])
            if max_acc > best_acc:
                best_acc = max_acc
                best_stage = stage
    
    if best_stage is None:
        print("⚠️ No valid training data found")
        return
    
    # 确定模型路径
    model_path_map = {
        'stage1': Config.STAGE1_MODEL_PATH,
        'stage2a': Config.STAGE2A_MODEL_PATH,
        'stage2b': Config.STAGE2B_MODEL_PATH,
        'stage3a': Config.STAGE3A_MODEL_PATH,
        'stage3b': Config.STAGE3B_MODEL_PATH,
        'stage3c': Config.FINAL_MODEL_PATH,
    }
    
    best_model_path = model_path_map.get(best_stage)
    
    # 导出信息
    info = {
        'best_stage': best_stage,
        'best_accuracy': best_acc,
        'model_path': best_model_path,
        'training_config': {
            'char_size': Config.CHAR_SIZE,
            'chars_per_label': Config.CHARS_PER_LABEL,
            'img_height': Config.IMG_HEIGHT,
            'img_width': Config.IMG_WIDTH,
        },
        'architecture': {
            'stage1': 'SimpleCNN + SimpleLSTM',
            'stage2a': 'SimpleCNN(frozen) + AttentionLSTM',
            'stage2b': 'ResidualCNN + SimpleLSTM(frozen)',
            'stage3a': 'ResidualCNN(frozen) + AttentionLSTM',
            'stage3b': 'ResidualCNN + AttentionLSTM(frozen)',
            'stage3c': 'ResidualCNN + AttentionLSTM (joint)',
        }[best_stage],
        'all_results': {
            stage: max(hist['val_accuracy']) 
            for stage, hist in history_all.items() 
            if 'val_accuracy' in hist
        }
    }
    
    info_path = "models/best_model_info.json"
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=2)
    
    print("\n" + "="*80)
    print("BEST MODEL INFORMATION")
    print("="*80)
    print(f"\nStage: {best_stage}")
    print(f"Architecture: {info['architecture']}")
    print(f"Validation Accuracy: {best_acc:.4f}")
    print(f"Model Path: {best_model_path}")
    print(f"\nInfo exported to: {info_path}")
    print("="*80 + "\n")


# ========== 程序入口 ==========

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "train":
            # 训练模式
            try:
                final_model = main()
                print("✅ Training completed successfully!")
                export_best_model_info()
            except KeyboardInterrupt:
                print("\n⚠️ Training interrupted by user")
            except Exception as e:
                print(f"\n❌ Training failed with error: {e}")
                import traceback
                traceback.print_exc()
        
        elif command == "compare":
            # 比较模式
            compare_models()
        
        elif command == "plot":
            # 绘图模式
            plot_training_history()
        
        elif command == "info":
            # 信息导出模式
            export_best_model_info()
        
        else:
            print(f"Unknown command: {command}")
            print("Usage:")
            print("  python train.py train      - Run full training pipeline")
            print("  python train.py compare    - Compare all saved models")
            print("  python train.py plot       - Plot training history")
            print("  python train.py info       - Export best model info")
    
    else:
        # 默认运行训练
        try:
            final_model = main()
            print("✅ Training completed successfully!")
            export_best_model_info()
            
            # 自动运行比较和绘图
            print("\n" + "="*80)
            compare_models()
            plot_training_history()
            
        except KeyboardInterrupt:
            print("\n⚠️ Training interrupted by user")
        except Exception as e:
            print(f"\n❌ Training failed with error: {e}")
            import traceback
            traceback.print_exc()