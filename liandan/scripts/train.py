# Copyright (C) 2025 Langning Chen
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
    EPOCHS_STAGE1 = 100
    EPOCHS_STAGE2 = 15
    EPOCHS_STAGE3 = 150
    
    # 模型路径
    STAGE1_MODEL_PATH = "models/stage1_cnn_bilstm.keras"
    STAGE2_MODEL_PATH = "models/stage2_with_attention.keras"
    FINAL_MODEL_PATH = "models/luoguCaptcha_final.keras"
    
    # 控制开关
    SKIP_STAGE1 = True   # 设为 True 跳过阶段1
    SKIP_STAGE2 = False  # 设为 True 跳过阶段2
    SKIP_STAGE3 = False  # 设为 True 跳过阶段3


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
    """构建阶段1模型：普通CNN + BiLSTM"""
    inputs = keras.Input(shape=(Config.IMG_HEIGHT, Config.IMG_WIDTH, 1), name="input")

    # CNN部分
    x = layers.Conv2D(64, 3, padding="same", activation="relu", name="cnn_conv1")(inputs)
    x = layers.BatchNormalization(name="cnn_bn1")(x)
    x = layers.MaxPooling2D((2, 2), name="cnn_pool1")(x)

    x = layers.Conv2D(128, 3, padding="same", activation="relu", name="cnn_conv2")(x)
    x = layers.BatchNormalization(name="cnn_bn2")(x)
    x = layers.MaxPooling2D((2, 2), name="cnn_pool2")(x)

    # 转换为序列
    cnn_shape = x.shape
    x = layers.Reshape((cnn_shape[1] * cnn_shape[2], cnn_shape[3]), name="reshape_to_seq")(x)

    # BiLSTM
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True), name="bilstm_1")(x)
    x = layers.Dropout(0.3, name="dropout_1")(x)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=False), name="bilstm_2")(x)
    x = layers.Dropout(0.3, name="dropout_2")(x)

    # 输出层
    x = layers.Dense(Config.CHARS_PER_LABEL * Config.CHAR_SIZE, activation="softmax", name="dense_output")(x)
    outputs = layers.Reshape((Config.CHARS_PER_LABEL, Config.CHAR_SIZE), name="reshape_output")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="Stage1_CNN_BiLSTM")
    return model


def build_stage2_model(baseline_model):
    """构建阶段2模型：添加Self-Attention，冻结CNN"""
    inputs = keras.Input(shape=(Config.IMG_HEIGHT, Config.IMG_WIDTH, 1), name="input")
    
    # 构建CNN部分（结构与stage1相同）
    x = inputs
    x = layers.Conv2D(64, 3, padding="same", activation="relu", name="cnn_conv1_frozen")(x)
    x = layers.BatchNormalization(name="cnn_bn1_frozen")(x)
    x = layers.MaxPooling2D((2, 2), name="cnn_pool1_frozen")(x)
    x = layers.Conv2D(128, 3, padding="same", activation="relu", name="cnn_conv2_frozen")(x)
    x = layers.BatchNormalization(name="cnn_bn2_frozen")(x)
    x = layers.MaxPooling2D((2, 2), name="cnn_pool2_frozen")(x)
    
    # 创建临时模型用于复制权重
    temp_model = keras.Model(inputs=inputs, outputs=x, name="temp_cnn")
    
    # 复制CNN权重
    print("Copying CNN weights from Stage 1...")
    for i in range(1, 7):  # 6个CNN层
        try:
            weights = baseline_model.layers[i].get_weights()
            if weights:
                temp_model.layers[i].set_weights(weights)
                print(f"  ✓ Copied layer {i}: {baseline_model.layers[i].name}")
        except Exception as e:
            print(f"  ⚠ Skipped layer {i}: {e}")
    
    # 转换为序列
    cnn_shape = x.shape
    x = layers.Reshape((cnn_shape[1] * cnn_shape[2], cnn_shape[3]), name="reshape_to_seq")(x)
    
    # Self-Attention（新增）
    attention_output = layers.MultiHeadAttention(
        num_heads=4, key_dim=64, dropout=0.1, name="self_attention"
    )(x, x)
    x = layers.LayerNormalization(name="attn_norm")(x + attention_output)
    
    # BiLSTM（重新初始化）
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True), name="bilstm_1_new")(x)
    x = layers.Dropout(0.3, name="dropout_1_new")(x)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=False), name="bilstm_2_new")(x)
    x = layers.Dropout(0.3, name="dropout_2_new")(x)
    
    # 输出层
    x = layers.Dense(Config.CHARS_PER_LABEL * Config.CHAR_SIZE, activation="softmax", name="dense_output_new")(x)
    outputs = layers.Reshape((Config.CHARS_PER_LABEL, Config.CHAR_SIZE), name="reshape_output_new")(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name="Stage2_CNN_Attention_BiLSTM")
    
    # 冻结CNN层
    print("\nFreezing CNN layers...")
    for i in range(1, 7):
        model.layers[i].trainable = False
        print(f"  Frozen: {model.layers[i].name}")
    
    return model


# ========== 训练阶段 ==========
class StageTrainer:
    """三阶段训练管理器"""
    
    def __init__(self, train_dataset, val_dataset):
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.history_stage1 = None
        self.history_stage2 = None
        self.history_stage3 = None
        os.makedirs("models", exist_ok=True)
    
    def run_stage1(self):
        """阶段1：训练基线模型（CNN + BiLSTM）"""
        print("\n" + "="*60)
        print("STAGE 1: Training Plain CNN + BiLSTM")
        print("="*60 + "\n")
        
        if Config.SKIP_STAGE1 and os.path.exists(Config.STAGE1_MODEL_PATH):
            print(f"⊙ Skipping Stage 1 (loading from {Config.STAGE1_MODEL_PATH})")
            model = keras.models.load_model(Config.STAGE1_MODEL_PATH)
            
            # 评估模型
            print("Evaluating loaded model...")
            val_loss, val_acc = model.evaluate(self.val_dataset, verbose=0)
            print(f"  Validation accuracy: {val_acc:.4f}")
            print(f"  Validation loss: {val_loss:.4f}")
            
            # 创建假history
            self.history_stage1 = type('obj', (object,), {
                'history': {
                    'val_accuracy': [val_acc],
                    'loss': [val_loss],
                    'accuracy': [val_acc],
                    'val_loss': [val_loss]
                }
            })()
            
            return model
        
        # 构建并训练模型
        model = build_stage1_model()
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        
        print("Model Summary:")
        model.summary()
        
        self.history_stage1 = model.fit(
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
        
        model.save(Config.STAGE1_MODEL_PATH)
        best_acc = max(self.history_stage1.history['val_accuracy'])
        print(f"\n✓ Stage 1 completed! Best val_acc: {best_acc:.4f}")
        print(f"✓ Model saved to {Config.STAGE1_MODEL_PATH}\n")
        
        return model
    
    def run_stage2(self, baseline_model):
        """阶段2：添加Self-Attention，冻结CNN"""
        print("\n" + "="*60)
        print("STAGE 2: Adding Self-Attention, Freezing CNN")
        print("="*60 + "\n")
        
        if Config.SKIP_STAGE2 and os.path.exists(Config.STAGE2_MODEL_PATH):
            print(f"⊙ Skipping Stage 2 (loading from {Config.STAGE2_MODEL_PATH})")
            model = keras.models.load_model(Config.STAGE2_MODEL_PATH)
            
            val_loss, val_acc = model.evaluate(self.val_dataset, verbose=0)
            print(f"  Validation accuracy: {val_acc:.4f}")
            
            self.history_stage2 = type('obj', (object,), {
                'history': {
                    'val_accuracy': [val_acc],
                    'loss': [val_loss],
                    'accuracy': [val_acc],
                    'val_loss': [val_loss]
                }
            })()
            
            return model
        
        # 构建模型
        model = build_stage2_model(baseline_model)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.002),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        
        print("\nModel Summary:")
        model.summary()
        
        trainable_count = sum([tf.size(w).numpy() for w in model.trainable_weights])
        non_trainable_count = sum([tf.size(w).numpy() for w in model.non_trainable_weights])
        print(f"\nTrainable params: {trainable_count:,}")
        print(f"Non-trainable params: {non_trainable_count:,}\n")
        
        self.history_stage2 = model.fit(
            self.train_dataset,
            validation_data=self.val_dataset,
            epochs=Config.EPOCHS_STAGE2,
            callbacks=[
                keras.callbacks.ReduceLROnPlateau(
                    monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5, verbose=1
                ),
                keras.callbacks.ModelCheckpoint(
                    Config.STAGE2_MODEL_PATH, monitor="val_accuracy", save_best_only=True, verbose=1
                ),
            ],
        )
        
        best_acc = max(self.history_stage2.history['val_accuracy'])
        print(f"\n✓ Stage 2 completed! Best val_acc: {best_acc:.4f}")
        print(f"✓ Model saved to {Config.STAGE2_MODEL_PATH}\n")
        
        return model
    
    def run_stage3(self, model):
        """阶段3：解冻全部，微调"""
        print("\n" + "="*60)
        print("STAGE 3: Unfreezing All Layers, Fine-tuning")
        print("="*60 + "\n")
        
        if Config.SKIP_STAGE3:
            print("⊙ Skipping Stage 3")
            self.history_stage3 = type('obj', (object,), {
                'history': {
                    'val_accuracy': [0],
                    'loss': [0],
                    'accuracy': [0],
                    'val_loss': [0]
                }
            })()
            return model
        
        # 解冻所有层
        print("Unfreezing all layers...")
        for layer in model.layers:
            layer.trainable = True
            if hasattr(layer, 'name') and 'frozen' in layer.name:
                print(f"  Unfrozen: {layer.name}")
        
        # 重新编译（小学习率）
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0001),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        
        print(f"\nTraining for up to {Config.EPOCHS_STAGE3} epochs...\n")
        
        self.history_stage3 = model.fit(
            self.train_dataset,
            validation_data=self.val_dataset,
            epochs=Config.EPOCHS_STAGE3,
            callbacks=[
                keras.callbacks.EarlyStopping(
                    monitor="val_loss", patience=15, restore_best_weights=True, verbose=1
                ),
                keras.callbacks.ReduceLROnPlateau(
                    monitor="val_loss", factor=0.5, patience=5, min_lr=1e-7, verbose=1
                ),
                keras.callbacks.ModelCheckpoint(
                    Config.FINAL_MODEL_PATH, monitor="val_accuracy", save_best_only=True, verbose=1
                ),
            ],
        )
        
        best_acc = max(self.history_stage3.history['val_accuracy'])
        print(f"\n✓ Stage 3 completed! Best val_acc: {best_acc:.4f}")
        print(f"✓ Final model saved to {Config.FINAL_MODEL_PATH}\n")
        
        return model
    
    def save_training_history(self):
        """保存所有阶段的训练历史"""
        history_all = {
            'stage1': {k: [float(v) for v in vals] for k, vals in self.history_stage1.history.items()},
            'stage2': {k: [float(v) for v in vals] for k, vals in self.history_stage2.history.items()},
            'stage3': {k: [float(v) for v in vals] for k, vals in self.history_stage3.history.items()},
        }
        
        history_path = "models/training_history_all_stages.json"
        with open(history_path, 'w') as f:
            json.dump(history_all, f, indent=2)
        print(f"✓ Training history saved to {history_path}")
    
    def print_summary(self):
        """打印训练总结"""
        print("\n" + "="*60)
        print("TRAINING SUMMARY")
        print("="*60)
        
        stage1_best = max(self.history_stage1.history['val_accuracy'])
        stage2_best = max(self.history_stage2.history['val_accuracy'])
        stage3_best = max(self.history_stage3.history['val_accuracy']) if not Config.SKIP_STAGE3 else stage2_best
        
        print(f"\nStage 1 (CNN+BiLSTM):          {stage1_best:.4f}")
        print(f"Stage 2 (+Attention, CNN frozen): {stage2_best:.4f}")
        if not Config.SKIP_STAGE3:
            print(f"Stage 3 (Fine-tuning):         {stage3_best:.4f}")
        
        # 改进分析
        print("\n" + "-"*60)
        print("IMPROVEMENT ANALYSIS")
        print("-"*60)
        
        improvement_1_2 = stage2_best - stage1_best
        print(f"\nStage 1 → Stage 2: {stage1_best:.4f} → {stage2_best:.4f} ({improvement_1_2:+.4f})")
        if improvement_1_2 > 0.01:
            print(f"  ✓ Self-Attention helped (+{improvement_1_2:.2%})")
        elif improvement_1_2 > -0.02:
            print(f"  → Maintained performance")
        else:
            print(f"  ✗ Performance dropped ({improvement_1_2:.2%})")
        
        if not Config.SKIP_STAGE3:
            improvement_2_3 = stage3_best - stage2_best
            print(f"\nStage 2 → Stage 3: {stage2_best:.4f} → {stage3_best:.4f} ({improvement_2_3:+.4f})")
            if improvement_2_3 > 0.01:
                print(f"  ✓ Fine-tuning helped (+{improvement_2_3:.2%})")
            elif improvement_2_3 > -0.02:
                print(f"  → Maintained performance")
            else:
                print(f"  ✗ Performance dropped ({improvement_2_3:.2%})")
            
            improvement_total = stage3_best - stage1_best
            print(f"\nOverall: {stage1_best:.4f} → {stage3_best:.4f} ({improvement_total:+.4f})")
        
        # 建议
        print("\n" + "="*60)
        print("RECOMMENDATIONS")
        print("="*60 + "\n")
        
        final_acc = stage3_best if not Config.SKIP_STAGE3 else stage2_best
        
        if final_acc > 0.92:
            print("✓✓ SUCCESS! Model exceeded 0.92 target")
            print(f"   Use: keras.models.load_model('{Config.FINAL_MODEL_PATH}')")
        elif improvement_1_2 < -0.02:
            print("⚠ Self-Attention hurt performance")
            print("  Suggestions:")
            print("  - Try using Stage 1 model only")
            print("  - Reduce attention heads: num_heads=4 → 2")
            print("  - Increase LSTM units before adding attention")
        elif not Config.SKIP_STAGE3 and improvement_2_3 < -0.02:
            print("⚠ Fine-tuning hurt performance")
            print("  Suggestions:")
            print("  - Use Stage 2 model instead")
            print("  - Lower learning rate: 0.0001 → 0.00005")
            print("  - Only unfreeze upper layers")
        else:
            print("→ Model trained successfully but has room for improvement")
            print("  Suggestions:")
            print("  - Train Stage 1 longer for higher baseline")
            print("  - Increase model capacity (more LSTM units)")
            print("  - Try data augmentation")
        
        print("\n" + "="*60 + "\n")


# ========== 主函数 ==========
def main():
    """主训练流程"""
    print("\n" + "="*60)
    print("LUOGU CAPTCHA - THREE-STAGE TRAINING")
    print("="*60 + "\n")
    
    # 配置
    print("Configuration:")
    print(f"  Stage 1: {'SKIP' if Config.SKIP_STAGE1 else 'RUN'}")
    print(f"  Stage 2: {'SKIP' if Config.SKIP_STAGE2 else 'RUN'}")
    print(f"  Stage 3: {'SKIP' if Config.SKIP_STAGE3 else 'RUN'}")
    print()
    
    # 设置GPU
    setup_gpu()
    
    # 加载数据
    print("\nLoading datasets...")
    train_dataset, val_dataset = load_datasets(Config.TFRECORD_DIR)
    
    # 创建训练器
    trainer = StageTrainer(train_dataset, val_dataset)
    
    # 阶段1
    model_stage1 = trainer.run_stage1()
    
    # 阶段2
    model_stage2 = trainer.run_stage2(model_stage1)
    
    # 阶段3
    final_model = trainer.run_stage3(model_stage2)
    
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
    except KeyboardInterrupt:
        print("\n⚠ Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()