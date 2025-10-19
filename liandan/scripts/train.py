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
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# 自动选择设备
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Using GPU: {gpus}")
    except Exception as e:
        print(f"GPU setup error: {e}")
else:
    print("Using CPU")

# 参数
CHAR_SIZE = 256
CHARS_PER_LABEL = 4
IMG_HEIGHT, IMG_WIDTH = 35, 90
EPOCHS_STAGE1 = 100  # 阶段1：训练到0.80
EPOCHS_STAGE2 = 15   # 阶段2：冻结CNN，训练Attention
EPOCHS_STAGE3 = 150  # 阶段3：解冻微调
BATCH_SIZE = 256
TFRECORD_DIR = "data/luogu_captcha_tfrecord"


def parse_tfrecord(example_proto):
    """Parses a single TFRecord example into image and label."""
    feature_description = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "label": tf.io.FixedLenFeature([CHARS_PER_LABEL], tf.int64),
    }
    example = tf.io.parse_single_example(example_proto, feature_description)
    image = tf.io.decode_png(example["image"], channels=1)
    image = tf.image.convert_image_dtype(image, tf.float32)
    label = example["label"]
    return image, label


def load_and_preprocess_data(tfrecord_dir):
    """Loads and preprocesses data from TFRecord files."""
    train_files = sorted(glob.glob(os.path.join(tfrecord_dir, "train_part_*.tfrecord")))
    test_files = sorted(glob.glob(os.path.join(tfrecord_dir, "test_part_*.tfrecord")))

    if not train_files or not test_files:
        raise ValueError(f"No TFRecord files found in {tfrecord_dir}")

    print(f"Found {len(train_files)} train files and {len(test_files)} test files")

    train_ds = tf.data.TFRecordDataset(train_files, num_parallel_reads=tf.data.AUTOTUNE)
    test_ds = tf.data.TFRecordDataset(test_files, num_parallel_reads=tf.data.AUTOTUNE)

    train_ds = train_ds.map(parse_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)
    test_ds = test_ds.map(parse_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)

    train_ds = (
        train_ds.shuffle(buffer_size=10000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    )
    test_ds = test_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    return train_ds, test_ds


# Load data
try:
    train_dataset, val_dataset = load_and_preprocess_data(TFRECORD_DIR)
    print("Data loaded successfully")
except Exception as e:
    print(f"Error loading TFRecord data: {e}")
    exit(1)

# ========== 阶段1: 普通CNN + BiLSTM 训练到 0.80 ==========
print("\n" + "="*60)
print("STAGE 1: Training Plain CNN + BiLSTM to val_acc >= 0.80")
print("="*60 + "\n")

inputs = keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 1), name="input")

# === 普通CNN（2层）===
x = layers.Conv2D(64, 3, padding="same", activation="relu", name="cnn_conv1")(inputs)
x = layers.BatchNormalization(name="cnn_bn1")(x)
x = layers.MaxPooling2D((2, 2), name="cnn_pool1")(x)

x = layers.Conv2D(128, 3, padding="same", activation="relu", name="cnn_conv2")(x)
x = layers.BatchNormalization(name="cnn_bn2")(x)
x = layers.MaxPooling2D((2, 2), name="cnn_pool2")(x)

# === 转换为序列 ===
cnn_shape = x.shape
x = layers.Reshape((cnn_shape[1] * cnn_shape[2], cnn_shape[3]), name="reshape_to_seq")(x)

# === BiLSTM ===
x = layers.Bidirectional(layers.LSTM(128, return_sequences=True), name="bilstm_1")(x)
x = layers.Dropout(0.3, name="dropout_1")(x)
x = layers.Bidirectional(layers.LSTM(128, return_sequences=False), name="bilstm_2")(x)
x = layers.Dropout(0.3, name="dropout_2")(x)

# === 输出层 ===
x = layers.Dense(CHARS_PER_LABEL * CHAR_SIZE, activation="softmax", name="dense_output")(x)
outputs = layers.Reshape((CHARS_PER_LABEL, CHAR_SIZE), name="reshape_output")(x)

model_stage1 = keras.Model(inputs=inputs, outputs=outputs, name="Stage1_CNN_BiLSTM")

model_stage1.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

print("Stage 1 Model Summary:")
model_stage1.summary()

# 训练阶段1
history_stage1 = model_stage1.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS_STAGE1,
    callbacks=[
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=20,
            mode='max',
            restore_best_weights=True,
            verbose=1,
            baseline=0.80  # 达到0.80就可以停止
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=8,
            min_lr=1e-7,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            "models/stage1_baseline.keras",
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        ),
    ],
)

# 保存阶段1模型
os.makedirs("models", exist_ok=True)
model_stage1.save("models/stage1_cnn_bilstm.keras")
print(f"\n{'='*60}")
print(f"Stage 1 completed!")
print(f"Best val_accuracy: {max(history_stage1.history['val_accuracy']):.4f}")
print(f"Model saved to models/stage1_cnn_bilstm.keras")
print(f"{'='*60}\n")

# 检查是否达到目标
best_val_acc = max(history_stage1.history['val_accuracy'])
if best_val_acc < 0.80:
    print(f"⚠️  Warning: Stage 1 only reached {best_val_acc:.4f}, lower than target 0.80")
    print("Continuing to Stage 2 anyway...")

# ========== 阶段2: 添加 Self-Attention，冻结CNN ==========
print("\n" + "="*60)
print("STAGE 2: Adding Self-Attention, Freezing CNN")
print("="*60 + "\n")

# 加载阶段1的模型
baseline_model = keras.models.load_model("models/stage1_cnn_bilstm.keras")

# 构建新模型，复用CNN权重
inputs_stage2 = keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 1), name="input")

# === 复制CNN部分（前6层）===
x = inputs_stage2
cnn_layers = []
for i, layer in enumerate(baseline_model.layers[:6]):  # Conv1, BN1, Pool1, Conv2, BN2, Pool2
    # 创建新层并复制权重
    if isinstance(layer, layers.Conv2D):
        new_layer = layers.Conv2D(
            layer.filters,
            layer.kernel_size,
            padding=layer.padding,
            activation=layer.activation,
            name=layer.name
        )
    elif isinstance(layer, layers.BatchNormalization):
        new_layer = layers.BatchNormalization(name=layer.name)
    elif isinstance(layer, layers.MaxPooling2D):
        new_layer = layers.MaxPooling2D(layer.pool_size, name=layer.name)
    else:
        new_layer = layer
    
    x = new_layer(x)
    cnn_layers.append(new_layer)
    new_layer.trainable = False  # 冻结CNN

# 复制CNN权重
for i, layer in enumerate(cnn_layers):
    if hasattr(baseline_model.layers[i], 'get_weights'):
        weights = baseline_model.layers[i].get_weights()
        if weights:
            layer.set_weights(weights)

print("✓ CNN weights copied and frozen")

# === 新的序列处理部分 ===
cnn_shape = x.shape
x = layers.Reshape((cnn_shape[1] * cnn_shape[2], cnn_shape[3]), name="reshape_to_seq")(x)

# === Self-Attention（新增）===
attention_output = layers.MultiHeadAttention(
    num_heads=4,
    key_dim=64,
    dropout=0.1,
    name="self_attention"
)(x, x)
x = layers.LayerNormalization(name="attn_norm")(x + attention_output)

# === BiLSTM（重新初始化）===
x = layers.Bidirectional(layers.LSTM(128, return_sequences=True), name="bilstm_1_new")(x)
x = layers.Dropout(0.3, name="dropout_1_new")(x)
x = layers.Bidirectional(layers.LSTM(128, return_sequences=False), name="bilstm_2_new")(x)
x = layers.Dropout(0.3, name="dropout_2_new")(x)

# === 输出层 ===
x = layers.Dense(CHARS_PER_LABEL * CHAR_SIZE, activation="softmax", name="dense_output_new")(x)
outputs_stage2 = layers.Reshape((CHARS_PER_LABEL, CHAR_SIZE), name="reshape_output_new")(x)

model_stage2 = keras.Model(inputs=inputs_stage2, outputs=outputs_stage2, name="Stage2_CNN_Attention_BiLSTM")

model_stage2.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.002),  # 稍大的学习率
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

print("\nStage 2 Model Summary:")
model_stage2.summary()

# 训练阶段2
print(f"\nTraining Stage 2 for {EPOCHS_STAGE2} epochs...")
history_stage2 = model_stage2.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS_STAGE2,
    callbacks=[
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-5,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            "models/stage2_with_attention.keras",
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        ),
    ],
)

print(f"\n{'='*60}")
print(f"Stage 2 completed!")
print(f"Best val_accuracy: {max(history_stage2.history['val_accuracy']):.4f}")
print(f"{'='*60}\n")

# ========== 阶段3: 解冻全部，微调 ==========
print("\n" + "="*60)
print("STAGE 3: Unfreezing All Layers, Fine-tuning")
print("="*60 + "\n")

# 解冻所有层
for layer in model_stage2.layers:
    layer.trainable = True

print("✓ All layers unfrozen")

# 使用很小的学习率
model_stage2.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

# 训练阶段3
print(f"\nTraining Stage 3 for up to {EPOCHS_STAGE3} epochs...")
history_stage3 = model_stage2.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS_STAGE3,
    callbacks=[
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            "models/stage3_final.keras",
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        ),
    ],
)

# ========== 保存最终模型 ==========
final_model_path = "models/luoguCaptcha_final.keras"
model_stage2.save(final_model_path)

print("\n" + "="*60)
print("TRAINING COMPLETED!")
print("="*60)
print("\nFinal Results:")
print(f"  Stage 1 best val_acc: {max(history_stage1.history['val_accuracy']):.4f}")
print(f"  Stage 2 best val_acc: {max(history_stage2.history['val_accuracy']):.4f}")
print(f"  Stage 3 best val_acc: {max(history_stage3.history['val_accuracy']):.4f}")
print(f"\nFinal model saved to: {final_model_path}")
print("="*60 + "\n")

# 保存训练历史
import json
history_all = {
    'stage1': {k: [float(v) for v in vals] for k, vals in history_stage1.history.items()},
    'stage2': {k: [float(v) for v in vals] for k, vals in history_stage2.history.items()},
    'stage3': {k: [float(v) for v in vals] for k, vals in history_stage3.history.items()},
}

history_path = "models/training_history_all_stages.json"
with open(history_path, 'w') as f:
    json.dump(history_all, f, indent=2)
print(f"Training history saved to: {history_path}")

# 可视化训练过程（可选）
try:
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Three-Stage Training Process', fontsize=16)
    
    stages = [
        ('Stage 1: CNN+BiLSTM', history_stage1.history),
        ('Stage 2: +Attention (CNN Frozen)', history_stage2.history),
        ('Stage 3: Fine-tuning (All Unfrozen)', history_stage3.history)
    ]
    
    for idx, (title, history) in enumerate(stages):
        # Loss
        axes[0, idx].plot(history['loss'], label='Train Loss')
        axes[0, idx].plot(history['val_loss'], label='Val Loss')
        axes[0, idx].set_title(f'{title}\nLoss')
        axes[0, idx].set_xlabel('Epoch')
        axes[0, idx].set_ylabel('Loss')
        axes[0, idx].legend()
        axes[0, idx].grid(True)
        
        # Accuracy
        axes[1, idx].plot(history['accuracy'], label='Train Acc')
        axes[1, idx].plot(history['val_accuracy'], label='Val Acc')
        axes[1, idx].set_title(f'{title}\nAccuracy')
        axes[1, idx].set_xlabel('Epoch')
        axes[1, idx].set_ylabel('Accuracy')
        axes[1, idx].legend()
        axes[1, idx].grid(True)
    
    plt.tight_layout()
    plt.savefig('models/training_curves.png', dpi=150, bbox_inches='tight')
    print(f"Training curves saved to: models/training_curves.png")
    
except ImportError:
    print("Matplotlib not available, skipping visualization")

# 打印详细分析
print("\n" + "="*60)
print("DETAILED ANALYSIS")
print("="*60)

def analyze_stage(stage_name, history):
    print(f"\n{stage_name}:")
    print(f"  Initial → Final:")
    print(f"    Train Loss: {history['loss'][0]:.4f} → {history['loss'][-1]:.4f}")
    print(f"    Val Loss:   {history['val_loss'][0]:.4f} → {history['val_loss'][-1]:.4f}")
    print(f"    Train Acc:  {history['accuracy'][0]:.4f} → {history['accuracy'][-1]:.4f}")
    print(f"    Val Acc:    {history['val_accuracy'][0]:.4f} → {history['val_accuracy'][-1]:.4f}")
    print(f"  Best Val Accuracy: {max(history['val_accuracy']):.4f} (Epoch {history['val_accuracy'].index(max(history['val_accuracy'])) + 1})")
    print(f"  Total Epochs: {len(history['loss'])}")

analyze_stage("Stage 1 (CNN+BiLSTM Baseline)", history_stage1.history)
analyze_stage("Stage 2 (Added Attention, CNN Frozen)", history_stage2.history)
analyze_stage("Stage 3 (Fine-tuning All)", history_stage3.history)

# 对比分析
print("\n" + "-"*60)
print("IMPROVEMENT ANALYSIS:")
print("-"*60)
stage1_best = max(history_stage1.history['val_accuracy'])
stage2_best = max(history_stage2.history['val_accuracy'])
stage3_best = max(history_stage3.history['val_accuracy'])

print(f"\nStage 1 → Stage 2:")
improvement_1_2 = stage2_best - stage1_best
print(f"  Val Acc: {stage1_best:.4f} → {stage2_best:.4f} ({improvement_1_2:+.4f})")
if improvement_1_2 > 0.01:
    print(f"  ✓ Self-Attention helped! (+{improvement_1_2:.2%})")
elif improvement_1_2 > -0.02:
    print(f"  → Self-Attention maintained performance")
else:
    print(f"  ✗ Self-Attention hurt performance ({improvement_1_2:.2%})")

print(f"\nStage 2 → Stage 3:")
improvement_2_3 = stage3_best - stage2_best
print(f"  Val Acc: {stage2_best:.4f} → {stage3_best:.4f} ({improvement_2_3:+.4f})")
if improvement_2_3 > 0.01:
    print(f"  ✓ Fine-tuning helped! (+{improvement_2_3:.2%})")
elif improvement_2_3 > -0.02:
    print(f"  → Fine-tuning maintained performance")
else:
    print(f"  ✗ Fine-tuning hurt performance ({improvement_2_3:.2%})")

print(f"\nOverall (Stage 1 → Stage 3):")
improvement_total = stage3_best - stage1_best
print(f"  Val Acc: {stage1_best:.4f} → {stage3_best:.4f} ({improvement_total:+.4f})")
if stage3_best > 0.92:
    print(f"  ✓✓ SUCCESS! Exceeded 0.92 target")
elif improvement_total > 0.01:
    print(f"  ✓ Improved over baseline (+{improvement_total:.2%})")
else:
    print(f"  → Similar to baseline")

print("\n" + "="*60)
print("NEXT STEPS:")
print("="*60)

if stage3_best > 0.92:
    print("\n✓ Model successfully trained!")
    print(f"  Load model: keras.models.load_model('{final_model_path}')")
elif improvement_1_2 < -0.02:
    print("\n⚠️ Self-Attention hurt performance")
    print("  Suggestions:")
    print("  1. Try without Self-Attention (just use Stage 1 model)")
    print("  2. Reduce attention heads: num_heads=4 → 2")
    print("  3. Try different attention architecture")
elif improvement_2_3 < -0.02:
    print("\n⚠️ Fine-tuning hurt performance")
    print("  Suggestions:")
    print("  1. Use Stage 2 model instead")
    print("  2. Reduce learning rate in Stage 3: 0.0001 → 0.00005")
    print("  3. Freeze CNN in Stage 3, only fine-tune Attention+LSTM")
else:
    print("\n→ Model trained but didn't exceed baseline")
    print("  Suggestions:")
    print("  1. Train Stage 1 longer (higher baseline)")
    print("  2. Increase LSTM units: 128 → 256")
    print("  3. Try stacking multiple Attention layers")
    print("  4. Experiment with different Reshape strategies")

print("\n" + "="*60 + "\n")