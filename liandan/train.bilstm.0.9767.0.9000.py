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
from tqdm import tqdm

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
EPOCHS_STAGE1 = 20  # CNN+Dense 阶段
EPOCHS_STAGE2 = 20  # 冻结CNN+BiLSTM 阶段
EPOCHS_STAGE3 = 150  # 解冻全部模型阶段
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
    image = tf.image.convert_image_dtype(image, tf.float32)  # Normalize to [0, 1]
    label = example["label"]  # Shape: (4,)
    return image, label


def load_and_preprocess_data(tfrecord_dir):
    """Loads and preprocesses data from TFRecord files."""
    # Get train and test TFRecord files
    train_files = sorted(glob.glob(os.path.join(tfrecord_dir, "train_part_*.tfrecord")))
    test_files = sorted(glob.glob(os.path.join(tfrecord_dir, "test_part_*.tfrecord")))

    if not train_files or not test_files:
        raise ValueError(f"No TFRecord files found in {tfrecord_dir}")

    print(f"Found {len(train_files)} train files and {len(test_files)} test files")

    # Create tf.data datasets
    train_ds = tf.data.TFRecordDataset(train_files, num_parallel_reads=tf.data.AUTOTUNE)
    test_ds = tf.data.TFRecordDataset(test_files, num_parallel_reads=tf.data.AUTOTUNE)

    # Parse TFRecords
    train_ds = train_ds.map(parse_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)
    test_ds = test_ds.map(parse_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)

    # Shuffle and batch train dataset
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

# ========== 阶段1: CNN+Dense 训练20个epoch ==========
print("\n" + "="*50)
print("Stage 1: Training CNN+Dense for 20 epochs")
print("="*50 + "\n")

inputs = keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 1), name="input")
x = layers.Conv2D(64, 3, padding="same", activation="relu")(inputs)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D(2)(x)
x = layers.Conv2D(128, 4, padding="same", activation="relu")(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D(2)(x)
x = layers.Conv2D(256, 5, padding="same", activation="relu")(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D(2)(x)
x = layers.Flatten()(x)
x = layers.Dense(512, activation="relu")(x)
x = layers.Dropout(0.2)(x)
x = layers.Dense(CHARS_PER_LABEL * CHAR_SIZE, activation="softmax")(x)
outputs = layers.Reshape((CHARS_PER_LABEL, CHAR_SIZE))(x)

model = keras.Model(inputs=inputs, outputs=outputs, name="LuoguCaptcha_Stage1")
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.002),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
model.summary()

history1 = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS_STAGE1,
    callbacks=[
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.2, patience=3, min_lr=1e-5
        ),
    ],
)

# ========== 阶段2: 冻结CNN, 替换为BiLSTM, 训练20个epoch ==========
print("\n" + "="*50)
print("Stage 2: Freezing CNN, replacing with Bidirectional LSTM, training for 20 epochs")
print("="*50 + "\n")

# 创建新模型，复用CNN部分
inputs_lstm = keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 1), name="input")
x = layers.Conv2D(64, 3, padding="same", activation="relu")(inputs_lstm)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D(2)(x)
x = layers.Conv2D(128, 4, padding="same", activation="relu")(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D(2)(x)
x = layers.Conv2D(256, 5, padding="same", activation="relu")(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D(2)(x)

# 将CNN输出reshape为序列形式 (batch, time_steps, features)
# 当前x的shape: (batch, height, width, channels)
shape = x.shape
x = layers.Reshape((shape[1] * shape[2], shape[3]))(x)  # (batch, time_steps, features)

# 双向LSTM层
x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
x = layers.Dropout(0.2)(x)
x = layers.Bidirectional(layers.LSTM(128, return_sequences=False))(x)
x = layers.Dropout(0.2)(x)
x = layers.Dense(CHARS_PER_LABEL * CHAR_SIZE, activation="softmax")(x)
outputs_lstm = layers.Reshape((CHARS_PER_LABEL, CHAR_SIZE))(x)

model_lstm = keras.Model(inputs=inputs_lstm, outputs=outputs_lstm, name="LuoguCaptcha_Stage2")

# 复制CNN权重
for i, layer in enumerate(model_lstm.layers[:9]):  # 前9层是CNN部分
    layer.set_weights(model.layers[i].get_weights())
    layer.trainable = False  # 冻结CNN

model_lstm.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.002),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
model_lstm.summary()

history2 = model_lstm.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS_STAGE2,
    callbacks=[
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.2, patience=3, min_lr=1e-5
        ),
    ],
)

# ========== 阶段3: 解冻全部模型, 训练150个epoch ==========
print("\n" + "="*50)
print("Stage 3: Unfreezing all layers, training for 150 epochs")
print("="*50 + "\n")

# 解冻所有层
for layer in model_lstm.layers:
    layer.trainable = True

# 使用较小的学习率
model_lstm.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

history3 = model_lstm.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS_STAGE3,
    callbacks=[
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.2, patience=5, min_lr=1e-6
        ),
    ],
)

# 保存模型 (本地)
os.makedirs("models", exist_ok=True)
final_model_path = "models/luoguCaptcha.keras"
model_lstm.save(final_model_path)
print(f"Model saved to {final_model_path}")

# 提示上传
print(f"Run `python scripts/huggingface.py upload_model {final_model_path}` to upload.")