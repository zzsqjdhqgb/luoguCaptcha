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
EPOCHS_STAGE1 = 30  # CNN+Dense 阶段
EPOCHS_STAGE2 = 15  # 冻结CNN+LSTM 阶段
EPOCHS_STAGE3 = 50  # 解冻全部模型阶段
BATCH_SIZE = 256
TFRECORD_DIR = "data/luogu_captcha_tfrecord"


def uppercase_label(label):
    """
    将标签转换为大写。
    小写字母 a-z (ASCII 97-122) 转换为大写 A-Z (ASCII 65-90)
    """
    def convert_char(char_code):
        # 如果是小写字母 (97-122)，转换为大写 (65-90)
        return tf.where(
            tf.logical_and(char_code >= 97, char_code <= 122),
            char_code - 32,  # 转换为大写
            char_code        # 保持不变
        )
    
    return tf.map_fn(convert_char, label, dtype=tf.int64)


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
    
    # 将标签转换为大写
    label = uppercase_label(label)
    
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
    
    # 验证标签转换（可选）
    print("\n验证标签转换:")
    for images, labels in train_dataset.take(1):
        print(f"第一个batch的前5个标签:")
        for i in range(min(5, labels.shape[0])):
            label_chars = [chr(int(c)) for c in labels[i].numpy()]
            print(f"  样本 {i+1}: {label_chars} -> {''.join(label_chars)}")
        break
    
except Exception as e:
    print(f"Error loading TFRecord data: {e}")
    exit(1)

# ========== 阶段1: CNN+Dense 训练10个epoch ==========
print("\n" + "="*50)
print("Stage 1: Training CNN+Dense for 10 epochs")
print("="*50 + "\n")

inputs = keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 1), name="input")
x = layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D(2)(x)
x = layers.Conv2D(64, 4, padding="same", activation="relu")(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D(2)(x)
x = layers.Conv2D(128, 5, padding="same", activation="relu")(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D(2)(x)
cnn_output = x  # 保存CNN输出用于后续构建LSTM模型
x = layers.Flatten()(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(256, activation="relu")(x)
x = layers.Dropout(0.4)(x)
x = layers.Dense(CHARS_PER_LABEL * CHAR_SIZE, activation="softmax")(x)
outputs = layers.Reshape((CHARS_PER_LABEL, CHAR_SIZE))(x)

model = keras.Model(inputs=inputs, outputs=outputs, name="LuoguCaptcha_Stage1")
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
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

# ========== 阶段2: 冻结CNN, 替换为LSTM, 训练10个epoch ==========
print("\n" + "="*50)
print("Stage 2: Freezing CNN, replacing with LSTM, training for 10 epochs")
print("="*50 + "\n")

# 创建新模型，复用CNN部分
inputs_lstm = keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 1), name="input")
x = layers.Conv2D(32, 3, padding="same", activation="relu")(inputs_lstm)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D(2)(x)
x = layers.Conv2D(64, 4, padding="same", activation="relu")(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D(2)(x)
x = layers.Conv2D(128, 5, padding="same", activation="relu")(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D(2)(x)

# 将CNN输出reshape为序列形式 (batch, time_steps, features)
# 当前x的shape: (batch, height, width, channels)
shape = x.shape
x = layers.Reshape((shape[1] * shape[2], shape[3]))(x)  # (batch, time_steps, features)

# LSTM层
x = layers.LSTM(128, return_sequences=True)(x)
x = layers.Dropout(0.3)(x)
x = layers.LSTM(128, return_sequences=False)(x)
x = layers.Dropout(0.4)(x)
x = layers.Dense(CHARS_PER_LABEL * CHAR_SIZE, activation="softmax")(x)
outputs_lstm = layers.Reshape((CHARS_PER_LABEL, CHAR_SIZE))(x)

model_lstm = keras.Model(inputs=inputs_lstm, outputs=outputs_lstm, name="LuoguCaptcha_Stage2")

# 复制CNN权重
for i, layer in enumerate(model_lstm.layers[:9]):  # 前9层是CNN部分
    layer.set_weights(model.layers[i].get_weights())
    layer.trainable = False  # 冻结CNN

model_lstm.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
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

# ========== 阶段3: 解冻全部模型, 训练50个epoch ==========
print("\n" + "="*50)
print("Stage 3: Unfreezing all layers, training for 50 epochs")
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
            monitor="val_loss", patience=5, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.2, patience=3, min_lr=1e-5
        ),
    ],
)

# 保存模型 (本地)
os.makedirs("models", exist_ok=True)
final_model_path = "models/luoguCaptcha.CRNN.basic.cnnleading.keras"
model_lstm.save(final_model_path)
print(f"Model saved to {final_model_path}")

# 提示上传
print(f"Run `python scripts/huggingface.py upload_model {final_model_path}` to upload.")