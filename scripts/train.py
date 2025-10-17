# scripts/train.py
# Copyright (C) 2025 Langning Chen
# CRNN-CTC based captcha training script

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from datasets import load_dataset

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
NUM_CLASSES = CHAR_SIZE + 1  # 增加一个空白符给 CTC Loss
CHARS_PER_LABEL = 4
IMG_HEIGHT, IMG_WIDTH = 35, 90
EPOCHS = 15
BATCH_SIZE = 256
DATASET_PATH = "langningchen/luogu-captcha-dataset"

# 根据模型架构计算CNN输出的序列长度
# Input (90) -> Pool1(2) -> 45 -> Pool2(2) -> 22 -> Pool3(2) -> 11
SEQ_LEN = 11


# 数据加载与预处理 (适配 CTC)
def load_and_preprocess_data(dataset_path):
    """Loads pre-processed dataset from Hugging Face Hub and prepares it for CTC loss."""
    dataset_dict = load_dataset(dataset_path)
    train_ds_hf = dataset_dict["train"]
    val_ds_hf = dataset_dict["test"]

    train_ds = train_ds_hf.to_tf_dataset(
        columns="image",
        label_cols="label",
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    val_ds = val_ds_hf.to_tf_dataset(
        columns="image", label_cols="label", batch_size=BATCH_SIZE
    )

    # 准备 CTC Loss 所需的输入
    def prepare_for_ctc(image, label):
        batch_size = tf.shape(image)[0]
        input_length = tf.ones(shape=(batch_size, 1), dtype="int32") * SEQ_LEN
        label_length = tf.ones(shape=(batch_size, 1), dtype="int32") * CHARS_PER_LABEL

        inputs = {
            "image": image,
            "label": label,
            "input_length": input_length,
            "label_length": label_length,
        }
        outputs = tf.zeros(shape=(batch_size,))
        return inputs, outputs

    train_dataset = train_ds.map(prepare_for_ctc, num_parallel_calls=tf.data.AUTOTUNE)
    val_dataset = val_ds.map(prepare_for_ctc, num_parallel_calls=tf.data.AUTOTUNE)

    return train_dataset.prefetch(tf.data.AUTOTUNE), val_dataset.prefetch(
        tf.data.AUTOTUNE
    )


train_dataset, val_dataset = load_and_preprocess_data(DATASET_PATH)


# CRNN 模型架构
def build_crnn_model():
    """Builds the CRNN model architecture for prediction and a wrapped model for training."""
    image_input = layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 1), name="image", dtype="float32")

    # --- 优化 1: 归一化输入 ---
    x = layers.Rescaling(1.0 / 255)(image_input)

    # 1. 卷积层 (CNN) - 让卷积层接收归一化后的 x
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2, name="pool1")(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2, name="pool2")(x)
    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2, name="pool3")(x)

    # 2. Map-to-Sequence
    conv_shape = x.shape
    new_shape = (conv_shape[2], conv_shape[1] * conv_shape[3])
    x = layers.Reshape(target_shape=new_shape, name="reshape")(x)

    # --- 优化 3: 在 RNN 前稳定输入 ---
    x = layers.BatchNormalization()(x)

    x = layers.Dense(64, activation="relu", name="dense1")(x)
    x = layers.Dropout(0.2)(x)

    # 3. 循环层 (RNN)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.25))(x)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True, dropout=0.25))(x)

    # 4. 转录层
    output = layers.Dense(NUM_CLASSES, activation="softmax", name="output")(x)

    prediction_model = keras.Model(inputs=image_input, outputs=output, name="CRNN_Predictor")

    # ---- 为训练包装模型以使用 CTC Loss ----
    label_input = layers.Input(name="label", shape=[CHARS_PER_LABEL], dtype="int32")
    input_length_input = layers.Input(name="input_length", shape=[1], dtype="int32")
    label_length_input = layers.Input(name="label_length", shape=[1], dtype="int32")

    def ctc_loss_func(args):
        y_pred, y_true, input_len, label_len = args
        label_length = tf.squeeze(label_len, axis=-1)
        logit_length = tf.squeeze(input_len, axis=-1)
        loss = tf.nn.ctc_loss(
            labels=y_true,
            logits=y_pred,
            label_length=label_length,
            logit_length=logit_length,
            logits_time_major=False
        )
        return tf.reduce_mean(loss)

    loss_output = layers.Lambda(
        ctc_loss_func, output_shape=(1,), name="ctc_loss"
    )([output, label_input, input_length_input, label_length_input])

    training_model = keras.Model(
        inputs=[image_input, label_input, input_length_input, label_length_input],
        outputs=loss_output,
        name="CRNN_Trainer"
    )

    return training_model, prediction_model

# 构建模型
model, prediction_model = build_crnn_model()

# 编译训练模型
# IMPORTANT: 训练成功后，可以移除 run_eagerly=True 来提速
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss={"ctc_loss": lambda y_true, y_pred: y_pred},
    # run_eagerly=True, # 调试时使用，确认能运行后可注释掉
)

print("--- Training Model Summary ---")
model.summary()
print("\n--- Prediction Model Summary ---")
prediction_model.summary()

# 训练
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    callbacks=[
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.2, patience=3, min_lr=1e-5
        ),
    ],
)

# 保存用于推理的模型 (本地)
os.makedirs("models", exist_ok=True)
final_model_path = "models/luoguCaptcha_crnn.keras"
prediction_model.save(final_model_path)
print(f"Prediction model saved to {final_model_path}")

# 提示上传
print(f"Run `python scripts/huggingface.py upload_model {final_model_path}` to upload.")