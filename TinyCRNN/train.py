import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from device_config import setup_device
from data_loader import load_and_preprocess_data

# 自动选择设备
setup_device()

# 参数
CHAR_SIZE = 256
CHARS_PER_LABEL = 4
IMG_HEIGHT, IMG_WIDTH = 35, 90
EPOCHS_STAGE1 = 10  # CNN+Dense 阶段
EPOCHS_STAGE2 = 10  # 冻结CNN+LSTM 阶段
EPOCHS_STAGE3 = 50  # 解冻全部模型阶段
BATCH_SIZE = 256
TFRECORD_DIR = "data/luogu_captcha_tfrecord"

# Load data
try:
    train_dataset, val_dataset = load_and_preprocess_data(
        TFRECORD_DIR, 
        batch_size=BATCH_SIZE,
        chars_per_label=CHARS_PER_LABEL
    )
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


# 定义残差块
def residual_block(x, filters, kernel_size, strides=1, use_pooling=False):
    """
    残差块
    Args:
        x: 输入tensor
        filters: 卷积核数量
        kernel_size: 卷积核大小
        strides: 步长
        use_pooling: 是否使用池化
    """
    shortcut = x
    
    # 主路径
    x = layers.Conv2D(filters, kernel_size, strides=strides, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(filters, kernel_size, padding="same")(x)
    x = layers.BatchNormalization()(x)
    
    # 如果维度不匹配，调整shortcut
    if shortcut.shape[-1] != filters or strides != 1:
        shortcut = layers.Conv2D(filters, 1, strides=strides, padding="same")(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)
    
    # 残差连接
    x = layers.Add()([shortcut, x])
    x = layers.Activation("relu")(x)
    
    # 可选的池化层
    if use_pooling:
        x = layers.MaxPooling2D(2)(x)
    
    return x


# 构建CNN部分（带残差连接）
def build_cnn_with_residual(inputs):
    """构建带残差连接的CNN"""
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    
    # 残差块1
    x = residual_block(x, 32, 3, use_pooling=True)
    
    # 残差块2
    x = residual_block(x, 64, 4, use_pooling=True)
    
    # 残差块3
    x = residual_block(x, 128, 5, use_pooling=True)
    
    return x


# ========== 阶段1: CNN+Dense 训练10个epoch ==========
print("\n" + "="*50)
print("Stage 1: Training CNN+Dense for 10 epochs (with Residual Connections)")
print("="*50 + "\n")

inputs = keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 1), name="input")
x = build_cnn_with_residual(inputs)
cnn_output = x  # 保存CNN输出用于后续构建LSTM模型
x = layers.Flatten()(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(256, activation="relu")(x)
x = layers.Dropout(0.4)(x)
x = layers.Dense(CHARS_PER_LABEL * CHAR_SIZE, activation="softmax")(x)
outputs = layers.Reshape((CHARS_PER_LABEL, CHAR_SIZE))(x)

model = keras.Model(inputs=inputs, outputs=outputs, name="LuoguCaptcha_Stage1_ResNet")
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
x = build_cnn_with_residual(inputs_lstm)

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

model_lstm = keras.Model(inputs=inputs_lstm, outputs=outputs_lstm, name="LuoguCaptcha_Stage2_ResNet")

# 复制CNN权重（找到CNN部分的层）
# 由于使用了函数式构建，需要匹配层
cnn_layers_stage1 = []
cnn_layers_stage2 = []

for layer in model.layers:
    if 'flatten' in layer.name.lower() or 'dense' in layer.name.lower() or 'dropout' in layer.name.lower() or 'reshape' in layer.name.lower():
        break
    if layer.trainable and len(layer.get_weights()) > 0:
        cnn_layers_stage1.append(layer)

for layer in model_lstm.layers:
    if 'reshape' in layer.name.lower() or 'lstm' in layer.name.lower() or 'dense' in layer.name.lower() or 'dropout' in layer.name.lower():
        break
    if layer.trainable and len(layer.get_weights()) > 0:
        cnn_layers_stage2.append(layer)

# 复制权重并冻结
print(f"Copying weights from {len(cnn_layers_stage1)} CNN layers")
for layer1, layer2 in zip(cnn_layers_stage1, cnn_layers_stage2):
    layer2.set_weights(layer1.get_weights())
    layer2.trainable = False
    print(f"  Copied and froze: {layer2.name}")

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
final_model_path = "models/luoguCaptcha.CRNN.resnet.keras"
model_lstm.save(final_model_path)
print(f"Model saved to {final_model_path}")

# 提示上传
print(f"Run `python scripts/huggingface.py upload_model {final_model_path}` to upload.")