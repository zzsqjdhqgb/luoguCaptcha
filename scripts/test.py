import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, Bidirectional, LSTM, Dense, Reshape, Lambda
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K

def build_crnn_model(input_shape, num_classes):
    """
    构建 CRNN 模型。

    :param input_shape: 输入图像的尺寸 (height, width, channels)
    :param num_classes: 字符集的类别数量（包括空白符）
    :return: 编译好的 CRNN 模型
    """
    # 输入层
    inputs = Input(name='the_input', shape=input_shape, dtype='float32')

    # 卷积层 (CNN)
    inner = Conv2D(64, (3, 3), padding='same', name='conv1', kernel_initializer='he_normal')(inputs)
    inner = BatchNormalization()(inner)
    inner = tf.keras.layers.ReLU()(inner)
    inner = MaxPooling2D(pool_size=(2, 2), name='pool1')(inner)

    inner = Conv2D(128, (3, 3), padding='same', name='conv2', kernel_initializer='he_normal')(inner)
    inner = BatchNormalization()(inner)
    inner = tf.keras.layers.ReLU()(inner)
    inner = MaxPooling2D(pool_size=(2, 2), name='pool2')(inner)

    inner = Conv2D(256, (3, 3), padding='same', name='conv3', kernel_initializer='he_normal')(inner)
    inner = BatchNormalization()(inner)
    inner = tf.keras.layers.ReLU()(inner)
    inner = Conv2D(256, (3, 3), padding='same', name='conv4', kernel_initializer='he_normal')(inner)
    inner = BatchNormalization()(inner)
    inner = tf.keras.layers.ReLU()(inner)
    inner = MaxPooling2D(pool_size=(1, 2), name='pool3')(inner)

    inner = Conv2D(512, (3, 3), padding='same', name='conv5', kernel_initializer='he_normal')(inner)
    inner = BatchNormalization()(inner)
    inner = tf.keras.layers.ReLU()(inner)
    inner = Conv2D(512, (3, 3), padding='same', name='conv6', kernel_initializer='he_normal')(inner)
    inner = BatchNormalization()(inner)
    inner = tf.keras.layers.ReLU()(inner)
    inner = MaxPooling2D(pool_size=(1, 2), name='pool4')(inner)

    inner = Conv2D(512, (2, 2), padding='same', name='conv7', kernel_initializer='he_normal')(inner)
    inner = BatchNormalization()(inner)
    inner = tf.keras.layers.ReLU()(inner)

    # 将 CNN 的输出特征图转换为序列
    # 获取特征图的形状
    _, h, w, c = inner.shape
    # Reshape to (batch_size, width, height * channels)
    inner = Reshape(target_shape=(w, h * c), name='reshape')(inner)
    
    # 循环层 (RNN)
    # 使用双向 LSTM 来捕捉序列的前后文信息
    inner = Bidirectional(LSTM(256, return_sequences=True), name='bidirectional_lstm_1')(inner)
    inner = Bidirectional(LSTM(256, return_sequences=True), name='bidirectional_lstm_2')(inner)

    # 全连接层 + Softmax
    # 将 RNN 的输出映射到字符类别
    outputs = Dense(num_classes, activation='softmax', name='dense_output')(inner)

    # 定义模型
    model = Model(inputs=inputs, outputs=outputs)
    
    return model

# 示例：定义 CTC Loss 函数
# Keras 的 ctc_batch_cost 已经不推荐使用，建议直接使用 tf.nn.ctc_loss
def ctc_loss_func(y_true, y_pred):
    batch_len = tf.cast(tf.shape(y_pred)[0], dtype="int32")
    input_length = tf.cast(tf.shape(y_pred)[1], dtype="int32")
    label_length = tf.cast(tf.shape(y_true)[1], dtype="int32")

    input_length = input_length * tf.ones(shape=(batch_len,), dtype="int32")
    label_length = label_length * tf.ones(shape=(batch_len,), dtype="int32")

    # 使用 tf.nn.ctc_loss
    loss = K.ctc_batch_cost(y_true, y_pred, input_length, label_length)
    return loss


# 定义模型参数
IMG_HEIGHT = 32
IMG_WIDTH = 128
CHANNELS = 1
NUM_CLASSES = 80 # 假设字符集大小为 79 + 1 (空白符)

# 构建模型
crnn_model = build_crnn_model((IMG_HEIGHT, IMG_WIDTH, CHANNELS), NUM_CLASSES)
crnn_model.summary()

# 编译模型
# 这里需要一个自定义的 CTC loss 函数
# Keras 的 ctc_batch_cost 已经封装了 tf.nn.ctc_loss 的逻辑
crnn_model.compile(optimizer='adam', loss=ctc_loss_func, metrics=['accuracy'])