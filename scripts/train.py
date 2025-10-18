# train.py
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
import numpy as np
from datasets import load_dataset
import matplotlib.pyplot as plt

# GPU配置
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"使用GPU: {gpus}")
    except Exception as e:
        print(f"GPU设置错误: {e}")
else:
    print("使用CPU")

# 超参数配置
class Config:
    # 数据集配置
    DATASET_PATH = "langningchen/luogu-captcha-dataset"
    IMG_HEIGHT = 35
    IMG_WIDTH = 90
    IMG_CHANNELS = 1
    
    # 验证码配置
    CHARS_PER_LABEL = 4
    CHAR_SIZE = 256
    N_CLASSES = CHAR_SIZE  # 不使用CTC，直接分类
    
    # 训练配置
    BATCH_SIZE = 256
    EPOCHS = 50
    LEARNING_RATE = 0.001
    
    # 路径配置
    MODEL_DIR = "models"
    CHECKPOINT_PATH = os.path.join(MODEL_DIR, "best_model.keras")
    FINAL_MODEL_PATH = os.path.join(MODEL_DIR, "luoguCaptcha_crnn.keras")

    # >>> 新增：TFRecord 缓存路径配置
    TFRECORD_DIR = "tfrecords"
    TRAIN_TFRECORD_PATH = os.path.join(TFRECORD_DIR, "train.tfrecord")
    TEST_TFRECORD_PATH = os.path.join(TFRECORD_DIR, "test.tfrecord")
    METADATA_PATH = os.path.join(TFRECORD_DIR, "metadata.json")

config = Config()

# 创建CRNN模型
class CRNNModel:
    def __init__(self, config):
        self.config = config
        self.model = self._build_model()
    
    def _build_model(self):
        # 🔄 使用更好的初始化
        from tensorflow.keras.initializers import HeNormal, GlorotUniform
        
        inputs = layers.Input(shape=(35, 90, 1))
        
        # CNN - 使用He初始化（适合ReLU）
        x = layers.Conv2D(32, (3, 3), padding='same', activation='relu',
                         kernel_initializer=HeNormal())(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        
        x = layers.Conv2D(64, (3, 3), padding='same', activation='relu',
                         kernel_initializer=HeNormal())(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        
        x = layers.Conv2D(128, (3, 3), padding='same', activation='relu',
                         kernel_initializer=HeNormal())(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        
        # RNN
        shape = x.shape
        x = layers.Reshape(target_shape=(shape[2], shape[1] * shape[3]))(x)
        x = layers.Dense(128, activation='relu', 
                        kernel_initializer=HeNormal())(x)
        x = layers.Dropout(0.3)(x)
        
        # LSTM - 使用Glorot初始化
        x = layers.Bidirectional(
            layers.LSTM(128, return_sequences=True, dropout=0.2,
                       kernel_initializer=GlorotUniform(),
                       recurrent_initializer='orthogonal')  # 🔄 重要
        )(x)
        
        x = layers.Bidirectional(
            layers.LSTM(128, return_sequences=True, dropout=0.2,
                       kernel_initializer=GlorotUniform(),
                       recurrent_initializer='orthogonal')  # 🔄 重要
        )(x)
        
        # 输出
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dense(512, activation='relu',
                        kernel_initializer=HeNormal())(x)
        x = layers.Dropout(0.4)(x)
        
        outputs = []
        for i in range(4):
            out = layers.Dense(256, activation='softmax',
                             kernel_initializer=GlorotUniform(),  # 🔄 使用Glorot
                             name=f'char_{i}')(x)
            outputs.append(out)
        
        return Model(inputs=inputs, outputs=outputs)
    
    def compile_model(self):
        """编译模型"""
        optimizer = keras.optimizers.Adam(learning_rate=self.config.LEARNING_RATE)
        
        # 每个输出使用稀疏分类交叉熵
        losses = {f'char_{i}': 'sparse_categorical_crossentropy' 
                  for i in range(self.config.CHARS_PER_LABEL)}
        
        metrics = {f'char_{i}': ['accuracy'] 
                   for i in range(self.config.CHARS_PER_LABEL)}
        
        self.model.compile(
            optimizer=optimizer,
            loss=losses,
            metrics=metrics
        )
        
        return self.model

import json
from tqdm import tqdm

class DataLoader:
    def __init__(self, config):
        self.config = config
        self.train_size = 0
        self.val_size = 0

    def _serialize_example(self, image, label):
        """将单个样本序列化为 TFRecord Example Proto"""
        feature = {
            "image": tf.train.Feature(float_list=tf.train.FloatList(value=image.flatten())),
            "label": tf.train.Feature(int64_list=tf.train.Int64List(value=label)),
        }
        return tf.train.Example(features=tf.train.Features(feature=feature)).SerializeToString()

    def _create_tfrecords(self):
        """从HuggingFace下载数据并创建TFRecord文件"""
        print(f"TFRecord文件未找到。开始从 '{self.config.DATASET_PATH}' 创建缓存...")
        os.makedirs(self.config.TFRECORD_DIR, exist_ok=True)
        
        # 1. 加载HF数据集
        dataset_dict = load_dataset(self.config.DATASET_PATH)
        
        # 2. 处理并写入训练集
        with tf.io.TFRecordWriter(self.config.TRAIN_TFRECORD_PATH) as writer:
            train_ds = dataset_dict["train"]
            self.train_size = len(train_ds)
            for example in tqdm(train_ds, desc="正在处理训练集"):
                image = np.array(example["image"], dtype=np.float32) / 255.0
                label = example["label"]
                writer.write(self._serialize_example(image, label))
        
        # 3. 处理并写入验证集
        with tf.io.TFRecordWriter(self.config.TEST_TFRECORD_PATH) as writer:
            val_ds = dataset_dict["test"]
            self.val_size = len(val_ds)
            for example in tqdm(val_ds, desc="正在处理验证集"):
                image = np.array(example["image"], dtype=np.float32) / 255.0
                label = example["label"]
                writer.write(self._serialize_example(image, label))
        
        # 4. 保存数据集大小元数据
        metadata = {"train_size": self.train_size, "val_size": self.val_size}
        with open(self.config.METADATA_PATH, 'w') as f:
            json.dump(metadata, f)
            
        print(f"TFRecord 缓存创建完成。训练集: {self.train_size}, 验证集: {self.val_size}")

    def _parse_tfrecord_fn(self, example_proto):
        """解析TFRecord Example Proto"""
        feature_description = {
            "image": tf.io.FixedLenFeature([self.config.IMG_HEIGHT * self.config.IMG_WIDTH], tf.float32),
            "label": tf.io.FixedLenFeature([self.config.CHARS_PER_LABEL], tf.int64),
        }
        example = tf.io.parse_single_example(example_proto, feature_description)
        image = tf.reshape(example["image"], (self.config.IMG_HEIGHT, self.config.IMG_WIDTH, self.config.IMG_CHANNELS))
        label = tf.cast(example["label"], tf.int32)
        
        # 为主输出和辅助输出都提供标签
        outputs = {f'char_{i}': label[i] for i in range(self.config.CHARS_PER_LABEL)}
        outputs.update({f'aux_char_{i}': label[i] for i in range(self.config.CHARS_PER_LABEL)})
        
        return image, outputs

    def _augment_data(self, image, labels):
        """应用数据增强"""
        image = tf.image.random_brightness(image, max_delta=0.1)
        image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
        image = tf.clip_by_value(image, 0.0, 1.0)
        return image, labels

    def get_datasets(self):
        """获取准备好的训练和验证数据集"""
        if not os.path.exists(self.config.TRAIN_TFRECORD_PATH):
            self._create_tfrecords()
        else:
            print("从现有TFRecord缓存加载数据...")
            with open(self.config.METADATA_PATH, 'r') as f:
                metadata = json.load(f)
            self.train_size = metadata["train_size"]
            self.val_size = metadata["val_size"]
            print(f"训练集大小: {self.train_size}")
            print(f"验证集大小: {self.val_size}")
    
        # 创建TFRecordDataset
        raw_train_ds = tf.data.TFRecordDataset(self.config.TRAIN_TFRECORD_PATH)
        raw_val_ds = tf.data.TFRecordDataset(self.config.TEST_TFRECORD_PATH)
        
        # 🔧 关键修改：先 map 再 batch
        train_dataset = (
            raw_train_ds
            .map(self._parse_tfrecord_fn, num_parallel_calls=tf.data.AUTOTUNE)  # ✅ 移到 batch 前
            .shuffle(self.train_size)
            .map(self._augment_data, num_parallel_calls=tf.data.AUTOTUNE)
            .batch(self.config.BATCH_SIZE, drop_remainder=True)  # ✅ batch 在 map 后
            .prefetch(tf.data.AUTOTUNE)
        )
        
        val_dataset = (
            raw_val_ds
            .map(self._parse_tfrecord_fn, num_parallel_calls=tf.data.AUTOTUNE)  # ✅ 移到 batch 前
            .batch(self.config.BATCH_SIZE, drop_remainder=True)  # ✅ batch 在 map 后
            .prefetch(tf.data.AUTOTUNE)
        )
        
        return train_dataset, val_dataset

    def get_steps(self, dataset_size):
        """计算每个epoch的步数"""
        return dataset_size // self.config.BATCH_SIZE

# 自定义回调 - 计算整体准确率
class CaptchaAccuracyCallback(keras.callbacks.Callback):
    def __init__(self, validation_data, config):
        super().__init__()
        self.validation_data = validation_data
        self.config = config
        self.best_accuracy = 0.0
    
    def on_epoch_end(self, epoch, logs=None):
        """在每个epoch结束时计算完整验证码准确率"""
        correct = 0
        total = 0
        
        for images, labels_dict in self.validation_data:
            predictions = self.model.predict(images, verbose=0)
            
            batch_size = images.shape[0]
            for i in range(batch_size):
                # 获取真实标签
                true_label = [labels_dict[f'char_{j}'].numpy()[i] 
                             for j in range(self.config.CHARS_PER_LABEL)]
                
                # 获取预测标签
                pred_label = [np.argmax(predictions[j][i]) 
                             for j in range(self.config.CHARS_PER_LABEL)]
                
                if true_label == pred_label:
                    correct += 1
                total += 1
        
        accuracy = correct / total if total > 0 else 0
        print(f"\n完整验证码准确率: {accuracy:.4f} ({correct}/{total})")
        
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            print(f"最佳准确率更新: {self.best_accuracy:.4f}")
        
        logs['val_captcha_accuracy'] = accuracy

# 主训练函数
def train():
    """主训练流程"""
    # 创建模型目录
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    
    # >>> 修改：加载数据的方式
    data_loader = DataLoader(config)
    train_dataset, val_dataset = data_loader.get_datasets()
    
    # 计算步数
    train_steps = data_loader.get_steps(data_loader.train_size)
    val_steps = data_loader.get_steps(data_loader.val_size)
    
    print(f"\n每epoch训练步数: {train_steps}")
    print(f"每epoch验证步数: {val_steps}")
    
    # 构建和编译模型
    crnn = CRNNModel(config)
    model = crnn.compile_model()
    
    print("\n模型架构:")
    model.summary()
    
    # 回调函数
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            config.CHECKPOINT_PATH,
            save_best_only=True,
            monitor='val_loss',
            mode='min',
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=8,
            restore_best_weights=True,
            verbose=1
        ),
        CaptchaAccuracyCallback(val_dataset, config),
        keras.callbacks.TensorBoard(
            log_dir='logs',
            histogram_freq=1
        )
    ]
    
    # 训练模型 - 明确指定步数
    print("\n开始训练...")
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=config.EPOCHS,
        steps_per_epoch=train_steps,  # 明确指定
        validation_steps=val_steps,    # 明确指定
        callbacks=callbacks,
        verbose=1
    )
    
    # 保存最终模型
    model.save(config.FINAL_MODEL_PATH)
    print(f"\n模型已保存到: {config.FINAL_MODEL_PATH}")

    plot_training_curves(history, config)
    
    return model, history


def plot_training_curves(history, config):
    """绘制训练曲线"""
    import os
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    
    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. 总Loss曲线
    axes[0, 0].plot(history.history['loss'], label='Train Loss', linewidth=2)
    axes[0, 0].plot(history.history['val_loss'], label='Val Loss', linewidth=2)
    axes[0, 0].set_title('Total Loss', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 各位置平均准确率
    train_acc_avg = []
    val_acc_avg = []
    for epoch in range(len(history.history['loss'])):
        train_avg = sum(history.history[f'char_{i}_accuracy'][epoch] for i in range(4)) / 4
        val_avg = sum(history.history[f'val_char_{i}_accuracy'][epoch] for i in range(4)) / 4
        train_acc_avg.append(train_avg)
        val_acc_avg.append(val_avg)
    
    axes[0, 1].plot(train_acc_avg, label='Train Avg Acc', linewidth=2)
    axes[0, 1].plot(val_acc_avg, label='Val Avg Acc', linewidth=2)
    axes[0, 1].set_title('Average Accuracy', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim([0, 1.05])
    
    # 3. 各位置准确率对比
    for i in range(4):
        axes[1, 0].plot(history.history[f'char_{i}_accuracy'], 
                       label=f'Char {i}', linewidth=1.5)
    axes[1, 0].set_title('Per-Position Accuracy (Train)', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim([0, 1.05])
    
    # 4. 过拟合诊断
    gap = [history.history['loss'][i] - history.history['val_loss'][i] 
           for i in range(len(history.history['loss']))]
    axes[1, 1].plot(gap, label='Train - Val Loss Gap', linewidth=2, color='red')
    axes[1, 1].axhline(y=0, color='green', linestyle='--', label='No Gap')
    axes[1, 1].set_title('Overfitting Diagnosis', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss Gap')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存
    save_path = os.path.join(config.MODEL_DIR, 'training_curves.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ 训练曲线已保存: {save_path}")
    
    plt.close()

if __name__ == "__main__":
    model, history = train()
    print("\n训练完成！")