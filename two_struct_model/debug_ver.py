# train.py
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
import numpy as np
from datasets import load_dataset
import matplotlib.pyplot as plt
import json
from tqdm import tqdm

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
    
    # 训练配置
    BATCH_SIZE = 256
    EPOCHS = 15
    LEARNING_RATE = 0.001
    
    # 路径配置
    MODEL_DIR = "models"
    MODEL_PATH = os.path.join(MODEL_DIR, "luoguCaptcha.keras")
    
    # TFRecord 缓存路径配置
    TFRECORD_DIR = "tfrecords"
    TRAIN_TFRECORD_PATH = os.path.join(TFRECORD_DIR, "train.tfrecord")
    TEST_TFRECORD_PATH = os.path.join(TFRECORD_DIR, "test.tfrecord")
    METADATA_PATH = os.path.join(TFRECORD_DIR, "metadata.json")

config = Config()


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
        print(f"TFRecord文件未找到。开始从 '{self.config.DATASET_PATH}' 创建缓存...")
        os.makedirs(self.config.TFRECORD_DIR, exist_ok=True)

        dataset_dict = load_dataset(self.config.DATASET_PATH)

        # 处理训练集
        with tf.io.TFRecordWriter(self.config.TRAIN_TFRECORD_PATH) as writer:
            train_ds = dataset_dict["train"]
            self.train_size = len(train_ds)
            for example in tqdm(train_ds, desc="正在处理训练集"):
                image = np.array(example["image"], dtype=np.float32)
                # 修改：只在需要时归一化
                if image.max() > 1.0:
                    image = image / 255.0
                label = example["label"]
                writer.write(self._serialize_example(image, label))
        
        # 处理验证集
        with tf.io.TFRecordWriter(self.config.TEST_TFRECORD_PATH) as writer:
            val_ds = dataset_dict["test"]
            self.val_size = len(val_ds)
            for example in tqdm(val_ds, desc="正在处理验证集"):
                image = np.array(example["image"], dtype=np.float32)
                # 判断是否需要归一化
                if image.max() > 1.0:
                    image = image / 255.0
                label = example["label"]
                writer.write(self._serialize_example(image, label))
        
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
        
        return image, label

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

        raw_train_ds = tf.data.TFRecordDataset(self.config.TRAIN_TFRECORD_PATH)
        raw_val_ds = tf.data.TFRecordDataset(self.config.TEST_TFRECORD_PATH)
        
        train_dataset = (
            raw_train_ds
            .map(self._parse_tfrecord_fn, num_parallel_calls=tf.data.AUTOTUNE)
            .shuffle(min(10000, self.train_size))
            .batch(self.config.BATCH_SIZE)
            .prefetch(tf.data.AUTOTUNE)
        )
        
        val_dataset = (
            raw_val_ds
            .map(self._parse_tfrecord_fn, num_parallel_calls=tf.data.AUTOTUNE)
            .batch(self.config.BATCH_SIZE)
            .prefetch(tf.data.AUTOTUNE)
        )
        
        return train_dataset, val_dataset


def build_model(config):
    """构建原版模型 - 修正版"""
    inputs = keras.Input(
        shape=(config.IMG_HEIGHT, config.IMG_WIDTH, config.IMG_CHANNELS), 
        name="input"
    )
    
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    
    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    
    x = layers.Flatten()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(512, activation="relu")(x)
    
    # 关键修改：先不用 softmax，改用 linear，然后用 from_logits=True
    x = layers.Dense(config.CHARS_PER_LABEL * config.CHAR_SIZE)(x)  # 去掉 activation
    outputs = layers.Reshape((config.CHARS_PER_LABEL, config.CHAR_SIZE))(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name="LuoguCaptcha")
    
    # 使用 from_logits=True
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    
    return model


class CaptchaAccuracyCallback(keras.callbacks.Callback):
    """计算完整验证码准确率"""
    def __init__(self, validation_data, config):
        super().__init__()
        self.validation_data = validation_data
        self.config = config
        self.best_accuracy = 0.0
    
    def on_epoch_end(self, epoch, logs=None):
        correct = 0
        total = 0
        
        for images, labels in self.validation_data:
            predictions = self.model.predict(images, verbose=0)
            
            batch_size = images.shape[0]
            for i in range(batch_size):
                true_label = labels[i].numpy()
                pred_label = np.argmax(predictions[i], axis=1)
                
                if np.array_equal(true_label, pred_label):
                    correct += 1
                total += 1
        
        accuracy = correct / total if total > 0 else 0
        print(f"\n完整验证码准确率: {accuracy:.4f} ({correct}/{total})")
        
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            print(f"最佳准确率更新: {self.best_accuracy:.4f}")
        
        logs['val_captcha_accuracy'] = accuracy


def plot_training_curves(history, config):
    """绘制训练曲线"""
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Loss曲线
    axes[0, 0].plot(history.history['loss'], label='Train Loss', linewidth=2)
    axes[0, 0].plot(history.history['val_loss'], label='Val Loss', linewidth=2)
    axes[0, 0].set_title('Loss', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 准确率曲线
    axes[0, 1].plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
    axes[0, 1].plot(history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
    axes[0, 1].set_title('Accuracy', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim([0, 1.05])
    
    # 3. 完整验证码准确率
    if 'val_captcha_accuracy' in history.history:
        axes[1, 0].plot(history.history['val_captcha_accuracy'], 
                       label='Captcha Accuracy', linewidth=2, color='green')
        axes[1, 0].set_title('Full Captcha Accuracy', fontsize=14, fontweight='bold')
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
    
    save_path = os.path.join(config.MODEL_DIR, 'training_curves.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ 训练曲线已保存: {save_path}")
    plt.close()


def train():
    """主训练流程"""
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    
    # 加载数据
    print("加载数据集...")
    data_loader = DataLoader(config)
    train_dataset, val_dataset = data_loader.get_datasets()
    
    # 构建模型
    print("\n构建模型...")
    model = build_model(config)
    
    print("\n模型架构:")
    model.summary()
    
    # 回调函数
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            config.MODEL_PATH,
            save_best_only=True,
            monitor='val_loss',
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss", 
            patience=5, 
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", 
            factor=0.2, 
            patience=3, 
            min_lr=1e-5
        ),
        CaptchaAccuracyCallback(val_dataset, config),
        keras.callbacks.TensorBoard(
            log_dir='logs',
            histogram_freq=1
        )
    ]
    
    # 训练
    print("\n开始训练...")
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=config.EPOCHS,
        callbacks=callbacks,
        verbose=1
    )
    
    # 保存模型
    model.save(config.MODEL_PATH)
    print(f"\n✅ 模型已保存到: {config.MODEL_PATH}")
    
    # 绘制训练曲线
    plot_training_curves(history, config)
    
    print("\n" + "="*80)
    print("训练完成！")
    print("="*80)
    
    return model, history


if __name__ == "__main__":
    model, history = train()
    print("\n✅ 所有流程已完成！")