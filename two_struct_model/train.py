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
    
    # 训练配置 - 阶段1 (CNN+Dense)
    BATCH_SIZE = 256
    EPOCHS_STAGE1 = 15
    LEARNING_RATE_STAGE1 = 0.001
    
    # 训练配置 - 阶段2 (CNN冻结+LSTM)
    EPOCHS_STAGE2 = 30
    LEARNING_RATE_STAGE2 = 0.0005
    
    # 路径配置
    MODEL_DIR = "models"
    STAGE1_MODEL_PATH = os.path.join(MODEL_DIR, "stage1_cnn_dense.keras")
    STAGE2_MODEL_PATH = os.path.join(MODEL_DIR, "stage2_cnn_lstm.keras")
    FINAL_MODEL_PATH = os.path.join(MODEL_DIR, "luoguCaptcha_final.keras")
    
    # TFRecord 缓存路径配置
    TFRECORD_DIR = "tfrecords"
    TRAIN_TFRECORD_PATH = os.path.join(TFRECORD_DIR, "train.tfrecord")
    TEST_TFRECORD_PATH = os.path.join(TFRECORD_DIR, "test.tfrecord")
    METADATA_PATH = os.path.join(TFRECORD_DIR, "metadata.json")

config = Config()


class ModelBuilder:
    """模型构建器 - 严格按照表现良好的架构"""
    
    def __init__(self, config):
        self.config = config
    
    def build_stage1_model(self):
        """阶段1: 严格按照你提供的CNN+Dense架构"""
        inputs = keras.Input(
            shape=(self.config.IMG_HEIGHT, self.config.IMG_WIDTH, self.config.IMG_CHANNELS), 
            name="input"
        )
        
        # 完全按照你的架构
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
        x = layers.Dense(
            self.config.CHARS_PER_LABEL * self.config.CHAR_SIZE, 
            activation="softmax"
        )(x)
        outputs = layers.Reshape((self.config.CHARS_PER_LABEL, self.config.CHAR_SIZE))(x)
        
        model = keras.Model(inputs=inputs, outputs=outputs, name="Stage1_LuoguCaptcha")
        return model
    
    def build_stage2_model(self, pretrained_stage1_path):
        """阶段2: 保留CNN骨干，替换为LSTM"""
        # 加载阶段1模型
        stage1_model = keras.models.load_model(pretrained_stage1_path)
        
        # 创建新输入
        inputs = keras.Input(
            shape=(self.config.IMG_HEIGHT, self.config.IMG_WIDTH, self.config.IMG_CHANNELS), 
            name="input"
        )
        
        # 提取CNN部分（到最后一个MaxPooling2D）
        x = inputs
        for layer in stage1_model.layers:
            if isinstance(layer, layers.MaxPooling2D):
                # 冻结CNN层
                layer.trainable = False
                x = layer(x)
            elif isinstance(layer, (layers.Conv2D, layers.BatchNormalization)):
                # 冻结CNN层
                layer.trainable = False
                x = layer(x)
            elif isinstance(layer, (layers.Flatten, layers.Dense, layers.Dropout, layers.Reshape)):
                # 停止，不使用Dense部分
                break
        
        # 保存CNN输出的shape用于Reshape
        cnn_output = x
        shape = cnn_output.shape  # (batch, height, width, channels)
        
        # 将CNN特征图转换为序列 (时间步, 特征)
        # shape[1] = height, shape[2] = width, shape[3] = channels
        x = layers.Reshape(target_shape=(shape[2], shape[1] * shape[3]), name='reshape')(x)
        
        # LSTM层
        x = layers.Dense(128, activation='relu', name='dense_before_lstm')(x)
        x = layers.Dropout(0.3)(x)
        
        x = layers.Bidirectional(
            layers.LSTM(128, return_sequences=True, dropout=0.2),
            name='bilstm1'
        )(x)
        x = layers.Bidirectional(
            layers.LSTM(128, return_sequences=False, dropout=0.2),
            name='bilstm2'
        )(x)
        
        # 分类头
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(
            self.config.CHARS_PER_LABEL * self.config.CHAR_SIZE, 
            activation="softmax"
        )(x)
        outputs = layers.Reshape((self.config.CHARS_PER_LABEL, self.config.CHAR_SIZE))(x)
        
        model = keras.Model(inputs=inputs, outputs=outputs, name="Stage2_CNN_LSTM")
        return model
    
    def compile_model(self, model, learning_rate):
        """编译模型 - 使用和你相同的配置"""
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model


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
        
        dataset_dict = load_dataset(self.config.DATASET_PATH)
        
        # 处理训练集
        with tf.io.TFRecordWriter(self.config.TRAIN_TFRECORD_PATH) as writer:
            train_ds = dataset_dict["train"]
            self.train_size = len(train_ds)
            for example in tqdm(train_ds, desc="正在处理训练集"):
                image = np.array(example["image"], dtype=np.float32)
                # 判断是否需要归一化（检查值是否在0-1范围内）
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
                # 判断是否需要归一化（检查值是否在0-1范围内）
                if image.max() > 1.0:
                    image = image / 255.0
                label = example["label"]
                writer.write(self._serialize_example(image, label))
        
        metadata = {"train_size": self.train_size, "val_size": self.val_size}
        with open(self.config.METADATA_PATH, 'w') as f:
            json.dump(metadata, f)
            
        print(f"TFRecord 缓存创建完成。训练集: {self.train_size}, 验证集: {self.val_size}")

    def _parse_tfrecord_fn(self, example_proto):
        """解析TFRecord Example Proto - 返回 (image, label) 格式"""
        feature_description = {
            "image": tf.io.FixedLenFeature([self.config.IMG_HEIGHT * self.config.IMG_WIDTH], tf.float32),
            "label": tf.io.FixedLenFeature([self.config.CHARS_PER_LABEL], tf.int64),
        }
        example = tf.io.parse_single_example(example_proto, feature_description)
        image = tf.reshape(example["image"], (self.config.IMG_HEIGHT, self.config.IMG_WIDTH, self.config.IMG_CHANNELS))
        label = tf.cast(example["label"], tf.int32)
        
        # 直接返回 (image, label)，label shape 为 (4,)
        return image, label

    def _augment_data(self, image, label):
        """应用数据增强"""
        image = tf.image.random_brightness(image, max_delta=0.1)
        image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
        image = tf.clip_by_value(image, 0.0, 1.0)
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
            .map(self._augment_data, num_parallel_calls=tf.data.AUTOTUNE)
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

    def get_steps(self, dataset_size):
        """计算每个epoch的步数"""
        return dataset_size // self.config.BATCH_SIZE


class CaptchaAccuracyCallback(keras.callbacks.Callback):
    """计算完整验证码准确率"""
    def __init__(self, validation_data, config, stage_name=""):
        super().__init__()
        self.validation_data = validation_data
        self.config = config
        self.best_accuracy = 0.0
        self.stage_name = stage_name
    
    def on_epoch_end(self, epoch, logs=None):
        correct = 0
        total = 0
        
        for images, labels in self.validation_data:
            predictions = self.model.predict(images, verbose=0)
            # predictions shape: (batch, 4, 256)
            # labels shape: (batch, 4)
            
            batch_size = images.shape[0]
            for i in range(batch_size):
                true_label = labels[i].numpy()  # (4,)
                pred_label = np.argmax(predictions[i], axis=1)  # (4,)
                
                if np.array_equal(true_label, pred_label):
                    correct += 1
                total += 1
        
        accuracy = correct / total if total > 0 else 0
        print(f"\n[{self.stage_name}] 完整验证码准确率: {accuracy:.4f} ({correct}/{total})")
        
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            print(f"[{self.stage_name}] 最佳准确率更新: {self.best_accuracy:.4f}")
        
        logs['val_captcha_accuracy'] = accuracy


def plot_training_curves(history, config, stage_name, save_path):
    """绘制训练曲线"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Loss曲线
    axes[0, 0].plot(history.history['loss'], label='Train Loss', linewidth=2)
    axes[0, 0].plot(history.history['val_loss'], label='Val Loss', linewidth=2)
    axes[0, 0].set_title(f'{stage_name} - Loss', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 准确率曲线
    axes[0, 1].plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
    axes[0, 1].plot(history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
    axes[0, 1].set_title(f'{stage_name} - Accuracy', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim([0, 1.05])
    
    # 3. 完整验证码准确率（如果有）
    if 'val_captcha_accuracy' in history.history:
        axes[1, 0].plot(history.history['val_captcha_accuracy'], 
                       label='Captcha Accuracy', linewidth=2, color='green')
        axes[1, 0].set_title(f'{stage_name} - Full Captcha Accuracy', 
                           fontsize=14, fontweight='bold')
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
    axes[1, 1].set_title(f'{stage_name} - Overfitting Diagnosis', 
                        fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss Gap')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ {stage_name} 训练曲线已保存: {save_path}")
    plt.close()


def train_stage1(train_dataset, val_dataset, data_loader, config):
    """阶段1训练: CNN + Dense（严格按照你的架构）"""
    print("\n" + "="*80)
    print("阶段1: 训练 CNN + Dense 基础模型")
    print("="*80)
    
    # 构建模型
    model_builder = ModelBuilder(config)
    model = model_builder.build_stage1_model()
    model = model_builder.compile_model(model, config.LEARNING_RATE_STAGE1)
    
    print("\n阶段1 模型架构:")
    model.summary()
    
    # 回调函数 - 使用和你相同的配置
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            config.STAGE1_MODEL_PATH,
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
        CaptchaAccuracyCallback(val_dataset, config, stage_name="Stage1"),
        keras.callbacks.TensorBoard(
            log_dir='logs/stage1',
            histogram_freq=1
        )
    ]
    
    # 训练
    print("\n开始阶段1训练...")
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=config.EPOCHS_STAGE1,
        callbacks=callbacks,
        verbose=1
    )
    
    # 保存模型
    model.save(config.STAGE1_MODEL_PATH)
    print(f"\n✅ 阶段1模型已保存到: {config.STAGE1_MODEL_PATH}")
    
    # 绘制训练曲线
    plot_training_curves(
        history, 
        config, 
        "Stage1 (CNN+Dense)", 
        os.path.join(config.MODEL_DIR, 'stage1_training_curves.png')
    )
    
    return model, history


def train_stage2(train_dataset, val_dataset, data_loader, config):
    """阶段2训练: 冻结CNN + LSTM"""
    print("\n" + "="*80)
    print("阶段2: 冻结CNN，训练 LSTM 层")
    print("="*80)
    
    # 检查阶段1模型是否存在
    if not os.path.exists(config.STAGE1_MODEL_PATH):
        raise FileNotFoundError(
            f"未找到阶段1模型: {config.STAGE1_MODEL_PATH}\n"
            "请先运行阶段1训练或确保模型文件存在"
        )
    
    # 构建模型
    model_builder = ModelBuilder(config)
    model = model_builder.build_stage2_model(config.STAGE1_MODEL_PATH)
    model = model_builder.compile_model(model, config.LEARNING_RATE_STAGE2)
    
    print("\n阶段2 模型架构:")
    model.summary()
    
    # 显示冻结层信息
    print("\n冻结层信息:")
    frozen_layers = [layer.name for layer in model.layers if not layer.trainable]
    trainable_layers = [layer.name for layer in model.layers if layer.trainable]
    print(f"冻结层 ({len(frozen_layers)}): {frozen_layers}")
    print(f"可训练层 ({len(trainable_layers)}): {trainable_layers}")
    
    # 回调函数
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            config.STAGE2_MODEL_PATH,
            save_best_only=True,
            monitor='val_loss',
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss", 
            patience=8, 
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", 
            factor=0.2, 
            patience=4, 
            min_lr=1e-6
        ),
        CaptchaAccuracyCallback(val_dataset, config, stage_name="Stage2"),
        keras.callbacks.TensorBoard(
            log_dir='logs/stage2',
            histogram_freq=1
        )
    ]
    
    # 训练
    print("\n开始阶段2训练...")
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=config.EPOCHS_STAGE2,
        callbacks=callbacks,
        verbose=1
    )
    
    # 保存模型
    model.save(config.STAGE2_MODEL_PATH)
    print(f"\n✅ 阶段2模型已保存到: {config.STAGE2_MODEL_PATH}")
    
    # 绘制训练曲线
    plot_training_curves(
        history, 
        config, 
        "Stage2 (CNN_frozen+LSTM)", 
        os.path.join(config.MODEL_DIR, 'stage2_training_curves.png')
    )
    
    return model, history


def evaluate_and_compare(val_dataset, config):
    """评估并对比两个阶段的模型"""
    print("\n" + "="*80)
    print("模型评估与对比")
    print("="*80)
    
    # 加载两个模型
    stage1_model = keras.models.load_model(config.STAGE1_MODEL_PATH)
    stage2_model = keras.models.load_model(config.STAGE2_MODEL_PATH)
    
    def calculate_accuracy(model, dataset, model_name):
        """计算完整验证码准确率"""
        correct = 0
        total = 0
        
        for images, labels in dataset:
            predictions = model.predict(images, verbose=0)
            
            batch_size = images.shape[0]
            for i in range(batch_size):
                true_label = labels[i].numpy()
                pred_label = np.argmax(predictions[i], axis=1)
                
                if np.array_equal(true_label, pred_label):
                    correct += 1
                total += 1
        
        accuracy = correct / total if total > 0 else 0
        print(f"\n{model_name} 完整验证码准确率: {accuracy:.4f} ({correct}/{total})")
        return accuracy
    
    # 评估验证集
    print("\n验证集评估:")
    stage1_acc = calculate_accuracy(stage1_model, val_dataset, "阶段1 (CNN+Dense)")
    stage2_acc = calculate_accuracy(stage2_model, val_dataset, "阶段2 (CNN+LSTM)")
    
    # 对比结果
    print("\n" + "-"*80)
    print("对比结果:")
    print(f"阶段1准确率: {stage1_acc:.4f}")
    print(f"阶段2准确率: {stage2_acc:.4f}")
    improvement = stage2_acc - stage1_acc
    print(f"提升幅度: {improvement:+.4f} ({improvement*100:+.2f}%)")
    print("-"*80)
    
    # 选择最佳模型作为最终模型
    if stage2_acc >= stage1_acc:
        best_model = stage2_model
        best_model_name = "阶段2 (CNN+LSTM)"
        print(f"\n✅ 最终模型: {best_model_name}")
    else:
        best_model = stage1_model
        best_model_name = "阶段1 (CNN+Dense)"
        print(f"\n✅ 最终模型: {best_model_name} (阶段2未能提升性能)")
    
    # 保存最终模型
    best_model.save(config.FINAL_MODEL_PATH)
    print(f"✅ 最终模型已保存到: {config.FINAL_MODEL_PATH}")
    
    return stage1_acc, stage2_acc


def train():
    """主训练流程"""
    # 创建模型目录
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    
    # 加载数据
    print("加载数据集...")
    data_loader = DataLoader(config)
    train_dataset, val_dataset = data_loader.get_datasets()
    
    print(f"\n数据集信息:")
    print(f"训练集大小: {data_loader.train_size}")
    print(f"验证集大小: {data_loader.val_size}")
    print(f"每epoch训练步数: {data_loader.get_steps(data_loader.train_size)}")
    print(f"每epoch验证步数: {data_loader.get_steps(data_loader.val_size)}")
    
    # 阶段1: 训练 CNN + Dense
    stage1_model, stage1_history = train_stage1(
        train_dataset, 
        val_dataset, 
        data_loader, 
        config
    )
    
    # 阶段2: 冻结CNN + 训练LSTM
    stage2_model, stage2_history = train_stage2(
        train_dataset, 
        val_dataset, 
        data_loader, 
        config
    )
    
    # 评估对比
    stage1_acc, stage2_acc = evaluate_and_compare(
        val_dataset, 
        config
    )
    
    print("\n" + "="*80)
    print("训练完成！")
    print("="*80)
    print(f"阶段1模型: {config.STAGE1_MODEL_PATH}")
    print(f"阶段2模型: {config.STAGE2_MODEL_PATH}")
    print(f"最终模型: {config.FINAL_MODEL_PATH}")
    print(f"\n阶段1准确率: {stage1_acc:.4f}")
    print(f"阶段2准确率: {stage2_acc:.4f}")
    print(f"性能提升: {(stage2_acc - stage1_acc)*100:+.2f}%")
    print("="*80)
    
    return {
        'stage1_model': stage1_model,
        'stage1_history': stage1_history,
        'stage1_accuracy': stage1_acc,
        'stage2_model': stage2_model,
        'stage2_history': stage2_history,
        'stage2_accuracy': stage2_acc
    }


if __name__ == "__main__":
    results = train()
    print("\n✅ 所有训练流程已完成！")