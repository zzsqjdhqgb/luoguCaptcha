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
    N_CLASSES = CHAR_SIZE
    
    # 训练配置 - 阶段1 (CNN+Dense)
    BATCH_SIZE = 256
    EPOCHS_STAGE1 = 20  # 第一阶段训练轮数
    LEARNING_RATE_STAGE1 = 0.001
    
    # 训练配置 - 阶段2 (CNN冻结+LSTM)
    EPOCHS_STAGE2 = 30  # 第二阶段训练轮数
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
    """模型构建器 - 支持两阶段架构"""
    
    def __init__(self, config):
        self.config = config
    
    def build_cnn_backbone(self, inputs):
        """构建CNN骨干网络（两阶段共享）"""
        x = layers.Conv2D(32, (3, 3), padding='same', activation='relu', name='conv1')(inputs)
        x = layers.BatchNormalization(name='bn1')(x)
        x = layers.MaxPooling2D(pool_size=(2, 2), name='pool1')(x)
        
        x = layers.Conv2D(64, (3, 3), padding='same', activation='relu', name='conv2')(x)
        x = layers.BatchNormalization(name='bn2')(x)
        x = layers.MaxPooling2D(pool_size=(2, 2), name='pool2')(x)
        
        x = layers.Conv2D(128, (3, 3), padding='same', activation='relu', name='conv3')(x)
        x = layers.BatchNormalization(name='bn3')(x)
        x = layers.MaxPooling2D(pool_size=(2, 2), name='pool3')(x)
        
        x = layers.Conv2D(256, (3, 3), padding='same', activation='relu', name='conv4')(x)
        x = layers.BatchNormalization(name='bn4')(x)
        x = layers.MaxPooling2D(pool_size=(2, 2), name='pool4')(x)
        
        return x
    
    def build_stage1_model(self):
        """阶段1: CNN + Dense (稳定训练)"""
        inputs = layers.Input(
            shape=(self.config.IMG_HEIGHT, self.config.IMG_WIDTH, self.config.IMG_CHANNELS),
            name='input_image'
        )
        
        # CNN骨干
        x = self.build_cnn_backbone(inputs)
        
        # Dense分类头
        x = layers.Flatten(name='flatten')(x)
        x = layers.Dropout(0.5, name='dropout1')(x)
        x = layers.Dense(512, activation='relu', name='dense_512')(x)
        x = layers.Dropout(0.4, name='dropout2')(x)
        
        # 多输出层
        outputs = []
        for i in range(self.config.CHARS_PER_LABEL):
            out = layers.Dense(
                self.config.N_CLASSES,
                activation='softmax',
                name=f'char_{i}'
            )(x)
            outputs.append(out)
        
        model = Model(inputs=inputs, outputs=outputs, name='Stage1_CNN_Dense')
        return model
    
    def build_stage2_model(self, pretrained_stage1_path):
        """阶段2: 冻结CNN + LSTM (精细调优)"""
        # 加载阶段1模型
        stage1_model = keras.models.load_model(pretrained_stage1_path)
        
        # 创建新输入
        inputs = layers.Input(
            shape=(self.config.IMG_HEIGHT, self.config.IMG_WIDTH, self.config.IMG_CHANNELS),
            name='input_image'
        )
        
        # 提取CNN骨干（到pool4层）
        x = inputs
        for layer in stage1_model.layers:
            if layer.name == 'pool4':
                x = layer(x)
                break
            elif 'input' not in layer.name:
                x = layer(x)
        
        # 冻结CNN层
        for layer in stage1_model.layers:
            if layer.name in ['flatten', 'dense_512'] or 'dropout' in layer.name or 'char_' in layer.name:
                break
            layer.trainable = False
        
        # 添加LSTM层
        shape = x.shape
        x = layers.Reshape(target_shape=(shape[2], shape[1] * shape[3]), name='reshape')(x)
        x = layers.Dense(128, activation='relu', name='dense_before_rnn')(x)
        x = layers.Dropout(0.3, name='dropout_rnn1')(x)
        
        # 双向LSTM
        x = layers.Bidirectional(
            layers.LSTM(128, return_sequences=True, dropout=0.2),
            name='bilstm1'
        )(x)
        x = layers.Bidirectional(
            layers.LSTM(128, return_sequences=True, dropout=0.2),
            name='bilstm2'
        )(x)
        
        # 全局特征聚合
        x = layers.GlobalAveragePooling1D(name='global_pool')(x)
        x = layers.Dense(512, activation='relu', name='dense_final')(x)
        x = layers.Dropout(0.4, name='dropout_final')(x)
        
        # 多输出层
        outputs = []
        for i in range(self.config.CHARS_PER_LABEL):
            out = layers.Dense(
                self.config.N_CLASSES,
                activation='softmax',
                name=f'char_{i}'
            )(x)
            outputs.append(out)
        
        model = Model(inputs=inputs, outputs=outputs, name='Stage2_CNN_LSTM')
        return model
    
    def compile_model(self, model, learning_rate):
        """编译模型"""
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        
        losses = {f'char_{i}': 'sparse_categorical_crossentropy' 
                  for i in range(self.config.CHARS_PER_LABEL)}
        
        metrics = {f'char_{i}': ['accuracy'] 
                   for i in range(self.config.CHARS_PER_LABEL)}
        
        model.compile(
            optimizer=optimizer,
            loss=losses,
            metrics=metrics
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
                image = np.array(example["image"], dtype=np.float32) / 255.0
                label = example["label"]
                writer.write(self._serialize_example(image, label))
        
        # 处理验证集
        with tf.io.TFRecordWriter(self.config.TEST_TFRECORD_PATH) as writer:
            val_ds = dataset_dict["test"]
            self.val_size = len(val_ds)
            for example in tqdm(val_ds, desc="正在处理验证集"):
                image = np.array(example["image"], dtype=np.float32) / 255.0
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
        
        outputs = {f'char_{i}': label[i] for i in range(self.config.CHARS_PER_LABEL)}
        
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

        raw_train_ds = tf.data.TFRecordDataset(self.config.TRAIN_TFRECORD_PATH)
        raw_val_ds = tf.data.TFRecordDataset(self.config.TEST_TFRECORD_PATH)
        
        train_dataset = (
            raw_train_ds
            .map(self._parse_tfrecord_fn, num_parallel_calls=tf.data.AUTOTUNE)
            .shuffle(self.train_size)
            .map(self._augment_data, num_parallel_calls=tf.data.AUTOTUNE)
            .batch(self.config.BATCH_SIZE, drop_remainder=True)
            .prefetch(tf.data.AUTOTUNE)
        )
        
        val_dataset = (
            raw_val_ds
            .map(self._parse_tfrecord_fn, num_parallel_calls=tf.data.AUTOTUNE)
            .batch(self.config.BATCH_SIZE, drop_remainder=True)
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
        
        for images, labels_dict in self.validation_data:
            predictions = self.model.predict(images, verbose=0)
            
            batch_size = images.shape[0]
            for i in range(batch_size):
                true_label = [labels_dict[f'char_{j}'].numpy()[i] 
                             for j in range(self.config.CHARS_PER_LABEL)]
                
                pred_label = [np.argmax(predictions[j][i]) 
                             for j in range(self.config.CHARS_PER_LABEL)]
                
                if true_label == pred_label:
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
    
    # 1. 总Loss曲线
    axes[0, 0].plot(history.history['loss'], label='Train Loss', linewidth=2)
    axes[0, 0].plot(history.history['val_loss'], label='Val Loss', linewidth=2)
    axes[0, 0].set_title(f'{stage_name} - Total Loss', fontsize=14, fontweight='bold')
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
    axes[0, 1].set_title(f'{stage_name} - Average Accuracy', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim([0, 1.05])
    
    # 3. 各位置准确率对比
    for i in range(4):
        axes[1, 0].plot(history.history[f'char_{i}_accuracy'], 
                       label=f'Char {i}', linewidth=1.5)
    axes[1, 0].set_title(f'{stage_name} - Per-Position Accuracy (Train)', fontsize=14, fontweight='bold')
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
    axes[1, 1].set_title(f'{stage_name} - Overfitting Diagnosis', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss Gap')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ {stage_name} 训练曲线已保存: {save_path}")
    plt.close()


def train_stage1(train_dataset, val_dataset, data_loader, config):
    """阶段1训练: CNN + Dense"""
    print("\n" + "="*80)
    print("阶段1: 训练 CNN + Dense 基础模型")
    print("="*80)
    
    # 构建模型
    model_builder = ModelBuilder(config)
    model = model_builder.build_stage1_model()
    model = model_builder.compile_model(model, config.LEARNING_RATE_STAGE1)
    
    print("\n阶段1 模型架构:")
    model.summary()
    
    # 计算步数
    train_steps = data_loader.get_steps(data_loader.train_size)
    val_steps = data_loader.get_steps(data_loader.val_size)
    
    # 回调函数
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            config.STAGE1_MODEL_PATH,
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
        steps_per_epoch=train_steps,
        validation_steps=val_steps,
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
    
    # 计算步数
    train_steps = data_loader.get_steps(data_loader.train_size)
    val_steps = data_loader.get_steps(data_loader.val_size)
    
    # 回调函数
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            config.STAGE2_MODEL_PATH,
            save_best_only=True,
            monitor='val_loss',
            mode='min',
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
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
        steps_per_epoch=train_steps,
        validation_steps=val_steps,
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


def evaluate_and_compare(train_dataset, val_dataset, config):
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
        
        for images, labels_dict in dataset:
            predictions = model.predict(images, verbose=0)
            
            batch_size = images.shape[0]
            for i in range(batch_size):
                true_label = [labels_dict[f'char_{j}'].numpy()[i] 
                             for j in range(config.CHARS_PER_LABEL)]
                
                pred_label = [np.argmax(predictions[j][i]) 
                             for j in range(config.CHARS_PER_LABEL)]
                
                if true_label == pred_label:
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
    
    print(f"\n每epoch训练步数: {data_loader.get_steps(data_loader.train_size)}")
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
        train_dataset, 
        val_dataset, 
        config
    )
    
    print("\n" + "="*80)
    print("训练完成！")
    print("="*80)
    print(f"阶段1模型: {config.STAGE1_MODEL_PATH}")
    print(f"阶段2模型: {config.STAGE2_MODEL_PATH}")
    print(f"最终模型: {config.FINAL_MODEL_PATH}")
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
    print("\n✅✅ 所有训练流程已完成！")
    
    # 打印最终统计
    print("\n" + "="*80)
    print("训练总结")
    print("="*80)
    print(f"阶段1 (CNN+Dense) 最终准确率: {results['stage1_accuracy']:.4f}")
    print(f"阶段2 (CNN+LSTM) 最终准确率: {results['stage2_accuracy']:.4f}")
    print(f"性能提升: {(results['stage2_accuracy'] - results['stage1_accuracy'])*100:+.2f}%")
    print("="*80)