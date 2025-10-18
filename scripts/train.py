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
    BATCH_SIZE = 128
    EPOCHS = 50
    LEARNING_RATE = 0.001
    
    # 路径配置
    MODEL_DIR = "models"
    CHECKPOINT_PATH = os.path.join(MODEL_DIR, "best_model.keras")
    FINAL_MODEL_PATH = os.path.join(MODEL_DIR, "luoguCaptcha_crnn.keras")

config = Config()

# 创建CRNN模型
class CRNNModel:
    def __init__(self, config):
        self.config = config
        self.model = self._build_model()
    
    def _build_model(self):
        """构建CRNN模型架构"""
        inputs = layers.Input(
            shape=(self.config.IMG_HEIGHT, self.config.IMG_WIDTH, self.config.IMG_CHANNELS),
            name='input_image'
        )
        
        # CNN特征提取
        x = layers.Conv2D(32, (3, 3), padding='same', activation='relu', name='conv1')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(pool_size=(2, 2), name='pool1')(x)
        
        x = layers.Conv2D(64, (3, 3), padding='same', activation='relu', name='conv2')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(pool_size=(2, 2), name='pool2')(x)
        
        x = layers.Conv2D(128, (3, 3), padding='same', activation='relu', name='conv3')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(pool_size=(2, 2), name='pool3')(x)
        
        x = layers.Conv2D(256, (3, 3), padding='same', activation='relu', name='conv4')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(pool_size=(2, 2), name='pool4')(x)
        
        # 特征图重塑为序列
        shape = x.shape
        x = layers.Reshape(target_shape=(shape[2], shape[1] * shape[3]), name='reshape')(x)
        x = layers.Dense(128, activation='relu', name='dense_before_rnn')(x)
        x = layers.Dropout(0.3)(x)
        
        # 双向LSTM层
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
        x = layers.Dropout(0.4)(x)
        
        # 多输出层 - 每个字符位置独立预测
        outputs = []
        for i in range(self.config.CHARS_PER_LABEL):
            out = layers.Dense(
                self.config.N_CLASSES,
                activation='softmax',
                name=f'char_{i}'
            )(x)
            outputs.append(out)
        
        model = Model(inputs=inputs, outputs=outputs, name='CRNN_Captcha')
        return model
    
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

# train.py - 修改 DataLoader 类

class DataLoader:
    def __init__(self, config):
        self.config = config
        self.train_size = 0
        self.val_size = 0
    
    def load_data(self):
        """从HuggingFace加载数据集"""
        print(f"加载数据集: {self.config.DATASET_PATH}")
        dataset_dict = load_dataset(self.config.DATASET_PATH)
        
        train_ds = dataset_dict["train"]
        val_ds = dataset_dict["test"]
        
        self.train_size = len(train_ds)
        self.val_size = len(val_ds)
        
        print(f"训练集大小: {self.train_size}")
        print(f"验证集大小: {self.val_size}")
        
        return train_ds, val_ds
    
    def prepare_dataset(self, dataset, is_training=True):
        """使用 to_tf_dataset 快速转换 - 零CPU预处理"""
        
        # 关键：直接转换为 TensorFlow 数据集
        tf_dataset = dataset.to_tf_dataset(
            columns="image",
            label_cols="label",
            batch_size=self.config.BATCH_SIZE,
            shuffle=is_training,  # 内置shuffle
            collate_fn=None,  # 不需要自定义collate
        )
        
        # 标签转换为多输出格式
        def prepare_multi_output(image, label):
            """将标签转换为多输出字典"""
            outputs = {
                f'char_{i}': label[:, i] 
                for i in range(self.config.CHARS_PER_LABEL)
            }
            return image, outputs
        
        # 应用标签转换
        tf_dataset = tf_dataset.map(
            prepare_multi_output,
            num_parallel_calls=tf.data.AUTOTUNE
        )
        
        # 预取
        tf_dataset = tf_dataset.prefetch(tf.data.AUTOTUNE)
        
        print(f"  ✅ 数据集准备完成!")
        
        return tf_dataset
    
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
    
    # 加载数据
    data_loader = DataLoader(config)
    train_ds_hf, val_ds_hf = data_loader.load_data()
    
    # 准备TensorFlow数据集
    train_dataset = data_loader.prepare_dataset(train_ds_hf, is_training=True)
    val_dataset = data_loader.prepare_dataset(val_ds_hf, is_training=False)
    
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