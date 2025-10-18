import os
import time
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
import numpy as np
from datasets import load_dataset

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
    
    # 训练配置
    BATCH_SIZE = 128
    EPOCHS = 50
    LEARNING_RATE = 0.001
    
    # 路径配置
    MODEL_DIR = "models"
    CHECKPOINT_PATH = os.path.join(MODEL_DIR, "best_model.keras")  # 改为.keras
    FINAL_MODEL_PATH = os.path.join(MODEL_DIR, "luoguCaptcha_crnn.keras")  # 改为.keras

config = Config()

# CRNN模型类（保持不变）
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
        
        # 多输出层
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

# 优化的DataLoader类
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
    
    def prepare_dataset_optimized(self, dataset, is_training=True):
        """优化的数据准备 - 使用批量索引"""
        print(f"正在转换数据集... (共 {len(dataset)} 个样本)")
        start_time = time.time()
        
        # 使用批量索引访问，比迭代器快很多
        chunk_size = 2000  # 每次处理2000个样本
        n_samples = len(dataset)
        
        images_chunks = []
        labels_chunks = []
        
        for i in range(0, n_samples, chunk_size):
            end_idx = min(i + chunk_size, n_samples)
            print(f"  处理 {i:6d}-{end_idx:6d}/{n_samples} ({end_idx/n_samples*100:5.1f}%)")
            
            # 关键：使用切片批量获取数据（比迭代快10-20倍）
            chunk = dataset[i:end_idx]
            
            # 批量转换
            chunk_images = np.array(chunk["image"], dtype=np.float32) / 255.0
            chunk_labels = np.array(chunk["label"], dtype=np.int32)
            
            # 添加通道维度（如果需要）
            if len(chunk_images.shape) == 3:  # (N, H, W)
                chunk_images = np.expand_dims(chunk_images, axis=-1)
            
            images_chunks.append(chunk_images)
            labels_chunks.append(chunk_labels)
        
        # 合并所有chunks
        print("  合并数据...")
        images = np.concatenate(images_chunks, axis=0)
        labels = np.concatenate(labels_chunks, axis=0)
        
        elapsed = time.time() - start_time
        print(f"  ✅ 转换完成! 耗时: {elapsed:.1f}秒")
        print(f"  数据形状 - 图像: {images.shape}, 标签: {labels.shape}")
        
        # 创建tf.data.Dataset
        tf_dataset = tf.data.Dataset.from_tensor_slices((images, labels))
        
        if is_training:
            tf_dataset = tf_dataset.shuffle(10000, reshuffle_each_iteration=True)
        
        tf_dataset = tf_dataset.batch(self.config.BATCH_SIZE)
        tf_dataset = tf_dataset.map(
            self._prepare_labels, 
            num_parallel_calls=tf.data.AUTOTUNE
        )
        tf_dataset = tf_dataset.prefetch(tf.data.AUTOTUNE)
        
        return tf_dataset
    
    def prepare_dataset_with_cache(self, dataset, is_training=True):
        """带缓存的数据准备 - 第二次训练超快"""
        cache_dir = "data_cache"
        os.makedirs(cache_dir, exist_ok=True)

        cache_file = os.path.join(
            cache_dir, 
            f"{'train' if is_training else 'val'}_cache.npz"
        )

        if os.path.exists(cache_file):
            print(f"✅ 从缓存加载: {cache_file}")
            start = time.time()

            data = np.load(cache_file)
            images = data['images']
            labels = data['labels']

            print(f"  加载耗时: {time.time()-start:.1f}秒")
        else:
            print(f"创建缓存: {cache_file}")

            # 使用优化方法加载
            chunk_size = 2000
            n_samples = len(dataset)

            images_chunks = []
            labels_chunks = []

            for i in range(0, n_samples, chunk_size):
                end = min(i + chunk_size, n_samples)
                print(f"  {i:6d}-{end:6d}/{n_samples}")

                chunk = dataset[i:end]
                chunk_images = np.array(chunk["image"], dtype=np.float32) / 255.0
                chunk_labels = np.array(chunk["label"], dtype=np.int32)

                if len(chunk_images.shape) == 3:
                    chunk_images = np.expand_dims(chunk_images, axis=-1)

                images_chunks.append(chunk_images)
                labels_chunks.append(chunk_labels)

            images = np.concatenate(images_chunks, axis=0)
            labels = np.concatenate(labels_chunks, axis=0)

            # 保存缓存
            print(f"  保存缓存...")
            np.savez_compressed(cache_file, images=images, labels=labels)
            print(f"  ✅ 缓存已保存!")

        # 创建TensorFlow数据集
        tf_dataset = tf.data.Dataset.from_tensor_slices((images, labels))

        if is_training:
            tf_dataset = tf_dataset.shuffle(10000, reshuffle_each_iteration=True)

        tf_dataset = tf_dataset.batch(self.config.BATCH_SIZE)
        tf_dataset = tf_dataset.map(
            self._prepare_labels, 
            num_parallel_calls=tf.data.AUTOTUNE
        )
        tf_dataset = tf_dataset.prefetch(tf.data.AUTOTUNE)

        return tf_dataset
    
    def prepare_dataset(self, dataset, is_training=True):
        """准备数据集 - 保留原方法作为备用"""
        print("正在转换数据集为NumPy数组...")
        images_list = []
        labels_list = []
        
        for sample in dataset:
            image = np.array(sample["image"], dtype=np.float32) / 255.0
            label = np.array(sample["label"], dtype=np.int32)
            
            if len(image.shape) == 2:
                image = np.expand_dims(image, axis=-1)
            
            images_list.append(image)
            labels_list.append(label)
        
        images = np.array(images_list)
        labels = np.array(labels_list)
        
        print(f"数据形状 - 图像: {images.shape}, 标签: {labels.shape}")
        
        tf_dataset = tf.data.Dataset.from_tensor_slices((images, labels))
        
        if is_training:
            tf_dataset = tf_dataset.shuffle(10000, reshuffle_each_iteration=True)
        
        tf_dataset = tf_dataset.batch(self.config.BATCH_SIZE)
        tf_dataset = tf_dataset.map(
            self._prepare_labels, 
            num_parallel_calls=tf.data.AUTOTUNE
        )
        tf_dataset = tf_dataset.prefetch(tf.data.AUTOTUNE)
        
        return tf_dataset
    
    def _prepare_labels(self, images, labels):
        """将标签转换为多输出格式"""
        outputs = {}
        for i in range(self.config.CHARS_PER_LABEL):
            outputs[f'char_{i}'] = labels[:, i]
        return images, outputs
    
    def get_steps(self, dataset_size):
        """计算每个epoch的步数"""
        return dataset_size // self.config.BATCH_SIZE

# 自定义回调
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
                true_label = [labels_dict[f'char_{j}'].numpy()[i] 
                             for j in range(self.config.CHARS_PER_LABEL)]
                
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

class ValidationLossPrinter(keras.callbacks.Callback):
    """专门打印验证损失的回调"""
    
    def __init__(self):
        super().__init__()
        self.best_val_loss = float('inf')
    
    def on_epoch_end(self, epoch, logs=None):
        val_loss = logs.get('val_loss', 0)
        train_loss = logs.get('loss', 0)
        
        improvement = ""
        if val_loss < self.best_val_loss:
            improvement = f" ✅ 改进 {self.best_val_loss - val_loss:.4f}"
            self.best_val_loss = val_loss
        else:
            improvement = f" ⚠️  比最佳高 {val_loss - self.best_val_loss:.4f}"
        
        print(f"\nEpoch {epoch + 1} 损失: "
              f"训练={train_loss:.4f}, 验证={val_loss:.4f}{improvement}")

# 主训练函数
def train():
    """主训练流程"""
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    
    print("\n" + "="*70)
    print("CRNN验证码识别模型训练".center(70))
    print("="*70 + "\n")
    
    # 加载数据
    data_loader = DataLoader(config)
    train_ds_hf, val_ds_hf = data_loader.load_data()
    
    # 使用优化的数据准备方法
    print("\n准备训练集...")
    train_dataset = data_loader.prepare_dataset_with_cache(train_ds_hf, is_training=True)
    
    print("\n准备验证集...")
    val_dataset = data_loader.prepare_dataset_with_cache(val_ds_hf, is_training=False)
    
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
        ValidationLossPrinter(),
        # TqdmCallback(verbose=2, leave=True),  # 添加这个，放第一个
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
    
    # 训练模型
    print("\n" + "="*70)
    print("开始训练...".center(70))
    print("="*70 + "\n")
    
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=config.EPOCHS,
        steps_per_epoch=train_steps,
        validation_steps=val_steps,
        callbacks=callbacks,
        verbose=1
    )
    
    # 保存最终模型
    print(f"\n保存最终模型...")
    model.save(config.FINAL_MODEL_PATH, save_format='keras')
    print(f"✅ 模型已保存到: {config.FINAL_MODEL_PATH}")
    
    # 打印训练总结
    print("\n" + "="*70)
    print("训练完成！".center(70))
    print("="*70)
    
    return model, history

if __name__ == "__main__":
    model, history = train()
    print("\n训练完成！")