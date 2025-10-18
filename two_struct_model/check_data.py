# check_data.py
import tensorflow as tf
import numpy as np
import json

# 加载一个batch检查
def check_data():
    TRAIN_TFRECORD_PATH = "tfrecords/train.tfrecord"
    BATCH_SIZE = 256
    IMG_HEIGHT = 35
    IMG_WIDTH = 90
    CHARS_PER_LABEL = 4
    
    def _parse_tfrecord_fn(example_proto):
        feature_description = {
            "image": tf.io.FixedLenFeature([IMG_HEIGHT * IMG_WIDTH], tf.float32),
            "label": tf.io.FixedLenFeature([CHARS_PER_LABEL], tf.int64),
        }
        example = tf.io.parse_single_example(example_proto, feature_description)
        image = tf.reshape(example["image"], (IMG_HEIGHT, IMG_WIDTH, 1))
        label = tf.cast(example["label"], tf.int32)
        return image, label
    
    dataset = tf.data.TFRecordDataset(TRAIN_TFRECORD_PATH)
    dataset = dataset.map(_parse_tfrecord_fn).batch(BATCH_SIZE)
    
    for images, labels in dataset.take(1):
        print("="*60)
        print("数据检查")
        print("="*60)
        print(f"Images shape: {images.shape}")
        print(f"Images dtype: {images.dtype}")
        print(f"Images min: {tf.reduce_min(images).numpy():.4f}")
        print(f"Images max: {tf.reduce_max(images).numpy():.4f}")
        print(f"Images mean: {tf.reduce_mean(images).numpy():.4f}")
        print()
        print(f"Labels shape: {labels.shape}")
        print(f"Labels dtype: {labels.dtype}")
        print(f"Labels min: {tf.reduce_min(labels).numpy()}")
        print(f"Labels max: {tf.reduce_max(labels).numpy()}")
        print()
        print("前5个样本的标签:")
        for i in range(min(5, labels.shape[0])):
            label = labels[i].numpy()
            chars = ''.join([chr(c) for c in label])
            print(f"  样本{i}: {label} -> '{chars}'")
        print("="*60)
        
        # 检查标签分布
        unique, counts = np.unique(labels.numpy().flatten(), return_counts=True)
        print(f"\n标签值范围: {unique.min()} - {unique.max()}")
        print(f"不同字符数: {len(unique)}")
        
        break

if __name__ == "__main__":
    check_data()