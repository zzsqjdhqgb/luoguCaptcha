# Copyright (C) 2025 Langning Chen
# Modified by zzsqjdhqgb
#
# This file is part of luoguCaptcha.
#
# luoguCaptcha is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# luoguCaptcha is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with luoguCaptcha.  If not, see <https://www.gnu.org/licenses/>.

import os
import glob
import tensorflow as tf
from config import Config


def parse_tfrecord(example_proto):
    """
    解析单个TFRecord样本
    
    Args:
        example_proto: TFRecord序列化样本
        
    Returns:
        (image, label): 图像和标签的元组
    """
    feature_description = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "label": tf.io.FixedLenFeature([Config.CHARS_PER_LABEL], tf.int64),
    }
    example = tf.io.parse_single_example(example_proto, feature_description)
    image = tf.io.decode_png(example["image"], channels=1)
    image = tf.image.convert_image_dtype(image, tf.float32)  # Normalize to [0, 1]
    label = example["label"]
    return image, label


def load_datasets(tfrecord_dir=None):
    """
    加载训练和验证数据集
    
    Args:
        tfrecord_dir: TFRecord文件目录，默认使用Config中的配置
        
    Returns:
        (train_dataset, val_dataset): 训练集和验证集
    """
    if tfrecord_dir is None:
        tfrecord_dir = Config.TFRECORD_DIR
    
    train_files = sorted(glob.glob(os.path.join(tfrecord_dir, "train_part_*.tfrecord")))
    test_files = sorted(glob.glob(os.path.join(tfrecord_dir, "test_part_*.tfrecord")))

    if not train_files or not test_files:
        raise ValueError(f"No TFRecord files found in {tfrecord_dir}")

    print(f"✓ Found {len(train_files)} train files and {len(test_files)} test files")

    # 创建数据集
    train_ds = tf.data.TFRecordDataset(train_files, num_parallel_reads=tf.data.AUTOTUNE)
    test_ds = tf.data.TFRecordDataset(test_files, num_parallel_reads=tf.data.AUTOTUNE)

    # 解析和预处理
    train_ds = train_ds.map(parse_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)
    test_ds = test_ds.map(parse_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)

    # 批处理和预取
    train_ds = (
        train_ds.shuffle(buffer_size=10000)
        .batch(Config.BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )
    test_ds = test_ds.batch(Config.BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    return train_ds, test_ds