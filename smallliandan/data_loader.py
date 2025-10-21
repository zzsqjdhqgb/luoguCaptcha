import os
import glob
import tensorflow as tf


def uppercase_label(label):
    """
    将标签转换为大写。
    小写字母 a-z (ASCII 97-122) 转换为大写 A-Z (ASCII 65-90)
    """
    def convert_char(char_code):
        return tf.where(
            tf.logical_and(char_code >= 97, char_code <= 122),
            char_code - 32,
            char_code
        )
    
    return tf.map_fn(convert_char, label, dtype=tf.int64)


def parse_tfrecord(example_proto, chars_per_label=4):
    """Parses a single TFRecord example into image and label."""
    feature_description = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "label": tf.io.FixedLenFeature([chars_per_label], tf.int64),
    }
    example = tf.io.parse_single_example(example_proto, feature_description)
    image = tf.io.decode_png(example["image"], channels=1)
    image = tf.image.convert_image_dtype(image, tf.float32)  # Normalize to [0, 1]
    label = example["label"]  # Shape: (4,)
    
    # 将标签转换为大写
    label = uppercase_label(label)
    
    return image, label


def load_and_preprocess_data(tfrecord_dir, batch_size=256, chars_per_label=4):
    """Loads and preprocesses data from TFRecord files."""
    # Get train and test TFRecord files
    train_files = sorted(glob.glob(os.path.join(tfrecord_dir, "train_part_*.tfrecord")))
    test_files = sorted(glob.glob(os.path.join(tfrecord_dir, "test_part_*.tfrecord")))

    if not train_files or not test_files:
        raise ValueError(f"No TFRecord files found in {tfrecord_dir}")

    print(f"Found {len(train_files)} train files and {len(test_files)} test files")

    # Create tf.data datasets
    train_ds = tf.data.TFRecordDataset(train_files, num_parallel_reads=tf.data.AUTOTUNE)
    test_ds = tf.data.TFRecordDataset(test_files, num_parallel_reads=tf.data.AUTOTUNE)

    # Parse TFRecords
    train_ds = train_ds.map(
        lambda x: parse_tfrecord(x, chars_per_label),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    test_ds = test_ds.map(
        lambda x: parse_tfrecord(x, chars_per_label),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    # Shuffle and batch train dataset
    train_ds = (
        train_ds.shuffle(buffer_size=10000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    )
    test_ds = test_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return train_ds, test_ds