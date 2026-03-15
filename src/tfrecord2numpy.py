# Copyright (C) 2025 zzsqjdhqgb
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

"""
将已有的 TFRecord 文件转换为 NumPy .npz 格式。

Usage:
  python tfrecord2numpy.py [--tfrecord-dir DIR] [--output-dir DIR]

Examples:
  python tfrecord2numpy.py
  python tfrecord2numpy.py --tfrecord-dir data/luogu_captcha_tfrecord --output-dir data/luogu_captcha_numpy
"""

import argparse
import glob
import os

import numpy as np
import tensorflow as tf
from tqdm import tqdm

# ── 常量 ──────────────────────────────────────────────────────────
TFRECORD_DIR = "data/luogu_captcha_tfrecord"
OUTPUT_DIR = "data/luogu_captcha_numpy"
CHARS_PER_LABEL = 4
IMG_HEIGHT, IMG_WIDTH = 35, 90


def parse_tfrecord(example_proto):
    """解析单条 TFRecord 样本。"""
    feature_description = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "label": tf.io.FixedLenFeature([CHARS_PER_LABEL], tf.int64),
    }
    example = tf.io.parse_single_example(example_proto, feature_description)
    image = tf.io.decode_png(example["image"], channels=1)  # (H, W, 1), uint8
    label = example["label"]  # (4,), int64
    return image, label


def convert_split(tfrecord_files: list, split_name: str, output_dir: str):
    """
    读取一组 TFRecord 文件，转换为 NumPy 数组并保存为 .npz。
    - images: uint8, shape (N, IMG_HEIGHT, IMG_WIDTH, 1)
    - labels: int32, shape (N, CHARS_PER_LABEL)
    """
    if not tfrecord_files:
        print(f"  No TFRecord files found for split '{split_name}', skipping.")
        return

    print(f"\nConverting '{split_name}': {len(tfrecord_files)} TFRecord file(s)")

    # 先数一下总样本数（用于预分配）
    dataset = tf.data.TFRecordDataset(tfrecord_files)
    total = 0
    for _ in dataset:
        total += 1
    print(f"  Total samples: {total}")

    # 预分配数组
    images = np.empty((total, IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.uint8)
    labels = np.empty((total, CHARS_PER_LABEL), dtype=np.int32)

    # 重新读取并解析
    dataset = tf.data.TFRecordDataset(tfrecord_files)
    dataset = dataset.map(parse_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)

    idx = 0
    for image, label in tqdm(dataset, total=total, desc=f"  Reading"):
        images[idx] = image.numpy()  # uint8, (H, W, 1)
        labels[idx] = label.numpy().astype(np.int32)
        idx += 1

    # 保存
    output_path = os.path.join(output_dir, f"{split_name}.npz")
    np.savez_compressed(output_path, images=images, labels=labels)
    print(f"  Saved {idx} samples to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert TFRecord files to NumPy .npz format."
    )
    parser.add_argument(
        "--tfrecord-dir",
        type=str,
        default=TFRECORD_DIR,
        help=f"Directory containing TFRecord files (default: {TFRECORD_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=OUTPUT_DIR,
        help=f"Output directory for .npz files (default: {OUTPUT_DIR})",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 查找 train 和 test 文件
    train_files = sorted(
        glob.glob(os.path.join(args.tfrecord_dir, "train_part_*.tfrecord"))
    )
    test_files = sorted(
        glob.glob(os.path.join(args.tfrecord_dir, "test_part_*.tfrecord"))
    )

    if not train_files and not test_files:
        print(f"Error: No TFRecord files found in {args.tfrecord_dir}")
        print("Expected files like: train_part_0000.tfrecord, test_part_0000.tfrecord")
        return

    print(f"Found {len(train_files)} train files, {len(test_files)} test files")

    convert_split(train_files, "train", args.output_dir)
    convert_split(test_files, "test", args.output_dir)

    print(f"\n✅ Done! NumPy files saved to: {args.output_dir}")
    print("You can now run train.py to train the model.")


if __name__ == "__main__":
    main()