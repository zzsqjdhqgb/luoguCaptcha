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
从 Hugging Face Hub 拉取数据集，转换为 NumPy .npz 格式。

Usage:
  python pull_data_numpy.py [--repo-id REPO_ID] [--output-dir DIR]

Examples:
  python pull_data_numpy.py
  python pull_data_numpy.py --repo-id langningchen/luogu-captcha-dataset --output-dir data/luogu_captcha_numpy
"""

import argparse
import os

import numpy as np
from datasets import load_dataset
from tqdm import tqdm

# ── 常量 ──────────────────────────────────────────────────────────
DATASET_REPO_ID = "langningchen/luogu-captcha-dataset"
OUTPUT_DIR = "data/luogu_captcha_numpy"
CHAR_SIZE = 256
CHARS_PER_LABEL = 4
IMG_HEIGHT, IMG_WIDTH = 35, 90


def convert_split(dataset, split_name: str, output_dir: str):
    """
    将一个 split 的数据转换为两个 NumPy 数组并保存为 .npz。
    - images: uint8, shape (N, IMG_HEIGHT, IMG_WIDTH, 1)
    - labels: int32, shape (N, CHARS_PER_LABEL)
    """
    num_samples = len(dataset)
    images = np.empty((num_samples, IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.uint8)
    labels = np.empty((num_samples, CHARS_PER_LABEL), dtype=np.int32)

    skipped = 0
    valid_idx = 0

    for i in tqdm(range(num_samples), desc=f"Converting '{split_name}'"):
        try:
            sample = dataset[i]
            image = np.asarray(sample["image"], dtype=np.float32)
            label = sample["label"]

            # 归一化检查：如果是 [0,255] 范围则转换
            if image.max() > 1.0:
                image = image / 255.0

            # 确保形状为 (H, W)
            if image.ndim == 3 and image.shape[-1] == 1:
                image = image.squeeze(-1)

            if image.shape != (IMG_HEIGHT, IMG_WIDTH):
                raise ValueError(
                    f"Unexpected image shape: {image.shape}, "
                    f"expected ({IMG_HEIGHT}, {IMG_WIDTH})"
                )

            # 转为 uint8 存储（节省空间，训练时再转 float）
            image_uint8 = (image * 255).astype(np.uint8)

            # 标签
            label_list = (
                label.tolist() if isinstance(label, np.ndarray) else list(label)
            )
            if len(label_list) != CHARS_PER_LABEL:
                raise ValueError(
                    f"Unexpected label length: {len(label_list)}, "
                    f"expected {CHARS_PER_LABEL}"
                )

            images[valid_idx, :, :, 0] = image_uint8
            labels[valid_idx] = label_list
            valid_idx += 1

        except Exception as e:
            print(f"  ⚠ Skipped sample {i} in '{split_name}': {e}")
            skipped += 1
            continue

    # 裁剪到实际有效数量
    images = images[:valid_idx]
    labels = labels[:valid_idx]

    # 保存
    output_path = os.path.join(output_dir, f"{split_name}.npz")
    np.savez_compressed(output_path, images=images, labels=labels)
    print(
        f"  Saved {valid_idx} samples to {output_path} "
        f"(skipped {skipped})"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Download dataset from Hugging Face Hub and convert to NumPy .npz."
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default=DATASET_REPO_ID,
        help=f"Hugging Face dataset repo ID (default: {DATASET_REPO_ID})",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=OUTPUT_DIR,
        help=f"Output directory for .npz files (default: {OUTPUT_DIR})",
    )
    args = parser.parse_args()

    # 1. 拉取数据集
    print(f"Downloading dataset from Hugging Face Hub: {args.repo_id} ...")
    try:
        dataset_dict = load_dataset(args.repo_id)
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("Please check the repo ID and your network/authentication.")
        return

    for split_name, ds in dataset_dict.items():
        print(f"  Split '{split_name}': {len(ds)} samples")

    # 2. 转换为 .npz
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"\nConverting to NumPy .npz (output: {args.output_dir}) ...")

    for split_name in dataset_dict.keys():
        convert_split(dataset_dict[split_name], split_name, args.output_dir)

    print(f"\n✅ Done! NumPy files saved to: {args.output_dir}")
    print("You can now run train.py to train the model.")


if __name__ == "__main__":
    main()