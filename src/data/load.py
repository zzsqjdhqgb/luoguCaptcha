# Copyright (C) 2026 zzsqjdhqgb
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
数据加载模块。

从 NumPy .npz 文件加载验证码数据，自动检查并执行归一化，
将标签中的小写字母统一转为大写（验证码不区分大小写），
返回可直接用于 Keras model.fit() 的 NumPy 数组。

Usage:
    from data.load import load_captcha_data

    (x_train, y_train), (x_test, y_test) = load_captcha_data("data/luogu_captcha_numpy")
"""

import os
import numpy as np

from config import NUMPY_DIR, IMG_HEIGHT, IMG_WIDTH, CHARS_PER_LABEL


def _check_and_normalize(images: np.ndarray, name: str) -> np.ndarray:
    """
    检查图像数组并在必要时进行归一化。

    规则：
      - 如果 dtype 为 uint8 或最大值 > 1.0，执行 / 255.0 归一化
      - 如果已经在 [0, 1] 范围内（float 且 max <= 1.0），跳过归一化
      - 打印归一化状态以便排查问题

    Args:
        images: 图像数组，形状 (N, H, W, 1)
        name: 数据集名称（用于日志输出）

    Returns:
        float32 数组，值域 [0, 1]
    """
    original_dtype = images.dtype
    vmin, vmax = float(images.min()), float(images.max())

    # 判断是否需要归一化
    needs_normalize = False

    if np.issubdtype(original_dtype, np.integer):
        # 整型（uint8 等），一定需要归一化
        needs_normalize = True
    elif vmax > 1.0:
        # 浮点但值域超过 [0,1]，说明未归一化或被重复加载为 float 但未除 255
        needs_normalize = True

    if needs_normalize:
        images = images.astype(np.float32) / 255.0
        new_vmin, new_vmax = float(images.min()), float(images.max())
        print(
            f"  [{name}] Normalized: dtype {original_dtype} [{vmin:.1f}, {vmax:.1f}] "
            f"→ float32 [{new_vmin:.4f}, {new_vmax:.4f}]"
        )
    else:
        images = images.astype(np.float32)
        print(
            f"  [{name}] Already normalized: dtype {original_dtype} "
            f"[{vmin:.4f}, {vmax:.4f}], skipping normalization"
        )

    return images


def _normalize_labels_to_upper(labels: np.ndarray, name: str) -> np.ndarray:
    """
    将标签中的小写字母 ASCII 码统一转为大写。

    验证码不区分大小写，因此将 'a'-'z' (97-122) 映射到 'A'-'Z' (65-90)，
    减少输出类别冗余。

    Args:
        labels: 标签数组，shape (N, CHARS_PER_LABEL)，值为 ASCII 码
        name: 数据集名称（用于日志输出）

    Returns:
        转换后的标签数组（int64）
    """
    labels = labels.copy()
    # 小写字母 ASCII 范围：97 ('a') ~ 122 ('z')
    lowercase_mask = (labels >= ord('a')) & (labels <= ord('z'))
    num_converted = int(lowercase_mask.sum())

    if num_converted > 0:
        # 小写转大写：减去 32 (ord('a') - ord('A') == 32)
        labels[lowercase_mask] -= 32
        total_chars = labels.size
        print(
            f"  [{name}] Labels uppercased: {num_converted}/{total_chars} chars converted "
            f"({num_converted / total_chars * 100:.1f}%)"
        )
    else:
        print(f"  [{name}] Labels already uppercase, no conversion needed")

    return labels


def _validate_shapes(
    images: np.ndarray, labels: np.ndarray, name: str
) -> tuple[np.ndarray, np.ndarray]:
    """
    验证图像和标签的形状，必要时修正。

    Args:
        images: 图像数组
        labels: 标签数组
        name: 数据集名称

    Returns:
        (images, labels) 形状验证后的数组
    """
    n = images.shape[0]

    # 图像：确保为 (N, H, W, 1)
    if images.ndim == 3:
        # (N, H, W) → (N, H, W, 1)
        images = images[..., np.newaxis]
        print(f"  [{name}] Added channel dimension: {images.shape}")

    if images.shape[1:] != (IMG_HEIGHT, IMG_WIDTH, 1):
        raise ValueError(
            f"[{name}] Unexpected image shape: {images.shape}, "
            f"expected (N, {IMG_HEIGHT}, {IMG_WIDTH}, 1)"
        )

    # 标签：确保为 (N, CHARS_PER_LABEL)，int64
    if labels.shape != (n, CHARS_PER_LABEL):
        raise ValueError(
            f"[{name}] Unexpected label shape: {labels.shape}, "
            f"expected ({n}, {CHARS_PER_LABEL})"
        )

    labels = labels.astype(np.int64)

    return images, labels


def load_captcha_data(
    data_dir: str | None = None,
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """
    从 .npz 文件加载训练集和测试集。

    自动执行：
      1. 形状验证（确保 images 为 (N, H, W, 1)，labels 为 (N, 4)）
      2. 归一化检查（防止多次归一化）
      3. 标签大小写统一（小写转大写，消除冗余类别）

    Args:
        data_dir: 包含 train.npz 和 test.npz 的目录路径。
                  默认使用 config.NUMPY_DIR。

    Returns:
        ((x_train, y_train), (x_test, y_test))
        - x_train, x_test: float32, shape (N, 35, 90, 1), 值域 [0, 1]
        - y_train, y_test: int64, shape (N, 4), 标签为大写字母/数字的 ASCII 码
    """
    if data_dir is None:
        data_dir = NUMPY_DIR

    train_path = os.path.join(data_dir, "train.npz")
    test_path = os.path.join(data_dir, "test.npz")

    # 检查文件存在
    for fpath, label in [(train_path, "train"), (test_path, "test")]:
        if not os.path.exists(fpath):
            raise FileNotFoundError(
                f"{label} data not found: {fpath}\n"
                f"Run `python -m data.pull_data_numpy` or "
                f"`python -m data.tfrecord2numpy` first."
            )

    print(f"Loading data from: {data_dir}")

    # 加载 train
    print("  Loading train.npz ...")
    train_data = np.load(train_path)
    x_train, y_train = train_data["images"], train_data["labels"]
    x_train, y_train = _validate_shapes(x_train, y_train, "train")
    x_train = _check_and_normalize(x_train, "train")
    y_train = _normalize_labels_to_upper(y_train, "train")
    print(f"  [train] {x_train.shape[0]} samples loaded")

    # 加载 test
    print("  Loading test.npz ...")
    test_data = np.load(test_path)
    x_test, y_test = test_data["images"], test_data["labels"]
    x_test, y_test = _validate_shapes(x_test, y_test, "test")
    x_test = _check_and_normalize(x_test, "test")
    y_test = _normalize_labels_to_upper(y_test, "test")
    print(f"  [test]  {x_test.shape[0]} samples loaded")

    print(
        f"Data loaded: {x_train.shape[0]} train + {x_test.shape[0]} test = "
        f"{x_train.shape[0] + x_test.shape[0]} total"
    )

    return (x_train, y_train), (x_test, y_test)