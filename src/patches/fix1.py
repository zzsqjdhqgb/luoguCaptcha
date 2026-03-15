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
修复模型脚本：将使用 ops.slice 的旧模型重新保存为跨后端兼容版本。

问题描述:
  ExtractCLSTokens 层原先使用 ops.slice(x, [0,0,0], [-1, N, -1])，
  其中 size=-1 在 JAX 后端表示"取到末尾"，但在 PyTorch 后端被误解为
  size-1，导致最后一维从 128 变成 127，加载时报错。

修复方法:
  1. 用能正常加载的后端（JAX）加载旧模型
  2. 提取权重
  3. 用修复后的代码重新构建模型
  4. 载入权重并保存

用法:
  # 确保已安装 JAX 后端
  python fix1.py [--input models/luoguCaptcha.keras] [--output models/luoguCaptcha.keras]
  python fix1.py --backup  # 自动备份旧模型
"""

import os
import sys
import shutil
import argparse
from datetime import datetime


def detect_working_backend():
    """检测能正常加载旧模型的后端（优先 JAX）。"""
    backends = []

    try:
        import jax  # noqa: F401
        backends.append("jax")
    except ImportError:
        pass

    try:
        import tensorflow  # noqa: F401
        backends.append("tensorflow")
    except ImportError:
        pass

    # PyTorch 是有问题的后端，放最后作为 fallback
    try:
        import torch  # noqa: F401
        backends.append("torch")
    except ImportError:
        pass

    if not backends:
        print("Error: No deep learning backend found!")
        print("Please install at least one: jax, tensorflow, or torch")
        sys.exit(1)

    # 优先选择 JAX（已知能正常加载旧模型）
    if "jax" in backends:
        return "jax"
    if "tensorflow" in backends:
        return "tensorflow"

    print("Warning: Only PyTorch backend available.")
    print("The old model may not load correctly with PyTorch.")
    print("Consider installing JAX: pip install jax jaxlib")
    return "torch"


def main():
    parser = argparse.ArgumentParser(
        description="Fix model for cross-backend compatibility"
    )
    parser.add_argument(
        "--input",
        default=None,
        help="Path to the old model file (default: from config.MODEL_PATH)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to save the fixed model (default: same as input, overwrite)",
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        help="Create a backup of the old model before overwriting",
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "jax", "torch", "tensorflow"],
        default="auto",
        help="Backend to use for loading the old model (default: auto)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        default=True,
        help="Verify the fixed model by running a dummy inference (default: True)",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip verification step",
    )
    args = parser.parse_args()

    # ── 选择后端（必须在 import keras 之前）──
    if args.backend == "auto":
        backend = detect_working_backend()
    else:
        backend = args.backend

    os.environ["KERAS_BACKEND"] = backend
    print(f"Using backend: {backend}")

    # ── 现在才能 import keras 和项目模块 ──
    import keras
    import numpy as np

    import sys
    from pathlib import Path

    # 将 src 目录添加到系统路径，以便能够导入 model 和 config
    # 假设 fix1.py 位于 src/patches/ 目录下
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent  # 指向 src 目录
    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))

    # 触发自定义层注册
    import model as custom_model_module  # noqa: F401
    from model.vit import build_vit_captcha_model
    from config import MODEL_PATH, IMG_HEIGHT, IMG_WIDTH

    input_path = args.input or MODEL_PATH
    output_path = args.output or input_path

    print(f"Keras backend: {keras.backend.backend()}")
    print(f"Input model:   {input_path}")
    print(f"Output model:  {output_path}")

    # ── 检查输入文件 ──
    if not os.path.exists(input_path):
        print(f"Error: Model file not found: {input_path}")
        sys.exit(1)

    # ── 备份 ──
    if args.backup and os.path.exists(output_path) and os.path.abspath(input_path) == os.path.abspath(output_path):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = f"{output_path}.backup_{timestamp}"
        print(f"Creating backup: {backup_path}")
        shutil.copy2(output_path, backup_path)

    # ── Step 1: 加载旧模型 ──
    print("\n[Step 1/4] Loading old model...")
    try:
        old_model = keras.models.load_model(input_path)
        print(f"  Model loaded successfully: {old_model.name}")
    except Exception as e:
        print(f"  Error loading model: {e}")
        print(f"\n  The model cannot be loaded with the '{backend}' backend.")
        if backend != "jax":
            print("  Try installing JAX and running again:")
            print("    pip install jax jaxlib")
            print(f"    python {sys.argv[0]} --backend jax")
        sys.exit(1)

    # ── Step 2: 提取模型超参数和权重 ──
    print("\n[Step 2/4] Extracting model configuration and weights...")

    # 从旧模型的层中提取超参数
    config_extracted = {}
    for layer in old_model.layers:
        layer_config = layer.get_config()
        if layer.name == "patch_embedding":
            config_extracted["patch_size"] = layer_config["patch_size"]
            config_extracted["d_model"] = layer_config["d_model"]
        elif layer.name == "transformer_encoder":
            config_extracted["num_heads"] = layer_config["num_heads"]
            config_extracted["num_layers"] = layer_config["num_layers"]
            config_extracted["dff"] = layer_config["dff"]
            config_extracted["dropout_rate"] = layer_config["dropout_rate"]

    print(f"  Extracted config: {config_extracted}")

    # 保存权重到临时文件
    weights_tmp = output_path + ".tmp.weights.h5"
    old_model.save_weights(weights_tmp)
    print(f"  Weights saved to temporary file: {weights_tmp}")

    # ── Step 3: 用修复后的代码重建模型 ──
    print("\n[Step 3/4] Building new model with fixed layers...")
    new_model = build_vit_captcha_model(**config_extracted)

    # 加载权重
    new_model.load_weights(weights_tmp)
    print("  Weights loaded into new model")

    # 清理临时文件
    os.remove(weights_tmp)
    print(f"  Temporary file removed: {weights_tmp}")

    # ── Step 4: 保存修复后的模型 ──
    print(f"\n[Step 4/4] Saving fixed model to {output_path}...")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    new_model.save(output_path)
    print("  Model saved successfully!")

    # ── 验证 ──
    if args.verify and not args.no_verify:
        print("\n[Verify] Running dummy inference...")
        try:
            dummy_input = np.random.rand(1, IMG_HEIGHT, IMG_WIDTH, 1).astype(
                np.float32
            )
            output = new_model.predict(dummy_input, verbose=0)
            print(f"  Input shape:  (1, {IMG_HEIGHT}, {IMG_WIDTH}, 1)")
            print(f"  Output shape: {output.shape}")
            print(f"  Output sum per position (should be ~1.0 each): "
                  f"{[f'{s:.4f}' for s in output[0].sum(axis=-1)]}")
            print("  ✓ Verification passed!")
        except Exception as e:
            print(f"  ✗ Verification failed: {e}")
            print("  The model was saved but may have issues.")
            sys.exit(1)

    # ── 完成 ──
    print(f"\n{'='*60}")
    print("Fix completed successfully!")
    print(f"Fixed model: {output_path}")
    print(f"You can now use this model with any backend (jax/torch/tensorflow).")


if __name__ == "__main__":
    main()