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
将 Keras (.keras) 模型导出为 ONNX 格式。

流程: Keras (JAX backend) → keras.export() → TF SavedModel → tf2onnx → ONNX

Usage:
  python scripts/export_onnx.py
  python scripts/export_onnx.py --output models/captcha.onnx --opset 17
  python scripts/export_onnx.py --no-verify

Dependencies:
  pip install jax jaxlib tensorflow tf2onnx onnx onnxruntime
"""

import argparse
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np


def check_dependencies():
    """检查所有必要依赖是否已安装。"""
    missing = []

    try:
        import jax  # noqa: F401
    except ImportError:
        missing.append("jax jaxlib")

    try:
        import tensorflow  # noqa: F401
    except ImportError:
        missing.append("tensorflow")

    try:
        import tf2onnx  # noqa: F401
    except ImportError:
        missing.append("tf2onnx")

    try:
        import onnx  # noqa: F401
    except ImportError:
        missing.append("onnx")

    if missing:
        print("Error: Missing dependencies!")
        print(f"  pip install {' '.join(missing)}")
        sys.exit(1)


def export_savedmodel(model_path: str, saved_model_dir: str, img_height: int, img_width: int):
    """
    用 JAX 后端加载 Keras 模型，导出为 TF SavedModel。
    """
    os.environ["KERAS_BACKEND"] = "jax"

    import keras
    import model as custom_model_module  # noqa: F401

    print(f"Loading model from {model_path} ...")
    keras_model = keras.models.load_model(model_path)
    keras_model.summary()

    # 验证推理
    dummy = np.zeros((1, img_height, img_width, 1), dtype=np.float32)
    output = keras_model.predict(dummy, verbose=0)
    print(f"Keras output shape: {output.shape}")
    print(f"Keras output range: [{output.min():.6f}, {output.max():.6f}]")

    # 导出 SavedModel
    print(f"Exporting SavedModel to {saved_model_dir} ...")
    keras_model.export(saved_model_dir, format="tf_saved_model")
    print("SavedModel exported successfully.")

    return output  # 返回参考输出用于验证


def savedmodel_to_onnx(saved_model_dir: str, output_path: str, opset: int):
    """
    使用 tf2onnx 命令行工具将 SavedModel 转换为 ONNX。

    tf2onnx 的 Python API (convert.from_saved_model) 在某些版本中不存在，
    但命令行工具 `python -m tf2onnx.convert` 始终可用且最稳定。
    """
    print(f"\nConverting SavedModel → ONNX (opset {opset}) ...")

    cmd = [
        sys.executable, "-m", "tf2onnx.convert",
        "--saved-model", saved_model_dir,
        "--output", output_path,
        "--opset", str(opset),
    ]

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    # 打印 tf2onnx 的输出
    if result.stdout:
        # 只打印关键行，过滤冗长的 verbose 输出
        for line in result.stdout.strip().split("\n"):
            if any(kw in line.lower() for kw in [
                "onnx", "opset", "graph", "model", "save", "convert",
                "input", "output", "error", "warn", "fail",
            ]):
                print(f"  [tf2onnx] {line}")

    if result.returncode != 0:
        print(f"\ntf2onnx stderr:\n{result.stderr}")
        raise RuntimeError(f"tf2onnx failed with return code {result.returncode}")

    if not os.path.exists(output_path):
        # 有时 tf2onnx 成功但没有打印错误，检查 stderr
        print(f"\ntf2onnx stderr:\n{result.stderr}")
        raise RuntimeError(f"ONNX file was not created at {output_path}")

    file_size = os.path.getsize(output_path) / 1024 / 1024
    print(f"ONNX model saved: {output_path} ({file_size:.2f} MB)")


def rename_onnx_io(onnx_path: str):
    """
    重命名 ONNX 模型的输入输出为友好名称。
    tf2onnx 生成的名称通常很长（如 'serving_default_input:0'），这里做简化。
    """
    try:
        import onnx

        model = onnx.load(onnx_path)

        # 重命名输入
        if len(model.graph.input) == 1:
            old_input_name = model.graph.input[0].name
            new_input_name = "input"

            if old_input_name != new_input_name:
                print(f"Renaming input: '{old_input_name}' → '{new_input_name}'")
                model.graph.input[0].name = new_input_name

                # 更新所有引用
                for node in model.graph.node:
                    for i, inp in enumerate(node.input):
                        if inp == old_input_name:
                            node.input[i] = new_input_name

        # 重命名输出
        if len(model.graph.output) == 1:
            old_output_name = model.graph.output[0].name
            new_output_name = "output"

            if old_output_name != new_output_name:
                print(f"Renaming output: '{old_output_name}' → '{new_output_name}'")
                model.graph.output[0].name = new_output_name

                for node in model.graph.node:
                    for i, out in enumerate(node.output):
                        if out == old_output_name:
                            node.output[i] = new_output_name

        onnx.save(model, onnx_path)
        print("I/O names updated.")

    except Exception as e:
        print(f"Warning: Could not rename I/O: {e} (non-critical)")


def verify_onnx(onnx_path: str, img_height: int, img_width: int,
                reference_output: np.ndarray = None):
    """验证导出的 ONNX 模型。"""
    print("\n" + "=" * 60)
    print("Verifying ONNX model")
    print("=" * 60)

    import onnx
    import onnxruntime as ort

    # 1. 结构检查
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("✓ ONNX model structure is valid")

    # 2. 输入输出信息
    print("\nInputs:")
    for inp in onnx_model.graph.input:
        shape = []
        for d in inp.type.tensor_type.shape.dim:
            if d.dim_param:
                shape.append(d.dim_param)
            elif d.dim_value:
                shape.append(d.dim_value)
            else:
                shape.append("?")
        print(f"  {inp.name}: {shape}")

    print("Outputs:")
    for out in onnx_model.graph.output:
        shape = []
        for d in out.type.tensor_type.shape.dim:
            if d.dim_param:
                shape.append(d.dim_param)
            elif d.dim_value:
                shape.append(d.dim_value)
            else:
                shape.append("?")
        print(f"  {out.name}: {shape}")

    # 3. 推理测试
    print("\nRunning inference test ...")
    session = ort.InferenceSession(onnx_path)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    dummy = np.zeros((1, img_height, img_width, 1), dtype=np.float32)
    result = session.run([output_name], {input_name: dummy})[0]
    print(f"  Input:  {input_name} {dummy.shape}")
    print(f"  Output: {output_name} {result.shape} dtype={result.dtype}")
    print(f"  Range:  [{result.min():.6f}, {result.max():.6f}]")

    # 4. Softmax 检查
    prob_sums = result[0].sum(axis=-1)
    print(f"  Per-position prob sums: {prob_sums}")
    if np.allclose(prob_sums, 1.0, atol=1e-3):
        print("✓ Softmax outputs valid (each position sums to ~1.0)")
    else:
        print("⚠ WARNING: Softmax outputs may not sum to 1.0!")

    # 5. 与 Keras 输出对比
    if reference_output is not None:
        diff = np.abs(result - reference_output)
        max_diff = diff.max()
        mean_diff = diff.mean()
        print(f"\n  vs Keras reference:")
        print(f"    Max  abs diff: {max_diff:.8f}")
        print(f"    Mean abs diff: {mean_diff:.8f}")
        if max_diff < 1e-4:
            print("✓ Output matches Keras reference (< 1e-4)")
        elif max_diff < 1e-2:
            print("~ Output approximately matches (< 1e-2, likely float precision)")
        else:
            print("⚠ WARNING: Large difference from Keras reference!")

    # 6. 性能基准
    print("\nBenchmark (100 iterations) ...")
    times = []
    for _ in range(100):
        t0 = time.perf_counter()
        session.run([output_name], {input_name: dummy})
        times.append(time.perf_counter() - t0)

    times_ms = np.array(times) * 1000
    print(f"  Mean:   {times_ms.mean():.2f} ms")
    print(f"  Median: {np.median(times_ms):.2f} ms")
    print(f"  Min:    {times_ms.min():.2f} ms")
    print(f"  Std:    {times_ms.std():.2f} ms")

    # 7. 文件大小
    size_mb = os.path.getsize(onnx_path) / 1024 / 1024
    print(f"\nModel size: {size_mb:.2f} MB")
    print("✓ Verification complete!")


def main():
    parser = argparse.ArgumentParser(
        description="Export Keras captcha model to ONNX",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/export_onnx.py
  python scripts/export_onnx.py --output models/captcha.onnx
  python scripts/export_onnx.py --opset 15
  python scripts/export_onnx.py --no-verify
        """,
    )
    parser.add_argument(
        "--model", "-m", default=None,
        help="Keras model path (default: from config.py)",
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help="Output ONNX path (default: same as model with .onnx)",
    )
    parser.add_argument(
        "--opset", type=int, default=17,
        help="ONNX opset version (default: 17)",
    )
    parser.add_argument(
        "--no-verify", action="store_true",
        help="Skip verification after export",
    )
    parser.add_argument(
        "--keep-savedmodel", default=None,
        help="Keep intermediate SavedModel at this path (default: use temp dir)",
    )
    args = parser.parse_args()

    # 导入配置
    try:
        from config import MODEL_PATH, IMG_HEIGHT, IMG_WIDTH
        default_model_path = MODEL_PATH
        img_height, img_width = IMG_HEIGHT, IMG_WIDTH
    except ImportError:
        default_model_path = "models/luoguCaptcha.keras"
        img_height, img_width = 35, 90
        print("Warning: config.py not found, using defaults")

    model_path = args.model or default_model_path
    if not os.path.exists(model_path):
        print(f"Error: Model not found: {model_path}")
        sys.exit(1)

    output_path = args.output or str(Path(model_path).with_suffix(".onnx"))
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    print(f"Model:   {model_path}")
    print(f"Output:  {output_path}")
    print(f"Opset:   {args.opset}")
    print(f"Image:   {img_width}×{img_height}")
    print()

    check_dependencies()

    # 步骤 1: Keras → SavedModel
    if args.keep_savedmodel:
        saved_model_dir = args.keep_savedmodel
        os.makedirs(saved_model_dir, exist_ok=True)
        ref_output = export_savedmodel(model_path, saved_model_dir, img_height, img_width)
        # 步骤 2: SavedModel → ONNX
        savedmodel_to_onnx(saved_model_dir, output_path, args.opset)
    else:
        with tempfile.TemporaryDirectory() as tmpdir:
            saved_model_dir = os.path.join(tmpdir, "saved_model")
            ref_output = export_savedmodel(model_path, saved_model_dir, img_height, img_width)
            # 步骤 2: SavedModel → ONNX
            savedmodel_to_onnx(saved_model_dir, output_path, args.opset)

    # 步骤 3: 重命名 I/O
    rename_onnx_io(output_path)

    # 步骤 4: 验证
    if not args.no_verify:
        try:
            verify_onnx(output_path, img_height, img_width, reference_output=ref_output)
        except ImportError as e:
            print(f"\nSkipping verification: {e}")
            print("Install: pip install onnxruntime")
        except Exception as e:
            print(f"\nVerification error: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'=' * 60}")
    print(f"✓ Export complete: {output_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()