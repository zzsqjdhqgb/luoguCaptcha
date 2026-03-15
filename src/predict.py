# Copyright (C) 2025 Langning Chen
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

# ---- 必须在 import keras 之前设置后端 ----
import os
os.environ["KERAS_BACKEND"] = "jax"

import numpy as np
import jax
import keras
from PIL import Image
import sys
import io
import json
import base64
import http.server

# 导入自定义层（触发注册），使 load_model 能识别自定义对象
import model as custom_model_module  # noqa: F401  # 重命名避免与后续 model 变量冲突
from config import CHAR_SIZE, CHARS_PER_LABEL, IMG_HEIGHT, IMG_WIDTH, MODEL_PATH

# 打印 JAX 后端设备信息
devices = jax.devices()
print(f"Keras backend: {keras.backend.backend()}")
print(f"JAX devices: {devices}")
if any(d.platform == "gpu" for d in devices):
    print(f"Using GPU: {[d for d in devices if d.platform == 'gpu']}")
else:
    print("Using CPU")

model_path = MODEL_PATH

# 加载模型
try:
    captcha_model = keras.models.load_model(model_path)
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    print(f"Please ensure the model is saved at {model_path}")
    sys.exit(1)


def preprocess_image(img: Image.Image) -> np.ndarray:
    """将 PIL Image 预处理为模型输入张量。

    灰度转换 -> NumPy -> 归一化 -> 添加通道 -> 添加 Batch 维度
    目标形状: (1, IMG_HEIGHT, IMG_WIDTH, 1)
    """
    image_np = np.array(img.convert("L"), dtype=np.float32) / 255.0
    image_np = np.expand_dims(image_np, axis=-1)   # (H, W, 1)
    image_np = np.expand_dims(image_np, axis=0)    # (1, H, W, 1)
    return image_np


def decode_prediction(pred_probabilities: np.ndarray) -> str:
    """将模型输出概率解码为验证码字符串。

    pred_probabilities 形状: (1, CHARS_PER_LABEL, CHAR_SIZE)
    """
    # 使用 numpy argmax（JAX 数组也兼容）
    predicted_ascii_codes = np.argmax(pred_probabilities, axis=-1)[0]
    predicted_captcha = "".join(map(chr, predicted_ascii_codes))
    return predicted_captcha


def predict_captcha(image_path: str) -> str:
    """预测单个图像文件"""
    try:
        img = Image.open(image_path)
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        sys.exit(1)

    image_np = preprocess_image(img)
    pred_probabilities = captcha_model.predict(image_np)
    return decode_prediction(pred_probabilities)


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1].isdigit():
        port = int(sys.argv[1])
        print(f"Starting HTTP server on port {port}...")

        class CaptchaHandler(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                try:
                    # 读取请求数据
                    length = int(self.headers["Content-Length"])
                    data = json.loads(self.rfile.read(length))

                    # 解码 base64 图像
                    image_data = base64.b64decode(data["image"])
                    image = Image.open(io.BytesIO(image_data))

                    # 预处理图像
                    image_np = preprocess_image(image)

                    # 预测
                    pred_probabilities = captcha_model.predict(image_np)
                    predicted_captcha = decode_prediction(pred_probabilities)

                    # 返回 JSON 响应
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    response = json.dumps({"prediction": predicted_captcha})
                    self.wfile.write(response.encode())

                except Exception as e:
                    # 错误处理
                    self.send_response(500)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    error_response = json.dumps({"error": str(e)})
                    self.wfile.write(error_response.encode())

        # 启动服务器
        server = http.server.HTTPServer(("", port), CaptchaHandler)
        print(f"Server running at http://localhost:{port}")
        print('Send POST request with JSON: {"image": "base64_encoded_image"}')
        server.serve_forever()

    elif len(sys.argv) == 2:
        # 单张图片预测
        result = predict_captcha(sys.argv[1])
        print(f"Predicted captcha: {result}")

    else:
        print("Usage:")
        print("  python predict.py <image_path>          # Predict single image")
        print("  python predict.py <port>                # Start HTTP server")
        print("                                           (port must be a number)")
        print("")
        print("HTTP Server Usage:")
        print("  POST http://localhost:<port>/")
        print('  Body: {"image": "base64_encoded_image"}')
        print('  Response: {"prediction": "captcha_text"}')