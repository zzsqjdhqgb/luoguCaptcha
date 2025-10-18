# test_model.py
"""快速测试模型性能"""

import os
import tensorflow as tf
from tensorflow import keras
import numpy as np
from datasets import load_dataset
from tqdm import tqdm

def test_model(model_path, test_size=1000):
    """测试模型在测试集上的表现"""
    print(f"加载模型: {model_path}")
    model = keras.models.load_model(model_path)
    
    print("加载测试数据集...")
    dataset = load_dataset("langningchen/luogu-captcha-dataset", split='test')
    
    test_size = min(test_size, len(dataset))
    dataset = dataset.select(range(test_size))
    
    correct = 0
    char_correct = [0, 0, 0, 0]
    total = test_size
    
    print(f"\n开始测试 {test_size} 个样本...")
    
    for example in tqdm(dataset):
        # 预处理
        image = np.array(example["image"], dtype=np.float32)
        if image.max() > 1.0:
            image = image / 255.0
        image = np.expand_dims(image, axis=0)
        
        true_label = example["label"]
        
        # 预测
        predictions = model.predict(image, verbose=0)
        pred_label = [np.argmax(predictions[0, i, :]) for i in range(4)]
        
        # 统计
        if pred_label == true_label:
            correct += 1
        
        for i in range(4):
            if pred_label[i] == true_label[i]:
                char_correct[i] += 1
    
    # 计算准确率
    full_acc = correct / total
    char_acc = [c / total for c in char_correct]
    avg_char_acc = np.mean(char_acc)
    
    # 打印结果
    print("\n" + "="*60)
    print("测试结果")
    print("="*60)
    print(f"完整验证码准确率: {full_acc:.4f} ({correct}/{total})")
    print(f"平均字符准确率: {avg_char_acc:.4f}")
    print("\n各位置字符准确率:")
    for i, acc in enumerate(char_acc):
        print(f"  位置 {i}: {acc:.4f} ({char_correct[i]}/{total})")
    print("="*60)
    
    return full_acc, char_acc


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='models/luoguCaptcha.keras')
    parser.add_argument('--size', type=int, default=1000, help='测试样本数量')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model):
        print(f"❌ 错误: 模型文件不存在: {args.model}")
        print("请先运行 train.py 训练模型")
    else:
        test_model(args.model, args.size)