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
        image = np.array(example["image"], dtype=np.float32) / 255.0
        image = np.expand_dims(image, axis=0)
        
        true_label = example["label"]
        
        # 预测
        predictions = model.predict(image, verbose=0)
        # predictions shape: (1, 4, 256)
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
    parser.add_argument('--model', type=str, default='models/luoguCaptcha_final.keras')
    parser.add_argument('--size', type=int, default=1000, help='测试样本数量')
    parser.add_argument('--compare', action='store_true', help='对比两个阶段的模型')
    
    args = parser.parse_args()
    
    if args.compare:
        print("\n对比两个阶段的模型性能...\n")
        
        if os.path.exists('models/stage1_cnn_dense.keras'):
            print("阶段1 (CNN+Dense):")
            stage1_acc, _ = test_model('models/stage1_cnn_dense.keras', args.size)
        else:
            print("⚠️  阶段1模型不存在")
            stage1_acc = 0
        
        if os.path.exists('models/stage2_cnn_lstm.keras'):
            print("\n阶段2 (CNN+LSTM):")
            stage2_acc, _ = test_model('models/stage2_cnn_lstm.keras', args.size)
        else:
            print("⚠️  阶段2模型不存在")
            stage2_acc = 0
        
        if stage1_acc > 0 and stage2_acc > 0:
            print("\n" + "="*60)
            print("对比总结")
            print("="*60)
            print(f"阶段1准确率: {stage1_acc:.4f}")
            print(f"阶段2准确率: {stage2_acc:.4f}")
            print(f"提升幅度: {(stage2_acc - stage1_acc)*100:+.2f}%")
            print("="*60)
    else:
        test_model(args.model, args.size)