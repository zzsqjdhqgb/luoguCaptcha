# check_labels.py
from datasets import load_dataset
import numpy as np

dataset = load_dataset("langningchen/luogu-captcha-dataset")
sample = dataset["train"][0]

print("检查标签格式:")
print(f"标签值: {sample['label']}")
print(f"标签类型: {type(sample['label'])}")
print(f"转为字符: {''.join(chr(c) for c in sample['label'])}")

# 检查标签范围
all_labels = []
for i in range(min(1000, len(dataset["train"]))):
    all_labels.extend(dataset["train"][i]["label"])

print(f"\n标签统计:")
print(f"最小值: {min(all_labels)}")
print(f"最大值: {max(all_labels)}")
print(f"唯一值数量: {len(set(all_labels))}")

# 🔍 关键检查：是否超出范围
if max(all_labels) >= 256:
    print(f"🚨 错误：标签值 {max(all_labels)} 超出模型输出范围 [0, 255]!")