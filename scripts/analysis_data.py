# diagnose_dataset_optimized.py
import numpy as np
from datasets import load_dataset
from collections import Counter, defaultdict
import json
from tqdm import tqdm
import hashlib

print("="*70)
print("数据集质量诊断工具 (优化版 - 使用哈希表)".center(70))
print("="*70)

# 加载数据集
print("\n加载数据集...")
dataset_dict = load_dataset("langningchen/luogu-captcha-dataset")
train_ds = dataset_dict["train"]
test_ds = dataset_dict["test"]

print(f"训练集大小: {len(train_ds)}")
print(f"测试集大小: {len(test_ds)}")

# ============================================================
# 辅助函数
# ============================================================

def get_label_string(label):
    """将标签转换为字符串"""
    return ''.join([chr(c) for c in label])

def fast_image_hash(image):
    """快速计算图像哈希 - 使用MD5"""
    img_array = np.array(image, dtype=np.uint8)
    return hashlib.md5(img_array.tobytes()).hexdigest()

# ============================================================
# 1. 标签分析 (使用哈希表)
# ============================================================
print("\n" + "="*70)
print("1. 标签重复分析")
print("="*70)

print("\n处理训练集标签...")
train_labels_dict = {}  # {label_string: [indices]}
for idx in tqdm(range(len(train_ds)), desc="训练集"):
    label_str = get_label_string(train_ds[idx]["label"])
    if label_str not in train_labels_dict:
        train_labels_dict[label_str] = []
    train_labels_dict[label_str].append(idx)

print("\n处理测试集标签...")
test_labels_dict = {}  # {label_string: [indices]}
for idx in tqdm(range(len(test_ds)), desc="测试集"):
    label_str = get_label_string(test_ds[idx]["label"])
    if label_str not in test_labels_dict:
        test_labels_dict[label_str] = []
    test_labels_dict[label_str].append(idx)

# 统计
train_unique_labels = len(train_labels_dict)
test_unique_labels = len(test_labels_dict)

print(f"\n训练集唯一标签数: {train_unique_labels}")
print(f"测试集唯一标签数: {test_unique_labels}")

# 计算重复次数
train_label_counts = {label: len(indices) for label, indices in train_labels_dict.items()}
test_label_counts = {label: len(indices) for label, indices in test_labels_dict.items()}

# 排序找出最常见的
train_sorted = sorted(train_label_counts.items(), key=lambda x: x[1], reverse=True)
test_sorted = sorted(test_label_counts.items(), key=lambda x: x[1], reverse=True)

print(f"\n训练集中最常见的20个标签:")
for label, count in train_sorted[:20]:
    percentage = count / len(train_ds) * 100
    print(f"  '{label}': {count:5d} 次 ({percentage:5.2f}%)")

max_train_dup = train_sorted[0][1] if train_sorted else 0
print(f"\n训练集单个标签最大重复次数: {max_train_dup}")

highly_duplicated = sum(1 for label, count in train_label_counts.items() if count > 100)
print(f"重复超过100次的标签数量: {highly_duplicated}")

# ============================================================
# 2. 训练集/测试集标签重叠分析
# ============================================================
print("\n" + "="*70)
print("2. 训练集/测试集标签重叠分析 (数据泄漏检测)")
print("="*70)

train_label_set = set(train_labels_dict.keys())
test_label_set = set(test_labels_dict.keys())

overlap_labels = train_label_set & test_label_set
only_train = train_label_set - test_label_set
only_test = test_label_set - train_label_set

print(f"\n仅在训练集的标签: {len(only_train)}")
print(f"仅在测试集的标签: {len(only_test)}")
print(f"同时出现的标签: {len(overlap_labels)}")

if train_label_set:
    print(f"重叠比例 (训练集): {len(overlap_labels)/len(train_label_set)*100:.2f}%")
if test_label_set:
    print(f"重叠比例 (测试集): {len(overlap_labels)/len(test_label_set)*100:.2f}%")

if overlap_labels:
    print(f"\n⚠️  警告: 发现 {len(overlap_labels)} 个标签同时出现在训练集和测试集!")
    
    # 统计重叠标签涉及的样本数
    overlap_train_samples = sum(len(train_labels_dict[label]) for label in overlap_labels)
    overlap_test_samples = sum(len(test_labels_dict[label]) for label in overlap_labels)
    
    print(f"涉及训练样本: {overlap_train_samples}/{len(train_ds)} ({overlap_train_samples/len(train_ds)*100:.2f}%)")
    print(f"涉及测试样本: {overlap_test_samples}/{len(test_ds)} ({overlap_test_samples/len(test_ds)*100:.2f}%)")
    
    print(f"\n重叠标签示例 (前20个):")
    for i, label in enumerate(list(overlap_labels)[:20], 1):
        train_count = len(train_labels_dict[label])
        test_count = len(test_labels_dict[label])
        print(f"  {i:2d}. '{label}': 训练集{train_count:4d}次, 测试集{test_count:3d}次")

# ============================================================
# 3. 图像哈希分析 (使用MD5哈希表)
# ============================================================
print("\n" + "="*70)
print("3. 图像重复分析 (基于MD5哈希)")
print("="*70)

print("\n计算训练集图像哈希...")
train_image_dict = {}  # {hash: [indices]}
for idx in tqdm(range(len(train_ds)), desc="训练集图像"):
    img_hash = fast_image_hash(train_ds[idx]["image"])
    if img_hash not in train_image_dict:
        train_image_dict[img_hash] = []
    train_image_dict[img_hash].append(idx)

train_unique_images = len(train_image_dict)
train_duplicate_groups = sum(1 for indices in train_image_dict.values() if len(indices) > 1)
train_duplicated_samples = sum(len(indices) - 1 for indices in train_image_dict.values() if len(indices) > 1)

print(f"\n训练集唯一图像数: {train_unique_images}")
print(f"训练集总样本数: {len(train_ds)}")
print(f"重复图像组数: {train_duplicate_groups}")
print(f"重复样本总数: {train_duplicated_samples}")
print(f"重复比例: {train_duplicated_samples/len(train_ds)*100:.2f}%")

# 找出重复最多的图像
if train_duplicate_groups > 0:
    train_dup_sorted = sorted(
        [(img_hash, indices) for img_hash, indices in train_image_dict.items() if len(indices) > 1],
        key=lambda x: len(x[1]),
        reverse=True
    )
    
    print(f"\n重复最多的10组图像:")
    for i, (img_hash, indices) in enumerate(train_dup_sorted[:10], 1):
        labels = [get_label_string(train_ds[idx]["label"]) for idx in indices[:5]]
        label_set = set(get_label_string(train_ds[idx]["label"]) for idx in indices)
        print(f"  {i:2d}. 重复{len(indices):4d}次, "
              f"标签{'相同' if len(label_set)==1 else '不同'}, "
              f"示例: {labels}")

# 测试集
print("\n计算测试集图像哈希...")
test_image_dict = {}  # {hash: [indices]}
for idx in tqdm(range(len(test_ds)), desc="测试集图像"):
    img_hash = fast_image_hash(test_ds[idx]["image"])
    if img_hash not in test_image_dict:
        test_image_dict[img_hash] = []
    test_image_dict[img_hash].append(idx)

test_unique_images = len(test_image_dict)
test_duplicate_groups = sum(1 for indices in test_image_dict.values() if len(indices) > 1)
test_duplicated_samples = sum(len(indices) - 1 for indices in test_image_dict.values() if len(indices) > 1)

print(f"\n测试集唯一图像数: {test_unique_images}")
print(f"测试集总样本数: {len(test_ds)}")
print(f"重复图像组数: {test_duplicate_groups}")
print(f"重复样本总数: {test_duplicated_samples}")
print(f"重复比例: {test_duplicated_samples/len(test_ds)*100:.2f}%")

# ============================================================
# 4. 训练集/测试集图像重叠分析
# ============================================================
print("\n" + "="*70)
print("4. 训练集/测试集图像重叠分析 (严重数据泄漏)")
print("="*70)

train_hash_set = set(train_image_dict.keys())
test_hash_set = set(test_image_dict.keys())

image_overlap = train_hash_set & test_hash_set

print(f"\n同时出现在训练和测试集的相同图像数: {len(image_overlap)}")

if train_hash_set:
    print(f"重叠比例 (相对训练集唯一图像): {len(image_overlap)/len(train_hash_set)*100:.2f}%")
if test_hash_set:
    print(f"重叠比例 (相对测试集唯一图像): {len(image_overlap)/len(test_hash_set)*100:.2f}%")

if image_overlap:
    # 计算涉及的样本数
    overlap_train_samples = sum(len(train_image_dict[h]) for h in image_overlap)
    overlap_test_samples = sum(len(test_image_dict[h]) for h in image_overlap)
    
    print(f"\n🚨 严重警告: 发现 {len(image_overlap)} 个完全相同的图像!")
    print(f"涉及训练样本: {overlap_train_samples}/{len(train_ds)} ({overlap_train_samples/len(train_ds)*100:.2f}%)")
    print(f"涉及测试样本: {overlap_test_samples}/{len(test_ds)} ({overlap_test_samples/len(test_ds)*100:.2f}%)")
    
    print(f"\n重叠图像示例 (前20个):")
    for i, img_hash in enumerate(list(image_overlap)[:20], 1):
        train_idx = train_image_dict[img_hash][0]
        test_idx = test_image_dict[img_hash][0]
        train_label = get_label_string(train_ds[train_idx]["label"])
        test_label = get_label_string(test_ds[test_idx]["label"])
        match = "✅" if train_label == test_label else "❌ 不同!"
        
        print(f"  {i:2d}. 训练[{train_idx:5d}]:'{train_label}' <=> 测试[{test_idx:4d}]:'{test_label}' {match}")

# ============================================================
# 5. 字符分布分析
# ============================================================
print("\n" + "="*70)
print("5. 字符分布分析")
print("="*70)

train_char_counter = Counter()
for label in train_labels_dict.keys():
    train_char_counter.update(label)

test_char_counter = Counter()
for label in test_labels_dict.keys():
    test_char_counter.update(label)

print(f"\n训练集唯一字符数: {len(train_char_counter)}")
print(f"测试集唯一字符数: {len(test_char_counter)}")

print(f"\n训练集字符分布:")
for char, count in train_char_counter.most_common(30):
    print(f"  '{char}' (ASCII {ord(char):3d}): {count:5d} 次")

# ============================================================
# 6. 极端情况检测
# ============================================================
print("\n" + "="*70)
print("6. 极端情况检测")
print("="*70)

print("\n检查是否真的只有少数几个唯一字符串...")

# 检查训练集是否只有很少的唯一组合
if train_unique_labels <= 50:
    print(f"🚨🚨🚨 极严重: 训练集只有 {train_unique_labels} 个唯一标签!")
    print(f"\n所有唯一标签列表:")
    for i, label in enumerate(sorted(train_labels_dict.keys()), 1):
        count = len(train_labels_dict[label])
        print(f"  {i:2d}. '{label}': {count:5d} 次 ({count/len(train_ds)*100:.2f}%)")

# 检查测试集是否是训练集的子集
if test_label_set.issubset(train_label_set):
    print(f"\n🚨 测试集的所有标签都出现在训练集中 (100%泄漏)!")

# 检查是否完全重复
if train_unique_images == test_unique_images and len(image_overlap) == train_unique_images:
    print(f"\n🚨🚨🚨 致命: 训练集和测试集的图像完全相同!")

# ============================================================
# 7. 生成报告
# ============================================================
print("\n" + "="*70)
print("7. 生成诊断报告")
print("="*70)

report = {
    "dataset_size": {
        "train": len(train_ds),
        "test": len(test_ds)
    },
    "label_analysis": {
        "train_unique_labels": train_unique_labels,
        "test_unique_labels": test_unique_labels,
        "total_unique_labels": len(train_label_set | test_label_set),
        "max_train_label_duplicates": max_train_dup,
        "labels_over_100_duplicates": highly_duplicated,
        "top_10_train_labels": train_sorted[:10]
    },
    "label_leakage": {
        "overlap_count": len(overlap_labels),
        "overlap_percentage_train": len(overlap_labels)/len(train_label_set)*100 if train_label_set else 0,
        "overlap_percentage_test": len(overlap_labels)/len(test_label_set)*100 if test_label_set else 0,
        "only_in_train": len(only_train),
        "only_in_test": len(only_test),
        "overlap_samples_train": sum(len(train_labels_dict[label]) for label in overlap_labels),
        "overlap_samples_test": sum(len(test_labels_dict[label]) for label in overlap_labels)
    },
    "image_duplication": {
        "train_unique_images": train_unique_images,
        "train_duplicate_groups": train_duplicate_groups,
        "train_duplicated_samples": train_duplicated_samples,
        "train_duplication_rate": train_duplicated_samples/len(train_ds)*100,
        "test_unique_images": test_unique_images,
        "test_duplicate_groups": test_duplicate_groups,
        "test_duplicated_samples": test_duplicated_samples,
        "test_duplication_rate": test_duplicated_samples/len(test_ds)*100
    },
    "image_leakage": {
        "overlap_count": len(image_overlap),
        "overlap_percentage_train": len(image_overlap)/len(train_hash_set)*100 if train_hash_set else 0,
        "overlap_percentage_test": len(image_overlap)/len(test_hash_set)*100 if test_hash_set else 0,
        "overlap_samples_train": sum(len(train_image_dict[h]) for h in image_overlap),
        "overlap_samples_test": sum(len(test_image_dict[h]) for h in image_overlap)
    },
    "character_distribution": {
        "train_unique_chars": len(train_char_counter),
        "test_unique_chars": len(test_char_counter),
        "train_top_chars": train_char_counter.most_common(20),
        "test_top_chars": test_char_counter.most_common(20)
    }
}

report_path = "dataset_diagnosis_report.json"
with open(report_path, 'w', encoding='utf-8') as f:
    json.dump(report, f, indent=2, ensure_ascii=False)

print(f"\n✅ 详细报告已保存: {report_path}")

# ============================================================
# 8. 最终诊断结论
# ============================================================
print("\n" + "="*70)
print("最终诊断结论".center(70))
print("="*70)

issues = []
severity_score = 0  # 严重程度评分

# 检查1: 唯一标签数量
if train_unique_labels <= 20:
    issues.append(("🚨🚨🚨 致命", f"训练集只有 {train_unique_labels} 个唯一标签，模型就是在查表!"))
    severity_score += 100
elif train_unique_labels <= 100:
    issues.append(("🚨 严重", f"训练集只有 {train_unique_labels} 个唯一标签，数据多样性极低"))
    severity_score += 50
elif train_unique_labels < 1000:
    issues.append(("⚠️  警告", f"训练集只有 {train_unique_labels} 个唯一标签，数据多样性不足"))
    severity_score += 20

# 检查2: 标签重复
if max_train_dup > len(train_ds) * 0.05:
    issues.append(("⚠️  警告", f"单个标签重复 {max_train_dup} 次，占总数 {max_train_dup/len(train_ds)*100:.1f}%"))
    severity_score += 15

# 检查3: 标签泄漏
label_leak_rate = len(overlap_labels)/len(test_label_set)*100 if test_label_set else 0
if label_leak_rate > 90:
    issues.append(("🚨🚨 严重", f"测试集 {label_leak_rate:.1f}% 的标签在训练集出现，几乎完全泄漏!"))
    severity_score += 80
elif label_leak_rate > 50:
    issues.append(("🚨 严重", f"测试集 {label_leak_rate:.1f}% 的标签在训练集出现，存在严重泄漏"))
    severity_score += 50
elif label_leak_rate > 20:
    issues.append(("⚠️  警告", f"测试集 {label_leak_rate:.1f}% 的标签在训练集出现"))
    severity_score += 20

# 检查4: 图像完全相同的泄漏
image_leak_rate = len(image_overlap)/len(test_hash_set)*100 if test_hash_set else 0
if image_leak_rate > 50:
    issues.append(("🚨🚨🚨 致命", f"测试集 {image_leak_rate:.1f}% 的图像与训练集完全相同!"))
    severity_score += 100
elif image_leak_rate > 10:
    issues.append(("🚨🚨 严重", f"测试集 {image_leak_rate:.1f}% 的图像与训练集完全相同"))
    severity_score += 60
elif image_leak_rate > 0:
    issues.append(("🚨 严重", f"发现 {len(image_overlap)} 个相同图像在训练和测试集"))
    severity_score += 30

# 检查5: 训练集图像重复
train_dup_rate = train_duplicated_samples/len(train_ds)*100
if train_dup_rate > 50:
    issues.append(("⚠️  警告", f"训练集 {train_dup_rate:.1f}% 的样本是重复图像"))
    severity_score += 15

# 检查6: 测试集是否是训练集子集
if test_label_set.issubset(train_label_set):
    issues.append(("🚨🚨 严重", "测试集所有标签都在训练集中 (100%标签泄漏)"))
    severity_score += 70

# 输出结论
if not issues:
    print("\n✅✅✅ 数据集质量良好，未发现严重问题!")
    print(f"严重程度评分: {severity_score}/500 (越低越好)")
else:
    print(f"\n发现 {len(issues)} 个问题 (严重程度评分: {severity_score}/500):\n")
    for i, (level, issue) in enumerate(issues, 1):
        print(f"{i}. [{level}] {issue}")
    
    print("\n" + "="*70)
    print("数据集质量评级".center(70))
    print("="*70)
    
    if severity_score >= 200:
        quality = "❌ 不可用"
        recommendation = "强烈建议放弃此数据集"
        color = "\033[91m"  # 红色
    elif severity_score >= 100:
        quality = "⚠️  较差"
        recommendation = "建议重新清洗或寻找其他数据集"
        color = "\033[93m"  # 黄色
    elif severity_score >= 50:
        quality = "⚠️  一般"
        recommendation = "需要进行数据清洗"
        color = "\033[93m"  # 黄色
    else:
        quality = "✅ 可用"
        recommendation = "数据质量尚可，但仍有改进空间"
        color = "\033[92m"  # 绿色
    
    reset_color = "\033[0m"
    
    print(f"\n{color}数据集质量: {quality}{reset_color}")
    print(f"建议: {recommendation}")
    
    print("\n" + "="*70)
    print("改进建议".center(70))
    print("="*70)
    
    suggestions = []
    
    if train_unique_labels < 1000:
        suggestions.append("1. 生成更多样化的验证码 (建议至少10000个唯一字符串)")
    
    if len(image_overlap) > 0:
        suggestions.append("2. 【必须】从测试集中删除与训练集相同的图像")
        suggestions.append(f"   - 需要删除的测试样本索引: {[idx for h in image_overlap for idx in test_image_dict[h]][:100]}... (共{sum(len(test_image_dict[h]) for h in image_overlap)}个)")
    
    if label_leak_rate > 50:
        suggestions.append("3. 【必须】重新划分训练/测试集，确保标签不重叠")
    
    if train_dup_rate > 30:
        suggestions.append("4. 清理训练集中的重复图像")
    
    suggestions.append("5. 考虑使用以下策略生成新数据集:")
    suggestions.append("   - 使用随机字符串生成器")
    suggestions.append("   - 确保训练/测试集字符串不重叠")
    suggestions.append("   - 每个验证码图像只生成一次")
    suggestions.append("   - 建议训练集10000+唯一验证码，测试集2000+唯一验证码")
    
    for suggestion in suggestions:
        print(f"  {suggestion}")

print("\n" + "="*70)
print("详细报告文件".center(70))
print("="*70)
print(f"\n  📊 JSON报告: {report_path}")
print(f"  📝 可以使用以下命令查看:")
print(f"     cat {report_path} | python -m json.tool")

# ============================================================
# 9. 生成清理脚本 (如果需要)
# ============================================================
if len(image_overlap) > 0 or train_duplicated_samples > 0:
    print("\n" + "="*70)
    print("10. 生成数据清理建议")
    print("="*70)
    
    cleanup_script = "cleanup_dataset.py"
    with open(cleanup_script, 'w', encoding='utf-8') as f:
        f.write("# cleanup_dataset.py\n")
        f.write("# 自动生成的数据集清理脚本\n\n")
        f.write("from datasets import load_dataset, Dataset, DatasetDict\n")
        f.write("import numpy as np\n\n")
        
        f.write("# 加载数据集\n")
        f.write('dataset_dict = load_dataset("langningchen/luogu-captcha-dataset")\n')
        f.write("train_ds = dataset_dict['train']\n")
        f.write("test_ds = dataset_dict['test']\n\n")
        
        if len(image_overlap) > 0:
            # 需要从测试集删除的索引
            indices_to_remove = sorted([idx for h in image_overlap for idx in test_image_dict[h]])
            
            f.write("# 从测试集中删除与训练集相同的图像\n")
            f.write(f"# 共需删除 {len(indices_to_remove)} 个样本\n")
            f.write(f"indices_to_remove = {indices_to_remove}\n\n")
            
            f.write("# 创建保留索引\n")
            f.write("keep_indices = [i for i in range(len(test_ds)) if i not in set(indices_to_remove)]\n\n")
            
            f.write("# 创建清理后的测试集\n")
            f.write("clean_test_ds = test_ds.select(keep_indices)\n\n")
            
            f.write("print(f'原测试集大小: {len(test_ds)}')\n")
            f.write("print(f'清理后测试集大小: {len(clean_test_ds)}')\n")
            f.write("print(f'删除样本数: {len(test_ds) - len(clean_test_ds)}')\n\n")
        
        if train_duplicated_samples > 0:
            f.write("# 从训练集删除重复图像\n")
            f.write("# TODO: 需要根据具体需求决定保留哪些重复样本\n\n")
        
        f.write("# 保存清理后的数据集\n")
        f.write("# clean_dataset = DatasetDict({\n")
        f.write("#     'train': train_ds,  # 或清理后的训练集\n")
        f.write("#     'test': clean_test_ds\n")
        f.write("# })\n")
        f.write("# clean_dataset.save_to_disk('cleaned_dataset')\n")
        f.write("# 或推送到 Hugging Face Hub\n")
        f.write("# clean_dataset.push_to_hub('your-username/cleaned-captcha-dataset')\n")
    
    print(f"\n✅ 清理脚本已生成: {cleanup_script}")
    print(f"   运行 'python {cleanup_script}' 来清理数据集")

print("\n" + "="*70 + "\n")