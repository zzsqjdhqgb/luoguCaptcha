# diagnose.py
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from datasets import load_dataset
from PIL import Image
import matplotlib.pyplot as plt
from collections import Counter

class CaptchaDiagnosis:
    def __init__(self, model_path, dataset_path):
        self.model = load_model(model_path)
        self.dataset_path = dataset_path
        
    def analyze_dataset_distribution(self):
        """分析数据集的字符分布"""
        print("\n" + "="*70)
        print("数据集分布分析".center(70))
        print("="*70 + "\n")
        
        dataset_dict = load_dataset(self.dataset_path)
        train_ds = dataset_dict["train"]
        test_ds = dataset_dict["test"]
        
        # 统计字符分布
        train_chars = []
        test_chars = []
        
        for sample in train_ds:
            train_chars.extend(sample["label"])
        
        for sample in test_ds:
            test_chars.extend(sample["label"])
        
        train_counter = Counter(train_chars)
        test_counter = Counter(test_chars)
        
        print("训练集字符分布 (Top 20):")
        for char_code, count in train_counter.most_common(20):
            char = chr(char_code)
            print(f"  '{char}' (ASCII {char_code:3d}): {count:5d} 次")
        
        print("\n测试集字符分布 (Top 20):")
        for char_code, count in test_counter.most_common(20):
            char = chr(char_code)
            print(f"  '{char}' (ASCII {char_code:3d}): {count:5d} 次")
        
        # 检查字符集差异
        train_chars_set = set(train_counter.keys())
        test_chars_set = set(test_counter.keys())
        
        only_in_train = train_chars_set - test_chars_set
        only_in_test = test_chars_set - train_chars_set
        
        if only_in_train:
            print(f"\n⚠️  仅在训练集中出现的字符: {[chr(c) for c in only_in_train]}")
        
        if only_in_test:
            print(f"\n⚠️  仅在测试集中出现的字符: {[chr(c) for c in only_in_test]}")
        
        print(f"\n训练集唯一字符数: {len(train_chars_set)}")
        print(f"测试集唯一字符数: {len(test_chars_set)}")
        print(f"交集字符数: {len(train_chars_set & test_chars_set)}")
        
        return train_counter, test_counter
    
    def analyze_image_statistics(self):
        """分析图像统计特征"""
        print("\n" + "="*70)
        print("图像统计分析".center(70))
        print("="*70 + "\n")
        
        dataset_dict = load_dataset(self.dataset_path)
        train_ds = dataset_dict["train"]
        test_ds = dataset_dict["test"]
        
        # 采样分析
        sample_size = min(1000, len(train_ds), len(test_ds))
        
        train_stats = self._compute_image_stats(train_ds, sample_size, "训练集")
        test_stats = self._compute_image_stats(test_ds, sample_size, "测试集")
        
        # 对比
        print("\n对比分析:")
        print(f"{'指标':<20} {'训练集':<15} {'测试集':<15} {'差异'}")
        print("-"*70)
        
        metrics = ['mean', 'std', 'min', 'max']
        for metric in metrics:
            train_val = train_stats[metric]
            test_val = test_stats[metric]
            diff = abs(train_val - test_val)
            print(f"{metric.upper():<20} {train_val:<15.4f} {test_val:<15.4f} {diff:.4f}")
        
        return train_stats, test_stats
    
    def _compute_image_stats(self, dataset, sample_size, name):
        """计算图像统计信息"""
        pixels = []
        
        for i, sample in enumerate(dataset):
            if i >= sample_size:
                break
            image = np.array(sample["image"], dtype=np.float32) / 255.0
            pixels.extend(image.flatten())
        
        pixels = np.array(pixels)
        
        stats = {
            'mean': np.mean(pixels),
            'std': np.std(pixels),
            'min': np.min(pixels),
            'max': np.max(pixels)
        }
        
        print(f"{name}图像统计 (采样 {sample_size}):")
        print(f"  均值: {stats['mean']:.4f}")
        print(f"  标准差: {stats['std']:.4f}")
        print(f"  最小值: {stats['min']:.4f}")
        print(f"  最大值: {stats['max']:.4f}")
        
        return stats
    
    def visualize_samples(self, num_samples=10):
        """可视化训练集和测试集样本"""
        dataset_dict = load_dataset(self.dataset_path)
        train_ds = dataset_dict["train"]
        test_ds = dataset_dict["test"]
        
        fig, axes = plt.subplots(2, num_samples, figsize=(20, 4))
        fig.suptitle('训练集 vs 测试集样本对比', fontsize=16)
        
        # 训练集样本
        for i in range(num_samples):
            sample = train_ds[i]
            image = sample["image"]
            label = ''.join([chr(c) for c in sample["label"]])
            
            axes[0, i].imshow(image, cmap='gray')
            axes[0, i].set_title(f'Train: {label}', fontsize=10)
            axes[0, i].axis('off')
        
        # 测试集样本
        for i in range(num_samples):
            sample = test_ds[i]
            image = sample["image"]
            label = ''.join([chr(c) for c in sample["label"]])
            
            axes[1, i].imshow(image, cmap='gray')
            axes[1, i].set_title(f'Test: {label}', fontsize=10)
            axes[1, i].axis('off')
        
        plt.tight_layout()
        plt.savefig('dataset_samples.png', dpi=150, bbox_inches='tight')
        print(f"\n✅ 样本对比图已保存: dataset_samples.png")
        plt.show()
    
    def test_real_world_image(self, image_path):
        """测试真实世界的图像"""
        print("\n" + "="*70)
        print("真实图像测试".center(70))
        print("="*70 + "\n")
        
        # 加载图像
        img = Image.open(image_path).convert('L')
        print(f"原始图像大小: {img.size}")
        print(f"原始图像模式: {img.mode}")
        
        # 预处理（和训练时一致）
        img_resized = img.resize((90, 35))
        img_array = np.array(img_resized, dtype=np.float32) / 255.0
        
        # 统计信息
        print(f"\n预处理后统计:")
        print(f"  形状: {img_array.shape}")
        print(f"  均值: {np.mean(img_array):.4f}")
        print(f"  标准差: {np.std(img_array):.4f}")
        print(f"  最小值: {np.min(img_array):.4f}")
        print(f"  最大值: {np.max(img_array):.4f}")
        
        # 添加通道和batch维度
        if len(img_array.shape) == 2:
            img_array = np.expand_dims(img_array, axis=-1)
        img_array = np.expand_dims(img_array, axis=0)
        
        # 预测
        predictions = self.model.predict(img_array, verbose=0)
        
        # 解码
        if isinstance(predictions, list):
            pred_chars = [np.argmax(pred[0]) for pred in predictions]
        else:
            pred_chars = tf.math.argmax(predictions, axis=-1).numpy()[0]
        
        pred_text = ''.join([chr(c) for c in pred_chars])
        
        print(f"\n预测结果: {pred_text}")
        print(f"预测ASCII: {pred_chars}")
        
        # 显示置信度
        print(f"\n各位置置信度:")
        if isinstance(predictions, list):
            for i, pred in enumerate(predictions):
                confidence = np.max(pred[0]) * 100
                char = chr(np.argmax(pred[0]))
                print(f"  位置 {i}: '{char}' (置信度: {confidence:.2f}%)")
        
        # 可视化
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        axes[0].imshow(img, cmap='gray')
        axes[0].set_title('原始图像')
        axes[0].axis('off')
        
        axes[1].imshow(img_resized, cmap='gray')
        axes[1].set_title(f'预处理后 (预测: {pred_text})')
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.savefig('real_world_test.png', dpi=150, bbox_inches='tight')
        print(f"\n✅ 测试结果已保存: real_world_test.png")
        plt.show()
        
        return pred_text
    
    def compare_preprocessing(self, image_path):
        """对比不同预处理方式"""
        print("\n" + "="*70)
        print("预处理方式对比".center(70))
        print("="*70 + "\n")
        
        img = Image.open(image_path)
        
        methods = {
            '原始灰度': lambda x: x.convert('L'),
            '自适应二值化': lambda x: self._adaptive_threshold(x),
            '直方图均衡化': lambda x: self._histogram_equalization(x),
            '去噪': lambda x: self._denoise(x),
        }
        
        fig, axes = plt.subplots(2, len(methods), figsize=(20, 8))
        
        for idx, (name, method) in enumerate(methods.items()):
            # 预处理
            processed = method(img)
            processed_resized = processed.resize((90, 35))
            
            # 预测
            img_array = np.array(processed_resized, dtype=np.float32) / 255.0
            if len(img_array.shape) == 2:
                img_array = np.expand_dims(img_array, axis=-1)
            img_array = np.expand_dims(img_array, axis=0)
            
            predictions = self.model.predict(img_array, verbose=0)
            
            if isinstance(predictions, list):
                pred_chars = [np.argmax(pred[0]) for pred in predictions]
            else:
                pred_chars = tf.math.argmax(predictions, axis=-1).numpy()[0]
            
            pred_text = ''.join([chr(c) for c in pred_chars])
            
            # 显示
            axes[0, idx].imshow(processed, cmap='gray')
            axes[0, idx].set_title(f'{name}')
            axes[0, idx].axis('off')
            
            axes[1, idx].imshow(processed_resized, cmap='gray')
            axes[1, idx].set_title(f'预测: {pred_text}')
            axes[1, idx].axis('off')
            
            print(f"{name:<20}: {pred_text}")
        
        plt.tight_layout()
        plt.savefig('preprocessing_comparison.png', dpi=150, bbox_inches='tight')
        print(f"\n✅ 预处理对比已保存: preprocessing_comparison.png")
        plt.show()
    
    def _adaptive_threshold(self, img):
        """自适应二值化"""
        import cv2
        img_array = np.array(img.convert('L'))
        binary = cv2.adaptiveThreshold(
            img_array, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        return Image.fromarray(binary)
    
    def _histogram_equalization(self, img):
        """直方图均衡化"""
        import cv2
        img_array = np.array(img.convert('L'))
        equalized = cv2.equalizeHist(img_array)
        return Image.fromarray(equalized)
    
    def _denoise(self, img):
        """去噪"""
        import cv2
        img_array = np.array(img.convert('L'))
        denoised = cv2.fastNlMeansDenoising(img_array, None, 10, 7, 21)
        return Image.fromarray(denoised)

def main():
    MODEL_PATH = "models/luoguCaptcha_crnn.h5"
    DATASET_PATH = "langningchen/luogu-captcha-dataset"
    
    print("\n" + "="*70)
    print("验证码识别问题诊断工具".center(70))
    print("="*70)
    
    diagnosis = CaptchaDiagnosis(MODEL_PATH, DATASET_PATH)
    
    # 1. 分析数据集分布
    diagnosis.analyze_dataset_distribution()
    
    # 2. 分析图像统计
    diagnosis.analyze_image_statistics()
    
    # 3. 可视化样本
    diagnosis.visualize_samples(num_samples=10)
    
    # 4. 测试真实图像（如果有）
    print("\n" + "="*70)
    print("真实图像测试")
    print("="*70)
    real_image_path = input("\n请输入真实验证码图片路径 (按Enter跳过): ").strip()
    
    if real_image_path and os.path.exists(real_image_path):
        diagnosis.test_real_world_image(real_image_path)
        diagnosis.compare_preprocessing(real_image_path)
    else:
        print("跳过真实图像测试")
    
    print("\n" + "="*70)
    print("诊断完成！".center(70))
    print("="*70 + "\n")

if __name__ == "__main__":
    main()