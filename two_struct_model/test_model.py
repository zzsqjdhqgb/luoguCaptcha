# inference.py
import os
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

class CaptchaPredictor:
    """验证码预测器"""
    
    def __init__(self, model_path, img_height=35, img_width=90):
        self.model = keras.models.load_model(model_path)
        self.img_height = img_height
        self.img_width = img_width
        print(f"✅ 模型加载成功: {model_path}")
    
    def preprocess_image(self, image_path):
        """预处理图像"""
        # 读取图像
        img = Image.open(image_path).convert('L')  # 转为灰度图
        img = img.resize((self.img_width, self.img_height))
        
        # 转为numpy数组并归一化
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=-1)  # 添加通道维度
        img_array = np.expand_dims(img_array, axis=0)   # 添加批次维度
        
        return img_array, img
    
    def predict(self, image_path, visualize=True):
        """预测验证码"""
        # 预处理
        img_array, original_img = self.preprocess_image(image_path)
        
        # 预测
        predictions = self.model.predict(img_array, verbose=0)
        
        # 解码预测结果
        predicted_chars = []
        confidences = []
        
        for i in range(len(predictions)):
            char_probs = predictions[i][0]
            predicted_code = np.argmax(char_probs)
            confidence = char_probs[predicted_code]
            
            predicted_chars.append(chr(predicted_code))
            confidences.append(confidence)
        
        result = ''.join(predicted_chars)
        avg_confidence = np.mean(confidences)
        
        # 可视化
        if visualize:
            self._visualize_prediction(
                original_img, 
                result, 
                confidences, 
                avg_confidence
            )
        
        return result, avg_confidence, confidences
    
    def _visualize_prediction(self, image, result, confidences, avg_confidence):
        """可视化预测结果"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # 显示图像
        ax1.imshow(image, cmap='gray')
        ax1.axis('off')
        ax1.set_title(f'预测结果: {result}\n平均置信度: {avg_confidence:.2%}', 
                     fontsize=14, fontweight='bold')
        
        # 显示每个字符的置信度
        positions = range(len(result))
        colors = ['green' if c > 0.9 else 'orange' if c > 0.7 else 'red' 
                 for c in confidences]
        
        bars = ax2.bar(positions, confidences, color=colors, alpha=0.7)
        ax2.set_xlabel('字符位置', fontsize=12)
        ax2.set_ylabel('置信度', fontsize=12)
        ax2.set_title('各字符预测置信度', fontsize=14, fontweight='bold')
        ax2.set_xticks(positions)
        ax2.set_xticklabels([f'{i}\n{result[i]}' for i in positions])
        ax2.set_ylim([0, 1])
        ax2.axhline(y=0.9, color='green', linestyle='--', alpha=0.5, label='高置信度')
        ax2.axhline(y=0.7, color='orange', linestyle='--', alpha=0.5, label='中置信度')
        ax2.grid(axis='y', alpha=0.3)
        ax2.legend()
        
        # 在柱状图上标注具体数值
        for bar, conf in zip(bars, confidences):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{conf:.2%}',
                    ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.show()
    
    def batch_predict(self, image_folder, save_results=True):
        """批量预测文件夹中的图像"""
        results = []
        
        # 获取所有图像文件
        image_files = [f for f in os.listdir(image_folder) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        
        print(f"\n开始批量预测，共 {len(image_files)} 张图像...")
        
        for img_file in image_files:
            img_path = os.path.join(image_folder, img_file)
            
            try:
                result, avg_conf, char_confs = self.predict(img_path, visualize=False)
                
                results.append({
                    'filename': img_file,
                    'prediction': result,
                    'avg_confidence': avg_conf,
                    'char_confidences': char_confs
                })
                
                print(f"✅ {img_file}: {result} (置信度: {avg_conf:.2%})")
                
            except Exception as e:
                print(f"❌ {img_file}: 预测失败 - {str(e)}")
        
        # 保存结果
        if save_results and results:
            import csv
            output_file = os.path.join(image_folder, 'predictions.csv')
            
            with open(output_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=[
                    'filename', 'prediction', 'avg_confidence', 
                    'char0_conf', 'char1_conf', 'char2_conf', 'char3_conf'
                ])
                writer.writeheader()
                
                for r in results:
                    row = {
                        'filename': r['filename'],
                        'prediction': r['prediction'],
                        'avg_confidence': f"{r['avg_confidence']:.4f}",
                        'char0_conf': f"{r['char_confidences'][0]:.4f}",
                        'char1_conf': f"{r['char_confidences'][1]:.4f}",
                        'char2_conf': f"{r['char_confidences'][2]:.4f}",
                        'char3_conf': f"{r['char_confidences'][3]:.4f}",
                    }
                    writer.writerow(row)
            
            print(f"\n✅ 预测结果已保存到: {output_file}")
        
        return results


def main():
    """主函数 - 演示用法"""
    import argparse
    
    parser = argparse.ArgumentParser(description='洛谷验证码识别推理脚本')
    parser.add_argument('--model', type=str, default='models/luoguCaptcha_final.keras',
                       help='模型路径')
    parser.add_argument('--image', type=str, help='单张图像路径')
    parser.add_argument('--folder', type=str, help='图像文件夹路径（批量预测）')
    parser.add_argument('--stage', type=str, choices=['1', '2', 'final'], 
                       default='final', help='使用哪个阶段的模型')
    
    args = parser.parse_args()
    
    # 根据stage参数选择模型
    if args.stage == '1':
        model_path = 'models/stage1_cnn_dense.keras'
    elif args.stage == '2':
        model_path = 'models/stage2_cnn_lstm.keras'
    else:
        model_path = args.model
    
    # 检查模型是否存在
    if not os.path.exists(model_path):
        print(f"❌ 错误: 模型文件不存在: {model_path}")
        print("请先运行 train.py 训练模型")
        return
    
    # 创建预测器
    predictor = CaptchaPredictor(model_path)
    
    # 单张图像预测
    if args.image:
        if not os.path.exists(args.image):
            print(f"❌ 错误: 图像文件不存在: {args.image}")
            return
        
        print(f"\n预测图像: {args.image}")
        result, confidence, char_confs = predictor.predict(args.image)
        
        print(f"\n预测结果: {result}")
        print(f"平均置信度: {confidence:.2%}")
        print("各字符置信度:")
        for i, (char, conf) in enumerate(zip(result, char_confs)):
            print(f"  位置 {i} ('{char}'): {conf:.2%}")
    
    # 批量预测
    elif args.folder:
        if not os.path.exists(args.folder):
            print(f"❌ 错误: 文件夹不存在: {args.folder}")
            return
        
        results = predictor.batch_predict(args.folder)
        
        # 统计
        if results:
            avg_conf = np.mean([r['avg_confidence'] for r in results])
            print(f"\n批量预测完成！")
            print(f"总图像数: {len(results)}")
            print(f"平均置信度: {avg_conf:.2%}")
    
    else:
        print("请使用 --image 指定单张图像或 --folder 指定文件夹")
        print("示例:")
        print("  python inference.py --image test.png")
        print("  python inference.py --folder test_images/")
        print("  python inference.py --image test.png --stage 1  # 使用阶段1模型")


if __name__ == "__main__":
    main()