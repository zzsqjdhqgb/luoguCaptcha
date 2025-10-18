# predict.py
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import sys
import os
import matplotlib.pyplot as plt

# 自动选择设备
physical_devices = tf.config.list_physical_devices("GPU")
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print("Using GPU:", physical_devices[0])
else:
    print("Using CPU")

# 配置参数
CharSize = 256
CharsPerLabel = 4
IMG_HEIGHT, IMG_WIDTH = 35, 90

# 模型路径（支持两种格式）
model_path = os.path.join("models", "luoguCaptcha_crnn.h5")
if not os.path.exists(model_path):
    model_path = os.path.join("models", "luoguCaptcha.keras")

def preprocess_image(img_path, show_steps=False):
    """
    预处理输入图像
    
    Args:
        img_path: 图像路径
        show_steps: 是否显示预处理步骤
    
    Returns:
        image_np: 预处理后的图像数组
        images_dict: 包含各个处理步骤图像的字典（用于可视化）
    """
    try:
        img = Image.open(img_path)
    except FileNotFoundError:
        print(f"Error: Image file not found at {img_path}")
        sys.exit(1)
    
    # 保存处理步骤的图像
    images_dict = {}
    images_dict['original'] = img.copy()
    
    # 步骤1: 灰度转换
    img_gray = img.convert("L")
    images_dict['grayscale'] = img_gray.copy()
    
    # 步骤2: 调整大小（如果需要）
    if img_gray.size != (IMG_WIDTH, IMG_HEIGHT):
        print(f"Resizing image from {img_gray.size} to ({IMG_WIDTH}, {IMG_HEIGHT})")
        img_resized = img_gray.resize((IMG_WIDTH, IMG_HEIGHT), Image.Resampling.LANCZOS)
        images_dict['resized'] = img_resized.copy()
    else:
        img_resized = img_gray
        images_dict['resized'] = img_resized.copy()
    
    # 步骤3: 转为NumPy并归一化
    image_np = np.array(img_resized, dtype=np.float32) / 255.0
    images_dict['normalized'] = image_np.copy()
    
    # 步骤4: 添加通道维度
    image_np = np.expand_dims(image_np, axis=-1)
    
    # 步骤5: 添加 Batch 维度
    image_np = np.expand_dims(image_np, axis=0)
    images_dict['final'] = np.squeeze(image_np[0])
    
    # 打印统计信息
    print(f"\nImage preprocessing info:")
    print(f"  Original size: {img.size}, mode: {img.mode}")
    print(f"  Final shape: {image_np.shape}")
    print(f"  Pixel value range: [{image_np.min():.3f}, {image_np.max():.3f}]")
    print(f"  Mean: {image_np.mean():.3f}, Std: {image_np.std():.3f}")
    
    return image_np, images_dict

def decode_prediction(predictions, verbose=True):
    """
    解码模型预测结果
    支持两种输出格式：
    1. 列表格式 [array(1,256), array(1,256), array(1,256), array(1,256)]
    2. 单个数组格式 (1, 4, 256)
    """
    predicted_ascii_codes = []
    confidences = []
    
    # 判断输出格式
    if isinstance(predictions, list):
        # 多输出格式（CRNN模型）
        for i, pred in enumerate(predictions):
            char_idx = np.argmax(pred[0])
            confidence = np.max(pred[0])
            predicted_ascii_codes.append(char_idx)
            confidences.append(confidence)
            
            if verbose:
                char = chr(char_idx)
                print(f"  Position {i}: '{char}' (ASCII {char_idx}, confidence: {confidence:.4f})")
    else:
        # 单输出格式（原项目格式）
        predicted_ascii_codes = tf.math.argmax(predictions, axis=-1).numpy()[0].tolist()
        for i, char_idx in enumerate(predicted_ascii_codes):
            confidence = predictions[0, i, char_idx]
            confidences.append(confidence)
            
            if verbose:
                char = chr(char_idx)
                print(f"  Position {i}: '{char}' (ASCII {char_idx}, confidence: {confidence:.4f})")
    
    # 转换为字符
    predicted_captcha = "".join(map(chr, predicted_ascii_codes))
    
    if verbose:
        avg_confidence = np.mean(confidences)
        print(f"\nAverage confidence: {avg_confidence:.4f} ({avg_confidence*100:.2f}%)")
    
    return predicted_captcha, confidences

def visualize_preprocessing(images_dict, prediction_result, confidences, save_path='preprocessing_visualization.png'):
    """
    可视化预处理步骤和预测结果
    
    Args:
        images_dict: 包含各处理步骤图像的字典
        prediction_result: 预测结果文本
        confidences: 各位置的置信度
        save_path: 保存路径
    """
    # 创建子图
    fig = plt.figure(figsize=(16, 10))
    
    # 定义网格布局
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 原始图像
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(images_dict['original'])
    ax1.set_title(f"1. Original Image\nSize: {images_dict['original'].size}", fontsize=10)
    ax1.axis('off')
    
    # 灰度图像
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(images_dict['grayscale'], cmap='gray')
    ax2.set_title(f"2. Grayscale\nMode: L", fontsize=10)
    ax2.axis('off')
    
    # 调整大小后
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(images_dict['resized'], cmap='gray')
    ax3.set_title(f"3. Resized\nSize: {IMG_WIDTH}×{IMG_HEIGHT}", fontsize=10)
    ax3.axis('off')
    
    # 归一化后
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.imshow(images_dict['normalized'], cmap='gray')
    stats_text = f"Mean: {images_dict['normalized'].mean():.3f}\nStd: {images_dict['normalized'].std():.3f}"
    ax4.set_title(f"4. Normalized [0,1]\n{stats_text}", fontsize=10)
    ax4.axis('off')
    
    # 最终输入
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.imshow(images_dict['final'], cmap='gray')
    ax5.set_title(f"5. Final Input\nShape: (1, {IMG_HEIGHT}, {IMG_WIDTH}, 1)", fontsize=10)
    ax5.axis('off')
    
    # 预测结果（大图）
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.imshow(images_dict['final'], cmap='gray')
    result_color = 'green' if np.mean(confidences) > 0.9 else 'orange' if np.mean(confidences) > 0.7 else 'red'
    ax6.set_title(f"Prediction: {prediction_result}", 
                  fontsize=16, fontweight='bold', color=result_color)
    ax6.axis('off')
    
    # 置信度柱状图
    ax7 = fig.add_subplot(gs[2, :])
    positions = [f"Pos {i}\n'{prediction_result[i]}'" for i in range(len(prediction_result))]
    colors = ['green' if c > 0.9 else 'orange' if c > 0.7 else 'red' for c in confidences]
    bars = ax7.bar(positions, confidences, color=colors, alpha=0.7, edgecolor='black')
    ax7.set_ylabel('Confidence', fontsize=12)
    ax7.set_title('Per-Character Confidence', fontsize=12, fontweight='bold')
    ax7.set_ylim([0, 1])
    ax7.axhline(y=0.9, color='green', linestyle='--', linewidth=1, alpha=0.5, label='High (>0.9)')
    ax7.axhline(y=0.7, color='orange', linestyle='--', linewidth=1, alpha=0.5, label='Medium (>0.7)')
    ax7.axhline(y=0.5, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Low (<0.7)')
    ax7.legend(loc='upper right')
    ax7.grid(axis='y', alpha=0.3)
    
    # 在柱状图上添加数值标签
    for bar, conf in zip(bars, confidences):
        height = bar.get_height()
        ax7.text(bar.get_x() + bar.get_width()/2., height,
                f'{conf:.3f}\n({conf*100:.1f}%)',
                ha='center', va='bottom', fontsize=10)
    
    plt.suptitle(f'CAPTCHA Prediction Visualization', fontsize=16, fontweight='bold')
    
    # 保存图像
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Visualization saved to: {save_path}")
    
    # 显示图像
    plt.show()

def predict_captcha(img_path, model_path, show_visualization=True):
    """预测验证码"""
    # 加载模型
    try:
        model = load_model(model_path)
        print(f"✅ Model loaded from: {model_path}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print(f"Please ensure the model is saved at {model_path}")
        sys.exit(1)
    
    # 预处理图像
    print(f"\nPreprocessing image: {img_path}")
    image_np, images_dict = preprocess_image(img_path, show_steps=True)
    
    # 预测
    print(f"\nRunning prediction...")
    predictions = model.predict(image_np, verbose=0)
    
    # 解码
    print(f"\nDecoding prediction:")
    predicted_captcha, confidences = decode_prediction(predictions, verbose=True)
    
    # 可视化
    if show_visualization:
        visualize_preprocessing(images_dict, predicted_captcha, confidences)
    
    return predicted_captcha

if __name__ == "__main__":
    if len(sys.argv) >= 2:
        img_path = sys.argv[1]
        
        # 检查是否需要隐藏可视化（添加 --no-viz 参数）
        show_viz = "--no-viz" not in sys.argv
        
        # 预测
        result = predict_captcha(img_path, model_path, show_visualization=show_viz)
        
        print(f"\n{'='*60}")
        print(f"PREDICTED CAPTCHA: {result}")
        print(f"{'='*60}")
        
    else:
        print("Usage: python predict.py <image_path> [--no-viz]")
        print("\nExamples:")
        print("  python predict.py test_image.png")
        print("  python predict.py test_image.png --no-viz  # 不显示可视化")