# evaluate_detailed.py
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from datasets import load_dataset
from tqdm import tqdm
import json

# GPU配置
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"使用GPU: {gpus}")
    except Exception as e:
        print(f"GPU设置错误: {e}")
else:
    print("使用CPU")

# 配置参数
class Config:
    DATASET_PATH = "langningchen/luogu-captcha-dataset"
    MODEL_PATH = "models/luoguCaptcha_crnn.h5"  # 或 .keras
    IMG_HEIGHT = 35
    IMG_WIDTH = 90
    IMG_CHANNELS = 1
    CHARS_PER_LABEL = 4
    CHAR_SIZE = 256
    
    # 输出配置
    OUTPUT_DIR = "evaluation_results"
    SAVE_DETAILED_RESULTS = True
    SAVE_ERRORS_ONLY = True  # 只保存错误样本
    MAX_DISPLAY_SAMPLES = 50  # 终端显示的最大样本数

config = Config()

class DetailedEvaluator:
    def __init__(self, model_path, config):
        self.config = config
        self.model = self._load_model(model_path)
        self.results = []
        
    def _load_model(self, model_path):
        """加载模型"""
        print(f"加载模型: {model_path}")
        
        # 尝试不同的路径
        if not os.path.exists(model_path):
            # 尝试 .keras 后缀
            alternative_path = model_path.replace('.h5', '.keras')
            if os.path.exists(alternative_path):
                model_path = alternative_path
            else:
                raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        model = load_model(model_path)
        print(f"✅ 模型加载成功")
        print(f"输入形状: {model.input_shape}")
        print(f"输出数量: {len(model.outputs) if isinstance(model.outputs, list) else 1}")
        
        return model
    
    def _decode_prediction(self, predictions):
        """解码预测结果"""
        predicted_chars = []
        
        if isinstance(predictions, list):
            # 多输出格式
            for pred in predictions:
                char_idx = np.argmax(pred[0])
                predicted_chars.append(char_idx)
        else:
            # 单输出格式 (batch, seq_len, num_classes)
            predicted_chars = tf.math.argmax(predictions, axis=-1).numpy()[0]
        
        return list(predicted_chars)
    
    def _chars_to_string(self, char_list):
        """将字符索引列表转换为字符串"""
        return ''.join([chr(c) for c in char_list])
    
    def evaluate_dataset(self):
        """评估整个测试集"""
        print(f"\n加载数据集: {self.config.DATASET_PATH}")
        dataset_dict = load_dataset(self.config.DATASET_PATH)
        test_ds = dataset_dict["test"]
        
        print(f"测试集大小: {len(test_ds)}")
        print("\n开始评估...\n")
        
        # 统计变量
        correct_count = 0
        total_count = 0
        position_correct = [0] * self.config.CHARS_PER_LABEL
        position_total = [0] * self.config.CHARS_PER_LABEL
        error_samples = []
        
        # 逐个样本评估
        for idx, sample in enumerate(tqdm(test_ds, desc="评估进度")):
            # 预处理图像
            image = np.array(sample["image"], dtype=np.float32) / 255.0
            if len(image.shape) == 2:
                image = np.expand_dims(image, axis=-1)
            image = np.expand_dims(image, axis=0)  # 添加batch维度
            
            # 真实标签
            true_label_indices = sample["label"]
            true_label_str = self._chars_to_string(true_label_indices)
            
            # 预测
            predictions = self.model.predict(image, verbose=0)
            pred_label_indices = self._decode_prediction(predictions)
            pred_label_str = self._chars_to_string(pred_label_indices)
            
            # 判断是否完全正确
            is_correct = (pred_label_indices == true_label_indices)
            
            # 统计
            total_count += 1
            if is_correct:
                correct_count += 1
            
            # 按位置统计
            for pos in range(self.config.CHARS_PER_LABEL):
                position_total[pos] += 1
                if pred_label_indices[pos] == true_label_indices[pos]:
                    position_correct[pos] += 1
            
            # 记录结果
            result = {
                'index': idx,
                'true_label': true_label_str,
                'true_indices': true_label_indices,
                'pred_label': pred_label_str,
                'pred_indices': pred_label_indices,
                'is_correct': is_correct,
                'position_results': [
                    pred_label_indices[i] == true_label_indices[i] 
                    for i in range(self.config.CHARS_PER_LABEL)
                ]
            }
            
            self.results.append(result)
            
            # 保存错误样本
            if not is_correct:
                error_samples.append(result)
        
        # 计算准确率
        overall_accuracy = correct_count / total_count * 100 if total_count > 0 else 0
        position_accuracies = [
            position_correct[i] / position_total[i] * 100 if position_total[i] > 0 else 0
            for i in range(self.config.CHARS_PER_LABEL)
        ]
        
        # 输出结果
        self._print_results(
            total_count, correct_count, overall_accuracy,
            position_correct, position_total, position_accuracies,
            error_samples
        )
        
        # 保存结果到文件
        if self.config.SAVE_DETAILED_RESULTS:
            self._save_results(
                overall_accuracy, position_accuracies, error_samples
            )
        
        return overall_accuracy
    
    def _print_results(self, total_count, correct_count, overall_accuracy,
                      position_correct, position_total, position_accuracies,
                      error_samples):
        """打印评估结果"""
        print("\n" + "="*70)
        print("评估结果汇总".center(70))
        print("="*70)
        
        print(f"\n总样本数: {total_count}")
        print(f"正确数量: {correct_count}")
        print(f"错误数量: {total_count - correct_count}")
        print(f"\n{'整体准确率':<20}: {overall_accuracy:.2f}%")
        
        print(f"\n{'='*70}")
        print("各位置准确率:")
        print(f"{'='*70}")
        for i in range(self.config.CHARS_PER_LABEL):
            print(f"  位置 {i}: {position_accuracies[i]:6.2f}% "
                  f"({position_correct[i]}/{position_total[i]})")
        
        # 显示部分样本
        print(f"\n{'='*70}")
        print("样本详情 (显示前 {}/{} 个样本)".format(
            min(self.config.MAX_DISPLAY_SAMPLES, len(self.results)),
            len(self.results)
        ).center(70))
        print(f"{'='*70}")
        print(f"{'序号':<6} {'真实标签':<12} {'预测标签':<12} {'结果':<8} {'位置匹配'}")
        print(f"{'-'*70}")
        
        for i, result in enumerate(self.results[:self.config.MAX_DISPLAY_SAMPLES]):
            status = "✅ 正确" if result['is_correct'] else "❌ 错误"
            position_str = ''.join(['✓' if x else '✗' for x in result['position_results']])
            
            print(f"{result['index']:<6} "
                  f"{result['true_label']:<12} "
                  f"{result['pred_label']:<12} "
                  f"{status:<8} "
                  f"{position_str}")
        
        if len(self.results) > self.config.MAX_DISPLAY_SAMPLES:
            print(f"\n... 还有 {len(self.results) - self.config.MAX_DISPLAY_SAMPLES} 个样本未显示")
        
        # 显示错误样本
        if error_samples:
            print(f"\n{'='*70}")
            print(f"错误样本详情 (共 {len(error_samples)} 个)".center(70))
            print(f"{'='*70}")
            print(f"{'序号':<6} {'真实':<12} {'预测':<12} {'真实ASCII':<20} {'预测ASCII'}")
            print(f"{'-'*70}")
            
            for error in error_samples[:20]:  # 只显示前20个错误
                true_ascii = str(error['true_indices'])
                pred_ascii = str(error['pred_indices'])
                print(f"{error['index']:<6} "
                      f"{error['true_label']:<12} "
                      f"{error['pred_label']:<12} "
                      f"{true_ascii:<20} "
                      f"{pred_ascii}")
            
            if len(error_samples) > 20:
                print(f"\n... 还有 {len(error_samples) - 20} 个错误样本未显示")
        
        print(f"\n{'='*70}\n")
    
    def _save_results(self, overall_accuracy, position_accuracies, error_samples):
        """保存结果到文件"""
        os.makedirs(self.config.OUTPUT_DIR, exist_ok=True)
        
        # 保存汇总结果
        summary = {
            'overall_accuracy': overall_accuracy,
            'position_accuracies': position_accuracies,
            'total_samples': len(self.results),
            'correct_samples': sum(1 for r in self.results if r['is_correct']),
            'error_samples': len(error_samples)
        }
        
        summary_path = os.path.join(self.config.OUTPUT_DIR, 'summary.json')
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"✅ 汇总结果已保存: {summary_path}")
        
        # 保存详细结果
        if self.config.SAVE_ERRORS_ONLY:
            # 只保存错误样本
            details_to_save = error_samples
            filename = 'error_samples.json'
        else:
            # 保存所有样本
            details_to_save = self.results
            filename = 'all_samples.json'
        
        details_path = os.path.join(self.config.OUTPUT_DIR, filename)
        with open(details_path, 'w', encoding='utf-8') as f:
            json.dump(details_to_save, f, indent=2, ensure_ascii=False)
        print(f"✅ 详细结果已保存: {details_path} ({len(details_to_save)} 个样本)")
        
        # 生成可读的文本报告
        report_path = os.path.join(self.config.OUTPUT_DIR, 'report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("验证码识别模型评估报告\n")
            f.write("="*70 + "\n\n")
            
            f.write(f"模型路径: {self.config.MODEL_PATH}\n")
            f.write(f"数据集: {self.config.DATASET_PATH}\n")
            f.write(f"测试集大小: {len(self.results)}\n\n")
            
            f.write(f"整体准确率: {overall_accuracy:.2f}%\n\n")
            
            f.write("各位置准确率:\n")
            for i, acc in enumerate(position_accuracies):
                f.write(f"  位置 {i}: {acc:.2f}%\n")
            
            f.write(f"\n错误样本数: {len(error_samples)}\n")
            f.write(f"正确样本数: {len(self.results) - len(error_samples)}\n\n")
            
            if error_samples:
                f.write("="*70 + "\n")
                f.write("错误样本列表\n")
                f.write("="*70 + "\n")
                f.write(f"{'序号':<8} {'真实':<12} {'预测':<12} {'位置匹配'}\n")
                f.write("-"*70 + "\n")
                
                for error in error_samples:
                    position_str = ''.join(['✓' if x else '✗' for x in error['position_results']])
                    f.write(f"{error['index']:<8} "
                           f"{error['true_label']:<12} "
                           f"{error['pred_label']:<12} "
                           f"{position_str}\n")
        
        print(f"✅ 文本报告已保存: {report_path}")
        
        # 生成CSV格式（便于Excel打开）
        csv_path = os.path.join(self.config.OUTPUT_DIR, 'results.csv')
        with open(csv_path, 'w', encoding='utf-8-sig') as f:  # utf-8-sig for Excel
            f.write("序号,真实标签,预测标签,是否正确,位置0,位置1,位置2,位置3\n")
            for result in (error_samples if self.config.SAVE_ERRORS_ONLY else self.results):
                f.write(f"{result['index']},"
                       f"{result['true_label']},"
                       f"{result['pred_label']},"
                       f"{'正确' if result['is_correct'] else '错误'},")
                f.write(','.join(['✓' if x else '✗' for x in result['position_results']]))
                f.write('\n')
        
        print(f"✅ CSV结果已保存: {csv_path}")

def main():
    """主函数"""
    # 检查模型文件
    if not os.path.exists(config.MODEL_PATH):
        # 尝试查找可用的模型
        model_dir = os.path.dirname(config.MODEL_PATH)
        if os.path.exists(model_dir):
            available_models = [f for f in os.listdir(model_dir) 
                              if f.endswith(('.h5', '.keras'))]
            if available_models:
                print(f"⚠️  指定的模型不存在: {config.MODEL_PATH}")
                print(f"发现以下可用模型:")
                for i, model_file in enumerate(available_models):
                    print(f"  {i+1}. {model_file}")
                
                choice = input(f"\n请选择模型 (1-{len(available_models)}) 或按Enter使用第一个: ").strip()
                if choice.isdigit() and 1 <= int(choice) <= len(available_models):
                    selected_model = available_models[int(choice) - 1]
                else:
                    selected_model = available_models[0]
                
                config.MODEL_PATH = os.path.join(model_dir, selected_model)
                print(f"✅ 使用模型: {config.MODEL_PATH}\n")
            else:
                print(f"❌ 错误: 模型目录 {model_dir} 中没有找到模型文件")
                print("请先训练模型或将模型文件放入 models/ 目录")
                return
        else:
            print(f"❌ 错误: 模型文件不存在: {config.MODEL_PATH}")
            print(f"请确保模型文件存在，或先运行训练脚本")
            return
    
    # 创建评估器
    evaluator = DetailedEvaluator(config.MODEL_PATH, config)
    
    # 执行评估
    try:
        overall_accuracy = evaluator.evaluate_dataset()
        
        print("\n" + "="*70)
        print("评估完成！".center(70))
        print("="*70)
        print(f"\n最终准确率: {overall_accuracy:.2f}%\n")
        
        if config.SAVE_DETAILED_RESULTS:
            print(f"详细结果已保存到目录: {config.OUTPUT_DIR}/")
            print(f"  - summary.json       : 汇总统计")
            print(f"  - error_samples.json : 错误样本详情" if config.SAVE_ERRORS_ONLY else "  - all_samples.json   : 所有样本详情")
            print(f"  - report.txt         : 文本报告")
            print(f"  - results.csv        : CSV格式(可用Excel打开)")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  评估被用户中断")
    except Exception as e:
        print(f"\n❌ 评估过程中出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("\n" + "="*70)
    print("验证码识别模型详细评估工具".center(70))
    print("="*70 + "\n")
    
    main()