# evaluate_detailed_parallel.py
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from datasets import load_dataset
from tqdm import tqdm
import json
from concurrent.futures import ThreadPoolExecutor
import time

# GPU配置 - 优化显存和性能
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            # 启用混合精度（可选，进一步加速）
            # tf.config.optimizer.set_experimental_options({"auto_mixed_precision": True})
        print(f"使用GPU: {gpus}")
    except Exception as e:
        print(f"GPU设置错误: {e}")
else:
    print("使用CPU")

# 优化TensorFlow性能
tf.config.threading.set_intra_op_parallelism_threads(8)
tf.config.threading.set_inter_op_parallelism_threads(8)

# 配置参数
class Config:
    DATASET_PATH = "langningchen/luogu-captcha-dataset"
    MODEL_PATH = "models/luoguCaptcha_crnn.h5"
    IMG_HEIGHT = 35
    IMG_WIDTH = 90
    IMG_CHANNELS = 1
    CHARS_PER_LABEL = 4
    CHAR_SIZE = 256
    
    # 并行优化参数
    BATCH_SIZE = 256  # 增大批次，充分利用GPU
    PREFETCH_SIZE = 4  # 预取批次数
    NUM_WORKERS = 8    # 数据加载线程数
    
    # 输出配置
    OUTPUT_DIR = "evaluation_results"
    SAVE_DETAILED_RESULTS = True
    SAVE_ERRORS_ONLY = True
    MAX_DISPLAY_SAMPLES = 50

config = Config()

class ParallelEvaluator:
    def __init__(self, model_path, config):
        self.config = config
        self.model = self._load_model(model_path)
        self.results = []
        
    def _load_model(self, model_path):
        """加载模型"""
        print(f"加载模型: {model_path}")
        
        if not os.path.exists(model_path):
            alternative_path = model_path.replace('.h5', '.keras')
            if os.path.exists(alternative_path):
                model_path = alternative_path
            else:
                raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        model = load_model(model_path)
        print(f"✅ 模型加载成功")
        
        # 预热模型（JIT编译）
        print("预热模型...")
        dummy_input = np.random.random((1, self.config.IMG_HEIGHT, 
                                       self.config.IMG_WIDTH, 
                                       self.config.IMG_CHANNELS)).astype(np.float32)
        _ = model.predict(dummy_input, verbose=0)
        print("✅ 模型预热完成")
        
        return model
    
    def _prepare_batch_data(self, samples):
        """批量准备数据（并行处理）"""
        images = []
        labels = []
        
        for sample in samples:
            image = np.array(sample["image"], dtype=np.float32) / 255.0
            if len(image.shape) == 2:
                image = np.expand_dims(image, axis=-1)
            images.append(image)
            labels.append(sample["label"])
        
        return np.array(images), labels
    
    def _decode_batch_predictions(self, predictions, batch_size):
        """批量解码预测结果"""
        if isinstance(predictions, list):
            # 多输出格式: [array(batch, 256), array(batch, 256), ...]
            pred_labels = []
            for i in range(batch_size):
                pred_chars = [np.argmax(predictions[j][i]) 
                             for j in range(self.config.CHARS_PER_LABEL)]
                pred_labels.append(pred_chars)
            return pred_labels
        else:
            # 单输出格式: (batch, 4, 256)
            return [list(tf.math.argmax(pred, axis=-1).numpy()) 
                   for pred in predictions]
    
    def _chars_to_string(self, char_list):
        """将字符索引列表转换为字符串"""
        return ''.join([chr(c) for c in char_list])
    
    def evaluate_dataset(self):
        """并行批量评估整个测试集"""
        print(f"\n加载数据集: {self.config.DATASET_PATH}")
        dataset_dict = load_dataset(self.config.DATASET_PATH)
        test_ds = dataset_dict["test"]
        
        total_samples = len(test_ds)
        print(f"测试集大小: {total_samples}")
        print(f"批次大小: {self.config.BATCH_SIZE}")
        print(f"总批次数: {(total_samples + self.config.BATCH_SIZE - 1) // self.config.BATCH_SIZE}")
        print("\n开始批量评估...\n")
        
        # 统计变量
        correct_count = 0
        total_count = 0
        position_correct = [0] * self.config.CHARS_PER_LABEL
        position_total = [0] * self.config.CHARS_PER_LABEL
        error_samples = []
        
        # 记录时间
        start_time = time.time()
        
        # 批量处理
        num_batches = (total_samples + self.config.BATCH_SIZE - 1) // self.config.BATCH_SIZE
        
        with tqdm(total=total_samples, desc="评估进度", unit="样本") as pbar:
            for batch_idx in range(num_batches):
                # 获取当前批次的样本
                start_idx = batch_idx * self.config.BATCH_SIZE
                end_idx = min(start_idx + self.config.BATCH_SIZE, total_samples)
                batch_samples = [test_ds[i] for i in range(start_idx, end_idx)]
                current_batch_size = len(batch_samples)
                
                # 批量准备数据
                batch_images, batch_true_labels = self._prepare_batch_data(batch_samples)
                
                # 批量预测（这里是GPU加速的关键）
                batch_predictions = self.model.predict(
                    batch_images, 
                    batch_size=self.config.BATCH_SIZE,
                    verbose=0
                )
                
                # 批量解码预测
                batch_pred_labels = self._decode_batch_predictions(
                    batch_predictions, 
                    current_batch_size
                )
                
                # 处理批次结果
                for i in range(current_batch_size):
                    sample_idx = start_idx + i
                    true_label = batch_true_labels[i]
                    pred_label = batch_pred_labels[i]
                    
                    true_label_str = self._chars_to_string(true_label)
                    pred_label_str = self._chars_to_string(pred_label)
                    
                    is_correct = (pred_label == true_label)
                    
                    # 统计
                    total_count += 1
                    if is_correct:
                        correct_count += 1
                    
                    # 按位置统计
                    position_results = []
                    for pos in range(self.config.CHARS_PER_LABEL):
                        position_total[pos] += 1
                        pos_correct = (pred_label[pos] == true_label[pos])
                        if pos_correct:
                            position_correct[pos] += 1
                        position_results.append(pos_correct)
                    
                    # 记录结果
                    result = {
                        'index': sample_idx,
                        'true_label': true_label_str,
                        'true_indices': true_label,
                        'pred_label': pred_label_str,
                        'pred_indices': pred_label,
                        'is_correct': is_correct,
                        'position_results': position_results
                    }
                    
                    self.results.append(result)
                    
                    if not is_correct:
                        error_samples.append(result)
                
                # 更新进度条
                pbar.update(current_batch_size)
                
                # 实时显示当前准确率
                current_acc = (correct_count / total_count * 100) if total_count > 0 else 0
                pbar.set_postfix({
                    'Acc': f'{current_acc:.2f}%',
                    'Err': len(error_samples)
                })
        
        # 计算总耗时和速度
        end_time = time.time()
        total_time = end_time - start_time
        samples_per_second = total_samples / total_time
        
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
            error_samples, total_time, samples_per_second
        )
        
        # 保存结果
        if self.config.SAVE_DETAILED_RESULTS:
            self._save_results(
                overall_accuracy, position_accuracies, error_samples,
                total_time, samples_per_second
            )
        
        return overall_accuracy
    
    def _print_results(self, total_count, correct_count, overall_accuracy,
                      position_correct, position_total, position_accuracies,
                      error_samples, total_time, samples_per_second):
        """打印评估结果"""
        print("\n" + "="*70)
        print("评估结果汇总".center(70))
        print("="*70)
        
        print(f"\n{'性能统计':-^70}")
        print(f"总耗时: {total_time:.2f} 秒")
        print(f"处理速度: {samples_per_second:.2f} 样本/秒")
        print(f"平均延迟: {1000/samples_per_second:.2f} 毫秒/样本")
        
        print(f"\n{'准确率统计':-^70}")
        print(f"总样本数: {total_count}")
        print(f"正确数量: {correct_count}")
        print(f"错误数量: {total_count - correct_count}")
        print(f"\n整体准确率: {overall_accuracy:.2f}%")
        
        print(f"\n{'各位置准确率':-^70}")
        for i in range(self.config.CHARS_PER_LABEL):
            print(f"  位置 {i}: {position_accuracies[i]:6.2f}% "
                  f"({position_correct[i]}/{position_total[i]})")
        
        # 显示部分样本
        print(f"\n{'='*70}")
        print("样本详情 (显示前 {}/{} 个)".format(
            min(self.config.MAX_DISPLAY_SAMPLES, len(self.results)),
            len(self.results)
        ).center(70))
        print(f"{'='*70}")
        print(f"{'序号':<6} {'真实标签':<12} {'预测标签':<12} {'结果':<8} {'位置'}")
        print(f"{'-'*70}")
        
        for result in self.results[:self.config.MAX_DISPLAY_SAMPLES]:
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
            print(f"{'序号':<6} {'真实':<12} {'预测':<12} {'真实ASCII':<25} {'预测ASCII'}")
            print(f"{'-'*70}")
            
            for error in error_samples[:20]:
                true_ascii = str(error['true_indices'])
                pred_ascii = str(error['pred_indices'])
                print(f"{error['index']:<6} "
                      f"{error['true_label']:<12} "
                      f"{error['pred_label']:<12} "
                      f"{true_ascii:<25} "
                      f"{pred_ascii}")
            
            if len(error_samples) > 20:
                print(f"\n... 还有 {len(error_samples) - 20} 个错误样本未显示")
        
        print(f"\n{'='*70}\n")
    
    def _save_results(self, overall_accuracy, position_accuracies, 
                     error_samples, total_time, samples_per_second):
        """保存结果到文件"""
        os.makedirs(self.config.OUTPUT_DIR, exist_ok=True)
        
        # 保存汇总结果（包含性能信息）
        summary = {
            'overall_accuracy': overall_accuracy,
            'position_accuracies': position_accuracies,
            'total_samples': len(self.results),
            'correct_samples': sum(1 for r in self.results if r['is_correct']),
            'error_samples': len(error_samples),
            'performance': {
                'total_time_seconds': total_time,
                'samples_per_second': samples_per_second,
                'average_latency_ms': 1000 / samples_per_second,
                'batch_size': self.config.BATCH_SIZE
            }
        }
        
        summary_path = os.path.join(self.config.OUTPUT_DIR, 'summary.json')
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"✅ 汇总结果已保存: {summary_path}")
        
        # 保存详细结果
        if self.config.SAVE_ERRORS_ONLY:
            details_to_save = error_samples
            filename = 'error_samples.json'
        else:
            details_to_save = self.results
            filename = 'all_samples.json'
        
        details_path = os.path.join(self.config.OUTPUT_DIR, filename)
        with open(details_path, 'w', encoding='utf-8') as f:
            json.dump(details_to_save, f, indent=2, ensure_ascii=False)
        print(f"✅ 详细结果已保存: {details_path} ({len(details_to_save)} 个样本)")
        
        # 生成文本报告
        report_path = os.path.join(self.config.OUTPUT_DIR, 'report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("验证码识别模型评估报告\n")
            f.write("="*70 + "\n\n")
            
            f.write(f"模型路径: {self.config.MODEL_PATH}\n")
            f.write(f"数据集: {self.config.DATASET_PATH}\n")
            f.write(f"测试集大小: {len(self.results)}\n")
            f.write(f"批次大小: {self.config.BATCH_SIZE}\n\n")
            
            f.write("性能统计:\n")
            f.write(f"  总耗时: {total_time:.2f} 秒\n")
            f.write(f"  处理速度: {samples_per_second:.2f} 样本/秒\n")
            f.write(f"  平均延迟: {1000/samples_per_second:.2f} 毫秒/样本\n\n")
            
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
        
        # 生成CSV格式
        csv_path = os.path.join(self.config.OUTPUT_DIR, 'results.csv')
        with open(csv_path, 'w', encoding='utf-8-sig') as f:
            f.write("序号,真实标签,预测标签,是否正确,位置0,位置1,位置2,位置3\n")
            for result in (error_samples if self.config.SAVE_ERRORS_ONLY else self.results):
                f.write(f"{result['index']},"
                       f"{result['true_label']},"
                       f"{result['pred_label']},"
                       f"{'正确' if result['is_correct'] else '错误'},")
                f.write(','.join(['✓' if x else '✗' for x in result['position_results']]))
                f.write('\n')
        
        print(f"✅ CSV结果已保存: {csv_path}")
    
    def analyze_confusion(self):
        """分析常见的混淆字符对"""
        confusion_pairs = {}
        char_errors = {}  # 统计每个字符的错误次数
        
        for result in self.results:
            if not result['is_correct']:
                for pos in range(self.config.CHARS_PER_LABEL):
                    true_char = chr(result['true_indices'][pos])
                    pred_char = chr(result['pred_indices'][pos])
                    
                    if true_char != pred_char:
                        # 记录混淆对
                        pair = f"{true_char}→{pred_char}"
                        confusion_pairs[pair] = confusion_pairs.get(pair, 0) + 1
                        
                        # 记录每个字符的错误
                        char_errors[true_char] = char_errors.get(true_char, 0) + 1
        
        # 排序并输出
        sorted_pairs = sorted(confusion_pairs.items(), key=lambda x: x[1], reverse=True)
        sorted_chars = sorted(char_errors.items(), key=lambda x: x[1], reverse=True)
        
        print("\n" + "="*70)
        print("字符混淆分析".center(70))
        print("="*70)
        
        print(f"\n{'最常见的字符混淆 (Top 15)':-^70}")
        print(f"{'混淆对':<15} {'次数':<10} {'百分比'}")
        print("-"*70)
        total_errors = sum(confusion_pairs.values())
        for pair, count in sorted_pairs[:15]:
            percentage = count / total_errors * 100 if total_errors > 0 else 0
            print(f"{pair:<15} {count:<10} {percentage:.2f}%")
        
        print(f"\n{'最容易识别错误的字符 (Top 10)':-^70}")
        print(f"{'字符':<10} {'错误次数':<15} {'ASCII'}")
        print("-"*70)
        for char, count in sorted_chars[:10]:
            print(f"{char:<10} {count:<15} {ord(char)}")
        
        print("="*70 + "\n")
        
        # 保存混淆分析
        confusion_path = os.path.join(self.config.OUTPUT_DIR, 'confusion_analysis.json')
        confusion_data = {
            'confusion_pairs': sorted_pairs,
            'char_errors': sorted_chars,
            'total_errors': total_errors
        }
        with open(confusion_path, 'w', encoding='utf-8') as f:
            json.dump(confusion_data, f, indent=2, ensure_ascii=False)
        print(f"✅ 混淆分析已保存: {confusion_path}")
        
        return confusion_pairs, char_errors

def main():
    """主函数"""
    print("\n" + "="*70)
    print("验证码识别模型并行评估工具 (GPU加速版)".center(70))
    print("="*70 + "\n")
    
    # 检查模型文件
    if not os.path.exists(config.MODEL_PATH):
        # 尝试查找可用的模型
        model_dir = os.path.dirname(config.MODEL_PATH)
        if os.path.exists(model_dir):
            available_models = [f for f in os.listdir(model_dir) 
                              if f.endswith(('.h5', '.keras'))]
            if available_models:
                print(f"⚠️  指定的模型不存在: {config.MODEL_PATH}")
                print(f"\n发现以下可用模型:")
                for i, model_file in enumerate(available_models, 1):
                    model_path = os.path.join(model_dir, model_file)
                    size_mb = os.path.getsize(model_path) / (1024 * 1024)
                    print(f"  {i}. {model_file:<30} ({size_mb:.2f} MB)")
                
                choice = input(f"\n请选择模型 (1-{len(available_models)}) 或按Enter使用第一个: ").strip()
                if choice.isdigit() and 1 <= int(choice) <= len(available_models):
                    selected_model = available_models[int(choice) - 1]
                else:
                    selected_model = available_models[0]
                
                config.MODEL_PATH = os.path.join(model_dir, selected_model)
                print(f"\n✅ 使用模型: {config.MODEL_PATH}\n")
            else:
                print(f"❌ 错误: 模型目录 {model_dir} 中没有找到模型文件")
                print("请先训练模型或将模型文件放入 models/ 目录")
                return
        else:
            print(f"❌ 错误: 模型文件不存在: {config.MODEL_PATH}")
            print(f"请确保模型文件存在，或先运行训练脚本")
            return
    
    # 显示配置信息
    print("="*70)
    print("配置信息".center(70))
    print("="*70)
    print(f"模型路径    : {config.MODEL_PATH}")
    print(f"数据集      : {config.DATASET_PATH}")
    print(f"批次大小    : {config.BATCH_SIZE}")
    print(f"预取批次    : {config.PREFETCH_SIZE}")
    print(f"工作线程    : {config.NUM_WORKERS}")
    print(f"结果目录    : {config.OUTPUT_DIR}")
    print("="*70 + "\n")
    
    # 创建评估器
    try:
        evaluator = ParallelEvaluator(config.MODEL_PATH, config)
    except Exception as e:
        print(f"❌ 创建评估器失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 执行评估
    try:
        print("开始评估...\n")
        overall_accuracy = evaluator.evaluate_dataset()
        
        # 混淆分析
        print("\n执行混淆分析...")
        evaluator.analyze_confusion()
        
        # 最终总结
        print("\n" + "="*70)
        print("评估完成！".center(70))
        print("="*70)
        print(f"\n最终准确率: {overall_accuracy:.2f}%")
        
        if config.SAVE_DETAILED_RESULTS:
            print(f"\n详细结果已保存到目录: {config.OUTPUT_DIR}/")
            print(f"  📊 summary.json           : 汇总统计 + 性能指标")
            print(f"  📝 {'error_samples.json' if config.SAVE_ERRORS_ONLY else 'all_samples.json':<20} : {'错误' if config.SAVE_ERRORS_ONLY else '所有'}样本详情")
            print(f"  📄 report.txt             : 文本报告")
            print(f"  📈 results.csv            : CSV格式(Excel可打开)")
            print(f"  🔍 confusion_analysis.json: 字符混淆分析")
        
        print("\n" + "="*70 + "\n")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  评估被用户中断")
    except Exception as e:
        print(f"\n❌ 评估过程中出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()