# Copyright (C) 2025 zzsqjdhqgb
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

"""
三阶段渐进式训练主入口

训练策略：
    阶段1: 训练基线模型（CNN + BiLSTM）至 0.80 准确率
    阶段2: 冻结CNN，添加Self-Attention，训练Attention+LSTM
    阶段3: 解冻全部层，端到端微调

使用方法：
    1. 修改 config.py 中的参数
    2. 设置 SKIP_STAGE1/2/3 控制训练流程
    3. 运行: python train.py
"""

import sys
from config import Config
from data_loader import load_datasets
from trainer import StageTrainer
from utils import setup_gpu, print_stage_header


def print_configuration():
    """打印当前配置"""
    print("\n" + "="*60)
    print("TRAINING CONFIGURATION")
    print("="*60)
    print("\nStage Control:")
    print(f"  Stage 1: {'SKIP' if Config.SKIP_STAGE1 else 'RUN'} "
          f"(target acc: {Config.STAGE1_TARGET_ACC})")
    print(f"  Stage 2: {'SKIP' if Config.SKIP_STAGE2 else 'RUN'}")
    print(f"  Stage 3: {'SKIP' if Config.SKIP_STAGE3 else 'RUN'}")
    
    print("\nModel Hyperparameters:")
    print(f"  CNN Filters: {Config.CNN_FILTERS}")
    print(f"  LSTM Units: {Config.LSTM_UNITS}")
    print(f"  Attention Heads: {Config.ATTENTION_HEADS}")
    print(f"  Dropout Rate: {Config.DROPOUT_RATE}")
    
    print("\nTraining Parameters:")
    print(f"  Batch Size: {Config.BATCH_SIZE}")
    print(f"  Learning Rates: {Config.LR_STAGE1} / {Config.LR_STAGE2} / {Config.LR_STAGE3}")
    print(f"  Max Epochs: {Config.EPOCHS_STAGE1} / {Config.EPOCHS_STAGE2} / {Config.EPOCHS_STAGE3}")
    
    print("\nData:")
    print(f"  TFRecord Dir: {Config.TFRECORD_DIR}")
    
    print("\nOutput:")
    print(f"  Final Model: {Config.FINAL_MODEL_PATH}")
    print(f"  History: {Config.HISTORY_PATH}")
    print("="*60 + "\n")


def main():
    """主训练流程"""
    try:
        # 打印配置
        print_configuration()
        
        # 设置GPU
        setup_gpu()
        
        # 加载数据
        print("\nLoading datasets...")
        train_dataset, val_dataset = load_datasets()
        
        # 创建训练器
        trainer = StageTrainer(train_dataset, val_dataset)
        
        # ========== 阶段1 ==========
        model_stage1 = trainer.run_stage1()
        
        # ========== 阶段2 ==========
        model_stage2 = trainer.run_stage2(model_stage1)
        
        # ========== 阶段3 ==========
        final_model = trainer.run_stage3(model_stage2)
        
        # 保存历史
        trainer.save_training_history()
        
        # 打印总结
        trainer.print_summary()
        
        print("✓ Training completed successfully!\n")
        return final_model
        
    except KeyboardInterrupt:
        print("\n⚠ Training interrupted by user")
        sys.exit(1)
        
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    final_model = main()