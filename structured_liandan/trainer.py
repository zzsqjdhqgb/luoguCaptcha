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

import os
import json
import tensorflow as tf
from tensorflow import keras
from config import Config
from models import build_stage1_model, build_stage2_model
from utils import print_stage_header, analyze_improvement, print_model_info


class StageTrainer:
    """三阶段训练管理器"""
    
    def __init__(self, train_dataset, val_dataset):
        """
        初始化训练器
        
        Args:
            train_dataset: 训练数据集
            val_dataset: 验证数据集
        """
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.history_stage1 = None
        self.history_stage2 = None
        self.history_stage3 = None
        os.makedirs(Config.MODEL_DIR, exist_ok=True)
    
    def _load_existing_model(self, model_path, stage):
        """
        加载现有模型并评估
        
        Args:
            model_path: 模型路径
            stage: 阶段编号（1, 2, 3）
            
        Returns:
            model: 加载的Keras模型
        """
        print(f"⊙ Skipping Stage {stage} (loading from {model_path})")
        model = keras.models.load_model(model_path)
        
        print("Evaluating loaded model...")
        val_loss, val_acc = model.evaluate(self.val_dataset, verbose=0)
        print(f"  Validation accuracy: {val_acc:.4f}")
        print(f"  Validation loss: {val_loss:.4f}\n")
        
        # 创建假history用于后续分析
        fake_history = type('obj', (object,), {
            'history': {
                'val_accuracy': [val_acc],
                'loss': [val_loss],
                'accuracy': [val_acc],
                'val_loss': [val_loss]
            }
        })()
        
        if stage == 1:
            self.history_stage1 = fake_history
        elif stage == 2:
            self.history_stage2 = fake_history
        elif stage == 3:
            self.history_stage3 = fake_history
        
        return model
    
    def _get_stage1_callbacks(self):
        """获取阶段1的训练回调"""
        return [
            keras.callbacks.EarlyStopping(
                monitor="val_accuracy",
                patience=Config.PATIENCE_STAGE1,
                mode='max',
                restore_best_weights=True,
                verbose=1,
                baseline=Config.STAGE1_TARGET_ACC
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", 
                factor=0.5, 
                patience=8, 
                min_lr=1e-7, 
                verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                Config.STAGE1_MODEL_PATH, 
                monitor="val_accuracy", 
                save_best_only=True, 
                verbose=1
            ),
        ]
    
    def _get_stage2_callbacks(self):
        """获取阶段2的训练回调"""
        return [
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", 
                factor=0.5, 
                patience=Config.PATIENCE_STAGE2, 
                min_lr=1e-5, 
                verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                Config.STAGE2_MODEL_PATH, 
                monitor="val_accuracy", 
                save_best_only=True, 
                verbose=1
            ),
        ]
    
    def _get_stage3_callbacks(self):
        """获取阶段3的训练回调"""
        return [
            keras.callbacks.EarlyStopping(
                monitor="val_loss", 
                patience=Config.PATIENCE_STAGE3, 
                restore_best_weights=True, 
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", 
                factor=0.5, 
                patience=5, 
                min_lr=1e-7, 
                verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                Config.FINAL_MODEL_PATH, 
                monitor="val_accuracy", 
                save_best_only=True, 
                verbose=1
            ),
        ]
    
    def _print_stage_result(self, stage, history):
        """
        打印阶段训练结果
        
        Args:
            stage: 阶段编号
            history: 训练历史
        """
        best_acc = max(history.history['val_accuracy'])
        print(f"\n{'='*60}")
        print(f"Stage {stage} completed!")
        print(f"Best validation accuracy: {best_acc:.4f}")
        print(f"{'='*60}\n")
    
    def run_stage1(self):
        """
        阶段1：训练基线模型（CNN + BiLSTM）
        
        Returns:
            model: 训练好的模型
        """
        print_stage_header("STAGE 1: Training Plain CNN + BiLSTM")
        
        # 如果设置了跳过且模型存在
        if Config.SKIP_STAGE1 and os.path.exists(Config.STAGE1_MODEL_PATH):
            return self._load_existing_model(Config.STAGE1_MODEL_PATH, stage=1)
        
        # 构建并训练模型
        model = build_stage1_model()
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=Config.LR_STAGE1),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        
        print("Model Summary:")
        model.summary()
        print_model_info(model)
        
        print(f"\nTraining for up to {Config.EPOCHS_STAGE1} epochs...")
        print(f"Target: val_accuracy >= {Config.STAGE1_TARGET_ACC}\n")
        
        self.history_stage1 = model.fit(
            self.train_dataset,
            validation_data=self.val_dataset,
            epochs=Config.EPOCHS_STAGE1,
            callbacks=self._get_stage1_callbacks(),
        )
        
        model.save(Config.STAGE1_MODEL_PATH)
        self._print_stage_result(1, self.history_stage1)
        
        return model
    
    def run_stage2(self, baseline_model):
        """
        阶段2：添加Self-Attention，冻结CNN
        
        Args:
            baseline_model: 阶段1训练好的模型
            
        Returns:
            model: 训练好的模型
        """
        print_stage_header("STAGE 2: Adding Self-Attention, Freezing CNN")
        
        # 如果设置了跳过且模型存在
        if Config.SKIP_STAGE2 and os.path.exists(Config.STAGE2_MODEL_PATH):
            return self._load_existing_model(Config.STAGE2_MODEL_PATH, stage=2)
        
        # 构建模型
        model = build_stage2_model(baseline_model)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=Config.LR_STAGE2),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        
        print("Model Summary:")
        model.summary()
        print_model_info(model)
        
        print(f"\nTraining for {Config.EPOCHS_STAGE2} epochs...\n")
        
        self.history_stage2 = model.fit(
            self.train_dataset,
            validation_data=self.val_dataset,
            epochs=Config.EPOCHS_STAGE2,
            callbacks=self._get_stage2_callbacks(),
        )
        
        self._print_stage_result(2, self.history_stage2)
        
        return model
    
    def run_stage3(self, model):
        """
        阶段3：解冻全部，微调
        
        Args:
            model: 阶段2训练好的模型
            
        Returns:
            model: 训练好的模型
        """
        print_stage_header("STAGE 3: Unfreezing All Layers, Fine-tuning")
        
        # 如果设置了跳过
        if Config.SKIP_STAGE3:
            print("⊙ Skipping Stage 3\n")
            self.history_stage3 = type('obj', (object,), {
                'history': {
                    'val_accuracy': [0],
                    'loss': [0],
                    'accuracy': [0],
                    'val_loss': [0]
                }
            })()
            return model
        
        # 解冻所有层
        print("Unfreezing all layers...")
        unfrozen_count = 0
        for layer in model.layers:
            if not layer.trainable:
                layer.trainable = True
                unfrozen_count += 1
                print(f"  Unfrozen: {layer.name}")
        
        if unfrozen_count == 0:
            print("  (All layers already trainable)")
        print()
        
        # 重新编译（小学习率）
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=Config.LR_STAGE3),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        
        print_model_info(model)
        print(f"\nTraining for up to {Config.EPOCHS_STAGE3} epochs...\n")
        
        self.history_stage3 = model.fit(
            self.train_dataset,
            validation_data=self.val_dataset,
            epochs=Config.EPOCHS_STAGE3,
            callbacks=self._get_stage3_callbacks(),
        )
        
        model.save(Config.FINAL_MODEL_PATH)
        self._print_stage_result(3, self.history_stage3)
        
        return model
    
    def save_training_history(self):
        """保存所有阶段的训练历史到JSON文件"""
        history_all = {
            'stage1': {
                k: [float(v) for v in vals] 
                for k, vals in self.history_stage1.history.items()
            },
            'stage2': {
                k: [float(v) for v in vals] 
                for k, vals in self.history_stage2.history.items()
            },
            'stage3': {
                k: [float(v) for v in vals] 
                for k, vals in self.history_stage3.history.items()
            },
        }
        
        with open(Config.HISTORY_PATH, 'w') as f:
            json.dump(history_all, f, indent=2)
        
        print(f"✓ Training history saved to {Config.HISTORY_PATH}")
    
    def print_summary(self):
        """打印完整的训练总结和分析"""
        print("\n" + "="*60)
        print("TRAINING SUMMARY")
        print("="*60)
        
        # 获取各阶段最佳准确率
        stage1_best = max(self.history_stage1.history['val_accuracy'])
        stage2_best = max(self.history_stage2.history['val_accuracy'])
        stage3_best = max(self.history_stage3.history['val_accuracy']) if not Config.SKIP_STAGE3 else stage2_best
        
        print(f"\nStage 1 (CNN+BiLSTM):            {stage1_best:.4f}")
        print(f"Stage 2 (+Attention, CNN frozen): {stage2_best:.4f}")
        if not Config.SKIP_STAGE3:
            print(f"Stage 3 (Fine-tuning):            {stage3_best:.4f}")
        
        # ========== 改进分析 ==========
        print("\n" + "-"*60)
        print("IMPROVEMENT ANALYSIS")
        print("-"*60)
        
        improvement_1_2 = analyze_improvement(
            "Stage 1 → Stage 2", stage1_best, stage2_best
        )
        
        if not Config.SKIP_STAGE3:
            improvement_2_3 = analyze_improvement(
                "Stage 2 → Stage 3", stage2_best, stage3_best
            )
            
            improvement_total = analyze_improvement(
                "Overall (Stage 1 → 3)", stage1_best, stage3_best
            )
        
        # ========== 建议 ==========
        print("\n" + "="*60)
        print("RECOMMENDATIONS")
        print("="*60 + "\n")
        
        final_acc = stage3_best if not Config.SKIP_STAGE3 else stage2_best
        
        if final_acc > 0.92:
            print("✓✓ SUCCESS! Model exceeded 0.92 target")
            print(f"\nTo use the model:")
            print(f"  model = keras.models.load_model('{Config.FINAL_MODEL_PATH}')")
        
        elif improvement_1_2 < -0.02:
            print("⚠ Self-Attention hurt performance")
            print("\nSuggestions:")
            print("  1. Use Stage 1 model only (without Attention)")
            print("  2. Reduce attention heads in config.py:")
            print("     ATTENTION_HEADS = 2  # instead of 4")
            print("  3. Increase LSTM capacity before adding Attention:")
            print("     LSTM_UNITS = 256  # instead of 128")
        
        elif not Config.SKIP_STAGE3 and improvement_2_3 < -0.02:
            print("⚠ Fine-tuning hurt performance")
            print("\nSuggestions:")
            print("  1. Use Stage 2 model instead:")
            print(f"     model = keras.models.load_model('{Config.STAGE2_MODEL_PATH}')")
            print("  2. Lower learning rate in config.py:")
            print("     LR_STAGE3 = 0.00005  # instead of 0.0001")
            print("  3. Freeze more layers (edit trainer.py run_stage3)")
        
        else:
            print("→ Model trained successfully but has room for improvement")
            print("\nSuggestions:")
            print("  1. Train Stage 1 longer for higher baseline:")
            print("     EPOCHS_STAGE1 = 150  # instead of 100")
            print("  2. Increase model capacity in config.py:")
            print("     LSTM_UNITS = 256")
            print("     CNN_FILTERS = [96, 192]")
            print("  3. Add data augmentation (edit data_loader.py)")
        
        print("\n" + "="*60 + "\n")