# rebuild_model_without_optimizer.py
import sys
import tensorflow as tf
import os

def rebuild_model_clean(input_path, output_path):
    """通过重建模型来彻底移除优化器"""
    print(f"Loading model from: {input_path}")
    model = tf.keras.models.load_model(input_path, compile=False)  # 关键：不编译
    
    original_size = os.path.getsize(input_path) / (1024 * 1024)
    print(f"Original size: {original_size:.2f} MB")
    
    # 获取模型架构和权重
    print("\nExtracting model architecture and weights...")
    config = model.get_config()
    weights = model.get_weights()
    
    # 重建模型（全新的，没有任何训练状态）
    print("Rebuilding model...")
    clean_model = tf.keras.Model.from_config(config)
    clean_model.set_weights(weights)
    
    # 保存干净的模型
    print(f"Saving clean model to: {output_path}")
    clean_model.save(output_path, save_format='keras')
    
    new_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\nNew size: {new_size:.2f} MB")
    print(f"Saved: {original_size - new_size:.2f} MB ({(1-new_size/original_size)*100:.1f}%)")
    
    # 验证
    print("\nVerifying clean model...")
    reloaded = tf.keras.models.load_model(output_path)
    
    # 安全检查 optimizer
    has_optimizer = hasattr(reloaded, 'optimizer') and reloaded.optimizer is not None
    print(f"Has optimizer: {has_optimizer}")
    
    if has_optimizer:
        print("⚠️ Still has optimizer (this shouldn't happen)")
    else:
        print("✓ Optimizer successfully removed!")
    
    # 测试模型可用性
    print("\nTesting model inference...")
    print(f"Input shape: {clean_model.input.shape}")
    print(f"Output shape: {clean_model.output.shape}")
    
    import numpy as np
    test_input = np.random.random((1, 35, 90, 1)).astype(np.float32)
    
    # 测试原模型和新模型输出是否一致
    print("\nComparing outputs...")
    original_output = model.predict(test_input, verbose=0)
    new_output = clean_model.predict(test_input, verbose=0)
    reloaded_output = reloaded.predict(test_input, verbose=0)
    
    diff1 = np.max(np.abs(original_output - new_output))
    diff2 = np.max(np.abs(original_output - reloaded_output))
    
    print(f"Original vs New model diff: {diff1:.10f}")
    print(f"Original vs Reloaded diff: {diff2:.10f}")
    
    if diff1 < 1e-5 and diff2 < 1e-5:
        print("✓ All models produce identical outputs!")
    else:
        print("⚠️ Warning: Outputs differ slightly (but this is usually okay)")
    
    print(f"\n{'='*80}")
    print("Summary:")
    print(f"{'='*80}")
    print(f"✓ Model successfully cleaned and saved to: {output_path}")
    print(f"✓ File size reduced from {original_size:.2f} MB to {new_size:.2f} MB")
    print(f"✓ Reduction: {original_size - new_size:.2f} MB ({(1-new_size/original_size)*100:.1f}%)")
    
    return clean_model

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python rebuild_model_without_optimizer.py <input.keras> <output.keras>")
        print("\nExample:")
        print("  python rebuild_model_without_optimizer.py models/model.keras models/model_clean.keras")
        sys.exit(1)
    
    rebuild_model_clean(sys.argv[1], sys.argv[2])