import sys
import tensorflow as tf
from tensorflow import keras

def show_model_summary(model_path):
    """
    加载并显示Keras模型的摘要信息
    
    Args:
        model_path: 模型文件路径 (.keras, .h5等)
    """
    try:
        print(f"Loading model from: {model_path}")
        print("=" * 80)
        
        # 加载模型
        model = keras.models.load_model(model_path)
        
        print(f"\nModel Name: {model.name}")
        print("=" * 80)
        
        # 显示模型摘要
        model.summary()
        
        print("\n" + "=" * 80)
        print("Additional Information:")
        print("=" * 80)
        
        # 显示输入输出形状
        print(f"\nInput shape(s):")
        if isinstance(model.input, list):
            for i, inp in enumerate(model.input):
                print(f"  Input {i}: {inp.shape}")
        else:
            print(f"  {model.input.shape}")
        
        print(f"\nOutput shape(s):")
        if isinstance(model.output, list):
            for i, out in enumerate(model.output):
                print(f"  Output {i}: {out.shape}")
        else:
            print(f"  {model.output.shape}")
        
        # 显示总参数量
        total_params = model.count_params()
        print(f"\nTotal parameters: {total_params:,}")
        
        # 显示编译信息（如果已编译）
        if model.optimizer is not None:
            print(f"\nOptimizer: {model.optimizer.__class__.__name__}")
            print(f"Learning rate: {float(model.optimizer.learning_rate)}")
        
        if hasattr(model, 'loss') and model.loss is not None:
            print(f"Loss function: {model.loss}")
        
        if hasattr(model, 'metrics') and model.metrics:
            print(f"Metrics: {[m.name for m in model.metrics]}")
        
        print("\n" + "=" * 80)
        print("Model loaded successfully!")
        
    except Exception as e:
        print(f"\nError loading model: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python show_model_summary.py <model_path>")
        print("\nExample:")
        print("  python show_model_summary.py models/luoguCaptcha.CRNN.resnet.keras")
        sys.exit(1)
    
    model_path = sys.argv[1]
    show_model_summary(model_path)