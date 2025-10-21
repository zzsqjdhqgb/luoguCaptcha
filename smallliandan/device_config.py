import tensorflow as tf


def setup_device():
    """自动配置GPU/CPU设备"""
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Using GPU: {gpus}")
        except Exception as e:
            print(f"GPU setup error: {e}")
    else:
        print("Using CPU")