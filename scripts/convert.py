# 从 .h5 迁移到 .keras

import keras

# 步骤1: 加载旧模型
old_model = keras.models.load_model('models/luoguCaptcha_crnn.h5')

# 步骤2: 保存为新格式
old_model.save('models/luoguCaptcha_crnn.keras')

# 步骤3: 验证
new_model = keras.models.load_model('models/luoguCaptcha_crnn.keras')

# 步骤4: 测试预测是否一致
import numpy as np
test_input = np.random.random((1, 35, 90, 1))
assert np.allclose(
    old_model.predict(test_input),
    new_model.predict(test_input)
), "模型预测结果不一致！"

print("✅ 迁移成功！")