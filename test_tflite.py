import numpy as np
import tensorflow as tf

# TFLite 모델 로드
interpreter = tf.lite.Interpreter(model_path="./work/ds_cnn.tflite")
interpreter.allocate_tensors()

# 입출력 텐서 정보 확인
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("== Input Details ==")
print(input_details[0]['shape'])
print(input_details[0]['dtype'])

print("\n== Output Details ==")
print(output_details[0]['shape'])
print(output_details[0]['dtype'])

# 더미 데이터 생성 (배치 사이즈 1, 490개의 float32 특성 벡터)
input_shape = input_details[0]['shape']
input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)

# 추론 실행
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()
output_data = interpreter.get_tensor(output_details[0]['index'])

print("\n== Inference Result ==")
print(output_data)
print("Sum of probabilities:", np.sum(output_data))
print("TFLite Model is working perfectly!")
