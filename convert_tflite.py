"""
DS-CNN Frozen Graph -> TFLite 변환 스크립트
fingerprint_input 기반으로 별도의 frozen graph를 생성한 후 TFLite로 변환합니다.
"""
import tensorflow as tf
import sys
import os

sys.path.insert(0, '.')
import input_data
import models
from tensorflow.python.framework import graph_util

# === 설정 (학습 시 사용한 값과 동일하게) ===
WANTED_WORDS = 'yes,no'
SAMPLE_RATE = 16000
CLIP_DURATION_MS = 1000
WINDOW_SIZE_MS = 40
WINDOW_STRIDE_MS = 20
DCT_COEFFICIENT_COUNT = 10
MODEL_SIZE_INFO = [5, 64, 10, 4, 2, 2, 64, 3, 3, 1, 1, 64, 3, 3, 1, 1, 64, 3, 3, 1, 1, 64, 3, 3, 1, 1]
CHECKPOINT_PATH = './work/ds_cnn_train/best/ds_cnn_9687.ckpt-7500'
OUTPUT_PB = './work/ds_cnn_fingerprint_frozen.pb'
OUTPUT_TFLITE = './work/ds_cnn.tflite'

# === 1. 모델 설정 계산 ===
words_list = input_data.prepare_words_list(WANTED_WORDS.split(','))
model_settings = models.prepare_model_settings(
    len(words_list), SAMPLE_RATE, CLIP_DURATION_MS,
    WINDOW_SIZE_MS, WINDOW_STRIDE_MS, DCT_COEFFICIENT_COUNT, 100)

fingerprint_size = model_settings['fingerprint_size']
print(f'[INFO] fingerprint_size: {fingerprint_size}')
print(f'[INFO] label_count: {model_settings["label_count"]}')

# === 2. fingerprint 입력 기반 추론 그래프 생성 ===
tf.reset_default_graph()
fingerprint_input = tf.placeholder(
    tf.float32, [1, fingerprint_size], name='fingerprint_input')

logits = models.create_ds_cnn_model(
    fingerprint_input, model_settings, MODEL_SIZE_INFO, is_training=False)
output = tf.nn.softmax(logits, name='labels_softmax')

# === 3. 체크포인트 로드 ===
sess = tf.Session()
saver = tf.train.Saver()
saver.restore(sess, CHECKPOINT_PATH)
print('[INFO] 체크포인트 로드 완료')

# === 4. Frozen Graph 생성 ===
frozen_graph_def = graph_util.convert_variables_to_constants(
    sess, sess.graph_def, ['labels_softmax'])
tf.train.write_graph(
    frozen_graph_def,
    os.path.dirname(OUTPUT_PB),
    os.path.basename(OUTPUT_PB),
    as_text=False)
print(f'[INFO] Frozen Graph 저장: {OUTPUT_PB}')

# === 5. TFLite 변환 ===
converter = tf.lite.TFLiteConverter.from_frozen_graph(
    OUTPUT_PB,
    input_arrays=['fingerprint_input'],
    output_arrays=['labels_softmax'],
    input_shapes={'fingerprint_input': [1, fingerprint_size]}
)
tflite_model = converter.convert()

with open(OUTPUT_TFLITE, 'wb') as f:
    f.write(tflite_model)

size_kb = len(tflite_model) / 1024
print(f'[SUCCESS] TFLite 변환 완료!')
print(f'[SUCCESS] 파일: {OUTPUT_TFLITE}')
print(f'[SUCCESS] 크기: {len(tflite_model)} bytes ({size_kb:.1f} KB)')

sess.close()
