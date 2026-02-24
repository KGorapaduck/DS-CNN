import tensorflow as tf
from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio
import os

# 1. MFCC 추출용 세션 생성
sess = tf.Session()
wav_placeholder = tf.placeholder(tf.float32, [16000, 1], name='wav_input')

spectrogram = contrib_audio.audio_spectrogram(
    wav_placeholder,
    window_size=int(16000 * 0.04),
    stride=int(16000 * 0.02),
    magnitude_squared=True)

mfcc_op = contrib_audio.mfcc(
    spectrogram,
    16000,
    dct_coefficient_count=10)

mfcc_flatten = tf.reshape(mfcc_op, [1, -1], name='mfcc_output')

# 2. TFLiteConverter로 변환 (Frozen Graph 없이 Session에서 바로 변환)
converter = tf.lite.TFLiteConverter.from_session(sess, [wav_placeholder], [mfcc_flatten])
tflite_model = converter.convert()

# 3. 모델 저장
os.makedirs('work', exist_ok=True)
output_path = 'work/mfcc_engine.tflite'
with open(output_path, 'wb') as f:
    f.write(tflite_model)

print(f"✅ MFCC 추출 엔진 TFLite 모델 생성 완료: {output_path}")
