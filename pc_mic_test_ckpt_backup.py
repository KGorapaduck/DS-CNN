import pyaudio
import numpy as np
import tensorflow as tf
from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio
import collections
import os
import sys

# --- 1. 파라미터 설정 ---
SAMPLE_RATE = 16000
CHUNK_DURATION_MS = 250
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION_MS / 1000)
WINDOW_SIZE_MS = 40.0
WINDOW_STRIDE_MS = 20.0
DCT_COEFFICIENT_COUNT = 10
CLIP_DURATION_SAMPLES = SAMPLE_RATE

LABELS = ['_silence_', '_unknown_', 'quiz', 'understand']
CHECKPOINT_DIR = "./work/ds_cnn_korean/best"

# --- 2. TF 세션 및 그래프 구축 ---
tf.reset_default_graph()
sess = tf.InteractiveSession()

# 입력 플레이스홀더
wav_placeholder = tf.placeholder(tf.float32, [CLIP_DURATION_SAMPLES, 1])

# MFCC 전처리 그래프
spectrogram = contrib_audio.audio_spectrogram(
    wav_placeholder,
    window_size=int(SAMPLE_RATE * WINDOW_SIZE_MS / 1000),
    stride=int(SAMPLE_RATE * WINDOW_STRIDE_MS / 1000),
    magnitude_squared=True)
mfcc_op = contrib_audio.mfcc(
    spectrogram,
    SAMPLE_RATE,
    dct_coefficient_count=DCT_COEFFICIENT_COUNT)
mfcc_flatten = tf.reshape(mfcc_op, [1, -1]) # [1, 490]

# --- 3. DS-CNN 아키텍처 재구축 (체크포인트 파라미터 로딩용) ---
# NOTE: To load a .ckpt without freezing, we must recreate the exact same network graph.
# Since importing models.py might be tricky dynamically, we can use the original project's models module.
try:
    import models
except ImportError:
    print("models.py not found in current directory. Please run this script from KWS-DS-CNN-for-embedded root.")
    sys.exit(1)

model_settings = models.prepare_model_settings(
      len(LABELS), SAMPLE_RATE, 1000, WINDOW_SIZE_MS,
      WINDOW_STRIDE_MS, DCT_COEFFICIENT_COUNT, 32)

fingerprint_input = tf.placeholder(
      tf.float32, [None, model_settings['fingerprint_size']], name='fingerprint_input')

# '5 64 10 4 2 2 64 3 3 1 1 64 3 3 1 1 64 3 3 1 1 64 3 3 1 1'
model_size_info = [5, 64, 10, 4, 2, 2, 64, 3, 3, 1, 1, 64, 3, 3, 1, 1, 64, 3, 3, 1, 1, 64, 3, 3, 1, 1]

logits = models.create_model(
    fingerprint_input=fingerprint_input,
    model_settings=model_settings,
    model_architecture="ds_cnn",
    model_size_info=model_size_info,
    is_training=False)

# Softmax for probabilities
probabilities_op = tf.nn.softmax(logits, name='labels_softmax')

# Load the weights from the latest best checkpoint
saver = tf.train.Saver(tf.global_variables())
checkpoint_state = tf.train.get_checkpoint_state(CHECKPOINT_DIR)
if not checkpoint_state or not checkpoint_state.model_checkpoint_path:
    print("No checkpoint found in", CHECKPOINT_DIR)
    sys.exit(1)
print(f"Loading checkpoint: {checkpoint_state.model_checkpoint_path}")
saver.restore(sess, checkpoint_state.model_checkpoint_path)

# --- 4. 마이크 스트리밍 세팅 ---
window_history = collections.deque(maxlen=2)
suppression_counter = 0
SUPPRESSION_PULL_DOWN = 6

audio_buffer = np.zeros(CLIP_DURATION_SAMPLES, dtype=np.float32)

p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16,
                channels=1,
                rate=SAMPLE_RATE,
                input=True,
                frames_per_buffer=CHUNK_SIZE)

print("\n==== 🎤 실시간 마이크 KWS(한국어 .ckpt 추론) 시작 ====")
print("('퀴즈' 또는 '이해하셨나요'를 말해보세요!)")
print("=========================================================")

try:
    while True:
        data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
        audio_chunk = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
        volume = np.max(np.abs(audio_chunk))
        
        audio_buffer = np.roll(audio_buffer, -CHUNK_SIZE)
        audio_buffer[-CHUNK_SIZE:] = audio_chunk
        
        # 1. 1초 오디오 -> MFCC Feature 추출
        mfcc_feat = sess.run(mfcc_flatten, feed_dict={wav_placeholder: audio_buffer.reshape(-1, 1)})
        
        # 2. MFCC -> Logits -> Probabilities (using the restored DS-CNN)
        probs = sess.run(probabilities_op, feed_dict={fingerprint_input: mfcc_feat})[0]
        
        window_history.append(probs)
        
        if len(window_history) == window_history.maxlen:
            smoothed_output = np.mean(window_history, axis=0)
            top_index = np.argmax(smoothed_output)
            top_score = smoothed_output[top_index]
            prediction = LABELS[top_index]
            
            if suppression_counter > 0:
                suppression_counter -= 1
                msg = f"  (감지 보류 중... 볼륨: {volume:.2f})           "
                print(msg, end='\r', flush=True)
                continue
            
            if volume < 0.02:
                prediction = '_silence_'
            
            # 타겟 한국어 포착 확인 (단어별 독립 민감도 적용)
            # 'quiz'는 비교적 명확하게 잡히므로 0.5 유지, 'understand'는 길어서 점수가 분산되므로 0.45로 더 민감하게 설정
            if (prediction == 'quiz' and top_score > 0.4) or (prediction == 'understand' and top_score > 0.45):
                print(f"🔥 포착됨: '{prediction}' (신뢰도: {top_score*100:.1f}%, 볼륨: {volume:.2f})")
                suppression_counter = SUPPRESSION_PULL_DOWN
            else:
                msg = f"  ({prediction}: {top_score*100:.1f}%, 볼륨: {volume:.2f})           "
                print(msg, end='\r', flush=True)

except KeyboardInterrupt:
    print("\n==== 🛑 마이크 KWS 종료 ====")
finally:
    stream.stop_stream()
    stream.close()
    p.terminate()
    sess.close()
