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
# 2. Frozen Graph(.pb) 로드
graph_def = tf.compat.v1.GraphDef()
with tf.io.gfile.GFile("work/ds_cnn_korean_frozen.pb", 'rb') as f:
    graph_def.ParseFromString(f.read())
    
# 현재 Graph에 로드한 노드 추가
tf.import_graph_def(graph_def, name='frozen_model')

# Graph 내 텐서(Tensor) 가져오기
model_graph = tf.compat.v1.get_default_graph()
fingerprint_input = model_graph.get_tensor_by_name("frozen_model/Reshape:0")
probabilities_op = model_graph.get_tensor_by_name("frozen_model/labels_softmax:0")


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
            
            # 타겟 한국어 포착 확인 (AIHub 대규모 검증 기반 최적 Threshold 적용)
            # 'quiz' 평가 Median: 0.98 -> 보수적으로 0.6 설정
            # 'understand' 평가 Median: 0.83 (Lowest 0.007 분산 심함) -> 민감하게 0.3 설정
            if (prediction == 'quiz' and top_score >= 0.6) or (prediction == 'understand' and top_score >= 0.3):
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
