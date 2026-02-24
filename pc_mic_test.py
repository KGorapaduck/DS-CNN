import pyaudio
import numpy as np
import tensorflow as tf
from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio
import collections
import time

# --- 1. 파라미터 설정 ---
SAMPLE_RATE = 16000
CHUNK_DURATION_MS = 250  # 0.25초마다 마이크에서 가져옴
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION_MS / 1000)  # 4000 samples
WINDOW_SIZE_MS = 40.0
WINDOW_STRIDE_MS = 20.0
DCT_COEFFICIENT_COUNT = 10
CLIP_DURATION_SAMPLES = SAMPLE_RATE  # 1초 분량 (16000)

LABELS = ['_silence_', '_unknown_', 'yes', 'no']

# --- 2. TFLite 모델 로딩 ---
interpreter = tf.lite.Interpreter(model_path="./work/ds_cnn.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# --- 3. 오디오 특징 추출(MFCC)을 위한 TF 그래프 구축 ---
tf.reset_default_graph()
sess = tf.Session()

wav_placeholder = tf.placeholder(tf.float32, [CLIP_DURATION_SAMPLES, 1])
spectrogram = contrib_audio.audio_spectrogram(
    wav_placeholder,
    window_size=int(SAMPLE_RATE * WINDOW_SIZE_MS / 1000),
    stride=int(SAMPLE_RATE * WINDOW_STRIDE_MS / 1000),
    magnitude_squared=True)
mfcc_op = contrib_audio.mfcc(
    spectrogram,
    SAMPLE_RATE,
    dct_coefficient_count=DCT_COEFFICIENT_COUNT)
# output shape of mfcc: [1, 49, 10] (if 1 sec window). => Flatten to [1, 490]
mfcc_flatten = tf.reshape(mfcc_op, [1, -1])

# --- 4. 안정성 로직 세팅 ---
window_history = collections.deque(maxlen=2)  # 최근 4번(1초)의 결과를 평균 내어 스무딩
suppression_counter = 0                       # 감지 후 일정 시간 동안 중복 감지 무시
SUPPRESSION_PULL_DOWN = 6                     # 감지 후 6틱(1.5초) 동안은 무시

# --- 5. 마이크 스트리밍 세팅 ---
# 1초 분량(16000 패딩)의 버퍼 큐 생성
audio_buffer = np.zeros(CLIP_DURATION_SAMPLES, dtype=np.float32)

p = pyaudio.PyAudio()
# paInt16으로 받고 수동 정규화하는 것이 안전
stream = p.open(format=pyaudio.paInt16,
                channels=1,
                rate=SAMPLE_RATE,
                input=True,
                frames_per_buffer=CHUNK_SIZE)

print("\n==== 🎤 실시간 마이크 KWS 시작 ====")
print("('yes' 또는 'no'를 말해보세요!)")
print("===================================")

try:
    while True:
        # 마이크로부터 청크(0.25초 분량) 읽기
        data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
        # Int16 -> [-1.0, 1.0] 스케일로 정규화 (학습 시와 완벽 동일)
        audio_chunk = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
        
        # 음성 볼륨 체크 (마이크가 제대로 작동하는지 확인용)
        volume = np.max(np.abs(audio_chunk))
        
        # 이전 음성을 뒤로 밀고, 새로운 음성을 앞에 추가 (Sliding Window 방식)
        audio_buffer = np.roll(audio_buffer, -CHUNK_SIZE)
        audio_buffer[-CHUNK_SIZE:] = audio_chunk
        
        # --- (A) 1초 분량 오디오 -> MFCC 변환 ---
        feed_dict = {wav_placeholder: audio_buffer.reshape(-1, 1)}
        fingerprint = sess.run(mfcc_flatten, feed_dict=feed_dict)
        
        # --- (B) MFCC -> TFLite 추론 ---
        interpreter.set_tensor(input_details[0]['index'], fingerprint)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])[0]
        
        # 히스토리에 현재 예측 저장
        window_history.append(output_data)
        
        # --- (C) 예측 스무딩 (Smoothing) 및 포착 로직 ---
        if len(window_history) == window_history.maxlen:
            # 최근 기록의 평균을 구해 튀는 값을 방지
            smoothed_output = np.mean(window_history, axis=0)
            top_index = np.argmax(smoothed_output)
            top_score = smoothed_output[top_index]
            prediction = LABELS[top_index]
            
            # 억제(Suppression) 진행 중이면 카운터 감소
            if suppression_counter > 0:
                suppression_counter -= 1
                msg = f"  (감지 대기 중... 볼륨: {volume:.2f})           "
                print(msg, end='\r', flush=True)
                continue
            
            # 볼륨이 너무 작으면 배경 잡음으로 간주
            if volume < 0.05:
                prediction = '_silence_'
            
            # 포착 (Smoothing된 신뢰도가 80% 이상이고 타깃일 때)
            if top_score > 0.8 and prediction in ['yes', 'no']:
                print(f"🔥 포착됨: '{prediction}' (신뢰도: {top_score*100:.1f}%, 볼륨: {volume:.2f})")
                suppression_counter = SUPPRESSION_PULL_DOWN  # 중복 포착 방지
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
