import wave
import numpy as np
import tensorflow as tf
from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio
import collections
import time
import os

# --- 1. 파라미터 설정 ---
SAMPLE_RATE = 16000
CHUNK_DURATION_MS = 250  # 0.25초마다 모델 추론 (Overlap)
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION_MS / 1000)  # 4000 samples
WINDOW_SIZE_MS = 40.0
WINDOW_STRIDE_MS = 20.0
DCT_COEFFICIENT_COUNT = 10
CLIP_DURATION_SAMPLES = SAMPLE_RATE  # 모델의 입력 사이즈는 무조건 1초 (16000)

LABELS = ['_silence_', '_unknown_', 'quiz', 'understand']
WAV_FILE_PATH = "korean_virtual_mic_test.wav"  # 사용자가 라즈베리파이로 옮길 파일 이름
PB_MODEL_PATH = "work/ds_cnn_korean_frozen.pb"

if not os.path.exists(PB_MODEL_PATH):
    PB_MODEL_PATH = "ds_cnn_korean_frozen.pb" # 라즈베리파이 같은 폴더에 있을 경우 대비

if not os.path.exists(WAV_FILE_PATH):
    print(f"❌ 오류: '{WAV_FILE_PATH}' 파일을 찾을 수 없습니다!")
    print("스크립트와 같은 경로(현재 폴더)에 해당 wav 파일이 있는지 확인해 주세요.")
    print("PC에 있는 녹음 파일을 복사해서 가져오세요!")
    exit(1)

if not os.path.exists(PB_MODEL_PATH):
    print(f"❌ 오류: '{PB_MODEL_PATH}' 파일을 찾을 수 없습니다!")
    print("스크립트와 같은 경로(현재 폴더)에 해당 pb 파일이 있는지 확인해 주세요.")
    exit(1)

# --- 2. TF 세션 및 그래프 구축 ---
tf.reset_default_graph()
sess = tf.compat.v1.InteractiveSession()

wav_placeholder = tf.placeholder(tf.float32, [CLIP_DURATION_SAMPLES, 1])

# MFCC 추출 그래프 (contrib_audio)
spectrogram = contrib_audio.audio_spectrogram(
    wav_placeholder,
    window_size=int(SAMPLE_RATE * WINDOW_SIZE_MS / 1000),
    stride=int(SAMPLE_RATE * WINDOW_STRIDE_MS / 1000),
    magnitude_squared=True)
mfcc_op = contrib_audio.mfcc(
    spectrogram,
    SAMPLE_RATE,
    dct_coefficient_count=DCT_COEFFICIENT_COUNT)
mfcc_flatten = tf.reshape(mfcc_op, [1, -1])

# --- 3. Frozen Graph(.pb) 로드 ---
graph_def = tf.compat.v1.GraphDef()
with tf.io.gfile.GFile(PB_MODEL_PATH, 'rb') as f:
    graph_def.ParseFromString(f.read())
    
tf.import_graph_def(graph_def, name='frozen_model')

model_graph = tf.compat.v1.get_default_graph()
fingerprint_input = model_graph.get_tensor_by_name("frozen_model/Reshape:0")
probabilities_op = model_graph.get_tensor_by_name("frozen_model/labels_softmax:0")

# --- 4. 안정성 로직 세팅 (PC 마이크 최적화와 동일) ---
window_history = collections.deque(maxlen=2)
suppression_counter = 0                       
SUPPRESSION_PULL_DOWN = 6                     

# 1초 분량(16000)의 빈 버퍼(배경음)
audio_buffer = np.zeros(CLIP_DURATION_SAMPLES, dtype=np.float32)

print(f"\n==== 🎧 가상 마이크(WAV) 읽기 시작 (Docker + .pb) ====")
print(f"모델 파일: {PB_MODEL_PATH}")
print(f"재생 파일: {WAV_FILE_PATH}")
print("========================================================\n")

wf = wave.open(WAV_FILE_PATH, 'rb')
elapsed_time_ms = 0

def format_time(ms):
    """밀리초를 분:초 단위로 변환하는 헬퍼 함수"""
    ms = int(ms)
    minutes = ms // 60000
    seconds = (ms % 60000) // 1000
    milliseconds = ms % 1000
    return f"{minutes:02d}:{seconds:02d}.{milliseconds:03d}"

try:
    while True:
        # 1. 가상 마이크(WAV 파일)에서 0.25초 분량(4000개 샘플)씩 잘라서 읽기
        data = wf.readframes(CHUNK_SIZE)
        
        # 파일이 끝났으면 종료
        if len(data) == 0:
            print("\n==== 🏁 WAV 파일 재생 완료 ====")
            break
            
        # 프레임 수가 부족한 경우(파일의 맨 끝부분) 제로 패딩으로 크기 맞추기
        audio_chunk_int16 = np.frombuffer(data, dtype=np.int16)
        if len(audio_chunk_int16) < CHUNK_SIZE:
            padded_chunk = np.zeros(CHUNK_SIZE, dtype=np.int16)
            padded_chunk[:len(audio_chunk_int16)] = audio_chunk_int16
            audio_chunk_int16 = padded_chunk
            
        # Int16 -> Float32 [-1.0, 1.0] 정규화
        audio_chunk = audio_chunk_int16.astype(np.float32) / 32768.0
        volume = np.max(np.abs(audio_chunk))
        
        # 슬라이딩 윈도우: 버퍼 안의 내용물을 왼쪽으로 0.25초 밀고 새 0.25초 채우기
        audio_buffer = np.roll(audio_buffer, -CHUNK_SIZE)
        audio_buffer[-CHUNK_SIZE:] = audio_chunk
        
        # --- (A) 1초 분량 오디오 -> MFCC 변환 ---
        mfcc_feat = sess.run(mfcc_flatten, feed_dict={wav_placeholder: audio_buffer.reshape(-1, 1)})
        
        # --- (B) MFCC -> DS-CNN(.pb) 추론 ---
        probs = sess.run(probabilities_op, feed_dict={fingerprint_input: mfcc_feat})[0]
        
        window_history.append(probs)
        
        # --- (C) 예측 스무딩 및 판정 ---
        if len(window_history) == window_history.maxlen:
            smoothed_output = np.mean(window_history, axis=0)
            top_index = np.argmax(smoothed_output)
            top_score = smoothed_output[top_index]
            prediction = LABELS[top_index]
            
            if suppression_counter > 0:
                suppression_counter -= 1
                msg = f"  (감지 보류 중... 볼륨: {volume:.2f})"
                print(msg.ljust(50), end='\r', flush=True)
                continue
            
            if volume < 0.02:
                prediction = '_silence_'
            
            # Threshold 차등 적용 (퀴즈: 0.6, 이해: 0.3)
            if (prediction == 'quiz' and top_score >= 0.6) or (prediction == 'understand' and top_score >= 0.3):
                timestamp = format_time(elapsed_time_ms)
                print(f"🔥 [{timestamp}] 포착됨: '{prediction}' (신뢰도: {top_score*100:.1f}%, 볼륨: {volume:.2f})")
                
                # 💡 소켓 혹은 HTTP API 통신을 하려면 여기에 통신 코드를 추가하면 됩니다!
                # if prediction == 'quiz':
                #    socket.send("TRIGGER_QUIZ")
                
                suppression_counter = SUPPRESSION_PULL_DOWN
            else:
                msg = f"  ({prediction}: {top_score*100:.1f}%, 볼륨: {volume:.2f})"
                print(msg.ljust(50), end='\r', flush=True)
                    
        # 가상 마이크처럼 보이도록 의도적인 시간 지연 (1배속 재생)
        time.sleep(0.25)
        elapsed_time_ms += CHUNK_DURATION_MS

except KeyboardInterrupt:
    print("\n==== 🛑 강제 종료 ====")
finally:
    wf.close()
    sess.close()
