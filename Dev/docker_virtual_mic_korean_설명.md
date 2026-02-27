# docker_virtual_mic_korean.py 코드 분석 및 설명

이 문서는 `docker_virtual_mic_korean.py` 파일의 구조와 핵심 동작 원리에 대해 상세히 분석하여 설명합니다.

## 1. 개요 (Overview)
이 스크립트는 실제 마이크(Microphone) 하드웨어를 사용하는 대신, 사전에 녹음된 **`.wav` 오디오 파일을 스트리밍 방식으로 읽어 들여 가상의 마이크처럼 에뮬레이션(Emulation)하는 역할**을 수행합니다. 읽어 들인 오디오 데이터는 실시간으로 전처리되어 고정 그래프(Frozen Graph, `.pb`) 형태의 DS-CNN 모델에 입력되며, 지정된 키워드('quiz', 'understand')를 검출하는 키워드 스포팅(Keyword Spotting, KWS) 작업을 수행합니다.

## 2. 전체 소스 코드
```python
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
```

## 3. 단계별 핵심 로직 설명

### 3.1 파라미터 및 환경 설정 (Parameter Configuration)
오디오 전처리 및 모델 추론을 위한 상수를 정의하고 외부 파일의 존재 여부를 검증합니다.

| 변수명 | 설정값 | 설명 |
| :--- | :--- | :--- |
| `SAMPLE_RATE` | `16000` | 오디오 샘플링 속도 (1초당 16,000개 샘플) |
| `CHUNK_DURATION_MS` | `250` | 한 번에 읽어 들일 오디오 청크(Chunk)의 길이 (0.25초) |
| `CLIP_DURATION_SAMPLES`| `16000` | 모델이 요구하는 입력 버퍼 사이즈 (1초 분량) |
| `LABELS` | `['_silence_', '_unknown_', 'quiz', 'understand']` | 모델이 분류하는 4가지 클래스 레이블 |
| 경로 변수 | `WAV_FILE_PATH`, `PB_MODEL_PATH` | 재생할 WAV 파일과 추론용 모델(.pb) 파일 경로 |

### 3.2 텐서플로우(TensorFlow) 그래프 구축 및 모델 로드
TensorFlow 1.x 환경(또는 호환 모드)을 활용하여 오디오 스트림을 처리하고 모델을 메모리에 올립니다.

* **MFCC 추출 그래프 (`contrib_audio`)**:
  오디오 파형(Waveform)을 모델이 잘 인식할 수 있는 주파수 특성인 **MFCC(Mel-frequency cepstral coefficients)** 로 변환하는 연산을 정의합니다.
* **모델 로드 및 입출력 텐서(Tensor) 바인딩**:
  `ds_cnn_korean_frozen.pb` 파일에서 그래프 정의를 파싱합니다.
  * 입력 텐서: `frozen_model/Reshape:0`
  * 출력 텐서: `frozen_model/labels_softmax:0` (각 레이블에 대한 확률값 반환)

### 3.3 오디오 슬라이딩 윈도우 (Sliding Window Buffer)
실시간 감지 환경과 동일한 조건을 맞추기 위해 1초 크기의 버퍼(`audio_buffer`)를 운영합니다. 파일 라인 110번부터 112번 라인까지의 코드가 이에 해당합니다.

* 매 회전마다 WAV 파일에서 `0.25초(CHUNK_SIZE)` 분량의 데이터를 새롭게 읽어옵니다.
* 기존 `audio_buffer`의 데이터를 왼쪽으로 `0.25초`만큼 밀어내고, 가장 오른쪽 빈 공간에 새롭게 읽은 `0.25초` 분량의 데이터를 이어 붙입니다 (**Overlap 기법**).
* 결과적으로 버퍼는 항상 **가장 최근의 1초 분량 오디오 데이터**를 유지하게 됩니다.

### 3.4 추론 및 판정 로직 (Inference & Smoothing)
MFCC 변환 후 모델에 입력하여 결과를 얻고, 노이즈나 일시적 오류로 인한 오작동을 방지하는 후처리(Post-processing)를 진행합니다.

1. **예측 스무딩 (Prediction Smoothing)**:
   * `window_history`에 최근 2회의 예측 확률(Probability) 값을 저장하고 평균(`np.mean`)을 냅니다. 이는 순간적인 예측 튐 현상(False Positive)을 방지합니다.
2. **볼륨(Volume) 기반 필터링**:
   * 청크의 최대 볼륨 값이 0.02 미만일 경우 강제로 `_silence_`로 판정합니다.
3. **클래스별 임계값(Threshold) 차등 적용**:
   * 모델의 특성에 맞추어 `quiz` 키워드는 예측 신뢰도가 **0.6 (60%)** 이상일 때, `understand` 키워드는 **0.3 (30%)** 이상일 때 검출된 것으로 인정합니다.
4. **억제 타이머 (Suppression Logic / Debounce)**:
   * `suppression_counter` 변수를 사용하여 키워드가 한 번 인식되면 일정 시간(루프 6회, 즉 1.5초) 동안은 추가 인식을 보류합니다. 하나의 발화가 여러 번 중복 인식되는 것을 막기 위한 필수 장치입니다.

### 3.5 실시간 에뮬레이션 지연 (Time Delay)
* 파일 153번 라인의 `time.sleep(0.25)` 코드는 매우 짧은 시간 안에 오디오 파일을 끝까지 읽어버리는 것을 방지하기 위해 존재합니다.
* 읽어 들인 청크의 크기(0.25초)만큼 의도적인 대기 시간을 주어, 마치 실제 마이크가 시간에 따라 소리를 받아들이는 것과 동일한 속도(1배속 재생)로 동작하게 만듭니다.
