import wave
import numpy as np
import time
from python_speech_features import mfcc
# 라즈베리파이에서는 tensorflow 전체 대신 가벼운 tflite_runtime을 사용합니다.
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    print("tflite_runtime이 설치되지 않았습니다. TFLite 대신 일반 tensorflow를 시도합니다.")
    import tensorflow as tf
    tflite = tf.lite

# --- 1. 파라미터 설정 ---
SAMPLE_RATE = 16000
CHUNK_DURATION_MS = 250  # 0.25초 단위
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION_MS / 1000)  # 4000 samples
WINDOW_SIZE_MS = 40.0
WINDOW_STRIDE_MS = 20.0
DCT_COEFFICIENT_COUNT = 10
CLIP_DURATION_SAMPLES = SAMPLE_RATE  # 1초 분량 (16000)

LABELS = ['_silence_', '_unknown_', 'yes', 'no']

# 라즈베리파이에 넣은 오디오 및 모델 경로 (경로는 라즈베리파이 환경에 맞게 수정)
VIRTUAL_MIC_WAV = "virtual_mic_test.wav"
TFLITE_MODEL = "ds_cnn.tflite"

# --- 2. TFLite 모델 로딩 ---
interpreter = tflite.Interpreter(model_path=TFLITE_MODEL)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# --- 3. 안정성 로직 세팅 ---
# 최근 4번(1초)의 결과를 평균 내어 스무딩
window_history = []
suppression_counter = 0                       # 감지 후 일정 시간 동안 중복 감지 무시
SUPPRESSION_PULL_DOWN = 6                     # 감지 후 6틱(1.5초) 동안은 무시

# 1초 분량(16000 패딩)의 버퍼 큐 생성
audio_buffer = np.zeros(CLIP_DURATION_SAMPLES, dtype=np.float32)

print(f"\n==== 🎧 [라즈베리파이용] 가상 마이크(WAV 스트리밍) 시작 ====")
print(f"소스 파일: {VIRTUAL_MIC_WAV}")
print("=========================================================\n")

# --- 4. 가상 마이크 스트리밍 시작 ---
try:
    with wave.open(VIRTUAL_MIC_WAV, 'rb') as wf:
        if wf.getframerate() != SAMPLE_RATE:
            print(f"경고: 샘플 레이트가 {SAMPLE_RATE}Hz가 아닙니다! ({wf.getframerate()}Hz)")
        
        total_frames = wf.getnframes()
        processed_frames = 0
        
        while processed_frames < total_frames:
            # 시간 지연으로 실시간(Real-time) 마이크 수음 환경 모방
            time.sleep(CHUNK_DURATION_MS / 1000.0)
            
            # 마이크로부터 청크(0.25초 분량) 읽어오기 (WAV에서 추출)
            data = wf.readframes(CHUNK_SIZE)
            if not data:
                break
            
            # 읽어온 데이터 정규화 (-1.0 ~ 1.0)
            audio_chunk = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            if len(audio_chunk) < CHUNK_SIZE:
                padded = np.zeros(CHUNK_SIZE, dtype=np.float32)
                padded[:len(audio_chunk)] = audio_chunk
                audio_chunk = padded
                
            volume = np.max(np.abs(audio_chunk))
            
            # 버퍼 갱신 (Sliding Window)
            audio_buffer = np.roll(audio_buffer, -CHUNK_SIZE)
            audio_buffer[-CHUNK_SIZE:] = audio_chunk
            
            # --- (A) 순수 Python으로 MFCC 변환 ---
            # TensorFlow 1.15의 contrib_audio.mfcc와 100% 동일하지는 않지만 근사치로 동작하도록 세팅
            # 실전 배포 시에는 ARM CMSIS-NN C++ 코드로 대체됨
            mfcc_feat = mfcc(audio_buffer, 
                             samplerate=SAMPLE_RATE, 
                             winlen=WINDOW_SIZE_MS/1000, 
                             winstep=WINDOW_STRIDE_MS/1000, 
                             numcep=DCT_COEFFICIENT_COUNT, 
                             nfilt=40, 
                             nfft=1024)
            # TFLite 입력 차원에 맞게 평탄화 후 타입 캐스팅 [1, 490]
            fingerprint = np.reshape(mfcc_feat, (1, -1)).astype(np.float32)
            
            # --- (B) MFCC -> TFLite 추론 ---
            interpreter.set_tensor(input_details[0]['index'], fingerprint)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])[0]
            
            # 기록 저장 및 스무딩
            window_history.append(output_data)
            if len(window_history) > 4:
                window_history.pop(0)
            
            # --- (C) 예측 스무딩 및 포착 로직 ---
            if len(window_history) == 4:
                smoothed_output = np.mean(window_history, axis=0)
                top_index = np.argmax(smoothed_output)
                top_score = smoothed_output[top_index]
                prediction = LABELS[top_index]
                
                # 현재 스트리밍 시간 계산
                current_time_sec = (processed_frames + CHUNK_SIZE) / SAMPLE_RATE
                time_str = f"[{int(current_time_sec//60):02d}:{int(current_time_sec%60):02d}]"
                
                if suppression_counter > 0:
                    suppression_counter -= 1
                    print(f"{time_str} (감지 대기 중... 볼륨: {volume:.2f})           ", end='\r', flush=True)
                else:
                    if volume < 0.05:
                        prediction = '_silence_'
                    
                    if top_score > 0.8 and prediction in ['yes', 'no']:
                        print(f"\n🔥 {time_str} 포착됨: '{prediction}' (신뢰도: {top_score*100:.1f}%, 볼륨: {volume:.2f})")
                        # (TODO) 이 블록에 TCP 소켓 [TRIGGER_ON 전송] 추가 구현 (rasp_socket_practice.py 코드 병합)
                        suppression_counter = SUPPRESSION_PULL_DOWN
                    else:
                        print(f"{time_str} ({prediction}: {top_score*100:.1f}%, 볼륨: {volume:.2f})           ", end='\r', flush=True)

            processed_frames += CHUNK_SIZE

    print("\n\n==== 🎤 가상 스트리밍 파일 재생 종료 ====")
    
except FileNotFoundError:
    print(f"\n[오류] 파일을 찾을 수 없습니다: {VIRTUAL_MIC_WAV}")
except KeyboardInterrupt:
    print("\n\n==== 🛑 가상 마이크 KWS 강제 종료 ====")
