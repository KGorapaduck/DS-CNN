import socket
import numpy as np
import tensorflow as tf
from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio
import collections
import time
import os

# --- 1. 파라미터 및 통신 설정 ---
HOST = '0.0.0.0'  # 모든 인터페이스에서 접속 허용
PORT = 9999       # 포트 번호

SAMPLE_RATE = 16000
CHUNK_DURATION_MS = 250
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION_MS / 1000)  # 4000 samples
WINDOW_SIZE_MS = 40.0
WINDOW_STRIDE_MS = 20.0
DCT_COEFFICIENT_COUNT = 10
CLIP_DURATION_SAMPLES = SAMPLE_RATE

LABELS = ['_silence_', '_unknown_', 'quiz', 'understand']
PB_MODEL_PATH = "work/ds_cnn_korean_frozen.pb"

if not os.path.exists(PB_MODEL_PATH):
    PB_MODEL_PATH = "ds_cnn_korean_frozen.pb"

if not os.path.exists(PB_MODEL_PATH):
    print(f"❌ 오류: '{PB_MODEL_PATH}' 파일을 찾을 수 없습니다!")
    exit(1)

# --- 2. TF 세션 및 그래프 구축 ---
tf.reset_default_graph()
sess = tf.compat.v1.InteractiveSession()
wav_placeholder = tf.placeholder(tf.float32, [CLIP_DURATION_SAMPLES, 1])

# MFCC 추출 그래프
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

# --- 3. Frozen Graph 로드 ---
graph_def = tf.compat.v1.GraphDef()
with tf.io.gfile.GFile(PB_MODEL_PATH, 'rb') as f:
    graph_def.ParseFromString(f.read())
tf.import_graph_def(graph_def, name='frozen_model')

model_graph = tf.compat.v1.get_default_graph()
fingerprint_input = model_graph.get_tensor_by_name("frozen_model/Reshape:0")
probabilities_op = model_graph.get_tensor_by_name("frozen_model/labels_softmax:0")

# --- 4. 안정성 로직 세팅 ---
window_history = collections.deque(maxlen=2)
suppression_counter = 0                       
SUPPRESSION_PULL_DOWN = 6                     
audio_buffer = np.zeros(CLIP_DURATION_SAMPLES, dtype=np.float32)

def recvall(sock, n):
    """지정된 바이트 수(n)만큼 소켓에서 완전히 읽어오는 헬퍼 함수"""
    data = bytearray()
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data.extend(packet)
    return data

# --- 5. 소켓 서버 구동 ---
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server_socket.bind((HOST, PORT))
server_socket.listen(1)

print(f"\n==== 📡 소켓 서버 시작 (포트: {PORT}) ====")
print(f"모델 파일: {PB_MODEL_PATH}")
print("PC(클라이언트)의 연결을 대기 중입니다...\n")

while True:
    client_socket, addr = server_socket.accept()
    print(f"\n✅ 클라이언트 연결됨: {addr}")
    
    # 연결될 때마다 버퍼 초기화
    audio_buffer = np.zeros(CLIP_DURATION_SAMPLES, dtype=np.float32)
    window_history.clear()
    suppression_counter = 0
    
    try:
        while True:
            # 16bit(2 bytes) * 4000 samples = 8000 bytes
            raw_data = recvall(client_socket, CHUNK_SIZE * 2)
            
            if not raw_data:
                print(f"클라이언트({addr}) 연결 종료")
                break
            
            # Int16 -> Float32 [-1.0, 1.0] 정규화
            audio_chunk_int16 = np.frombuffer(raw_data, dtype=np.int16)
            audio_chunk = audio_chunk_int16.astype(np.float32) / 32768.0
            volume = np.max(np.abs(audio_chunk))
            
            # 슬라이딩 윈도우
            audio_buffer = np.roll(audio_buffer, -CHUNK_SIZE)
            audio_buffer[-CHUNK_SIZE:] = audio_chunk
            
            # MFCC 변환 및 추론
            mfcc_feat = sess.run(mfcc_flatten, feed_dict={wav_placeholder: audio_buffer.reshape(-1, 1)})
            probs = sess.run(probabilities_op, feed_dict={fingerprint_input: mfcc_feat})[0]
            
            window_history.append(probs)
            
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
                
                if (prediction == 'quiz' and top_score >= 0.6) or (prediction == 'understand' and top_score >= 0.3):
                    print(f"🔥 포착됨: '{prediction}' (신뢰도: {top_score*100:.1f}%, 볼륨: {volume:.2f})")
                    
                    # 클라이언트로 트리거 이벤트 전송 (newline 포함)
                    send_msg = f"TRIGGER_{prediction.upper()}\n"
                    client_socket.sendall(send_msg.encode('utf-8'))
                    
                    suppression_counter = SUPPRESSION_PULL_DOWN
                else:
                    msg = f"  ({prediction}: {top_score*100:.1f}%, 볼륨: {volume:.2f})"
                    print(msg.ljust(50), end='\r', flush=True)

    except ConnectionResetError:
        print(f"⚠️ 클라이언트({addr}) 비정상 종료")
    finally:
        client_socket.close()
