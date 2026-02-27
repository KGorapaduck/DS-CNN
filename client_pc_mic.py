import socket
import pyaudio
import threading
import sys
import argparse

# --- 파라미터 ---
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK_DURATION_MS = 250
CHUNK = int(RATE * CHUNK_DURATION_MS / 1000)  # 4000 samples

def receive_thread(sock):
    """서버(라즈베리파이)로부터 오는 트리거 알림을 비동기적으로 수신하는 스레드"""
    print("가동 준비 완료. 라즈베리파이로부터의 이벤트를 대기합니다...")
    try:
        # 버퍼 리더 역할을 통해 라인 단위로 읽음
        f = sock.makefile('r', encoding='utf-8')
        while True:
            line = f.readline()
            if not line:
                break
            
            line = line.strip()
            if line.startswith("TRIGGER_"):
                keyword = line.replace("TRIGGER_", "")
                print(f"\n[🔔 서버 응답] 키워드 감지 완료! -> '{keyword}'")
                
                if keyword == 'QUIZ':
                    print(">> 퀴즈 프로세스를 실행합니다 (PC 측 로직 처리)")
                elif keyword == 'UNDERSTAND':
                    print(">> 이해도 확인 프로세스를 실행합니다 (PC 측 로직 처리)")
                
    except Exception as e:
        print(f"\n[수신 스레드 종료]: {e}")

def main():
    parser = argparse.ArgumentParser(description="PC 마이크 오디오를 소켓을 통해 라즈베리파이로 스트리밍합니다.")
    parser.add_argument("--ip", type=str, required=True, help="라즈베리파이의 IP 주소 (예: 192.168.0.x)")
    parser.add_argument("--port", type=int, default=9999, help="라즈베리파이 접속 포트 (기본값: 5000)")
    args = parser.parse_args()

    server_ip = args.ip
    server_port = args.port

    # 1. 소켓 연결
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        print(f"서버({server_ip}:{server_port})에 연결 중...")
        client_socket.connect((server_ip, server_port))
        print("연결 성공!")
    except Exception as e:
        print(f"❌ 접속 실패: {e}")
        sys.exit(1)

    # 2. 서버 이벤트 수신 스레드 시작
    tr = threading.Thread(target=receive_thread, args=(client_socket,), daemon=True)
    tr.start()

    # 3. 마이크 설정 (PyAudio)
    audio = pyaudio.PyAudio()
    try:
        stream = audio.open(format=FORMAT,
                            channels=CHANNELS,
                            rate=RATE,
                            input=True,
                            frames_per_buffer=CHUNK)
        
        print("\n==== 🎙️ 마이크 스트리밍 시작 ====")
        print(">> 발화를 시작해 주세요 ('퀴즈', '이해했나요')")
        print(">> (종료하려면 Ctrl+C를 누르세요)")
        
        while True:
            # 마이크에서 0.25초 분량(4000개 샘플)의 바이트 데이터를 읽어옴 (8000 bytes)
            data = stream.read(CHUNK, exception_on_overflow=False)
            
            # 서버로 바로 전송
            client_socket.sendall(data)
            
    except KeyboardInterrupt:
        print("\n중지 중...")
    except Exception as e:
        print(f"오류 발생: {e}")
    finally:
        print("마이크 및 소켓 접속 해제 중...")
        stream.stop_stream()
        stream.close()
        audio.terminate()
        client_socket.close()

if __name__ == "__main__":
    main()
