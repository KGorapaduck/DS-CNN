import socket
import time

SERVER_IP = '172.16.206.93' 
PORT = 9999

client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

try:
    client_socket.connect((SERVER_IP, PORT))
    print(f"서버({SERVER_IP})에 연결되었습니다.")

    while True:
        # 1. 서버에 데이터 요청
        client_socket.send("GET_TEMP".encode('utf-8'))
        
        # 2. 데이터 수신 및 출력
        response = client_socket.recv(1024).decode('utf-8')
        print(f"수신 데이터: {response}")
        
        # 3. 지연 시간 추가 (1초 대기)
        time.sleep(1)

except KeyboardInterrupt:
    print("\n사용자에 의해 종료되었습니다.")
except Exception as e:
    print(f"오류 발생: {e}")
finally:
    client_socket.close()
    print("연결 종료")