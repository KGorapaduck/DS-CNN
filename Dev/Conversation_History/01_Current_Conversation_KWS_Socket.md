# 대화 기록: KWS 실시간 소켓 통신 및 가상 마이크 구현 (현재)

**ID**: 90a6a8d2-cc3a-43f2-9062-c3c89ae74f58
**날짜**: 2026-02-27

## 주요 내용
1. **가상 마이크 스크립트 분석**: `docker_virtual_mic_korean.py` 코드의 슬라이딩 윈도우, MFCC 변환, 추론 로직을 분석하고 설명 문서를 생성함.
2. **실시간 소켓 통신 설계**: PC의 마이크 데이터를 라즈베리파이로 스트리밍하여 추론을 수행하는 TCP 기반 아키텍처 수립.
3. **컴포넌트 구현**: 
   - `server_pi_socket.py`: 라즈베리파이용 추론 서버.
   - `client_pc_mic.py`: PC용 오디오 스트리밍 클라이언트.
4. **문제 해결**: 
   - `protobuf` 버전 호환성 문제 해결 방안 제시 (`protobuf<=3.20.3` 다운그레이드).
   - 클라이언트 실행 시 인자값(`--ip`) 사용법 및 포트 포워딩(`-p 9999:9999`) 안내.

## 생성 및 수정된 파일
- `docker_virtual_mic_korean_설명.md`
- `realtime_mic_inf.md`
- `server_pi_socket.py`
- `client_pc_mic.py`
