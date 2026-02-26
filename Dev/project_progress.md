# 📌 KWS DS-CNN 프로젝트 진행 현황

> **최종 수정:** 2026-02-24

---

## Phase 1: 파이프라인 관통 (1주차 목표)

### 모델 학습 및 변환
- [x] 프로젝트 구조 분석
- [x] 라즈베리파이 3 B+ (1GB RAM) 적합성 분석
- [x] Speech Commands v0.01 데이터셋 다운로드
- [x] Conda `kws` 환경 구축 (Python 3.7 + TF 1.15)
- [x] DS-CNN 모델 학습 (yes/no, Float32) — Test Accuracy 96.72%
- [x] `freeze.py`로 체크포인트 → Frozen Graph (`.pb`) 변환
- [x] Frozen Graph → TFLite (`.tflite`) 변환
- [x] TFLite 모델 추론 테스트 (PC에서)

### 라즈베리파이 배포
- [x] 라즈베리파이 OS 세팅 (Lite 버전 권장)
- [x] 가상 마이크 스트리밍 테스트 스크립트 작성 (`pi_virtual_mic_test.py`)
- [ ] USB 마이크 수음 환경 세팅 (pyaudio, 16kHz)
- [ ] `tflite_runtime` 설치 및 `.tflite` 모델 배포 (진행 중 - 파이썬 버전 충돌 및 용량 부족 이슈 해결 중)
- [ ] 키워드 감지 시 TCP/UDP 소켓 통신 (`TRIGGER_ON`) 구현

---

## Phase 2: 프론트엔드 연동 및 서버 융합 (2주차 목표)

### 서버 (교수자 PC)
- [ ] `TRIGGER_ON` 수신 대기 소켓 서버 구축
- [ ] 수신 시 오디오 스트림 수신 체계 구현
- [ ] 로컬 STT (Whisper.cpp 등) 실행 및 텍스트 추출
- [ ] ChatGPT API 연동 → JSON 포맷 퀴즈 생성

### 프론트엔드
- [ ] WebSocket으로 퀴즈 팝업 배포 연동
- [ ] (선택) 메인 Whisper 자막 시스템과 연동 테스트

---

## Phase 3: 모델 최적화 및 안정화 (3주차 목표)

### 양자화 및 최적화
- [ ] `trainTest_quant.py`로 DS-CNN 양자화(INT8) 재학습
- [ ] 양자화 모델 정확도 vs Float32 비교 테스트
- [ ] (여유 시) C/C++ 포팅 시도

### 한국어 키워드 모델 전환 및 검증
- [x] 한국어 당면 과제: 키워드 데이터 수집/녹음 
  - [x] '퀴즈' 키워드: 웹 음원 기반 증강(Augmentation) 데이터 구축 완료
  - [x] '이해하셨나요' 키워드: 사용자 음성 12개 기반 Truncation & 증강 데이터 3,100개 구축 완료
  - [x] 비-타겟(`Unknown`) 보충: AI Hub 대화 데이터셋 기반 1초 단위 랜덤 추출(3,100개) 
- [x] 커스텀 한국어 데이터로 DS-CNN 재학습 (Best Accuracy 99.99% 달성)
- [x] **[신규]** PC 기반 실시간 마이크 KWS 추론 테스트 및 디버깅 (`OperatorNotAllowedInGraphError` 등 텐서플로우 오류 해결)
- [x] **[신규]** AIHub 데이터셋 일괄 추론을 통한 키워드별 최적 민감도(Threshold) 정밀 도출 및 적용 완료 ('quiz': 0.6, 'understand': 0.3)
- [x] 엣지 배포용 모델 포맷 변환 완료 (Freezing -> `.pb`) 및 단독 검증 통과 
- ~~[ ] (추가 선택) 가벼운 `.tflite` (TensorFlow Lite) 포맷 변환 시도 (실패 12~14 사유로 생략, 원본 Docker 및 .pb 구동으로 직행)~~
- [x] 라즈베리파이 한국어 KWS 실전 통합 테스트 (`docker_virtual_mic_korean.py` 등 100% 감지 완료)

### 안정화
- [ ] 예외 처리 (마이크 에러, LLM 응답 포맷 에러 등) 방어 코드
- [ ] 전체 시스템 통합 테스트
