# 🚨 라즈베리파이 환경 구축 트러블슈팅 로그

> **작성일:** 2026-02-25
> **목적:** 라즈베리파이에 `tflite-runtime`을 설치하며 겪은 실패 과정과 해결책 기록

---

## 1. 실패 1: 외부 관리 환경 (PEP 668) 에러
**상황:**
- 최신 라즈베리파이 OS (Debian 12 Bookworm)에서 `pip install` 시도 시 발생.
- `error: externally-managed-environment` 메세지 발생.
**원인:**
- 라즈베리파이 OS가 시스템 패키지 매니저(`apt`)와 `pip`의 충돌을 방지하기 위해 전역 설치를 차단함.
**해결 대안:**
- `python3 -m venv`로 가상 환경 구축 후 설치 또는 `--break-system-packages` 플래그 사용.

## 2. 실패 2: 파이썬 버전에 맞는 라이브러리 부재
**상황:**
- `pip install tflite-runtime` 시 `No matching distribution found` 에러 발생.
**원인:**
- 현재 라즈베리파이의 파이썬 버전(3.13)에 호환되는 사전 컴파일된 `tflite-runtime` Wheel(.whl) 파일이 공식 및 서드파티 저장소에 존재하지 않음.

## 3. 실패 3: 거대 `tensorflow` 패키지 설치 중 용량 부족 (OS Error 28)
**상황:**
- 대안으로 `pip install tensorflow` 명렁어를 통해 전체 묶음을 설치하려 시도.
- 약 260MB짜리 덩어리를 받아 압축을 푸는 과정에서 라즈베리파이 SD 카드 가용 용량(6.3GB 수준)을 캐시가 전부 잡아먹고 디스크 풀(Disk Full) 현상 발생.
**조치 사항:**
- `pip cache purge` 및 `pip uninstall tensorflow` 등 명령어로 캐시와 찌꺼기 삭제하여 용량 복구.

## 4. 🌟 최종 해결책: `pyenv`를 통한 Python 다운그레이드 (진행 중)
**결론:**
- 라즈베리파이에 무거운 Tensorflow를 올리는 대신 가벼운 `tflite-runtime`을 쓰는 것이 맞으나, 파이썬 3.13 버전의 벽에 막힘.
- 따라서, **`pyenv`를 설치하여 호환성이 가장 검증된 Python 3.9.18 버전 환경을 독립적으로 빌드**하여 구성하는 방향으로 선회.
- 현재 라즈베리파이 기기 내에서 Python 3.9 소스의 컴파일 작업이 진행 중임.
