# 🐳 라즈베리파이 도커 환경 구축 트러블슈팅 로그

> **작성일:** 2026-02-25
> **목적:** 라즈베리파이 환경에서 구형 TensorFlow 1.15를 구동하기 위해 Docker를 도입하며 겪은 문제들과 해결 과정을 기록

---

## 1. 실패 1: 라즈베리파이(ARM)용 공식 Docker Image 부재
**상황:**
- `docker pull lhelontra/tensorflow-on-arm:1.15.0` 등 기존 커뮤니티에서 제공하던 레퍼런스 ARM용 TF 1.15 이미지 다운로드 시도 시 `pull access denied` (Repository 삭제) 오류 리턴.
**원인:**
- 구형 1.x 버전의 ARM용 도커 이미지 배포자들이 저장소를 삭제하거나 비공개 처리함.
- `tensorflow/tensorflow` 공식 저장소 이미지는 `amd64` 전용이므로, 라즈베리파이(ARM)에서 구동할 경우 `exec format error`가 발생하며 호환되지 않음.
**해결 대안:**
- 완성된 이미지가 없다면 깡통 파이썬 이미지 위에 직접 라즈베리파이용 텐서플로우 파일(`.whl`)을 설치하는 나만의 `Dockerfile` 레시피 작성 방식으로 해결.

## 2. 실패 2: 운영체제(64비트)와 파일(32비트)의 Architecture 불일치
**상황:**
- `Dockerfile`을 만들어 `python:3.7-slim-bullseye` 위에서 32비트용(`armv7l`) 텐서플로우 휠 파일 설치를 시도했으나 `is not a supported wheel on this platform` 오류 리턴.
**원인:**
- 라즈베리파이가 64비트(`aarch64`) 운영체제이기 때문에, 도커 컨테이너의 베이스 OS도 호스트 구조를 따라 64비트로 세팅됨.
- 반면 인터넷상에는 파이썬 3.7 + aarch64를 모두 만족하는 TensorFlow 1.15 바이너리 휠(`.whl`)이 전 세계 어디에도 없음. 타임머신을 타고 과거(32비트 라즈베리파이)로 돌아가야만 하는 상황.
**해결 대안:**
- `Dockerfile`의 첫 줄(Base Image)을 강제로 32비트인 `FROM arm32v7/python:3.7-slim-bullseye` 로 선언하여, **64비트 라즈베리파이 OS 위에서 32비트 도커 컨테이너를 에뮬레이션**하는 방식으로 극복. 호환성 문제 완벽 회피 성공.

## 3. 실패 3: numpy 설치 중 c/c++ 의존성 빌드 에러 및 병목 현상
**상황:**
- `arm32v7` 도커 환경에서 `pip install numpy scipy wave` 설치 시, 파일 용량 초과와 오랜 시간 대기 후 `cannot link a simple C program` 등의 컴파일 에러가 발생하며 Exit code 1 리턴.
**원인:**
- 베이스 이미지(`slim`)에 무거운 라이브러리를 자력으로 해석할 수 있는 기초 C 묶음(compiler)이 들어있지 않아서 직접 소스코드를 빌드(Compile)하다가 뻗음.
- 라즈베리파이 3 B+의 빈약한 성능(RAM 1GB)으로는 거대한 수학 패키지를 혼자서 직접 컴파일하기엔 시간이 매우 오래 걸리며(`Getting requirements to build wheel...`), 종종 메모리 부족으로 도중 강제종료됨.
**최종 해결책:**
- `Dockerfile` 내에 `gcc`, `g++` 등의 패키지를 먼저 깔게 조치.
- 나아가서, 억지로 수십 분 동안 라즈베리파이가 무거운 의존성을 직접 빌드하게 놔둘 것이 아니라, **라즈베리파이용 32비트 완성품들이 전부 모여 있는 공식 `piwheels` 저장소 인덱스** 주소를 추가하여, 컴파일 대신 미리 구워진 완성품(`.whl`)을 즉시 내려받도록 유도함.

---

## 🚀 4. 최종 완성본: 마법의 Dockerfile 레시피 (성공)

이 파란만장한 과정 끝에 얻어낸 100% 작동을 보장하는 최후의 도커 빌드 코드는 다음과 같습니다.

```dockerfile
# 강제 32비트 아키텍처 지정 (매우 중요)
FROM arm32v7/python:3.7-slim-bullseye

# C++ 의존성 에러 회피를 위한 기초 컴파일 도구 마운트
RUN apt-get update && apt-get install -y libhdf5-dev libatlas-base-dev gcc g++ \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip setuptools

WORKDIR /app

# 미리 준비된 32비트용 특제 텐서플로우 1.15 파일 복사
# wget https://github.com/Qengineering/TensorFlow-Raspberry-Pi/raw/master/tensorflow-1.15.2-cp37-cp37m-linux_armv7l.whl
COPY tensorflow-1.15.2-cp37-cp37m-linux_armv7l.whl /tmp/

# 핵심 트릭 1: piwheels 저장소 이용 우회
# 핵심 트릭 2: 32비트 휠 파일 강제 설치
RUN pip install /tmp/tensorflow-1.15.2-cp37-cp37m-linux_armv7l.whl --extra-index-url https://www.piwheels.org/simple/

# 핵심 연산 라이브러리들 또한 원클릭 설치를 위해 piwheels 배급사 이용
RUN pip install numpy scipy wave --extra-index-url https://www.piwheels.org/simple/

CMD ["bash"]
```
*(위 도커파일 기반으로 라즈베리파이 3 B+에서 빌드했을 때 약 45~50분 소요 끝에 정상 `FINISHED` 확인 완료.)*
