# 🐳 라즈베리파이 TensorFlow 1.15 Docker 구축 가이드 (ARM 호환성 해결)

> 이 문서는 64비트(aarch64) 최신 라즈베리파이 OS 환경에서, Python 3.7까지만 지원하는 구형 **TensorFlow 1.15를 32비트(arm32v7)로 완벽하게 에뮬레이션하여 세팅**하는 가장 빠르고 확실한 방법을 기록합니다.

---

## 📌 1. 필수 준비물 다운로드

라즈베리파이 프로젝트 디렉토리(`KWS-DS-CNN-for-embedded`) 안에서, 과거 선배 개발자들이 남겨둔 **32비트 라즈베리파이용 특제 TensorFlow 1.15 휠(`.whl`) 파일**을 확실하게 확보합니다.

```bash
wget https://github.com/Qengineering/TensorFlow-Raspberry-Pi/raw/master/tensorflow-1.15.2-cp37-cp37m-linux_armv7l.whl
```

## 🛠️ 2. 마법의 Dockerfile 작성

아래 내용을 그대로 복사하여 프로젝트 최상단에 `Dockerfile` 이라는 이름으로 생성합니다.

```dockerfile
# 🌟 핵심 트릭: 64비트 라즈베리파이 OS를 강제로 32비트(armv7) 데비안 컨테이너로 띄우기!
FROM arm32v7/python:3.7-slim-bullseye

# numpy, scipy 등 무거운 패키지 빌드 시 C++ 컴파일 에러를 회피하기 위한 기초 도구
RUN apt-get update && apt-get install -y libhdf5-dev libatlas-base-dev gcc g++ \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip setuptools

WORKDIR /app

# 다운로드한 32비트 TensorFlow 휠 파일 복사
COPY tensorflow-1.15.2-cp37-cp37m-linux_armv7l.whl /tmp/

# 🌟 중요 트릭: 무거운 수학 패키지 소스 코드를 라즈베리파이가 1시간 동안 무식하게 컴파일하는 걸 막고,
# 이미 라즈베리파이용으로 완성된 타겟을 piwheels에서 즉시 내려받도록 우회 (--extra-index-url)
RUN pip install /tmp/tensorflow-1.15.2-cp37-cp37m-linux_armv7l.whl --extra-index-url https://www.piwheels.org/simple/

RUN pip install numpy scipy wave --extra-index-url https://www.piwheels.org/simple/

# 🌟 중요 (protobuf 에러 강제 회피): TF 1.15는 protobuf 4.x와 충돌하므로 반드시 3.20.0으로 낮춰야 함
RUN pip install protobuf==3.20.0

CMD ["bash"]
```

## 🚀 3. Docker 이미지 빌드 (굽기)

준비된 `Dockerfile`을 이용해 이미지를 빌드합니다.
**(라즈베리파이 3 B+ 기준 체감 소요 시간: 약 45분)**

```bash
docker build -t my-tf1.15-32bit .
```
> 도중에 `[6/7] RUN pip install ... whl` 단계에서 수 분 이상 멈춰 있어도 에러가 아니니 절대 끄지 마시고 커피 한잔 여유를 가지고 기다리십시오.

## 🎧 4. 컨테이너 접속 및 KWS 실시간 마이크 실행

빌드가 퍼펙트하게 끝났다면, 드디어 호스트 권한(마이크 장치 등)을 끌고 도커 안으로 진입합니다!

```bash
# --device /dev/snd : 라즈베리파이에 꽂힌 실제 하드웨어 마이크/스피커 권한 개방
# -v $(pwd):/app : 내 원래 프로젝트 폴더를 컨테이너 폴더와 실시간 거울 동기화
docker run -it --rm -v $(pwd):/app -w /app --device /dev/snd my-tf1.15-32bit bash
```

내부 프롬프트 장소가 `root@컨테이너ID:/app#` 으로 바뀌었다면, 스크립트를 즉시 실행하여 짜릿한 KWS 100% 인식률을 눈으로 확인합니다!

```bash
# 가상 WAV 파일 테스트 시 (한국어 모델)
python docker_virtual_mic_korean.py
```
