# 🍓 라즈베리파이 3 B+ KWS 세팅 가이드 (Phase 2)

PC에서 성공적으로 검증된 KWS(Keyword Spotting) 파이프라인을 라즈베리파이에 이식하기 위한 완벽한 가이드입니다. 

우리가 만든 모델은 학습 시 TensorFlow 1.15의 내장 C++ 모듈(`contrib_audio`)을 사용하여 오디오 특징(MFCC)을 추출했습니다. 이 연산은 표준 TFLite 런타임에 호환되지 않으므로, 라즈베리파이에도 가벼운 **임베디드용 TensorFlow 1.15**를 설치해야 완벽하게 동일한 성능을 낼 수 있습니다.

---

이 가이드는 사용자의 최신 OS(Debian 13 Trixie, aarch64, Python 3.13)를 **재설치하지 않고 그대로 유지**하면서, TensorFlow 1.15 구동에 필요한 Python 3.7 환경을 **Docker 컨테이너**로 안전하게 띄워서 실행하는 방법을 설명합니다.

---

## 1단계: 라즈베리파이에 Docker 설치 (호스트 터미널 작업)

현재 사용 중인 최신 라즈베리파이 OS에 컨테이너 환경을 구동하기 위한 Docker 엔진을 설치합니다. 라즈베리파이 터미널에서 다음을 실행하세요.

```bash
# 최신 패키지 업데이트
sudo apt-get update
sudo apt-get upgrade -y

# Docker 간편 설치 스크립트 실행
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# 현재 사용자(kouy956 등)를 docker 그룹에 추가하여 sudo 없이 실행 (설정 후 SSH 재접속 권장)
sudo usermod -aG docker $USER
```

---

## 2단계: 호스트에 마이크 하드웨어 확인
라즈베리파이에 USB 마이크(또는 오디오 햇)를 꽂은 뒤, 리눅스 커널에 제대로 인식되었는지 확인합니다.
```bash
arecord -l
```
> 목록에 USB 오디오 디바이스가 보이면 정상입니다. (카드 번호를 꼭 확인해 주세요. 나중에 Docker로 스피커/마이크 장치 `/dev/snd` 를 통째로 넘겨줄 예정입니다.)

### 2. 마이크 연결 확인
USB 마이크(또는 오디오 햇)를 꽂은 뒤, 인식되었는지 확인합니다.
```bash
arecord -l
```
> 목록에 USB 오디오 디바이스가 보이면 정상입니다. (카드 번호와 디바이스 번호를 기억해두세요. 보통 `hw:1,0` 형식입니다.)

---

## 3단계: Dockerfile 작성 및 빌드 (라즈베리파이 내부 작업)

라즈베리파이 안에 프로젝트 폴더를 만들고, 내장될 파이썬 3.7 컨테이너 명세서(`Dockerfile`)를 작성합니다.

```bash
mkdir -p ~/KWS_Project && cd ~/KWS_Project
nano Dockerfile
```

아래 내용을 복사하여 `Dockerfile`에 붙여넣고 저장(`Ctrl+O`, `Enter`, `Ctrl+X`)합니다.

```dockerfile
# ARM64 기반 레거시 Debian(Python 3.7 호환) 이미지 사용
FROM arm64v8/debian:buster-slim

# 마이크/오디오 의존성 패키지 설치
RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-dev \
    wget libatlas-base-dev portaudio19-dev \
    alsa-utils gcc \
    && rm -rf /var/lib/apt/lists/*

# 파이썬 pip 최신화
RUN python3 -m pip install --upgrade pip

# ARM64용 TensorFlow 1.15 Wheel 설치 (aarch64 전용 사전 빌드 파일)
RUN wget https://github.com/lhelontra/tensorflow-on-arm/releases/download/v1.15.2/tensorflow-1.15.2-cp37-none-linux_aarch64.whl
RUN pip3 install tensorflow-1.15.2-cp37-none-linux_aarch64.whl pyaudio

WORKDIR /app
CMD ["python3", "pc_mic_test.py"]
```

작성 후, 해당 디렉토리에서 Docker 이미지를 빌드합니다. (종속성 다운로드로 인해 시간이 좀 걸립니다.)
```bash
docker build -t kws-tf1.15 .
```

---

## 4단계: 파일 전송 및 Docker 컨테이너 실행

PC에서 완성된 KWS 파일들을 라즈베리파이의 `~/KWS_Project` 폴더로 전송합니다. 전송 수단으로는 터미널의 `scp` 기능을 사용합니다.

**전송해야 할 핵심 파일 2가지:**
1. `work/ds_cnn_korean_frozen.pb` (동결된 배포용 모델 원본)
2. `docker_virtual_mic_korean.py` (우리가 PC에서 완성하고 한국어에 맞게 최적화한 실시간 추론 스크립트)

*(전송 시 `.pb` 모델의 경로가 스크립트 내부 설정과 일치하도록 주의하세요. 기본 스크립트는 모델이 같은 폴더나 하위 폴더에 있는 것을 자동으로 스캔합니다.)*

### 🚀 마이크 패스스루 옵션으로 추론 실행
하드웨어 마이크 제어권(`/dev/snd`)을 Docker 안으로 넘겨주어 스크립트를 실행원합니다.

```bash
cd ~/KWS_Project

# 오디오 권한을 컨테이너 밖의 하드웨어와 연동하여 실행
docker run -it --rm --device /dev/snd -v $(pwd):/app -w /app kws-tf1.15 bash

# 내부 터미널 진입 후 스크립트 실행
python docker_virtual_mic_korean.py
```

---

## (참고) 마이크 에러 발생 시 트러블슈팅
만약 `pc_mic_test.py` 실행 시 `[Errno -9998] Invalid number of channels` 또는 `Default Input Device` 관련 에러가 발생한다면, PyAudio에 하드웨어 마이크 번호를 직접 지정해야 합니다.

`pc_mic_test.py` 파일 내의 스트리밍 세팅 라인에 `input_device_index` 부분을 추가합니다.

```python
# 수정 전
stream = p.open(format=pyaudio.paInt16, ... )

# 수정 후: input_device_index 추가 (arecord -l 로 확인한 카드 번호가 1번일 때)
stream = p.open(format=pyaudio.paInt16,
                channels=1,
                rate=SAMPLE_RATE,
                input=True,
                input_device_index=1,  # <--- 이 부분 추가 (기기 환경에 맞게 숫자 변경)
                frames_per_buffer=CHUNK_SIZE)
```
