
# KWS-DS-CNN-for-embedded

본 리포지토리는 **키워드 스포팅(Keyword Spotting)**을 위한 **깊이별 분리 합성곱 신경망(Depthwise Separable Convolutional Neural Networks, DS-CNN)** 기반 TensorFlow 모델 학습용 수정된 Python 스크립트를 포함하고 있습니다. 해당 스크립트는 TensorFlow의 Speech Commands 예제를 기반으로 작성되었습니다.

또한 학습된 네트워크의 **양자화(Quantization)** 스크립트와, **단일 추론 테스트(Single-Inference Test)** 및 **연속 오디오 스트림(Continuous Audio Stream)** 환경에서의 성능 평가 스크립트도 포함되어 있습니다.

Cortex M4 기반 **FRDM K66F** 개발 보드에 사전 학습된 네트워크를 구현하기 위한 C++ 소스 코드도 함께 제공됩니다.

# 학습 및 배포 (Training and Deployment)

네트워크 학습 및 ARM Cortex-M 보드로의 배포는 [ARM의 가이드](https://github.com/ARM-software/ML-KWS-for-MCU)를 참고하시기 바랍니다.

> **⚠️ 참고:** CMSIS-NN 라이브러리를 클론(Clone)한 후, `CMSIS\NN\Source\ConvolutionFunctions\arm_depthwise_separable_conv_HWC_q7_nonsquare.c` 파일을 본 리포지토리에 포함된 수정 버전으로 교체해야 합니다.

# PC 환경 설정 및 구동 가이드 (PC Setup & Run Guide)

다른 PC 환경(예: Celeron CPU 등)에서 본 프로젝트를 원활하게 구동하기 위한 초기 설정 방법입니다.

## 1. 요구 사항 (Requirements)
- **Python 3.7** (TensorFlow 1.15 버전과의 호환성을 위해 권장합니다.)
- 필요한 패키지는 `requirements.txt`에 명시되어 있습니다.

## 2. 환경 구축 및 설치 방법
명령 프롬프트(cmd) 또는 터미널을 열고 다음 명령어를 순서대로 실행해 주십시오.

```bash
# 1. 저장소 클론 (Clone Repository)
git clone https://github.com/KGorapaduck/DS-CNN.git
cd DS-CNN

# 2. 가상 환경 생성 (권장)
# (Conda를 사용하는 경우)
conda create -n kws_pc python=3.7
conda activate kws_pc

# 3. 패키지 설치
pip install -r requirements.txt
```

> **PyAudio 설치 시 참고 사항:**
> Windows 환경에서 PyAudio 설치 중 에러가 발생할 경우, 비공식 바이너리(whl) 파일을 다운로드하여 설치하시기 바랍니다.

## 3. PC 마이크 실시간 테스트 실행
환경 설정이 완료되면 아래 스크립트를 통해 PC에 연결된 마이크를 사용하여 실시간 추론 테스트를 진행할 수 있습니다.

```bash
python pc_mic_test.py
```
- 모델 입력용으로 실시간 오디오 버퍼를 수집하며 "yes" 와 "no" 키워드를 인식하고 출력합니다.
