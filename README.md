
# KWS-DS-CNN-for-embedded

본 리포지토리는 **키워드 스포팅(Keyword Spotting)**을 위한 **깊이별 분리 합성곱 신경망(Depthwise Separable Convolutional Neural Networks, DS-CNN)** 기반 TensorFlow 모델 학습용 수정된 Python 스크립트를 포함하고 있습니다. 해당 스크립트는 TensorFlow의 Speech Commands 예제를 기반으로 작성되었습니다.

또한 학습된 네트워크의 **양자화(Quantization)** 스크립트와, **단일 추론 테스트(Single-Inference Test)** 및 **연속 오디오 스트림(Continuous Audio Stream)** 환경에서의 성능 평가 스크립트도 포함되어 있습니다.

Cortex M4 기반 **FRDM K66F** 개발 보드에 사전 학습된 네트워크를 구현하기 위한 C++ 소스 코드도 함께 제공됩니다.

# 학습 및 배포 (Training and Deployment)

네트워크 학습 및 ARM Cortex-M 보드로의 배포는 [ARM의 가이드](https://github.com/ARM-software/ML-KWS-for-MCU)를 참고하시기 바랍니다.

> **⚠️ 참고:** CMSIS-NN 라이브러리를 클론(Clone)한 후, `CMSIS\NN\Source\ConvolutionFunctions\arm_depthwise_separable_conv_HWC_q7_nonsquare.c` 파일을 본 리포지토리에 포함된 수정 버전으로 교체해야 합니다.
