# 📋 KWS-DS-CNN 개발 로그

> **프로젝트:** KWS-DS-CNN-for-embedded  
> **목적:** 키워드 스포팅(KWS) DS-CNN 모델 학습 및 라즈베리파이 배포 파이프라인 구축  
> **작성일:** 2026-02-24  
> **최종 수정:** 2026-02-24

---

## 1. 개발 환경

### 하드웨어
| 항목 | 사양 |
|------|------|
| GPU | NVIDIA GeForce RTX 4060 Laptop (8GB VRAM) |
| OS | Windows |
| 타깃 디바이스 | Raspberry Pi 3 B+ (1GB RAM, ARM Cortex-A53) |

### 소프트웨어 (시스템)
| 항목 | 버전 |
|------|------|
| Python (시스템) | 3.10.11 |
| Conda | 25.5.1 |
| pip (시스템) | 23.0.1 |

### 학습용 Conda 환경: `kws`
| 항목 | 버전 | 비고 |
|------|------|------|
| Python | **3.7** | TF 1.15가 Python 3.7까지만 지원 |
| TensorFlow | **1.15.0** (CPU) | `tensorflow.contrib` API 사용 필수 |
| protobuf | **3.20.0** | TF 1.15와의 호환을 위해 다운그레이드 |
| numpy | 1.21.6 | TF 1.15 설치 시 자동 설치 |
| six | 1.17.0 | TF 1.15 설치 시 자동 설치 |

#### 환경 생성 명령어
```bash
# Conda 환경 생성
conda create -n kws python=3.7 -y

# TensorFlow 및 의존성 설치
conda run -n kws pip install tensorflow==1.15.0 numpy six

# protobuf 다운그레이드 (필수!)
conda run -n kws pip install protobuf==3.20.0
```

#### 학습 실행 명령어
```bash
# Python 직접 경로로 실행 (conda run 대신)
C:/ProgramData/anaconda3/envs/kws/python.exe train.py \
  --data_url= \
  --data_dir=./speech_dataset \
  --wanted_words=yes,no \
  --model_architecture=ds_cnn \
  --model_size_info 5 64 10 4 2 2 64 3 3 1 1 64 3 3 1 1 64 3 3 1 1 64 3 3 1 1 \
  --how_many_training_steps=10000,3000 \
  --learning_rate=0.001,0.0001 \
  --window_size_ms=40 \
  --window_stride_ms=20 \
  --dct_coefficient_count=10 \
  --train_dir=./work/ds_cnn_train \
  --batch_size=50 \
  --eval_step_interval=500
```

---

## 2. 핵심 라이브러리 및 의존 관계

```
TensorFlow 1.15.0
├── tensorflow.contrib.slim          → 모델 학습 (train_op 생성)
├── tensorflow.contrib.framework     → 오디오 처리 (audio_ops)
├── tensorflow.contrib.signal        → mel-spectrogram 변환
├── tensorflow.contrib.layers        → 레이어 유틸리티
└── protobuf 3.20.0                  → 직렬화 (4.x와 비호환!)
```

> ⚠️ **`tensorflow.contrib`는 TF 2.x에서 제거됨.** 따라서 이 프로젝트는 반드시 TF 1.x에서만 실행 가능.

---

## 3. 데이터셋

| 항목 | 상세 |
|------|------|
| 이름 | Google Speech Commands v0.01 |
| 다운로드 URL | `http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz` |
| 용량 | 약 1.4GB (압축), 압축 해제 후 약 2.2GB |
| 저장 위치 | `./speech_dataset/` |
| 형식 | 16kHz, 1초, mono WAV |
| 포함 단어 | 30개 단어 클래스 + `_background_noise_/` |
| 학습에 사용한 키워드 | `yes`, `no` (+ `_silence_`, `_unknown_`) |

---

## 4. 진행 과정

### ✅ 완료된 작업

| # | 작업 | 상태 | 비고 |
|---|------|------|------|
| 1 | 프로젝트 구조 분석 | ✅ | TF 1.x 기반 DS-CNN KWS 프로젝트 확인 |
| 2 | 라즈베리파이 적합성 분석 | ✅ | `role_strategy.md` 검토, KWS 추론만 엣지에서 수행하는 분산 아키텍처 검증 |
| 3 | Git 연결 해제 | ✅ | `.git` 폴더 삭제 |
| 4 | 한국어 KWS 데이터셋 조사 | ✅ | 직접 매칭되는 공개 데이터셋 없음. AI Hub 활용 또는 직접 녹음 필요 |
| 5 | Speech Commands v0.01 다운로드 | ✅ | `./speech_dataset/`에 저장 |
| 6 | Conda `kws` 환경 생성 | ✅ | Python 3.7 + TF 1.15.0 |
| 7 | DS-CNN 모델 학습 (yes/no) | ✅ | 13,000 steps, ~35분 소요 (CPU) |
| 8 | Freeze (체크포인트 → `.pb`) | ✅ | `freeze.py` 수정 후 47개 변수 동결, `ds_cnn_frozen.pb` 생성 |

### 📊 학습 결과

| 지표 | 수치 |
|------|------|
| Final Test Accuracy | **96.72%** (N=610) |
| Best Validation Accuracy | **96.87%** |
| 모델 크기 | ~270KB |
| 학습 시간 | ~35분 (CPU) |

#### Confusion Matrix (Test)
```
              _silence_  _unknown_   yes    no
_silence_        51         0         0     0
_unknown_         0        46         3     2
yes               0         4       250     2
no                0         4         5   243
```

### 📁 산출물
- 모델 그래프: `work/ds_cnn_train/ds_cnn.pbtxt`
- 레이블: `work/ds_cnn_train/ds_cnn_labels.txt` (`_silence_`, `_unknown_`, `yes`, `no`)
- Best 체크포인트: `work/ds_cnn_train/best/` (5개, 최고 96.87%)
- Frozen Graph: `work/ds_cnn_frozen.pb` (47개 변수 동결)

---

## 5. ❌ 실패 및 트러블슈팅 기록

### 실패 1: TensorFlow 미설치 상태
- **증상:** `ModuleNotFoundError: No module named 'tensorflow'`
- **원인:** 시스템 Python 3.10에 TF 미설치
- **해결:** TF 1.x는 Python 3.7까지만 지원하므로 conda 환경 별도 생성
- **교훈:** TF 1.x 프로젝트는 반드시 Python 3.7 이하 환경 필요

### 실패 2: protobuf 버전 비호환
- **증상:** TF import 시 crash (protobuf 관련 에러)
- **원인:** pip가 자동으로 protobuf 4.24.4를 설치했으나, TF 1.15는 protobuf 3.x만 지원
- **해결:** `pip install protobuf==3.20.0`으로 다운그레이드
- **교훈:** TF 1.15 + protobuf 4.x는 비호환. **반드시 3.20.0 이하 사용**

### 실패 3: `conda run --no-banner` 미지원
- **증상:** `unrecognized arguments: --no-banner`
- **원인:** conda 25.5.1에서 `--no-banner` 옵션 제거됨
- **해결:** `--no-banner` 옵션 제거 후 재실행
- **교훈:** conda 버전에 따라 CLI 옵션이 다를 수 있음

### 실패 4: `conda run`으로 실행 시 Segmentation Fault (Exit Code 139)
- **증상:** `conda run -n kws python train.py ...` 실행 시 Exit Code 139
- **원인:** `conda run`의 stdout 래핑 문제로 TF 세션과 충돌
- **해결:** conda 환경의 Python을 직접 호출: `C:/ProgramData/anaconda3/envs/kws/python.exe train.py ...`
- **교훈:** TF 1.x 학습 시 `conda run` 대신 **Python 실행 파일 직접 경로** 사용 권장

### 실패 5: 체크포인트 저장 시 Segmentation Fault
- **증상:** Validation 후 best 모델 저장 시점에서 Segfault (Exit Code 139)
- **에러 메시지:** `경로를 찾을 수 없습니다. ./work/ds_cnn_train/best/ds_cnn_9467.ckpt-...`
- **원인:** `train.py`의 best 모델 저장 경로(`./work/ds_cnn_train/best/`)가 존재하지 않음. TF의 `saver.save()`가 디렉토리를 자동 생성하지 않아 crash 발생
- **해결:** 학습 전 디렉토리 수동 생성: `mkdir -p ./work/ds_cnn_train/best`
- **교훈:** **`train.py` 실행 전 반드시 `train_dir/best/` 디렉토리를 미리 생성해야 함**

### 실패 6: RTX 4060 GPU와 TF 1.15 비호환
- **증상:** `Could not load dynamic library 'cudart64_100.dll'`
- **원인:** TF 1.15는 CUDA 10.0을 요구하지만, RTX 4060은 CUDA 12.x 이상 드라이버 사용
- **해결:** GPU 무시하고 CPU로 학습 진행 (Warning 메시지만 출력, 학습 자체는 정상)
- **교훈:** Ampere/Ada Lovelace 이상 GPU에서는 TF 1.x GPU 사용 불가. CPU 학습 또는 TF 2.x 마이그레이션 필요

### 실패 7: `freeze.py` 반환값 불일치
- **증상:** `freeze.py` 실행 시 `create_model()` 반환값 언패킹 에러
- **원인:** `freeze.py`는 `ds_cnn_quant` (양자화) 전용으로 작성되어 5개 반환값을 기대하지만, `ds_cnn`은 `logits` 1개만 반환
- **배경:** 원래 저자의 워크플로우는 `train.py(ds_cnn)` → `trainTest_quant.py(ds_cnn_quant 재학습)` → `freeze.py(ds_cnn_quant)` 순서. 우리는 1주차 빠른 검증을 위해 Float32 `ds_cnn`으로 바로 freeze 시도
- **해결:** `isinstance(model_output, tuple)` 분기 처리로 `ds_cnn`/`ds_cnn_quant` 양쪽 호환
- **교훈:** 프로젝트의 전체 파이프라인 흐름을 사전에 파악한 후 실행해야 함. 상세: `Dev/freeze_issue.md`

### 실패 8: `toco_from_protos` 명령어 인식 불가
- **증상:** TFLite 변환 시 `'toco_from_protos' is not recognized as an internal or external command` 에러 발생
- **원인:** TF 1.15 환경(Windows)에서 `tf.lite.TFLiteConverter.from_frozen_graph` 내부적으로 호출하는 `toco_from_protos` 실행 파일이 시스템 PATH에 등록되지 않음
- **해결:** `C:/ProgramData/anaconda3/envs/kws/Scripts/toco.exe` (또는 `toco_from_protos.exe`) 바이너리를 직접 호출하는 방식으로 우회 시도

### 실패 9: TFLite 변환 시 Input Shape 불일치
- **증상:** `toco.exe` 실행 시 `Dimension 1 in both shapes must be equal, but are 490 and 240. Shapes are [1,490] and [1,240].` 에러 발생
- **원인:** 학습 시 사용한 파라미터(window_size_ms=40, window_stride_ms=20 등)로 인해 도출된 `fingerprint_size`가 우리가 변환 스크립트에 수동으로 입력한 shape(`1,240`)과 달리 `490`임
- **해결:** (진행 예정) 변환 스크립트에서 입력 shape을 `[1, 490]`으로 수정하여 재변환 필요

### 실패 10: PC 마이크 실시간 테스트 초기 오탐지(False Alarm)
- **증상:** `pc_mic_test.py` 실행 직후 약 1~2초 동안 계속해서 "no" 키워드를 80% 이상의 신뢰도로 오탐지함. 이후에는 정상적으로 `_silence_`를 출력.
- **원인:** 스크립트 실행 시 1초 분량의 오디오 버퍼를 수학적으로 완벽한 `0.0`(Zero) 배열로 초기화함. 인공지능 모델은 학습 시 이런 완벽한 무음 데이터를 본 적이 없어 예측 불가능한(Out-of-distribution) 값으로 받아들이고 오작동함.
- **해결방안:** 라즈베리파이에 이식할 때는 프로그램 시작 후 초기 1~2초(버퍼가 실제 마이크의 배경 잡음으로 완전히 채워질 때까지) 동안은 모델의 추론 결과를 무시(Bypass)하도록 예외 처리 코드를 추가해야 함.

