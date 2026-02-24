# 📌 freeze.py 분석

> **파일:** `freeze.py` (218줄)  
> **역할:** 학습된 체크포인트(Checkpoint)를 배포용 Frozen Graph(`.pb`)로 변환  
> **작성일:** 2026-02-24

---

## 1. 핵심 개념: 체크포인트 vs Frozen Graph

### 체크포인트(Checkpoint)란?
학습 중 **모델의 가중치(weight)를 저장한 파일 세트**. 학습 재개나 추가 학습에 사용.

```
work/ds_cnn_train/best/
├── ds_cnn_9687.ckpt-7500.data-00000-of-00001   ← 가중치 데이터 (숫자들)
├── ds_cnn_9687.ckpt-7500.index                  ← 가중치 이름 ↔ 위치 매핑
├── ds_cnn_9687.ckpt-7500.meta                   ← 그래프 구조 (노드, 연산 정보)
└── checkpoint                                    ← 최신 ckpt 경로 기록
```

### 비교표

| 구분 | 체크포인트(`.ckpt`) | Frozen Graph(`.pb`) |
|------|---------------------|---------------------|
| **용도** | 학습 재개, 추가 학습 | 추론(Inference) 전용 |
| **내용** | 가중치 + 그래프 (분리됨) | 가중치가 그래프에 **내장**(frozen) |
| **파일 수** | 3~4개 | **1개** |
| **배포** | ❌ 부적합 | ✅ 모바일/임베디드용 |

---

## 2. freeze.py 내부 동작 (3단계)

```
[1. 추론용 그래프 생성] → [2. 체크포인트에서 가중치 로드] → [3. 변수→상수 변환, .pb 저장]
```

### Stage 1: 추론용 그래프 생성 (`create_inference_graph`, 55~131줄)

학습 시 사용되지 않는 노드(Dropout, 역전파 등)를 제외하고 추론 전용 파이프라인 구성:

```
WAV 입력 (wav_data)
    ↓ decode_wav
PCM 오디오 데이터
    ↓ audio_spectrogram
스펙트로그램
    ↓ MFCC 또는 log-mel
특징 벡터 (fingerprint)
    ↓ DS-CNN 모델
    ↓ softmax
키워드 확률 출력 (labels_softmax)
```

- **입력 노드:** `wav_data` (WAV 바이너리) 또는 `decoded_sample_data` (PCM float)
- **출력 노드:** `labels_softmax` (각 클래스별 확률)

### Stage 2: 체크포인트 로드 (143줄)

```python
models.load_variables_from_checkpoint(sess, FLAGS.start_checkpoint)
```

`best/` 폴더의 `.ckpt` 파일에서 학습된 가중치를 복원.

### Stage 3: Freeze 및 저장 (146~152줄)

```python
frozen_graph_def = graph_util.convert_variables_to_constants(
    sess, sess.graph_def, ['labels_softmax'])
```

모든 Variable(가중치)을 Constant(상수)로 변환 → 단일 `.pb` 파일로 저장.

---

## 3. 명령줄 인자

| 인자 | 기본값 | 설명 |
|------|--------|------|
| `--sample_rate` | 16000 | 오디오 샘플레이트 (Hz) |
| `--clip_duration_ms` | 1000 | 오디오 클립 길이 (ms) |
| `--window_size_ms` | 30.0 | 스펙트로그램 윈도우 크기 |
| `--window_stride_ms` | 10.0 | 스펙트로그램 스트라이드 |
| `--dct_coefficient_count` | 40 | MFCC 계수 수 |
| `--model_architecture` | conv | 모델 아키텍처 |
| `--model_size_info` | [128,128,128] | 모델 차원 정보 |
| `--start_checkpoint` | (필수) | 체크포인트 경로 |
| `--output_file` | (필수) | 출력 `.pb` 파일 경로 |
| `--wanted_words` | yes,no,...,go | 인식 대상 단어 |
| `--input_type` | MFCC | 입력 특징 타입 (MFCC/log-mel) |

> ⚠️ **중요:** `train.py` 학습 시 사용한 인자(`window_size_ms`, `dct_coefficient_count` 등)와 **동일한 값**을 넘겨야 함!

---

## 4. 실행 예시 (현재 프로젝트 기준)

```bash
C:/ProgramData/anaconda3/envs/kws/python.exe freeze.py \
  --wanted_words=yes,no \
  --model_architecture=ds_cnn \
  --model_size_info 5 64 10 4 2 2 64 3 3 1 1 64 3 3 1 1 64 3 3 1 1 64 3 3 1 1 \
  --window_size_ms=40 \
  --window_stride_ms=20 \
  --dct_coefficient_count=10 \
  --start_checkpoint=./work/ds_cnn_train/best/ds_cnn_9687.ckpt-7500 \
  --output_file=./work/ds_cnn_frozen.pb
```

---

## 5. 전체 배포 파이프라인에서의 위치

```
train.py        →    freeze.py      →    TFLite 변환     →    라즈베리파이
(학습)               (모델 동결)          (.pb → .tflite)       (tflite_runtime 추론)
.ckpt 생성            .pb 생성             .tflite 생성          실시간 KWS
```
