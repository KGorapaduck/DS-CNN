# ⚠️ freeze.py 반환값 불일치 이슈

> **발생일:** 2026-02-24  
> **상태:** 해결 완료

---

## 1. 문제

`freeze.py` 실행 시 `create_model()` 반환값 개수 불일치로 에러 발생.

```python
# freeze.py 원본 (123줄) — 5개 반환값 기대
logits, net_c1, fingerprint_4d, net_avg_pool, layer_out = models.create_model(...)
```

그러나 학습에 사용한 `ds_cnn` 모델은 `is_training=False`일 때 `logits` **1개만** 반환:

```python
# models.py — create_ds_cnn_model (992줄)
return logits  # 1개
```

---

## 2. 원인

`freeze.py`는 원래 **`ds_cnn_quant`** (양자화 버전) 전용으로 작성되어 있었음.

```python
# models.py — create_ds_cnn_model_quant (1134~1135줄)
return logits, net_c1, fingerprint_4d, net_avg_pool  # 4개 반환
```

### 저자가 의도한 원래 워크플로우
```
train.py (ds_cnn, Float32 학습)
    ↓
trainTest_quant.py (ds_cnn_quant, INT8 양자화 재학습)
    ↓
freeze.py (ds_cnn_quant로 freeze)  ← freeze.py는 이 단계용
    ↓
배포
```

### 우리의 워크플로우 (1주차 빠른 검증)
```
train.py (ds_cnn, Float32 학습)
    ↓
freeze.py (ds_cnn으로 직접 freeze)  ← 반환값 불일치 발생!
    ↓
TFLite 변환 → 라즈베리파이 테스트
```

---

## 3. 왜 Float32(ds_cnn)로 먼저 학습했는가

`role_strategy.md`의 3주 마일스톤 전략에 따름:

- **1주차:** Float32 기반 가벼운 KWS 모델을 `tflite_runtime`으로 **우선 구동** (파이프라인 관통)
- **3주차:** KWS 모델 **양자화(INT8)** 적용 및 정확도 테스트 (최적화)

→ 1주차 목표 달성을 위해 Float32로 먼저 파이프라인을 검증하는 것이 올바른 순서였음.

---

## 4. 해결

`freeze.py` 123~131줄을 `ds_cnn`과 `ds_cnn_quant` **양쪽 모두 호환**되도록 수정:

```python
# 수정 후 (양쪽 호환)
model_output = models.create_model(
    reshaped_input, model_settings, model_architecture, model_size_info,
    is_training=False, runtime_settings=runtime_settings)
if isinstance(model_output, tuple):
  logits = model_output[0]
else:
  logits = model_output
tf.nn.softmax(logits, name='labels_softmax')
```

---

## 5. 교훈

- `freeze.py`는 `ds_cnn_quant` 전용으로 작성되어 있었으므로, **비양자화 모델로 사용 시 반환값 분기 처리 필요**
- 프로젝트의 전체 파이프라인 흐름(`train` → `trainTest_quant` → `freeze`)을 **사전에 파악한 후 실행**해야 이런 불일치를 예방할 수 있음
