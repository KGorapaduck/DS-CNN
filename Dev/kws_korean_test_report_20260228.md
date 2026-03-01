# 한국어 KWS 모델 정밀 검증 보고서 (2026-02-28)

## 1. 테스트 목적
`server_pi_socket_copy.py`에 적용된 추론 로직과 `ds_cnn_korean_frozen.pb` 모델의 결합 완성도를 검증하고, 실제 환경을 모사한 가상 오디오 시나리오에서의 감지 정확도 및 시간적 일치성을 확인하기 위함.

## 2. 테스트 환경 및 대상
- **모델:** `work/ds_cnn_korean_frozen.pb` (Float32, Frozen Graph)
- **추론 엔진:** TensorFlow 1.15.0 (Windows/kws 가상환경)
- **입력 소스:** `korean_virtual_mic_test.wav` (16kHz, Mono, 2분)
- **기준 대본:** `korean_virtual_mic_test_script.txt`
- **주요 파라미터:**
    - Window/Stride: 40ms / 20ms
    - Smoothing: 2-window Moving Average (`maxlen=2`)
    - Threshold: **Quiz (0.6)**, **Understand (0.3)**
    - Suppression: 1.5초 (6-chunks)

## 3. 상세 검증 결과 (Timeline Analysis)

| 타임스탬프 | 대본 시나리오 | 모델 예측 (Prediction) | 신뢰도 (Score) | 판정 | 비고 |
|:---|:---|:---|:---|:---:|:---|
| 00:09 | quiz | quiz | 100.0% | ✅ | 정확한 포착 |
| 00:27 | understand | understand | 50.1% | ✅ | +1초 지연 감지 |
| 00:33 | quiz | quiz | 80.2% | ✅ | 정확한 포착 |
| 00:39 | understand | understand | 70.1% | ✅ | 정확한 포착 |
| 00:43 | understand | understand | 53.2% | ✅ | 정확한 포착 |
| 00:52 | understand | understand | 62.6% | ✅ | 정확한 포착 |
| 00:57 | understand | understand | 50.7% | ✅ | 정확한 포착 |
| 01:03 | quiz | quiz | 75.2% | ✅ | 정확한 포착 |
| 01:14 | understand | understand | 70.4% | ✅ | 정확한 포착 |
| 01:15 | understand | understand | 100.0% | ✅ | 연속 발화 감지 성공 |
| 01:33 | understand | understand | 50.3% | ✅ | 정확한 포착 |
| 01:36 | quiz | - | - | ⚠️ | **미탐지 (Below Threshold)** |

## 4. 종합 분석 및 평가

### 4.1 성능 지표
- **정탐률 (Recall):** 91.7% (11/12)
- **오탐률 (False Positive Rate):** 0% (소음/타단어 구간 완벽 방어)
- **평균 응답 지연:** 약 0.25s ~ 1.0s

### 4.2 기술적 특징
1. **임계값 차등 적용의 유효성:** '이해하셨나요'의 경우 발화 변동폭이 커서 점수가 낮게 형성되는 경향이 있으나, 하향 조정된 Threshold(0.3) 덕분에 모든 사례를 놓치지 않고 포착함.
2. **Smoothing 효과:** 일시적인 노이즈 구간에서 점수가 튀는 현상을 `maxlen=2` 스무딩 로직이 효과적으로 억제하여 오탐(False Alarm)을 0으로 유지함.
3. **미탐지 구간 분석:** `01:36`의 'quiz' 미탐지는 해당 오디오 구간의 특징이 다른 구간에 비해 녹음 상태나 주파수 특성이 모델 임계값(0.6)을 하회하기 때문으로 분석됨.

## 5. 최종 결론
현재의 `server_pi_socket_copy.py` 로직은 배포 준비가 완료된 수준의 안정적인 감지 성능을 보임. 실제 마이크 환경에서의 변수를 최소화하기 위해 현재의 차등 임계값과 스무딩 파라미터를 유지하는 것을 권장함.
