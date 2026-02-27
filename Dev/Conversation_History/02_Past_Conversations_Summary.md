# 이전 대화 기록 요약 목록

이 문서는 본 프로젝트 진행 과정에서 발생한 주요 대화 기록들의 요약을 포함합니다.

## 1. Raspberry Pi Deployment Success (8314c9e4)
- **내용**: 라즈베리파이에 컨테이너를 배포하고 `.pb` 모델의 추론 성능을 100% 정확도로 검증 완료함.

## 2. Data Pipeline Integration (507afe52)
- **내용**: AI 허브 데이터셋에서 '퀴즈', '이해' 키워드를 추출하여 학습 파이프라인에 통합.

## 3. Augmenting Audio Data (bf25d269)
- **내용**: 피치 시프트, 대역통과 필터 등을 사용하여 데이터 증강(Augmentation) 수행 (2500개 파일 생성).

## 4. Raspberry Pi Docker Setup (608c91f7)
- **내용**: 라즈베리파이 환경에서 TensorFlow 1.15를 실행하기 위한 전용 도커 이미지 구축.

## 5. DS-CNN Model Training & Tuning (57ee7d50, d07811cb)
- **내용**: 기존 CNN 모델을 DS-CNN(Depthwise Separable CNN)으로 변경하고 하이퍼파라미터 튜닝을 통해 최적화 수행.

## 6. KWS Model Debugging (fe6d3a09)
- **내용**: 오탐(False Positive) 방지를 위한 노이즈 데이터 증강 및 추론 로직 보정.

## 기타 기록
- **Edge UI Project Analysis** (21d1824b, 15e4dd95)
- **Database Extension Setup** (3548d971)
