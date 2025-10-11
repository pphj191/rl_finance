# SSL Features Guide

**Self-Supervised Learning 기반 특성 추출 및 미래 예측**

## 개요

`trading_env/ssl_features.py` 모듈은 기존의 규칙 기반(rule-based) 기술적 지표와는 달리, **학습 기반(learning-based)** 특성 추출을 제공합니다.

### 기존 방식 vs SSL 방식

| 방식 | 기술적 지표 (indicators.py) | SSL 특성 (ssl_features.py) |
|------|---------------------------|---------------------------|
| 접근법 | 규칙 기반 (Rule-based) | 학습 기반 (Learning-based) |
| 계산 | 수식으로 직접 계산 (SMA, RSI 등) | 신경망 모델 학습 후 추출 |
| 데이터 요구 | 실시간 계산 가능 | 사전 학습 필요 |
| 특성 | 명시적, 해석 가능 | 암묵적 표현, 강력한 패턴 인식 |
| 예시 | SMA, EMA, RSI, MACD | Representation 벡터, 미래 예측 |

---

## 주요 기능

### 1. Contrastive Learning (대조 학습)
- **목적**: 유사한 시계열 패턴을 가까운 벡터로, 다른 패턴을 먼 벡터로 학습
- **출력**: 고차원 representation 벡터 (hidden_dim)
- **활용**: 패턴 유사도, 이상 탐지

```python
# 학습 후 representation 추출
features = ssl_extractor.extract_features(df)
contrastive_repr = features['contrastive_repr']  # (seq_len, hidden_dim)
```

### 2. Masked Prediction (마스킹 예측)
- **목적**: 시계열 데이터의 일부를 마스킹하고 예측 (BERT 방식)
- **출력**: Masked prediction representation
- **활용**: 시계열의 문맥 이해

```python
masked_repr = features['masked_repr']  # (seq_len, hidden_dim)
```

### 3. Temporal Pattern Classification (시간적 패턴 분류)
- **목적**: 시계열 패턴을 8개 클래스로 분류
  - 강한 상승, 약한 상승, 횡보, 약한 하락, 강한 하락
  - 높은 변동성, 낮은 변동성, 반전 패턴
- **출력**: 각 클래스에 대한 확률 분포
- **활용**: 시장 상태 인식

```python
pattern_probs = features['pattern_probs']  # (seq_len, 8)
```

### 4. Future Price Prediction (미래 가격 예측)
- **목적**: 1분, 5분, 15분, 30분, 60분 후 가격 변화 예측
- **출력**: Multi-horizon 예측값
- **활용**: 미래 시장 예측, 보상 신호 개선

```python
future_preds = features['future_predictions']  # (seq_len, num_horizons)
# future_preds[:, 0] = 1분 후 예측
# future_preds[:, 1] = 5분 후 예측
# future_preds[:, 2] = 15분 후 예측
# ...
```

---

## 사용 방법

### 1단계: 설정 생성

```python
from trading_env.ssl_features import SSLConfig, SSLFeatureExtractor

config = SSLConfig(
    hidden_dim=128,              # 은닉층 차원
    num_layers=2,                # LSTM 레이어 수
    dropout=0.1,                 # 드롭아웃 비율

    batch_size=32,               # 배치 크기
    learning_rate=1e-3,          # 학습률
    num_epochs=100,              # 에포크 수

    temperature=0.07,            # Contrastive learning temperature
    mask_ratio=0.15,             # Masked prediction 마스킹 비율

    prediction_horizons=[1, 5, 15, 30, 60],  # 예측 시간 (분)
    model_dir="models/ssl"       # 모델 저장 경로
)
```

### 2단계: 모델 학습

```python
# SSL 추출기 생성
ssl_extractor = SSLFeatureExtractor(config, device="cuda")

# 모든 모델 학습 (최초 1회만 필요)
ssl_extractor.train_all_models(db_path="data/market_data.db")

# 또는 개별 모델 학습
# ssl_extractor.train_contrastive_model(data_loader, db_path)
# ssl_extractor.train_masked_prediction_model(data_loader, db_path)
# ssl_extractor.train_pattern_classifier(data_loader, db_path)
# ssl_extractor.train_future_predictor(data_loader, db_path)
```

**참고**: 학습 로직은 현재 TODO로 표시되어 있습니다. 구현이 필요합니다.

### 3단계: 학습된 모델 로드

```python
# 학습된 모델 로드
input_dim = 50  # 입력 특성 차원 (OHLCV + indicators)
ssl_extractor.load_all_models(input_dim=input_dim)

# 또는 개별 모델 로드
# ssl_extractor.load_model('contrastive', input_dim)
# ssl_extractor.load_model('masked', input_dim)
# ssl_extractor.load_model('pattern', input_dim)
# ssl_extractor.load_model('future', input_dim)
```

### 4단계: 특성 추출

```python
import pandas as pd

# 시장 데이터 준비 (OHLCV + indicators)
df = pd.DataFrame({
    'close': [...],
    'volume': [...],
    'rsi': [...],
    'macd': [...],
    # ... 기타 기술적 지표
})

# SSL 특성 추출
ssl_features = ssl_extractor.extract_features(df)

# 결과
print(ssl_features.keys())
# dict_keys(['contrastive_repr', 'masked_repr', 'pattern_probs', 'future_predictions'])
```

---

## DataPipeline 통합

SSL 특성을 DataPipeline에 통합하여 캐싱 및 자동 처리가 가능합니다.

### 옵션 1: DataPipeline에서 SSL 비활성화 (현재 기본값)

```python
from trading_env import DataPipeline, MarketDataStorage

storage = MarketDataStorage("data/market_data.db")
pipeline = DataPipeline(
    storage=storage,
    mode="offline",
    include_ssl=False  # SSL 특성 제외
)

# 기술적 지표만 포함
processed_data = pipeline.process_data("KRW-BTC")
```

### 옵션 2: SSL 특성 별도 추가 (권장)

```python
# 1. 기술적 지표 처리
pipeline = DataPipeline(storage, mode="offline", include_ssl=False)
processed_data = pipeline.process_data("KRW-BTC")

# 2. SSL 특성 추가
ssl_config = SSLConfig()
ssl_extractor = SSLFeatureExtractor(ssl_config, device="cpu")
ssl_extractor.load_all_models(input_dim=processed_data.shape[1])

ssl_features = ssl_extractor.extract_features(processed_data)

# 3. 결합 (RL 에이전트 입력으로 사용)
# 예: contrastive_repr만 사용
import numpy as np
combined_features = np.concatenate([
    processed_data.values,
    ssl_features['contrastive_repr']
], axis=1)
```

---

## RL 에이전트 통합 예시

```python
from trading_env import TradingEnvironment, TradingConfig
from trading_env.ssl_features import SSLFeatureExtractor, SSLConfig
import pandas as pd

# 1. 데이터 준비 (기술적 지표 포함)
# ... prepare processed_data ...

# 2. SSL 특성 추가
ssl_config = SSLConfig()
ssl_extractor = SSLFeatureExtractor(ssl_config)
ssl_extractor.load_all_models(input_dim=processed_data.shape[1])

ssl_features = ssl_extractor.extract_features(processed_data)

# 3. 특성 결합 (선택적)
# 예: contrastive representation + future predictions만 사용
enhanced_data = processed_data.copy()
enhanced_data['ssl_repr_mean'] = ssl_features['contrastive_repr'].mean(axis=1)
enhanced_data['future_1min'] = ssl_features['future_predictions'][:, 0]
enhanced_data['future_5min'] = ssl_features['future_predictions'][:, 1]

# 4. RL 환경 생성
config = TradingConfig()
env = TradingEnvironment(
    config=config,
    market="KRW-BTC",
    data=enhanced_data  # SSL 특성이 포함된 데이터
)

# 5. 학습 진행
# ... training loop ...
```

---

## 모델 아키텍처

### ContrastiveEncoder
```
Input (batch, seq_len, input_dim)
  ↓
Linear(input_dim → hidden_dim) + LayerNorm + ReLU
  ↓
LSTM(hidden_dim, num_layers=2)
  ↓
Representation (batch, hidden_dim)
  ↓
Projection Head (hidden_dim → hidden_dim/2)
```

### MaskedPredictor
```
Masked Input (batch, seq_len, input_dim)
  ↓
LSTM(input_dim → hidden_dim, num_layers=2)
  ↓
Linear(hidden_dim → input_dim)
  ↓
Predictions (batch, seq_len, input_dim)
```

### TemporalPatternClassifier
```
Input (batch, seq_len, input_dim)
  ↓
LSTM(input_dim → hidden_dim, num_layers=2)
  ↓
Linear(hidden_dim → hidden_dim/2) + ReLU
  ↓
Linear(hidden_dim/2 → 8)
  ↓
Class Logits (batch, 8)
```

### FuturePricePredictor
```
Input (batch, seq_len, input_dim)
  ↓
LSTM(input_dim → hidden_dim, num_layers=2)
  ↓
Multi-Task Heads (x5)
  ↓
Predictions (batch, 5)
  [1분, 5분, 15분, 30분, 60분 후]
```

---

## TODO: 학습 구현 필요 항목

현재 `ssl_features.py`에는 모델 아키텍처와 특성 추출 로직이 구현되어 있지만, 실제 학습 로직은 TODO로 표시되어 있습니다.

### 구현 필요 항목

1. **Data Loader 구현** (`create_data_loader`)
   - SQLite에서 데이터 읽기
   - PyTorch Dataset/DataLoader 생성
   - 배치 처리

2. **Contrastive Learning 학습** (`train_contrastive_model`)
   - Data augmentation (노이즈 추가, 시간 왜곡)
   - Positive/Negative pair 생성
   - NT-Xent loss 구현 및 최적화

3. **Masked Prediction 학습** (`train_masked_prediction_model`)
   - Random/Block masking 전략
   - MSE loss로 마스킹된 부분 예측

4. **Pattern Classification 학습** (`train_pattern_classifier`)
   - 패턴 레이블 자동 생성 (트렌드, 변동성 기반)
   - Cross-entropy loss로 분류

5. **Future Prediction 학습** (`train_future_predictor`)
   - Multi-horizon target 생성
   - Multi-task learning
   - Loss balancing

### 참고 자료
- SimCLR (Contrastive Learning): https://arxiv.org/abs/2002.05709
- BERT (Masked Prediction): https://arxiv.org/abs/1810.04805
- Time Series SSL: https://arxiv.org/abs/2106.10466

---

## 성능 고려사항

### 메모리 사용량
- 모델 크기: ~1-5MB (hidden_dim=128 기준)
- GPU 메모리: ~500MB-2GB (학습 시)
- CPU 메모리: ~100MB (추론 시)

### 속도
- 학습: ~1-10분/에포크 (데이터 크기에 따라)
- 추론: ~10-50ms/배치 (CPU 기준)

### 권장 사항
- 학습: GPU 사용 권장 (`device="cuda"`)
- 추론: CPU도 충분 (`device="cpu"`)
- 모델 저장: 학습 후 체크포인트 저장 (`_save_model`)
- 재학습: 주기적으로 재학습 (월 1회 등)

---

## FAQ

**Q: SSL 특성을 반드시 사용해야 하나요?**
A: 아니요. 기술적 지표만으로도 RL 학습이 가능합니다. SSL 특성은 선택적 추가 특성입니다.

**Q: 학습 없이 사용할 수 있나요?**
A: 아니요. SSL 특성을 사용하려면 먼저 모델을 학습해야 합니다. 학습 로직은 현재 TODO로 표시되어 구현이 필요합니다.

**Q: 어떤 SSL 특성을 사용해야 하나요?**
A: 실험을 통해 결정하는 것이 좋습니다. 초기에는 `contrastive_repr`와 `future_predictions`를 추천합니다.

**Q: 기존 indicators.py의 SSL 메서드는 어떻게 되나요?**
A: 모두 제거되었습니다. `ssl_features.py`를 사용해주세요.

---

## 마이그레이션 가이드

기존 코드에서 `indicators.extract_ssl_features()`를 사용하던 경우:

### Before (이전)
```python
from trading_env.indicators import FeatureExtractor

extractor = FeatureExtractor()
ssl_features = extractor.extract_ssl_features(df)  # ❌ 더 이상 작동하지 않음
```

### After (이후)
```python
from trading_env.ssl_features import SSLFeatureExtractor, SSLConfig

# 설정 및 모델 로드
ssl_config = SSLConfig()
ssl_extractor = SSLFeatureExtractor(ssl_config)
ssl_extractor.load_all_models(input_dim=df.shape[1])

# 특성 추출
ssl_features = ssl_extractor.extract_features(df)  # ✅ 새로운 방식
```

---

## 요약

- **indicators.py**: 규칙 기반 기술적 지표 (SMA, RSI, MACD 등)
- **ssl_features.py**: 학습 기반 representation 및 미래 예측
- **통합 방식**: 별도로 추출 후 결합 (유연성 확보)
- **학습 필요**: SSL 특성 사용 전 모델 학습 필수
- **TODO**: 학습 로직 구현 필요 (데이터 로더, 학습 루프)

더 자세한 내용은 `trading_env/ssl_features.py` 소스 코드를 참고하세요.
