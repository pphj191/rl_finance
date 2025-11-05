# Trading Environment (trading_env)

강화학습 기반 트레이딩 시스템의 핵심 환경 모듈

---

## 📋 개요

`trading_env` 모듈은 강화학습 에이전트를 위한 트레이딩 환경을 제공합니다. 데이터 수집, 저장, 전처리, 지표 계산, 환경 구성까지 트레이딩 시스템에 필요한 모든 기능을 포함합니다.

이 문서는 크게 두 부분으로 구성되어 있습니다:
- **[데이터 시스템](#-데이터-시스템-상세)**: 데이터 수집, 저장, 파이프라인
- **[지표 시스템](#-지표-시스템-상세)**: 기술적 지표, SSL 특성 추출

---

## 📁 파일 구조

### 🎯 핵심 환경 파일

#### `rl_env.py`
**강화학습 트레이딩 환경**
- Gymnasium (OpenAI Gym) 기반 트레이딩 환경 구현
- 상태(state), 행동(action), 보상(reward) 정의
- 포지션 관리 및 거래 실행 시뮬레이션
- 오프라인/실시간 모드 지원

```python
from trading_env.rl_env import TradingEnvironment
from trading_env.base_env import TradingConfig

config = TradingConfig()
env = TradingEnvironment(config, market="KRW-BTC", mode="offline")
```

#### `base_env.py`
**환경 기본 설정**
- `TradingConfig`: 트레이딩 환경 설정 (초기 자금, 수수료, 윈도우 크기 등)
- `ActionSpace`: 행동 공간 정의 (매수/매도/홀드)
- 공통 설정 및 데이터 클래스 정의

---

### 💾 데이터 관리 파일

#### `data_collection.py` ✨ NEW
**통합 데이터 수집 모듈**
- DB와 API를 조합하여 데이터 제공
- DB에 없는 데이터는 API로 자동 수집 및 저장
- Multi-timeframe 데이터 수집 지원
- 타임스탬프 자동 정규화

**주요 기능:**
```python
from trading_env.data_collection import DataCollector

collector = DataCollector(db_path="data/market_data.db")

# 1. 끝시간 + 개수로 데이터 수집
df = collector.get_candles_by_count("KRW-BTC", "1m", count=100)

# 2. 시간 범위로 데이터 수집
df = collector.get_candles_by_range(
    "KRW-BTC", "1h", 
    start_time=datetime(2025, 10, 1),
    end_time=datetime(2025, 10, 10)
)

# 3. Multi-timeframe 데이터 수집
data_dict = collector.get_multi_timeframe_data(
    "KRW-BTC",
    timeframes=['1m', '1h', '1d'],
    count_per_timeframe={'1m': 100, '1h': 24, '1d': 7}
)
```

#### `data_storage.py`
**SQLite 데이터베이스 관리**
- 순수 SQLite 연동 기능 (CRUD)
- 시장 데이터 저장 및 조회
- 데이터 존재 확인 및 통계
- 타임스탬프 정규화 유틸리티

**주요 기능:**
```python
from trading_env.data_storage import MarketDataStorage

storage = MarketDataStorage(db_path="data/market_data.db")

# 데이터 조회
df = storage.load_data("KRW-BTC", "1m", start_time, end_time)

# 데이터 저장
storage.save_data("KRW-BTC", "1m", df)

# 데이터 존재 확인
has_data = storage.has_data("KRW-BTC", "1m", start_time, end_time)
```

#### `market_data.py`
**시장 데이터 수집 및 전처리**
- `UpbitDataCollector`: Upbit API를 통한 데이터 수집
- `DataNormalizer`: 데이터 정규화 (Standard, MinMax, Robust)
- 실시간 데이터 수집 지원

---

### 📊 데이터 파이프라인

#### `env_pipeline.py`
**통합 데이터 파이프라인**
- 데이터 수집 → 지표 계산 → 특성 추출 → 저장/로드
- 오프라인 및 실시간 모드 지원
- 캐싱 메커니즘으로 성능 최적화
- 데이터 해시 기반 중복 방지

**파이프라인 흐름:**
```
Upbit API → 기술적 지표 계산 → SSL 특성 추출 → 정규화 → SQLite 저장
```

**사용 예시:**
```python
from trading_env.env_pipeline import DataPipeline
from trading_env.data_storage import MarketDataStorage

storage = MarketDataStorage()
pipeline = DataPipeline(storage, mode="offline", cache_enabled=True)

# 데이터 파이프라인 실행
features = pipeline.process(
    market="KRW-BTC",
    start_time="2025-01-01",
    end_time="2025-10-12"
)
```

---

### 📈 기술적 지표 파일

#### `indicators_basic.py`
**기본 기술적 지표**
- `FeatureExtractor`: 기술적 지표 계산 및 특성 추출
- 이동평균 (SMA, EMA)
- 가격 변화율 (ROC)
- 거래량 지표
- RSI, Bollinger Bands 등

**지표 목록:**
- 가격: SMA, EMA, 상대 가격 위치
- 거래량: 거래량 변화율, 거래량 MA
- 모멘텀: ROC, RSI
- 변동성: Bollinger Bands, ATR

#### `indicators_custom.py`
**커스텀 기술적 지표**
- 개인적으로 개발한 트레이딩 지표 구현
- `pullback_index`: 눌림목 지수 (상승 추세의 일시적 하락 패턴 감지)
- 기타 실험적 지표들

**예시:**
```python
from trading_env.indicators_custom import CustomIndicators

# 눌림목 지수 계산
pullback_idx = CustomIndicators.pullback_index(
    df, 
    lookback=20, 
    pullback_threshold=0.02
)
```

#### `indicators_ssl.py`
**Self-Supervised Learning 특성 추출**
- 딥러닝 기반 representation 벡터 추출
- 미래 가격 예측을 위한 SSL 모델
- PyTorch 기반 Transformer/LSTM 모델
- 학습된 모델로부터 고차원 특성 추출

**주요 컴포넌트:**
- `SSLConfig`: SSL 모델 설정
- `TimeSeriesSSL`: 시계열 SSL 모델
- `SSLFeatureExtractor`: 학습된 모델로부터 특성 추출

**사용 예시:**
```python
from trading_env.indicators_ssl import SSLFeatureExtractor, SSLConfig

config = SSLConfig(hidden_dim=128, num_layers=2)
extractor = SSLFeatureExtractor(config)

# 모델 학습
extractor.train(train_data)

# 특성 추출
ssl_features = extractor.extract_features(test_data)
```

#### `indicators.py`
**지표 통합 인터페이스**
- 현재 비어있음 (향후 확장용)
- 모든 지표를 통합하는 인터페이스 제공 예정

---

## 🔄 데이터 흐름

### 오프라인 모드
```
1. DataCollector
   ↓ (DB 확인 → 없으면 API 호출)
2. MarketDataStorage (SQLite 저장)
   ↓
3. DataPipeline (지표 계산)
   ↓
4. FeatureExtractor (기본 지표)
   ↓
5. SSLFeatureExtractor (SSL 특성)
   ↓
6. DataNormalizer (정규화)
   ↓
7. TradingEnvironment (RL 환경)
```

### 실시간 모드
```
1. UpbitDataCollector (WebSocket)
   ↓
2. DataPipeline (실시간 처리)
   ↓
3. TradingEnvironment (실시간 거래)
```

---

## 📚 모듈 의존성

```
rl_env.py
├── base_env.py (설정)
├── market_data.py (데이터 수집/정규화)
└── env_pipeline.py
    ├── data_storage.py (DB)
    ├── data_collection.py (통합 수집) ✨ NEW
    ├── indicators_basic.py (기본 지표)
    ├── indicators_custom.py (커스텀 지표)
    └── indicators_ssl.py (SSL 특성)
```

---

## 🚀 빠른 시작

### 1. 데이터 수집 및 저장
```python
from trading_env.data_collection import DataCollector

# DataCollector 생성
collector = DataCollector(db_path="data/market_data.db")

# 데이터 수집 (DB에 없으면 자동으로 API 호출)
df = collector.get_candles_by_count(
    market="KRW-BTC",
    timeframe="1m",
    count=1000
)
```

### 2. 데이터 파이프라인 실행
```python
from trading_env.env_pipeline import DataPipeline
from trading_env.data_storage import MarketDataStorage

storage = MarketDataStorage()
pipeline = DataPipeline(storage, mode="offline")

# 특성 추출
features = pipeline.process(
    market="KRW-BTC",
    start_time="2025-01-01",
    end_time="2025-10-12"
)
```

### 3. 트레이딩 환경 생성
```python
from trading_env.rl_env import TradingEnvironment
from trading_env.base_env import TradingConfig

config = TradingConfig(
    initial_balance=1000000,
    transaction_fee=0.0005,
    lookback_window=60
)

env = TradingEnvironment(
    config=config,
    market="KRW-BTC",
    mode="offline",
    db_path="data/market_data.db"
)

# 환경 사용
obs, info = env.reset()
action = env.action_space.sample()
obs, reward, done, truncated, info = env.step(action)
```

---

## 🔧 설정 파일

### TradingConfig (base_env.py)
```python
config = TradingConfig(
    # 환경 설정
    initial_balance=1000000.0,      # 초기 자금
    max_position=1.0,                # 최대 포지션 비율
    transaction_fee=0.0005,          # 거래 수수료
    
    # 데이터 설정
    lookback_window=60,              # 과거 데이터 윈도우
    update_interval=60,              # 업데이트 간격(초)
    
    # 정규화 설정
    normalization_method="robust",   # 정규화 방법
    feature_window=252,              # rolling window
    
    # 모델 설정
    model_type="dqn",                # 모델 타입
    hidden_size=256,                 # 은닉층 크기
    num_layers=3                     # 레이어 수
)
```

---

## 📊 지원하는 타임프레임

| Timeframe | 설명 | API 엔드포인트 |
|-----------|------|----------------|
| 1m, 3m, 5m | 분봉 | /v1/candles/minutes/{unit} |
| 15m, 30m | 분봉 | /v1/candles/minutes/{unit} |
| 1h, 4h | 시간봉 | /v1/candles/minutes/60, 240 |
| 1d | 일봉 | /v1/candles/days |
| 1w | 주봉 | /v1/candles/weeks |
| 1M | 월봉 | /v1/candles/months |

---

## 🎯 주요 특징

### 1. 모듈화된 구조
- 각 파일이 명확한 책임을 가짐
- 독립적으로 사용 가능
- 테스트 및 유지보수 용이

### 2. 데이터 관리
- **data_collection.py**: DB와 API를 자동으로 조합 ✨
- **data_storage.py**: 순수 SQLite 연동
- 중복 방지 및 캐싱 지원

### 3. 지표 시스템
- 기본 지표 (indicators_basic.py)
- 커스텀 지표 (indicators_custom.py)
- SSL 기반 고급 특성 (indicators_ssl.py)

### 4. 유연한 환경
- 오프라인/실시간 모드 지원
- 다양한 정규화 방법
- 설정 기반 환경 구성

---

## 📝 TODO 및 개선 사항

상세한 리팩토링 계획은 [`data_get_TODO.md`](./data_get_TODO.md)를 참조하세요.

### 완료된 작업 ✅
- [x] data_collection.py 생성 (통합 데이터 수집)
- [x] data_storage.py 리팩토링 (순수 SQLite)
- [x] 200개 이상 데이터 자동 분할 수집
- [x] 타임스탬프 자동 정규화

### 계획된 작업 📋
- [ ] Bithumb API 연동
- [ ] 데이터 갭 자동 감지 및 보정
- [ ] 병렬 처리 지원
- [ ] 추가 커스텀 지표 개발

---

## 🔗 관련 문서

- [`data_get_TODO.md`](./data_get_TODO.md) - 데이터 수집 시스템 리팩토링 계획
- [`/reports/DATA_COLLECTION_REFACTOR_COMPLETE.md`](../reports/DATA_COLLECTION_REFACTOR_COMPLETE.md) - 리팩토링 완료 보고서

---

## 📞 문의 및 기여

문제가 발생하거나 개선 사항이 있으면 이슈를 등록해주세요.

---

**최종 업데이트**: 2025-11-05  
**버전**: 1.0.0
