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

# 📊 데이터 시스템 상세

데이터 시스템은 시장 데이터의 수집, 저장, 관리를 담당합니다.

## 데이터 아키텍처

```
┌─────────────────────────────────────────────────────────┐
│                    DataCollector                        │
│  (통합 데이터 수집 - DB와 API 자동 조합)                  │
└──────────────┬──────────────────────────────────────────┘
               │
               ├── DB 확인 (has_data?)
               │   ├── 있음 → MarketDataStorage.load_data()
               │   └── 없음 → UpbitAPI 호출
               │
               ↓
┌──────────────────────────────────────────────────────────┐
│               MarketDataStorage                          │
│        (SQLite 기반 데이터베이스 관리)                     │
│  ┌────────────────────────────────────────────────┐     │
│  │ market_1m  │ market_1h  │ market_1d  │ ...    │     │
│  │ (타임프레임별 분리된 테이블)                     │     │
│  └────────────────────────────────────────────────┘     │
└──────────────────────────────────────────────────────────┘
               │
               ↓
┌──────────────────────────────────────────────────────────┐
│                    UpbitAPI                              │
│         (외부 거래소 API 연동)                            │
│  - 분봉: get_candles_minutes_bulk()                      │
│  - 일봉: get_candles_days_bulk()                         │
│  - 주봉: get_candles_weeks_bulk()                        │
│  - 월봉: get_candles_months_bulk()                       │
└──────────────────────────────────────────────────────────┘
```

---

## 1. DataCollector (data_collection.py)

### 핵심 개념
`DataCollector`는 **DB 우선 전략**을 사용합니다:
1. 요청받은 데이터가 DB에 있는지 확인
2. DB에 있으면 → DB에서 로드
3. DB에 없으면 → API 호출 후 DB에 저장
4. 결과 반환

### 주요 메서드

#### 1.1 개별 타임프레임 수집

##### `get_candles_by_count()`
개수 기반 데이터 수집 (가장 많이 사용)

```python
from trading_env.data_collection import DataCollector

collector = DataCollector(db_path="data/market_data.db")

# 최근 100개 캔들 수집
df = collector.get_candles_by_count(
    market="KRW-BTC",
    timeframe="1m",
    count=100,
    end_time=None,  # None이면 현재 시간
    force_api=False  # True면 DB 무시하고 API 직접 호출
)
```

**내부 동작:**
1. `end_time` 정규화 (타임프레임에 맞게 초/밀리초 제거)
2. `start_time` 계산: `end_time - (count × timeframe_interval)`
3. DB에서 `[start_time, end_time]` 범위 조회
4. 데이터가 `count`개 미만이면 API 호출
5. API 데이터를 DB에 저장 후 병합하여 반환

##### `get_candles_by_range()`
시간 범위 기반 데이터 수집

```python
from datetime import datetime

# 특정 기간의 데이터 수집
df = collector.get_candles_by_range(
    market="KRW-BTC",
    timeframe="1h",
    start_time=datetime(2025, 10, 1, 0, 0, 0),
    end_time=datetime(2025, 10, 10, 0, 0, 0),
    force_api=False
)
```

**내부 동작:**
1. 시간 범위를 바탕으로 필요한 캔들 개수 계산
2. `get_candles_by_count()` 호출

#### 1.2 Multi-Timeframe 수집

##### `get_multi_timeframe_data()`
여러 타임프레임 동시 수집

```python
data_dict = collector.get_multi_timeframe_data(
    market="KRW-BTC",
    timeframes=['1m', '1h', '1d'],
    count_per_timeframe={
        '1m': 1000,
        '1h': 168,  # 1주일
        '1d': 30
    },
    end_time=None,
    force_api=False
)

# 결과: {'1m': DataFrame, '1h': DataFrame, '1d': DataFrame}
print(f"1분봉: {len(data_dict['1m'])} rows")
print(f"1시간봉: {len(data_dict['1h'])} rows")
print(f"1일봉: {len(data_dict['1d'])} rows")
```

##### `get_multi_timeframe_data_by_range()`
시간 범위로 여러 타임프레임 동시 수집

```python
data_dict = collector.get_multi_timeframe_data_by_range(
    market="KRW-BTC",
    timeframes=['1m', '1h', '1d'],
    start_time=datetime(2025, 10, 1),
    end_time=datetime(2025, 10, 10),
    force_api=False
)
```

### 내부 헬퍼 메서드

#### `_calculate_required_count()`
시간 범위로부터 필요한 데이터 개수 계산

```python
# 예: 1시간봉, 2025-10-01 ~ 2025-10-10
# → 10일 × 24시간 = 240개 + 1
count = collector._calculate_required_count(
    timeframe="1h",
    start_time=datetime(2025, 10, 1),
    end_time=datetime(2025, 10, 10)
)
# count = 241
```

#### `_parse_timeframe_for_api()`
타임프레임을 Upbit API 파라미터로 변환

```python
# 입력: '1m' → 출력: ('minutes', 1)
# 입력: '1h' → 출력: ('minutes', 60)
# 입력: '1d' → 출력: ('days', None)
# 입력: '1w' → 출력: ('weeks', None)
candle_type, unit = collector._parse_timeframe_for_api('1h')
# candle_type = 'minutes', unit = 60
```

#### `_fetch_from_api()`
API에서 실제 데이터 수집

```python
# 내부적으로 호출됨 (직접 호출 불가)
# 1. API 파라미터 파싱
# 2. 적절한 Upbit API 메서드 호출
# 3. DataFrame 변환 및 정규화
# 4. 중복 제거 및 시간 순 정렬
```

### 타임스탬프 정규화

모든 시간 데이터는 타임프레임에 맞게 정규화됩니다:

```python
from trading_env.data_storage import align_timestamp

# 1분봉: 초/밀리초 제거
dt = datetime(2025, 10, 9, 14, 23, 45, 123456)
aligned = align_timestamp(dt, '1m')
# → 2025-10-09 14:23:00

# 1시간봉: 분/초/밀리초 제거
aligned = align_timestamp(dt, '1h')
# → 2025-10-09 14:00:00

# 1일봉: 시/분/초/밀리초 제거
aligned = align_timestamp(dt, '1d')
# → 2025-10-09 00:00:00
```

### 지원하는 타임프레임

| 타임프레임 | 설명 | 분 단위 | API 엔드포인트 |
|-----------|------|---------|----------------|
| `1m` | 1분봉 | 1 | `/v1/candles/minutes/1` |
| `3m` | 3분봉 | 3 | `/v1/candles/minutes/3` |
| `5m` | 5분봉 | 5 | `/v1/candles/minutes/5` |
| `10m` | 10분봉 | 10 | `/v1/candles/minutes/10` |
| `15m` | 15분봉 | 15 | `/v1/candles/minutes/15` |
| `30m` | 30분봉 | 30 | `/v1/candles/minutes/30` |
| `60m`, `1h` | 1시간봉 | 60 | `/v1/candles/minutes/60` |
| `240m`, `4h` | 4시간봉 | 240 | `/v1/candles/minutes/240` |
| `1d` | 1일봉 | 1440 | `/v1/candles/days` |
| `1w` | 1주봉 | 10080 | `/v1/candles/weeks` |
| `1M` | 1월봉 | 43200 | `/v1/candles/months` |

---

## 2. MarketDataStorage (data_storage.py)

### 핵심 개념
`MarketDataStorage`는 **순수 SQLite 연동 기능**만 제공합니다:
- 데이터 CRUD (Create, Read, Update, Delete)
- 타임프레임별 분리된 테이블 관리
- 인덱스 최적화로 빠른 조회

### 데이터베이스 구조

#### 테이블 스키마
타임프레임마다 별도 테이블로 분리:

```sql
-- 1분봉 테이블
CREATE TABLE market_1m (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    market TEXT NOT NULL,                 -- 예: 'KRW-BTC'
    timestamp TEXT NOT NULL,              -- ISO 형식: '2025-10-09 14:23:00'
    open REAL,
    high REAL,
    low REAL,
    close REAL,
    volume REAL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(market, timestamp)            -- 중복 방지
);

-- 인덱스 (빠른 조회)
CREATE INDEX idx_market_1m_market_timestamp ON market_1m(market, timestamp);
CREATE INDEX idx_market_1m_timestamp ON market_1m(timestamp);

-- 1시간봉, 1일봉 등도 동일한 구조
CREATE TABLE market_1h (...);
CREATE TABLE market_1d (...);
```

**테이블 분리 이유:**
- 각 타임프레임의 데이터 양이 매우 다름 (1분봉 >> 1일봉)
- 쿼리 성능 향상 (WHERE 조건에 timeframe 불필요)
- 인덱스 효율성 증가

### 주요 메서드

#### 2.1 데이터 조회 (Read)

##### `load_data()`
시장 데이터 조회

```python
from trading_env.data_storage import MarketDataStorage
from datetime import datetime

storage = MarketDataStorage(db_path="data/market_data.db")

# 특정 시간 범위 조회
df = storage.load_data(
    market="KRW-BTC",
    timeframe="1m",
    start_time=datetime(2025, 10, 1, 0, 0, 0),
    end_time=datetime(2025, 10, 2, 0, 0, 0),
    limit=None  # 최대 개수 제한 (None이면 전체)
)

# 반환: DataFrame (timestamp가 index)
print(df.head())
#                      open     high      low    close    volume
# timestamp
# 2025-10-01 00:00:00  50000    50100    49900  50050    123.45
# 2025-10-01 00:01:00  50050    50200    50000  50150    234.56
# ...
```

##### `get_data_range()`
저장된 데이터의 시간 범위 조회

```python
start, end = storage.get_data_range(
    market="KRW-BTC",
    timeframe="1m"
)

if start and end:
    print(f"데이터 범위: {start} ~ {end}")
else:
    print("데이터 없음")
```

##### `get_data_count()`
저장된 데이터 개수 조회

```python
count = storage.get_data_count(
    market="KRW-BTC",
    timeframe="1m",
    start_time=datetime(2025, 10, 1),
    end_time=datetime(2025, 10, 10)
)
print(f"데이터 개수: {count}")
```

##### `has_data()`
특정 시간 범위의 데이터 존재 여부 확인

```python
exists = storage.has_data(
    market="KRW-BTC",
    timeframe="1m",
    start_time=datetime(2025, 10, 1),
    end_time=datetime(2025, 10, 10)
)
print(f"데이터 존재: {exists}")  # True or False
```

#### 2.2 데이터 저장 (Create/Update)

##### `save_data()`
시장 데이터 저장 (대량 삽입)

```python
import pandas as pd

# DataFrame 준비 (timestamp가 index 또는 컬럼)
df = pd.DataFrame({
    'timestamp': pd.date_range('2025-10-01', periods=100, freq='1min'),
    'open': [50000] * 100,
    'high': [50100] * 100,
    'low': [49900] * 100,
    'close': [50050] * 100,
    'volume': [100.0] * 100
})

# 저장
saved_count = storage.save_data(
    market="KRW-BTC",
    timeframe="1m",
    df=df,
    replace=True  # True면 중복 시 교체, False면 무시
)
print(f"{saved_count}개 저장됨")
```

**내부 동작:**
1. DataFrame 준비 (index가 timestamp면 컬럼으로 변환)
2. timestamp를 ISO 형식 문자열로 변환
3. `INSERT OR REPLACE` (replace=True) 또는 `INSERT OR IGNORE` (replace=False)

##### `update_data()`
특정 타임스탬프의 데이터 업데이트

```python
updated = storage.update_data(
    market="KRW-BTC",
    timeframe="1m",
    timestamp=datetime(2025, 10, 1, 0, 0, 0),
    close=50100,  # 종가 업데이트
    volume=150.0  # 거래량 업데이트
)
print(f"업데이트 성공: {updated}")
```

#### 2.3 데이터 삭제 (Delete)

##### `delete_data()`
데이터 삭제

```python
deleted_count = storage.delete_data(
    market="KRW-BTC",
    timeframe="1m",
    start_time=datetime(2025, 10, 1),
    end_time=datetime(2025, 10, 2)
)
print(f"{deleted_count}개 삭제됨")
```

#### 2.4 유틸리티

##### `get_available_markets()`
DB에 저장된 모든 마켓 목록 조회

```python
markets = storage.get_available_markets()
print(markets)  # ['KRW-BTC', 'KRW-ETH', 'KRW-XRP', ...]
```

##### `get_available_timeframes()`
특정 마켓의 사용 가능한 타임프레임 목록

```python
timeframes = storage.get_available_timeframes("KRW-BTC")
print(timeframes)  # ['1m', '1h', '1d']
```

##### `get_database_stats()`
데이터베이스 통계 정보

```python
stats = storage.get_database_stats()
print(f"총 행 개수: {stats['total_rows']}")

for stat in stats['market_stats']:
    print(f"{stat['market']} {stat['timeframe']}: "
          f"{stat['count']}개 ({stat['start']} ~ {stat['end']})")
```

### 성능 최적화

#### 인덱스 전략
1. **복합 인덱스**: `(market, timestamp)` - 가장 자주 사용되는 조회 패턴
2. **단독 인덱스**: `timestamp` - 시간 범위 조회

#### 트랜잭션
- 모든 쓰기 작업은 트랜잭션으로 처리
- 대량 삽입 시 일괄 커밋으로 성능 향상

#### 중복 방지
- `UNIQUE(market, timestamp)` 제약 조건
- 동일한 데이터 재저장 시 자동으로 무시 또는 교체

---

## 3. DataPipeline (env_pipeline.py)

### 핵심 개념
`DataPipeline`은 **데이터 수집 → 지표 계산 → 특성 추출**의 전체 흐름을 관리합니다.

### 파이프라인 흐름

```
1. 데이터 수집 (DataCollector)
   ↓
2. 기본 지표 계산 (FeatureExtractor)
   ↓
3. SSL 특성 추출 (SSLFeatureExtractor) [선택적]
   ↓
4. 데이터 정규화 (DataNormalizer)
   ↓
5. 캐싱 및 저장 (선택적)
```

### 사용 예시

```python
from trading_env.env_pipeline import DataPipeline
from trading_env.data_storage import MarketDataStorage

storage = MarketDataStorage(db_path="data/market_data.db")
pipeline = DataPipeline(
    storage=storage,
    mode="offline",  # 'offline' 또는 'realtime'
    cache_enabled=True
)

# 데이터 처리
features = pipeline.process(
    market="KRW-BTC",
    start_time="2025-01-01",
    end_time="2025-10-12"
)

print(features.columns)
# ['open', 'high', 'low', 'close', 'volume',
#  'sma_5', 'sma_20', 'rsi', 'macd', ...]
```

### 캐싱 메커니즘
- 동일한 시간 범위의 데이터를 재요청할 때 캐시에서 반환
- 데이터 해시 기반 중복 방지
- 메모리 효율적인 캐시 관리

---

## 4. 데이터 시스템 사용 예제

### 예제 1: 오프라인 학습용 데이터 준비

```python
from trading_env.data_collection import DataCollector
from datetime import datetime

# 1. DataCollector 생성
collector = DataCollector(db_path="data/market_data.db")

# 2. 학습용 데이터 수집 (과거 1년)
df = collector.get_candles_by_range(
    market="KRW-BTC",
    timeframe="1m",
    start_time=datetime(2024, 1, 1),
    end_time=datetime(2025, 1, 1)
)

print(f"수집된 데이터: {len(df)} rows")
print(f"기간: {df.index[0]} ~ {df.index[-1]}")

# 3. 데이터 확인
print(df.describe())
```

### 예제 2: Multi-Timeframe 데이터 수집

```python
# 여러 타임프레임 동시 수집
data_dict = collector.get_multi_timeframe_data(
    market="KRW-BTC",
    timeframes=['1m', '5m', '1h', '1d'],
    count_per_timeframe={
        '1m': 1440,   # 1일
        '5m': 288,    # 1일
        '1h': 168,    # 1주
        '1d': 365     # 1년
    }
)

for tf, df in data_dict.items():
    print(f"{tf}: {len(df)} rows, {df.index[0]} ~ {df.index[-1]}")
```

### 예제 3: 데이터베이스 직접 조작

```python
from trading_env.data_storage import MarketDataStorage

storage = MarketDataStorage(db_path="data/market_data.db")

# 통계 확인
stats = storage.get_database_stats()
print(f"총 데이터: {stats['total_rows']} rows")

# 특정 마켓의 데이터 범위 확인
start, end = storage.get_data_range("KRW-BTC", "1m")
print(f"KRW-BTC 1분봉 범위: {start} ~ {end}")

# 데이터 조회
df = storage.load_data(
    market="KRW-BTC",
    timeframe="1m",
    start_time=start,
    end_time=end,
    limit=1000  # 최근 1000개만
)
```

### 예제 4: 데이터 갭 확인 및 수정

```python
# 1. 저장된 데이터 확인
df = storage.load_data("KRW-BTC", "1m", start_time, end_time)

# 2. 타임스탬프 연속성 확인
time_diff = df.index.to_series().diff()
gaps = time_diff[time_diff > pd.Timedelta('1min')]

if len(gaps) > 0:
    print(f"발견된 갭: {len(gaps)}개")
    for gap_time, gap_size in gaps.items():
        print(f"  {gap_time}: {gap_size}")

    # 3. 갭 메우기 (API 재수집)
    for gap_time in gaps.index:
        df_fill = collector.get_candles_by_range(
            market="KRW-BTC",
            timeframe="1m",
            start_time=gap_time - pd.Timedelta('1hour'),
            end_time=gap_time + pd.Timedelta('1hour'),
            force_api=True  # API에서 강제 수집
        )
        print(f"갭 메움: {len(df_fill)} rows")
```

---

# 📈 지표 시스템 상세

지표 시스템은 시장 데이터로부터 다양한 기술적 지표와 머신러닝 특성을 추출합니다.

## 지표 아키텍처

```
┌──────────────────────────────────────────────────────┐
│              Raw Market Data (OHLCV)                 │
│         (timestamp, open, high, low, close, volume)  │
└────────────────────┬─────────────────────────────────┘
                     │
         ┌───────────┴───────────┐
         │                       │
┌────────▼───────────┐  ┌────────▼──────────────────┐
│ FeatureExtractor   │  │ CustomIndicators          │
│ (기본 기술적 지표)   │  │ (커스텀 지표)              │
│ - SMA, EMA         │  │ - pullback_index          │
│ - RSI, MACD        │  │ - 기타 실험적 지표          │
│ - Bollinger Bands  │  │                           │
│ - Stochastic       │  │                           │
│ - ATR, OBV         │  │                           │
└────────┬───────────┘  └────────┬──────────────────┘
         │                       │
         └───────────┬───────────┘
                     │
         ┌───────────▼────────────┐
         │ SSLFeatureExtractor    │
         │ (딥러닝 기반 특성)       │
         │ - Contrastive Repr.    │
         │ - Masked Prediction    │
         │ - Pattern Classification│
         │ - Future Prediction    │
         └────────────┬───────────┘
                      │
         ┌────────────▼───────────┐
         │   Combined Features    │
         │  (통합 특성 벡터)        │
         └────────────────────────┘
```

---

## 1. FeatureExtractor (indicators_basic.py)

### 핵심 개념
기본 기술적 지표를 계산하고 특성 벡터를 생성합니다.
모든 지표는 **수동 계산**으로 구현되어 외부 라이브러리 의존성이 없습니다.

### 주요 메서드

#### 1.1 지표 계산

##### `extract_technical_indicators()`
모든 기술적 지표를 한 번에 계산

```python
from trading_env.indicators_basic import FeatureExtractor
import pandas as pd

extractor = FeatureExtractor()

# 원본 OHLCV 데이터
df = pd.DataFrame({
    'timestamp': pd.date_range('2025-01-01', periods=1000, freq='1min'),
    'open': [...],
    'high': [...],
    'low': [...],
    'close': [...],
    'volume': [...]
})

# 지표 계산
features = extractor.extract_technical_indicators(df)

# 추가된 지표들
print(features.columns)
# ['open', 'high', 'low', 'close', 'volume',
#  'sma_5', 'sma_20', 'sma_60',
#  'ema_12', 'ema_26',
#  'bb_upper', 'bb_middle', 'bb_lower', 'bb_width',
#  'rsi', 'macd', 'macd_signal', 'macd_histogram',
#  'stoch_k', 'stoch_d',
#  'volume_sma', 'obv',
#  'atr',
#  'price_change_1', 'price_change_5', 'price_change_20']
```

### 계산되는 지표들

#### 1.1.1 가격 지표

##### 이동평균 (Moving Averages)
```python
# 단순이동평균 (SMA)
features['sma_5'] = close_prices.rolling(window=5).mean()
features['sma_20'] = close_prices.rolling(window=20).mean()
features['sma_60'] = close_prices.rolling(window=60).mean()

# 지수이동평균 (EMA)
features['ema_12'] = close_prices.ewm(span=12).mean()
features['ema_26'] = close_prices.ewm(span=26).mean()
```

**용도:**
- 추세 확인 (가격이 이동평균 위/아래)
- 지지/저항선
- 골든크로스/데드크로스 (단기 MA가 장기 MA 교차)

##### 볼린저 밴드 (Bollinger Bands)
```python
bb_period = 20
bb_std = 2
sma_20 = close_prices.rolling(window=bb_period).mean()
std_20 = close_prices.rolling(window=bb_period).std()

features['bb_upper'] = sma_20 + (std_20 * bb_std)   # 상단 밴드
features['bb_middle'] = sma_20                      # 중심선
features['bb_lower'] = sma_20 - (std_20 * bb_std)   # 하단 밴드
features['bb_width'] = (upper - lower) / middle     # 밴드 너비 (변동성)
```

**용도:**
- 과매수/과매도 판단 (가격이 상단/하단 밴드 근처)
- 변동성 측정 (밴드 너비)
- 추세 전환 신호

#### 1.1.2 모멘텀 지표

##### RSI (Relative Strength Index)
```python
rsi_period = 14
delta = close_prices.diff()

# 상승분과 하락분 분리
gain = delta.copy()
loss = delta.copy()
gain[gain < 0] = 0
loss[loss > 0] = 0
loss = -loss

# 평균 계산
gain_avg = gain.rolling(window=rsi_period).mean()
loss_avg = loss.rolling(window=rsi_period).mean()

# RSI 계산
rs = gain_avg / loss_avg
features['rsi'] = 100 - (100 / (1 + rs))
```

**해석:**
- RSI > 70: 과매수 (overbought)
- RSI < 30: 과매도 (oversold)
- 50 기준선: 상승/하락 추세

##### MACD (Moving Average Convergence Divergence)
```python
exp1 = close_prices.ewm(span=12).mean()  # 단기 EMA
exp2 = close_prices.ewm(span=26).mean()  # 장기 EMA

features['macd'] = exp1 - exp2                          # MACD 선
features['macd_signal'] = macd.ewm(span=9).mean()       # 시그널 선
features['macd_histogram'] = macd - macd_signal         # 히스토그램
```

**용도:**
- MACD가 시그널선 상향 돌파: 매수 신호
- MACD가 시그널선 하향 돌파: 매도 신호
- 히스토그램 크기: 추세 강도

##### 스토캐스틱 (Stochastic Oscillator)
```python
k_period = 14
lowest_low = low_prices.rolling(window=k_period).min()
highest_high = high_prices.rolling(window=k_period).max()

# %K 라인
features['stoch_k'] = 100 * ((close - lowest_low) / (highest_high - lowest_low))

# %D 라인 (K의 이동평균)
features['stoch_d'] = stoch_k.rolling(window=3).mean()
```

**해석:**
- %K > 80: 과매수
- %K < 20: 과매도
- %K와 %D의 교차: 매매 신호

#### 1.1.3 거래량 지표

##### 거래량 이동평균
```python
features['volume_sma'] = volume.rolling(window=20).mean()
```

**용도:**
- 거래량이 평균 이상: 강한 추세
- 거래량이 평균 이하: 약한 추세

##### OBV (On-Balance Volume)
```python
obv = [0.0]
for i in range(1, len(df)):
    if close_prices.iloc[i] > close_prices.iloc[i-1]:
        obv.append(obv[-1] + volume.iloc[i])      # 상승 시 거래량 더함
    elif close_prices.iloc[i] < close_prices.iloc[i-1]:
        obv.append(obv[-1] - volume.iloc[i])      # 하락 시 거래량 뺌
    else:
        obv.append(obv[-1])                       # 보합 시 유지

features['obv'] = obv
```

**용도:**
- OBV 상승 + 가격 상승: 강한 상승 추세
- OBV 하락 + 가격 하락: 강한 하락 추세
- OBV와 가격의 다이버전스: 추세 전환 신호

#### 1.1.4 변동성 지표

##### ATR (Average True Range)
```python
# True Range 계산
high_low = high_prices - low_prices
high_close = np.abs(high_prices - close_prices.shift())
low_close = np.abs(low_prices - close_prices.shift())

true_range = pd.DataFrame({
    'hl': high_low,
    'hc': high_close,
    'lc': low_close
}).max(axis=1)

# ATR (True Range의 이동평균)
features['atr'] = true_range.rolling(window=14).mean()
```

**용도:**
- 변동성 측정
- 손절/익절 레벨 설정 (예: ATR의 2배)
- 포지션 크기 조정

#### 1.1.5 가격 변화율

```python
features['price_change_1'] = close_prices.pct_change(1)    # 1분 전 대비
features['price_change_5'] = close_prices.pct_change(5)    # 5분 전 대비
features['price_change_20'] = close_prices.pct_change(20)  # 20분 전 대비
```

**용도:**
- 단기/중기 모멘텀 측정
- 급등/급락 감지

### 1.2 시퀀스 데이터 준비

#### `create_time_windows()`
슬라이딩 윈도우 생성 (RNN/LSTM용)

```python
# 60분 윈도우, 1분 step
windows = extractor.create_time_windows(
    data=features,
    window_size=60,
    step_size=1
)

print(f"생성된 윈도우 개수: {len(windows)}")
# 각 윈도우는 (60, num_features) 크기의 DataFrame
```

#### `prepare_sequence_data()`
시퀀스 데이터 변환 (X, y)

```python
# LSTM 학습용 데이터 준비
X, y = extractor.prepare_sequence_data(
    data=features,
    sequence_length=60,
    target_columns=['close']  # 예측 대상
)

print(X.shape)  # (num_samples, 60, num_features)
print(y.shape)  # (num_samples, 1)
```

### 1.3 특성 벡터 추출

#### `get_feature_vector()`
DataFrame을 numpy array로 변환

```python
feature_vector, feature_names = extractor.get_feature_vector(
    df=features,
    exclude_columns=['timestamp', 'id']  # 제외할 컬럼
)

print(feature_vector.shape)  # (num_samples, num_features)
print(feature_names)  # ['open', 'high', ..., 'rsi', 'macd', ...]
```

---

## 2. CustomIndicators (indicators_custom.py)

### 핵심 개념
개인적으로 개발한 커스텀 지표를 구현합니다.

### 주요 지표

#### 2.1 Pullback Index (눌림목 지수)

```python
from trading_env.indicators_custom import CustomIndicators

# 눌림목 지수 계산
pullback_idx = CustomIndicators.pullback_index(
    df=df,
    lookback=20,              # 추세 판단 기간
    pullback_threshold=0.02   # 하락 임계값 (2%)
)
```

**개념:**
- 상승 추세 중 일시적 하락 (눌림목) 감지
- 매수 타이밍 포착

**계산 방법:**
1. 과거 N일 추세 확인 (SMA 기울기)
2. 최근 하락폭 계산
3. 거래량 확인
4. 종합 점수 산출

---

## 3. SSLFeatureExtractor (indicators_ssl.py)

### 핵심 개념
딥러닝 기반 Self-Supervised Learning으로 고차원 특성을 추출합니다.
**현재 모델 구조만 정의되어 있으며, 학습 로직은 TODO 상태입니다.**

### SSL 모델 종류

#### 3.1 Contrastive Encoder
**목적:** 유사한 패턴은 가까운 벡터로, 다른 패턴은 먼 벡터로 매핑

```python
from trading_env.indicators_ssl import SSLFeatureExtractor, SSLConfig

config = SSLConfig(
    hidden_dim=128,
    num_layers=2,
    learning_rate=1e-3
)

extractor = SSLFeatureExtractor(config, device="cuda")

# TODO: 모델 학습 (아직 구현 안 됨)
# extractor.train_contrastive_model(data_loader, db_path="data/market_data.db")

# 특성 추출 (학습된 모델 필요)
# features = extractor.extract_features(df)
# contrastive_repr = features['contrastive_repr']  # (seq_len, hidden_dim)
```

**학습 방법 (TODO):**
1. Data augmentation (노이즈 추가, 시간 왜곡)
2. Positive pair 생성 (같은 데이터의 augmented versions)
3. Negative pair 생성 (다른 시간대 데이터)
4. NT-Xent loss 최소화

#### 3.2 Masked Predictor
**목적:** BERT-style masked prediction for time series

```python
# TODO: 구현 필요
# extractor.train_masked_prediction_model(data_loader, db_path)

# 특성 추출
# masked_repr = features['masked_repr']  # (seq_len, hidden_dim)
```

**학습 방법 (TODO):**
1. 랜덤하게 일부 timestep 마스킹 (15%)
2. 마스킹된 부분 예측
3. MSE loss 최소화

#### 3.3 Pattern Classifier
**목적:** 시계열 패턴 분류 (상승, 하락, 횡보, 변동성 등)

```python
# TODO: 구현 필요
# extractor.train_pattern_classifier(data_loader, db_path)

# 패턴 분류
# pattern_probs = features['pattern_probs']  # (seq_len, 8)
```

**패턴 클래스:**
0. 강한 상승 (strong uptrend)
1. 약한 상승 (weak uptrend)
2. 횡보 (sideways)
3. 약한 하락 (weak downtrend)
4. 강한 하락 (strong downtrend)
5. 높은 변동성 (high volatility)
6. 낮은 변동성 (low volatility)
7. 반전 패턴 (reversal pattern)

#### 3.4 Future Predictor
**목적:** Multi-horizon 미래 가격 예측

```python
# TODO: 구현 필요
# extractor.train_future_predictor(data_loader, db_path)

# 미래 예측
# future_preds = features['future_predictions']  # (seq_len, num_horizons)
# 각 horizon: [1분, 5분, 15분, 30분, 60분 후 가격 변화율]
```

### SSL 모델 구조

#### ContrastiveEncoder
```python
class ContrastiveEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2):
        # 1. Linear encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # 2. LSTM for temporal encoding
        self.lstm = nn.LSTM(
            hidden_dim, hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1
        )

        # 3. Projection head for contrastive learning
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )
```

#### MaskedPredictor
```python
class MaskedPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2):
        # LSTM encoder
        self.encoder = nn.LSTM(
            input_dim, hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )

        # Decoder to reconstruct masked values
        self.decoder = nn.Linear(hidden_dim, input_dim)
```

#### TemporalPatternClassifier
```python
class TemporalPatternClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_classes=8):
        # LSTM encoder
        self.encoder = nn.LSTM(
            input_dim, hidden_dim,
            num_layers=2,
            batch_first=True
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_classes)
        )
```

#### FuturePricePredictor
```python
class FuturePricePredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_horizons=5):
        # LSTM encoder
        self.encoder = nn.LSTM(
            input_dim, hidden_dim,
            num_layers=2,
            batch_first=True
        )

        # Multi-task prediction heads (각 horizon마다 별도 head)
        self.prediction_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1)  # 가격 변화율 예측
            )
            for _ in range(num_horizons)
        ])
```

### 모델 사용 예시 (학습 완료 후)

```python
# 1. 설정
config = SSLConfig(
    hidden_dim=128,
    num_layers=2,
    prediction_horizons=[1, 5, 15, 30, 60]  # 분 단위
)

extractor = SSLFeatureExtractor(config, device="cuda")

# 2. 모델 학습 (TODO: 아직 구현 안 됨)
# extractor.train_all_models(db_path="data/market_data.db")

# 3. 모델 로드
extractor.load_all_models(input_dim=100)  # 특성 개수

# 4. 특성 추출
features = extractor.extract_features(df)

# 5. 결과 확인
if 'contrastive_repr' in features:
    print(f"Contrastive representation: {features['contrastive_repr'].shape}")
if 'pattern_probs' in features:
    print(f"Pattern probabilities: {features['pattern_probs'].shape}")
if 'future_predictions' in features:
    print(f"Future predictions: {features['future_predictions'].shape}")
```

---

## 4. 지표 시스템 사용 예제

### 예제 1: 기본 지표 계산

```python
from trading_env.indicators_basic import FeatureExtractor
from trading_env.data_collection import DataCollector

# 1. 데이터 수집
collector = DataCollector()
df = collector.get_candles_by_count(
    market="KRW-BTC",
    timeframe="1m",
    count=1000
)

# 2. 지표 계산
extractor = FeatureExtractor()
features = extractor.extract_technical_indicators(df)

# 3. 결과 확인
print(features[['close', 'sma_20', 'rsi', 'macd']].tail())

# 4. 시각화 (matplotlib 사용)
import matplotlib.pyplot as plt

plt.figure(figsize=(15, 10))

# 가격과 이동평균
plt.subplot(3, 1, 1)
plt.plot(features.index, features['close'], label='Close')
plt.plot(features.index, features['sma_20'], label='SMA 20')
plt.plot(features.index, features['ema_12'], label='EMA 12')
plt.legend()
plt.title('Price and Moving Averages')

# RSI
plt.subplot(3, 1, 2)
plt.plot(features.index, features['rsi'], label='RSI')
plt.axhline(y=70, color='r', linestyle='--', label='Overbought')
plt.axhline(y=30, color='g', linestyle='--', label='Oversold')
plt.legend()
plt.title('RSI')

# MACD
plt.subplot(3, 1, 3)
plt.plot(features.index, features['macd'], label='MACD')
plt.plot(features.index, features['macd_signal'], label='Signal')
plt.bar(features.index, features['macd_histogram'], label='Histogram', alpha=0.3)
plt.legend()
plt.title('MACD')

plt.tight_layout()
plt.show()
```

### 예제 2: 특성 벡터 생성 (강화학습용)

```python
# 1. 지표 계산
features = extractor.extract_technical_indicators(df)

# 2. NaN 제거 (초기 지표 계산 불가 구간)
features = features.dropna()

# 3. 특성 벡터 추출
feature_vector, feature_names = extractor.get_feature_vector(
    df=features,
    exclude_columns=['timestamp']
)

print(f"Feature vector shape: {feature_vector.shape}")
print(f"Feature names: {feature_names}")

# 4. 정규화 (옵션)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
normalized_features = scaler.fit_transform(feature_vector)

print(f"Normalized features shape: {normalized_features.shape}")
```

### 예제 3: 시퀀스 데이터 준비 (LSTM용)

```python
# 1. 지표 계산
features = extractor.extract_technical_indicators(df)
features = features.dropna()

# 2. 시퀀스 데이터 생성
X, y = extractor.prepare_sequence_data(
    data=features,
    sequence_length=60,  # 60분 윈도우
    target_columns=['close']  # 종가 예측
)

print(f"X shape: {X.shape}")  # (num_samples, 60, num_features)
print(f"y shape: {y.shape}")  # (num_samples, 1)

# 3. Train/Test split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False  # 시계열은 shuffle=False
)

# 4. PyTorch 텐서 변환
import torch

X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.FloatTensor(y_train)
X_test_tensor = torch.FloatTensor(X_test)
y_test_tensor = torch.FloatTensor(y_test)
```

### 예제 4: 매매 신호 생성

```python
# 1. 지표 계산
features = extractor.extract_technical_indicators(df)

# 2. 매매 신호 로직
def generate_signals(features):
    signals = pd.DataFrame(index=features.index)
    signals['price'] = features['close']

    # RSI 기반 신호
    signals['rsi_oversold'] = features['rsi'] < 30
    signals['rsi_overbought'] = features['rsi'] > 70

    # MACD 기반 신호
    signals['macd_buy'] = (features['macd'] > features['macd_signal']) & \
                          (features['macd'].shift(1) <= features['macd_signal'].shift(1))
    signals['macd_sell'] = (features['macd'] < features['macd_signal']) & \
                           (features['macd'].shift(1) >= features['macd_signal'].shift(1))

    # 볼린저 밴드 기반 신호
    signals['bb_buy'] = features['close'] < features['bb_lower']
    signals['bb_sell'] = features['close'] > features['bb_upper']

    # 종합 매수/매도 신호
    signals['buy_signal'] = (
        signals['rsi_oversold'] |
        signals['macd_buy'] |
        signals['bb_buy']
    )

    signals['sell_signal'] = (
        signals['rsi_overbought'] |
        signals['macd_sell'] |
        signals['bb_sell']
    )

    return signals

# 3. 신호 생성
signals = generate_signals(features)

# 4. 결과 확인
print(f"매수 신호: {signals['buy_signal'].sum()}개")
print(f"매도 신호: {signals['sell_signal'].sum()}개")

# 5. 백테스팅 (간단한 예시)
initial_balance = 1000000
balance = initial_balance
position = 0

for i in range(len(signals)):
    if signals['buy_signal'].iloc[i] and balance > 0:
        # 전량 매수
        position = balance / signals['price'].iloc[i]
        balance = 0
        print(f"{signals.index[i]}: 매수 @ {signals['price'].iloc[i]}")

    elif signals['sell_signal'].iloc[i] and position > 0:
        # 전량 매도
        balance = position * signals['price'].iloc[i]
        position = 0
        print(f"{signals.index[i]}: 매도 @ {signals['price'].iloc[i]}")

# 최종 평가
final_value = balance + (position * signals['price'].iloc[-1] if position > 0 else 0)
profit = final_value - initial_balance
profit_rate = (profit / initial_balance) * 100

print(f"\n초기 자금: {initial_balance:,.0f}원")
print(f"최종 자금: {final_value:,.0f}원")
print(f"수익: {profit:,.0f}원 ({profit_rate:.2f}%)")
```

---

## 📞 문의 및 기여

문제가 발생하거나 개선 사항이 있으면 이슈를 등록해주세요.

---

**최종 업데이트**: 2025-11-06
**버전**: 2.0.0
