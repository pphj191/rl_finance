# Trading Environment Package

강화학습 기반 암호화폐 트레이딩 환경을 제공하는 패키지입니다.

## 📦 패키지 구조

```
trading_env/
├── core/                    # 핵심 환경
│   ├── base_env.py         # 기본 설정 및 Enum
│   ├── rl_env.py           # TradingEnvironment (Gymnasium 기반)
│   └── env_pipeline.py     # 데이터 파이프라인
│
├── data/                    # 데이터 관리
│   ├── storage.py          # SQLite 저장/로드
│   ├── collection.py       # 데이터 수집기
│   └── market_data.py      # 시장 데이터 처리 및 정규화
│
├── indicators/              # 기술적 지표
│   ├── basic.py            # 기본 지표 (SMA, EMA, RSI, MACD)
│   ├── custom.py           # 커스텀 지표 (눌림목, 변동성)
│   └── ssl.py              # Self-Supervised Learning 특성
│
└── docs/                    # 상세 문서
    ├── README.md           # 전체 문서
    └── data_collection_TODO.md
```

## 🚀 빠른 시작

### 기본 사용법

```python
from trading_env import TradingEnvironment, TradingConfig

# 설정
config = TradingConfig(
    initial_balance=1000000,
    trading_fee=0.0005,
    window_size=60
)

# 환경 생성
env = TradingEnvironment(
    config=config,
    market="KRW-BTC",
    db_path="./data/market_data.db",
    mode="offline"
)

# 강화학습 루프
obs, info = env.reset()
for _ in range(1000):
    action = env.action_space.sample()  # 랜덤 액션
    obs, reward, done, truncated, info = env.step(action)
    if done:
        break
```

### 데이터 수집 및 저장

```python
from trading_env import MarketDataStorage, collect_and_store_data

# 데이터 수집
collect_and_store_data(
    market="KRW-BTC",
    count=1000,
    unit=1,  # 1분봉
    db_path="./data/market_data.db"
)

# 데이터 로드
storage = MarketDataStorage("./data/market_data.db")
data = storage.load_market_data(
    market="KRW-BTC",
    timeframe="1m",
    days=30
)
```

### 기술적 지표 계산

```python
from trading_env import FeatureExtractor

extractor = FeatureExtractor()
features = extractor.extract_features(ohlcv_data)
```

## 📚 주요 컴포넌트

### TradingEnvironment
- Gymnasium 기반 강화학습 환경
- 오프라인/실시간 모드 지원
- 자동 데이터 캐싱 및 특성 추출

### DataPipeline
- 데이터 수집 → 지표 계산 → 정규화 → 저장
- 멀티 타임프레임 지원
- 효율적인 캐싱 시스템

### MarketDataStorage
- SQLite 기반 시계열 데이터 저장
- 타임프레임별 테이블 분리
- 빠른 시간 범위 쿼리

### FeatureExtractor
- 50+ 기술적 지표
- 이동평균, 모멘텀, 변동성, 거래량 지표
- SSL 기반 미래 예측 특성

## 🔧 고급 기능

### SSL (Self-Supervised Learning) 특성
```python
from trading_env import SSLFeatureExtractor, SSLConfig

ssl_config = SSLConfig(hidden_dim=128, num_layers=3)
ssl_extractor = SSLFeatureExtractor(ssl_config)
ssl_features = ssl_extractor.extract_features(data)
```

### 커스텀 지표
```python
from trading_env import CustomIndicators

custom = CustomIndicators()
pullback_idx = custom.calculate_pullback_index(data)
```

## 📖 상세 문서

더 자세한 내용은 [docs/README.md](docs/README.md)를 참조하세요.

## 🛠️ 의존성

- Python 3.8+
- gymnasium
- numpy, pandas
- torch (SSL 기능 사용 시)
- sklearn

## 📝 라이센스

MIT License
