# 데이터 파이프라인 사용 가이드

> **최종 업데이트**: 2025년 10월 05일 20:30

하이브리드 데이터 파이프라인으로 **오프라인 학습은 빠르게, 실시간 트레이딩은 안정적으로!**

---

## 🚀 빠른 시작

### 1. 오프라인 학습 (추천 ⭐)

```bash
# 1단계: 데이터 준비 (한 번만 실행)
python scripts/prepare_offline_data.py --market KRW-BTC --days 30

# 2단계: 학습 (매우 빠름!)
python run_train.py --db data/market_data.db --episodes 1000
```

### 2. 실시간 트레이딩

```bash
# 캐시 활용 + 실시간 계산
python run_realtime_trading.py --db data/market_data.db --cache-enabled
```

---

## 📊 데이터 파이프라인 구조

```
┌─────────────────────────────────────────────────────────┐
│                  데이터 파이프라인                        │
└─────────────────────────────────────────────────────────┘

1. Upbit API → OHLCV 원본 데이터
   └─> SQLite (ohlcv_data 테이블)

2. 기술적 지표 계산
   - SMA, EMA, RSI, MACD, Bollinger Bands, ATR, OBV 등

3. 특성 추출
   - SSL 특성 (대조 학습, 마스킹 예측, 시간 패턴)
   - 정규화 (robust/standard/minmax)

4. 저장
   └─> SQLite (processed_data 테이블) ← 캐시!

5. TradingEnvironment
   └─> 학습 또는 실시간 트레이딩
```

---

## 🎯 모드별 동작

### 오프라인 모드 (`mode="offline"`)

- **특징**: SQLite 캐시만 사용
- **장점**: 매우 빠름 (API 호출 없음)
- **용도**: 빠른 학습, 실험

```python
from trading_env import TradingEnvironment, TradingConfig

config = TradingConfig()
env = TradingEnvironment(
    config,
    market="KRW-BTC",
    db_path="data/market_data.db",
    mode="offline",  # ← 오프라인 모드
    cache_enabled=True
)
```

### 실시간 모드 (`mode="realtime"`)

- **특징**: 캐시 우선, 없으면 계산 후 저장
- **장점**: 안정적, 최신 데이터 사용 가능
- **용도**: 실시간 트레이딩

```python
env = TradingEnvironment(
    config,
    market="KRW-BTC",
    db_path="data/market_data.db",
    mode="realtime",  # ← 실시간 모드
    cache_enabled=True
)
```

---

## 📝 상세 사용법

### 데이터 준비

#### 방법 1: 스크립트 사용 (추천)

```bash
# 기본 (7일)
python scripts/prepare_offline_data.py --market KRW-BTC

# 30일 데이터
python scripts/prepare_offline_data.py --market KRW-BTC --days 30

# 여러 마켓
python scripts/prepare_offline_data.py --market KRW-BTC --days 30
python scripts/prepare_offline_data.py --market KRW-ETH --days 30
python scripts/prepare_offline_data.py --market KRW-XRP --days 30

# 정규화 방법 지정
python scripts/prepare_offline_data.py --market KRW-BTC --normalization standard

# SSL 특성 제외
python scripts/prepare_offline_data.py --market KRW-BTC --no-ssl
```

#### 방법 2: Python 코드

```python
from trading_env.data_pipeline import prepare_offline_data

# 데이터 준비
prepare_offline_data(
    market="KRW-BTC",
    days=30,
    db_path="data/market_data.db",
    normalization_method="robust",
    include_ssl=True
)
```

### 학습

#### 오프라인 학습

```bash
# 데이터 준비 + 학습 한 번에
python run_train.py --collect-data --db data/market_data.db --episodes 500

# 또는 분리
python scripts/prepare_offline_data.py --market KRW-BTC --days 30
python run_train.py --db data/market_data.db --episodes 1000
```

#### 실시간 학습 (API 사용)

```bash
# 데이터베이스 없이 (느림, API 호출)
python run_train.py --episodes 100
```

### 백테스팅

```bash
# 오프라인 데이터로 백테스팅 (빠름)
python run_backtest.py --db data/market_data.db --model models/best_model.pth
```

### 실시간 트레이딩

```bash
# 캐시 활용 (추천)
python run_realtime_trading.py --db data/market_data.db --cache-enabled

# 데모 모드
python run_realtime_trading.py --db data/market_data.db --demo
```

---

## 🔧 고급 기능

### 캐시 무효화

설정이 바뀌면 자동으로 캐시 무효화:

```python
# 정규화 방법 변경 → 자동 재계산
prepare_offline_data(
    market="KRW-BTC",
    normalization_method="standard"  # robust → standard
)
```

### 수동 데이터 관리

```python
from trading_env.data_storage import MarketDataStorage

storage = MarketDataStorage("data/market_data.db")

# 데이터 정보 확인
count = storage.get_data_count("KRW-BTC")
min_time, max_time = storage.get_data_range("KRW-BTC")

print(f"저장된 데이터: {count}건")
print(f"범위: {min_time} ~ {max_time}")

# 처리된 데이터 로드
processed_data = storage.load_processed_data("KRW-BTC")
print(processed_data.head())

# 데이터 삭제
storage.clear_data("KRW-BTC")  # 특정 마켓
storage.clear_data()  # 전체
```

### 커스텀 파이프라인

```python
from trading_env.data_pipeline import DataPipeline
from trading_env.data_storage import MarketDataStorage

storage = MarketDataStorage("data/market_data.db")
pipeline = DataPipeline(
    storage=storage,
    mode="realtime",
    cache_enabled=True,
    normalization_method="robust",
    include_ssl=True
)

# 데이터 처리
from datetime import datetime, timedelta
processed_data = pipeline.process_data(
    market="KRW-BTC",
    start_time=datetime.now() - timedelta(days=7),
    end_time=datetime.now(),
    force_recalculate=False  # True면 캐시 무시하고 재계산
)
```

---

## 📊 성능 비교

| 방식 | 1000 에피소드 학습 시간 | 특징 |
|------|-------------------------|------|
| **실시간 API** | ~60분 | 매번 API 호출, 느림 |
| **오프라인 캐시** | ~6분 | SQLite 캐시, 매우 빠름 ⚡ |
| **하이브리드** | ~7-10분 | 캐시 우선, 안정적 |

**결론**: 오프라인 모드가 **10배 빠름!** 🚀

---

## ⚠️ 주의사항

### 오프라인 모드
- ✅ 빠름
- ✅ 재현 가능
- ❌ 데이터 미리 준비 필요
- ❌ 최신 데이터 아님

### 실시간 모드
- ✅ 최신 데이터
- ✅ 캐시 활용 가능
- ❌ API 호출 필요 (느릴 수 있음)
- ❌ API 제한 주의

---

## 🐛 문제 해결

### Q1: "SQLite에 데이터가 없습니다" 에러

```bash
# 해결: 데이터 먼저 준비
python scripts/prepare_offline_data.py --market KRW-BTC --days 7
```

### Q2: 설정을 바꿨는데 캐시가 안 바뀌어요

```python
# 해결: config_hash가 자동으로 변경됨, force_recalculate 사용
pipeline.process_data(market="KRW-BTC", force_recalculate=True)
```

### Q3: 데이터베이스 크기가 너무 커요

```python
# 오래된 데이터 삭제
storage = MarketDataStorage("data/market_data.db")
storage.clear_data("KRW-BTC")

# 또는 특정 기간만 유지
# (수동 SQL 필요)
```

---

## 📚 참고

- **[CHANGELOG.md](../.github/docs/CHANGELOG.md)**: 개발 로그
- **[SQLITE_USAGE.md](SQLITE_USAGE.md)**: SQLite 상세 가이드
- **[README.md](../README.md)**: 프로젝트 개요

---

**Happy Trading with Hybrid Pipeline! 🚀**
