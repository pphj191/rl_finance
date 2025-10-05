# SQLite 데이터 사용 가이드

> **최종 업데이트**: 2025년 10월 05일

## 📚 개요

매번 Upbit API에서 데이터를 받아오는 대신 SQLite 데이터베이스에 데이터를 저장하고 재사용할 수 있습니다.

### 장점
- ✅ **빠른 학습**: API 호출 없이 로컬 데이터 사용
- ✅ **재현성**: 동일한 데이터로 반복 학습 가능
- ✅ **API 제한 회피**: Upbit API 호출 제한 걱정 없음
- ✅ **오프라인 학습**: 인터넷 연결 없이 학습 가능

---

## 🚀 사용법

### 1. 데이터 수집 및 저장

먼저 Upbit에서 데이터를 수집하여 SQLite에 저장합니다:

```bash
# 방법 1: run_train.py의 --collect-data 옵션 사용
python run_train.py --collect-data --market KRW-BTC --data-count 1000 --db data/market_data.db

# 방법 2: Python 스크립트로 직접 수집
python -c "from trading_env.data_storage import collect_and_store_data; collect_and_store_data('KRW-BTC', count=1000, unit=1, db_path='data/market_data.db')"
```

### 2. SQLite 데이터로 학습

저장된 데이터를 사용하여 학습합니다:

```bash
# SQLite 데이터베이스 사용
python run_train.py --db data/market_data.db --episodes 1000 --market KRW-BTC

# 데이터 수집 + 학습 동시 실행
python run_train.py --collect-data --db data/market_data.db --episodes 500
```

### 3. 여러 마켓 데이터 수집

```bash
# 비트코인 데이터 수집
python run_train.py --collect-data --market KRW-BTC --data-count 2000 --db data/market_data.db

# 이더리움 데이터 추가 수집
python run_train.py --collect-data --market KRW-ETH --data-count 2000 --db data/market_data.db

# 각 마켓별로 학습
python run_train.py --db data/market_data.db --market KRW-BTC --episodes 1000
python run_train.py --db data/market_data.db --market KRW-ETH --episodes 1000
```

---

## 💻 Python 코드 예제

### 데이터 수집

```python
from trading_env.data_storage import MarketDataStorage, collect_and_store_data

# 1. 데이터 수집
collect_and_store_data(
    market="KRW-BTC",
    count=1000,  # 1000개의 1분봉 데이터
    unit=1,      # 1분봉
    db_path="data/market_data.db"
)

# 2. 저장된 데이터 확인
storage = MarketDataStorage("data/market_data.db")

# 데이터 개수
count = storage.get_data_count("KRW-BTC")
print(f"저장된 데이터: {count}건")

# 데이터 범위
min_time, max_time = storage.get_data_range("KRW-BTC")
print(f"데이터 범위: {min_time} ~ {max_time}")
```

### SQLite 데이터로 학습

```python
from trading_env import TradingConfig, TradingEnvironment
from rl_agent import TradingTrainer

# 설정
config = TradingConfig(
    initial_balance=1000000,
    model_type="dqn",
    learning_rate=1e-4
)

# SQLite 데이터로 트레이너 생성
trainer = TradingTrainer(
    config=config,
    market="KRW-BTC",
    db_path="data/market_data.db"  # SQLite 사용
)

# 학습 실행
results = trainer.train(num_episodes=500)
```

### DataFrame으로 직접 데이터 제공

```python
import pandas as pd
from trading_env import TradingEnvironment, TradingConfig

# 1. 데이터 로드
storage = MarketDataStorage("data/market_data.db")
data = storage.load_ohlcv_data("KRW-BTC")

# 2. 데이터를 직접 환경에 제공
config = TradingConfig()
env = TradingEnvironment(
    config=config,
    market="KRW-BTC",
    data=data  # DataFrame 직접 제공
)

# 3. 환경 사용
obs, info = env.reset()
```

---

## 📊 데이터 관리

### 데이터 조회

```python
from trading_env.data_storage import MarketDataStorage
from datetime import datetime, timedelta

storage = MarketDataStorage("data/market_data.db")

# 특정 기간 데이터 로드
start_time = datetime.now() - timedelta(days=7)
end_time = datetime.now()

data = storage.load_ohlcv_data(
    market="KRW-BTC",
    start_time=start_time,
    end_time=end_time
)

print(f"로드된 데이터: {len(data)}건")
print(data.head())
```

### 데이터 삭제

```python
from trading_env.data_storage import MarketDataStorage

storage = MarketDataStorage("data/market_data.db")

# 특정 마켓 데이터 삭제
storage.clear_data(market="KRW-BTC")

# 모든 데이터 삭제
storage.clear_data()
```

---

## 🔧 데이터베이스 구조

### OHLCV 데이터 테이블

| 컬럼 | 타입 | 설명 |
|------|------|------|
| id | INTEGER | 기본 키 (자동 증가) |
| market | TEXT | 마켓 코드 (예: KRW-BTC) |
| timestamp | INTEGER | Unix 타임스탬프 (초) |
| open | REAL | 시가 |
| high | REAL | 고가 |
| low | REAL | 저가 |
| close | REAL | 종가 |
| volume | REAL | 거래량 |
| value | REAL | 거래대금 |
| created_at | TIMESTAMP | 데이터 저장 시간 |

### 인덱스

- `idx_ohlcv_market_timestamp`: (market, timestamp) 복합 인덱스
- UNIQUE 제약: (market, timestamp) - 중복 방지

---

## ⚙️ 고급 사용법

### 1. 대량 데이터 수집

```bash
# 여러 마켓의 데이터를 한 번에 수집
for market in KRW-BTC KRW-ETH KRW-XRP KRW-ADA; do
    python run_train.py --collect-data --market $market --data-count 2000
done
```

### 2. 학습 파이프라인

```bash
#!/bin/bash
# train_pipeline.sh

DB_PATH="data/market_data.db"
MARKET="KRW-BTC"

# 1. 데이터 수집
echo "데이터 수집 중..."
python run_train.py --collect-data --market $MARKET --data-count 2000 --db $DB_PATH

# 2. 모델 학습
echo "모델 학습 중..."
python run_train.py --db $DB_PATH --market $MARKET --episodes 1000 --model-dir models/$(date +%Y%m%d)

# 3. 백테스팅
echo "백테스팅 중..."
python run_backtest.py --model models/$(date +%Y%m%d)/best_model.pth
```

### 3. 데이터 증분 업데이트

```python
from trading_env.data_storage import MarketDataStorage, collect_and_store_data
from datetime import datetime

storage = MarketDataStorage("data/market_data.db")

# 마지막 데이터 시간 확인
_, last_time = storage.get_data_range("KRW-BTC")

if last_time:
    print(f"마지막 데이터: {last_time}")
    # 마지막 시간 이후 데이터만 수집
    collect_and_store_data(
        market="KRW-BTC",
        count=100,  # 최근 100개만
        unit=1,
        db_path="data/market_data.db"
    )
```

---

## ⚠️ 주의사항

1. **데이터 중복**: SQLite는 (market, timestamp) 조합이 중복되면 에러 발생
2. **디스크 공간**: 대량 데이터 수집 시 디스크 공간 확인 필요
3. **데이터 정합성**: 데이터 수집 중 오류 발생 시 부분 데이터 저장될 수 있음
4. **타임존**: 모든 타임스탬프는 UTC 기준

---

## 🔍 문제 해결

### Q: 데이터가 너무 많아요

```python
# 오래된 데이터 삭제
from datetime import datetime, timedelta

storage = MarketDataStorage("data/market_data.db")
cutoff_time = datetime.now() - timedelta(days=30)

# 30일 이전 데이터 삭제 (직접 SQL 실행)
import sqlite3
conn = sqlite3.connect("data/market_data.db")
cursor = conn.cursor()
cursor.execute(
    "DELETE FROM ohlcv_data WHERE market = ? AND timestamp < ?",
    ["KRW-BTC", int(cutoff_time.timestamp())]
)
conn.commit()
conn.close()
```

### Q: 데이터베이스가 손상되었어요

```bash
# 데이터베이스 무결성 검사
sqlite3 data/market_data.db "PRAGMA integrity_check;"

# 데이터베이스 재구축
sqlite3 data/market_data.db "VACUUM;"
```

### Q: API 모드와 SQLite 모드의 차이는?

| 특성 | API 모드 | SQLite 모드 |
|------|----------|-------------|
| 속도 | 느림 (네트워크) | 빠름 (로컬) |
| 재현성 | 낮음 (실시간 데이터) | 높음 (고정 데이터) |
| API 제한 | 있음 | 없음 |
| 데이터 신선도 | 최신 | 수집 시점 |

---

**더 자세한 정보는 [프로젝트 README](../README.md)를 참조하세요.**
