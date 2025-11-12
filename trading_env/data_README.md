# 데이터 수집 시스템 구조

이 문서는 `data_collection.py`, `data_storage.py` 간의 관계와 각 모듈의 역할을 설명합니다.

> **참고**: `market_data.py`는 레거시 모듈로 `backup/` 폴더로 이동되었습니다.

## 시스템 아키텍처

```
┌─────────────────────────────────────────────────────────────────┐
│                        Trading System                            │
└─────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│                   data_collection.py (통합 모듈)                  │
│                                                                   │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │              DataCollector (메인 클래스)                   │  │
│  │                                                            │  │
│  │  Public Methods:                                           │  │
│  │  - get_candles_by_count()      # 개수로 데이터 수집       │  │
│  │  - get_candles_by_range()      # 시간범위로 데이터 수집   │  │
│  │  - get_multi_timeframe_data()  # 멀티 타임프레임 수집     │  │
│  │                                                            │  │
│  └────────────────────────────────────────────────────────────┘  │
│           │                                  │                    │
│           │ 의존                            │ 의존               │
│           ▼                                  ▼                    │
│  ┌─────────────────────┐          ┌─────────────────────┐        │
│  │  MarketDataStorage  │          │    UpbitAPI         │        │
│  │  (from data_storage)│          │  (upbit_api 모듈)  │        │
│  └─────────────────────┘          └─────────────────────┘        │
└──────────────────────────────────────────────────────────────────┘
           │                                  │
           │ 조회/저장                        │ API 호출
           ▼                                  ▼

┌──────────────────────────────────────────────────────────────────┐
│                   data_storage.py (DB 모듈)                      │
│                                                                   │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │            MarketDataStorage (DB 클래스)                   │  │
│  │                                                            │  │
│  │  Data Operations:                                          │  │
│  │  - load_data()           # DB에서 데이터 조회             │  │
│  │  - save_data()           # DB에 데이터 저장               │  │
│  │  - has_data()            # 데이터 존재 확인               │  │
│  │  - get_data_range()      # 데이터 시간 범위 조회          │  │
│  │                                                            │  │
│  │  Utility:                                                  │  │
│  │  - align_timestamp()     # 타임프레임별 시간 정규화       │  │
│  └────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
           │
           │ SQLite 연결
           ▼
┌──────────────────────────────────────────────────────────────────┐
│                    SQLite Database                                │
│                                                                   │
│  Tables (타임프레임별 분리):                                      │
│  - market_1m  (1분봉)                                            │
│  - market_1h  (1시간봉)                                          │
│  - market_1d  (1일봉)                                            │
│  - market_1w  (1주봉)                                            │
│  - ... (기타 타임프레임)                                         │
│                                                                   │
│  Columns: market, timestamp, open, high, low, close, volume      │
└──────────────────────────────────────────────────────────────────┘
```

## 데이터 흐름 (Data Flow)

```
사용자 요청
    │
    ▼
┌─────────────────────────────────────────────────┐
│ DataCollector.get_candles_by_count()            │
│                                                 │
│ 1. DB 조회 (MarketDataStorage.load_data())     │
│    └─> DB에 충분한 데이터 있음? ───┐           │
│                                    │           │
│        YES                        NO          │
│         │                          │           │
│         └─> DB 데이터 반환          │           │
│                                    │           │
│                 2. API 호출 (UpbitAPI) ◄───────┤
│                    │                           │
│                    ▼                           │
│         3. DB 저장 (save_data())                │
│                    │                           │
│                    ▼                           │
│         4. DB + API 데이터 병합                 │
│                    │                           │
└────────────────────┼───────────────────────────┘
                     │
                     ▼
              최종 데이터 반환
```

## 모듈 상세 설명

### 1. data_storage.py (저장소 레이어)

**역할**: SQLite 데이터베이스와의 직접적인 상호작용을 담당하는 순수 저장소 레이어

**주요 클래스**:
- `MarketDataStorage`: SQLite 기반 데이터 저장소 클래스

**핵심 기능**:
- **데이터 조회 (Read)**
  - `load_data()`: 마켓, 타임프레임, 시간 범위로 데이터 조회
  - `get_data_range()`: 저장된 데이터의 최소/최대 시간 조회
  - `get_data_count()`: 특정 조건의 데이터 개수 조회
  - `has_data()`: 데이터 존재 여부 확인

- **데이터 저장 (Write)**
  - `save_data()`: DataFrame을 DB에 저장 (INSERT OR REPLACE)
  - `update_data()`: 특정 타임스탬프의 데이터 업데이트

- **데이터 관리 (Management)**
  - `delete_data()`: 조건에 맞는 데이터 삭제
  - `get_available_markets()`: 저장된 모든 마켓 목록 조회
  - `get_available_timeframes()`: 특정 마켓의 타임프레임 목록 조회
  - `get_database_stats()`: DB 통계 정보 조회

**데이터베이스 구조**:
- 타임프레임별로 테이블 분리 (성능 최적화)
- 지원 타임프레임: `1m`, `3m`, `5m`, `10m`, `15m`, `30m`, `60m`, `1h`, `4h`, `1d`, `1w`, `1M`
- 각 테이블 스키마:
  ```sql
  CREATE TABLE market_1m (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      market TEXT NOT NULL,
      timestamp TEXT NOT NULL,
      open REAL,
      high REAL,
      low REAL,
      close REAL,
      volume REAL,
      created_at TEXT DEFAULT CURRENT_TIMESTAMP,
      UNIQUE(market, timestamp)
  )
  ```
- 복합 인덱스: `(market, timestamp)` - 빠른 조회를 위함

**유틸리티 함수**:
- `align_timestamp()`: 타임프레임에 맞게 타임스탬프 정규화
  - 예: `2025-10-09 14:23:45.123` → `2025-10-09 14:23:00` (1분봉)

**사용 예시**:
```python
# Storage 인스턴스 생성
storage = MarketDataStorage(db_path="data/market_data.db")

# 데이터 조회
df = storage.load_data(
    market="KRW-BTC",
    timeframe="1h",
    start_time=datetime(2025, 1, 1),
    end_time=datetime(2025, 1, 31)
)

# 데이터 저장
storage.save_data(market="KRW-BTC", timeframe="1h", df=candle_data)
```

---

### 2. data_collection.py (통합 레이어)

**역할**: DB와 API를 조합하여 스마트한 데이터 수집 제공. DB 캐시를 우선 사용하고 부족한 데이터만 API로 요청

**주요 클래스**:
- `DataCollector`: 통합 데이터 수집 클래스

**핵심 기능**:

- **개별 타임프레임 수집**
  - `get_candles_by_count()`: 끝시간과 개수로 데이터 수집
    - DB에 데이터가 있으면 DB에서 반환
    - 없으면 API 호출 → DB 저장 → 반환
  - `get_candles_by_range()`: 시작시간과 끝시간으로 데이터 수집

- **멀티 타임프레임 수집**
  - `get_multi_timeframe_data()`: 여러 타임프레임 데이터를 동시에 수집
  - `get_multi_timeframe_data_by_range()`: 시간 범위로 멀티 타임프레임 수집

**내부 헬퍼 메서드**:
- `_calculate_required_count()`: 시간 범위로부터 필요한 데이터 개수 계산
- `_parse_timeframe_for_api()`: 타임프레임을 API 파라미터로 변환
- `_fetch_from_api()`: API에서 데이터 수집 및 정규화

**지능형 캐싱 로직**:
1. 먼저 DB에서 데이터 조회
2. DB에 충분한 데이터가 있으면 즉시 반환
3. 부족하면 API 호출
4. API 데이터를 DB에 저장
5. DB + API 데이터 병합 후 반환

**의존성**:
- `MarketDataStorage` (data_storage.py): DB 작업
- `UpbitAPI` (upbit_api 모듈): API 호출

**사용 예시**:
```python
# Collector 인스턴스 생성
collector = DataCollector(db_path="data/market_data.db")

# 100개의 1분봉 데이터 수집 (DB 우선, 없으면 API)
df = collector.get_candles_by_count(
    market="KRW-BTC",
    timeframe="1m",
    count=100,
    end_time=datetime.now()
)

# 멀티 타임프레임 수집
data_dict = collector.get_multi_timeframe_data(
    market="KRW-BTC",
    timeframes=['1m', '1h', '1d'],
    count_per_timeframe={'1m': 100, '1h': 24, '1d': 7}
)
# 결과: {'1m': DataFrame, '1h': DataFrame, '1d': DataFrame}
```

---

## 의존성 관계

```
data_collection.py
    ├─> data_storage.py (DB 작업)
    └─> UpbitAPI (API 호출)

data_storage.py
    └─> SQLite (DB 연결)
```

**모듈 간 독립성**:
- `data_collection.py`: `data_storage.py`와 `UpbitAPI`에 의존
- `data_storage.py`: SQLite만 사용, 다른 모듈과 독립적

---

## 사용 시나리오

### 시나리오 1: 대용량 과거 데이터 수집 및 저장
→ **data_collection.py 사용**
```python
collector = DataCollector(db_path="data/market_data.db")
df = collector.get_candles_by_range(
    market="KRW-BTC",
    timeframe="1h",
    start_time=datetime(2024, 1, 1),
    end_time=datetime(2025, 1, 1)
)
```

### 시나리오 2: 멀티 타임프레임 분석
→ **data_collection.py 사용**
```python
collector = DataCollector()
data_dict = collector.get_multi_timeframe_data(
    market="KRW-BTC",
    timeframes=['1m', '5m', '1h', '1d'],
    count_per_timeframe={'1m': 100, '5m': 100, '1h': 24, '1d': 30}
)
```

### 시나리오 3: DB 직접 조회 (최고 성능)
→ **data_storage.py 사용**
```python
storage = MarketDataStorage(db_path="data/market_data.db")
df = storage.load_data(
    market="KRW-BTC",
    timeframe="1h",
    start_time=datetime(2025, 1, 1),
    limit=1000
)
```

---

## 설계 원칙

1. **단일 책임 원칙 (SRP)**
   - `data_storage.py`: 오직 DB 작업만 담당
   - `data_collection.py`: 데이터 수집 로직만 담당

2. **개방-폐쇄 원칙 (OCP)**
   - 새로운 거래소 추가 시 기존 코드 수정 없이 확장 가능
   - 새로운 타임프레임 추가 시 테이블만 추가

3. **의존성 역전 원칙 (DIP)**
   - `data_collection.py`는 `data_storage.py`의 인터페이스에만 의존
   - DB 구현 변경 시 `data_collection.py`는 영향 받지 않음

4. **캐싱 전략**
   - DB를 1차 캐시로 사용
   - API 호출 최소화로 Rate Limit 회피

---

## 성능 최적화

1. **타임프레임별 테이블 분리**
   - 단일 테이블 대비 쿼리 성능 향상
   - 인덱스 효율성 증가

2. **복합 인덱스 사용**
   - `(market, timestamp)` 인덱스로 빠른 조회

3. **스마트 캐싱**
   - DB 우선 조회로 API 호출 최소화
   - 중복 데이터 방지 (`UNIQUE` 제약)

4. **Bulk Insert**
   - DataFrame 단위로 데이터 저장

---

## 로깅

모든 모듈은 Python `logging` 모듈을 사용하여 상세한 로그를 제공합니다.

```python
# 로깅 레벨 설정
import logging
logging.basicConfig(level=logging.INFO)

# 각 모듈의 로그 레벨 개별 설정 가능
collector = DataCollector(log_level=logging.DEBUG)
```

**로그 예시**:
```
2025-10-12 14:23:45.123 - data_collection.DataCollector - INFO - get_candles_by_count: KRW-BTC 1m count=100
2025-10-12 14:23:45.234 - data_storage.MarketDataStorage - INFO - Loaded 50 rows for KRW-BTC 1m
2025-10-12 14:23:45.345 - data_collection.DataCollector - INFO - Insufficient data in DB (50 rows), fetching from API
2025-10-12 14:23:46.456 - data_collection.DataCollector - INFO - Fetched 100 rows from API for KRW-BTC 1m
```

---

## 추가 참고 자료

- [data_collection.py](data_collection.py) - 통합 데이터 수집 모듈
- [data_storage.py](data_storage.py) - SQLite 저장소 모듈
- [README.md](README.md) - 전체 Trading Environment 문서
