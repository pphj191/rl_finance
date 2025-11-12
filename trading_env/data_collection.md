# data_collection.py 구조도

## 개요
데이터 수집 통합 모듈로서 사용자 요청에 따라 DB에 없는 데이터를 API로 수집하여 제공합니다.

## 모듈 구성

### 1. 모듈 레벨 함수

```
setup_module_logging(level: int = LOG_LEVEL) -> logger
  └─ 역할: 모듈 레벨 로깅 설정
  └─ 위치: 33-45행
```

```
create_data_collector(db_path: str, log_level: int) -> DataCollector
  └─ 역할: DataCollector 인스턴스 생성 편의 함수
  └─ 위치: 420-434행
```

---

## 2. DataCollector 클래스

### 클래스 구조

```
DataCollector
├─ 클래스 변수
│  └─ TIMEFRAME_MINUTES: Dict[str, int]  # 타임프레임별 분 단위 매핑
│
├─ 초기화 (__init__)
│  ├─ self.storage: MarketDataStorage      # DB 연동
│  ├─ self.upbit_api: UpbitAPI            # Upbit API 연동
│  └─ self.logger: Logger                  # 로거
│
├─ [내부 헬퍼 메서드]
│  ├─ _calculate_required_count()         # 필요 데이터 개수 계산
│  ├─ _parse_timeframe_for_api()          # 타임프레임 파싱
│  └─ _fetch_from_api()                   # API에서 데이터 수집
│
├─ [공개 메서드: 개별 타임프레임]
│  ├─ get_candles_by_count()              # 개수 기반 데이터 수집
│  └─ get_candles_by_range()              # 시간 범위 기반 데이터 수집
│
└─ [공개 메서드: Multi-timeframe]
   ├─ get_multi_timeframe_data()          # 개수 기반 멀티 타임프레임 수집
   └─ get_multi_timeframe_data_by_range() # 시간 범위 기반 멀티 타임프레임 수집
```

---

### 2.1 클래스 변수

#### TIMEFRAME_MINUTES
```python
TIMEFRAME_MINUTES = {
    '1m': 1, '3m': 3, '5m': 5, '10m': 10, '15m': 15, '30m': 30,
    '60m': 60, '240m': 240, '1h': 60, '4h': 240,
    '1d': 1440, '1w': 10080, '1M': 43200
}
```
- **역할**: 타임프레임 문자열을 분 단위로 변환하는 매핑
- **위치**: 60-74행

---

### 2.2 초기화 메서드

#### `__init__(db_path, api_access_key, api_secret_key, log_level)`
**위치**: 76-103행

**역할**: DataCollector 인스턴스 초기화

**의존성**:
```
__init__()
├─ MarketDataStorage 인스턴스 생성 → self.storage
└─ UpbitAPI 인스턴스 생성 → self.upbit_api
```

**파라미터**:
- `db_path`: SQLite DB 파일 경로 (기본값: "data/market_data.db")
- `api_access_key`: Upbit API Access Key (선택)
- `api_secret_key`: Upbit API Secret Key (선택)
- `log_level`: 로깅 레벨 (기본값: LOG_LEVEL)

---

### 2.3 내부 헬퍼 메서드

#### `_calculate_required_count(timeframe, start_time, end_time) -> int`
**위치**: 109-134행

**역할**: 시간 범위로부터 필요한 데이터 개수 계산

**의존성**:
```
_calculate_required_count()
└─ self.TIMEFRAME_MINUTES (클래스 변수 참조)
```

**로직**:
1. 타임프레임 유효성 검증
2. 총 시간(분)을 캔들 단위 시간으로 나누어 개수 계산
3. 계산된 개수 반환

---

#### `_parse_timeframe_for_api(timeframe) -> Tuple[str, Optional[int]]`
**위치**: 136-162행

**역할**: 타임프레임을 API 파라미터로 변환

**반환값**:
- `candle_type`: 'minutes', 'days', 'weeks', 'months'
- `unit`: 분 캔들의 경우 단위 (1, 3, 5, 15, 30, 60, 240), 그 외는 None

**변환 규칙**:
```
'1m', '3m', '5m', '10m', '15m', '30m', '60m', '240m' → ('minutes', unit)
'1h' → ('minutes', 60)
'4h' → ('minutes', 240)
'1d' → ('days', None)
'1w' → ('weeks', None)
'1M' → ('months', None)
```

---

#### `_fetch_from_api(market, timeframe, count, to) -> pd.DataFrame`
**위치**: 164-238행

**역할**: API에서 데이터 수집

**의존성**:
```
_fetch_from_api()
├─ _parse_timeframe_for_api() → (candle_type, unit)
└─ self.upbit_api
   ├─ get_candles_minutes_bulk()
   ├─ get_candles_days_bulk()
   ├─ get_candles_weeks_bulk()
   └─ get_candles_months_bulk()
```

**처리 흐름**:
1. 타임프레임 파싱 (`_parse_timeframe_for_api`)
2. API 호출 (candle_type에 따라 적절한 메서드 선택)
3. DataFrame 변환 및 컬럼 매핑
4. 타임스탬프 정규화 (`align_timestamp`)
5. 시간 순 정렬 및 중복 제거
6. DataFrame 반환

**반환 컬럼**: `timestamp`, `open`, `high`, `low`, `close`, `volume`

---

### 2.4 공개 메서드: 개별 타임프레임 데이터 수집

#### `get_candles_by_count(market, timeframe, count, end_time, force_api) -> pd.DataFrame`
**위치**: 244-305행

**역할**: 끝시간과 개수로 데이터 수집

**의존성**:
```
get_candles_by_count()
├─ align_timestamp() → end_time 정규화
├─ _fetch_from_api() (force_api=True 또는 DB 부족 시)
├─ self.storage.load_data() (DB 조회)
└─ self.storage.save_data() (API 데이터 저장)
```

**처리 흐름**:
1. `end_time` 정규화
2. `force_api=True`이면 바로 API 호출
3. 아니면:
   - 시작 시간 계산
   - DB 조회
   - 데이터 충분하면 반환
   - 부족하면 API 호출 및 병합
4. 최종 데이터 반환

**파라미터**:
- `market`: 마켓 코드 (예: "KRW-BTC")
- `timeframe`: 타임프레임 (예: "1m", "1h", "1d")
- `count`: 데이터 개수
- `end_time`: 종료 시간 (기본값: 현재 시간)
- `force_api`: DB 무시하고 API 직접 호출 (기본값: False)

---

#### `get_candles_by_range(market, timeframe, start_time, end_time, force_api) -> pd.DataFrame`
**위치**: 307-340행

**역할**: 시작시간과 끝시간으로 데이터 수집

**의존성**:
```
get_candles_by_range()
├─ align_timestamp() → start_time, end_time 정규화
├─ _calculate_required_count() → 필요 개수 계산
└─ get_candles_by_count() → 실제 데이터 수집
```

**처리 흐름**:
1. 시간 정규화
2. 필요 데이터 개수 계산
3. `get_candles_by_count()` 호출

**파라미터**:
- `market`: 마켓 코드
- `timeframe`: 타임프레임
- `start_time`: 시작 시간
- `end_time`: 종료 시간 (기본값: 현재 시간)
- `force_api`: DB 무시 여부 (기본값: False)

---

### 2.5 공개 메서드: Multi-timeframe 데이터 수집

#### `get_multi_timeframe_data(market, timeframes, count_per_timeframe, end_time, force_api) -> Dict[str, pd.DataFrame]`
**위치**: 346-381행

**역할**: 여러 타임프레임의 데이터를 동시에 수집 (개수 기반)

**의존성**:
```
get_multi_timeframe_data()
└─ get_candles_by_count() (각 타임프레임마다 호출)
```

**처리 흐름**:
1. 각 타임프레임에 대해 반복
2. `get_candles_by_count()` 호출
3. 결과를 딕셔너리에 저장
4. 딕셔너리 반환

**파라미터**:
- `market`: 마켓 코드
- `timeframes`: 타임프레임 리스트 (예: `['1m', '1h', '1d']`)
- `count_per_timeframe`: 타임프레임별 개수 (예: `{'1m': 100, '1h': 24}`)
- `end_time`: 종료 시간 (기본값: 현재 시간)
- `force_api`: DB 무시 여부 (기본값: False)

**반환값**: `{timeframe: DataFrame}` 형태의 딕셔너리

---

#### `get_multi_timeframe_data_by_range(market, timeframes, start_time, end_time, force_api) -> Dict[str, pd.DataFrame]`
**위치**: 383-416행

**역할**: 시간 범위로 여러 타임프레임의 데이터를 동시에 수집

**의존성**:
```
get_multi_timeframe_data_by_range()
└─ get_candles_by_range() (각 타임프레임마다 호출)
```

**처리 흐름**:
1. 각 타임프레임에 대해 반복
2. `get_candles_by_range()` 호출
3. 결과를 딕셔너리에 저장
4. 딕셔너리 반환

**파라미터**:
- `market`: 마켓 코드
- `timeframes`: 타임프레임 리스트
- `start_time`: 시작 시간
- `end_time`: 종료 시간 (기본값: 현재 시간)
- `force_api`: DB 무시 여부 (기본값: False)

**반환값**: `{timeframe: DataFrame}` 형태의 딕셔너리

---

## 3. 메서드 호출 관계도

### 3.1 데이터 흐름 (개수 기반)
```
get_candles_by_count()
├─ [force_api=True 경로]
│  ├─ _fetch_from_api()
│  │  ├─ _parse_timeframe_for_api()
│  │  └─ upbit_api.get_candles_*_bulk()
│  └─ storage.save_data()
│
└─ [force_api=False 경로]
   ├─ storage.load_data()
   ├─ [데이터 부족 시]
   │  ├─ _fetch_from_api()
   │  └─ storage.save_data()
   └─ DataFrame 병합 및 반환
```

### 3.2 데이터 흐름 (시간 범위 기반)
```
get_candles_by_range()
├─ _calculate_required_count()
│  └─ TIMEFRAME_MINUTES 참조
└─ get_candles_by_count()
   └─ (위의 3.1 참조)
```

### 3.3 데이터 흐름 (Multi-timeframe, 개수 기반)
```
get_multi_timeframe_data()
└─ for each timeframe:
   └─ get_candles_by_count()
      └─ (위의 3.1 참조)
```

### 3.4 데이터 흐름 (Multi-timeframe, 시간 범위 기반)
```
get_multi_timeframe_data_by_range()
└─ for each timeframe:
   └─ get_candles_by_range()
      └─ (위의 3.2 참조)
```

---

## 4. 외부 의존성

### 4.1 모듈 의존성
```
data_collection.py
├─ trading_env.data_storage
│  ├─ MarketDataStorage
│  └─ align_timestamp
│
└─ upbit_api.upbit_api
   └─ UpbitAPI
```

### 4.2 주요 라이브러리
- `pandas`: DataFrame 처리
- `logging`: 로깅
- `datetime`: 시간 처리
- `typing`: 타입 힌팅

---

## 5. 사용 예제

### 5.1 초기화
```python
collector = DataCollector(db_path="data/market_data.db")
```

### 5.2 개수 기반 수집
```python
df = collector.get_candles_by_count(
    market="KRW-BTC",
    timeframe="1m",
    count=100,
    end_time=datetime(2025, 10, 10, 12, 0, 0)
)
```

### 5.3 시간 범위 기반 수집
```python
df = collector.get_candles_by_range(
    market="KRW-BTC",
    timeframe="1h",
    start_time=datetime(2025, 10, 9, 0, 0, 0),
    end_time=datetime(2025, 10, 10, 0, 0, 0)
)
```

### 5.4 Multi-timeframe 수집
```python
data_dict = collector.get_multi_timeframe_data(
    market="KRW-BTC",
    timeframes=['1m', '1h', '1d'],
    count_per_timeframe={'1m': 60, '1h': 24, '1d': 7},
    end_time=datetime(2025, 10, 10, 12, 0, 0)
)
```

---

## 6. 핵심 설계 패턴

### 6.1 계층적 메서드 구조
- **최상위**: Multi-timeframe 메서드 (여러 타임프레임 처리)
- **중간층**: 개별 타임프레임 메서드 (단일 타임프레임 처리)
- **하위층**: 헬퍼 메서드 (API 호출, 파싱, 계산)

### 6.2 데이터 우선순위
1. DB에 데이터가 있으면 DB 사용
2. DB에 데이터가 없거나 부족하면 API 호출
3. `force_api=True`이면 DB 무시하고 API 직접 호출

### 6.3 데이터 정규화
- 모든 타임스탬프는 `align_timestamp()`로 정규화
- 중복 데이터 자동 제거
- 시간 순 정렬 보장

---

## 7. 주요 특징

1. **DB 우선 전략**: API 호출 최소화
2. **Multi-timeframe 지원**: 여러 타임프레임 동시 수집
3. **유연한 인터페이스**: 개수/시간 범위 기반 수집 지원
4. **자동 데이터 저장**: API로 수집한 데이터를 DB에 자동 저장
5. **타임스탬프 정규화**: 데이터 일관성 보장
6. **로깅**: 모든 주요 작업에 대한 상세 로깅
