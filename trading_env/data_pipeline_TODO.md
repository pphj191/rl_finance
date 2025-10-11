# 멀티 타임프레임 데이터 파이프라인 설계 (개선안)

> **최종 업데이트**: 2025-10-10

## 📋 TODO 체크리스트

### 🐛 버그 수정 (긴급)
- [ ] **타임스탬프 검증 로직 추가**
  - **문제**: 미래 시간 데이터가 저장됨 (예: 새벽 1시인데 "~ 07:15:00" 데이터 존재)
  - **원인**:
    1. Upbit API가 UTC 시간 반환하는데 KST로 잘못 해석
    2. 또는 데이터 저장 시 timezone 정규화 누락
  - **해결책**:
    - timezone 테스트를 통한 검증 (UTC,KST 변환 확인)
    - 테스트를 통해 수집된 타임스탬프가 현재 시간보다 미래인지 체크

- [ ] **중간 구간 Gap 감지 불가 문제**
  - **문제**: `get_data_range_by_timeframe()`이 MIN/MAX만 조회하여 중간 누락 감지 못함
  - **예시**: 10일 데이터 중 중간 3일 누락되어도 "전체 존재"로 표시
  - **해결책 옵션**:
    1. **간단**: 데이터 개수 검증 (기대 개수 vs 실제 개수 비교)
    2. **철저**: 연속성 검사 (샘플링 또는 전수 조사)
    3. **효율적**: 메타데이터 테이블 추가 (연속 구간 기록)
  - **우선순위**: Phase 2에서 구현

### 📝 개선 작업
- [ ] 타임스탬프 정규화 일관성 검증
- [ ] 로깅 메시지에 timezone 명시 (UTC/KST 혼동 방지)
- [ ] 데이터 수집 후 유효성 검사 추가 (범위, 개수, 중복)

---

## 📊 예상 성능

### 데이터 수집 시간 (KRW-BTC 기준)

| 기간 | 1분봉 개수 | 예상 요청 수 | 예상 소요 시간 |
|------|-----------|-------------|--------------|
| 1시간 | 60 | 1 | ~0.1초 |
| 1일 | 1,440 | 8 | ~1초 |
| 7일 | 10,080 | 51 | ~6초 |
| 30일 | 43,200 | 216 | ~24초 |

*Rate Limit (0.11초/요청) 고려*

### 저장 공간 (30일 기준)

| 타임프레임 | 개수 | 예상 크기 |
|-----------|------|----------|
| 1분봉 | 43,200 | ~2MB |
| 1시간봉 | 720 | ~40KB |
| 1일봉 | 30 | ~2KB |
| **합계** | **43,950** | **~2MB** |

---

## 🔍 사용 예제

### 예제 1: 최근 1일 데이터 수집 (end_time 지정)

```python
from datetime import datetime, timedelta
from trading_env.data_storage import collect_multi_timeframe_data

end_time = datetime.now()
start_time = end_time - timedelta(days=1)

collect_multi_timeframe_data(
    market="KRW-BTC",
    start_time=start_time,
    end_time=end_time,
    hourly_lookback_count=24,  # 추가 24시간
    daily_lookback_count=30,   # 추가 30일
    db_path="data/market_data.db"
)
```

### 예제 2: 특정 개수만 수집 (minute_candles_count 지정)

```python
from datetime import datetime
from trading_env.data_storage import collect_multi_timeframe_data

start_time = datetime.now() - timedelta(hours=2)

collect_multi_timeframe_data(
    market="KRW-BTC",
    start_time=start_time,
    minute_candles_count=120,  # 120개 1분봉 → end_time 자동 계산
    hourly_lookback_count=24,
    daily_lookback_count=30,
    db_path="data/market_data.db"
)
```

### 예제 3: 증분 수집 (기존 데이터에 추가)

```python
# 1차 수집
collect_multi_timeframe_data(
    market="KRW-BTC",
    start_time=datetime(2025, 10, 1),
    end_time=datetime(2025, 10, 5),
    db_path="data/market_data.db"
)

# 2차 수집 (10월 5일 ~ 10일) → 누락분만 수집
collect_multi_timeframe_data(
    market="KRW-BTC",
    start_time=datetime(2025, 10, 1),  # 동일한 시작점
    end_time=datetime(2025, 10, 10),   # 종료 시간만 연장
    db_path="data/market_data.db"
)
# → 10월 5일 ~ 10일 데이터만 추가 수집됨
```

---

## 📚 참고 자료

- [Upbit API 문서](https://docs.upbit.com/reference)
- [Upbit Rate Limits](https://docs.upbit.com/kr/reference/rate-limits)
- [SQLite 최적화 가이드](https://www.sqlite.org/optoverview.html)
- [tqdm 진행률 표시](https://github.com/tqdm/tqdm)

---

**이 문서는 멀티 타임프레임 데이터 파이프라인 개선을 위한 상세 설계서입니다.**
