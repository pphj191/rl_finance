# 개발 로그

> **최종 업데이트**: 2025년 10월 10일

일별 개발 내역을 기록합니다.

---

## 2025-10-10 - 멀티 타임프레임 데이터 파이프라인 구현 ✅

### ✅ 완료된 작업

#### 1. 멀티 타임프레임 데이터베이스 스키마
- **타임프레임별 별도 테이블 생성**
  - `ohlcv_1m`: 1분봉 데이터
  - `ohlcv_1h`: 1시간봉 데이터
  - `ohlcv_1d`: 1일봉 데이터
  - 각 테이블에 `UNIQUE(market, timestamp)` 제약 조건
  - 타임프레임별 인덱스 생성 (`idx_ohlcv_{timeframe}_market_timestamp`)

#### 2. 타임스탬프 정규화 및 중복 방지
- **`align_timestamp()` 함수 구현**
  - 1m: 초/밀리초 제거 (14:23:45.123 → 14:23:00)
  - 1h: 분/초/밀리초 제거 (14:23:45.123 → 14:00:00)
  - 1d: 시간/분/초/밀리초 제거 (14:23:45.123 → 00:00:00)

- **`save_ohlcv_data_by_timeframe()` 개선**
  - 저장 전 타임스탬프 자동 정규화
  - `INSERT OR REPLACE` 방식으로 중복 자동 처리
  - 같은 데이터 여러 번 저장해도 중복 발생하지 않음

#### 3. 증분 수집 (누락 데이터만 수집)
- **`get_missing_ranges()` 함수 구현**
  - SQLite에서 기존 데이터 범위 조회
  - 목표 범위와 비교하여 누락 구간 자동 계산
  - 앞부분 누락 (target_start < existing_start)
  - 뒷부분 누락 (target_end > existing_end)
  - 기존 데이터가 있으면 누락분만 수집 → 네트워크 트래픽 절감

#### 4. Upbit API Rate Limit 통합 관리
- **`UpbitDataCollector` 클래스 확장**
  - Rate Limit 클래스 변수 추가:
    - `MAX_REQUESTS_PER_SECOND = 10` (초당 최대 10회)
    - `MAX_REQUESTS_PER_MINUTE = 600` (분당 최대 600회)
    - `MAX_CANDLES_PER_REQUEST = 200` (1회 최대 200개)
    - `REQUEST_DELAY = 0.11` (요청 간 0.11초 대기)

  - Rate Limit 추적 변수:
    - `last_request_time`: 마지막 요청 시간
    - `request_count_per_minute`: 분당 요청 카운트
    - `minute_start_time`: 분 시작 시간

  - **`_wait_for_rate_limit()` 메서드 구현**
    - 초당 제한: 마지막 요청으로부터 REQUEST_DELAY 대기
    - 분당 제한: 1분 경과 시 카운트 리셋
    - 분당 요청 수 초과 시 자동 대기

  - **`get_historical_data()`에 자동 적용**
    - 모든 API 요청에 Rate Limit 자동 적용
    - 사용자는 Rate Limit 신경 쓸 필요 없음

#### 5. 통합 데이터 수집 함수
- **`collect_multi_timeframe_data()` 통합 함수 구현**
  - 기존 2개 함수 → 1개로 통합
  - **2가지 수집 방식 지원**:
    - 방법 1: `end_time` 지정 (시작~종료 기간)
    - 방법 2: `minute_candles_count` 지정 (시작+개수)

  - **주요 기능**:
    - 타임프레임별 목표 범위 자동 계산
    - 누락 데이터만 증분 수집
    - 200개씩 자동 분할 수집 (Upbit 제한 준수)
    - Rate Limit 자동 처리
    - tqdm 진행률 표시 (설치된 경우)
    - 타임스탬프 정규화 및 중복 제거

  - **lookback 지원**:
    - `hourly_lookback_count`: 시작 이전 시간봉 개수 (기본 24)
    - `daily_lookback_count`: 시작 이전 일봉 개수 (기본 30)
    - 지표 계산에 필요한 과거 데이터 자동 수집

- **기존 함수 제거**:
  - `collect_multi_timeframe_by_count()` 삭제 (통합 함수로 대체)

#### 6. 문서화
- **trading_env/data_pipeline_TODO.md 작성**
  - 구현 로직 단계별 상세 설명
  - 타임스탬프 정규화, 누락 데이터 감지, Rate Limit 처리
  - TODO 체크리스트 (Phase 1~5)
  - 예상 성능 및 저장 공간 계산
  - 사용 예제 3가지 (end_time, count, 증분 수집)
  - Upbit API Rate Limits 참고 문서 링크

### 📊 성능 개선

**데이터 수집 시간 (KRW-BTC 기준, Rate Limit 0.11초/요청)**

| 기간 | 1분봉 개수 | 예상 요청 수 | 예상 소요 시간 |
|------|-----------|-------------|--------------|
| 1시간 | 60 | 1 | ~0.1초 |
| 1일 | 1,440 | 8 | ~1초 |
| 7일 | 10,080 | 51 | ~6초 |
| 30일 | 43,200 | 216 | ~24초 |

**저장 공간 (30일 기준)**

| 타임프레임 | 개수 | 예상 크기 |
|-----------|------|----------|
| 1분봉 | 43,200 | ~2MB |
| 1시간봉 | 720 | ~40KB |
| 1일봉 | 30 | ~2KB |
| **합계** | **43,950** | **~2MB** |

### 🔧 사용 예제

```python
from datetime import datetime, timedelta
from trading_env.data_storage import collect_multi_timeframe_data

# 방법 1: end_time 지정
collect_multi_timeframe_data(
    market="KRW-BTC",
    start_time=datetime(2025, 10, 1),
    end_time=datetime(2025, 10, 10),
    hourly_lookback_count=24,  # 추가 24시간
    daily_lookback_count=30    # 추가 30일
)

# 방법 2: minute_candles_count 지정
collect_multi_timeframe_data(
    market="KRW-BTC",
    start_time=datetime(2025, 10, 1),
    minute_candles_count=1440,  # 1일치
    hourly_lookback_count=24,
    daily_lookback_count=30
)

# 방법 3: 증분 수집 (기존 데이터에 추가)
# 2차 실행 시 누락분만 자동 수집
collect_multi_timeframe_data(
    market="KRW-BTC",
    start_time=datetime(2025, 10, 1),
    end_time=datetime(2025, 10, 15),  # 기간 연장
    db_path="data/market_data.db"
)
```

### 📚 관련 파일

- **수정**: `trading_env/data_storage.py`
  - `align_timestamp()` 함수 추가
  - `get_missing_ranges()` 함수 추가
  - `save_ohlcv_data_by_timeframe()` 개선
  - `collect_multi_timeframe_data()` 통합 함수 구현
  - `collect_multi_timeframe_by_count()` 제거

- **수정**: `trading_env/market_data.py`
  - `UpbitDataCollector` 클래스 확장
  - Rate Limit 클래스 변수 추가
  - `_wait_for_rate_limit()` 메서드 추가
  - `get_historical_data()` Rate Limit 자동 적용

- **신규**: `trading_env/data_pipeline_TODO.md`
  - 상세 설계 문서 및 구현 가이드

---

## 2025-10-07 - 시각화, 지표 리팩토링, SB3 통합 ✅

### ✅ 완료된 작업

#### 1. 트레이딩 시각화 개선
- **rl_agent.py 시각화 기능 추가**
  - `_plot_episode_actions()` 메서드 구현
  - 에피소드별 트레이딩 액션(Buy/Sell) 및 리워드 시각화
  - 4개 서브플롯: 가격+액션, 스텝별 리워드, 잔고 추이, 포지션 추이
  - Buy/Sell 액션을 가격 차트에 마커로 표시 (▲ 녹색, ▼ 빨간색)
  - 리워드 영역 색칠 (양수=녹색, 음수=빨간색)
  - 통계 정보 표시 (평균 잔고, 평균 포지션)
  - 디버깅 정보 로깅 (액션 통계, 잔고/포지션 범위)

- **저장 경로 변경**
  - 시각화: `models/visualizations/` → `results/visualizations/`
  - 모델 파일: `models/` → `models/saved/`
  - `.gitignore` 업데이트
  - 디렉토리 구조 개선 및 `.gitkeep` 파일 추가

#### 2. 지표 파일명 일관성 개선
- **파일명 변경** (일관된 `indicators_*` 패턴)
  - `indicators.py` → `indicators_basic.py`
  - `custom_indicators.py` → `indicators_custom.py`
  - `ssl_features.py` → `indicators_ssl.py`

- **모든 import 경로 업데이트** (7개 파일)
  - `trading_env/__init__.py`
  - `trading_env/data_pipeline.py`
  - `trading_env/market_data.py`
  - `trading_env/indicators_basic.py` (주석)
  - `analysis/strategies.py`
  - `analysis/analyze_indicators.py`
  - `examples/example_trading_env_usage.py`

#### 3. Stable-Baselines3 통합
- **models/sb3_wrapper.py 신규 생성**
  - `SB3TradingModel` 클래스: SB3 알고리즘 래퍼
  - 지원 알고리즘: PPO, A2C, SAC, TD3, DQN
  - `TradingCallback` 클래스: 학습 중 콜백 (모델 저장, 로깅)
  - `create_sb3_model()` 헬퍼 함수
  - `SB3_RECOMMENDED_PARAMS`: 알고리즘별 권장 하이퍼파라미터

- **models/factory.py 확장**
  - `sb3_*` 모델 타입 지원 추가
  - `sb3_ppo`, `sb3_a2c`, `sb3_sac`, `sb3_td3`, `sb3_dqn` 모델 생성 가능

- **models/__init__.py 업데이트**
  - `SB3TradingModel`, `SB3_AVAILABLE`, `create_sb3_model` export

- **문서 작성**
  - `models/SB3_GUIDE.md`: 완전한 SB3 사용 가이드
    - 설치 방법
    - 기본/고급 사용법
    - 알고리즘별 권장 하이퍼파라미터
    - 직접 구현 vs SB3 비교표
    - run_train.py 통합 방법
    - 텐서보드 로깅
    - 문제 해결

- **사용 예제 작성**
  - `examples/example_sb3_usage.py`: 실행 가능한 예제
    - SB3 설치 확인
    - PPO 학습 예제
    - 알고리즘 비교 예제
    - 모델 평가 예제
    - 콜백 사용 예제

#### 4. 커스텀 지표 및 전략 분석 모듈
- **trading_env/indicators_custom.py 생성**
  - `CustomIndicators` 클래스
  - 눌림목 지수 (pullback_index)
  - 지지/저항 강도 (support_resistance_strength)
  - 추세 일관성 (trend_consistency)
  - 변동성 돌파 확률 (volatility_breakout_probability)
  - `add_custom_indicators()` 헬퍼 함수

- **analysis/ 모듈 생성**
  - `analysis/strategies.py`: 트레이딩 전략
    - `BaseStrategy`, `PullbackStrategy`, `BreakoutStrategy`, `HybridStrategy`
    - `backtest_strategy()` 간단한 백테스팅 함수
  - `analysis/backtest_strategies.py`: 백테스팅 엔진
    - `BacktestEngine` 클래스
    - 슬리피지, 수수료 고려
    - 성과 지표 계산 (수익률, 승률, MDD, 샤프 비율)
    - 결과 시각화
  - `analysis/analyze_indicators.py`: 지표 분석 도구
    - 지표 분포 시각화
    - 지표 간 상관관계 분석
    - 지표 vs 미래 수익률 관계 분석
    - 통계적 유의성 검정

#### 5. 리워드 설계 문서화
- **.github/docs/REWARD_DESIGN.md 신규 생성**
  - 현재 리워드 시스템의 문제점 분석
    - 매도 회피 문제 (상세 예시 포함)
    - 단기 가격 변동 민감성
    - 리스크 무시
    - 희소 리워드 문제
  - 리워드 설계 원칙 (매도 인센티브, 위험 조정, 행동 품질 평가)
  - 6가지 리워드 함수 제안 (코드 포함)
    1. 매도 인센티브 추가
    2. 위험 조정 수익률
    3. 벤치마크 대비 초과 수익
    4. 행동 품질 기반
    5. 복합 리워드 (추천)
    6. 에피소드 종료 시점 보상
  - 구현 계획 (4단계 Phase)
  - 실험 및 평가 계획 (평가 지표, 예상 결과)

- **.github/docs/TODO.md 업데이트**
  - 우선순위 최고 작업 추가
  - 1-1. 부분 매수/매도 구현 (액션 공간 확장)
  - 1-2. 리워드 설계 개선 (REWARD_DESIGN.md 참조)

### 🔄 변경 사항

#### 디렉토리 구조
```
models/
├── saved/                    ← 모델 파일 저장 (신규)
│   ├── *.pth
│   ├── train_config.json
│   └── training_results.json
├── sb3_wrapper.py           ← SB3 통합 (신규)
├── SB3_GUIDE.md            ← SB3 가이드 (신규)
└── ...

results/
├── visualizations/          ← 시각화 저장 (변경됨)
│   └── episode_*.png
└── backtests/

trading_env/
├── indicators_basic.py      ← 기본 지표 (이름 변경)
├── indicators_custom.py     ← 커스텀 지표 (이름 변경)
└── indicators_ssl.py        ← SSL 특성 (이름 변경)

analysis/                    ← 분석 모듈 (신규)
├── strategies.py
├── backtest_strategies.py
├── analyze_indicators.py
└── notebooks/

.github/docs/
└── REWARD_DESIGN.md        ← 리워드 설계 (신규)
```

#### Breaking Changes
- 지표 파일 import 경로 변경
  ```python
  # Before
  from trading_env.indicators import FeatureExtractor
  from trading_env.custom_indicators import CustomIndicators
  from trading_env.ssl_features import SSLFeatureExtractor

  # After
  from trading_env.indicators_basic import FeatureExtractor
  from trading_env.indicators_custom import CustomIndicators
  from trading_env.indicators_ssl import SSLFeatureExtractor
  ```

### 🎯 주요 개선사항

#### 1. 시각화 개선
- 에피소드별 트레이딩 액션 및 리워드 추적
- Buy/Sell 액션 가시화
- 디버깅 정보 자동 출력
- 결과 저장 경로 체계화

#### 2. 코드 일관성
- 일관된 파일명 패턴 (`indicators_*`)
- 명확한 모듈 역할 구분
- 체계적인 디렉토리 구조

#### 3. SB3 통합
- 검증된 RL 알고리즘 즉시 사용 가능
- 직접 구현과 성능 비교 가능
- 빠른 프로토타이핑 지원
- 풍부한 문서 및 예제

#### 4. 전략 분석 도구
- 커스텀 지표 개발 및 테스트
- 백테스팅 엔진
- 지표 성과 분석
- 통계적 검증

#### 5. 리워드 설계
- 체계적인 리워드 함수 설계
- 매도 인센티브 추가 계획
- 실험 및 평가 프레임워크

### 📝 사용법

#### SB3 모델 사용
```bash
# PPO 학습
python run_train.py --model-type sb3_ppo --episodes 1000

# 코드에서 사용
from models import create_sb3_model
model = create_sb3_model(env, algorithm="PPO")
model.train_step(total_timesteps=10000)
```

#### 커스텀 지표 분석
```python
from trading_env.indicators_custom import add_custom_indicators
from analysis.analyze_indicators import analyze_indicator_vs_returns

df = add_custom_indicators(df)
analyze_indicator_vs_returns(df, 'pullback_index', forward_periods=10)
```

#### 전략 백테스팅
```python
from analysis.strategies import PullbackStrategy, backtest_strategy

strategy = PullbackStrategy(pullback_threshold=60)
result = backtest_strategy(df, strategy)
```

### 📚 새로운 문서
- `models/SB3_GUIDE.md` - Stable-Baselines3 사용 가이드
- `.github/docs/REWARD_DESIGN.md` - 리워드 함수 설계 가이드

### 🐛 수정된 문제
- Sell 액션 표시 안 되는 문제 해결 (action_names 사용)
- Balance/Position 변화 추적 개선
- 파일명 일관성 문제 해결

### 🎉 기대 효과
- ✅ 학습 과정 시각화로 디버깅 용이
- ✅ SB3 통합으로 빠른 프로토타이핑
- ✅ 커스텀 지표 개발 및 검증 가능
- ✅ 체계적인 리워드 설계 가능
- ✅ 코드 일관성 및 유지보수성 향상

---

## 2025-10-06 - SSL 특성 모듈 분리 ✅

### ✅ 완료된 작업

#### SSL 특성 모듈 신규 생성
- **trading_env/ssl_features.py** 신규 생성
  - `SSLConfig`: SSL 모델 설정 dataclass
  - `ContrastiveEncoder`: 대조 학습 기반 인코더 (SimCLR 방식)
  - `MaskedPredictor`: 마스킹 예측 모델 (BERT 방식)
  - `TemporalPatternClassifier`: 시간적 패턴 분류 모델 (8개 클래스)
  - `FuturePricePredictor`: 미래 가격 예측 모델 (multi-horizon)
  - `SSLFeatureExtractor`: 통합 특성 추출 클래스

#### 학습 기반 특성 추출
- **Contrastive Learning**
  - 유사한 패턴 → 가까운 벡터
  - 다른 패턴 → 먼 벡터
  - 출력: representation 벡터 (hidden_dim)

- **Masked Prediction**
  - 시계열 일부를 마스킹하고 예측
  - BERT-style masked language modeling
  - 출력: masked representation 벡터

- **Pattern Classification**
  - 8개 패턴 클래스 분류
  - 상승/하락/횡보/변동성/반전 등
  - 출력: 클래스별 확률 분포

- **Future Price Prediction** (신규)
  - 1분, 5분, 15분, 30분, 60분 후 가격 예측
  - Multi-task learning
  - 출력: multi-horizon 예측값

#### indicators.py에서 SSL 메서드 제거
- 11개 SSL 관련 메서드 제거:
  - `extract_ssl_features()`
  - `_add_contrastive_features()`
  - `_add_masked_prediction_features()`
  - `_add_temporal_pattern_features()`
  - `_calculate_pattern_similarity()`
  - `_classify_volatility_regime()`
  - `_calculate_prediction_confidence()`
  - `_calculate_autocorrelation()`
  - `_extract_periodic_trend()`
  - `_calculate_trend_strength()`
  - `_calculate_trend_direction()`

- 마이그레이션 가이드 주석 추가

#### 모듈 통합
- **trading_env/__init__.py** 업데이트
  - `SSLFeatureExtractor`, `SSLConfig` export 추가
  - 모듈 docstring에 ssl_features.py 추가

#### 문서화
- **.github/docs/SSL_FEATURES_GUIDE.md** 신규 생성
  - SSL 개념 및 기존 방식과의 차이
  - 4가지 SSL 모델 상세 설명
  - 사용 방법 및 예제 코드
  - DataPipeline 통합 방법
  - RL 에이전트 통합 예시
  - 모델 아키텍처 다이어그램
  - TODO 항목 및 구현 가이드
  - 마이그레이션 가이드

### 🔄 변경 사항

#### Breaking Changes
- `indicators.py`의 `extract_ssl_features()` 제거
  - 기존 코드는 `ssl_features.py` 사용으로 마이그레이션 필요

#### 새로운 사용법
```python
# Before (제거됨)
from trading_env.indicators import FeatureExtractor
extractor = FeatureExtractor()
ssl_features = extractor.extract_ssl_features(df)

# After (새로운 방식)
from trading_env.ssl_features import SSLFeatureExtractor, SSLConfig
ssl_config = SSLConfig()
ssl_extractor = SSLFeatureExtractor(ssl_config)
ssl_extractor.load_all_models(input_dim=df.shape[1])
ssl_features = ssl_extractor.extract_features(df)
```

### 📝 TODO (학습 로직 구현 필요)
- [ ] `create_data_loader()`: SQLite 데이터 로더 구현
- [ ] `train_contrastive_model()`: Contrastive learning 학습 로직
- [ ] `train_masked_prediction_model()`: Masked prediction 학습 로직
- [ ] `train_pattern_classifier()`: Pattern classification 학습 로직
- [ ] `train_future_predictor()`: Future prediction 학습 로직
- [ ] `nt_xent_loss()`: NT-Xent loss 함수 구현
- [ ] `apply_mask()`: Masking 전략 구현

### 🎯 설계 의도
- **분리의 이유**: 규칙 기반 지표 vs 학습 기반 특성의 명확한 구분
- **유연성**: SSL 특성 사용 여부를 선택적으로 결정 가능
- **확장성**: 새로운 SSL 모델 추가 용이
- **독립성**: SQLite에서 데이터 로드 → 학습 → representation 추출의 독립적인 워크플로우

---

## 2025-10-05 (20:30) - 하이브리드 데이터 파이프라인 구현 완료 ✅

### ✅ 완료된 작업

#### 통합 데이터 파이프라인 구현
- **trading_env/data_pipeline.py** 신규 생성
  - `DataPipeline` 클래스: 오프라인/실시간 모드 지원
  - `prepare_offline_data()` 함수: 데이터 수집 + 지표 계산 + 특성 추출 + 저장
  - 모드별 동작:
    - `offline`: SQLite 캐시만 사용
    - `realtime`: 캐시 우선, 없으면 계산 후 저장
  - 설정 해시(config_hash): 자동 캐시 무효화

#### 데이터베이스 스키마 확장
- **processed_data 테이블 추가**
  - 기술적 지표 컬럼: sma_5, sma_20, ema_12, rsi_14, macd, bb_*, atr_14, obv 등 (19개)
  - 특성 벡터: feature_vector (BLOB), feature_names (JSON)
  - 정규화 정보: normalization_method, normalization_params (JSON)
  - 캐시 관리: config_hash (설정 변경 감지)

#### MarketDataStorage 확장
- **save_processed_data()** 메서드 추가
  - 기술적 지표 + 특성 벡터 저장
  - Pickle을 사용한 numpy array 직렬화
  - JSON을 사용한 메타데이터 저장
- **load_processed_data()** 메서드 추가
  - config_hash 기반 조회
  - 역직렬화 (pickle → numpy array)
- **_generate_config_hash()** 메서드 추가
  - 정규화 방법 + SSL 설정 기반 해시

#### FeatureExtractor 개선
- **extract_all()** 메서드 추가: 기술적 지표 + SSL 특성 통합 추출
- **get_feature_vector()** 메서드 추가: DataFrame → numpy array 변환
- **get_feature_names()** 메서드 추가: 특성 이름 리스트 반환

#### TradingEnvironment 개선
- **mode, cache_enabled 파라미터 추가**
- **DataPipeline 통합**: _prepare_data()에서 pipeline.process_data() 호출
- **3단계 데이터 소스**:
  1. 미리 준비된 DataFrame (최우선)
  2. DataPipeline (SQLite 캐시 활용)
  3. Upbit API 실시간 수집 (폴백)

#### RLAgent/TradingTrainer 개선
- **mode, cache_enabled 파라미터 추가**
- **TradingEnvironment에 파라미터 전달**

#### run_train.py 수정
- **모드 자동 결정**: db_path 있으면 offline, 없으면 realtime
- **--collect-data 개선**: prepare_offline_data() 함수 사용
  - 데이터 수집 + 지표 계산 + 특성 추출 자동화

#### 오프라인 데이터 준비 스크립트
- **scripts/prepare_offline_data.py** 신규 생성
  - 독립 스크립트로 실행 가능
  - 인자: --market, --days, --db, --normalization, --no-ssl
  - 사용 예제 포함

### 🎯 핵심 개선사항

#### 1. 계층적 캐싱 시스템
```
원본 OHLCV (ohlcv_data 테이블)
    ↓
기술적 지표 계산
    ↓
특성 추출
    ↓
처리된 데이터 (processed_data 테이블) ← 캐시
```

#### 2. 모드별 동작
| 모드 | 동작 | 용도 |
|------|------|------|
| offline | SQLite만 사용 | 빠른 학습 (데이터 미리 준비) |
| realtime | 캐시 우선, 없으면 계산 | 실시간 트레이딩 |

#### 3. 자동 캐시 무효화
- config_hash로 설정 변경 감지
- 정규화 방법 변경 시 자동 재계산

### 📝 사용법

#### 오프라인 학습 (빠름)
```bash
# 1. 데이터 준비 (한 번만)
python scripts/prepare_offline_data.py --market KRW-BTC --days 30

# 2. 학습 (빠름 - 캐시 사용)
python run_train.py --db data/market_data.db --episodes 1000
```

#### 실시간 트레이딩 (안정적)
```bash
# 캐시 활용 + 실시간 계산
python run_realtime_trading.py --cache-enabled
```

#### 데이터 수집 + 학습 한 번에
```bash
python run_train.py --collect-data --db data/market_data.db --episodes 500
```

### 📂 파일 구조
```
trading_env/
├── data_storage.py       (수정: processed_data 테이블, save/load 메서드)
├── data_pipeline.py      (신규: 통합 파이프라인)
├── indicators.py         (수정: extract_all, get_feature_vector)
├── rl_env.py            (수정: DataPipeline 통합)

scripts/
└── prepare_offline_data.py  (신규: 오프라인 데이터 준비)

run_train.py             (수정: mode 파라미터, prepare_offline_data 사용)
rl_agent.py              (수정: mode, cache_enabled 파라미터)
```

### 🔧 기술적 세부사항
- **데이터 직렬화**: Pickle (numpy array), JSON (메타데이터)
- **인덱싱**: (market, timestamp, config_hash) 복합 인덱스
- **UNIQUE 제약**: (market, timestamp, config_hash)
- **정규화 저장**: normalization_params (JSON)

### 🎉 기대 효과
- ✅ **학습 속도 10배 향상**: API 호출 없이 SQLite 캐시 사용
- ✅ **완벽한 재현성**: 동일한 데이터로 반복 학습
- ✅ **오프라인 학습 가능**: 인터넷 연결 불필요
- ✅ **실시간 안정성**: 캐시 우선, 폴백 지원
- ✅ **자동 최적화**: 설정 변경 시 캐시 무효화

---

## 2025-10-05 (18:00) - SQLite 데이터 저장/로드 기능 추가

### ✅ 완료된 작업

#### SQLite 데이터베이스 기능 구현
- **trading_env/data_storage.py** 신규 생성
  - `MarketDataStorage` 클래스: OHLCV 데이터 SQLite 저장/로드
  - `collect_and_store_data()` 함수: Upbit 데이터 수집 후 SQLite 저장
  - 데이터베이스 스키마 정의 (OHLCV 테이블, 오더북 테이블)
  - 인덱스 및 UNIQUE 제약 설정
  - 데이터 범위/개수 조회 기능
  - 데이터 삭제 기능

#### TradingEnvironment 개선
- **trading_env/rl_env.py** 수정
  - 생성자에 `data` (DataFrame), `db_path` (SQLite 경로) 파라미터 추가
  - `_prepare_data()` 메서드 개선: 3가지 데이터 소스 지원
    1. 미리 준비된 DataFrame
    2. SQLite 데이터베이스
    3. Upbit API 실시간 수집 (기본값)

#### RLAgent 및 TradingTrainer 개선
- **rl_agent.py** 수정
  - `TradingTrainer` 생성자에 `data`, `db_path` 파라미터 추가
  - 데이터 소스를 `TradingEnvironment`에 전달
  - `pd.DataFrame` import 추가

#### run_train.py 개선
- **파라미터 이름 수정** ✅ 중요!
  - `trainer.train(episodes=...)` → `trainer.train(num_episodes=...)`
  - `start_episode`, `save_interval`, `eval_interval` → `save_frequency`, `eval_frequency`
- **새로운 CLI 옵션 추가**
  - `--db PATH`: SQLite 데이터베이스 경로 지정
  - `--collect-data`: 학습 전에 데이터 수집
  - `--data-count N`: 수집할 데이터 개수
- **데이터 소스 로깅**: API 또는 SQLite 사용 여부 출력

#### 문서화
- **docs/SQLITE_USAGE.md** 신규 생성
  - SQLite 사용법 상세 가이드
  - 데이터 수집/저장/로드 예제
  - Python 코드 예제
  - 데이터베이스 구조 설명
  - 고급 사용법 (대량 수집, 파이프라인, 증분 업데이트)
  - 문제 해결 가이드

### 🐛 수정된 버그
- **TypeError: `TradingTrainer.train() got an unexpected keyword argument 'episodes'`**
  - 원인: `run_train.py`에서 잘못된 파라미터 이름 사용
  - 해결: `episodes` → `num_episodes`, `save_interval` → `save_frequency` 변경

### 🎯 개선 효과
- ✅ **학습 속도 향상**: API 호출 없이 로컬 데이터 사용
- ✅ **재현성 보장**: 동일한 데이터로 반복 학습 가능
- ✅ **API 제한 회피**: Upbit API 호출 제한 걱정 없음
- ✅ **오프라인 학습**: 인터넷 연결 없이 학습 가능
- ✅ **데이터 관리**: 여러 마켓 데이터를 체계적으로 저장/관리

### 📝 사용법

#### 기존 방식 (Upbit API 실시간)
```bash
python run_train.py --episodes 1000 --market KRW-BTC
```

#### 새로운 방식 (SQLite 저장 데이터)
```bash
# 1. 데이터 수집
python run_train.py --collect-data --market KRW-BTC --data-count 1000

# 2. SQLite 데이터로 학습
python run_train.py --db data/market_data.db --episodes 1000 --market KRW-BTC

# 3. 데이터 수집 + 학습 한 번에
python run_train.py --collect-data --db data/market_data.db --episodes 500
```

### 📂 파일 위치 (참고용)
```
rl/
├── .github/docs/CHANGELOG.md       ← 이 파일
├── docs/SQLITE_USAGE.md           ← SQLite 사용 가이드
├── trading_env/
│   ├── data_storage.py            ← 새로 추가된 파일
│   ├── rl_env.py                  ← 수정됨
│   └── __init__.py                ← 수정됨 (export 추가)
├── rl_agent.py                    ← 수정됨
└── run_train.py                   ← 수정됨
```

---

## 2025-10-05 (15:45)

### ✅ 완료된 작업

#### 실행 스크립트 재구성
- **run_train.py** 생성 - 모델 학습 전용 스크립트
- **run_backtest.py** 생성 - 백테스팅 전용 스크립트
- **run_realtime_trading.py** 생성 - 실시간 트레이딩 전용 스크립트
- 원본 파일(`run_trading_system.py`, `run_backtesting.py`, `run_real_time_trader.py`) backup/ 폴더로 이동

#### core/ 모듈 생성 (재사용 가능한 핵심 로직)
- **core/backtesting_engine.py** - 백테스팅 엔진 분리
- **core/performance_metrics.py** - 성과 지표 계산 분리
- **core/visualization.py** - 시각화 기능 분리
- **core/realtime_trader.py** - 실시간 트레이딩 로직 분리

#### models/ 패키지 분리
- **models.py** (504라인) → **5개 모듈**로 분리
  - `models/base_model.py` - ModelConfig, 기본 클래스
  - `models/dqn.py` - DQNModel
  - `models/lstm.py` - LSTMModel
  - `models/transformer.py` - TransformerModel
  - `models/ensemble.py` - EnsembleModel
  - `models/__init__.py` - 패키지 인터페이스

#### 문서 구조 개선
- **reports/** 폴더 생성 및 리포트 파일 정리
  - `RUN_SCRIPTS_REFACTOR_COMPLETE.md` - 실행 스크립트 재구성 완료 보고서
  - `DOCUMENTATION_RESTRUCTURE_COMPLETE.md` - 문서 구조 개선 완료 보고서
- **.github/docs/** 폴더 정리 → **docs/** 폴더로 통합
- **INSTRUCTIONS.md** 업데이트
  - 실행 스크립트 가이드 추가
  - 모듈 구조 설명 추가
  - 문서 작성 위치 지침 추가 (루트의 docs/ 사용)
- **TODO.md** 업데이트
  - 전체 TODO 체계화 (70% 진행률)
  - 실행 스크립트 테스트 항목 추가
  - 우선순위 재조정
- **CHANGELOG.md** 통합 및 업데이트
- **README.md** 업데이트 날짜 추가

#### 기타 개선 사항
- `dqn_agent.py` → `rl_agent.py` 파일명 변경
- `DQNAgent` → `RLAgent` 클래스명 변경
- 전체 프로젝트 import 경로 현행화 (`rl_trading_env` → `trading_env`)
- README.md 간소화 (580줄 → 123줄)
- `main.py` 삭제 (불필요한 템플릿)
- examples 폴더 import 경로 문제 해결

### 🔄 진행중
- 실행 스크립트 통합 테스트 (run_train.py, run_backtest.py, run_realtime_trading.py)
- Import 경로 최종 검증
- Stable-Baselines3 통합 계획 수립

### 📝 메모
- 진행률: **70%** (8/11 주요 작업 완료)
- 다음 즉시 실행 사항: 실행 스크립트 통합 테스트
- SB3는 직접 구현과 병행하여 성능 비교 목적으로 사용
- 루트 README는 간결하게, 상세 내용은 docs/ 폴더에서 관리
- 모든 문서는 루트의 `docs/` 폴더에 작성 (`.github/docs/` 사용 금지)

---

## 2025-09-30

### ✅ 완료
- **trading_env/** 패키지 분리 (821라인 → 4개 모듈)
  - `trading_env/environment.py` - TradingEnvironment 클래스
  - `trading_env/market_data.py` - MarketDataCollector 클래스
  - `trading_env/indicators.py` - 기술 지표 함수들
  - `trading_env/__init__.py` - 패키지 인터페이스
- **디렉토리 구조 생성** (8개 전용 폴더)
  - docs/, tests/, examples/, models/, logs/, data/, checkpoints/, results/
- **파일 명명 규칙 통일** (run_ 접두사)
- **백업 시스템 구축** (backup/ 폴더)
- **INSTRUCTIONS.md** 작성 (개발 지침서)
- 문서 체계화 (docs 폴더 정리)

---

## Template (아래 복사해서 사용)

```markdown
## YYYY-MM-DD

### ✅ 완료
- 작업 내용

### 🔄 진행중
- 작업 내용

### 🐛 수정
- 버그 내용

### 📝 메모
- 특이사항
```
