# 개발 로그

> **최종 업데이트**: 2025년 10월 05일 20:30

일별 개발 내역을 기록합니다.

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
