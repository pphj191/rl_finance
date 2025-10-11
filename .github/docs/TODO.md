# TODO 목록

> **최종 업데이트**: 2025년 10월 06일

## 📊 전체 진행률: 70% (8/11 주요 작업 완료)

---

## 🔴 우선순위 최고 (즉시 실행)

### 0. 멀티 타임프레임 데이터 입력 구조 설계 ⭐ NEW

**현재 상황**:
- RL 에이전트 입력: **분단위 데이터(OHLCV + indicators)만 사용**
- `rl_env.py`의 `_get_observation()`: `lookback_window` 크기의 1분봉 데이터 + 포트폴리오 정보

**목표**:
분단위, 시간단위, 일단위 데이터를 **동시에** 입력으로 활용
- **단기 패턴** (1분봉): 즉각적인 가격 변동, 단기 노이즈
- **중기 패턴** (1시간봉): 트렌드 및 모멘텀, 지지/저항선
- **장기 패턴** (1일봉): 전체적인 시장 방향성, 매크로 트렌드

**검토 필요 항목**:

#### 1. DB 스키마 설계
- [ ] **옵션 1: 각 타임프레임별 별도 테이블**
  ```sql
  CREATE TABLE ohlcv_1m (timestamp, open, high, low, close, volume, ...)
  CREATE TABLE ohlcv_1h (timestamp, open, high, low, close, volume, ...)
  CREATE TABLE ohlcv_1d (timestamp, open, high, low, close, volume, ...)
  CREATE TABLE processed_data_1m (...)
  CREATE TABLE processed_data_1h (...)
  CREATE TABLE processed_data_1d (...)
  ```
  - 장점: 타임프레임별 독립적 관리, 쿼리 성능 최적화 용이
  - 단점: 테이블 증가, 코드 중복 가능성

- [ ] **옵션 2: 단일 테이블 + timeframe 컬럼**
  ```sql
  CREATE TABLE ohlcv (
      market TEXT,
      timeframe TEXT,  -- '1m', '1h', '1d'
      timestamp DATETIME,
      ...
      PRIMARY KEY (market, timeframe, timestamp)
  )
  ```
  - 장점: 단일 테이블로 관리 용이
  - 단점: 쿼리 시 timeframe 필터링 필요, 인덱스 복잡도 증가

- [ ] **옵션 3: processed_data 테이블에 멀티 타임프레임 특성 통합**
  ```sql
  CREATE TABLE processed_data (
      ...
      features_1m BLOB,   -- 1분봉 특성
      features_1h BLOB,   -- 1시간봉 특성
      features_1d BLOB,   -- 1일봉 특성
      ...
  )
  ```
  - 장점: 시간 동기화된 데이터 저장
  - 단점: 각 row마다 모든 타임프레임 데이터 중복 저장

#### 2. DataPipeline 확장
- [ ] `process_multi_timeframe_data()` 메서드 추가
  - 여러 타임프레임 데이터를 동시에 로드
  - 타임스탬프 정렬 및 동기화 (1분 기준으로 1시간, 1일 데이터 매칭)
  - 각 타임프레임별 지표 계산 (SMA, RSI 등의 window는 타임프레임에 맞게 조정)
- [ ] 타임프레임 리샘플링 로직
  - 1분봉 → 1시간봉 aggregation (60개 candle → 1개)
  - 1분봉 → 1일봉 aggregation (1440개 candle → 1개)
  - Upbit API에서 직접 가져오기 vs 로컬 리샘플링 성능 비교

#### 3. Environment 입력 구조 설계
- [ ] **옵션 1: 각 타임프레임을 별도 채널로 concat**
  ```python
  observation = {
      '1m': np.array([60, features_1m]),    # 60분 lookback
      '1h': np.array([24, features_1h]),    # 24시간 lookback
      '1d': np.array([30, features_1d])     # 30일 lookback
  }
  # DQN 네트워크에서 각각 처리 후 fusion
  ```
  - 장점: 타임프레임별 독립적인 패턴 학습
  - 단점: 네트워크 복잡도 증가

- [ ] **옵션 2: Flatten 후 단일 벡터로 결합**
  ```python
  observation = np.concatenate([
      features_1m.flatten(),   # 1분봉 특성
      features_1h.flatten(),   # 1시간봉 특성
      features_1d.flatten(),   # 1일봉 특성
      portfolio_features
  ])
  ```
  - 장점: 기존 DQN 네트워크 구조 재사용 가능
  - 단점: 시계열 정보 손실, 차원 폭발

- [ ] **옵션 3: 각 타임프레임에서 요약 통계만 추출**
  ```python
  summary_1m = extract_summary(features_1m)  # mean, std, min, max, trend 등
  summary_1h = extract_summary(features_1h)
  summary_1d = extract_summary(features_1d)
  observation = np.concatenate([summary_1m, summary_1h, summary_1d, portfolio])
  ```
  - 장점: 차원 관리 용이, 해석 가능성
  - 단점: 정보 손실 가능성

#### 4. RL 네트워크 아키텍처 변경
- [ ] Multi-scale feature extraction
  - 각 타임프레임별 LSTM/CNN 인코더
  - Feature pyramid network 방식
- [ ] Temporal fusion 메커니즘
  - Attention-based fusion (어떤 타임프레임이 중요한지 학습)
  - Hierarchical representation learning
- [ ] 타임프레임별 importance weighting
  - 시장 상황에 따라 동적으로 가중치 조정
  - 예: 급등/급락 시 → 단기 중시, 횡보 시 → 장기 중시

#### 5. 데이터 수집 및 저장
- [ ] Upbit API에서 멀티 타임프레임 데이터 수집 스크립트
  ```python
  # scripts/collect_multi_timeframe_data.py
  collector.collect_all_timeframes(
      market="KRW-BTC",
      timeframes=['1m', '5m', '1h', '1d'],
      days=30
  )
  ```
- [ ] 타임프레임 간 일관성 검증
  - 1분봉 60개 aggregation == 1시간봉 1개 검증
  - 데이터 무결성 체크

**참고 자료**:
- Multi-Horizon Forecasting: https://arxiv.org/abs/1912.09363
- Temporal Fusion Transformers: https://arxiv.org/abs/1912.09363
- Multi-Scale Deep Neural Network: https://arxiv.org/abs/1703.03130

---

## 🔴 우선순위 최고 (즉시 실행)

### 1. 트레이딩 액션 및 리워드 시스템 개선 ⭐ NEW

#### 1-1. 부분 매수/매도 구현
**현재 문제**:
- 전액 매수/매도만 가능 (all-in/all-out)
- 포지션 조절 불가능
- 리스크 관리 어려움

**개선 방안**:
- [ ] **액션 공간 확장**
  - 옵션 A: Discrete(9) - [HOLD, BUY_25%, BUY_50%, BUY_75%, BUY_100%, SELL_25%, SELL_50%, SELL_75%, SELL_100%]
  - 옵션 B: MultiDiscrete([3, 4]) - [BUY/HOLD/SELL, 25%/50%/75%/100%]
  - 옵션 C: Continuous(2) - [action_type (-1~1), amount_ratio (0~1)]

- [ ] **환경 수정** (`trading_env/rl_env.py`)
  ```python
  # _execute_action 수정
  def _execute_action(self, action: int, amount_ratio: float):
      if action == BUY:
          cost = self.balance * amount_ratio
          # ...
      elif action == SELL:
          coins_to_sell = self.position * amount_ratio
          # ...
  ```

- [ ] **에이전트 수정** (`rl_agent.py`)
  - select_action 반환값: (action, amount_ratio)
  - 네트워크 출력: Q-values for each (action, ratio) pair

#### 1-2. 리워드 설계 개선
**상세 설계는 `REWARD_DESIGN.md` 참조**

- [ ] **매도 인센티브 추가**
  - 수익 실현 보너스
  - 손절 최소화 리워드
  - 과도한 보유 페널티

- [ ] **리워드 함수 구현** (`trading_env/reward_functions.py` 생성)
  - [ ] 기본 리워드: 포트폴리오 수익률
  - [ ] 매도 인센티브 리워드
  - [ ] 위험 조정 리워드
  - [ ] 복합 리워드

- [ ] **리워드 테스트 및 비교**
  - [ ] 각 리워드 함수별 학습 결과 비교
  - [ ] 매수/매도 빈도 분석
  - [ ] 수익률 및 안정성 평가

### 2. 실행 스크립트 통합 테스트
- [ ] **run_train.py 테스트**
  - [ ] 모델 생성 확인
  - [ ] 학습 루프 동작 확인
  - [ ] 체크포인트 저장/로드 확인

- [ ] **run_backtest.py 테스트**
  - [ ] 백테스팅 엔진 동작 확인
  - [ ] 성과 지표 계산 확인
  - [ ] 시각화 결과 확인

- [ ] **run_realtime_trading.py 테스트**
  - [ ] API 연결 확인
  - [ ] 실시간 데이터 수집 확인
  - [ ] 거래 실행 로직 확인 (데모 모드)

### 2. Import 경로 최종 검증
- [ ] **models 패키지 import 확인**
  - [ ] 모든 파일에서 `from models import` 정상 동작 확인
  - [ ] 순환 import 없는지 확인

- [ ] **core 모듈 import 확인**
  - [ ] run_backtest.py에서 core 모듈 import 확인
  - [ ] run_realtime_trading.py에서 core 모듈 import 확인

---

## 🟡 우선순위 높음 (이번 주 완료)

### 3. Stable-Baselines3 통합 🆕

**Stable-Baselines3 설치 및 환경 설정**
- [ ] 패키지 설치: `uv add stable-baselines3 sb3-contrib`
- [ ] 의존성 확인 (gymnasium, torch 버전 호환)
- [ ] 설치 가이드 문서 작성

**SB3 래퍼 클래스 구현**
- [ ] `sb3_wrapper/sb3_trading_env.py` - TradingEnvironment를 SB3 호환 Gym 환경으로 변환
- [ ] `sb3_wrapper/sb3_agent.py` - SB3 알고리즘 통합 인터페이스
- [ ] `sb3_wrapper/callbacks.py` - 학습 콜백 및 모니터링

**SB3 알고리즘 테스트 및 비교**
- [ ] PPO (Proximal Policy Optimization) 구현 및 테스트
- [ ] A2C (Advantage Actor-Critic) 구현 및 테스트
- [ ] SAC (Soft Actor-Critic) 구현 및 테스트
- [ ] DQN (기존 직접 구현 vs SB3 구현) 성능 비교
- [ ] 각 알고리즘별 하이퍼파라미터 튜닝

**SB3 실행 스크립트 작성**
- [ ] `run_sb3_training.py` - SB3 학습 실행
- [ ] `run_sb3_comparison.py` - 직접 구현 vs SB3 비교
- [ ] `examples/example_sb3_usage.py` - 사용 예제

**문서화**
- [ ] SB3 vs 직접 구현 비교 가이드
- [ ] SB3 알고리즘별 특성 및 선택 가이드
- [ ] 성능 벤치마크 결과 문서

### 4. 데이터 수집 및 저장 시스템 구축

**SQLite 데이터베이스 설계**
- [ ] 스키마 설계 (orderbook, trades, candles 테이블)
- [ ] 인덱스 최적화
- [ ] 데이터 무결성 제약조건 설정

**실시간 데이터 수집 모듈 구현**
- [ ] WebSocket 기반 orderbook 데이터 수집
- [ ] 거래 체결 데이터 수집
- [ ] 캔들 데이터 수집 및 저장
- [ ] 데이터 정합성 검증 로직

**데이터 관리 유틸리티**
- [ ] 데이터 백업 및 복구 기능
- [ ] 오래된 데이터 정리 기능
- [ ] 데이터 조회 및 분석 헬퍼 함수

### 5. SSL 기반 데이터 예측을 입력으로 사용
**확률적 회귀 모델 구현 (Probabilistic Regression)**
- [ ] **잠시후, 1시간 후, 1일 후 데이터 예측**
- [ ] **현재 추이가 예상과 너무 다른경우 (매크로 시장 변동성)에 대한 감지 및 알림**
- [ ] Gaussian NLL Loss 함수 구현
- [ ] 가격 예측기: 다음 N분 후 가격의 평균(mean)과 분산(variance) 예측
- [ ] 변동성 예측기: 미래 변동성의 평균과 분산 예측
- [ ] 예측 모델 아키텍처 설계 (Encoder + Dual Heads: mean_head, log_var_head)

**불확실성 정량화 (Uncertainty Quantification)**
- [ ] 예측 신뢰구간 계산 (95% Confidence Interval)
- [ ] 신뢰도 스코어 계산 함수: `confidence = 1 / (1 + std)`
- [ ] MC Dropout으로 불확실성 검증
- [ ] 예측 정확도와 신뢰도 상관관계 분석

**RL State 통합**
- [ ] SSL 예측값을 state에 추가 (mean, std, confidence)
- [ ] 신뢰도 기반 reward shaping 구현
- [ ] 예측 오차를 보조 reward로 활용
- [ ] State 차원 확장 테스트 및 검증

**SSL 학습 파이프라인**
- [ ] 자가지도 학습 데이터셋 생성 (과거→미래 예측)
- [ ] SSL 사전 학습(Pre-training) 스크립트 작성
- [ ] RL 학습과 병렬 fine-tuning 구조 설계
- [ ] 학습 안정성 모니터링 (loss, 예측 정확도)

**성능 평가 및 비교**
- [ ] SSL 예측 정확도 측정 (MAE, RMSE, R²)
- [ ] 신뢰구간 커버리지 검증 (실제 값이 예측 구간 내 포함 비율)
- [ ] SSL 유무에 따른 RL 성능 비교 (수익률, 샤프비율)
- [ ] 계산 비용 vs 성능 개선 trade-off 분석

### 6. 중간 크기 파일 최적화
- [ ] **dqn_agent.py (444라인) 리팩토링**
  - [ ] 에이전트 기본 클래스 분리
  - [ ] 메모리 관리 로직 분리
  - [ ] 학습 알고리즘 개선

- [ ] **setup_check.py (350라인) 개선**
  - [ ] 모듈별 체크 분리
  - [ ] 더 상세한 진단 정보 제공

### 7. 테스트 코드 작성

**models/ 패키지 테스트**
- [ ] `tests/test_models_dqn.py`
- [ ] `tests/test_models_lstm.py`
- [ ] `tests/test_models_transformer.py`
- [ ] `tests/test_models_ensemble.py`

**core/ 모듈 테스트**
- [ ] `tests/test_backtesting_engine.py`
- [ ] `tests/test_performance_metrics.py`
- [ ] `tests/test_visualization.py`
- [ ] `tests/test_realtime_trader.py`

**trading_env/ 패키지 테스트**
- [ ] `tests/test_trading_env.py`
- [ ] `tests/test_market_data.py`
- [ ] `tests/test_indicators.py`

**SB3 관련 테스트** 🆕
- [ ] `tests/test_sb3_wrapper.py` - SB3 래퍼 테스트
- [ ] `tests/test_sb3_integration.py` - SB3 vs 직접 구현 통합 테스트

### 8. 예제 코드 보완
**기본 예제**
- [ ] `examples/example_train_model.py` - 모델 학습 예제
- [ ] `examples/example_backtest_simple.py` - 간단한 백테스팅
- [ ] `examples/example_custom_strategy.py` - 커스텀 전략

**고급 예제**
- [ ] `examples/example_multi_exchange.py` - 다중 거래소
- [ ] `examples/example_ensemble_model.py` - 앙상블 모델
- [ ] `examples/example_optimization.py` - 하이퍼파라미터 최적화

---

## 🟢 우선순위 중간 (장기 계획)

### 9. 성능 최적화
**메모리 최적화**
- [ ] 대용량 데이터 스트리밍 처리
- [ ] 메모리 프로파일링 및 최적화
- [ ] 가비지 컬렉션 최적화

**처리 속도 개선**
- [ ] 멀티프로세싱 백테스팅
- [ ] GPU 가속 모델 학습
- [ ] 데이터 로딩 병렬화

### 10. 문서화 자동화
**API 문서 생성**
- [ ] Sphinx 설정
- [ ] Docstring 일관성 검토
- [ ] API 레퍼런스 자동 생성

**사용자 가이드**
- [ ] 설치 가이드 작성
- [ ] 튜토리얼 작성
- [ ] FAQ 작성
- [ ] SB3 사용 가이드 작성 🆕

### 11. CI/CD 구축
**자동화 파이프라인**
- [ ] GitHub Actions 설정
- [ ] 자동 테스트 실행
- [ ] 코드 품질 검사
- [ ] 자동 배포

### 12. 추가 기능
**다른 거래소 지원**
- [ ] 바이낸스 API 연동
- [ ] 코인원 API 연동
- [ ] 통합 거래소 인터페이스

**고급 기능**
- [ ] 실시간 알림 시스템
- [ ] 웹 대시보드
- [ ] 포트폴리오 관리

### 13. 보안 및 운영
**보안 강화**
- [ ] API 키 암호화
- [ ] 로그 민감정보 마스킹
- [ ] 권한 관리

**모니터링**
- [ ] 성능 모니터링
- [ ] 오류 추적
- [ ] 알림 시스템

---

## ✅ 완료된 작업

### 2025-10-05
- [x] **models.py 패키지 분리** (504라인 → 5개 모듈)
- [x] **실행 스크립트 재구성** (run_train, run_backtest, run_realtime_trading)
- [x] **core/ 모듈 생성** (backtesting_engine, performance_metrics, visualization, realtime_trader)
- [x] **문서 구조 개선** (reports/, .github/ 정리)
- [x] **원본 파일 백업** (backup/ 폴더)
- [x] **CHANGELOG 작성**

### 2025-09-30
- [x] **trading_env 패키지 분리** (821라인 → 4개 모듈)
- [x] **디렉토리 구조 생성** (8개 전용 폴더)
- [x] **파일 명명 규칙 통일** (run_ 접두사)
- [x] **백업 시스템 구축**
- [x] **INSTRUCTIONS.md 작성**

---

## 📝 작업 우선순위 가이드

### 🔴 즉시 실행 (1-2일)
- 시스템 안정성에 직접 영향
- 다른 작업의 선행 조건
- 버그 수정 및 긴급 개선

### 🟡 높은 우선순위 (1주일)
- 코드 품질 향상
- 테스트 커버리지 개선
- 중요 기능 추가

### 🟢 중간 우선순위 (1개월)
- 장기 개선 사항
- 편의 기능 추가
- 문서화 완성

---

## 🎯 마일스톤

### 마일스톤 1: 안정화 (70% → 80%)
- [ ] 모든 실행 스크립트 테스트 통과
- [ ] 기본 테스트 코드 작성
- [ ] Import 경로 검증 완료

**목표 완료일**: 2025-10-07

### 마일스톤 2: 품질 향상 (80% → 90%)
- [ ] 중간 크기 파일 최적화
- [ ] 테스트 커버리지 50% 이상
- [ ] 예제 코드 완성

**목표 완료일**: 2025-10-14

### 마일스톤 3: 완성 (90% → 100%)
- [ ] 성능 최적화 완료
- [ ] 문서화 100% 완성
- [ ] CI/CD 구축

**목표 완료일**: 2025-10-31

## 💡 Stable-Baselines3 통합 세부 계획 🆕

### 왜 SB3를 사용하는가?
- ✅ **검증된 구현**: 논문 기반 정확한 알고리즘 구현
- ✅ **다양한 알고리즘**: PPO, A2C, SAC, TD3, DQN 등 즉시 사용
- ✅ **빠른 프로토타이핑**: 직접 구현 대비 개발 시간 단축
- ✅ **성능 비교**: 직접 구현과 비교하여 개선점 발견
- ✅ **커뮤니티 지원**: 활발한 커뮤니티와 문서

### 직접 구현 vs SB3
| 항목 | 직접 구현 (현재) | Stable-Baselines3 |
|------|------------------|-------------------|
| **장점** | 완전한 제어, 커스터마이징 자유 | 검증된 구현, 빠른 개발 |
| **단점** | 버그 가능성, 시간 소요 | 커스터마이징 제약 |
| **용도** | 연구, 실험, 최적화 | 빠른 테스트, 벤치마크 |

### 통합 전략
1. **병행 사용**: 두 가지 방식 모두 유지
2. **비교 분석**: 성능 및 안정성 비교
3. **최적 선택**: 프로덕션에는 더 나은 방식 선택

---

**이 문서는 프로젝트와 함께 지속적으로 업데이트됩니다.**
