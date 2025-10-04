# 프로젝트 구조 분석

> **최종 업데이트**: 2025년 10월 04일 22:27

## 핵심 모듈 (Core Modules)

### 1. **rl_trading_env.py** (821라인) - 메인 환경
```
역할: 강화학습 환경의 핵심
포함: TradingEnvironment, FeatureExtractor, DataNormalizer
상태: ⚠️ 너무 크다 - 분리 필요
```

### 2. **models.py** (504라인) - 신경망 모델
```
역할: DQN, LSTM, Transformer, Ensemble 모델
포함: ModelConfig, 모든 모델 클래스
상태: ✅ 적절한 크기 - 유지
```

### 3. **dqn_agent.py** (444라인) - 에이전트
```
역할: DQN 에이전트, 학습 로직
포함: DQNAgent, ReplayBuffer, TradingTrainer
상태: ✅ 적절한 크기 - 유지
```

## 거래소 API

### 4. **upbit_api/** - Upbit 거래소
```
upbit_api.py: Upbit API 클라이언트
test_api.py: 테스트 코드
example.py: 사용 예제
상태: ✅ 잘 정리됨 - 유지
```

### 5. **bithumb_api/** - Bithumb 거래소
```
bithumb_api.py: Bithumb API 클라이언트
test_api.py: 테스트 코드
상태: ✅ 잘 정리됨 - 유지
```

## 실행 및 평가 모듈

### 6. **backtesting.py** (509라인) - 백테스팅
```
역할: 과거 데이터 기반 성능 평가
포함: Backtester, 성능 지표, 시각화
상태: ✅ 적절한 크기 - 유지
```

### 7. **real_time_trader.py** (480라인) - 실시간 거래
```
역할: 실시간 거래 실행, 리스크 관리
포함: RealTimeTrader, RiskManager
상태: ✅ 적절한 크기 - 유지
```

### 8. **run_trading_system.py** (217라인) - 통합 실행
```
역할: CLI 인터페이스, 시스템 통합
포함: 명령줄 인터페이스, 설정 관리
상태: ✅ 적절한 크기 - 유지
```

## 테스트 및 예제

### 9. **advanced_example.py** (403라인) - 고급 예제
```
역할: 복잡한 사용 예제
상태: ⚠️ 크다 - 간단하게 줄이거나 분리
```

### 10. **setup_check.py** (350라인) - 설정 검증
```
역할: 프로젝트 환경 검증
상태: ✅ 적절한 크기 - 유지
```

### 11. **quick_start.py** (183라인) - 빠른 시작
```
역할: 기본 기능 테스트
상태: ✅ 적절한 크기 - 유지
```

### 12. **quick_test.py** (82라인) - 간단 테스트
```
역할: 빠른 동작 확인
상태: ✅ 작고 명확 - 유지
```

## 정리 권장사항

### 🚨 즉시 정리가 필요한 파일

#### 1. **rl_trading_env.py 분리** (우선순위: 높음)
현재 821라인으로 너무 크며, 다음과 같이 분리 권장:

```
📂 trading_env/
├── __init__.py
├── environment.py          # TradingEnvironment 클래스
├── feature_extractor.py    # FeatureExtractor 클래스
├── data_normalizer.py      # DataNormalizer 클래스
├── data_collector.py       # UpbitDataCollector 클래스
└── action_space.py         # ActionSpace 관련
```

#### 2. **advanced_example.py 간소화** (우선순위: 중간)
403라인은 예제치고 너무 크므로:

```
📂 examples/
├── __init__.py
├── basic_example.py        # 기본 사용법 (50라인 내외)
├── advanced_training.py    # 고급 학습 예제
├── multi_exchange.py       # 다중 거래소 예제
└── strategy_examples.py    # 전략 예제
```

### ✅ 현재 상태 유지 권장

다음 파일들은 적절한 크기와 명확한 역할을 가지고 있어 현재 상태 유지:

- **models.py** - 모델 아키텍처 (504라인)
- **dqn_agent.py** - 에이전트 구현 (444라인)
- **backtesting.py** - 백테스팅 (509라인)
- **real_time_trader.py** - 실시간 거래 (480라인)
- **run_trading_system.py** - 메인 실행 (217라인)

## 파일 크기 현황

| 파일명 | 라인 수 | 상태 | 우선순위 |
|--------|---------|------|----------|
| **rl_trading_env_backup.py** | 821 | ✅ 분리완료 | - |
| **models.py** | 504 | 🔴 분리필요 | 높음 |
| **run_backtesting.py** | 509 | 🔴 분리필요 | 높음 |
| **run_real_time_trader.py** | 480 | 🟡 검토필요 | 중간 |
| **dqn_agent.py** | 444 | 🟡 검토필요 | 중간 |
| **setup_check.py** | 350 | 🟢 양호 | 낮음 |
| **run_trading_system.py** | 217 | 🟢 양호 | - |
| **quick_start.py** | 183 | 🟢 양호 | - |

## 완료된 작업 ✅
- ✅ **trading_env 패키지 분리** (821라인 → 4개 모듈)
- ✅ **디렉토리 구조 생성** (7개 전용 폴더)
- ✅ **파일 명명 규칙 적용** (run_ 접두사 통일)
- ✅ **개발 지침 문서화** (INSTRUCTIONS.md)
