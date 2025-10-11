# 강화학습 기반 암호화폐 트레이딩 시스템

> **최종 업데이트**: 2025-10-07

강화학습(Reinforcement Learning)을 활용한 자동화된 암호화폐 트레이딩 시스템입니다.

---

## 🚀 주요 기능

### 1. 다양한 RL 모델 지원
- **직접 구현**: DQN, LSTM, Transformer, Ensemble
- **Stable-Baselines3**: PPO, A2C, SAC, TD3, DQN
- 모델 간 성능 비교 및 선택 가능

### 2. 커스텀 지표 및 전략
- **기본 지표**: SMA, EMA, RSI, MACD, Bollinger Bands, ATR 등
- **커스텀 지표**: 눌림목 지수, 지지/저항 강도, 추세 일관성, 변동성 돌파 확률
- **SSL 특성**: Self-Supervised Learning 기반 특성 추출
- 전략 백테스팅 및 성과 분석 도구

### 3. 데이터 관리
- SQLite 기반 데이터 저장/로드
- 오프라인 학습 지원 (캐싱)
- 실시간 데이터 수집 및 처리
- 하이브리드 데이터 파이프라인 (offline/realtime)

### 4. 학습 과정 시각화
- 에피소드별 트레이딩 액션 시각화 (Buy/Sell 표시)
- 스텝별 리워드 추적
- 잔고/포지션 변화 모니터링
- 자동 저장 (`results/visualizations/`)

### 5. 유연한 리워드 설계
- 6가지 리워드 함수 제안
- 매도 인센티브, 위험 조정, 행동 품질 평가
- 실험 및 평가 프레임워크

---

## 📦 설치

```bash
# 1. 저장소 클론
git clone <repository-url>
cd rl

# 2. 환경 설정
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3. 기본 패키지 설치
pip install -r requirements.txt

# 4. Stable-Baselines3 사용 시 (선택)
pip install stable-baselines3

# 5. API 키 설정 (.env 파일 생성)
echo "UPBIT_ACCESS_KEY=your_key" > .env
echo "UPBIT_SECRET_KEY=your_secret" >> .env
```

---

## 🎯 빠른 시작

### 1. 기본 학습 (Upbit API 실시간)

```bash
python run_train.py --episodes 1000 --market KRW-BTC
```

### 2. 오프라인 학습 (SQLite 캐시)

```bash
# 1단계: 데이터 수집
python run_train.py --collect-data --market KRW-BTC --data-count 1000

# 2단계: 학습 (빠름!)
python run_train.py --db data/market_data.db --episodes 1000
```

### 3. SB3 모델 사용

```bash
# PPO 알고리즘으로 학습
python run_train.py --model-type sb3_ppo --episodes 1000

# A2C 알고리즘으로 학습
python run_train.py --model-type sb3_a2c --episodes 500
```

### 4. 커스텀 지표 분석

```python
from trading_env.indicators_custom import add_custom_indicators
from analysis.analyze_indicators import analyze_indicator_vs_returns

# 데이터에 커스텀 지표 추가
df = add_custom_indicators(df)

# 눌림목 지수 분석
analyze_indicator_vs_returns(df, 'pullback_index', forward_periods=10)
```

### 5. 전략 백테스팅

```python
from analysis.strategies import PullbackStrategy, backtest_strategy

# 눌림목 전략 생성
strategy = PullbackStrategy(pullback_threshold=60)

# 백테스팅 실행
result = backtest_strategy(df, strategy)
print(f"수익률: {result['profit_rate']:.2f}%")
```

---

## 📁 프로젝트 구조

```
rl/
├── trading_env/              # 트레이딩 환경
│   ├── rl_env.py            # 강화학습 환경
│   ├── indicators_basic.py  # 기본 기술 지표
│   ├── indicators_custom.py # 커스텀 지표 (눌림목 지수 등)
│   ├── indicators_ssl.py    # SSL 특성 추출
│   ├── data_storage.py      # SQLite 저장/로드
│   └── data_pipeline.py     # 통합 데이터 파이프라인
│
├── models/                   # 신경망 모델
│   ├── dqn.py               # Deep Q-Network
│   ├── lstm.py              # LSTM 모델
│   ├── transformer.py       # Transformer 모델
│   ├── ensemble.py          # 앙상블 모델
│   ├── sb3_wrapper.py       # Stable-Baselines3 통합
│   ├── factory.py           # 모델 팩토리
│   ├── saved/               # 학습된 모델 저장
│   └── SB3_GUIDE.md        # SB3 사용 가이드
│
├── analysis/                 # 전략 분석 도구
│   ├── strategies.py        # 트레이딩 전략
│   ├── backtest_strategies.py # 백테스팅 엔진
│   ├── analyze_indicators.py # 지표 분석
│   └── notebooks/           # Jupyter 노트북
│
├── results/                  # 학습 결과
│   ├── visualizations/      # 시각화 그래프
│   └── backtests/           # 백테스팅 결과
│
├── rl_agent.py              # RL 에이전트
├── run_train.py             # 학습 실행 스크립트
├── run_backtest.py          # 백테스팅 스크립트
└── run_realtime_trading.py # 실시간 트레이딩 스크립트
```

---

## 📚 문서

### 사용 가이드
- **[models/SB3_GUIDE.md](models/SB3_GUIDE.md)** - Stable-Baselines3 사용 가이드
- **[.github/docs/REWARD_DESIGN.md](.github/docs/REWARD_DESIGN.md)** - 리워드 함수 설계 가이드
- **[.github/docs/SSL_FEATURES_GUIDE.md](.github/docs/SSL_FEATURES_GUIDE.md)** - SSL 특성 추출 가이드
- **[docs/SQLITE_USAGE.md](docs/SQLITE_USAGE.md)** - SQLite 데이터 저장 가이드
- **[docs/DATA_PIPELINE_GUIDE.md](docs/DATA_PIPELINE_GUIDE.md)** - 데이터 파이프라인 가이드

### 개발 문서
- **[.github/docs/CHANGELOG.md](.github/docs/CHANGELOG.md)** - 개발 로그
- **[.github/docs/TODO.md](.github/docs/TODO.md)** - TODO 목록
- **[.github/INSTRUCTIONS.md](.github/INSTRUCTIONS.md)** - 개발 지침

### API 문서
- **[upbit_api/README.md](upbit_api/README.md)** - Upbit API 문서
- **[bithumb_api/README.md](bithumb_api/README.md)** - Bithumb API 문서

---

## 🎨 최근 업데이트 (2025-10-07)

### ✨ 새로운 기능
- ✅ **트레이딩 시각화**: 에피소드별 Buy/Sell 액션 및 리워드 그래프 자동 생성
- ✅ **SB3 통합**: Stable-Baselines3 알고리즘 즉시 사용 가능 (PPO, A2C, SAC, TD3, DQN)
- ✅ **커스텀 지표**: 눌림목 지수, 지지/저항 강도, 추세 일관성, 변동성 돌파 확률
- ✅ **전략 분석 도구**: 백테스팅 엔진, 지표 성과 분석, 통계적 검증
- ✅ **리워드 설계 가이드**: 6가지 리워드 함수 제안 및 구현 계획

### 🔧 개선 사항
- ✅ 지표 파일명 일관성 (`indicators_basic`, `indicators_custom`, `indicators_ssl`)
- ✅ 저장 경로 체계화 (`models/saved/`, `results/visualizations/`)
- ✅ 디버깅 정보 자동 출력 (액션 통계, 잔고/포지션 범위)

---

## 🛠️ 지원 모델

### 직접 구현
| 모델 | 설명 | 용도 |
|-----|------|-----|
| DQN | Deep Q-Network | 기본 이산 액션 |
| LSTM | 순환 신경망 | 시계열 패턴 학습 |
| Transformer | Self-Attention | 장기 의존성 |
| Ensemble | 앙상블 모델 | 안정성 향상 |

### Stable-Baselines3
| 알고리즘 | 설명 | 추천 |
|---------|------|-----|
| **PPO** | Proximal Policy Optimization | ⭐ 추천 |
| A2C | Advantage Actor-Critic | 빠른 학습 |
| SAC | Soft Actor-Critic | 연속 액션 |
| TD3 | Twin Delayed DDPG | 고급 제어 |
| DQN | Deep Q-Network | 비교용 |

---

## 📊 성과 지표

학습된 모델은 다음 지표로 평가됩니다:
- **총 수익률**: (최종 자본 - 초기 자본) / 초기 자본
- **샤프 비율**: 위험 조정 수익률
- **최대 낙폭(MDD)**: 최고점 대비 최대 하락폭
- **승률**: 수익 거래 / 전체 거래
- **거래 횟수**: 에피소드당 평균 거래 횟수

---

## 🔬 실험 및 개발

### 커스텀 지표 개발
```python
from trading_env.indicators_custom import CustomIndicators

# 새로운 지표 추가
class MyIndicators(CustomIndicators):
    @staticmethod
    def my_custom_indicator(df, window=20):
        # 지표 계산 로직
        return result
```

### 리워드 함수 실험
```python
# .github/docs/REWARD_DESIGN.md 참고
# 6가지 리워드 함수 중 선택 또는 커스터마이징
```

### 전략 개발
```python
from analysis.strategies import BaseStrategy

class MyStrategy(BaseStrategy):
    def generate_signals(self, df):
        # 시그널 생성 로직
        return signals
```

---

## ⚠️ 주의사항

- 실제 트레이딩 전 충분한 백테스팅 필요
- API 키는 환경변수 또는 `.env` 파일로 관리
- 초기 자본 설정 시 실제 리스크 고려
- 학습 데이터와 실제 시장 차이 고려

---

## 📝 라이선스

이 프로젝트는 개인 학습 및 연구 목적으로 개발되었습니다.

---

## 🤝 기여

버그 리포트, 기능 제안, 코드 기여 환영합니다!

---

## 📧 문의

프로젝트 관련 문의사항은 Issues를 통해 남겨주세요.

---

**Happy Trading! 📈**
