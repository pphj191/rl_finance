# 🤖 강화학습 암호화폐 트레이딩 시스템

> **최종 업데이트**: 2024년 10월 4일 01:25

> **AI 기반 자동 암호화폐 거래 시스템** - Upbit & Bithumb 지원

딥러닝과 강화학습을 활용하여 암호화폐 시장에서 자동으로 거래 결정을 내리는 시스템입니다.
복수 거래소를 지원하며, 다양한 AI 모델(DQN, LSTM, Transformer)을 제공합니다.

## ✨ 주요 특징

- 🔗 **다중 거래소**: Upbit, Bithumb 통합 지원
- 🧠 **AI 모델**: DQN, LSTM, Transformer, Ensemble
- 📊 **백테스팅**: 과거 데이터 기반 성능 검증  
- ⚡ **실시간 거래**: WebSocket 기반 실시간 트레이딩
- 🛡️ **리스크 관리**: 손절/익절, 포지션 관리
- 🔧 **쉬운 설정**: 가상환경 및 uv 패키지 관리자 지원

## 🎯 빠른 시작

### 30초 만에 시작하기

```bash
# 1. 저장소 클론
git clone <repository-url>
cd rl

# 2. 환경 설정
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3. 패키지 설치
pip install uv
uv add torch gymnasium scikit-learn matplotlib pandas requests python-dotenv

# 4. 설정 확인
python setup_check.py

# 5. 빠른 테스트
python quick_test.py
```

### API 키 설정

`.env` 파일을 생성하고 거래소 API 키 추가:

```env
# 최소 하나의 거래소 API 키 필요
UPBIT_ACCESS_KEY=your_upbit_access_key
UPBIT_SECRET_KEY=your_upbit_secret_key

# 선택사항: Bithumb API 키
BITHUMB_ACCESS_KEY=your_bithumb_access_key  
BITHUMB_SECRET_KEY=your_bithumb_secret_key
```

## 🚀 프로젝트 설정 가이드

### 1. 환경 준비

#### Python 가상환경 설정
```bash
# Python 가상환경 생성
python -m venv .venv

# 가상환경 활성화 (macOS/Linux)
source .venv/bin/activate

# 가상환경 활성화 (Windows)
.venv\Scripts\activate
```

#### 패키지 관리자 설치 (uv 사용)
```bash
# uv 설치
pip install uv

# 패키지 설치
uv add torch gymnasium scikit-learn matplotlib seaborn pandas numpy
uv add requests PyJWT websocket-client python-dotenv ta
```

### 2. 환경 설정

`.env` 파일을 생성하고 거래소 API 키를 설정:

```env
# Upbit API 키
UPBIT_ACCESS_KEY=your_upbit_access_key_here
UPBIT_SECRET_KEY=your_upbit_secret_key_here

# Bithumb API 키 (NEW!)
BITHUMB_ACCESS_KEY=your_bithumb_access_key_here
BITHUMB_SECRET_KEY=your_bithumb_secret_key_here
```

**⚠️ 중요**: `.env` 파일을 Git에 커밋하지 마세요!

### 3. 프로젝트 구조 확인

```bash
# 프로젝트 구조 검증
python setup_check.py
```

## 📁 프로젝트 구조

```
rl/
├── 📂 upbit_api/              # Upbit API 패키지
│   ├── __init__.py            # 패키지 초기화
│   ├── upbit_api.py           # Upbit API 클래스
│   └── README.md              # API 상세 문서
├── 📂 bithumb_api/            # Bithumb API 패키지 (NEW!)
│   ├── __init__.py            # 패키지 초기화
│   ├── bithumb_api.py         # Bithumb API 클래스
│   ├── README.md              # API 상세 문서
│   └── test_api.py            # 패키지 테스트
├── 📂 models/                 # 신경망 모델 저장 폴더
├── rl_trading_env.py          # 강화학습 환경
├── models.py                  # 신경망 모델 아키텍처
├── dqn_agent.py              # DQN 에이전트 및 학습 로직
├── backtesting.py            # 백테스팅 시스템
├── real_time_trader.py       # 실시간 트레이딩
├── run_trading_system.py     # 통합 실행 스크립트
├── setup_check.py            # 프로젝트 구조 확인
├── quick_test.py             # 빠른 테스트
├── example.py                # 기본 사용 예제
├── advanced_example.py       # 고급 사용 예제
├── test.py                   # 종합 테스트
├── README.md                 # 프로젝트 문서
└── .env                      # 환경 설정 (API 키)
```

## 🌟 주요 구성 요소

### 🏪 거래소 API (`upbit_api/`, `bithumb_api/`)
- **통합 인터페이스**: 동일한 함수명으로 여러 거래소 지원
- **Upbit API**: 업비트 거래소 완전 지원
- **Bithumb API**: 빗썸 거래소 완전 지원 (NEW!)
- **자동 전환**: 거래소 간 쉬운 전환 가능

### 🧠 신경망 모델 (`models.py`)
- **DQNModel**: 기본 Dueling Double DQN
- **LSTMModel**: LSTM + Attention 기반 시계열 모델
- **TransformerModel**: Self-Attention 기반 모델
- **EnsembleModel**: 여러 모델을 결합한 앙상블

### 🏋️ 강화학습 에이전트 (`dqn_agent.py`)
- **DQNAgent**: Q-learning 기반 의사결정 에이전트
- **ReplayBuffer**: 경험 재생 버퍼
- **TradingTrainer**: 통합 학습 관리자

### 🌍 트레이딩 환경 (`rl_trading_env.py`)
- **TradingEnvironment**: Gymnasium 호환 트레이딩 환경
- **FeatureExtractor**: 기술적 지표 및 SSL 특성 추출
- **DataNormalizer**: 다양한 정규화 방법 제공
- **UpbitDataCollector**: 실시간 시장 데이터 수집

### 📊 백테스팅 및 실거래 (`backtesting.py`, `real_time_trader.py`)
- **Backtester**: 과거 데이터 기반 성능 평가
- **RealTimeTrader**: 실시간 거래 실행
- **RiskManager**: 포지션 및 리스크 관리

## 💡 사용법

### 0. 다중 거래소 지원 🆕

이제 Upbit과 Bithumb 두 거래소를 모두 지원합니다!

```python
# Upbit 사용
from upbit_api import UpbitAPI
upbit = UpbitAPI()
ticker_upbit = upbit.get_ticker('KRW-BTC')

# Bithumb 사용 (동일한 인터페이스!)
from bithumb_api import BithumbAPI  
bithumb = BithumbAPI()
ticker_bithumb = bithumb.get_ticker('KRW-BTC')

# 간편한 전환을 위한 팩토리 패턴
def get_exchange_api(exchange='upbit'):
    if exchange == 'upbit':
        from upbit_api import UpbitAPI
        return UpbitAPI()
    elif exchange == 'bithumb':
        from bithumb_api import BithumbAPI
        return BithumbAPI()
    else:
        raise ValueError(f"Unsupported exchange: {exchange}")

# 사용 예시
api = get_exchange_api('bithumb')  # 거래소 선택
markets = api.get_market_all()     # 동일한 함수명
```

### 1. 빠른 시작

```bash
# 전체 시스템 테스트
python quick_test.py

# 기본 사용 예제
python example.py

# 고급 사용 예제  
python advanced_example.py
```

### 2. 모델 학습

```bash
# DQN 모델 학습
python run_trading_system.py --mode train --model-type dqn --episodes 1000

# LSTM 모델 학습
python run_trading_system.py --mode train --model-type lstm --episodes 1000

# Transformer 모델 학습
python run_trading_system.py --mode train --model-type transformer --episodes 1000

# 앙상블 모델 학습
python run_trading_system.py --mode train --model-type ensemble --episodes 1000
```

### 3. 백테스팅

```bash
# 학습된 모델 백테스팅
python run_trading_system.py --mode backtest --model models/best_model.pth --start-date 2024-01-01 --end-date 2024-12-31

# 특정 모델 타입으로 백테스팅
python run_trading_system.py --mode backtest --model-type dqn --model models/dqn_model.pth
```

### 4. 실시간 트레이딩 (⚠️ 실제 자금 사용)

```bash
# 실시간 트레이딩 시작
python run_trading_system.py --mode live --model models/best_model.pth --duration 60

# 데모 모드로 실행 (실제 거래 없음)
python run_trading_system.py --mode live --model models/best_model.pth --demo
```

## � 설정 옵션

### 모델 설정 (`models.py`)
```python
from models import ModelConfig, PRESET_CONFIGS

# 사전 정의된 설정 사용
config = PRESET_CONFIGS["medium_lstm"]

# 커스텀 설정
config = ModelConfig(
    model_type="transformer",    # "dqn", "lstm", "transformer", "ensemble"
    hidden_size=512,            # 은닉층 크기
    d_model=512,               # Transformer 모델 차원
    nhead=8,                   # Attention 헤드 수
    sequence_length=60,        # 시계열 길이
    dropout=0.1                # 드롭아웃 비율
)
```

### 트레이딩 설정 (`rl_trading_env.py`)
```python
from rl_trading_env import TradingConfig

config = TradingConfig(
    model_type="dqn",
    hidden_size=256,
    learning_rate=0.001,
    batch_size=32,
    memory_size=10000,
    normalization="robust",    # "standard", "minmax", "robust"
    initial_balance=1000000,   # 초기 자금 (KRW)
    transaction_fee=0.0005,    # 거래 수수료 (0.05%)
    max_position_size=0.3      # 최대 포지션 크기 (30%)
)
```

### 리스크 관리 설정
```python
from real_time_trader import RiskConfig

risk_config = RiskConfig(
    max_position_size=0.1,     # 최대 포지션 크기 (10%)
    stop_loss_pct=0.05,        # 손절 비율 (5%)
    take_profit_pct=0.1,       # 익절 비율 (10%)
    max_daily_trades=10,       # 일일 최대 거래 수
    min_trade_interval=300,    # 최소 거래 간격 (초)
    max_drawdown_pct=0.2       # 최대 낙폭 (20%)
)
```

## 🧪 모델 성능 비교

| 모델 타입 | 파라미터 수 | 학습 시간 | 메모리 사용량 | 추천 용도 |
|----------|------------|----------|-------------|-----------|
| DQN | ~100K | 빠름 | 낮음 | 빠른 프로토타이핑 |
| LSTM | ~500K | 보통 | 보통 | 시계열 패턴 학습 |
| Transformer | ~1M+ | 느림 | 높음 | 복잡한 패턴 인식 |
| Ensemble | ~1.5M+ | 매우 느림 | 매우 높음 | 최고 성능 |

## 📊 특성 추출 (Feature Engineering)

### 기술적 지표
- **트렌드**: SMA, EMA, MACD
- **변동성**: 볼린저 밴드, ATR
- **모멘텀**: RSI, 스토캐스틱
- **거래량**: OBV, VWAP

### SSL 특성 (Self-Supervised Learning)
- **대조 학습**: 가격 패턴 유사도 분석
- **마스킹 예측**: 미래 가격 예측 신뢰도
- **시간적 패턴**: 주기성 및 트렌드 분석

## 🛡️ 리스크 관리

### 포지션 관리
- 최대 포지션 크기 제한
- 손절매/익절매 자동 실행
- 일일 거래 횟수 제한
- 동적 포지션 크기 조정

### 자금 관리
- 최대 낙폭 모니터링
- 최소 거래 간격 설정
- 거래 수수료 고려
- 슬리피지 관리

## 📈 성과 지표

- **총 수익률**: 전체 기간 수익률
- **연간 수익률**: 연환산 수익률  
- **샤프 비율**: 위험 대비 수익률
- **소르티노 비율**: 하방 위험 대비 수익률
- **최대 낙폭**: 최대 손실 구간
- **승률**: 수익 거래 비율
- **수익 팩터**: 총 이익/총 손실
- **칼마 비율**: 연간 수익률/최대 낙폭

## 🔍 디버깅 및 모니터링

### 로그 설정
```python
import logging

# 로그 레벨 설정
logging.basicConfig(level=logging.INFO)

# 상세 로그 확인
logging.basicConfig(level=logging.DEBUG)
```

### 성능 모니터링
```bash
# 실시간 성능 모니터링
python run_trading_system.py --mode monitor --model models/best_model.pth

# 백테스팅 결과 시각화
python run_trading_system.py --mode visualize --results results/backtest_results.json
```

## 🧩 확장성

### 새로운 모델 추가
1. `models.py`에 새 모델 클래스 추가
2. `create_model` 함수에 모델 타입 등록
3. `ModelConfig`에 새 설정 추가

### 새로운 특성 추가  
1. `FeatureExtractor` 클래스 확장
2. 정규화 방법 추가
3. SSL 특성 확장

### 새로운 거래소 지원
1. 새 API 클래스 작성
2. `DataCollector` 인터페이스 구현
3. 환경 설정 업데이트

## ⚠️ 주의사항

### 실제 거래 관련
- **자금 관리**: 손실 감수 가능한 금액으로만 거래
- **리스크 관리**: 적절한 손절매 설정 필수
- **API 제한**: Upbit API 호출 제한 준수
- **세금**: 암호화폐 거래 세금 고려

### 기술적 주의사항
- **과적합**: 백테스팅에서만 좋은 성능 주의
- **슬리피지**: 실제 거래 시 예상보다 큰 슬리피지 발생 가능
- **지연**: 네트워크 지연으로 인한 가격 차이
- **데이터 품질**: 누락된 데이터나 이상치 처리

## 📁 프로젝트 파일 구조 분석

### 🎯 핵심 모듈 (권장 유지)

| 파일명 | 크기 | 역할 | 상태 |
|--------|------|------|------|
| `rl_trading_env.py` | 821줄 | 강화학습 환경 | ⚠️ **분리 필요** |
| `models.py` | 504줄 | 신경망 모델 | ✅ 적절 |
| `dqn_agent.py` | 444줄 | DQN 에이전트 | ✅ 적절 |
| `backtesting.py` | 509줄 | 백테스팅 | ✅ 적절 |
| `real_time_trader.py` | 480줄 | 실시간 거래 | ✅ 적절 |

### 🔧 유틸리티 & 예제

| 파일명 | 크기 | 역할 | 상태 |
|--------|------|------|------|
| `advanced_example.py` | 403줄 | 고급 예제 | ⚠️ **간소화 권장** |
| `setup_check.py` | 350줄 | 환경 검증 | ✅ 적절 |
| `run_trading_system.py` | 217줄 | 통합 실행 | ✅ 적절 |
| `quick_start.py` | 183줄 | 빠른 시작 | ✅ 적절 |
| `quick_test.py` | 82줄 | 간단 테스트 | ✅ 적절 |

### 📋 정리 권장사항

#### 🚨 즉시 정리 필요
1. **`rl_trading_env.py` 분리** (821줄 → 여러 파일)
   ```
   trading_env/
   ├── environment.py      # TradingEnvironment
   ├── feature_extractor.py # FeatureExtractor  
   ├── data_normalizer.py   # DataNormalizer
   └── action_space.py      # ActionSpace
   ```

2. **`advanced_example.py` 간소화** (403줄 → 100줄 이하)
   ```
   examples/
   ├── basic_usage.py      # 기본 사용법
   ├── multi_exchange.py   # 다중 거래소
   └── strategy_demo.py    # 전략 예제
   ```

#### ✅ 현재 상태 유지
- 나머지 모든 파일들은 적절한 크기와 명확한 역할
- 거래소 API 패키지들은 잘 구조화됨

## 📚 개발자 가이드

### 🔗 문서
- **[INSTRUCTIONS.md](./INSTRUCTIONS.md)** - 개발 지침서 (핵심)
- **[docs/TODO.md](./docs/TODO.md)** - 작업 목록 및 진행 상황
- **[docs/PROJECT_STRUCTURE.md](./docs/PROJECT_STRUCTURE.md)** - 프로젝트 구조 분석
- **[docs/FILE_NAMING.md](./docs/FILE_NAMING.md)** - 파일 명명 규칙
- **[docs/CODE_STANDARDS.md](./docs/CODE_STANDARDS.md)** - 코드 작성 표준
- **[docs/DEVELOPMENT_WORKFLOW.md](./docs/DEVELOPMENT_WORKFLOW.md)** - 개발 워크플로우
- **[upbit_api/README.md](./upbit_api/README.md)** - Upbit API 문서
- **[bithumb_api/README.md](./bithumb_api/README.md)** - Bithumb API 문서

### 🛠️ 개발 환경
```bash
# 개발 의존성 설치
uv add --dev pytest black isort mypy

# 코드 품질 검사
python -m black .
python -m isort .
python -m mypy .
```

### 📊 파일 크기 모니터링
```bash
# 파일 크기 확인
wc -l *.py | sort -nr

# 500줄 초과 파일 찾기
find . -name "*.py" -exec wc -l {} + | awk '$1 > 500 {print $0}'
```

## 🔧 문제 해결

### 일반적인 문제들

#### 1. 패키지 설치 오류
```bash
# uv 재설치
pip uninstall uv
pip install uv

# 캐시 클리어
uv cache clean
```

#### 2. GPU 사용 오류
```python
# CUDA 사용 가능 여부 확인
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device count: {torch.cuda.device_count()}")
```

#### 3. API 연결 오류
```python
# API 키 확인
from upbit_api import UpbitAPI
api = UpbitAPI()
print(api.get_accounts())  # 계정 정보 확인
```

#### 4. 메모리 부족
```python
# 배치 크기 줄이기
config.batch_size = 16  # 기본값: 32

# 시퀀스 길이 줄이기  
config.sequence_length = 30  # 기본값: 60
```

## 📚 추가 자료

### 강화학습 이론
- [Sutton & Barto - Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book.html)
- [Deep Q-Networks (DQN) 논문](https://arxiv.org/abs/1312.5602)
- [Dueling DQN 논문](https://arxiv.org/abs/1511.06581)

### 금융 강화학습
- [Financial Trading with RL](https://arxiv.org/abs/1911.10107)
- [Deep Reinforcement Learning for Trading](https://arxiv.org/abs/2106.00123)

### Upbit API 문서
- [Upbit 개발자 센터](https://docs.upbit.com/)
- [API 사용 가이드](https://docs.upbit.com/docs/upbit-quotation-restful-api)

## 🤝 기여하기

1. 이슈 생성 또는 기존 이슈 선택
2. 브랜치 생성: `git checkout -b feature/새기능`
3. 변경사항 커밋: `git commit -m "새 기능 추가"`
4. 브랜치에 푸시: `git push origin feature/새기능`  
5. Pull Request 생성

## 📄 라이선스

MIT License - 자세한 내용은 LICENSE 파일 참조

## 📞 지원

- **이슈**: GitHub Issues 페이지에서 버그 리포트 및 기능 요청
- **토론**: GitHub Discussions에서 질문 및 아이디어 공유

---

**⚡ 성공적인 트레이딩을 위해! ⚡**

> 💡 **팁**: 실제 거래 전에 충분한 백테스팅과 테스트를 거쳐 전략을 검증하세요.
├── backtesting.py         # 백테스팅 시스템
├── real_time_trader.py    # 실시간 트레이딩
├── run_trading_system.py  # 통합 실행 스크립트
├── setup_check.py         # 프로젝트 구조 확인
├── quick_test.py          # 빠른 테스트
├── example.py             # 기본 사용 예제
├── advanced_example.py    # 고급 사용 예제
├── test.py                # 종합 테스트
├── README.md              # 프로젝트 문서
└── .env                   # 환경 설정 (API 키)
```

## 🚀 설치 및 설정

### 1. 의존성 설치

```bash
pip install torch gymnasium scikit-learn matplotlib seaborn pandas numpy requests PyJWT websocket-client python-dotenv ta
```

### 2. 환경 설정

`.env` 파일을 생성하고 Upbit API 키를 설정:

```env
UPBIT_ACCESS_KEY=your_access_key_here
UPBIT_SECRET_KEY=your_secret_key_here
```

## 💡 사용법

### 1. 모델 학습

```bash
python run_trading_system.py --mode train --episodes 1000
```

### 2. 백테스팅

```bash
python run_trading_system.py --mode backtest --model models/best_model.pth --start-date 2024-01-01 --end-date 2024-12-31
```

### 3. 실시간 트레이딩

```bash
python run_trading_system.py --mode live --model models/best_model.pth --duration 60
```

### 4. 빠른 테스트

```bash
python quick_test.py
```

## 🏗️ 시스템 아키텍처

### 강화학습 환경 (rl_trading_env.py)
- **TradingEnvironment**: Gymnasium 호환 트레이딩 환경
- **FeatureExtractor**: 기술적 지표 및 SSL 특성 추출
- **DataNormalizer**: 다양한 정규화 방법 제공
- **UpbitDataCollector**: 실시간 시장 데이터 수집

### 모델 구조
- **DQNModel**: 기본 Deep Q-Network
- **LSTMModel**: LSTM 기반 시계열 모델
- **TransformerModel**: Self-Attention 기반 모델
- **EnsembleModel**: 여러 모델을 결합한 앙상블

### 트레이딩 시스템
- **DQNAgent**: Q-learning 기반 의사결정
- **RealTimeTrader**: 실시간 거래 실행
- **RiskManager**: 포지션 및 리스크 관리
- **Backtester**: 과거 데이터 기반 성능 평가

## 📊 특성 추출 (Feature Engineering)

### 기술적 지표
- 이동평균 (SMA, EMA)
- 볼린저 밴드
- RSI (Relative Strength Index)
- MACD
- 스토캐스틱
- ATR (Average True Range)
- OBV (On-Balance Volume)

### SSL 특성 (Self-Supervised Learning)
- **대조 학습**: 가격 패턴 유사도 분석
- **마스킹 예측**: 미래 가격 예측 신뢰도
- **시간적 패턴**: 주기성 및 트렌드 분석

## 🛡️ 리스크 관리

### 포지션 관리
- 최대 포지션 크기 제한
- 손절매/익절매 자동 실행
- 일일 거래 횟수 제한

### 자금 관리
- 최대 낙폭 모니터링
- 동적 포지션 크기 조정
- 최소 거래 간격 설정

## 📈 성과 지표

- **총 수익률**: 전체 기간 수익률
- **연간 수익률**: 연환산 수익률
- **샤프 비율**: 위험 대비 수익률
- **최대 낙폭**: 최대 손실 구간
- **승률**: 수익 거래 비율
- **수익 팩터**: 총 이익/총 손실

## 🔧 설정 옵션

### TradingConfig
```python
config = TradingConfig(
    model_type="dqn",          # "dqn", "lstm", "transformer", "ensemble"
    hidden_size=256,           # 은닉층 크기
    learning_rate=0.001,       # 학습률
    batch_size=32,             # 배치 크기
    memory_size=10000,         # 경험 재생 버퍼 크기
    normalization="robust",    # "standard", "minmax", "robust"
    sequence_length=60,        # LSTM/Transformer 시퀀스 길이
    d_model=256,              # Transformer 모델 차원
    nhead=8,                  # Transformer 헤드 수
)
```

### RiskConfig
```python
risk_config = RiskConfig(
    max_position_size=0.1,     # 최대 포지션 크기 (10%)
    stop_loss_pct=0.05,        # 손절 비율 (5%)
    take_profit_pct=0.1,       # 익절 비율 (10%)
    max_daily_trades=10,       # 일일 최대 거래 수
    min_trade_interval=300,    # 최소 거래 간격 (초)
    max_drawdown_pct=0.2,      # 최대 낙폭 (20%)
)
```

## 📋 사용 예제

### 기본 사용법
```python
from rl_trading_env import TradingEnvironment, TradingConfig
from dqn_agent import DQNAgent, TradingTrainer

# 설정
config = TradingConfig(model_type="dqn", hidden_size=256)

# 학습
trainer = TradingTrainer(config, market="KRW-BTC")
results = trainer.train(episodes=1000)

# 백테스팅
from backtesting import Backtester
backtester = Backtester(config)
backtest_results = backtester.run_backtest(trainer.agent, trainer.env)
```

### 고급 사용법
```python
# 앙상블 모델 사용
config = TradingConfig(model_type="ensemble")

# SSL 특성 추출
from rl_trading_env import FeatureExtractor
extractor = FeatureExtractor()
features = extractor.extract_ssl_features(market_data)

# 실시간 트레이딩
from real_time_trader import RealTimeTrader, RiskConfig
trader = RealTimeTrader(config, RiskConfig(), "models/best_model.pth")
trader.start_trading()
```

## ⚠️ 주의사항

1. **실제 거래 주의**: 실시간 트레이딩은 실제 자금이 사용됩니다.
2. **API 키 보안**: `.env` 파일을 Git에 커밋하지 마세요.
3. **백테스팅 한계**: 과거 성과가 미래 수익을 보장하지 않습니다.
4. **리스크 관리**: 적절한 리스크 관리 없이 큰 금액을 투자하지 마세요.

## 🤝 기여

버그 리포트, 기능 제안, 코드 기여를 환영합니다!

## 📄 라이선스

MIT License

## 📞 지원

문제가 있으시면 이슈를 등록해 주세요.

---

**⚡ Happy Trading! ⚡**
