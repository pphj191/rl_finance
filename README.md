# 🤖 강화학습 암호화폐 트레이딩 시스템

> **최종 업데이트**: 2025년 10월 05일 15:42

> **AI 기반 자동 암호화폐 거래 시스템** - Upbit & Bithumb 지원

딥러닝과 강화학습을 활용하여 암호화폐 시장에서 자동으로 거래 결정을 내리는 시스템입니다.
복수 거래소를 지원하며, 다양한 AI 모델(DQN, LSTM, Transformer)을 제공합니다.

## ✨ 주요 특징

- 🔗 **다중 거래소**: Upbit, Bithumb 통합 지원
- 🧠 **AI 모델**: DQN, LSTM, Transformer, Ensemble
- 📊 **백테스팅**: 과거 데이터 기반 성능 검증
- ⚡ **실시간 거래**: WebSocket 기반 실시간 트레이딩
- 🛡️ **리스크 관리**: 손절/익절, 포지션 관리

## 🎯 빠른 시작

```bash
# 1. 저장소 클론
git clone <repository-url>
cd rl

# 2. 환경 설정
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3. 패키지 설치
pip install uv
uv add torch gymnasium scikit-learn matplotlib seaborn pandas requests PyJWT websocket-client python-dotenv

# 4. API 키 설정 (.env 파일 생성)
echo "UPBIT_ACCESS_KEY=your_key" > .env
echo "UPBIT_SECRET_KEY=your_secret" >> .env

# 5. 빠른 테스트
python quick_start.py
```

## 📁 프로젝트 구조

```
rl/
├── 📂 models/                 # 신경망 모델 패키지
├── 📂 trading_env/            # 강화학습 트레이딩 환경
├── 📂 upbit_api/              # Upbit API 패키지
├── 📂 bithumb_api/            # Bithumb API 패키지
├── 📂 examples/               # 사용 예제 모음
├── 📂 tests/                  # 테스트 코드
├── 📂 docs/                   # 상세 문서
├── rl_agent.py                # 강화학습 에이전트 (DQN, LSTM, Transformer 지원)
├── run_backtesting.py         # 백테스팅 실행
├── run_real_time_trader.py    # 실시간 트레이딩 실행
├── run_trading_system.py      # 통합 실행 스크립트 (메인 진입점)
├── quick_start.py             # 빠른 시작 가이드
└── setup_check.py             # 프로젝트 구조 확인
```

## 💡 사용법

### 1. 모델 학습

```bash
# 기본 학습
python run_train.py

# 특정 모델로 학습
python run_train.py --model dqn --episodes 1000

# 학습 재개
python run_train.py --resume --checkpoint models/checkpoint.pth
```

### 2. 백테스팅

```bash
# 기본 백테스팅
python run_backtest.py

# 기간 지정
python run_backtest.py --start 2024-01-01 --end 2024-12-31

# 결과 저장
python run_backtest.py --save-results --output results/
```

### 3. 실시간 트레이딩 (⚠️ 실제 자금 사용)

```bash
# 데모 모드 (실제 거래 안함)
python run_realtime_trading.py --demo

# 실제 트레이딩 (주의!)
python run_realtime_trading.py --live
```

**⚠️ 중요**: 실시간 트레이딩은 실제 자금이 사용됩니다. 반드시 데모 모드로 충분히 테스트 후 사용하세요.

## 🌟 주요 구성 요소

### 강화학습 에이전트 (`rl_agent.py`)
- **RLAgent**: DQN, LSTM, Transformer, Ensemble 모델 지원
- **ReplayBuffer**: 경험 재생 버퍼
- **TradingTrainer**: 학습 관리자

### 신경망 모델 (`models/`)
- **DQN**: 기본 Dueling Double DQN
- **LSTM**: LSTM + Attention 기반 시계열 모델
- **Transformer**: Self-Attention 기반 모델
- **Ensemble**: 여러 모델을 결합한 앙상블

### 트레이딩 환경 (`trading_env/`)
- **TradingEnvironment**: Gymnasium 호환 트레이딩 환경
- **FeatureExtractor**: 기술적 지표 추출
- **MarketData**: 실시간 시장 데이터 수집

### 거래소 API
- **Upbit API** (`upbit_api/`): 업비트 거래소 완전 지원
- **Bithumb API** (`bithumb_api/`): 빗썸 거래소 완전 지원

## 📚 문서

### 사용자 가이드
- **[docs/USER_GUIDE.md](docs/USER_GUIDE.md)** - 사용자 매뉴얼
- **[docs/API_REFERENCE.md](docs/API_REFERENCE.md)** - API 레퍼런스
- **[docs/FAQ.md](docs/FAQ.md)** - 자주 묻는 질문

### 개발자 가이드
- **[.github/INSTRUCTIONS.md](.github/INSTRUCTIONS.md)** - 개발 지침서
- **[docs/PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md)** - 프로젝트 구조 상세
- **[docs/CODE_STANDARDS.md](docs/CODE_STANDARDS.md)** - 코드 작성 표준
- **[docs/TODO.md](docs/TODO.md)** - 작업 목록 및 진행 상황
- **[docs/CHANGELOG.md](docs/CHANGELOG.md)** - 개발 로그

## ⚠️ 주의사항

- **실제 거래 주의**: 실시간 트레이딩은 실제 자금이 사용됩니다
- **API 키 보안**: `.env` 파일을 Git에 커밋하지 마세요
- **백테스팅 한계**: 과거 성과가 미래 수익을 보장하지 않습니다
- **리스크 관리**: 손실 감수 가능한 금액으로만 거래하세요

## 📄 라이선스

MIT License

---

**⚡ 성공적인 트레이딩을 위해! ⚡**

> 💡 **팁**: 실제 거래 전에 충분한 백테스팅과 테스트를 거쳐 전략을 검증하세요.
