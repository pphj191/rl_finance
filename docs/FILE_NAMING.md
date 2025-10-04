# 파일 명명 규칙

> **최종 업데이트**: 2025년 10월 04일 22:27

## 파일 명명 규칙

### 실행 파일 (run_ 접두사 통일)
```python
run_backtesting.py        # 백테스팅 실행
run_real_time_trader.py   # 실시간 트레이딩 실행
run_trading_system.py     # 메인 트레이딩 시스템
```

### 테스트 파일 (test_ 접두사 필수)
```python
test_models.py            # 모델 테스트
test_dqn_agent.py         # 에이전트 테스트
test_backtesting.py       # 백테스팅 테스트
```

### 예제 파일 (example_ 접두사 권장)
```python
example_basic_usage.py        # 기본 사용법 예제
example_multi_exchange.py     # 다중 거래소 예제
example_advanced_trading.py   # 고급 트레이딩 예제
```

### 핵심 기능 파일
```python
models.py              # 신경망 모델
dqn_agent.py          # DQN 에이전트
trading_env/          # 트레이딩 환경 패키지
```

### 유틸리티 파일
```python
setup_check.py        # 환경 검증
quick_start.py        # 빠른 시작 도구
```

### 백업 파일 (backup/ 폴더에 위치)
```python
filename_backup.py    # 원본파일명_backup.확장자 형식
backup_manager.sh     # 백업 관리 스크립트
```

## 백업 파일 관리 규칙

```bash
# 백업 파일 생성 (자동으로 backup/ 폴더에 저장)
./scripts/backup_manager.sh -c models.py

# 기존 백업 파일들을 backup/ 폴더로 정리
./scripts/backup_manager.sh -m

# 백업 파일 목록 확인
./scripts/backup_manager.sh -l

# 백업 파일 명명 규칙
# 원본: models.py → 백업: backup/models_backup.py
# 원본: config.json → 백업: backup/config_backup.json
```

## 디렉토리 구조 규칙

```
📂 rl/                    # 프로젝트 루트
├── 📂 core/              # 핵심 기능 모듈
├── 📂 tests/             # 모든 테스트 파일
├── 📂 examples/          # 모든 예제 파일
├── 📂 docs/              # 문서 파일
├── 📂 scripts/           # 유틸리티 스크립트
├── 📂 backup/            # 백업 파일 보관소
├── 📂 trading_env/       # 트레이딩 환경 패키지
├── 📂 upbit_api/         # Upbit API 패키지
├── 📂 bithumb_api/       # Bithumb API 패키지
├── run_*.py              # 실행 파일들
├── *.py                  # 핵심 기능 파일들
└── README.md             # 프로젝트 문서
```
