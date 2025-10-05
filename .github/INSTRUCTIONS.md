# 개발 지침서 (Development Instructions)

> **최종 업데이트**: 2025년 10월 05일 18:15

**Claude Code, github Copilot 위한 프로젝트 개발 핵심 규칙**

---

## 📋 문서 참조 우선순위

개발 중 상세한 정보가 필요할 때 다음 순서로 문서를 참조하세요:

1. **이 파일 (INSTRUCTIONS.md)** - 핵심 규칙 및 빠른 참조
2. **[.github/docs/TODO.md](docs/TODO.md)** - 현재 작업 목록 및 우선순위
3. **[.github/docs/PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md)** - 프로젝트 구조 상세
4. **[.github/docs/CODE_STANDARDS.md](docs/CODE_STANDARDS.md)** - 코드 작성 표준
5. **[.github/docs/DEVELOPMENT_WORKFLOW.md](docs/DEVELOPMENT_WORKFLOW.md)** - 개발 워크플로우
6. **[.github/docs/CHANGELOG.md](docs/CHANGELOG.md)** - 개발 이력

---

## 🎯 핵심 개발 규칙

### 1. 파일 크기 관리
- **최대 500라인** 권장
- **800라인 초과** 시 분리 검토
- **1000라인 초과** 시 필수 분리

### 2. 파일 명명 규칙
| 파일 타입 | 접두사/위치 | 예시 |
|---------|----------|------|
| 실행 스크립트 | `run_*.py` | `run_train.py`, `run_backtest.py` |
| 테스트 파일 | `test_*.py` (tests/) | `tests/test_models_dqn.py` |
| 예제 파일 | `example_*.py` (examples/) | `examples/example_trading.py` |
| 코어 모듈 | `core/*.py` | `core/backtesting_engine.py` |

### 3. 모듈 구조
```
rl/
├── models/              # 신경망 모델 (DQN, LSTM, Transformer, Ensemble)
├── trading_env/         # 트레이딩 환경 (RL 환경, 데이터 수집, 지표)
├── core/                # 핵심 로직 (백테스팅, 성과측정, 시각화, 실시간 트레이딩)
├── upbit_api/           # Upbit API 클라이언트
├── bithumb_api/         # Bithumb API 클라이언트
├── tests/               # 테스트 코드
├── examples/            # 사용 예제
└── docs/                # 📚 모든 문서는 여기에 작성
```

### 4. 코드 품질 기준
- ✅ **단일 책임 원칙** 준수
- ✅ **Type hints** 완전 적용
- ✅ **Docstring** 작성 필수 (Google 스타일)
- ✅ **함수 크기** 20라인 이하 권장
- ✅ **의존성 최소화** (순환 import 방지)

### 5. 문서화 규칙

#### 📍 문서 작성 위치
**⚠️ 중요: 개발 관련 문서는 `/.github/docs/` 폴더에 작성**

```
✅ 올바른 위치 (개발 문서):
  /.github/docs/TODO.md
  /.github/docs/CHANGELOG.md
  /.github/docs/PROJECT_STRUCTURE.md
  /.github/docs/CODE_STANDARDS.md

📚 사용자 문서 위치:
  /docs/SQLITE_USAGE.md     (사용자 가이드)
  /docs/API_REFERENCE.md    (API 레퍼런스)
  /docs/FAQ.md              (자주 묻는 질문)
```

#### 📅 문서 업데이트 날짜 표시
모든 `.md` 파일 상단에 명시:
```markdown
> **최종 업데이트**: 2025년 10월 05일 16:00
```

#### 📝 개발 로그 작성
매일 작업 후 `.github/docs/CHANGELOG.md`에 기록:
```markdown
## 2025-10-05
### ✅ 완료
- 작업 내용
### 🔄 진행중
- 작업 내용
### 🐛 수정
- 버그 내용
```

---

## 🚀 실행 스크립트

### 1. 학습 (run_train.py)
```bash
python run_train.py --model dqn --episodes 1000
```

### 2. 백테스팅 (run_backtest.py)
```bash
python run_backtest.py --start 2024-01-01 --end 2024-12-31
```

### 3. 실시간 트레이딩 (run_realtime_trading.py)
```bash
# 데모 모드
python run_realtime_trading.py --demo

# 실제 트레이딩 (⚠️ 주의!)
python run_realtime_trading.py --live
```

---

## ⚠️ 개발 시 주의사항

### 필수 체크리스트
- [ ] 파일 크기 500라인 이하 유지
- [ ] Type hints 모든 함수에 적용
- [ ] Docstring 작성 완료
- [ ] 순환 import 없음
- [ ] 테스트 코드 작성 (새 기능)
- [ ] API 키 하드코딩 없음
- [ ] 변경 사항 CHANGELOG.md에 기록

### 보안 주의사항
- 🔒 API 키는 `.env` 파일에만 저장
- 🔒 `.env` 파일 Git 커밋 절대 금지
- 🔒 로그에 민감 정보 출력 금지

---

## 🔧 유용한 명령어

### 파일 크기 확인
```bash
# 500줄 초과 파일 찾기
find . -name "*.py" -exec wc -l {} + | awk '$1 > 500 {print $0}'
```

### 코드 품질 검사
```bash
# 포맷팅 + 타입 검사
python -m black . && python -m mypy .
```

### 테스트 실행
```bash
# 모든 테스트 + 커버리지
python -m pytest tests/ --cov=.
```

---

## 📌 현재 우선순위 (2025-10-05)

상세 내용은 **[.github/docs/TODO.md](docs/TODO.md)** 참조

### 🔴 즉시 실행
1. 실행 스크립트 통합 테스트
2. Import 경로 최종 검증

### 🟡 이번 주 완료
3. Stable-Baselines3 통합
4. 데이터 수집 시스템 구축
5. SSL 기반 데이터 예측

### 🟢 장기 계획
- 성능 최적화
- 문서화 자동화
- CI/CD 구축

**진행률: 70%** (8/11 주요 작업 완료)

---

## 📚 상세 문서 링크

### 개발 문서 (/.github/docs/)
- **[.github/docs/TODO.md](docs/TODO.md)** - 작업 목록 및 진행 상황
- **[.github/docs/CHANGELOG.md](docs/CHANGELOG.md)** - 개발 이력
- **[.github/docs/PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md)** - 프로젝트 구조 상세
- **[.github/docs/CODE_STANDARDS.md](docs/CODE_STANDARDS.md)** - 코드 작성 표준
- **[.github/docs/FILE_NAMING.md](docs/FILE_NAMING.md)** - 파일 명명 규칙
- **[.github/docs/DEVELOPMENT_WORKFLOW.md](docs/DEVELOPMENT_WORKFLOW.md)** - 개발 워크플로우

### 사용자 문서 (/docs/)
- **[docs/SQLITE_USAGE.md](../docs/SQLITE_USAGE.md)** - SQLite 사용 가이드

### API 문서
- **[upbit_api/README.md](../upbit_api/README.md)** - Upbit API 문서
- **[bithumb_api/README.md](../bithumb_api/README.md)** - Bithumb API 문서

---

**이 지침서는 프로젝트의 일관성과 품질을 유지하기 위한 핵심 규칙을 담고 있습니다.**
