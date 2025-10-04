# 강화학습 암호화폐 트레이딩 시스템 개발 지침서

> **최종 업데이트**: 2025년 10월 04일 22:27

이 문서는 프로젝트의 개발, 유지보수, 확장을 위한 핵심 지침을 제공합니다.

## 📚 문서 구조

상세한 개발 가이드는 아래 문서들을 참조하세요:

- **[TODO.md](./docs/TODO.md)** - 작업 목록 및 진행 상황
- **[PROJECT_STRUCTURE.md](./docs/PROJECT_STRUCTURE.md)** - 프로젝트 구조 상세 분석
- **[FILE_NAMING.md](./docs/FILE_NAMING.md)** - 파일 명명 규칙
- **[CODE_STANDARDS.md](./docs/CODE_STANDARDS.md)** - 코드 작성 표준
- **[DEVELOPMENT_WORKFLOW.md](./docs/DEVELOPMENT_WORKFLOW.md)** - 개발 워크플로우

## 🎯 핵심 원칙

### 1. 파일 크기 관리
- **최대 500라인** 권장
- 800라인 초과 시 분리 검토
- 1000라인 초과 시 필수 분리

### 2. 명명 규칙
- 실행 파일: `run_*.py`
- 테스트 파일: `test_*.py`
- 예제 파일: `example_*.py`

### 3. 코드 품질
- 단일 책임 원칙 준수
- Type hints 완전 적용
- Docstring 작성 필수
- 함수는 20라인 이하 권장

### 4. 문서화
모든 `.md` 파일은 상단에 업데이트 날짜 및 시간 명시:
```markdown
> **최종 업데이트**: YYYY년 MM월 DD일 HH:MM
```

## 🚀 빠른 시작

### 개발 환경 설정
```bash
# 가상환경 생성 및 활성화
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 패키지 설치
pip install uv
uv add torch gymnasium scikit-learn matplotlib pandas requests python-dotenv

# 환경 검증
python setup_check.py
```

### 새 기능 개발
```bash
# 브랜치 생성
git checkout -b feature/new-feature

# 개발 후 테스트
python -m pytest tests/

# 코드 품질 검사
python -m black .
python -m mypy .
```

## 📋 현재 작업 상황

### 완료된 작업 ✅
- ✅ **파일 구조 정리** (trading_env 패키지 분리)
- ✅ **디렉토리 구조 생성** (docs, tests, examples 폴더)
- ✅ **파일 명명 규칙 통일** (run_ 접두사 적용)
- ✅ **문서 체계화** (상세 문서 분리)
- ✅ **models.py 패키지 분리** (models/ 패키지 완성)

### 진행중인 작업 🔄
- 🔄 **데이터 수집 시스템 구축** (SQLite 기반 orderbook 저장)
- 🔄 **models.py 대체 작업** (새 패키지로 교체)
- 🔄 **import 경로 정리** (프로젝트 전체 수정)

상세한 TODO는 [docs/TODO.md](./docs/TODO.md)를 참조하세요.

## ⚠️ 주의사항

### 개발 시 주의할 점

1. **파일 크기 관리**
   - 500라인 초과 시 분리 검토
   - 기능별로 명확하게 분리

2. **의존성 관리**
   - 순환 의존성 방지
   - 최소한의 의존성 유지

3. **테스트 커버리지**
   - 새 기능은 반드시 테스트 작성
   - 기존 테스트 깨뜨리지 않기

4. **API 키 보안**
   - 코드에 하드코딩 금지
   - .env 파일 Git 커밋 금지

5. **성능 고려**
   - 대용량 데이터 처리 최적화
   - 메모리 누수 방지

## 🔧 유틸리티 명령어

### 파일 크기 확인
```bash
# 모든 Python 파일 크기 확인
wc -l *.py | sort -nr

# 500줄 초과 파일 찾기
find . -name "*.py" -exec wc -l {} + | awk '$1 > 500 {print $0}'
```

### 코드 품질 검사
```bash
# 포맷팅
python -m black .
python -m isort .

# 타입 검사
python -m mypy .

# 복잡도 측정
pip install radon
radon cc --show-complexity .
```

### 테스트 실행
```bash
# 모든 테스트 실행
python -m pytest tests/

# 커버리지 확인
pip install coverage
coverage run -m pytest
coverage report
```

## 📌 빠른 참조

### 현재 우선순위
1. 🔴 **데이터 수집 시스템 구축** (SQLite 기반 orderbook 저장)
2. 🔴 **models.py 대체** (새 패키지로 교체)
3. 🔴 **run_backtesting.py (509라인)** 분리

### 최근 완료 작업
- ✅ **문서 체계화** (상세 문서를 docs 폴더로 분리)
- ✅ **TODO.md 분리** (INSTRUCTIONS.md에서 독립)
- ✅ **models.py → models/ 패키지 분리 완료** (504라인 → 5개 모듈)

## 📚 참고 문서

### 내부 문서
- [README.md](./README.md) - 프로젝트 개요 및 사용법
- [docs/TODO.md](./docs/TODO.md) - 작업 목록
- [docs/PROJECT_STRUCTURE.md](./docs/PROJECT_STRUCTURE.md) - 프로젝트 구조
- [docs/FILE_NAMING.md](./docs/FILE_NAMING.md) - 명명 규칙
- [docs/CODE_STANDARDS.md](./docs/CODE_STANDARDS.md) - 코드 표준
- [docs/DEVELOPMENT_WORKFLOW.md](./docs/DEVELOPMENT_WORKFLOW.md) - 개발 워크플로우

### API 문서
- [upbit_api/README.md](./upbit_api/README.md) - Upbit API 문서
- [bithumb_api/README.md](./bithumb_api/README.md) - Bithumb API 문서

### 외부 참고 자료
- [Python 스타일 가이드 (PEP 8)](https://pep8.org/)
- [Type Hints (PEP 484)](https://www.python.org/dev/peps/pep-0484/)
- [Docstring 규약 (PEP 257)](https://www.python.org/dev/peps/pep-0257/)

---

**이 문서는 프로젝트와 함께 지속적으로 업데이트됩니다.**
