# 개발 워크플로우

> **최종 업데이트**: 2025년 10월 04일 22:27

## 새 기능 개발 절차

### 1. 브랜치 생성
```bash
git checkout -b feature/new-feature-name
```

### 2. 개발 환경 설정
```bash
# 가상환경 활성화
source .venv/bin/activate

# 의존성 설치
uv add new-package

# 설정 확인
python setup_check.py
```

### 3. 개발 환경

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

### 4. 테스트 작성
새 기능 추가 시 반드시 테스트 작성:

```python
# tests/test_new_feature.py
def test_new_feature():
    """새 기능 테스트"""
    pass

def test_edge_cases():
    """엣지 케이스 테스트"""
    pass
```

### 5. 커밋 및 푸시
```bash
# 변경사항 확인
git status
git diff

# 스테이징
git add .

# 커밋
git commit -m "feat: 새 기능 추가"

# 푸시
git push origin feature/new-feature-name
```

## 문서화 규칙

### Markdown 파일 작성 시 필수 사항
모든 `.md` 파일은 상단에 **업데이트 날짜 및 시간**을 반드시 명시해야 합니다.

**형식**:
```markdown
> **최종 업데이트**: YYYY년 MM월 DD일 HH:MM
```

**예시**:
```markdown
# 문서 제목

> **최종 업데이트**: 2025년 10월 04일 22:27

## 내용...
```

### 적용 대상
- 모든 README.md 파일
- 프로젝트 문서 (docs/ 폴더)
- 가이드 문서 (INSTRUCTIONS.md, CONTRIBUTING.md 등)
- 리포트 파일 (reports/ 폴더)

## 개발 도구

### 코드 품질 검사
```bash
# 개발 의존성 설치
uv add --dev pytest black isort mypy

# 코드 포맷팅
python -m black .
python -m isort .

# 타입 검사
python -m mypy .
```

### 파일 크기 모니터링
```bash
# 파일 크기 확인
wc -l *.py | sort -nr

# 500줄 초과 파일 찾기
find . -name "*.py" -exec wc -l {} + | awk '$1 > 500 {print $0}'
```

## 참고 자료

### 🔗 내부 문서
- [README.md](../README.md) - 사용자 가이드
- [TODO.md](./TODO.md) - 작업 목록
- [upbit_api/README.md](../upbit_api/README.md) - Upbit API 문서
- [bithumb_api/README.md](../bithumb_api/README.md) - Bithumb API 문서

### 📖 외부 문서
- [Python 스타일 가이드 (PEP 8)](https://pep8.org/)
- [Type Hints (PEP 484)](https://www.python.org/dev/peps/pep-0484/)
- [Docstring 규약 (PEP 257)](https://www.python.org/dev/peps/pep-0257/)

### 🏗️ 아키텍처 패턴
- **단일 책임 원칙** (Single Responsibility Principle)
- **의존성 주입** (Dependency Injection)
- **팩토리 패턴** (Factory Pattern)
- **전략 패턴** (Strategy Pattern)
