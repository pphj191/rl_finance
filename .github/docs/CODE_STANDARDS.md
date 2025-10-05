# 코드 작성 표준

> **최종 업데이트**: 2025년 10월 04일 22:27

## 코드 작성 가이드라인

### 파일 크기 제한
- **최대 500라인** 권장
- 800라인 초과 시 분리 검토
- 1000라인 초과 시 필수 분리

### 클래스 설계 원칙

```python
# 단일 책임 원칙
class TradingEnvironment:
    """강화학습 환경만 담당"""
    pass

class FeatureExtractor:
    """특성 추출만 담당"""
    pass
```

### 함수 설계 원칙

```python
# 함수는 20라인 이하 권장
def get_market_data(self, market: str) -> Dict:
    """한 가지 일만 하는 작은 함수"""
    pass

# 복잡한 로직은 여러 함수로 분리
def process_market_data(self, data: Dict) -> Dict:
    cleaned_data = self._clean_data(data)
    normalized_data = self._normalize_data(cleaned_data)
    features = self._extract_features(normalized_data)
    return features
```

### 문서화

```python
def new_function(param1: str, param2: int) -> Dict:
    """
    새 함수 설명

    Args:
        param1: 매개변수 1 설명
        param2: 매개변수 2 설명

    Returns:
        반환값 설명

    Example:
        >>> result = new_function("test", 123)
        >>> print(result)
    """
```

## 코드 리뷰 체크리스트

### 📋 기본 검사
- [ ] 파일 크기 500라인 이하
- [ ] 함수 크기 20라인 이하
- [ ] 클래스는 단일 책임 원칙 준수
- [ ] Type hints 완전 적용
- [ ] Docstring 작성 완료

### 🧪 테스트 검사
- [ ] 단위 테스트 작성
- [ ] 통합 테스트 실행
- [ ] `python setup_check.py` 통과
- [ ] 예제 코드 동작 확인

### 📚 문서 검사
- [ ] README.md 업데이트
- [ ] API 문서 업데이트
- [ ] 변경 사항 기록

## 리팩토링 가이드

### 대형 파일 분리 방법

#### 1. 기능별 분리
```python
# 기존: trading_env.py (800라인)
# 분리 후:
- environment.py      # 환경 클래스
- feature_extractor.py # 특성 추출
- data_normalizer.py   # 데이터 정규화
- action_space.py      # 액션 공간
```

#### 2. 계층별 분리
```python
# 기존: trading_system.py (600라인)
# 분리 후:
- core/           # 핵심 로직
- strategies/     # 거래 전략
- utils/         # 유틸리티
- config/        # 설정
```

#### 3. 의존성 관리
```python
# __init__.py에서 통합 export
from .environment import TradingEnvironment
from .feature_extractor import FeatureExtractor

__all__ = ['TradingEnvironment', 'FeatureExtractor']
```

## 코드 품질 지표

### 파일 크기 모니터링
```bash
# 주기적으로 실행
find . -name "*.py" -exec wc -l {} + | sort -nr | head -10
```

### 복잡도 측정
```bash
# 설치 후 사용
pip install radon
radon cc --show-complexity .
```

### 테스트 커버리지
```bash
# 설치 후 사용
pip install coverage
coverage run -m pytest
coverage report
```

## 성능 최적화

### 1. 프로파일링
```python
import cProfile
import pstats

# 성능 측정
profiler = cProfile.Profile()
profiler.enable()
# 코드 실행
profiler.disable()

# 결과 분석
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(10)
```

### 2. 메모리 사용량 모니터링
```python
import tracemalloc

# 메모리 추적 시작
tracemalloc.start()
# 코드 실행
current, peak = tracemalloc.get_traced_memory()
print(f"Current: {current / 1024 / 1024:.1f} MB")
print(f"Peak: {peak / 1024 / 1024:.1f} MB")
```

## 보안 가이드라인

### API 키 관리
```python
# ❌ 하드코딩 금지
API_KEY = "your_api_key_here"

# ✅ 환경변수 사용
import os
API_KEY = os.getenv('UPBIT_ACCESS_KEY')

# ✅ .env 파일 사용
from dotenv import load_dotenv
load_dotenv()
```

### 민감한 데이터 로깅 방지
```python
# ❌ API 키 로깅 금지
logger.info(f"Using API key: {api_key}")

# ✅ 마스킹 처리
logger.info(f"Using API key: {api_key[:8]}...")
```
