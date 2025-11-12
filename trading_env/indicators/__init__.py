"""
Technical Indicators Module

기술적 지표 계산 및 특성 추출을 담당하는 모듈입니다.

모듈 구성:
- basic.py: 기본 기술적 지표 (SMA, EMA, RSI, MACD, etc.)
- custom.py: 커스텀 지표 (눌림목 지수, 변동성 지표 등)
- ssl.py: Self-Supervised Learning 기반 특성 추출 및 미래 예측
"""

from .basic import FeatureExtractor
from .custom import CustomIndicators
from .ssl import SSLFeatureExtractor, SSLConfig

__all__ = [
    'FeatureExtractor',
    'CustomIndicators',
    'SSLFeatureExtractor',
    'SSLConfig',
]
