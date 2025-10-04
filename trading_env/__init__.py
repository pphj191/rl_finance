"""
Trading Environment Package

강화학습 기반 암호화폐 트레이딩 환경을 제공하는 패키지입니다.

모듈 구성:
- base_env.py: 기본 설정 및 공통 클래스
- rl_env.py: 강화학습 환경 (TradingEnvironment)
- market_data.py: 시장 데이터 수집 및 처리
- indicators.py: 기술적 지표 및 특성 추출
"""

from .base_env import TradingConfig, ActionSpace
from .rl_env import TradingEnvironment
from .market_data import UpbitDataCollector, DataNormalizer
from .indicators import FeatureExtractor

__all__ = [
    'TradingConfig',
    'ActionSpace', 
    'TradingEnvironment',
    'UpbitDataCollector',
    'DataNormalizer',
    'FeatureExtractor'
]

__version__ = "1.0.0"
