"""
Trading Environment Package

강화학습 기반 암호화폐 트레이딩 환경을 제공하는 패키지입니다.

패키지 구조:
- base_env.py: 기본 설정 및 공통 클래스
- rl_env.py: 강화학습 환경 (TradingEnvironment)
- env_pipeline.py: 통합 데이터 파이프라인
- data/: 데이터 수집 및 저장
- indicators/: 기술적 지표 및 특성 추출
- docs/: 상세 문서
"""

# Core environment
from .base_env import TradingConfig, ActionSpace
from .rl_env import TradingEnvironment
from .env_pipeline import DataPipeline, prepare_offline_data

# Data management
from .data import (
    MarketDataStorage,
    collect_and_store_data,
    DataCollector,
    UpbitDataCollector,
    DataNormalizer,
)

# Technical indicators
from .indicators import (
    FeatureExtractor,
    CustomIndicators,
    SSLFeatureExtractor,
    SSLConfig,
)

__all__ = [
    # Core
    'TradingConfig',
    'ActionSpace',
    'TradingEnvironment',
    'DataPipeline',
    'prepare_offline_data',
    # Data
    'MarketDataStorage',
    'collect_and_store_data',
    'DataCollector',
    'UpbitDataCollector',
    'DataNormalizer',
    # Indicators
    'FeatureExtractor',
    'CustomIndicators',
    'SSLFeatureExtractor',
    'SSLConfig',
]

__version__ = "1.0.0"
