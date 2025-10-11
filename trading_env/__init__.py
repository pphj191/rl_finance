"""
Trading Environment Package

강화학습 기반 암호화폐 트레이딩 환경을 제공하는 패키지입니다.

모듈 구성:
- base_env.py: 기본 설정 및 공통 클래스
- rl_env.py: 강화학습 환경 (TradingEnvironment)
- market_data.py: 시장 데이터 수집 및 처리
- indicators_basic.py: 기본 기술적 지표 및 특성 추출
- indicators_custom.py: 커스텀 지표 (눌림목 지수 등)
- indicators_ssl.py: Self-Supervised Learning 기반 특성 추출 및 미래 예측
- data_storage.py: SQLite 기반 데이터 저장/로드
- data_pipeline.py: 통합 데이터 파이프라인
"""

from .base_env import TradingConfig, ActionSpace
from .rl_env import TradingEnvironment
from .market_data import UpbitDataCollector, DataNormalizer
from .indicators_basic import FeatureExtractor
from .data_storage import MarketDataStorage, collect_and_store_data
from .data_pipeline import DataPipeline, prepare_offline_data
from .indicators_ssl import SSLFeatureExtractor, SSLConfig

__all__ = [
    'TradingConfig',
    'ActionSpace',
    'TradingEnvironment',
    'UpbitDataCollector',
    'DataNormalizer',
    'FeatureExtractor',
    'MarketDataStorage',
    'collect_and_store_data',
    'DataPipeline',
    'prepare_offline_data',
    'SSLFeatureExtractor',
    'SSLConfig'
]

__version__ = "1.0.0"
