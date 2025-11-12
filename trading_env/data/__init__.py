"""
Data Management Module

데이터 수집, 저장, 처리를 담당하는 모듈입니다.

모듈 구성:
- storage.py: SQLite 기반 데이터 저장/로드
- collection.py: 통합 데이터 수집기
- market_data.py: 시장 데이터 수집 및 정규화
"""

from .storage import MarketDataStorage, collect_and_store_data
from .collection import DataCollector
from .market_data import UpbitDataCollector, DataNormalizer

__all__ = [
    'MarketDataStorage',
    'collect_and_store_data',
    'DataCollector',
    'UpbitDataCollector',
    'DataNormalizer',
]
