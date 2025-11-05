"""
새로운 데이터 수집 시스템 테스트

data_storage_new.py, data_collection.py의 기능을 테스트합니다.
"""

import logging
from datetime import datetime, timedelta
import sys
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from trading_env.data_storage import MarketDataStorage
from trading_env.data_collection import DataCollector

# 로깅 설정 - 루트 로거에만 핸들러 추가 (중복 방지)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    force=True  # 기존 핸들러 제거 후 재설정
)

logger = logging.getLogger(__name__)

# 각 모듈의 로거가 루트 로거로 전파되지 않도록 설정
logging.getLogger('trading_env.data_storage_new').propagate = False
logging.getLogger('trading_env.data_collection').propagate = False


def test_storage():
    """data_storage_new.py 테스트"""
    logger.info("")
    logger.info("=" * 60)
    logger.info("data_storage_new.py 테스트 시작")
    logger.info("=" * 60)
    
    # Storage 생성
    storage = MarketDataStorage(db_path="data/market_data_test.db")
    
    # DB 통계 조회
    stats = storage.get_database_stats()
    logger.info(f"DB 통계: 총 {stats['total_rows']} rows")
    
    # 사용 가능한 마켓 조회
    markets = storage.get_available_markets()
    logger.info(f"사용 가능한 마켓: {markets}")
    
    if markets:
        market = markets[0]
        timeframes = storage.get_available_timeframes(market)
        logger.info(f"{market}의 타임프레임: {timeframes}")
        
        if timeframes:
            tf = timeframes[0]
            start, end = storage.get_data_range(market, tf)
            count = storage.get_data_count(market, tf)
            logger.info(f"{market} {tf} 데이터 범위: {start} ~ {end} ({count} rows)")
    
    logger.info("data_storage_new.py 테스트 완료")
    logger.info("")


def test_collector_basic():
    """data_collection.py 기본 테스트"""
    logger.info("")
    logger.info("=" * 60)
    logger.info("data_collection.py 기본 테스트 시작")
    logger.info("=" * 60)
    
    # Collector 생성
    collector = DataCollector(db_path="data/market_data_test.db")
    
    # 테스트 1: 끝시간과 개수로 데이터 수집 (최근 100개 1분봉)
    logger.info("")
    logger.info("[테스트 1] 끝시간과 개수로 데이터 수집")
    df = collector.get_candles_by_count(
        market="KRW-BTC",
        timeframe="1m",
        count=100,
        end_time=None  # 현재 시간
    )
    logger.info(f"수집된 데이터: {len(df)} rows")
    if not df.empty:
        logger.info(f"데이터 범위: {df.index[0]} ~ {df.index[-1]}")
        logger.info(f"샘플 데이터:")
        logger.info(f"{df.head(3)}")
    
    logger.info("data_collection.py 기본 테스트 완료")
    logger.info("")


def test_collector_range():
    """data_collection.py 시간 범위 테스트"""
    logger.info("")
    logger.info("=" * 60)
    logger.info("data_collection.py 시간 범위 테스트 시작")
    logger.info("=" * 60)
    
    collector = DataCollector(db_path="data/market_data_test.db")
    
    # 테스트 2: 시간 범위로 데이터 수집 (최근 24시간 1시간봉)
    logger.info("")
    logger.info("[테스트 2] 시간 범위로 데이터 수집")
    end_time = datetime.now()
    start_time = end_time - timedelta(hours=24)
    
    df = collector.get_candles_by_range(
        market="KRW-BTC",
        timeframe="1h",
        start_time=start_time,
        end_time=end_time
    )
    logger.info(f"수집된 데이터: {len(df)} rows")
    if not df.empty:
        logger.info(f"데이터 범위: {df.index[0]},{df.index[1]} ~ {df.index[-1]}")
        logger.info(f"샘플 데이터:")
        logger.info(f"\n{df.head(3)}")
    
    logger.info("data_collection.py 시간 범위 테스트 완료")
    logger.info("")


def test_collector_multi_timeframe():
    """data_collection.py Multi-timeframe 테스트"""
    logger.info("")
    logger.info("=" * 60)
    logger.info("data_collection.py Multi-timeframe 테스트 시작")
    logger.info("=" * 60)
    
    collector = DataCollector(db_path="data/market_data_test.db")
    
    # 테스트 3: Multi-timeframe 데이터 수집
    logger.info("")
    logger.info("[테스트 3] Multi-timeframe 데이터 수집")
    data_dict = collector.get_multi_timeframe_data(
        market="KRW-BTC",
        timeframes=['1m', '1h', '1d'],
        count_per_timeframe={'1m': 60, '1h': 24, '1d': 7},
        end_time=None
    )
    
    for tf, df in data_dict.items():
        logger.info(f"{tf}: {len(df)} rows")
        if not df.empty:
            logger.info(f"  범위: {df.index[0]} ~ {df.index[-1]}")
    
    logger.info("data_collection.py Multi-timeframe 테스트 완료")
    logger.info("")


def main():
    """메인 테스트 함수"""
    try:
        # 1. Storage 테스트
        test_storage()
        
        # 2. Collector 기본 테스트
        test_collector_basic()
        
        # 3. Collector 시간 범위 테스트
        test_collector_range()
        
        # 4. Collector Multi-timeframe 테스트
        test_collector_multi_timeframe()
        
        logger.info("")
        logger.info("=" * 60)
        logger.info("모든 테스트 완료!")
        logger.info("=" * 60)
        logger.info("")
        
    except Exception as e:
        logger.error(f"테스트 중 오류 발생: {e}", exc_info=True)
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
