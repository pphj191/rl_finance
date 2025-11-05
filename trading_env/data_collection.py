"""
데이터 수집 통합 모듈

역할:
- 사용자 요청에 따라 DB에 없는 데이터를 API로 수집하여 제공
- Upbit, Bithumb 등 거래소 API 연동 (현재는 Upbit만 지원)
- Multi-timeframe 데이터 수집
- 개별 timeframe 데이터 수집
- 시간 범위 기반 데이터 수집

Author: Trading System
Date: 2025-10-12
"""

import logging
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Tuple, Any
import pandas as pd
import sys
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from trading_env.data_storage import MarketDataStorage, align_timestamp
from upbit_api.upbit_api import UpbitAPI


# 모듈 레벨 로깅 설정
LOG_LEVEL = logging.INFO

def setup_module_logging(level: int = LOG_LEVEL):
    """모듈 레벨 로깅 설정"""
    logger = logging.getLogger(__name__)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s.%(msecs)03d - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


class DataCollector:
    """
    통합 데이터 수집 클래스
    
    기능:
    1. Multi-timeframe 데이터 수집
    2. 개별 timeframe 데이터 수집
    3. 시간 범위 기반 데이터 수집
    4. DB 연동 (없는 데이터만 API 요청)
    """
    
    # 타임프레임별 분 단위 매핑
    TIMEFRAME_MINUTES = {
        '1m': 1,
        '3m': 3,
        '5m': 5,
        '10m': 10,
        '15m': 15,
        '30m': 30,
        '60m': 60,
        '240m': 240,
        '1h': 60,
        '4h': 240,
        '1d': 1440,
        '1w': 10080,
        '1M': 43200  # 대략 30일
    }
    
    def __init__(
        self,
        db_path: str = "data/market_data.db",
        api_access_key: Optional[str] = None,
        api_secret_key: Optional[str] = None,
        log_level: int = LOG_LEVEL
    ):
        """
        DataCollector 초기화
        
        Args:
            db_path: SQLite DB 파일 경로
            api_access_key: Upbit API Access Key
            api_secret_key: Upbit API Secret Key
            log_level: 로깅 레벨
        """
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.logger.setLevel(log_level)
        
        # Storage 및 API 초기화
        self.storage = MarketDataStorage(db_path=db_path, log_level=log_level)
        self.upbit_api = UpbitAPI(
            access_key=api_access_key,
            secret_key=api_secret_key,
            log_level=log_level
        )
        
        self.logger.info(f"DataCollector initialized with DB: {db_path}")
    
    # =============================================================
    # 내부 헬퍼 메서드
    # =============================================================
    
    def _calculate_required_count(
        self,
        timeframe: str,
        start_time: datetime,
        end_time: datetime
    ) -> int:
        """
        시간 범위로부터 필요한 데이터 개수 계산
        
        Args:
            timeframe: 타임프레임 (1m, 1h, 1d 등)
            start_time: 시작 시간
            end_time: 종료 시간
            
        Returns:
            필요한 데이터 개수
        """
        if timeframe not in self.TIMEFRAME_MINUTES:
            raise ValueError(f"지원하지 않는 타임프레임: {timeframe}")
        
        minutes_per_candle = self.TIMEFRAME_MINUTES[timeframe]
        total_minutes = int((end_time - start_time).total_seconds() / 60)
        count = total_minutes // minutes_per_candle + 1
        
        self.logger.debug(f"Calculated count for {timeframe}: {count} (from {start_time} to {end_time})")
        return count
    
    def _parse_timeframe_for_api(self, timeframe: str) -> Tuple[str, Optional[int]]:
        """
        타임프레임을 API 파라미터로 변환
        
        Args:
            timeframe: 타임프레임 (1m, 1h, 1d 등)
            
        Returns:
            (candle_type, unit) 튜플
            - candle_type: 'minutes', 'days', 'weeks', 'months'
            - unit: 분 캔들의 경우 단위 (1, 3, 5, 15, 30, 60, 240), 그 외는 None
        """
        if timeframe in ['1m', '3m', '5m', '10m', '15m', '30m', '60m', '240m']:
            unit = int(timeframe.replace('m', ''))
            return 'minutes', unit
        elif timeframe == '1h':
            return 'minutes', 60
        elif timeframe == '4h':
            return 'minutes', 240
        elif timeframe == '1d':
            return 'days', None
        elif timeframe == '1w':
            return 'weeks', None
        elif timeframe == '1M':
            return 'months', None
        else:
            raise ValueError(f"지원하지 않는 타임프레임: {timeframe}")
    
    def _fetch_from_api(
        self,
        market: str,
        timeframe: str,
        count: int,
        to: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        API에서 데이터 수집
        
        Args:
            market: 마켓 코드
            timeframe: 타임프레임
            count: 데이터 개수
            to: 종료 시간
            
        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        candle_type, unit = self._parse_timeframe_for_api(timeframe)
        
        # ISO 형식으로 변환
        to_str = to.strftime('%Y-%m-%d %H:%M:%S') if to else None
        
        self.logger.info(f"Fetching {count} {timeframe} candles for {market} from API (to={to_str})")
        
        # API 호출
        if candle_type == 'minutes':
            data = self.upbit_api.get_candles_minutes_bulk(market, unit, count, to_str)
        elif candle_type == 'days':
            data = self.upbit_api.get_candles_days_bulk(market, count, to_str)
        elif candle_type == 'weeks':
            data = self.upbit_api.get_candles_weeks_bulk(market, count, to_str)
        elif candle_type == 'months':
            data = self.upbit_api.get_candles_months_bulk(market, count, to_str)
        else:
            raise ValueError(f"Unsupported candle type: {candle_type}")
        
        # DataFrame 변환
        if not data:
            self.logger.warning(f"No data received from API for {market} {timeframe}")
            return pd.DataFrame()
        
        df = pd.DataFrame(data)
        
        # 컬럼 이름 매핑 (Upbit API 응답 → 표준 형식)
        # timestamp 컬럼이 이미 있으면 먼저 제거
        if 'timestamp' in df.columns and 'candle_date_time_kst' in df.columns:
            df = df.drop(columns=['timestamp'])
        
        df = df.rename(columns={
            'candle_date_time_kst': 'timestamp',
            'opening_price': 'open',
            'high_price': 'high',
            'low_price': 'low',
            'trade_price': 'close',
            'candle_acc_trade_volume': 'volume'
        })
        
        # 필요한 컬럼만 선택
        required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        df = df[[col for col in required_cols if col in df.columns]]
        
        # timestamp를 datetime으로 변환 및 정규화
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['timestamp'] = df['timestamp'].apply(lambda x: align_timestamp(x, timeframe))
        
        # 시간 순 정렬 (과거 → 현재)
        df = df.sort_values('timestamp')
        
        # 중복 제거
        df = df.drop_duplicates(subset=['timestamp'], keep='last')
        
        self.logger.info(f"Fetched {len(df)} rows from API for {market} {timeframe}")
        return df
    
    # =============================================================
    # 공개 메서드: 개별 타임프레임 데이터 수집
    # =============================================================
    
    def get_candles_by_count(
        self,
        market: str,
        timeframe: str,
        count: int,
        end_time: Optional[datetime] = None,
        force_api: bool = False
    ) -> pd.DataFrame:
        """
        끝시간과 개수로 데이터 수집
        
        Args:
            market: 마켓 코드 (예: KRW-BTC)
            timeframe: 타임프레임 (1m, 1h, 1d 등)
            count: 데이터 개수
            end_time: 종료 시간 (None이면 현재 시간)
            force_api: True면 DB 조회 없이 API에서 직접 수집
            
        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        # end_time 기본값 설정 및 정규화
        if end_time is None:
            end_time = datetime.now()
        end_time = align_timestamp(end_time, timeframe)
        
        self.logger.info(f"get_candles_by_count: {market} {timeframe} count={count} end_time={end_time}")
        
        # force_api=True면 바로 API 호출
        if force_api:
            df = self._fetch_from_api(market, timeframe, count, end_time)
            # DB에 저장
            if not df.empty:
                self.storage.save_data(market, timeframe, df)
            return df
        
        # 시작 시간 계산
        minutes_per_candle = self.TIMEFRAME_MINUTES[timeframe]
        start_time = end_time - timedelta(minutes=minutes_per_candle * (count - 1))
        start_time = align_timestamp(start_time, timeframe)
        
        # DB에서 데이터 조회
        df_db = self.storage.load_data(market, timeframe, start_time, end_time)
        
        # DB에 충분한 데이터가 있는지 확인
        if len(df_db) >= count:
            self.logger.info(f"Sufficient data in DB: {len(df_db)} rows")
            return df_db.tail(count)
        
        # DB에 데이터가 부족하면 API 호출
        self.logger.info(f"Insufficient data in DB ({len(df_db)} rows), fetching from API")
        df_api = self._fetch_from_api(market, timeframe, count, end_time)
        
        # DB에 저장
        if not df_api.empty:
            self.storage.save_data(market, timeframe, df_api)
        
        # DB와 API 데이터 병합
        df_combined = pd.concat([df_db, df_api]).drop_duplicates(subset=['timestamp'], keep='last')
        df_combined = df_combined.sort_values('timestamp')
        
        return df_combined.tail(count)
    
    def get_candles_by_range(
        self,
        market: str,
        timeframe: str,
        start_time: datetime,
        end_time: Optional[datetime] = None,
        force_api: bool = False
    ) -> pd.DataFrame:
        """
        시작시간과 끝시간으로 데이터 수집
        
        Args:
            market: 마켓 코드
            timeframe: 타임프레임
            start_time: 시작 시간
            end_time: 종료 시간 (None이면 현재 시간)
            force_api: True면 DB 조회 없이 API에서 직접 수집
            
        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        # 시간 정규화
        start_time = align_timestamp(start_time, timeframe)
        if end_time is None:
            end_time = datetime.now()
        end_time = align_timestamp(end_time, timeframe)
        
        self.logger.info(f"get_candles_by_range: {market} {timeframe} {start_time} ~ {end_time}")
        
        # 필요한 데이터 개수 계산
        count = self._calculate_required_count(timeframe, start_time, end_time)
        
        # get_candles_by_count 호출
        return self.get_candles_by_count(market, timeframe, count, end_time, force_api)
    
    # =============================================================
    # 공개 메서드: Multi-timeframe 데이터 수집
    # =============================================================
    
    def get_multi_timeframe_data(
        self,
        market: str,
        timeframes: List[str],
        count_per_timeframe: Dict[str, int],
        end_time: Optional[datetime] = None,
        force_api: bool = False
    ) -> Dict[str, pd.DataFrame]:
        """
        여러 타임프레임의 데이터를 동시에 수집
        
        Args:
            market: 마켓 코드
            timeframes: 타임프레임 리스트 (예: ['1m', '1h', '1d'])
            count_per_timeframe: 타임프레임별 데이터 개수 (예: {'1m': 100, '1h': 24, '1d': 30})
            end_time: 종료 시간 (None이면 현재 시간)
            force_api: True면 DB 조회 없이 API에서 직접 수집
            
        Returns:
            {timeframe: DataFrame} 딕셔너리
        """
        if end_time is None:
            end_time = datetime.now()
        
        self.logger.info(f"get_multi_timeframe_data: {market} timeframes={timeframes} end_time={end_time}")
        
        result = {}
        for tf in timeframes:
            count = count_per_timeframe.get(tf, 200)
            self.logger.info(f"Collecting {tf} data: {count} candles")
            
            df = self.get_candles_by_count(market, tf, count, end_time, force_api)
            result[tf] = df
        
        self.logger.info(f"Multi-timeframe collection complete: {len(result)} timeframes")
        return result
    
    def get_multi_timeframe_data_by_range(
        self,
        market: str,
        timeframes: List[str],
        start_time: datetime,
        end_time: Optional[datetime] = None,
        force_api: bool = False
    ) -> Dict[str, pd.DataFrame]:
        """
        시간 범위로 여러 타임프레임의 데이터를 동시에 수집
        
        Args:
            market: 마켓 코드
            timeframes: 타임프레임 리스트
            start_time: 시작 시간
            end_time: 종료 시간 (None이면 현재 시간)
            force_api: True면 DB 조회 없이 API에서 직접 수집
            
        Returns:
            {timeframe: DataFrame} 딕셔너리
        """
        if end_time is None:
            end_time = datetime.now()
        
        self.logger.info(f"get_multi_timeframe_data_by_range: {market} {start_time} ~ {end_time}")
        
        result = {}
        for tf in timeframes:
            self.logger.info(f"Collecting {tf} data")
            df = self.get_candles_by_range(market, tf, start_time, end_time, force_api)
            result[tf] = df
        
        self.logger.info(f"Multi-timeframe collection complete: {len(result)} timeframes")
        return result


# 편의 함수들
def create_data_collector(
    db_path: str = "data/market_data.db",
    log_level: int = LOG_LEVEL
) -> DataCollector:
    """
    DataCollector 인스턴스 생성
    
    Args:
        db_path: DB 파일 경로
        log_level: 로깅 레벨
        
    Returns:
        DataCollector 인스턴스
    """
    return DataCollector(db_path=db_path, log_level=log_level)


# 모듈 로드 시 로깅 설정
setup_module_logging()


# 사용 예제
if __name__ == "__main__":
    # 로깅 레벨 설정
    logging.basicConfig(level=logging.INFO)
    
    # DataCollector 생성
    collector = DataCollector(db_path="data/market_data_test.db")
    
    # 예제 1: 끝시간과 개수로 데이터 수집
    print("")
    print("=== 예제 1: 끝시간과 개수로 데이터 수집 ===")
    df = collector.get_candles_by_count(
        market="KRW-BTC",
        timeframe="1m",
        count=100,
        end_time=datetime(2025, 10, 10, 12, 0, 0)
    )
    print(f"수집된 데이터: {len(df)} rows")
    if not df.empty:
        print(df.head())
    
    # 예제 2: 시간 범위로 데이터 수집
    print("")
    print("=== 예제 2: 시간 범위로 데이터 수집 ===")
    df = collector.get_candles_by_range(
        market="KRW-BTC",
        timeframe="1h",
        start_time=datetime(2025, 10, 9, 0, 0, 0),
        end_time=datetime(2025, 10, 10, 0, 0, 0)
    )
    print(f"수집된 데이터: {len(df)} rows")
    if not df.empty:
        print(df.head())
    
    # 예제 3: Multi-timeframe 데이터 수집
    print("")
    print("=== 예제 3: Multi-timeframe 데이터 수집 ===")
    data_dict = collector.get_multi_timeframe_data(
        market="KRW-BTC",
        timeframes=['1m', '1h', '1d'],
        count_per_timeframe={'1m': 60, '1h': 24, '1d': 7},
        end_time=datetime(2025, 10, 10, 12, 0, 0)
    )
    for tf, df in data_dict.items():
        print(f"{tf}: {len(df)} rows")
