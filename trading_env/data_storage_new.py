"""
SQLite 기반 시장 데이터 저장소 모듈

순수 SQLite 연동 기능만 제공합니다:
- 데이터베이스 연결 및 테이블 생성
- 데이터 조회 (읽기)
- 데이터 저장 (삽입/업데이트)
- 데이터 존재 여부 확인

Author: Trading System
Date: 2025-10-12
"""

import sqlite3
import pandas as pd
import numpy as np
import logging
from typing import Optional, List, Dict, Tuple, Any
from datetime import datetime, timedelta
from pathlib import Path


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


class MarketDataStorage:
    """
    SQLite 기반 시장 데이터 저장소 클래스
    
    순수 SQLite 연동 기능만 제공:
    - 데이터 조회
    - 데이터 저장
    - 데이터 존재 확인
    - 테이블 관리
    """
    
    def __init__(self, db_path: str = "data/market_data.db", log_level: int = LOG_LEVEL):
        """
        MarketDataStorage 초기화
        
        Args:
            db_path: SQLite DB 파일 경로
            log_level: 로깅 레벨
        """
        self.db_path = db_path
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.logger.setLevel(log_level)
        
        # DB 디렉토리 생성
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        
        # DB 연결 및 테이블 생성
        self._init_database()
        
        self.logger.info(f"MarketDataStorage initialized: {db_path}")
    
    def _init_database(self):
        """데이터베이스 테이블 초기화"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # market_data 테이블 생성
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS market_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    market TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume REAL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(market, timeframe, timestamp)
                )
            """)
            
            # 인덱스 생성
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_market_timeframe_timestamp
                ON market_data(market, timeframe, timestamp)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp
                ON market_data(timestamp)
            """)
            
            conn.commit()
            self.logger.debug("Database tables initialized")
    
    # =============================================================
    # 데이터 조회 (읽기)
    # =============================================================
    
    def load_data(
        self,
        market: str,
        timeframe: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """
        시장 데이터 조회
        
        Args:
            market: 마켓 코드 (예: KRW-BTC)
            timeframe: 타임프레임 (1m, 5m, 1h, 1d 등)
            start_time: 시작 시간 (포함)
            end_time: 종료 시간 (포함)
            limit: 최대 반환 개수
            
        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        query = """
            SELECT timestamp, open, high, low, close, volume
            FROM market_data
            WHERE market = ? AND timeframe = ?
        """
        params = [market, timeframe]
        
        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time.isoformat())
        
        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time.isoformat())
        
        query += " ORDER BY timestamp ASC"
        
        if limit:
            query += f" LIMIT {limit}"
        
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(query, conn, params=params)
        
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')
        
        self.logger.debug(f"Loaded {len(df)} rows for {market} {timeframe}")
        return df
    
    def get_data_range(
        self,
        market: str,
        timeframe: str
    ) -> Tuple[Optional[datetime], Optional[datetime]]:
        """
        저장된 데이터의 시간 범위 조회
        
        Args:
            market: 마켓 코드
            timeframe: 타임프레임
            
        Returns:
            (시작 시간, 종료 시간) 또는 (None, None) if no data
        """
        query = """
            SELECT MIN(timestamp), MAX(timestamp)
            FROM market_data
            WHERE market = ? AND timeframe = ?
        """
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(query, (market, timeframe))
            result = cursor.fetchone()
        
        if result and result[0] and result[1]:
            start = datetime.fromisoformat(result[0])
            end = datetime.fromisoformat(result[1])
            self.logger.debug(f"Data range for {market} {timeframe}: {start} ~ {end}")
            return start, end
        
        self.logger.debug(f"No data found for {market} {timeframe}")
        return None, None
    
    def get_data_count(
        self,
        market: str,
        timeframe: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> int:
        """
        저장된 데이터 개수 조회
        
        Args:
            market: 마켓 코드
            timeframe: 타임프레임
            start_time: 시작 시간 (포함)
            end_time: 종료 시간 (포함)
            
        Returns:
            데이터 개수
        """
        query = """
            SELECT COUNT(*)
            FROM market_data
            WHERE market = ? AND timeframe = ?
        """
        params = [market, timeframe]
        
        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time.isoformat())
        
        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time.isoformat())
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            count = cursor.fetchone()[0]
        
        self.logger.debug(f"Data count for {market} {timeframe}: {count}")
        return count
    
    def has_data(
        self,
        market: str,
        timeframe: str,
        start_time: datetime,
        end_time: datetime
    ) -> bool:
        """
        특정 시간 범위의 데이터 존재 여부 확인
        
        Args:
            market: 마켓 코드
            timeframe: 타임프레임
            start_time: 시작 시간
            end_time: 종료 시간
            
        Returns:
            True if data exists, False otherwise
        """
        count = self.get_data_count(market, timeframe, start_time, end_time)
        return count > 0
    
    # =============================================================
    # 데이터 저장 (삽입/업데이트)
    # =============================================================
    
    def save_data(
        self,
        market: str,
        timeframe: str,
        df: pd.DataFrame,
        replace: bool = True
    ) -> int:
        """
        시장 데이터 저장
        
        Args:
            market: 마켓 코드
            timeframe: 타임프레임
            df: DataFrame with columns: timestamp(index), open, high, low, close, volume
            replace: True면 중복 시 교체, False면 무시
            
        Returns:
            저장된 행 개수
        """
        if df.empty:
            self.logger.warning("Empty DataFrame provided, nothing to save")
            return 0
        
        # DataFrame 준비
        df_to_save = df.copy()
        
        # 인덱스가 timestamp인 경우 컬럼으로 변환
        if df_to_save.index.name == 'timestamp' or isinstance(df_to_save.index, pd.DatetimeIndex):
            df_to_save = df_to_save.reset_index()
        
        # timestamp 컬럼 확인
        if 'timestamp' not in df_to_save.columns:
            raise ValueError("DataFrame must have 'timestamp' column or DatetimeIndex")
        
        # timestamp를 ISO 형식 문자열로 변환
        df_to_save['timestamp'] = pd.to_datetime(df_to_save['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
        
        # 필요한 컬럼만 선택
        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        df_to_save = df_to_save[required_columns]
        
        # market, timeframe 컬럼 추가
        df_to_save['market'] = market
        df_to_save['timeframe'] = timeframe
        
        # DB에 저장
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            saved_count = 0

            if replace:
                # INSERT OR REPLACE: 중복 시 교체
                for _, row in df_to_save.iterrows():
                    cursor.execute("""
                        INSERT OR REPLACE INTO market_data
                        (market, timeframe, timestamp, open, high, low, close, volume)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        row['market'], row['timeframe'], row['timestamp'],
                        row['open'], row['high'], row['low'], row['close'], row['volume']
                    ))
                    if cursor.rowcount > 0:
                        saved_count += 1
            else:
                # INSERT OR IGNORE: 중복 시 무시
                for _, row in df_to_save.iterrows():
                    try:
                        cursor.execute("""
                            INSERT OR IGNORE INTO market_data
                            (market, timeframe, timestamp, open, high, low, close, volume)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            row['market'], row['timeframe'], row['timestamp'],
                            row['open'], row['high'], row['low'], row['close'], row['volume']
                        ))
                        if cursor.rowcount > 0:
                            saved_count += 1
                    except sqlite3.IntegrityError:
                        pass

            conn.commit()
        
        self.logger.info(f"Saved {saved_count} rows for {market} {timeframe}")
        return saved_count
    
    def update_data(
        self,
        market: str,
        timeframe: str,
        timestamp: datetime,
        **kwargs
    ) -> bool:
        """
        특정 타임스탬프의 데이터 업데이트
        
        Args:
            market: 마켓 코드
            timeframe: 타임프레임
            timestamp: 타임스탬프
            **kwargs: 업데이트할 필드 (open, high, low, close, volume)
            
        Returns:
            True if updated, False otherwise
        """
        if not kwargs:
            self.logger.warning("No fields to update")
            return False
        
        # UPDATE 쿼리 생성
        set_clause = ", ".join([f"{k} = ?" for k in kwargs.keys()])
        query = f"""
            UPDATE market_data
            SET {set_clause}
            WHERE market = ? AND timeframe = ? AND timestamp = ?
        """
        
        params = list(kwargs.values()) + [market, timeframe, timestamp.isoformat()]
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            updated = cursor.rowcount > 0
            conn.commit()
        
        if updated:
            self.logger.debug(f"Updated data for {market} {timeframe} at {timestamp}")
        else:
            self.logger.debug(f"No data found to update for {market} {timeframe} at {timestamp}")
        
        return updated
    
    # =============================================================
    # 유틸리티
    # =============================================================
    
    def delete_data(
        self,
        market: str,
        timeframe: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> int:
        """
        데이터 삭제
        
        Args:
            market: 마켓 코드
            timeframe: 타임프레임
            start_time: 시작 시간 (포함)
            end_time: 종료 시간 (포함)
            
        Returns:
            삭제된 행 개수
        """
        query = """
            DELETE FROM market_data
            WHERE market = ? AND timeframe = ?
        """
        params = [market, timeframe]
        
        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time.isoformat())
        
        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time.isoformat())
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            deleted_count = cursor.rowcount
            conn.commit()
        
        self.logger.info(f"Deleted {deleted_count} rows for {market} {timeframe}")
        return deleted_count
    
    def get_available_markets(self) -> List[str]:
        """
        DB에 저장된 모든 마켓 목록 조회
        
        Returns:
            마켓 코드 리스트
        """
        query = "SELECT DISTINCT market FROM market_data ORDER BY market"
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(query)
            markets = [row[0] for row in cursor.fetchall()]
        
        return markets
    
    def get_available_timeframes(self, market: str) -> List[str]:
        """
        특정 마켓의 사용 가능한 타임프레임 목록 조회
        
        Args:
            market: 마켓 코드
            
        Returns:
            타임프레임 리스트
        """
        query = """
            SELECT DISTINCT timeframe
            FROM market_data
            WHERE market = ?
            ORDER BY timeframe
        """
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(query, (market,))
            timeframes = [row[0] for row in cursor.fetchall()]
        
        return timeframes
    
    def get_database_stats(self) -> Dict[str, Any]:
        """
        데이터베이스 통계 정보 조회
        
        Returns:
            통계 정보 딕셔너리
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # 전체 행 개수
            cursor.execute("SELECT COUNT(*) FROM market_data")
            total_rows = cursor.fetchone()[0]
            
            # 마켓별 통계
            cursor.execute("""
                SELECT market, timeframe, COUNT(*), MIN(timestamp), MAX(timestamp)
                FROM market_data
                GROUP BY market, timeframe
                ORDER BY market, timeframe
            """)
            market_stats = []
            for row in cursor.fetchall():
                market_stats.append({
                    'market': row[0],
                    'timeframe': row[1],
                    'count': row[2],
                    'start': row[3],
                    'end': row[4]
                })
        
        return {
            'total_rows': total_rows,
            'market_stats': market_stats
        }


# 편의 함수들
def align_timestamp(dt, timeframe: str) -> datetime:
    """
    타임프레임에 맞게 타임스탬프 정규화 (초/밀리초 제거)

    Args:
        dt: datetime 객체, pandas Timestamp, 또는 Unix timestamp (int/float)
        timeframe: '1m', '1h', '1d'

    Returns:
        정규화된 datetime

    Examples:
        - 1m: 2025-10-09 14:23:45.123 → 2025-10-09 14:23:00
        - 1h: 2025-10-09 14:23:45.123 → 2025-10-09 14:00:00
        - 1d: 2025-10-09 14:23:45.123 → 2025-10-09 00:00:00
    """
    # pandas Timestamp를 datetime으로 변환
    if hasattr(dt, 'to_pydatetime'):
        dt = dt.to_pydatetime()
    # Unix timestamp (int 또는 float)를 datetime으로 변환
    elif isinstance(dt, (int, float)):
        dt = datetime.fromtimestamp(dt)

    if timeframe == '1m':
        return dt.replace(second=0, microsecond=0)
    elif timeframe == '1h':
        return dt.replace(minute=0, second=0, microsecond=0)
    elif timeframe == '1d':
        return dt.replace(hour=0, minute=0, second=0, microsecond=0)
    else:
        raise ValueError(f"지원하지 않는 타임프레임: {timeframe}")


# 모듈 로드 시 로깅 설정
setup_module_logging()
