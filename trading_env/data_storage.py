"""
SQLite 기반 시장 데이터 저장 및 로드

트레이딩 데이터를 SQLite 데이터베이스에 저장하고 효율적으로 로드합니다.
"""

import sqlite3
import pandas as pd
import numpy as np
import pickle
import json
import hashlib
from typing import Optional, List, Dict, Tuple, Any
from datetime import datetime, timedelta
import logging
from pathlib import Path


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


def get_missing_ranges(
    storage: 'MarketDataStorage',
    market: str,
    timeframe: str,
    target_start: datetime,
    target_end: datetime
) -> List[Tuple[datetime, datetime]]:
    """
    수집해야 할 누락 데이터 범위 계산

    Args:
        storage: MarketDataStorage 인스턴스
        market: 마켓 코드
        timeframe: '1m', '1h', '1d'
        target_start: 목표 시작 시간
        target_end: 목표 종료 시간

    Returns:
        [(시작1, 종료1), (시작2, 종료2), ...] 누락된 시간 범위 리스트
    """
    logger = logging.getLogger(__name__)

    # 1. 기존 데이터 범위 조회
    existing_start, existing_end = storage.get_data_range_by_timeframe(market, timeframe)

    # 2. 데이터가 없으면 전체 구간 수집
    if existing_start is None:
        logger.info(f"  [{timeframe}] 기존 데이터 없음 → 전체 구간 수집")
        return [(target_start, target_end)]

    logger.info(f"  [{timeframe}] 기존 데이터: {existing_start} ~ {existing_end}")

    # 3. 누락 구간 계산
    missing_ranges = []

    # 앞부분 누락 (target_start < existing_start)
    if target_start < existing_start:
        missing_ranges.append((target_start, existing_start))
        logger.info(f"  [{timeframe}] 앞부분 누락: {target_start} ~ {existing_start}")

    # 뒷부분 누락 (target_end > existing_end)
    if target_end > existing_end:
        missing_ranges.append((existing_end, target_end))
        logger.info(f"  [{timeframe}] 뒷부분 누락: {existing_end} ~ {target_end}")

    # TODO: 중간 구간 누락 감지 (Phase 2 확장)

    if not missing_ranges:
        logger.info(f"  [{timeframe}] 누락 구간 없음")

    return missing_ranges


class MarketDataStorage:
    """시장 데이터 SQLite 저장소"""

    def __init__(self, db_path: str = "data/market_data.db"):
        """
        Args:
            db_path: SQLite 데이터베이스 파일 경로
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)

        # 데이터베이스 초기화
        self._initialize_database()

    def _initialize_database(self):
        """데이터베이스 테이블 초기화 (멀티 타임프레임 지원)"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # 멀티 타임프레임 OHLCV 테이블 생성
            for timeframe in ['1m', '1h', '1d']:
                cursor.execute(f"""
                    CREATE TABLE IF NOT EXISTS ohlcv_{timeframe} (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        market TEXT NOT NULL,
                        timestamp INTEGER NOT NULL,
                        open REAL NOT NULL,
                        high REAL NOT NULL,
                        low REAL NOT NULL,
                        close REAL NOT NULL,
                        volume REAL NOT NULL,
                        value REAL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(market, timestamp)
                    )
                """)

            # 하위 호환성을 위해 기존 ohlcv_data 테이블도 유지 (1분봉으로 간주)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ohlcv_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    market TEXT NOT NULL,
                    timestamp INTEGER NOT NULL,
                    open REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    close REAL NOT NULL,
                    volume REAL NOT NULL,
                    value REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(market, timestamp)
                )
            """)

            # 오더북 데이터 테이블
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS orderbook_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    market TEXT NOT NULL,
                    timestamp INTEGER NOT NULL,
                    asks TEXT NOT NULL,  -- JSON 형식으로 저장
                    bids TEXT NOT NULL,  -- JSON 형식으로 저장
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(market, timestamp)
                )
            """)

            # 처리된 데이터 테이블 (기술적 지표 + 특성)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS processed_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    market TEXT NOT NULL,
                    timestamp INTEGER NOT NULL,
                    -- 기술적 지표 (컬럼으로 저장)
                    sma_5 REAL, sma_20 REAL, sma_60 REAL,
                    ema_12 REAL, ema_26 REAL,
                    rsi_14 REAL,
                    macd REAL, macd_signal REAL, macd_hist REAL,
                    bb_upper REAL, bb_middle REAL, bb_lower REAL,
                    bb_width REAL,
                    atr_14 REAL,
                    obv REAL,
                    stoch_k REAL, stoch_d REAL,
                    volume_sma REAL,
                    price_change_1 REAL, price_change_5 REAL, price_change_20 REAL,
                    -- 추출된 특성 (BLOB)
                    feature_vector BLOB,
                    feature_names TEXT,  -- JSON: 특성 이름 리스트
                    -- 정규화 정보
                    normalization_method TEXT,
                    normalization_params TEXT,  -- JSON: 정규화 파라미터
                    -- 메타데이터
                    config_hash TEXT NOT NULL,  -- 설정 해시 (캐시 무효화)
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(market, timestamp, config_hash)
                )
            """)

            # 인덱스 생성
            for timeframe in ['1m', '1h', '1d']:
                cursor.execute(f"""
                    CREATE INDEX IF NOT EXISTS idx_ohlcv_{timeframe}_market_timestamp
                    ON ohlcv_{timeframe}(market, timestamp)
                """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_ohlcv_market_timestamp
                ON ohlcv_data(market, timestamp)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_orderbook_market_timestamp
                ON orderbook_data(market, timestamp)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_processed_market_timestamp_hash
                ON processed_data(market, timestamp, config_hash)
            """)

            conn.commit()
            self.logger.info(f"데이터베이스 초기화 완료 (멀티 타임프레임 지원): {self.db_path}")

    def save_ohlcv_data(self, market: str, data: pd.DataFrame):
        """
        OHLCV 데이터 저장

        Args:
            market: 마켓 코드 (예: KRW-BTC)
            data: OHLCV 데이터프레임 (컬럼: timestamp, open, high, low, close, volume, value)
        """
        with sqlite3.connect(self.db_path) as conn:
            # 타임스탬프를 정수로 변환
            if 'timestamp' not in data.columns:
                data['timestamp'] = data.index

            data = data.copy()
            if pd.api.types.is_datetime64_any_dtype(data['timestamp']):
                data['timestamp'] = data['timestamp'].astype('int64') // 10**9

            # 데이터 삽입 (중복 무시)
            data['market'] = market
            data[['market', 'timestamp', 'open', 'high', 'low', 'close', 'volume', 'value']].to_sql(
                'ohlcv_data',
                conn,
                if_exists='append',
                index=False,
                method='multi'
            )

            conn.commit()
            self.logger.info(f"{market} OHLCV 데이터 {len(data)}건 저장 완료")

    def load_ohlcv_data(
        self,
        market: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """
        OHLCV 데이터 로드

        Args:
            market: 마켓 코드
            start_time: 시작 시간
            end_time: 종료 시간
            limit: 최대 데이터 개수

        Returns:
            OHLCV 데이터프레임
        """
        query = "SELECT timestamp, open, high, low, close, volume, value FROM ohlcv_data WHERE market = ?"
        params = [market]

        if start_time:
            query += " AND timestamp >= ?"
            params.append(int(start_time.timestamp()))

        if end_time:
            query += " AND timestamp <= ?"
            params.append(int(end_time.timestamp()))

        query += " ORDER BY timestamp ASC"

        if limit:
            query += f" LIMIT {limit}"

        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(query, conn, params=params)

        if len(df) == 0:
            self.logger.warning(f"{market} 데이터 없음")
            return pd.DataFrame()

        # 타임스탬프 변환
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        df = df.set_index('timestamp')

        self.logger.info(f"{market} OHLCV 데이터 {len(df)}건 로드 완료")
        return df

    def get_data_range(self, market: str) -> Tuple[Optional[datetime], Optional[datetime]]:
        """
        저장된 데이터의 시간 범위 조회

        Args:
            market: 마켓 코드

        Returns:
            (최소 시간, 최대 시간)
        """
        query = """
            SELECT MIN(timestamp) as min_time, MAX(timestamp) as max_time
            FROM ohlcv_data WHERE market = ?
        """

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(query, [market])
            result = cursor.fetchone()

        if result and result[0] and result[1]:
            min_time = datetime.fromtimestamp(result[0])
            max_time = datetime.fromtimestamp(result[1])
            return min_time, max_time

        return None, None

    def get_data_count(self, market: str) -> int:
        """
        저장된 데이터 개수 조회

        Args:
            market: 마켓 코드

        Returns:
            데이터 개수
        """
        query = "SELECT COUNT(*) FROM ohlcv_data WHERE market = ?"

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(query, [market])
            result = cursor.fetchone()

        return result[0] if result else 0

    def clear_data(self, market: Optional[str] = None):
        """
        데이터 삭제

        Args:
            market: 마켓 코드 (None이면 모든 데이터 삭제)
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            if market:
                cursor.execute("DELETE FROM ohlcv_data WHERE market = ?", [market])
                cursor.execute("DELETE FROM orderbook_data WHERE market = ?", [market])
                cursor.execute("DELETE FROM processed_data WHERE market = ?", [market])
                self.logger.info(f"{market} 데이터 삭제 완료")
            else:
                cursor.execute("DELETE FROM ohlcv_data")
                cursor.execute("DELETE FROM orderbook_data")
                cursor.execute("DELETE FROM processed_data")
                self.logger.info("모든 데이터 삭제 완료")

            conn.commit()

    def save_processed_data(
        self,
        market: str,
        data: pd.DataFrame,
        feature_vector: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
        normalization_method: str = "robust",
        normalization_params: Optional[Dict] = None,
        config_hash: Optional[str] = None
    ):
        """
        처리된 데이터(기술적 지표 + 특성) 저장

        Args:
            market: 마켓 코드
            data: 기술적 지표가 포함된 DataFrame
            feature_vector: 추출된 특성 벡터 (shape: [n_samples, n_features])
            feature_names: 특성 이름 리스트
            normalization_method: 정규화 방법
            normalization_params: 정규화 파라미터
            config_hash: 설정 해시
        """
        if config_hash is None:
            config_hash = self._generate_config_hash(
                normalization_method, normalization_params
            )

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # 데이터 준비
            data_to_save = data.copy()

            # 타임스탬프 변환
            if 'timestamp' not in data_to_save.columns:
                data_to_save['timestamp'] = data_to_save.index

            if pd.api.types.is_datetime64_any_dtype(data_to_save['timestamp']):
                data_to_save['timestamp'] = (
                    data_to_save['timestamp'].astype('int64') // 10**9
                )

            # 기술적 지표 컬럼 (테이블에 정의된 것)
            indicator_columns = [
                'sma_5', 'sma_20', 'sma_60',
                'ema_12', 'ema_26',
                'rsi_14',
                'macd', 'macd_signal', 'macd_hist',
                'bb_upper', 'bb_middle', 'bb_lower', 'bb_width',
                'atr_14', 'obv',
                'stoch_k', 'stoch_d',
                'volume_sma',
                'price_change_1', 'price_change_5', 'price_change_20'
            ]

            # 데이터 행별로 저장
            for idx, row in data_to_save.iterrows():
                timestamp = int(row['timestamp'])

                # 기술적 지표 값 추출
                indicator_values = {}
                for col in indicator_columns:
                    if col in data_to_save.columns:
                        val = row[col]
                        indicator_values[col] = (
                            float(val) if pd.notna(val) else None
                        )
                    else:
                        indicator_values[col] = None

                # 특성 벡터 직렬화
                feature_blob = None
                if feature_vector is not None and len(feature_vector) > idx:
                    feature_blob = pickle.dumps(feature_vector[idx])

                # 특성 이름 JSON 직렬화
                feature_names_json = (
                    json.dumps(feature_names) if feature_names else None
                )

                # 정규화 파라미터 JSON 직렬화
                norm_params_json = (
                    json.dumps(normalization_params) if normalization_params else None
                )

                # INSERT OR REPLACE
                cursor.execute("""
                    INSERT OR REPLACE INTO processed_data (
                        market, timestamp,
                        sma_5, sma_20, sma_60,
                        ema_12, ema_26,
                        rsi_14,
                        macd, macd_signal, macd_hist,
                        bb_upper, bb_middle, bb_lower, bb_width,
                        atr_14, obv,
                        stoch_k, stoch_d,
                        volume_sma,
                        price_change_1, price_change_5, price_change_20,
                        feature_vector, feature_names,
                        normalization_method, normalization_params,
                        config_hash
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, [
                    market, timestamp,
                    indicator_values['sma_5'], indicator_values['sma_20'], indicator_values['sma_60'],
                    indicator_values['ema_12'], indicator_values['ema_26'],
                    indicator_values['rsi_14'],
                    indicator_values['macd'], indicator_values['macd_signal'], indicator_values['macd_hist'],
                    indicator_values['bb_upper'], indicator_values['bb_middle'], indicator_values['bb_lower'], indicator_values['bb_width'],
                    indicator_values['atr_14'], indicator_values['obv'],
                    indicator_values['stoch_k'], indicator_values['stoch_d'],
                    indicator_values['volume_sma'],
                    indicator_values['price_change_1'], indicator_values['price_change_5'], indicator_values['price_change_20'],
                    feature_blob, feature_names_json,
                    normalization_method, norm_params_json,
                    config_hash
                ])

            conn.commit()
            self.logger.info(f"{market} 처리된 데이터 {len(data_to_save)}건 저장 완료 (hash: {config_hash[:8]})")

    def load_processed_data(
        self,
        market: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        config_hash: Optional[str] = None,
        limit: Optional[int] = None
    ) -> Optional[pd.DataFrame]:
        """
        처리된 데이터 로드

        Args:
            market: 마켓 코드
            start_time: 시작 시간
            end_time: 종료 시간
            config_hash: 설정 해시 (None이면 최신 것 사용)
            limit: 최대 데이터 개수

        Returns:
            처리된 데이터 DataFrame (feature_vector 포함)
        """
        query = """
            SELECT timestamp,
                   sma_5, sma_20, sma_60,
                   ema_12, ema_26,
                   rsi_14,
                   macd, macd_signal, macd_hist,
                   bb_upper, bb_middle, bb_lower, bb_width,
                   atr_14, obv,
                   stoch_k, stoch_d,
                   volume_sma,
                   price_change_1, price_change_5, price_change_20,
                   feature_vector, feature_names,
                   normalization_method, normalization_params,
                   config_hash
            FROM processed_data WHERE market = ?
        """
        params = [market]

        if config_hash:
            query += " AND config_hash = ?"
            params.append(config_hash)

        if start_time:
            query += " AND timestamp >= ?"
            params.append(int(start_time.timestamp()))

        if end_time:
            query += " AND timestamp <= ?"
            params.append(int(end_time.timestamp()))

        query += " ORDER BY timestamp ASC"

        if limit:
            query += f" LIMIT {limit}"

        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(query, conn, params=params)

        if len(df) == 0:
            self.logger.warning(f"{market} 처리된 데이터 없음")
            return None

        # 타임스탬프 변환
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        df = df.set_index('timestamp')

        # feature_vector 역직렬화
        if 'feature_vector' in df.columns:
            df['feature_vector'] = df['feature_vector'].apply(
                lambda x: pickle.loads(x) if x is not None else None
            )

        # feature_names 역직렬화
        if 'feature_names' in df.columns:
            df['feature_names'] = df['feature_names'].apply(
                lambda x: json.loads(x) if x is not None else None
            )

        # normalization_params 역직렬화
        if 'normalization_params' in df.columns:
            df['normalization_params'] = df['normalization_params'].apply(
                lambda x: json.loads(x) if x is not None else None
            )

        self.logger.info(
            f"{market} 처리된 데이터 {len(df)}건 로드 완료 "
            f"(hash: {df['config_hash'].iloc[0][:8] if len(df) > 0 else 'N/A'})"
        )
        return df

    def _generate_config_hash(
        self,
        normalization_method: str,
        normalization_params: Optional[Dict] = None
    ) -> str:
        """설정 기반 해시 생성 (캐시 무효화용)"""
        config_str = f"{normalization_method}_{normalization_params}"
        return hashlib.md5(config_str.encode()).hexdigest()

    # ========================================
    # 멀티 타임프레임 메서드 (1m, 1h, 1d)
    # ========================================

    def save_ohlcv_data_by_timeframe(
        self,
        market: str,
        data: pd.DataFrame,
        timeframe: str
    ):
        """
        타임프레임별 OHLCV 데이터 저장 (타임스탬프 정규화 + 중복 방지)

        Args:
            market: 마켓 코드 (예: KRW-BTC)
            data: OHLCV 데이터프레임 (컬럼: timestamp, open, high, low, close, volume, value)
            timeframe: 타임프레임 ('1m', '1h', '1d')
        """
        if timeframe not in ['1m', '1h', '1d']:
            raise ValueError(f"지원하지 않는 타임프레임: {timeframe}")

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # 타임스탬프 준비
            if 'timestamp' not in data.columns:
                data['timestamp'] = data.index

            data = data.copy()

            # datetime → 정규화 → Unix timestamp (int)
            if pd.api.types.is_datetime64_any_dtype(data['timestamp']):
                # 1. 타임스탬프 정규화 (초/밀리초 제거)
                data['timestamp'] = data['timestamp'].apply(lambda x: align_timestamp(x, timeframe))
                # 2. Unix timestamp로 변환
                data['timestamp'] = data['timestamp'].astype('int64') // 10**9

            # INSERT OR REPLACE 방식으로 저장 (중복 자동 처리)
            data['market'] = market

            for _, row in data.iterrows():
                cursor.execute(
                    f"""
                    INSERT OR REPLACE INTO ohlcv_{timeframe}
                    (market, timestamp, open, high, low, close, volume, value)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        market,
                        int(row['timestamp']),
                        float(row['open']),
                        float(row['high']),
                        float(row['low']),
                        float(row['close']),
                        float(row['volume']),
                        float(row.get('value', 0))
                    ]
                )

            conn.commit()
            self.logger.info(f"{market} OHLCV 데이터 {len(data)}건 저장 완료 (타임프레임: {timeframe})")

    def load_ohlcv_data_by_timeframe(
        self,
        market: str,
        timeframe: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """
        타임프레임별 OHLCV 데이터 로드

        Args:
            market: 마켓 코드
            timeframe: 타임프레임 ('1m', '1h', '1d')
            start_time: 시작 시간
            end_time: 종료 시간
            limit: 최대 데이터 개수

        Returns:
            OHLCV 데이터프레임
        """
        if timeframe not in ['1m', '1h', '1d']:
            raise ValueError(f"지원하지 않는 타임프레임: {timeframe}")

        query = f"SELECT timestamp, open, high, low, close, volume, value FROM ohlcv_{timeframe} WHERE market = ?"
        params = [market]

        if start_time:
            query += " AND timestamp >= ?"
            params.append(int(start_time.timestamp()))

        if end_time:
            query += " AND timestamp <= ?"
            params.append(int(end_time.timestamp()))

        query += " ORDER BY timestamp ASC"

        if limit:
            query += f" LIMIT {limit}"

        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(query, conn, params=params)

        if len(df) == 0:
            self.logger.warning(f"{market} 데이터 없음 (타임프레임: {timeframe})")
            return pd.DataFrame()

        # 타임스탬프 변환
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        df = df.set_index('timestamp')

        self.logger.info(f"{market} OHLCV 데이터 {len(df)}건 로드 완료 (타임프레임: {timeframe})")
        return df

    def get_data_range_by_timeframe(
        self,
        market: str,
        timeframe: str
    ) -> Tuple[Optional[datetime], Optional[datetime]]:
        """
        타임프레임별 데이터 시간 범위 조회

        Args:
            market: 마켓 코드
            timeframe: 타임프레임 ('1m', '1h', '1d')

        Returns:
            (최소 시간, 최대 시간)
        """
        if timeframe not in ['1m', '1h', '1d']:
            raise ValueError(f"지원하지 않는 타임프레임: {timeframe}")

        query = f"""
            SELECT MIN(timestamp) as min_time, MAX(timestamp) as max_time
            FROM ohlcv_{timeframe} WHERE market = ?
        """

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(query, [market])
            result = cursor.fetchone()

        if result and result[0] and result[1]:
            min_time = datetime.fromtimestamp(result[0])
            max_time = datetime.fromtimestamp(result[1])
            return min_time, max_time

        return None, None

    def get_data_count_by_timeframe(
        self,
        market: str,
        timeframe: str
    ) -> int:
        """
        타임프레임별 데이터 개수 조회

        Args:
            market: 마켓 코드
            timeframe: 타임프레임 ('1m', '1h', '1d')

        Returns:
            데이터 개수
        """
        if timeframe not in ['1m', '1h', '1d']:
            raise ValueError(f"지원하지 않는 타임프레임: {timeframe}")

        query = f"SELECT COUNT(*) FROM ohlcv_{timeframe} WHERE market = ?"

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(query, [market])
            result = cursor.fetchone()

        return result[0] if result else 0


def collect_multi_timeframe_data(
    market: str,
    start_time: datetime,
    end_time: Optional[datetime] = None,
    minute_candles_count: Optional[int] = None,
    hourly_lookback_count: int = 24,
    daily_lookback_count: int = 30,
    db_path: str = "data/market_data.db",
    show_progress: bool = True
):
    """
    멀티 타임프레임 데이터 수집 (통합 함수)

    Args:
        market: 마켓 코드 (예: KRW-BTC)
        start_time: 시작 시간
        end_time: 종료 시간 (minute_candles_count와 둘 중 하나 필수)
        minute_candles_count: 1분봉 개수 (end_time 대신 사용 가능)
        hourly_lookback_count: 시작 이전 시간봉 개수
        daily_lookback_count: 시작 이전 일봉 개수
        db_path: 데이터베이스 경로
        show_progress: 진행률 표시 여부

    Examples:
        # 방법 1: end_time 지정
        collect_multi_timeframe_data(
            market="KRW-BTC",
            start_time=datetime(2025, 10, 1),
            end_time=datetime(2025, 10, 10)
        )

        # 방법 2: minute_candles_count 지정
        collect_multi_timeframe_data(
            market="KRW-BTC",
            start_time=datetime(2025, 10, 1),
            minute_candles_count=1440  # 1일치
        )
    """
    from .market_data import UpbitDataCollector

    try:
        from tqdm import tqdm
        HAS_TQDM = True
    except ImportError:
        HAS_TQDM = False
        logging.warning("tqdm이 설치되지 않았습니다. 진행률 표시가 비활성화됩니다.")

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # 1. end_time 계산
    if end_time is None and minute_candles_count is not None:
        end_time = start_time + timedelta(minutes=minute_candles_count)
    elif end_time is None:
        raise ValueError("end_time 또는 minute_candles_count 중 하나는 필수입니다.")

    # 2. 타임프레임별 목표 범위 계산 + 정규화
    targets = {
        '1m': (
            align_timestamp(start_time, '1m'),
            align_timestamp(end_time, '1m')
        ),
        '1h': (
            align_timestamp(start_time - timedelta(hours=hourly_lookback_count), '1h'),
            align_timestamp(end_time, '1h')
        ),
        '1d': (
            align_timestamp(start_time - timedelta(days=daily_lookback_count), '1d'),
            align_timestamp(end_time, '1d')
        )
    }

    logger.info(f"[멀티 타임프레임 수집 시작] {market}")
    logger.info(f"  기간: {start_time} ~ {end_time}")
    logger.info(f"  추가 lookback: 시간봉 {hourly_lookback_count}개, 일봉 {daily_lookback_count}개")

    # 3. 데이터 수집 및 저장
    storage = MarketDataStorage(db_path)
    collector = UpbitDataCollector(market)

    for timeframe, (target_start, target_end) in targets.items():
        logger.info(f"[{timeframe}] 목표: {target_start} ~ {target_end}")

        # 누락 범위 계산
        missing_ranges = get_missing_ranges(storage, market, timeframe, target_start, target_end)

        if not missing_ranges:
            logger.info(f"  ✅ 이미 모든 데이터 존재 (수집 스킵)")
            continue

        # 누락 구간 수집
        for miss_start, miss_end in missing_ranges:
            logger.info(f"  📥 수집 중: {miss_start} ~ {miss_end}")

            # 필요 봉 개수 계산
            unit_minutes = {'1m': 1, '1h': 60, '1d': 1440}[timeframe]
            total_candles = int((miss_end - miss_start).total_seconds() / (unit_minutes * 60))

            # 200개씩 분할 수집
            MAX_PER_REQUEST = UpbitDataCollector.MAX_CANDLES_PER_REQUEST
            num_requests = (total_candles + MAX_PER_REQUEST - 1) // MAX_PER_REQUEST

            all_data = []
            pbar = None

            if show_progress and HAS_TQDM:
                pbar = tqdm(total=num_requests, desc=f"  [{timeframe}] 수집", leave=False)

            for i in range(num_requests):
                count = min(MAX_PER_REQUEST, total_candles - i * MAX_PER_REQUEST)

                # API 요청 (Rate Limit 자동 처리)
                data = collector.get_historical_data(count=count, unit=unit_minutes)

                if data is not None and not data.empty:
                    all_data.append(data)

                if pbar:
                    pbar.update(1)

            if pbar:
                pbar.close()

            # 데이터 병합 및 저장
            if all_data:
                merged_data = pd.concat(all_data).sort_index()

                # 타임스탬프 정규화
                merged_data.index = merged_data.index.map(lambda x: align_timestamp(x, timeframe))

                # 중복 제거 후 저장
                merged_data = merged_data[~merged_data.index.duplicated(keep='first')]
                storage.save_ohlcv_data_by_timeframe(market, merged_data, timeframe)
                logger.info(f"  ✅ {len(merged_data)}개 저장 완료")
            else:
                logger.error(f"  ❌ 수집 실패")

    # 4. 최종 결과
    logger.info("\n[수집 완료]")
    for tf in ['1m', '1h', '1d']:
        count = storage.get_data_count_by_timeframe(market, tf)
        data_range = storage.get_data_range_by_timeframe(market, tf)
        logger.info(f"  {tf}: {count}건, 범위: {data_range[0]} ~ {data_range[1]}")


def collect_and_store_data(
    market: str = "KRW-BTC",
    count: int = 500,
    unit: int = 1,
    db_path: str = "data/market_data.db"
):
    """
    Upbit에서 데이터를 수집하고 SQLite에 저장

    Args:
        market: 마켓 코드
        count: 수집할 캔들 개수
        unit: 캔들 단위 (분)
        db_path: 데이터베이스 경로
    """
    from .market_data import UpbitDataCollector

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # 데이터 수집
    logger.info(f"{market} 데이터 수집 시작 (count={count}, unit={unit}분)")
    collector = UpbitDataCollector(market)
    data = collector.get_historical_data(count=count, unit=unit)

    if data is None or len(data) == 0:
        logger.error("데이터 수집 실패")
        return

    # 데이터 저장
    storage = MarketDataStorage(db_path)
    storage.save_ohlcv_data(market, data)

    # 저장 결과 확인
    data_count = storage.get_data_count(market)
    data_range = storage.get_data_range(market)
    logger.info(f"저장 완료: {data_count}건, 범위: {data_range[0]} ~ {data_range[1]}")


if __name__ == "__main__":
    # 사용 예제
    collect_and_store_data(market="KRW-BTC", count=1000, unit=1)
