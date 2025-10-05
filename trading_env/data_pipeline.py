"""
통합 데이터 파이프라인

Upbit 데이터 수집 → 기술적 지표 계산 → 특성 추출 → SQLite 저장/로드
오프라인 및 실시간 모드 지원
"""

import pandas as pd
import numpy as np
import logging
import hashlib
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from .data_storage import MarketDataStorage
from .indicators import FeatureExtractor
from .market_data import DataNormalizer, UpbitDataCollector


class DataPipeline:
    """통합 데이터 파이프라인 (오프라인/실시간 지원)"""

    def __init__(
        self,
        storage: MarketDataStorage,
        mode: str = "offline",
        cache_enabled: bool = True,
        normalization_method: str = "robust",
        include_ssl: bool = True
    ):
        """
        Args:
            storage: MarketDataStorage 인스턴스
            mode: "offline" (SQLite만) | "realtime" (캐시+계산)
            cache_enabled: 캐시 사용 여부
            normalization_method: 정규화 방법 ("standard", "minmax", "robust")
            include_ssl: SSL 특성 포함 여부
        """
        self.storage = storage
        self.mode = mode
        self.cache_enabled = cache_enabled
        self.normalization_method = normalization_method
        self.include_ssl = include_ssl

        # 컴포넌트 초기화
        self.feature_extractor = FeatureExtractor()
        self.normalizer = DataNormalizer(method=normalization_method)

        # 로거
        self.logger = logging.getLogger(__name__)

    def process_data(
        self,
        market: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        force_recalculate: bool = False
    ) -> pd.DataFrame:
        """
        데이터 처리 파이프라인 (메인 메서드)

        Args:
            market: 마켓 코드
            start_time: 시작 시간
            end_time: 종료 시간
            force_recalculate: 강제 재계산 여부

        Returns:
            처리된 데이터 DataFrame (기술적 지표 + 특성)
        """
        # 설정 해시 생성
        config_hash = self._generate_config_hash()

        # 1. 캐시 확인 (오프라인 또는 캐시 활성화 시)
        if self.cache_enabled and not force_recalculate:
            cached_data = self._load_from_cache(
                market, start_time, end_time, config_hash
            )
            if cached_data is not None and len(cached_data) > 0:
                self.logger.info(f"캐시에서 데이터 로드: {market} ({len(cached_data)}건)")
                return cached_data

        # 2. 오프라인 모드에서 캐시 미스 시 에러
        if self.mode == "offline":
            raise ValueError(
                f"{market} 데이터가 캐시에 없습니다. "
                f"prepare_offline_data 스크립트로 데이터를 먼저 준비하세요."
            )

        # 3. 실시간 모드: 원본 데이터 로드 또는 수집
        raw_data = self._load_or_collect_raw_data(market, start_time, end_time)

        if raw_data is None or raw_data.empty:
            raise ValueError(f"{market} 데이터를 로드/수집할 수 없습니다.")

        # 4. 기술적 지표 계산 + 특성 추출
        processed_data = self._calculate_features(raw_data)

        # 5. 캐시 저장 (활성화 시)
        if self.cache_enabled:
            self._save_to_cache(market, processed_data, config_hash)

        return processed_data

    def _load_from_cache(
        self,
        market: str,
        start_time: Optional[datetime],
        end_time: Optional[datetime],
        config_hash: str
    ) -> Optional[pd.DataFrame]:
        """SQLite 캐시에서 데이터 로드"""
        try:
            return self.storage.load_processed_data(
                market=market,
                start_time=start_time,
                end_time=end_time,
                config_hash=config_hash
            )
        except Exception as e:
            self.logger.warning(f"캐시 로드 실패: {e}")
            return None

    def _load_or_collect_raw_data(
        self,
        market: str,
        start_time: Optional[datetime],
        end_time: Optional[datetime]
    ) -> Optional[pd.DataFrame]:
        """원본 OHLCV 데이터 로드 또는 수집"""
        # 1. SQLite에서 로드 시도
        raw_data = self.storage.load_ohlcv_data(market, start_time, end_time)

        if raw_data is not None and not raw_data.empty:
            return raw_data

        # 2. Upbit API에서 실시간 수집
        if self.mode == "realtime":
            self.logger.info(f"Upbit API에서 {market} 데이터 수집 중...")
            return self._collect_realtime_data(market)

        return None

    def _collect_realtime_data(self, market: str, count: int = 500) -> pd.DataFrame:
        """Upbit API에서 실시간 데이터 수집"""
        try:
            collector = UpbitDataCollector(market)
            data = collector.get_historical_data(count=count, unit=1)

            # SQLite에 저장 (나중에 재사용)
            if data is not None and not data.empty:
                self.storage.save_ohlcv_data(market, data)

            return data

        except Exception as e:
            self.logger.error(f"실시간 데이터 수집 실패: {e}")
            return pd.DataFrame()

    def _calculate_features(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """기술적 지표 계산 + 특성 추출"""
        # 1. 모든 특성 추출
        features = self.feature_extractor.extract_all(
            raw_data, include_ssl=self.include_ssl
        )

        # 2. 정규화
        numeric_columns = features.select_dtypes(include=[np.number]).columns.tolist()
        features = self.normalizer.fit_transform(features, numeric_columns)

        # NaN 처리
        features = features.ffill().fillna(0)

        return features

    def _save_to_cache(
        self,
        market: str,
        processed_data: pd.DataFrame,
        config_hash: str
    ):
        """처리된 데이터를 SQLite에 저장"""
        try:
            # 특성 벡터 추출
            feature_vector, feature_names = self.feature_extractor.get_feature_vector(
                processed_data
            )

            # 정규화 파라미터 (간단하게)
            normalization_params = {
                "method": self.normalization_method,
                "include_ssl": self.include_ssl
            }

            # SQLite에 저장
            self.storage.save_processed_data(
                market=market,
                data=processed_data,
                feature_vector=feature_vector,
                feature_names=feature_names,
                normalization_method=self.normalization_method,
                normalization_params=normalization_params,
                config_hash=config_hash
            )

        except Exception as e:
            self.logger.error(f"캐시 저장 실패: {e}")

    def _generate_config_hash(self) -> str:
        """설정 기반 해시 생성 (캐시 무효화용)"""
        config_str = f"{self.normalization_method}_{self.include_ssl}"
        return hashlib.md5(config_str.encode()).hexdigest()


def prepare_offline_data(
    market: str,
    days: int = 30,
    db_path: str = "data/market_data.db",
    normalization_method: str = "robust",
    include_ssl: bool = True
):
    """
    오프라인 학습용 데이터 준비

    Args:
        market: 마켓 코드
        days: 수집할 일수
        db_path: 데이터베이스 경로
        normalization_method: 정규화 방법
        include_ssl: SSL 특성 포함 여부
    """
    from .data_storage import collect_and_store_data

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # 1. 원본 데이터 수집
    logger.info(f"[1/3] 데이터 수집 중... ({market}, {days}일)")
    collect_and_store_data(
        market=market,
        count=days * 24 * 60,  # 1분봉 기준
        unit=1,
        db_path=db_path
    )

    # 2. 기술적 지표 계산 + 특성 추출
    logger.info("[2/3] 지표 계산 및 특성 추출 중...")
    storage = MarketDataStorage(db_path)
    pipeline = DataPipeline(
        storage=storage,
        mode="realtime",  # 계산 모드
        cache_enabled=True,
        normalization_method=normalization_method,
        include_ssl=include_ssl
    )

    start_time = datetime.now() - timedelta(days=days)
    end_time = datetime.now()

    processed_data = pipeline.process_data(
        market=market,
        start_time=start_time,
        end_time=end_time,
        force_recalculate=True  # 강제 재계산
    )

    # 3. 검증
    logger.info("[3/3] 데이터 검증 중...")
    logger.info(f"✅ 완료: {len(processed_data)}개 레코드 준비됨")
    logger.info(f"   데이터 범위: {processed_data.index[0]} ~ {processed_data.index[-1]}")
    logger.info(f"   특성 개수: {len(processed_data.columns)}")

    return processed_data


if __name__ == "__main__":
    # 테스트
    prepare_offline_data(market="KRW-BTC", days=7)
