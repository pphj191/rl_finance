"""
Market Data Collection and Processing

시장 데이터 수집, 정규화, 전처리를 담당하는 모듈입니다.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# Upbit API 가져오기 - 실제 프로젝트에서는 경로 수정 필요
try:
    from upbit_api.upbit_api import UpbitAPI
except ImportError:
    logging.warning("UpbitAPI를 가져올 수 없습니다. 상대 경로를 확인하세요.")
    UpbitAPI = None


class DataNormalizer:
    """데이터 정규화 클래스"""
    
    def __init__(self, method: str = "robust", window: int = 252):
        self.method = method
        self.window = window
        self.scalers = {}
    
    def fit_transform(self, data: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """데이터 정규화"""
        if columns is None:
            columns = data.select_dtypes(include=[np.number]).columns.tolist()
        
        normalized_data = data.copy()
        
        for col in columns:
            if col not in data.columns:
                continue
                
            if self.method == "standard":
                scaler = StandardScaler()
            elif self.method == "minmax":
                scaler = MinMaxScaler()
            elif self.method == "robust":
                scaler = RobustScaler()
            else:
                raise ValueError(f"Unknown normalization method: {self.method}")
            
            # Rolling window 정규화
            if len(data) > self.window:
                normalized_values = []
                for i in range(len(data)):
                    start_idx = max(0, i - self.window + 1)
                    end_idx = i + 1
                    window_data = data[col].iloc[start_idx:end_idx].values.reshape(-1, 1)
                    
                    scaler.fit(window_data)
                    normalized_val = scaler.transform([[data[col].iloc[i]]])[0, 0]
                    normalized_values.append(normalized_val)
                
                normalized_data[col] = normalized_values
            else:
                # 전체 데이터로 정규화
                values = np.array(data[col]).reshape(-1, 1)
                scaler.fit(values)
                normalized_data[col] = scaler.transform(values).flatten()
            
            self.scalers[col] = scaler
        
        return normalized_data
    
    def transform(self, data: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """기존 스케일러로 변환"""
        if columns is None:
            columns = list(self.scalers.keys())
        
        transformed_data = data.copy()
        
        for col in columns:
            if col in self.scalers and col in data.columns:
                values = np.array(data[col]).reshape(-1, 1)
                transformed_data[col] = self.scalers[col].transform(values).flatten()
        
        return transformed_data


class UpbitDataCollector:
    """Upbit 데이터 수집기"""
    
    def __init__(self, market: str = "KRW-BTC"):
        if UpbitAPI is None:
            raise ImportError("UpbitAPI를 가져올 수 없습니다. upbit_api 모듈을 확인하세요.")
        
        self.upbit = UpbitAPI()
        self.market = market
        
        # FeatureExtractor는 필요시 동적으로 import
        self._feature_extractor = None
    
    @property
    def feature_extractor(self):
        """지연 로딩으로 FeatureExtractor 가져오기"""
        if self._feature_extractor is None:
            from .indicators import FeatureExtractor
            self._feature_extractor = FeatureExtractor()
        return self._feature_extractor
    
    def get_historical_data(self, count: int = 200, unit: int = 1) -> pd.DataFrame:
        """과거 데이터 수집"""
        try:
            # 분봉 데이터 수집
            candles = self.upbit.get_candles_minutes(self.market, unit=unit, count=count)
            
            # DataFrame 변환
            df = pd.DataFrame(candles)
            df['timestamp'] = pd.to_datetime(df['candle_date_time_kst'])
            
            # 필요한 컬럼 선택 및 이름 변경
            df = df.rename(columns={
                'opening_price': 'open',
                'high_price': 'high',
                'low_price': 'low',
                'trade_price': 'close',
                'candle_acc_trade_volume': 'volume',
                'candle_acc_trade_price': 'value'
            })
            
            # 시간순 정렬
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            # 기술적 지표 추가 (feature_extractor 자동 로드됨)
            df = self.feature_extractor.extract_technical_indicators(df)
            
            return df
            
        except Exception as e:
            logging.error(f"데이터 수집 오류: {e}")
            return pd.DataFrame()
    
    def get_current_data(self) -> Dict[str, float]:
        """현재 시장 데이터 수집"""
        try:
            # 현재가 정보
            ticker = self.upbit.get_ticker(self.market)[0]
            
            # 호가 정보
            orderbook = self.upbit.get_orderbook(self.market)[0]
            
            return {
                'current_price': float(ticker['trade_price']),
                'volume': float(ticker['acc_trade_volume_24h']),
                'change_rate': float(ticker['change_rate']),
                'bid_price': float(orderbook['orderbook_units'][0]['bid_price']),
                'ask_price': float(orderbook['orderbook_units'][0]['ask_price']),
                'bid_size': float(orderbook['orderbook_units'][0]['bid_size']),
                'ask_size': float(orderbook['orderbook_units'][0]['ask_size']),
            }
            
        except Exception as e:
            logging.error(f"현재 데이터 수집 오류: {e}")
            return {}
