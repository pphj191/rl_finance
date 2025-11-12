"""
Technical Indicators and Feature Extraction

기술적 지표(개인들의 시선) 계산 및 특성 추출을 담당하는 모듈입니다.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any


class FeatureExtractor:
    """기술적 지표 및 특성 추출"""
    
    def __init__(self):
        self.features = []
    
    def extract_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """기술적 지표 계산"""
        features = df.copy()
        
        # 가격 관련 지표
        if 'close' in df.columns:
            close_prices = df['close'].astype(float)
            
            # 이동평균 (수동 계산)
            features['sma_5'] = close_prices.rolling(window=5).mean()
            features['sma_20'] = close_prices.rolling(window=20).mean()
            features['sma_60'] = close_prices.rolling(window=60).mean()
            
            # 지수이동평균 (수동 계산)
            features['ema_12'] = close_prices.ewm(span=12).mean()
            features['ema_26'] = close_prices.ewm(span=26).mean()
            
            # 볼린저 밴드 (수동 계산)
            bb_period = 20
            bb_std = 2
            sma_20 = close_prices.rolling(window=bb_period).mean()
            std_20 = close_prices.rolling(window=bb_period).std()
            features['bb_upper'] = sma_20 + (std_20 * bb_std)
            features['bb_middle'] = sma_20
            features['bb_lower'] = sma_20 - (std_20 * bb_std)
            features['bb_width'] = (features['bb_upper'] - features['bb_lower']) / features['bb_middle']
            
            # RSI (수동 계산 - 간단한 방식)
            rsi_period = 14
            delta = close_prices.diff()
            gain = delta.copy()
            loss = delta.copy()
            gain[gain < 0] = 0
            loss[loss > 0] = 0
            loss = -loss
            
            gain_avg = gain.rolling(window=rsi_period).mean()
            loss_avg = loss.rolling(window=rsi_period).mean()
            rs = gain_avg / loss_avg
            features['rsi'] = 100 - (100 / (1 + rs))
            
            # MACD (수동 계산)
            exp1 = close_prices.ewm(span=12).mean()
            exp2 = close_prices.ewm(span=26).mean()
            features['macd'] = exp1 - exp2
            features['macd_signal'] = features['macd'].ewm(span=9).mean()
            features['macd_histogram'] = features['macd'] - features['macd_signal']
            
            # 스토캐스틱 (수동 계산)
            if 'high' in df.columns and 'low' in df.columns:
                high_prices = df['high'].astype(float)
                low_prices = df['low'].astype(float)
                k_period = 14
                lowest_low = low_prices.rolling(window=k_period).min()
                highest_high = high_prices.rolling(window=k_period).max()
                features['stoch_k'] = 100 * ((close_prices - lowest_low) / (highest_high - lowest_low))
                features['stoch_d'] = features['stoch_k'].rolling(window=3).mean()
        
        # 거래량 지표
        if 'volume' in df.columns and 'close' in df.columns:
            volume = df['volume'].astype(float)
            close_prices = df['close'].astype(float)
            
            # 거래량 이동평균 (수동 계산)
            features['volume_sma'] = volume.rolling(window=20).mean()
            
            # 온밸런스 볼륨 (수동 계산)
            obv = [0.0]
            for i in range(1, len(df)):
                if close_prices.iloc[i] > close_prices.iloc[i-1]:
                    obv.append(obv[-1] + float(volume.iloc[i]))
                elif close_prices.iloc[i] < close_prices.iloc[i-1]:
                    obv.append(obv[-1] - float(volume.iloc[i]))
                else:
                    obv.append(obv[-1])
            features['obv'] = obv
        
        # 변동성 지표
        if 'high' in df.columns and 'low' in df.columns and 'close' in df.columns:
            high_prices = df['high'].astype(float)
            low_prices = df['low'].astype(float)
            close_prices = df['close'].astype(float)
            
            # ATR (Average True Range) - 수동 계산
            high_low = high_prices - low_prices
            high_close = np.abs(high_prices - close_prices.shift())
            low_close = np.abs(low_prices - close_prices.shift())
            
            # DataFrame으로 결합
            true_range_df = pd.DataFrame({
                'hl': high_low,
                'hc': high_close,
                'lc': low_close
            })
            true_range = true_range_df.max(axis=1)
            features['atr'] = true_range.rolling(window=14).mean()
        
        # 가격 변화율
        if 'close' in df.columns:
            close_prices = df['close'].astype(float)
            features['price_change_1'] = close_prices.pct_change(1)
            features['price_change_5'] = close_prices.pct_change(5)
            features['price_change_20'] = close_prices.pct_change(20)
        
        return features

    def create_time_windows(self, data: pd.DataFrame, window_size: int = 60, 
                           step_size: int = 1) -> List[pd.DataFrame]:
        """시계열 윈도우 생성"""
        windows = []
        
        for i in range(0, len(data) - window_size + 1, step_size):
            window = data.iloc[i:i + window_size].copy()
            windows.append(window)
        
        return windows
    
    def prepare_sequence_data(self, data: pd.DataFrame, sequence_length: int = 60,
                            target_columns: Optional[List[str]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """시퀀스 데이터 준비 (LSTM/RNN용)"""
        if target_columns is None:
            target_columns = ['close']
        
        # 수치형 컬럼만 선택
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        feature_data = data[numeric_columns].ffill().fillna(0)
        
        X, y = [], []
        
        for i in range(sequence_length, len(feature_data)):
            # 입력 시퀀스
            X.append(feature_data.iloc[i-sequence_length:i].values)

            # 타겟 (다음 시점의 목표 값들)
            target_values = []
            for col in target_columns:
                if col in feature_data.columns:
                    target_values.append(feature_data[col].iloc[i])
            y.append(target_values)

        return np.array(X), np.array(y)

    def extract_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        모든 기술적 지표 추출

        Args:
            df: 원본 OHLCV 데이터

        Returns:
            기술적 지표가 포함된 DataFrame

        Note:
            SSL 특성은 별도 모듈(ssl_features.py)에서 처리
        """
        # 기술적 지표 계산
        features = self.extract_technical_indicators(df)

        return features

    def get_feature_vector(
        self,
        df: pd.DataFrame,
        exclude_columns: Optional[List[str]] = None
    ) -> Tuple[np.ndarray, List[str]]:
        """
        DataFrame을 numpy array로 변환 (환경에서 사용)

        Args:
            df: 특성 DataFrame
            exclude_columns: 제외할 컬럼 리스트

        Returns:
            (feature_vector, feature_names)
        """
        if exclude_columns is None:
            exclude_columns = []

        # 수치형 컬럼 선택
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()

        # 제외할 컬럼 필터링
        feature_columns = [
            col for col in numeric_columns
            if col not in exclude_columns
        ]

        # numpy array로 변환
        feature_vector = df[feature_columns].values

        return feature_vector, feature_columns

    def get_feature_names(self, df: pd.DataFrame) -> List[str]:
        """
        특성 이름 리스트 반환

        Args:
            df: 특성 DataFrame

        Returns:
            특성 이름 리스트
        """
        return df.select_dtypes(include=[np.number]).columns.tolist()
