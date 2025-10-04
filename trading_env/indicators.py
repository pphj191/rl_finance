"""
Technical Indicators and Feature Extraction

기술적 지표 계산 및 특성 추출을 담당하는 모듈입니다.
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
    
    def extract_ssl_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """SSL (Self-Supervised Learning) 특성 추출"""
        ssl_features = data.copy()
        
        if 'close' in data.columns:
            close_prices = data['close'].astype(float)
            
            # 1. 시계열 대조 학습 기반 특성
            ssl_features = self._add_contrastive_features(ssl_features, close_prices)
            
            # 2. 마스킹 기반 예측 특성
            ssl_features = self._add_masked_prediction_features(ssl_features, close_prices)
            
            # 3. 시간적 패턴 인식 특성
            ssl_features = self._add_temporal_pattern_features(ssl_features, close_prices)
        
        return ssl_features
    
    def _add_contrastive_features(self, features: pd.DataFrame, prices: pd.Series) -> pd.DataFrame:
        """대조 학습 기반 특성 추출"""
        # 가격 변화 패턴 기반 유사도
        window_sizes = [5, 10, 20]
        
        for window in window_sizes:
            # 롤링 윈도우별 가격 변화 패턴
            price_changes = prices.pct_change(1).rolling(window=window)
            
            # 패턴 유사도 (현재 패턴과 과거 패턴들의 코사인 유사도)
            features[f'pattern_similarity_{window}'] = price_changes.apply(
                lambda x: self._calculate_pattern_similarity(x, prices, window), raw=False
            )
            
            # 변동성 클러스터링
            volatility = prices.rolling(window=window).std()
            features[f'volatility_regime_{window}'] = self._classify_volatility_regime(volatility)
        
        return features
    
    def _add_masked_prediction_features(self, features: pd.DataFrame, prices: pd.Series) -> pd.DataFrame:
        """마스킹 기반 예측 특성"""
        # 가격 예측 오차 기반 특성
        prediction_windows = [3, 7, 14]
        
        for window in prediction_windows:
            # 단순 이동평균 예측 vs 실제
            sma_pred = prices.rolling(window=window).mean().shift(1)
            prediction_error = np.abs(prices - sma_pred) / prices
            features[f'prediction_error_{window}'] = prediction_error
            
            # 예측 신뢰도 (과거 예측 정확도 기반)
            features[f'prediction_confidence_{window}'] = self._calculate_prediction_confidence(
                prices, window
            )
        
        return features
    
    def _add_temporal_pattern_features(self, features: pd.DataFrame, prices: pd.Series) -> pd.DataFrame:
        """시간적 패턴 인식 특성"""
        # 주기성 분석
        for period in [5, 10, 20, 50]:  # 5분, 10분, 20분, 50분 주기
            if len(prices) > period:
                # 자기상관 (autocorrelation)
                features[f'autocorr_{period}'] = prices.rolling(window=period*2).apply(
                    lambda x: self._calculate_autocorrelation(x, period), raw=False
                )
                
                # 주기적 트렌드
                features[f'periodic_trend_{period}'] = self._extract_periodic_trend(prices, period)
        
        # 트렌드 강도 및 방향
        features['trend_strength'] = self._calculate_trend_strength(prices)
        features['trend_direction'] = self._calculate_trend_direction(prices)
        
        return features
    
    def _calculate_pattern_similarity(self, current_pattern: pd.Series, all_prices: pd.Series, window: int) -> float:
        """현재 패턴과 과거 패턴들의 유사도 계산"""
        if len(current_pattern) < window or current_pattern.isna().all():
            return 0.0
        
        current_vec = np.array(current_pattern.values, dtype=float)
        if len(current_vec) < window:
            return 0.0
        
        # 정규화
        current_mean = float(np.mean(current_vec))
        current_std = float(np.std(current_vec))
        current_vec = (current_vec - current_mean) / (current_std + 1e-8)
        
        similarities = []
        for i in range(window, len(all_prices) - window):
            past_pattern = all_prices.iloc[i-window:i].pct_change(1).dropna().values
            if len(past_pattern) >= window-1:
                past_vec = np.array(past_pattern, dtype=float)
                past_mean = float(np.mean(past_vec))
                past_std = float(np.std(past_vec))
                past_vec = (past_vec - past_mean) / (past_std + 1e-8)
                
                # 코사인 유사도
                similarity = np.dot(current_vec[1:], past_vec) / (
                    np.linalg.norm(current_vec[1:]) * np.linalg.norm(past_vec) + 1e-8
                )
                similarities.append(float(similarity))
        
        return float(np.mean(similarities)) if similarities else 0.0
    
    def _classify_volatility_regime(self, volatility: pd.Series) -> pd.Series:
        """변동성 클러스터링"""
        # 변동성을 3개 구간으로 분류 (낮음, 보통, 높음)
        quantiles = volatility.quantile([0.33, 0.67])
        regime = pd.Series(1, index=volatility.index)  # 기본값: 보통 변동성
        regime[volatility <= quantiles.iloc[0]] = 0  # 낮은 변동성
        regime[volatility >= quantiles.iloc[1]] = 2  # 높은 변동성
        return regime
    
    def _calculate_prediction_confidence(self, prices: pd.Series, window: int) -> pd.Series:
        """예측 신뢰도 계산"""
        confidence = pd.Series(0.5, index=prices.index)  # 기본 신뢰도 50%
        
        for i in range(window*2, len(prices)):
            # 과거 window 구간에서의 예측 정확도 계산
            past_prices = prices.iloc[i-window*2:i-window]
            past_predictions = past_prices.rolling(window=window//2).mean()
            actual_prices = prices.iloc[i-window:i]
            
            if len(past_predictions) > 0 and len(actual_prices) > 0:
                # MAPE (Mean Absolute Percentage Error)의 역수로 신뢰도 계산
                actual_values = np.array(actual_prices.values, dtype=float)
                pred_values = np.array(past_predictions.values[-len(actual_prices):], dtype=float)
                errors = np.abs(actual_values - pred_values)
                mape = float(np.mean(errors / (actual_values + 1e-8)))
                confidence.iloc[i] = float(1.0 / (1.0 + mape))
        
        return confidence
    
    def _calculate_autocorrelation(self, series: pd.Series, lag: int) -> float:
        """자기상관 계산"""
        if len(series) <= lag:
            return 0.0
        
        y1 = np.array(series.values[:-lag], dtype=float)
        y2 = np.array(series.values[lag:], dtype=float)
        
        if len(y1) == 0 or len(y2) == 0:
            return 0.0
        
        # 피어슨 상관계수
        correlation = np.corrcoef(y1, y2)[0, 1]
        return float(correlation) if not np.isnan(correlation) else 0.0
    
    def _extract_periodic_trend(self, prices: pd.Series, period: int) -> pd.Series:
        """주기적 트렌드 추출"""
        trend = pd.Series(0.0, index=prices.index)
        
        for i in range(period, len(prices)):
            current_segment = prices.iloc[i-period:i]
            if len(current_segment) == period:
                # 선형 회귀 기울기로 트렌드 계산
                x = np.arange(len(current_segment))
                y = np.array(current_segment.values, dtype=float)
                slope = np.polyfit(x, y, 1)[0] if len(y) > 1 else 0.0
                trend.iloc[i] = float(slope)
        
        return trend
    
    def _calculate_trend_strength(self, prices: pd.Series, window: int = 20) -> pd.Series:
        """트렌드 강도 계산"""
        # R-squared 기반 트렌드 강도
        strength = pd.Series(0.0, index=prices.index)
        
        for i in range(window, len(prices)):
            segment = prices.iloc[i-window:i]
            x = np.arange(len(segment))
            y = np.array(segment.values, dtype=float)
            
            # 선형 회귀의 R-squared 값
            if len(y) > 1:
                slope, intercept = np.polyfit(x, y, 1)
                predicted = slope * x + intercept
                ss_res = np.sum((y - predicted) ** 2)
                y_mean = float(np.mean(y))
                ss_tot = np.sum((y - y_mean) ** 2)
                r_squared = 1 - (ss_res / (ss_tot + 1e-8))
                strength.iloc[i] = max(0, float(r_squared))
        
        return strength
    
    def _calculate_trend_direction(self, prices: pd.Series, window: int = 20) -> pd.Series:
        """트렌드 방향 계산"""
        # 이동평균 기울기 기반
        sma = prices.rolling(window=window).mean()
        direction = sma.diff() / sma
        return direction.fillna(0)
    
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
