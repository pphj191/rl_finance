"""
강화학습 기반 암호화폐 트레이딩 시스템

TODO:
1. 환경 구성
   - ✅ 기본 환경 클래스 구조 설계
   - ✅ 액션 스페이스 정의 (구매, 보류, 판매)
   - ✅ 상태 공간 정의 (시장 데이터 + 포트폴리오)
   - ✅ 리워드 함수 설계
   - ✅ 액션 마스킹 구현

2. 데이터 처리
   - ✅ Upbit 데이터 수집기
   - ✅ 데이터 정규화 (MinMax, Z-score, Robust)
   - ✅ 기술적 지표 계산
   - ✅ SSL (Self-Supervised Learning) 특성 추출기
   - ✅ 시계열 윈도우 생성

3. 모델 구조
   - ✅ DQN 기본 모델
   - ✅ LSTM/GRU 기반 모델
   - ✅ Transformer 기반 모델
   - ✅ 앙상블 모델

4. 학습 및 평가
   - ✅ 백테스팅 환경
   - ✅ 학습 루프
   - ✅ 성과 지표 계산
   - ✅ 모델 저장/로드

5. 실제 트레이딩
   - ⬜ 실시간 데이터 수집
   - ⬜ 모델 추론
   - ⬜ 주문 실행
   - ⬜ 리스크 관리
"""

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import logging
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import ta  # 기술적 분석 라이브러리

# Upbit API import 수정
from upbit_api import UpbitAPI
import warnings
warnings.filterwarnings('ignore')


@dataclass
class TradingConfig:
    """트레이딩 설정"""
    # 환경 설정
    initial_balance: float = 1000000.0  # 초기 자금 (100만원)
    max_position: float = 1.0  # 최대 포지션 비율
    transaction_fee: float = 0.0005  # 거래 수수료 (0.05%)
    
    # 데이터 설정
    lookback_window: int = 60  # 과거 데이터 윈도우 크기
    update_interval: int = 60  # 데이터 업데이트 간격 (초)
    
    # 정규화 설정
    normalization_method: str = "robust"  # "standard", "minmax", "robust"
    feature_window: int = 252  # 정규화를 위한 rolling window (1년)
    
    # 모델 설정
    model_type: str = "dqn"  # "dqn", "lstm", "transformer"
    hidden_size: int = 256
    num_layers: int = 3
    dropout: float = 0.1
    
    # 학습 설정
    learning_rate: float = 1e-4
    batch_size: int = 32
    memory_size: int = 10000
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995
    target_update: int = 100


class ActionSpace:
    """액션 공간 정의"""
    HOLD = 0    # 보류
    BUY = 1     # 구매
    SELL = 2    # 판매
    
    @classmethod
    def get_action_names(cls) -> List[str]:
        return ["HOLD", "BUY", "SELL"]
    
    @classmethod
    def get_num_actions(cls) -> int:
        return 3


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


class UpbitDataCollector:
    """Upbit 데이터 수집기"""
    
    def __init__(self, market: str = "KRW-BTC"):
        self.upbit = UpbitAPI()
        self.market = market
        self.feature_extractor = FeatureExtractor()
    
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
            
            # 기술적 지표 추가
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


class TradingEnvironment(gym.Env):
    """강화학습 트레이딩 환경"""
    
    def __init__(self, config: TradingConfig, market: str = "KRW-BTC"):
        super().__init__()
        
        self.config = config
        self.market = market
        self.data_collector = UpbitDataCollector(market)
        self.normalizer = DataNormalizer(
            method=config.normalization_method,
            window=config.feature_window
        )
        
        # 상태 및 액션 공간 정의
        self.action_space = spaces.Discrete(ActionSpace.get_num_actions())
        
        # 관측 공간 (나중에 실제 데이터 기반으로 조정)
        # self.observation_space = None  # reset에서 설정됨
        
        # 환경 상태
        self.reset()
    
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        """환경 초기화"""
        super().reset(seed=seed)
        
        # 포트폴리오 초기화
        self.balance = self.config.initial_balance
        self.position = 0.0  # 보유 코인 수량
        self.total_value = self.balance
        
        # 데이터 수집 및 전처리
        self.data = self._prepare_data()
        self.current_step = self.config.lookback_window
        
        # 관측 공간 정의 (데이터 기반)
        if self.observation_space is None:
            obs_shape = self._get_observation().shape
            self.observation_space = spaces.Box(
                low=-np.inf, 
                high=np.inf, 
                shape=obs_shape, 
                dtype=np.float32
            )
        
        return self._get_observation(), {}
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """환경 스텝 실행"""
        # 액션 마스킹 적용
        action = self._apply_action_mask(action)
        
        # 액션 실행
        reward = self._execute_action(action)
        
        # 다음 스텝으로 이동
        self.current_step += 1
        
        # 종료 조건 확인
        terminated = self.current_step >= len(self.data) - 1
        truncated = False
        
        # 관측값 업데이트
        observation = self._get_observation()
        
        # 추가 정보
        info = {
            'balance': self.balance,
            'position': self.position,
            'total_value': self.total_value,
            'action': ActionSpace.get_action_names()[action],
            'current_price': self._get_current_price()
        }
        
        return observation, reward, terminated, truncated, info
    
    def _prepare_data(self) -> pd.DataFrame:
        """데이터 준비 및 전처리"""
        # 과거 데이터 수집
        raw_data = self.data_collector.get_historical_data(
            count=500,  # 충분한 데이터 수집
            unit=1  # 1분봉
        )
        
        if raw_data.empty:
            raise ValueError("데이터 수집 실패")
        
        # 수치형 컬럼 선택
        numeric_columns = raw_data.select_dtypes(include=[np.number]).columns.tolist()
        
        # 정규화
        normalized_data = self.normalizer.fit_transform(raw_data, numeric_columns)
        
        # NaN 값 처리
        normalized_data = normalized_data.ffill().fillna(0)
        
        return normalized_data
    
    def _get_observation(self) -> np.ndarray:
        """현재 관측값 생성"""
        # 시장 데이터 (lookback window)
        start_idx = max(0, self.current_step - self.config.lookback_window)
        end_idx = self.current_step + 1
        
        market_data = self.data.iloc[start_idx:end_idx]
        
        # 수치형 데이터만 선택
        numeric_columns = market_data.select_dtypes(include=[np.number]).columns.tolist()
        market_features = market_data[numeric_columns].values
        
        # 윈도우 크기 맞추기
        if len(market_features) < self.config.lookback_window:
            padding = np.zeros((self.config.lookback_window - len(market_features), 
                              market_features.shape[1]))
            market_features = np.vstack([padding, market_features])
        
        # 포트폴리오 정보
        current_price = self._get_current_price()
        portfolio_features = np.array([
            self.balance / self.config.initial_balance,  # 정규화된 잔고
            self.position,  # 보유 수량
            (self.balance + self.position * current_price) / self.config.initial_balance,  # 총 가치
            self.position / (self.position + 1e-8) if current_price > 0 else 0,  # 포지션 비율
        ])
        
        # 관측값 결합
        observation = np.concatenate([
            np.array(market_features).flatten(),
            portfolio_features
        ])
        
        return observation.astype(np.float32)
    
    def _apply_action_mask(self, action: int) -> int:
        """액션 마스킹 적용"""
        current_price = self._get_current_price()
        
        # 판매 액션인데 보유 수량이 없는 경우
        if action == ActionSpace.SELL and self.position <= 0:
            return ActionSpace.HOLD
        
        # 구매 액션인데 잔고가 부족한 경우
        if action == ActionSpace.BUY and self.balance < current_price * (1 + self.config.transaction_fee):
            return ActionSpace.HOLD
        
        return action
    
    def _execute_action(self, action: int) -> float:
        """액션 실행 및 보상 계산"""
        current_price = self._get_current_price()
        prev_total_value = self.total_value
        
        if action == ActionSpace.BUY:
            # 매수: 전체 잔고로 구매
            if self.balance > 0:
                cost = self.balance
                fee = cost * self.config.transaction_fee
                coin_amount = (cost - fee) / current_price
                
                self.position += coin_amount
                self.balance = 0
        
        elif action == ActionSpace.SELL:
            # 매도: 전체 포지션 판매
            if self.position > 0:
                revenue = self.position * current_price
                fee = revenue * self.config.transaction_fee
                
                self.balance += (revenue - fee)
                self.position = 0
        
        # 현재 총 가치 계산
        self.total_value = self.balance + self.position * current_price
        
        # 보상 계산 (수익률 기반)
        reward = (self.total_value - prev_total_value) / prev_total_value
        
        return reward
    
    def _get_current_price(self) -> float:
        """현재 가격 반환"""
        if self.current_step < len(self.data):
            price_val = self.data.iloc[self.current_step]['close']
            return float(price_val.iloc[0] if hasattr(price_val, 'iloc') else price_val)
        else:
            price_val = self.data.iloc[-1]['close']
            return float(price_val.iloc[0] if hasattr(price_val, 'iloc') else price_val)
    
    def get_action_mask(self) -> np.ndarray:
        """유효한 액션 마스크 반환"""
        mask = np.ones(ActionSpace.get_num_actions(), dtype=bool)
        
        current_price = self._get_current_price()
        
        # 판매 불가능한 경우
        if self.position <= 0:
            mask[ActionSpace.SELL] = False
        
        # 구매 불가능한 경우
        if self.balance < current_price * (1 + self.config.transaction_fee):
            mask[ActionSpace.BUY] = False
        
        return mask


if __name__ == "__main__":
    # 기본 테스트
    config = TradingConfig()
    
    print("=== 강화학습 트레이딩 환경 테스트 ===")
    
    try:
        # 환경 생성
        env = TradingEnvironment(config)
        print(f"환경 생성 완료")
        print(f"관측 공간: {env.observation_space}")
        print(f"액션 공간: {env.action_space}")
        
        # 환경 리셋
        obs, info = env.reset()
        print(f"초기 관측값 크기: {obs.shape}")
        
        # 몇 스텝 실행
        for step in range(5):
            # 랜덤 액션 (마스킹 적용)
            action_mask = env.get_action_mask()
            valid_actions = np.where(action_mask)[0]
            action = np.random.choice(valid_actions)
            
            obs, reward, terminated, truncated, info = env.step(action)
            
            print(f"Step {step}: Action={ActionSpace.get_action_names()[action]}, "
                  f"Reward={reward:.4f}, Balance={info['balance']:.2f}, "
                  f"Position={info['position']:.6f}")
            
            if terminated or truncated:
                break
        
        print("환경 테스트 완료!")
        
    except Exception as e:
        print(f"테스트 오류: {e}")
        import traceback
        traceback.print_exc()
