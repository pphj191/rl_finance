"""
Reinforcement Learning Trading Environment

강화학습 기반 트레이딩 환경을 제공하는 모듈입니다.
"""

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Tuple, Optional

# 로컬 모듈 import
from .base_env import TradingConfig, ActionSpace
from .market_data import UpbitDataCollector, DataNormalizer


class TradingEnvironment(gym.Env):
    """강화학습 트레이딩 환경"""

    def __init__(
        self,
        config: TradingConfig,
        market: str = "KRW-BTC",
        data: Optional[pd.DataFrame] = None,
        db_path: Optional[str] = None,
        mode: str = "offline",
        cache_enabled: bool = True
    ):
        """
        Args:
            config: 트레이딩 설정
            market: 마켓 코드 (예: KRW-BTC)
            data: 미리 준비된 데이터 (DataFrame)
            db_path: SQLite 데이터베이스 경로
            mode: "offline" (SQLite만) | "realtime" (캐시+계산)
            cache_enabled: 캐시 사용 여부

        Note:
            data가 제공되면 해당 데이터 사용 (최우선)
            db_path가 제공되면 DataPipeline 사용 (캐시 활용)
            둘 다 None이면 Upbit API에서 실시간 데이터 수집
        """
        super().__init__()

        self.config = config
        self.market = market
        self.preloaded_data = data
        self.db_path = db_path
        self.mode = mode
        self.cache_enabled = cache_enabled

        # 데이터 파이프라인 설정
        if db_path is not None:
            # DataPipeline 사용 (SQLite 캐시 활용)
            from .data_storage import MarketDataStorage
            from .data_pipeline import DataPipeline

            storage = MarketDataStorage(db_path)
            self.pipeline = DataPipeline(
                storage=storage,
                mode=mode,
                cache_enabled=cache_enabled,
                normalization_method=config.normalization_method,
                include_ssl=True
            )
        else:
            self.pipeline = None
            # 폴백: 기존 방식
            if data is None:
                self.data_collector = UpbitDataCollector(market)
            else:
                self.data_collector = None

            self.normalizer = DataNormalizer(
                method=config.normalization_method,
                window=config.feature_window
            )

        # 상태 및 액션 공간 정의
        self.action_space = spaces.Discrete(ActionSpace.get_num_actions())

        # 관측 공간 (나중에 실제 데이터 기반으로 조정)
        self.observation_space = None  # reset에서 설정됨

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
        # 1. 미리 준비된 데이터 사용 (최우선)
        if self.preloaded_data is not None:
            return self.preloaded_data.copy()

        # 2. DataPipeline 사용 (SQLite 캐시 활용)
        if self.pipeline is not None:
            from datetime import datetime, timedelta

            # 기간 설정 (최근 7일)
            end_time = datetime.now()
            start_time = end_time - timedelta(days=7)

            # 파이프라인으로 처리된 데이터 로드
            processed_data = self.pipeline.process_data(
                market=self.market,
                start_time=start_time,
                end_time=end_time
            )

            return processed_data

        # 3. 폴백: Upbit API에서 실시간 수집
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


# 테스트용 스크립트
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
