"""
Base Environment Components

트레이딩 환경의 기본 설정과 공통 클래스를 정의합니다.
"""

from typing import List
from dataclasses import dataclass


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
