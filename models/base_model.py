"""
Base Model Configuration and Common Components

모든 모델의 기본 설정과 공통 구성 요소를 정의합니다.
"""

import torch
import torch.nn as nn
from typing import Optional, List
from dataclasses import dataclass


@dataclass
class ModelConfig:
    """모델 설정 클래스
    
    모든 신경망 모델에서 사용되는 공통 설정을 정의합니다.
    """
    model_type: str = "dqn"  # "dqn", "lstm", "gru", "transformer", "ensemble"
    hidden_size: int = 256
    num_layers: int = 3
    dropout: float = 0.3
    
    # LSTM/GRU 설정
    sequence_length: int = 60
    rnn_layers: int = 2
    bidirectional: bool = False
    
    # Transformer 설정
    d_model: int = 256
    nhead: int = 8
    num_encoder_layers: int = 6
    dim_feedforward: int = 512
    
    # Ensemble 설정
    ensemble_models: Optional[List[str]] = None
    
    def __post_init__(self):
        """설정 후 처리"""
        if self.ensemble_models is None:
            self.ensemble_models = ["dqn", "lstm", "transformer"]


# 모델 설정 프리셋
PRESET_CONFIGS = {
    "small_dqn": ModelConfig(
        model_type="dqn",
        hidden_size=128,
        num_layers=2,
        dropout=0.2
    ),
    
    "medium_dqn": ModelConfig(
        model_type="dqn",
        hidden_size=256,
        num_layers=3,
        dropout=0.3
    ),
    
    "large_dqn": ModelConfig(
        model_type="dqn",
        hidden_size=512,
        num_layers=4,
        dropout=0.3
    ),
    
    "small_lstm": ModelConfig(
        model_type="lstm",
        hidden_size=128,
        sequence_length=30,
        rnn_layers=1,
        dropout=0.2
    ),
    
    "medium_lstm": ModelConfig(
        model_type="lstm",
        hidden_size=256,
        sequence_length=30,
        rnn_layers=2,
        dropout=0.3
    ),
    
    "large_lstm": ModelConfig(
        model_type="lstm",
        hidden_size=512,
        sequence_length=60,
        rnn_layers=3,
        bidirectional=True,
        dropout=0.3
    ),
    
    "small_transformer": ModelConfig(
        model_type="transformer",
        d_model=256,
        nhead=4,
        num_encoder_layers=3,
        sequence_length=30,
        dropout=0.1
    ),
    
    "medium_transformer": ModelConfig(
        model_type="transformer",
        d_model=256,
        nhead=8,
        num_encoder_layers=4,
        sequence_length=60,
        dropout=0.1
    ),
    
    "large_transformer": ModelConfig(
        model_type="transformer",
        d_model=512,
        nhead=8,
        num_encoder_layers=6,
        sequence_length=60,
        dropout=0.1
    ),
    
    "ensemble_small": ModelConfig(
        model_type="ensemble",
        hidden_size=128,
        ensemble_models=["dqn", "lstm"],
        dropout=0.3
    ),
    
    "ensemble_medium": ModelConfig(
        model_type="ensemble",
        hidden_size=256,
        ensemble_models=["dqn", "lstm", "transformer"],
        dropout=0.3
    ),
    
    "ensemble_all": ModelConfig(
        model_type="ensemble",
        hidden_size=256,
        ensemble_models=["dqn", "lstm", "transformer"],
        dropout=0.3
    )
}