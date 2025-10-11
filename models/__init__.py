"""
Neural Network Models Package

강화학습을 위한 다양한 신경망 모델 아키텍처를 제공합니다.

모듈 구성:
- base_model.py: 공통 설정 및 기본 클래스
- dqn.py: Deep Q-Network 모델
- lstm.py: LSTM 기반 순환 신경망 모델
- transformer.py: Self-Attention 기반 Transformer 모델
- ensemble.py: 여러 모델을 결합한 앙상블 모델
- factory.py: 모델 생성 팩토리 함수

주요 클래스:
- ModelConfig: 모델 설정 클래스
- DQNModel: DQN 모델
- LSTMModel: LSTM 모델
- TransformerModel: Transformer 모델
- EnsembleModel: 앙상블 모델

사용 예시:
    from models import ModelConfig, create_model
    
    config = ModelConfig(model_type="dqn", hidden_size=256)
    model = create_model(config, state_size=50, action_size=3)
"""

# 공통 import
from .base_model import ModelConfig, PRESET_CONFIGS
from .dqn import DQNModel
from .lstm import LSTMModel
from .transformer import TransformerModel, PositionalEncoding
from .ensemble import EnsembleModel
from .sb3_wrapper import SB3TradingModel, SB3_AVAILABLE, create_sb3_model
from .factory import create_model, count_parameters, model_summary

__all__ = [
    # 설정 클래스
    'ModelConfig',
    'PRESET_CONFIGS',

    # 모델 클래스들
    'DQNModel',
    'LSTMModel',
    'TransformerModel',
    'PositionalEncoding',
    'EnsembleModel',
    'SB3TradingModel',

    # SB3 관련
    'SB3_AVAILABLE',
    'create_sb3_model',

    # 팩토리 함수들
    'create_model',
    'count_parameters',
    'model_summary'
]

__version__ = "1.0.0"