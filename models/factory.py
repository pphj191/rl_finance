"""
Model Factory and Utility Functions

모델 생성과 관련된 팩토리 함수와 유틸리티 함수들을 제공합니다.
"""

import torch
import torch.nn as nn
from typing import Union

from .base_model import ModelConfig
from .dqn import DQNModel
from .lstm import LSTMModel
from .transformer import TransformerModel
from .ensemble import EnsembleModel
from .sb3_wrapper import SB3TradingModel, SB3_AVAILABLE


def create_model(config: ModelConfig, state_size: int, action_size: int = 3) -> nn.Module:
    """모델 팩토리 함수
    
    설정에 따라 적절한 모델을 생성합니다.
    
    Args:
        config (ModelConfig): 모델 설정
        state_size (int): 상태 공간 크기
        action_size (int): 액션 공간 크기
        
    Returns:
        nn.Module: 생성된 모델
        
    Raises:
        ValueError: 지원하지 않는 모델 타입인 경우
    """
    
    if config.model_type == "dqn":
        return DQNModel(state_size, action_size, config)
    elif config.model_type in ["lstm", "gru"]:
        return LSTMModel(state_size, action_size, config)
    elif config.model_type == "transformer":
        return TransformerModel(state_size, action_size, config)
    elif config.model_type == "ensemble":
        return EnsembleModel(state_size, action_size, config)
    elif config.model_type.startswith("sb3_"):
        # Stable-Baselines3 모델
        if not SB3_AVAILABLE:
            raise ImportError(
                "Stable-Baselines3 not installed. "
                "Install with: pip install stable-baselines3"
            )

        # sb3_ppo, sb3_a2c, sb3_sac, sb3_td3, sb3_dqn
        algorithm = config.model_type.replace("sb3_", "").upper()

        return SB3TradingModel(
            state_size=state_size,
            action_size=action_size,
            algorithm=algorithm,
            learning_rate=config.learning_rate,
            **getattr(config, 'sb3_params', {})
        )
    else:
        supported_types = ["dqn", "lstm", "gru", "transformer", "ensemble", "sb3_ppo", "sb3_a2c", "sb3_sac", "sb3_td3", "sb3_dqn"]
        raise ValueError(f"지원하지 않는 모델 타입: {config.model_type}. "
                        f"지원되는 타입: {supported_types}")


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """모델 파라미터 수 계산
    
    Args:
        model (nn.Module): 대상 모델
        trainable_only (bool): True면 학습 가능한 파라미터만 계산
        
    Returns:
        int: 파라미터 개수
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


def get_model_size_mb(model: nn.Module) -> float:
    """모델 크기를 MB 단위로 계산
    
    Args:
        model (nn.Module): 대상 모델
        
    Returns:
        float: 모델 크기 (MB)
    """
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    total_size = param_size + buffer_size
    return total_size / (1024 ** 2)  # Convert to MB


def model_summary(model: nn.Module, input_size: tuple, device: str = "cpu") -> str:
    """모델 요약 정보 생성
    
    Args:
        model (nn.Module): 대상 모델
        input_size (tuple): 입력 크기 (배치 차원 제외)
        device (str): 디바이스 ("cpu" 또는 "cuda")
        
    Returns:
        str: 모델 요약 정보
    """
    # 모델을 지정된 디바이스로 이동
    model = model.to(device)
    
    # 기본 정보
    total_params = count_parameters(model, trainable_only=True)
    total_params_all = count_parameters(model, trainable_only=False)
    model_size = get_model_size_mb(model)
    
    # 모델 타입별 특수 정보
    model_type = type(model).__name__
    special_info = ""
    
    if hasattr(model, 'config'):
        config = model.config
        special_info += f"- 모델 설정: {config.model_type}\n"
        special_info += f"- 은닉 크기: {config.hidden_size}\n"
        special_info += f"- 레이어 수: {config.num_layers}\n"
        special_info += f"- 드롭아웃: {config.dropout}\n"
        
        if config.model_type in ["lstm", "gru"]:
            special_info += f"- 시퀀스 길이: {config.sequence_length}\n"
            special_info += f"- RNN 레이어: {config.rnn_layers}\n"
            special_info += f"- 양방향: {config.bidirectional}\n"
        elif config.model_type == "transformer":
            special_info += f"- 모델 차원: {config.d_model}\n"
            special_info += f"- 어텐션 헤드: {config.nhead}\n"
            special_info += f"- 인코더 레이어: {config.num_encoder_layers}\n"
        elif config.model_type == "ensemble":
            special_info += f"- 앙상블 모델들: {config.ensemble_models}\n"
    
    # 샘플 입력으로 출력 크기 확인
    try:
        sample_input = torch.randn(1, *input_size).to(device)
        model.eval()
        with torch.no_grad():
            sample_output = model(sample_input)
            output_shape = sample_output.shape
    except Exception as e:
        output_shape = f"계산 불가 ({str(e)})"
    
    summary = f"""
╔══════════════════════════════════════════════════════════════════════════════════╗
║                                  모델 요약                                        ║
╠══════════════════════════════════════════════════════════════════════════════════╣
║ 모델 정보:                                                                        ║
║ - 모델 타입: {model_type:<60} ║
║ - 입력 크기: {str(input_size):<60} ║
║ - 출력 크기: {str(output_shape):<60} ║
║ - 디바이스: {device:<63} ║
║                                                                                   ║
║ 파라미터 정보:                                                                     ║
║ - 학습 가능한 파라미터: {total_params:,} ║
║ - 전체 파라미터: {total_params_all:,} ║
║ - 모델 크기: {model_size:.2f} MB ║
║                                                                                   ║
║ 상세 설정:                                                                        ║
{special_info}║                                                                                   ║
╚══════════════════════════════════════════════════════════════════════════════════╝
"""
    
    return summary


def compare_models(models: dict, input_size: tuple, device: str = "cpu") -> str:
    """여러 모델 비교
    
    Args:
        models (dict): {name: model} 형태의 모델 딕셔너리
        input_size (tuple): 입력 크기
        device (str): 디바이스
        
    Returns:
        str: 모델 비교 표
    """
    comparison = "╔" + "═" * 100 + "╗\n"
    comparison += "║" + " " * 40 + "모델 비교" + " " * 40 + "║\n"
    comparison += "╠" + "═" * 100 + "╣\n"
    comparison += f"║ {'모델명':<20} │ {'타입':<15} │ {'파라미터':<15} │ {'크기(MB)':<10} │ {'출력 크기':<15} ║\n"
    comparison += "╠" + "═" * 100 + "╣\n"
    
    for name, model in models.items():
        model_type = type(model).__name__
        params = count_parameters(model)
        size_mb = get_model_size_mb(model)
        
        try:
            sample_input = torch.randn(1, *input_size).to(device)
            model.eval()
            with torch.no_grad():
                output = model(sample_input)
                output_shape = str(output.shape)
        except:
            output_shape = "계산 불가"
        
        comparison += f"║ {name:<20} │ {model_type:<15} │ {params:,:<15} │ {size_mb:<10.2f} │ {output_shape:<15} ║\n"
    
    comparison += "╚" + "═" * 100 + "╝\n"
    return comparison


def save_model_checkpoint(model: nn.Module, filepath: str, **kwargs):
    """모델 체크포인트 저장
    
    Args:
        model (nn.Module): 저장할 모델
        filepath (str): 저장 경로
        **kwargs: 추가 저장 정보
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'model_type': type(model).__name__,
        **kwargs
    }
    
    if hasattr(model, 'config'):
        checkpoint['config'] = model.config
    
    torch.save(checkpoint, filepath)
    print(f"모델 체크포인트 저장 완료: {filepath}")


def load_model_checkpoint(filepath: str, model: nn.Module = None) -> dict:
    """모델 체크포인트 로드
    
    Args:
        filepath (str): 체크포인트 경로
        model (nn.Module, optional): 가중치를 로드할 모델
        
    Returns:
        dict: 체크포인트 딕셔너리
    """
    checkpoint = torch.load(filepath, map_location='cpu')
    
    if model is not None:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"모델 가중치 로드 완료: {filepath}")
    
    return checkpoint