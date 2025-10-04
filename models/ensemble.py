"""
Ensemble Model

여러 개의 서로 다른 모델을 결합하여 더 강력한 예측 성능을 제공합니다.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any

from .base_model import ModelConfig
from .dqn import DQNModel
from .lstm import LSTMModel
from .transformer import TransformerModel


class EnsembleModel(nn.Module):
    """앙상블 모델
    
    여러 개의 서로 다른 신경망 모델을 결합하여 사용합니다.
    - 개별 모델들의 예측을 가중 평균
    - 메타 학습자를 통한 출력 조합
    - 학습 가능한 앙상블 가중치
    
    Args:
        state_size (int): 상태 공간 크기
        action_size (int): 액션 공간 크기
        config (ModelConfig): 모델 설정
    """
    
    def __init__(self, state_size: int, action_size: int, config: ModelConfig):
        super().__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.config = config
        
        # 앙상블 모델 리스트 확인
        if config.ensemble_models is None:
            config.ensemble_models = ["dqn", "lstm", "transformer"]
        
        self.model_names = config.ensemble_models
        
        # 개별 모델들 초기화
        self.models = nn.ModuleDict()
        
        for model_type in config.ensemble_models:
            # 각 모델별 설정 생성
            model_config = ModelConfig(
                model_type=model_type,
                hidden_size=config.hidden_size,
                num_layers=config.num_layers,
                dropout=config.dropout,
                sequence_length=config.sequence_length,
                d_model=config.d_model,
                nhead=config.nhead,
                num_encoder_layers=config.num_encoder_layers,
                rnn_layers=config.rnn_layers,
                bidirectional=config.bidirectional,
                dim_feedforward=config.dim_feedforward
            )
            
            # 모델 타입에 따라 생성
            if model_type == "dqn":
                self.models[model_type] = DQNModel(state_size, action_size, model_config)
            elif model_type == "lstm":
                self.models[model_type] = LSTMModel(state_size, action_size, model_config)
            elif model_type == "transformer":
                self.models[model_type] = TransformerModel(state_size, action_size, model_config)
            else:
                raise ValueError(f"지원되지 않는 모델 타입: {model_type}")
        
        # 앙상블 가중치 (학습 가능한 파라미터)
        self.ensemble_weights = nn.Parameter(
            torch.ones(len(config.ensemble_models)) / len(config.ensemble_models)
        )
        
        # 메타 학습자 (개별 모델 출력들을 조합)
        self.meta_learner = nn.Sequential(
            nn.Linear(action_size * len(config.ensemble_models), config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_size // 2, action_size)
        )
        
        # 최종 결합 가중치
        self.combination_weight = nn.Parameter(torch.tensor(0.7))  # 가중 평균 vs 메타 학습자
    
    def forward(self, x: torch.Tensor, action_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """순전파
        
        Args:
            x (torch.Tensor): 입력 텐서
            action_mask (torch.Tensor, optional): 유효한 액션 마스크
            
        Returns:
            torch.Tensor: 앙상블 Q-values
        """
        # 각 모델의 출력 계산
        model_outputs = []
        
        for model_name in self.model_names:
            model = self.models[model_name]
            try:
                output = model(x, action_mask)
                model_outputs.append(output)
            except Exception as e:
                # 개별 모델에서 오류 발생 시 0으로 채움
                print(f"Warning: {model_name} 모델에서 오류 발생: {e}")
                zero_output = torch.zeros(x.size(0), self.action_size, device=x.device)
                model_outputs.append(zero_output)
        
        # 앙상블 가중치 정규화
        weights = F.softmax(self.ensemble_weights, dim=0)
        
        # 가중 평균 계산
        weighted_sum = torch.zeros_like(model_outputs[0])
        for i, output in enumerate(model_outputs):
            weighted_sum += weights[i] * output
        
        # 메타 학습자를 통한 출력 조합
        concatenated = torch.cat(model_outputs, dim=-1)
        meta_output = self.meta_learner(concatenated)
        
        # 가중 평균과 메타 출력 결합
        combination_weight = torch.sigmoid(self.combination_weight)
        final_output = combination_weight * weighted_sum + (1 - combination_weight) * meta_output
        
        return final_output
    
    def get_individual_predictions(self, x: torch.Tensor, 
                                 action_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """개별 모델들의 예측 결과 반환
        
        Args:
            x (torch.Tensor): 입력 텐서
            action_mask (torch.Tensor, optional): 유효한 액션 마스크
            
        Returns:
            Dict[str, torch.Tensor]: 모델별 예측 결과
        """
        predictions = {}
        
        with torch.no_grad():
            for model_name in self.model_names:
                model = self.models[model_name]
                try:
                    output = model(x, action_mask)
                    predictions[model_name] = output
                except Exception as e:
                    print(f"Warning: {model_name} 모델에서 오류 발생: {e}")
                    predictions[model_name] = torch.zeros(x.size(0), self.action_size, device=x.device)
        
        return predictions
    
    def get_ensemble_weights(self) -> Dict[str, float]:
        """현재 앙상블 가중치 반환
        
        Returns:
            Dict[str, float]: 모델별 가중치
        """
        weights = F.softmax(self.ensemble_weights, dim=0)
        return {name: weight.item() for name, weight in zip(self.model_names, weights)}
    
    def set_model_freezing(self, model_name: str, freeze: bool = True):
        """특정 모델의 파라미터 고정/해제
        
        Args:
            model_name (str): 모델 이름
            freeze (bool): True면 고정, False면 해제
        """
        if model_name in self.models:
            for param in self.models[model_name].parameters():
                param.requires_grad = not freeze
            print(f"{model_name} 모델 파라미터 {'고정' if freeze else '해제'}됨")
        else:
            print(f"Warning: {model_name} 모델을 찾을 수 없습니다.")
    
    def get_model_complexity(self) -> Dict[str, int]:
        """각 모델의 파라미터 개수 반환
        
        Returns:
            Dict[str, int]: 모델별 파라미터 개수
        """
        complexity = {}
        
        for model_name in self.model_names:
            model = self.models[model_name]
            param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
            complexity[model_name] = param_count
        
        # 메타 학습자 파라미터
        meta_params = sum(p.numel() for p in self.meta_learner.parameters() if p.requires_grad)
        complexity['meta_learner'] = meta_params
        
        # 앙상블 가중치
        complexity['ensemble_weights'] = self.ensemble_weights.numel()
        
        # 전체 파라미터
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        complexity['total'] = total_params
        
        return complexity