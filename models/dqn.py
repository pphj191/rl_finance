"""
Deep Q-Network (DQN) Model

Deep Q-Network 구현을 제공합니다.
Dueling DQN 아키텍처를 사용하여 가치 함수와 어드밴티지 함수를 분리합니다.
"""

import torch
import torch.nn as nn
from typing import Optional

from .base_model import ModelConfig


class DQNModel(nn.Module):
    """Deep Q-Network 모델
    
    Dueling DQN 아키텍처를 구현합니다:
    - 특성 추출 레이어
    - 가치 함수 스트림 V(s)
    - 어드밴티지 함수 스트림 A(s,a)
    - Q(s,a) = V(s) + A(s,a) - mean(A(s,a))
    
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
        
        # 특성 추출 레이어
        layers = []
        input_size = state_size
        
        for i in range(config.num_layers):
            layers.extend([
                nn.Linear(input_size, config.hidden_size),
                nn.ReLU(),
                nn.Dropout(config.dropout)
            ])
            input_size = config.hidden_size
        
        # 마지막 드롭아웃 제거
        if layers:
            layers.pop()
        
        self.feature_extractor = nn.Sequential(*layers)
        
        # 가치 함수 스트림 (V(s))
        self.value_stream = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_size // 2, 1)
        )
        
        # 어드밴티지 함수 스트림 (A(s,a))
        self.advantage_stream = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_size // 2, action_size)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """가중치 초기화 (Xavier Uniform)"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor, action_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """순전파
        
        Args:
            x (torch.Tensor): 입력 상태 텐서 [batch_size, state_size]
            action_mask (torch.Tensor, optional): 유효한 액션 마스크 [batch_size, action_size]
            
        Returns:
            torch.Tensor: Q-values [batch_size, action_size]
        """
        # 배치 차원이 없으면 추가
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        # 특성 추출
        features = self.feature_extractor(x)
        
        # Dueling DQN: V(s) + A(s,a) - mean(A(s,a))
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        
        # Q-값 계산 (Dueling DQN 공식)
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        
        # 액션 마스킹 적용
        if action_mask is not None:
            if action_mask.dim() == 1:
                action_mask = action_mask.unsqueeze(0)
            # 불가능한 액션에 매우 낮은 값 할당
            q_values = q_values.masked_fill(~action_mask.bool(), -1e9)
        
        return q_values
    
    def get_action(self, state: torch.Tensor, epsilon: float = 0.0, 
                   action_mask: Optional[torch.Tensor] = None) -> int:
        """ε-greedy 액션 선택
        
        Args:
            state (torch.Tensor): 현재 상태
            epsilon (float): 탐험 확률
            action_mask (torch.Tensor, optional): 유효한 액션 마스크
            
        Returns:
            int: 선택된 액션
        """
        if torch.rand(1) < epsilon:
            # 무작위 액션 선택 (마스킹 고려)
            if action_mask is not None:
                valid_actions = torch.where(action_mask.bool())[0]
                if len(valid_actions) > 0:
                    return int(valid_actions[torch.randint(len(valid_actions), (1,))].item())
            return int(torch.randint(self.action_size, (1,)).item())
        else:
            # 그리디 액션 선택
            with torch.no_grad():
                q_values = self.forward(state, action_mask)
                return int(q_values.argmax().item())
    
    def get_q_values(self, state: torch.Tensor, 
                     action_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Q-값 반환 (추론 모드)
        
        Args:
            state (torch.Tensor): 현재 상태
            action_mask (torch.Tensor, optional): 유효한 액션 마스크
            
        Returns:
            torch.Tensor: Q-값들
        """
        with torch.no_grad():
            return self.forward(state, action_mask)