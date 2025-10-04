"""
LSTM-based Sequence Model

LSTM 기반 순환 신경망 모델을 구현합니다.
시계열 데이터를 처리하며 어텐션 메커니즘을 포함합니다.
"""

import torch
import torch.nn as nn
from typing import Optional

from .base_model import ModelConfig


class LSTMModel(nn.Module):
    """LSTM 기반 시계열 모델
    
    시계열 데이터를 처리하는 LSTM 기반 모델입니다.
    - 입력 투영 레이어
    - LSTM 레이어 (단방향/양방향 지원)
    - 멀티헤드 어텐션
    - Dueling DQN 출력 헤드
    
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
        self.sequence_length = config.sequence_length
        
        # 입력 투영 레이어
        self.input_projection = nn.Linear(state_size, config.d_model)
        
        # LSTM 레이어
        self.lstm = nn.LSTM(
            input_size=config.d_model,
            hidden_size=config.hidden_size,
            num_layers=config.rnn_layers,
            batch_first=True,
            dropout=config.dropout if config.rnn_layers > 1 else 0,
            bidirectional=config.bidirectional
        )
        
        # 출력 크기 계산
        lstm_output_size = config.hidden_size * (2 if config.bidirectional else 1)
        
        # 출력 레이어 (Dueling DQN)
        self.value_head = nn.Sequential(
            nn.Linear(lstm_output_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size // 2, 1)
        )
        
        self.advantage_head = nn.Sequential(
            nn.Linear(lstm_output_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size // 2, action_size)
        )
        
        # 어텐션 메커니즘
        self.attention = nn.MultiheadAttention(
            embed_dim=lstm_output_size,
            num_heads=min(8, lstm_output_size // 64),
            dropout=config.dropout,
            batch_first=True
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """가중치 초기화"""
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                nn.init.constant_(param.data, 0)
    
    def forward(self, x: torch.Tensor, action_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """순전파
        
        Args:
            x (torch.Tensor): 입력 텐서 [batch_size, sequence_length, state_size] 또는 [batch_size, state_size]
            action_mask (torch.Tensor, optional): 유효한 액션 마스크 [batch_size, action_size]
            
        Returns:
            torch.Tensor: Q-values [batch_size, action_size]
        """
        batch_size = x.size(0)
        
        # 시퀀스 형태로 변환 (batch_size, sequence_length, features)
        if x.dim() == 2:
            # 단일 타임스텝인 경우 시퀀스로 확장
            x = x.unsqueeze(1).repeat(1, self.sequence_length, 1)
        elif x.dim() == 3 and x.size(1) != self.sequence_length:
            # 시퀀스 길이 조정
            if x.size(1) < self.sequence_length:
                # 패딩 (앞쪽에 0으로 패딩)
                padding = torch.zeros(batch_size, self.sequence_length - x.size(1), x.size(2), 
                                    device=x.device, dtype=x.dtype)
                x = torch.cat([padding, x], dim=1)
            else:
                # 트런케이션 (최근 sequence_length만 사용)
                x = x[:, -self.sequence_length:, :]
        
        # 입력 투영
        x = self.input_projection(x)
        
        # LSTM 처리
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # 어텐션 적용 (self-attention)
        attended_out, attention_weights = self.attention(lstm_out, lstm_out, lstm_out)
        
        # 마지막 타임스텝 출력 사용
        final_output = attended_out[:, -1, :]
        
        # Dueling DQN: V(s) + A(s,a) - mean(A(s,a))
        value = self.value_head(final_output)
        advantage = self.advantage_head(final_output)
        
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        
        # 액션 마스킹 적용
        if action_mask is not None:
            if action_mask.dim() == 1:
                action_mask = action_mask.unsqueeze(0)
            q_values = q_values.masked_fill(~action_mask.bool(), -1e9)
        
        return q_values
    
    def get_hidden_state(self, x: torch.Tensor) -> tuple:
        """LSTM 히든 상태 추출
        
        Args:
            x (torch.Tensor): 입력 시퀀스
            
        Returns:
            tuple: (hidden_state, cell_state)
        """
        batch_size = x.size(0)
        
        # 시퀀스 형태로 변환
        if x.dim() == 2:
            x = x.unsqueeze(1).repeat(1, self.sequence_length, 1)
        elif x.dim() == 3 and x.size(1) != self.sequence_length:
            if x.size(1) < self.sequence_length:
                padding = torch.zeros(batch_size, self.sequence_length - x.size(1), x.size(2), 
                                    device=x.device, dtype=x.dtype)
                x = torch.cat([padding, x], dim=1)
            else:
                x = x[:, -self.sequence_length:, :]
        
        # 입력 투영
        x = self.input_projection(x)
        
        # LSTM 처리
        with torch.no_grad():
            _, (hidden, cell) = self.lstm(x)
        
        return hidden, cell
    
    def get_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """어텐션 가중치 반환
        
        Args:
            x (torch.Tensor): 입력 시퀀스
            
        Returns:
            torch.Tensor: 어텐션 가중치
        """
        batch_size = x.size(0)
        
        # 시퀀스 형태로 변환
        if x.dim() == 2:
            x = x.unsqueeze(1).repeat(1, self.sequence_length, 1)
        elif x.dim() == 3 and x.size(1) != self.sequence_length:
            if x.size(1) < self.sequence_length:
                padding = torch.zeros(batch_size, self.sequence_length - x.size(1), x.size(2), 
                                    device=x.device, dtype=x.dtype)
                x = torch.cat([padding, x], dim=1)
            else:
                x = x[:, -self.sequence_length:, :]
        
        # 입력 투영
        x = self.input_projection(x)
        
        # LSTM 처리
        lstm_out, _ = self.lstm(x)
        
        # 어텐션 가중치 추출
        with torch.no_grad():
            _, attention_weights = self.attention(lstm_out, lstm_out, lstm_out)
        
        return attention_weights