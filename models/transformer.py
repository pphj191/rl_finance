"""
Transformer-based Model

Self-Attention 기반 Transformer 모델을 구현합니다.
위치 인코딩과 멀티헤드 어텐션을 사용합니다.
"""

import torch
import torch.nn as nn
import math
from typing import Optional

from .base_model import ModelConfig


class PositionalEncoding(nn.Module):
    """위치 인코딩 모듈
    
    Transformer에서 시퀀스의 위치 정보를 제공합니다.
    Sinusoidal 위치 인코딩을 사용합니다.
    
    Args:
        d_model (int): 모델 차원
        max_len (int): 최대 시퀀스 길이
    """
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        # 위치 인코딩 계산
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        # 버퍼로 등록 (학습되지 않는 파라미터)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """위치 인코딩 적용
        
        Args:
            x (torch.Tensor): 입력 텐서 [seq_len, batch_size, d_model]
            
        Returns:
            torch.Tensor: 위치 인코딩이 적용된 텐서
        """
        seq_len = x.size(0)
        pe = self.get_buffer('pe')[:seq_len, :].to(x.device)
        return x + pe


class TransformerModel(nn.Module):
    """Transformer 기반 모델
    
    Self-Attention 메커니즘을 사용하는 Transformer 인코더 기반 모델입니다.
    - 입력 투영 레이어
    - 위치 인코딩
    - Transformer 인코더 스택
    - 글로벌 풀링
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
        
        # 위치 인코딩
        self.pos_encoder = PositionalEncoding(config.d_model, config.sequence_length)
        
        # Transformer 인코더 레이어
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-LN 구조 사용
        )
        
        # Transformer 인코더 스택
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=config.num_encoder_layers
        )
        
        # 레이어 정규화
        self.layer_norm = nn.LayerNorm(config.d_model)
        
        # 글로벌 풀링 (어텐션 기반)
        self.attention_pool = nn.MultiheadAttention(
            embed_dim=config.d_model,
            num_heads=config.nhead,
            dropout=config.dropout,
            batch_first=True
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.d_model))
        
        # 출력 헤드 (Dueling DQN)
        self.value_head = nn.Sequential(
            nn.Linear(config.d_model, config.hidden_size),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.GELU(),
            nn.Linear(config.hidden_size // 2, 1)
        )
        
        self.advantage_head = nn.Sequential(
            nn.Linear(config.d_model, config.hidden_size),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.GELU(),
            nn.Linear(config.hidden_size // 2, action_size)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """가중치 초기화 (Xavier Uniform)"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
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
        
        # CLS 토큰 추가
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # 위치 인코딩 적용 - 배치 우선 처리
        # [batch_size, seq_len, d_model] 형태에서 직접 처리
        seq_len = x.size(1)
        d_model = x.size(2)  # 임베딩 차원
        
        # 동적 위치 인코딩 생성
        pe = torch.zeros(seq_len, d_model, device=x.device)
        position = torch.arange(0, seq_len, dtype=torch.float, device=x.device).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2, device=x.device).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        
        # 배치 차원 추가 및 위치 인코딩 적용
        pe = pe.unsqueeze(0).expand(batch_size, -1, -1)
        x = x + pe
        
        # Transformer 인코더 처리
        encoded = self.transformer_encoder(x)
        
        # 레이어 정규화
        encoded = self.layer_norm(encoded)
        
        # CLS 토큰 사용 (첫 번째 토큰)
        cls_output = encoded[:, 0, :]
        
        # 어텐션 풀링 (선택사항)
        # attended_output, _ = self.attention_pool(cls_output.unsqueeze(1), encoded, encoded)
        # final_output = attended_output.squeeze(1)
        final_output = cls_output
        
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
    
    def get_attention_weights(self, x: torch.Tensor, layer_idx: int = -1) -> torch.Tensor:
        """어텐션 가중치 추출
        
        Args:
            x (torch.Tensor): 입력 텐서
            layer_idx (int): 어텐션을 추출할 레이어 인덱스 (-1은 마지막 레이어)
            
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
        
        # CLS 토큰 추가
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # 위치 인코딩
        x = x.transpose(0, 1)
        x = self.pos_encoder(x)
        x = x.transpose(0, 1)
        
        # 특정 레이어까지 처리하여 어텐션 가중치 추출
        # 간단한 구현을 위해 전체 처리 후 마지막 레이어의 어텐션 사용
        with torch.no_grad():
            encoded = self.transformer_encoder(x)
            
        # 실제 어텐션 가중치는 내부적으로 계산되므로
        # 여기서는 CLS 토큰과 다른 토큰들 간의 유사도를 반환
        cls_output = encoded[:, 0:1, :]  # [batch_size, 1, d_model]
        sequence_output = encoded[:, 1:, :]  # [batch_size, seq_len, d_model]
        
        # 코사인 유사도 계산
        similarity = torch.cosine_similarity(
            cls_output.expand(-1, sequence_output.size(1), -1),
            sequence_output,
            dim=-1
        )
        
        return similarity
    
    def get_representations(self, x: torch.Tensor) -> torch.Tensor:
        """중간 표현 추출
        
        Args:
            x (torch.Tensor): 입력 텐서
            
        Returns:
            torch.Tensor: Transformer 인코더 출력
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
        
        # CLS 토큰 추가
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # 위치 인코딩
        x = x.transpose(0, 1)
        x = self.pos_encoder(x)
        x = x.transpose(0, 1)
        
        # Transformer 인코더
        with torch.no_grad():
            encoded = self.transformer_encoder(x)
            encoded = self.layer_norm(encoded)
        
        return encoded