"""
강화학습 모델 아키텍처 모듈

다양한 신경망 모델 구조를 제공합니다:
- DQN: 기본 Deep Q-Network
- LSTM/GRU: 순환 신경망 기반 모델
- Transformer: Self-Attention 기반 모델 
- Ensemble: 여러 모델을 결합한 앙상블

TODO:
- ✅ DQN 기본 모델
- ✅ LSTM/GRU 기반 모델  
- ✅ Transformer 기반 모델
- ✅ 앙상블 모델
- ✅ 모델 팩토리 패턴
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, Any, List
import math
from dataclasses import dataclass


@dataclass
class ModelConfig:
    """모델 설정"""
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
        if self.ensemble_models is None:
            self.ensemble_models = ["dqn", "lstm", "transformer"]


class DQNModel(nn.Module):
    """기본 Deep Q-Network 모델"""
    
    def __init__(self, state_size: int, action_size: int, config: ModelConfig):
        super().__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.config = config
        
        # 특성 추출 레이어
        self.feature_extractor = nn.Sequential(
            nn.Linear(state_size, config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU()
        )
        
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
        """가중치 초기화"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor, action_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """순전파"""
        # 배치 차원이 없으면 추가
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        # 특성 추출
        features = self.feature_extractor(x)
        
        # Dueling DQN: V(s) + A(s,a) - mean(A(s,a))
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        
        # 액션 마스킹 적용
        if action_mask is not None:
            if action_mask.dim() == 1:
                action_mask = action_mask.unsqueeze(0)
            # 불가능한 액션에 매우 낮은 값 할당
            q_values = q_values.masked_fill(~action_mask.bool(), -1e9)
        
        return q_values


class LSTMModel(nn.Module):
    """LSTM 기반 시계열 모델"""
    
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
        
        # 출력 레이어
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
        """순전파"""
        batch_size = x.size(0)
        
        # 시퀀스 형태로 변환 (batch_size, sequence_length, features)
        if x.dim() == 2:
            # 단일 타임스텝인 경우 시퀀스로 확장
            x = x.unsqueeze(1).repeat(1, self.sequence_length, 1)
        elif x.dim() == 3 and x.size(1) != self.sequence_length:
            # 시퀀스 길이 조정
            if x.size(1) < self.sequence_length:
                # 패딩
                padding = torch.zeros(batch_size, self.sequence_length - x.size(1), x.size(2), 
                                    device=x.device, dtype=x.dtype)
                x = torch.cat([padding, x], dim=1)
            else:
                # 트런케이션
                x = x[:, -self.sequence_length:, :]
        
        # 입력 투영
        x = self.input_projection(x)
        
        # LSTM 처리
        lstm_out, _ = self.lstm(x)
        
        # 어텐션 적용
        attended_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # 마지막 타임스텝 출력 사용
        final_output = attended_out[:, -1, :]
        
        # Dueling DQN
        value = self.value_head(final_output)
        advantage = self.advantage_head(final_output)
        
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        
        # 액션 마스킹
        if action_mask is not None:
            if action_mask.dim() == 1:
                action_mask = action_mask.unsqueeze(0)
            q_values = q_values.masked_fill(~action_mask.bool(), -1e9)
        
        return q_values


class PositionalEncoding(nn.Module):
    """위치 인코딩"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pe = self.get_buffer('pe')[:x.size(0), :].to(x.device)
        return x + pe


class TransformerModel(nn.Module):
    """Transformer 기반 모델"""
    
    def __init__(self, state_size: int, action_size: int, config: ModelConfig):
        super().__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.config = config
        self.sequence_length = config.sequence_length
        
        # 입력 투영
        self.input_projection = nn.Linear(state_size, config.d_model)
        
        # 위치 인코딩
        self.pos_encoder = PositionalEncoding(config.d_model, config.sequence_length)
        
        # Transformer 인코더
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation='gelu',
            batch_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=config.num_encoder_layers
        )
        
        # 글로벌 풀링
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # 출력 헤드
        self.value_head = nn.Sequential(
            nn.Linear(config.d_model, config.hidden_size),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, 1)
        )
        
        self.advantage_head = nn.Sequential(
            nn.Linear(config.d_model, config.hidden_size),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, action_size)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """가중치 초기화"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x: torch.Tensor, action_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """순전파"""
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
        
        # 입력 투영 및 위치 인코딩
        x = self.input_projection(x)
        x = self.pos_encoder(x.transpose(0, 1)).transpose(0, 1)
        
        # Transformer 인코더
        encoded = self.transformer_encoder(x)
        
        # 글로벌 평균 풀링
        pooled = self.global_pool(encoded.transpose(1, 2)).squeeze(-1)
        
        # Dueling DQN
        value = self.value_head(pooled)
        advantage = self.advantage_head(pooled)
        
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        
        # 액션 마스킹
        if action_mask is not None:
            if action_mask.dim() == 1:
                action_mask = action_mask.unsqueeze(0)
            q_values = q_values.masked_fill(~action_mask.bool(), -1e9)
        
        return q_values


class EnsembleModel(nn.Module):
    """앙상블 모델"""
    
    def __init__(self, state_size: int, action_size: int, config: ModelConfig):
        super().__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.config = config
        
        # 앙상블 모델 리스트 확인
        if config.ensemble_models is None:
            config.ensemble_models = ["dqn", "lstm", "transformer"]
        
        # 개별 모델들 초기화
        self.models = nn.ModuleDict()
        
        for model_type in config.ensemble_models:
            model_config = ModelConfig(
                model_type=model_type,
                hidden_size=config.hidden_size,
                num_layers=config.num_layers,
                dropout=config.dropout,
                sequence_length=config.sequence_length,
                d_model=config.d_model,
                nhead=config.nhead,
                num_encoder_layers=config.num_encoder_layers
            )
            
            if model_type == "dqn":
                self.models[model_type] = DQNModel(state_size, action_size, model_config)
            elif model_type == "lstm":
                self.models[model_type] = LSTMModel(state_size, action_size, model_config)
            elif model_type == "transformer":
                self.models[model_type] = TransformerModel(state_size, action_size, model_config)
        
        # 앙상블 가중치
        self.ensemble_weights = nn.Parameter(
            torch.ones(len(config.ensemble_models)) / len(config.ensemble_models)
        )
        
        # 메타 학습자
        self.meta_learner = nn.Sequential(
            nn.Linear(action_size * len(config.ensemble_models), config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, action_size)
        )
    
    def forward(self, x: torch.Tensor, action_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """순전파"""
        # 각 모델의 출력 계산
        model_outputs = []
        for model_name, model in self.models.items():
            output = model(x, action_mask)
            model_outputs.append(output)
        
        # 가중 평균
        weights = F.softmax(self.ensemble_weights, dim=0)
        weighted_sum = sum(w * output for w, output in zip(weights, model_outputs))
        
        # 메타 학습자를 통한 최종 출력
        concatenated = torch.cat(model_outputs, dim=-1)
        meta_output = self.meta_learner(concatenated)
        
        # 가중 평균과 메타 출력 결합
        final_output = 0.7 * weighted_sum + 0.3 * meta_output
        
        return final_output


def create_model(config: ModelConfig, state_size: int, action_size: int = 3) -> nn.Module:
    """모델 팩토리 함수"""
    
    if config.model_type == "dqn":
        return DQNModel(state_size, action_size, config)
    elif config.model_type in ["lstm", "gru"]:
        return LSTMModel(state_size, action_size, config)
    elif config.model_type == "transformer":
        return TransformerModel(state_size, action_size, config)
    elif config.model_type == "ensemble":
        return EnsembleModel(state_size, action_size, config)
    else:
        raise ValueError(f"지원하지 않는 모델 타입: {config.model_type}")


def count_parameters(model: nn.Module) -> int:
    """모델 파라미터 수 계산"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_summary(model: nn.Module, input_size: tuple) -> str:
    """모델 요약 정보"""
    total_params = count_parameters(model)
    
    summary = f"""
모델 구조 요약:
- 모델 타입: {type(model).__name__}
- 총 파라미터 수: {total_params:,}
- 입력 크기: {input_size}
- 출력 크기: {model.action_size if hasattr(model, 'action_size') else 'N/A'}
"""
    return summary


# 모델 설정 프리셋
PRESET_CONFIGS = {
    "small_dqn": ModelConfig(
        model_type="dqn",
        hidden_size=128,
        num_layers=2,
        dropout=0.2
    ),
    
    "medium_lstm": ModelConfig(
        model_type="lstm",
        hidden_size=256,
        sequence_length=30,
        rnn_layers=2,
        dropout=0.3
    ),
    
    "large_transformer": ModelConfig(
        model_type="transformer",
        d_model=512,
        nhead=8,
        num_encoder_layers=6,
        sequence_length=60,
        dropout=0.1
    ),
    
    "ensemble_all": ModelConfig(
        model_type="ensemble",
        hidden_size=256,
        ensemble_models=["dqn", "lstm", "transformer"],
        dropout=0.3
    )
}


if __name__ == "__main__":
    # 모델 테스트
    config = PRESET_CONFIGS["medium_lstm"]
    state_size = 50
    action_size = 3
    
    model = create_model(config, state_size, action_size)
    print(model_summary(model, (1, state_size)))
    
    # 샘플 입력으로 테스트
    sample_input = torch.randn(1, state_size)
    sample_mask = torch.tensor([[True, True, False]])
    
    with torch.no_grad():
        output = model(sample_input, sample_mask)
        print(f"출력 형태: {output.shape}")
        print(f"Q-values: {output}")
