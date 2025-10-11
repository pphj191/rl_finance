"""
Self-Supervised Learning (SSL) Features Module

학습 기반 representation 벡터 추출 및 미래 예측을 위한 SSL 모듈입니다.
SQLite 데이터를 활용하여 모델을 학습하고, 학습된 모델로부터 특성을 추출합니다.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import pickle


@dataclass
class SSLConfig:
    """SSL 모델 설정"""
    # 모델 파라미터
    hidden_dim: int = 128
    num_layers: int = 2
    dropout: float = 0.1

    # 학습 파라미터
    batch_size: int = 32
    learning_rate: float = 1e-3
    num_epochs: int = 100

    # Contrastive Learning
    temperature: float = 0.07
    augmentation_strength: float = 0.1

    # Masked Prediction
    mask_ratio: float = 0.15

    # Future Prediction
    prediction_horizons: List[int] = None  # [1, 5, 15, 30, 60] 분 후 예측

    # 모델 저장 경로
    model_dir: str = "models/ssl"

    def __post_init__(self):
        if self.prediction_horizons is None:
            self.prediction_horizons = [1, 5, 15, 30, 60]


class ContrastiveEncoder(nn.Module):
    """대조 학습 기반 인코더

    시계열 데이터의 representation을 학습합니다.
    유사한 패턴은 가까운 벡터로, 다른 패턴은 먼 벡터로 매핑합니다.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 2):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # LSTM for temporal encoding
        self.lstm = nn.LSTM(
            hidden_dim, hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0
        )

        # Projection head for contrastive learning
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len, input_dim)

        Returns:
            representation: (batch, hidden_dim)
            projection: (batch, hidden_dim // 2)
        """
        # Encode
        encoded = self.encoder(x)  # (batch, seq_len, hidden_dim)

        # LSTM
        _, (h_n, _) = self.lstm(encoded)  # h_n: (num_layers, batch, hidden_dim)
        representation = h_n[-1]  # (batch, hidden_dim)

        # Projection for contrastive loss
        projection = self.projection(representation)

        return representation, projection


class MaskedPredictor(nn.Module):
    """마스킹 예측 모델

    시계열 데이터의 일부를 마스킹하고, 마스킹된 부분을 예측합니다.
    BERT-style masked prediction for time series.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 2):
        super().__init__()

        self.encoder = nn.LSTM(
            input_dim, hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0
        )

        self.decoder = nn.Linear(hidden_dim, input_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len, input_dim) - masked input

        Returns:
            predictions: (batch, seq_len, input_dim)
            hidden: (batch, seq_len, hidden_dim)
        """
        hidden, _ = self.encoder(x)  # (batch, seq_len, hidden_dim)
        predictions = self.decoder(hidden)  # (batch, seq_len, input_dim)

        return predictions, hidden


class TemporalPatternClassifier(nn.Module):
    """시간적 패턴 분류 모델

    시계열 데이터의 패턴을 분류합니다 (상승, 하락, 횡보, 변동성 등).
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128, num_classes: int = 8):
        super().__init__()

        self.encoder = nn.LSTM(
            input_dim, hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_dim)

        Returns:
            logits: (batch, num_classes)
        """
        _, (h_n, _) = self.encoder(x)
        logits = self.classifier(h_n[-1])

        return logits


class FuturePricePredictor(nn.Module):
    """미래 가격 예측 모델

    N분 후의 가격 변화를 예측합니다.
    Multi-horizon prediction (1분, 5분, 15분, 30분, 60분 후).
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128, num_horizons: int = 5):
        super().__init__()

        self.encoder = nn.LSTM(
            input_dim, hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )

        # Multi-task prediction heads
        self.prediction_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1)  # 가격 변화율 예측
            )
            for _ in range(num_horizons)
        ])

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Args:
            x: (batch, seq_len, input_dim)

        Returns:
            predictions: List of (batch, 1) for each horizon
        """
        _, (h_n, _) = self.encoder(x)
        representation = h_n[-1]  # (batch, hidden_dim)

        predictions = [head(representation) for head in self.prediction_heads]

        return predictions


class SSLFeatureExtractor:
    """SSL 기반 특성 추출기

    학습된 SSL 모델들을 사용하여 representation 벡터와 예측값을 추출합니다.
    """

    def __init__(self, config: SSLConfig, device: str = "cpu"):
        self.config = config
        self.device = device

        # 모델 초기화 (학습 후 로드됨)
        self.contrastive_encoder: Optional[ContrastiveEncoder] = None
        self.masked_predictor: Optional[MaskedPredictor] = None
        self.pattern_classifier: Optional[TemporalPatternClassifier] = None
        self.future_predictor: Optional[FuturePricePredictor] = None

        # 학습 여부 플래그
        self.is_trained = {
            'contrastive': False,
            'masked': False,
            'pattern': False,
            'future': False
        }

    def extract_features(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """모든 SSL 특성 추출

        Args:
            df: 시장 데이터 (OHLCV + indicators)

        Returns:
            features: {
                'contrastive_repr': (seq_len, hidden_dim),
                'masked_repr': (seq_len, hidden_dim),
                'pattern_probs': (seq_len, num_classes),
                'future_predictions': (seq_len, num_horizons)
            }
        """
        features = {}

        # 입력 데이터 준비
        x = self._prepare_input(df)

        # Contrastive representation
        if self.is_trained['contrastive']:
            contrastive_repr = self._extract_contrastive_features(x)
            features['contrastive_repr'] = contrastive_repr

        # Masked prediction representation
        if self.is_trained['masked']:
            masked_repr = self._extract_masked_features(x)
            features['masked_repr'] = masked_repr

        # Pattern classification
        if self.is_trained['pattern']:
            pattern_probs = self._classify_patterns(x)
            features['pattern_probs'] = pattern_probs

        # Future predictions
        if self.is_trained['future']:
            future_preds = self._predict_future(x)
            features['future_predictions'] = future_preds

        return features

    def _prepare_input(self, df: pd.DataFrame) -> torch.Tensor:
        """입력 데이터 준비"""
        # 수치형 컬럼만 선택
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        data = df[numeric_columns].values

        # Tensor 변환
        x = torch.FloatTensor(data).unsqueeze(0).to(self.device)  # (1, seq_len, input_dim)

        return x

    def _extract_contrastive_features(self, x: torch.Tensor) -> np.ndarray:
        """Contrastive representation 추출"""
        self.contrastive_encoder.eval()
        with torch.no_grad():
            representation, _ = self.contrastive_encoder(x)

        return representation.cpu().numpy()

    def _extract_masked_features(self, x: torch.Tensor) -> np.ndarray:
        """Masked prediction representation 추출"""
        self.masked_predictor.eval()
        with torch.no_grad():
            _, hidden = self.masked_predictor(x)

        return hidden.squeeze(0).cpu().numpy()

    def _classify_patterns(self, x: torch.Tensor) -> np.ndarray:
        """패턴 분류 확률 추출"""
        self.pattern_classifier.eval()
        with torch.no_grad():
            logits = self.pattern_classifier(x)
            probs = F.softmax(logits, dim=-1)

        return probs.cpu().numpy()

    def _predict_future(self, x: torch.Tensor) -> np.ndarray:
        """미래 가격 예측"""
        self.future_predictor.eval()
        with torch.no_grad():
            predictions = self.future_predictor(x)
            predictions = torch.cat(predictions, dim=-1)  # (batch, num_horizons)

        return predictions.cpu().numpy()

    # ============================================================================
    # 모델 학습 메서드들
    # ============================================================================

    def train_contrastive_model(self, data_loader, db_path: str):
        """대조 학습 모델 학습

        TODO: 구현 필요

        학습 절차:
        1. SQLite에서 시계열 데이터 로드
        2. Data augmentation (노이즈 추가, 시간 왜곡 등)
        3. Positive pair 생성 (같은 데이터의 augmented versions)
        4. Negative pair 생성 (다른 시간대 데이터)
        5. NT-Xent loss 계산 및 최적화

        참고:
        - SimCLR 방식의 contrastive learning
        - Temperature scaling
        - Hard negative mining

        Args:
            data_loader: SQLite 데이터 로더
            db_path: 데이터베이스 경로
        """
        # TODO: SQLite에서 데이터 로드
        # from .data_storage import MarketDataStorage
        # storage = MarketDataStorage(db_path)

        # TODO: 모델 초기화
        # input_dim = ...  # 입력 차원 계산
        # self.contrastive_encoder = ContrastiveEncoder(
        #     input_dim, self.config.hidden_dim, self.config.num_layers
        # ).to(self.device)

        # TODO: 옵티마이저 설정
        # optimizer = torch.optim.Adam(
        #     self.contrastive_encoder.parameters(),
        #     lr=self.config.learning_rate
        # )

        # TODO: 학습 루프
        # for epoch in range(self.config.num_epochs):
        #     for batch in data_loader:
        #         # 1. Data augmentation
        #         x1 = augment(batch)
        #         x2 = augment(batch)
        #
        #         # 2. Forward pass
        #         _, z1 = self.contrastive_encoder(x1)
        #         _, z2 = self.contrastive_encoder(x2)
        #
        #         # 3. NT-Xent loss
        #         loss = nt_xent_loss(z1, z2, temperature=self.config.temperature)
        #
        #         # 4. Backward pass
        #         optimizer.zero_grad()
        #         loss.backward()
        #         optimizer.step()

        # TODO: 모델 저장
        # self._save_model('contrastive')
        # self.is_trained['contrastive'] = True

        raise NotImplementedError("Contrastive learning 학습 로직 구현 필요")

    def train_masked_prediction_model(self, data_loader, db_path: str):
        """마스킹 예측 모델 학습

        TODO: 구현 필요

        학습 절차:
        1. SQLite에서 시계열 데이터 로드
        2. 랜덤하게 일부 timestep을 마스킹 (mask_ratio)
        3. 마스킹된 부분을 예측하도록 학습
        4. MSE loss 계산 및 최적화

        참고:
        - BERT-style masked language modeling for time series
        - 마스킹 전략: random, block, last

        Args:
            data_loader: SQLite 데이터 로더
            db_path: 데이터베이스 경로
        """
        # TODO: SQLite에서 데이터 로드

        # TODO: 모델 초기화
        # input_dim = ...
        # self.masked_predictor = MaskedPredictor(
        #     input_dim, self.config.hidden_dim, self.config.num_layers
        # ).to(self.device)

        # TODO: 옵티마이저 설정

        # TODO: 학습 루프
        # for epoch in range(self.config.num_epochs):
        #     for batch in data_loader:
        #         # 1. 마스킹
        #         masked_batch, mask_indices = apply_mask(
        #             batch, mask_ratio=self.config.mask_ratio
        #         )
        #
        #         # 2. Forward pass
        #         predictions, _ = self.masked_predictor(masked_batch)
        #
        #         # 3. MSE loss (마스킹된 부분만)
        #         loss = F.mse_loss(
        #             predictions[mask_indices],
        #             batch[mask_indices]
        #         )
        #
        #         # 4. Backward pass
        #         optimizer.zero_grad()
        #         loss.backward()
        #         optimizer.step()

        # TODO: 모델 저장
        # self._save_model('masked')
        # self.is_trained['masked'] = True

        raise NotImplementedError("Masked prediction 학습 로직 구현 필요")

    def train_pattern_classifier(self, data_loader, db_path: str):
        """패턴 분류 모델 학습

        TODO: 구현 필요

        학습 절차:
        1. SQLite에서 시계열 데이터 로드
        2. 패턴 레이블 생성 (상승, 하락, 횡보, 변동성 등)
        3. Classification loss 계산 및 최적화

        패턴 클래스:
        0: 강한 상승 (strong uptrend)
        1: 약한 상승 (weak uptrend)
        2: 횡보 (sideways)
        3: 약한 하락 (weak downtrend)
        4: 강한 하락 (strong downtrend)
        5: 높은 변동성 (high volatility)
        6: 낮은 변동성 (low volatility)
        7: 반전 패턴 (reversal pattern)

        Args:
            data_loader: SQLite 데이터 로더
            db_path: 데이터베이스 경로
        """
        # TODO: SQLite에서 데이터 로드

        # TODO: 모델 초기화
        # input_dim = ...
        # self.pattern_classifier = TemporalPatternClassifier(
        #     input_dim, self.config.hidden_dim, num_classes=8
        # ).to(self.device)

        # TODO: 옵티마이저 설정

        # TODO: 학습 루프
        # for epoch in range(self.config.num_epochs):
        #     for batch, labels in data_loader:
        #         # 1. Forward pass
        #         logits = self.pattern_classifier(batch)
        #
        #         # 2. Cross-entropy loss
        #         loss = F.cross_entropy(logits, labels)
        #
        #         # 3. Backward pass
        #         optimizer.zero_grad()
        #         loss.backward()
        #         optimizer.step()

        # TODO: 모델 저장
        # self._save_model('pattern')
        # self.is_trained['pattern'] = True

        raise NotImplementedError("Pattern classification 학습 로직 구현 필요")

    def train_future_predictor(self, data_loader, db_path: str):
        """미래 가격 예측 모델 학습

        TODO: 구현 필요

        학습 절차:
        1. SQLite에서 시계열 데이터 로드
        2. Multi-horizon target 생성 (1분, 5분, 15분, 30분, 60분 후 가격 변화율)
        3. Multi-task learning으로 모든 horizon 동시 학습
        4. MSE loss 계산 및 최적화

        참고:
        - Multi-task learning with uncertainty weighting
        - Horizon-specific loss balancing

        Args:
            data_loader: SQLite 데이터 로더
            db_path: 데이터베이스 경로
        """
        # TODO: SQLite에서 데이터 로드

        # TODO: 모델 초기화
        # input_dim = ...
        # num_horizons = len(self.config.prediction_horizons)
        # self.future_predictor = FuturePricePredictor(
        #     input_dim, self.config.hidden_dim, num_horizons
        # ).to(self.device)

        # TODO: 옵티마이저 설정

        # TODO: 학습 루프
        # for epoch in range(self.config.num_epochs):
        #     for batch, targets in data_loader:
        #         # targets: List of (batch, 1) for each horizon
        #
        #         # 1. Forward pass
        #         predictions = self.future_predictor(batch)
        #
        #         # 2. Multi-task loss
        #         losses = [
        #             F.mse_loss(pred, target)
        #             for pred, target in zip(predictions, targets)
        #         ]
        #         total_loss = sum(losses)
        #
        #         # 3. Backward pass
        #         optimizer.zero_grad()
        #         total_loss.backward()
        #         optimizer.step()

        # TODO: 모델 저장
        # self._save_model('future')
        # self.is_trained['future'] = True

        raise NotImplementedError("Future prediction 학습 로직 구현 필요")

    def train_all_models(self, db_path: str):
        """모든 SSL 모델 학습

        TODO: 구현 필요

        Args:
            db_path: SQLite 데이터베이스 경로
        """
        # TODO: 데이터 로더 생성
        # data_loader = create_data_loader(db_path, self.config)

        # TODO: 각 모델 순차적으로 학습
        # print("Training contrastive model...")
        # self.train_contrastive_model(data_loader, db_path)

        # print("Training masked prediction model...")
        # self.train_masked_prediction_model(data_loader, db_path)

        # print("Training pattern classifier...")
        # self.train_pattern_classifier(data_loader, db_path)

        # print("Training future predictor...")
        # self.train_future_predictor(data_loader, db_path)

        raise NotImplementedError("전체 모델 학습 로직 구현 필요")

    # ============================================================================
    # 모델 저장/로드 메서드들
    # ============================================================================

    def _save_model(self, model_type: str):
        """모델 저장

        Args:
            model_type: 'contrastive', 'masked', 'pattern', 'future'
        """
        model_dir = Path(self.config.model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)

        model_path = model_dir / f"{model_type}_model.pt"

        if model_type == 'contrastive':
            torch.save(self.contrastive_encoder.state_dict(), model_path)
        elif model_type == 'masked':
            torch.save(self.masked_predictor.state_dict(), model_path)
        elif model_type == 'pattern':
            torch.save(self.pattern_classifier.state_dict(), model_path)
        elif model_type == 'future':
            torch.save(self.future_predictor.state_dict(), model_path)

        print(f"Model saved: {model_path}")

    def load_model(self, model_type: str, input_dim: int):
        """모델 로드

        Args:
            model_type: 'contrastive', 'masked', 'pattern', 'future'
            input_dim: 입력 차원
        """
        model_dir = Path(self.config.model_dir)
        model_path = model_dir / f"{model_type}_model.pt"

        if not model_path.exists():
            print(f"Warning: Model not found at {model_path}")
            return

        if model_type == 'contrastive':
            self.contrastive_encoder = ContrastiveEncoder(
                input_dim, self.config.hidden_dim, self.config.num_layers
            ).to(self.device)
            self.contrastive_encoder.load_state_dict(torch.load(model_path, map_location=self.device))
            self.is_trained['contrastive'] = True

        elif model_type == 'masked':
            self.masked_predictor = MaskedPredictor(
                input_dim, self.config.hidden_dim, self.config.num_layers
            ).to(self.device)
            self.masked_predictor.load_state_dict(torch.load(model_path, map_location=self.device))
            self.is_trained['masked'] = True

        elif model_type == 'pattern':
            self.pattern_classifier = TemporalPatternClassifier(
                input_dim, self.config.hidden_dim, num_classes=8
            ).to(self.device)
            self.pattern_classifier.load_state_dict(torch.load(model_path, map_location=self.device))
            self.is_trained['pattern'] = True

        elif model_type == 'future':
            num_horizons = len(self.config.prediction_horizons)
            self.future_predictor = FuturePricePredictor(
                input_dim, self.config.hidden_dim, num_horizons
            ).to(self.device)
            self.future_predictor.load_state_dict(torch.load(model_path, map_location=self.device))
            self.is_trained['future'] = True

        print(f"Model loaded: {model_path}")

    def load_all_models(self, input_dim: int):
        """모든 모델 로드

        Args:
            input_dim: 입력 차원
        """
        for model_type in ['contrastive', 'masked', 'pattern', 'future']:
            try:
                self.load_model(model_type, input_dim)
            except Exception as e:
                print(f"Failed to load {model_type} model: {e}")


# ============================================================================
# 유틸리티 함수들
# ============================================================================

def create_data_loader(db_path: str, config: SSLConfig):
    """SQLite 데이터베이스로부터 데이터 로더 생성

    TODO: 구현 필요

    Args:
        db_path: SQLite 데이터베이스 경로
        config: SSL 설정

    Returns:
        data_loader: PyTorch 데이터 로더
    """
    # TODO: SQLite에서 데이터 읽기
    # from .data_storage import MarketDataStorage
    # storage = MarketDataStorage(db_path)

    # TODO: Dataset 클래스 정의
    # class MarketDataset(torch.utils.data.Dataset):
    #     def __init__(self, data):
    #         self.data = data
    #
    #     def __len__(self):
    #         return len(self.data)
    #
    #     def __getitem__(self, idx):
    #         return self.data[idx]

    # TODO: DataLoader 생성
    # dataset = MarketDataset(data)
    # data_loader = torch.utils.data.DataLoader(
    #     dataset,
    #     batch_size=config.batch_size,
    #     shuffle=True,
    #     num_workers=4
    # )

    raise NotImplementedError("데이터 로더 생성 로직 구현 필요")


def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """NT-Xent (Normalized Temperature-scaled Cross Entropy) Loss

    Contrastive learning의 표준 손실 함수입니다.

    TODO: 구현 필요

    Args:
        z1: (batch, dim) - first projection
        z2: (batch, dim) - second projection
        temperature: temperature parameter

    Returns:
        loss: scalar
    """
    # TODO: 구현
    # batch_size = z1.size(0)
    #
    # # Normalize
    # z1 = F.normalize(z1, dim=1)
    # z2 = F.normalize(z2, dim=1)
    #
    # # Similarity matrix
    # sim_matrix = torch.mm(z1, z1.t()) / temperature
    #
    # # Positive pairs
    # pos_sim = torch.mm(z1, z2.t()) / temperature
    #
    # # NT-Xent loss
    # ...

    raise NotImplementedError("NT-Xent loss 구현 필요")


def apply_mask(x: torch.Tensor, mask_ratio: float = 0.15) -> Tuple[torch.Tensor, torch.Tensor]:
    """시계열 데이터에 마스킹 적용

    TODO: 구현 필요

    Args:
        x: (batch, seq_len, input_dim)
        mask_ratio: 마스킹 비율

    Returns:
        masked_x: (batch, seq_len, input_dim)
        mask_indices: (num_masked,)
    """
    # TODO: 구현
    # batch_size, seq_len, input_dim = x.shape
    # num_masked = int(seq_len * mask_ratio)
    #
    # # Random masking
    # mask_indices = torch.randperm(seq_len)[:num_masked]
    #
    # masked_x = x.clone()
    # masked_x[:, mask_indices, :] = 0  # or use mask token

    raise NotImplementedError("Masking 로직 구현 필요")


# ============================================================================
# 테스트 코드
# ============================================================================

if __name__ == "__main__":
    print("=== SSL Feature Extractor 테스트 ===")

    # 설정
    config = SSLConfig()
    extractor = SSLFeatureExtractor(config)

    # 테스트 데이터
    test_data = pd.DataFrame({
        'close': np.random.randn(100),
        'volume': np.random.randn(100),
        'rsi': np.random.randn(100),
    })

    try:
        # 특성 추출 (학습되지 않은 상태)
        features = extractor.extract_features(test_data)
        print(f"Extracted features: {features.keys()}")

        # TODO: 모델 학습 테스트
        # extractor.train_all_models("data/market_data.db")

    except NotImplementedError as e:
        print(f"구현 필요: {e}")
    except Exception as e:
        print(f"오류: {e}")
        import traceback
        traceback.print_exc()
