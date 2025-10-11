"""
Stable-Baselines3 Integration

SB3 알고리즘을 사용하기 위한 래퍼 및 모델 클래스
"""

import numpy as np
from typing import Optional, Dict, Any, Callable
import logging

try:
    from stable_baselines3 import PPO, A2C, SAC, TD3, DQN
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import DummyVecEnv
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    logging.warning("Stable-Baselines3 not installed. SB3 models will not be available.")

from .base_model import BaseModel


class SB3TradingModel(BaseModel):
    """
    Stable-Baselines3 알고리즘 래퍼

    지원 알고리즘:
    - PPO: Proximal Policy Optimization
    - A2C: Advantage Actor-Critic
    - SAC: Soft Actor-Critic
    - TD3: Twin Delayed DDPG
    - DQN: Deep Q-Network
    """

    def __init__(
        self,
        state_size: int,
        action_size: int,
        algorithm: str = "PPO",
        policy: str = "MlpPolicy",
        learning_rate: float = 3e-4,
        **kwargs
    ):
        """
        Args:
            state_size: 상태 공간 크기
            action_size: 액션 공간 크기
            algorithm: SB3 알고리즘 (PPO, A2C, SAC, TD3, DQN)
            policy: 정책 네트워크 타입 (MlpPolicy, CnnPolicy 등)
            learning_rate: 학습률
            **kwargs: 알고리즘별 추가 파라미터
        """
        super().__init__(state_size, action_size)

        if not SB3_AVAILABLE:
            raise ImportError(
                "Stable-Baselines3 is not installed. "
                "Install it with: pip install stable-baselines3"
            )

        self.algorithm_name = algorithm.upper()
        self.policy = policy
        self.learning_rate = learning_rate
        self.kwargs = kwargs

        # 알고리즘 매핑
        self.algorithm_map = {
            "PPO": PPO,
            "A2C": A2C,
            "SAC": SAC,
            "TD3": TD3,
            "DQN": DQN
        }

        if self.algorithm_name not in self.algorithm_map:
            raise ValueError(
                f"Unknown algorithm: {algorithm}. "
                f"Available: {list(self.algorithm_map.keys())}"
            )

        self.model = None
        self.env = None

        logging.info(f"SB3Model initialized: {self.algorithm_name} with {policy}")

    def initialize_model(self, env):
        """
        환경을 받아서 SB3 모델 초기화

        Args:
            env: gymnasium 호환 환경
        """
        self.env = env

        # VecEnv로 래핑 (SB3는 vectorized env 필요)
        if not isinstance(env, DummyVecEnv):
            self.env = DummyVecEnv([lambda: env])

        # 알고리즘 클래스 가져오기
        AlgorithmClass = self.algorithm_map[self.algorithm_name]

        # 모델 생성
        self.model = AlgorithmClass(
            policy=self.policy,
            env=self.env,
            learning_rate=self.learning_rate,
            verbose=1,
            **self.kwargs
        )

        logging.info(f"SB3 {self.algorithm_name} model created")

        return self.model

    def forward(self, x):
        """
        순전파 (SB3는 내부적으로 처리하므로 직접 호출 불필요)
        """
        if self.model is None:
            raise RuntimeError("Model not initialized. Call initialize_model() first.")

        # SB3 모델에서 액션 예측
        action, _ = self.model.predict(x, deterministic=True)
        return action

    def train_step(self, total_timesteps: int = 1000):
        """
        SB3 학습 실행

        Args:
            total_timesteps: 학습할 타임스텝 수
        """
        if self.model is None:
            raise RuntimeError("Model not initialized. Call initialize_model() first.")

        self.model.learn(total_timesteps=total_timesteps)

    def save_model(self, path: str):
        """모델 저장"""
        if self.model is None:
            raise RuntimeError("Model not initialized.")

        self.model.save(path)
        logging.info(f"SB3 model saved to {path}")

    def load_model(self, path: str):
        """모델 로드"""
        AlgorithmClass = self.algorithm_map[self.algorithm_name]
        self.model = AlgorithmClass.load(path)
        logging.info(f"SB3 model loaded from {path}")

    def get_config(self) -> Dict[str, Any]:
        """모델 설정 반환"""
        return {
            "model_type": f"sb3_{self.algorithm_name.lower()}",
            "algorithm": self.algorithm_name,
            "policy": self.policy,
            "learning_rate": self.learning_rate,
            **self.kwargs
        }


class TradingCallback(BaseCallback):
    """
    SB3 학습 중 커스텀 콜백

    - 에피소드별 수익률 추적
    - 최고 성능 모델 저장
    - 텐서보드 로깅
    """

    def __init__(
        self,
        save_freq: int = 10000,
        save_path: str = "models/saved/",
        name_prefix: str = "sb3_model",
        verbose: int = 0
    ):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix
        self.best_mean_reward = -np.inf

    def _on_step(self) -> bool:
        """매 스텝마다 호출"""
        # 주기적으로 모델 저장
        if self.n_calls % self.save_freq == 0:
            path = f"{self.save_path}/{self.name_prefix}_{self.n_calls}_steps"
            self.model.save(path)
            if self.verbose > 0:
                print(f"Saving model checkpoint to {path}")

        return True

    def _on_rollout_end(self) -> None:
        """롤아웃 종료 시 호출"""
        # 평균 보상 계산
        if len(self.model.ep_info_buffer) > 0:
            mean_reward = np.mean([ep_info["r"] for ep_info in self.model.ep_info_buffer])

            # 최고 성능 모델 저장
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                path = f"{self.save_path}/{self.name_prefix}_best_model"
                self.model.save(path)
                if self.verbose > 0:
                    print(f"New best mean reward: {mean_reward:.2f}, saving to {path}")


# SB3 알고리즘별 권장 하이퍼파라미터
SB3_RECOMMENDED_PARAMS = {
    "PPO": {
        "n_steps": 2048,
        "batch_size": 64,
        "n_epochs": 10,
        "learning_rate": 3e-4,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.01,
    },
    "A2C": {
        "n_steps": 5,
        "learning_rate": 7e-4,
        "gamma": 0.99,
        "gae_lambda": 1.0,
        "ent_coef": 0.01,
    },
    "SAC": {
        "learning_rate": 3e-4,
        "buffer_size": 100000,
        "learning_starts": 1000,
        "batch_size": 256,
        "tau": 0.005,
        "gamma": 0.99,
        "train_freq": 1,
        "gradient_steps": 1,
    },
    "TD3": {
        "learning_rate": 1e-3,
        "buffer_size": 100000,
        "learning_starts": 1000,
        "batch_size": 100,
        "tau": 0.005,
        "gamma": 0.99,
        "train_freq": (1, "episode"),
        "gradient_steps": -1,
    },
    "DQN": {
        "learning_rate": 1e-4,
        "buffer_size": 100000,
        "learning_starts": 1000,
        "batch_size": 32,
        "tau": 1.0,
        "gamma": 0.99,
        "train_freq": 4,
        "gradient_steps": 1,
        "target_update_interval": 1000,
        "exploration_fraction": 0.1,
        "exploration_final_eps": 0.05,
    }
}


def create_sb3_model(
    env,
    algorithm: str = "PPO",
    policy: str = "MlpPolicy",
    learning_rate: float = 3e-4,
    use_recommended_params: bool = True,
    **kwargs
) -> SB3TradingModel:
    """
    SB3 모델 생성 헬퍼 함수

    Args:
        env: Trading environment
        algorithm: SB3 알고리즘
        policy: 정책 타입
        learning_rate: 학습률
        use_recommended_params: 권장 하이퍼파라미터 사용 여부
        **kwargs: 추가 파라미터

    Returns:
        초기화된 SB3TradingModel
    """
    # 권장 파라미터 사용
    if use_recommended_params and algorithm.upper() in SB3_RECOMMENDED_PARAMS:
        recommended = SB3_RECOMMENDED_PARAMS[algorithm.upper()].copy()
        recommended.update(kwargs)  # 사용자 지정 파라미터로 덮어쓰기
        kwargs = recommended

    # 환경에서 state/action 크기 추출
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    # 모델 생성
    model = SB3TradingModel(
        state_size=state_size,
        action_size=action_size,
        algorithm=algorithm,
        policy=policy,
        learning_rate=learning_rate,
        **kwargs
    )

    # 환경 연결
    model.initialize_model(env)

    return model


# 사용 예제
if __name__ == "__main__":
    print("=== Stable-Baselines3 Integration Test ===\n")

    if not SB3_AVAILABLE:
        print("❌ Stable-Baselines3 not installed")
        print("Install with: pip install stable-baselines3")
    else:
        print("✅ Stable-Baselines3 available")
        print(f"\nSupported algorithms:")
        for algo in ["PPO", "A2C", "SAC", "TD3", "DQN"]:
            print(f"  - {algo}")

        print("\n권장 하이퍼파라미터:")
        for algo, params in SB3_RECOMMENDED_PARAMS.items():
            print(f"\n{algo}:")
            for key, value in params.items():
                print(f"  {key}: {value}")
