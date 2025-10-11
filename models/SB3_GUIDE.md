# Stable-Baselines3 Integration Guide

## 📚 개요

이 프로젝트는 [Stable-Baselines3 (SB3)](https://stable-baselines3.readthedocs.io/)를 통합하여 검증된 강화학습 알고리즘을 쉽게 사용할 수 있도록 지원합니다.

### 지원 알고리즘

| 알고리즘 | 설명 | 적합한 경우 |
|---------|------|-----------|
| **PPO** | Proximal Policy Optimization | 안정적인 학습, 일반적인 용도 |
| **A2C** | Advantage Actor-Critic | 빠른 학습, 실시간 트레이딩 |
| **SAC** | Soft Actor-Critic | 연속 액션 공간 (향후 지원) |
| **TD3** | Twin Delayed DDPG | 고급 연속 제어 |
| **DQN** | Deep Q-Network | 이산 액션, SB3 구현 비교용 |

---

## 🚀 설치

### 1. Stable-Baselines3 설치

```bash
pip install stable-baselines3
```

또는 uv 사용:

```bash
uv add stable-baselines3
```

### 2. 추가 패키지 (선택사항)

텐서보드 로깅 및 추가 기능:

```bash
pip install stable-baselines3[extra]
```

---

## 💻 사용 방법

### 기본 사용법

```python
from trading_env import TradingConfig, TradingEnvironment
from models import create_sb3_model

# 1. 설정
config = TradingConfig(
    initial_balance=1000000,
    lookback_window=30,
    model_type="sb3_ppo"  # SB3 모델 타입
)

# 2. 환경 생성
env = TradingEnvironment(config, market="KRW-BTC")

# 3. SB3 모델 생성
model = create_sb3_model(
    env=env,
    algorithm="PPO",
    learning_rate=3e-4,
    use_recommended_params=True  # 권장 하이퍼파라미터 사용
)

# 4. 학습
model.train_step(total_timesteps=10000)

# 5. 저장
model.save_model("models/saved/my_ppo_model")
```

### 직접 구현 vs SB3

```python
# 직접 구현 DQN
from models import ModelConfig, create_model

config = ModelConfig(model_type="dqn", hidden_size=256)
model = create_model(config, state_size=50, action_size=3)

# SB3 DQN
from models import create_sb3_model

model = create_sb3_model(env, algorithm="DQN")
```

---

## 🎯 run_train.py에서 사용하기

### 명령줄에서 SB3 모델 학습

```bash
# PPO 학습
python run_train.py \
    --model-type sb3_ppo \
    --episodes 1000 \
    --market KRW-BTC

# A2C 학습
python run_train.py \
    --model-type sb3_a2c \
    --episodes 500

# DQN 학습 (SB3 구현)
python run_train.py \
    --model-type sb3_dqn \
    --episodes 1000
```

### 설정 파일 사용

`configs/sb3_ppo_config.json`:
```json
{
  "model_type": "sb3_ppo",
  "hidden_size": 256,
  "learning_rate": 0.0003,
  "batch_size": 64,
  "sb3_params": {
    "n_steps": 2048,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2
  }
}
```

사용:
```bash
python run_train.py --config configs/sb3_ppo_config.json
```

---

## 🔧 고급 사용법

### 1. 커스텀 하이퍼파라미터

```python
model = create_sb3_model(
    env=env,
    algorithm="PPO",
    learning_rate=1e-4,
    use_recommended_params=False,  # 권장값 사용 안 함
    # PPO 전용 파라미터
    n_steps=1024,
    batch_size=128,
    n_epochs=20,
    gamma=0.95,
    clip_range=0.3
)
```

### 2. 콜백 사용

```python
from models.sb3_wrapper import TradingCallback

callback = TradingCallback(
    save_freq=10000,
    save_path="models/saved/",
    name_prefix="ppo_trading",
    verbose=1
)

# SB3 모델에 직접 접근하여 콜백 사용
model.model.learn(total_timesteps=100000, callback=callback)
```

### 3. 텐서보드 로깅

```python
from stable_baselines3.common.vec_env import DummyVecEnv

# Vectorized environment
vec_env = DummyVecEnv([lambda: env])

# 텐서보드 로깅 활성화
from stable_baselines3 import PPO

model = PPO(
    "MlpPolicy",
    vec_env,
    verbose=1,
    tensorboard_log="./tensorboard_logs/"
)

model.learn(total_timesteps=100000)
```

텐서보드 실행:
```bash
tensorboard --logdir ./tensorboard_logs/
```

### 4. 모델 평가

```python
from stable_baselines3.common.evaluation import evaluate_policy

# 평가
mean_reward, std_reward = evaluate_policy(
    model.model,
    env,
    n_eval_episodes=10,
    deterministic=True
)

print(f"평균 보상: {mean_reward:.2f} +/- {std_reward:.2f}")
```

---

## 📊 권장 하이퍼파라미터

### PPO
```python
{
    "n_steps": 2048,           # 각 업데이트당 스텝 수
    "batch_size": 64,          # 미니배치 크기
    "n_epochs": 10,            # 업데이트 반복 횟수
    "learning_rate": 3e-4,     # 학습률
    "gamma": 0.99,             # 할인율
    "gae_lambda": 0.95,        # GAE 람다
    "clip_range": 0.2,         # 클리핑 범위
    "ent_coef": 0.01           # 엔트로피 계수
}
```

### A2C
```python
{
    "n_steps": 5,              # 각 업데이트당 스텝 수
    "learning_rate": 7e-4,     # 학습률
    "gamma": 0.99,             # 할인율
    "gae_lambda": 1.0,         # GAE 람다
    "ent_coef": 0.01           # 엔트로피 계수
}
```

### DQN
```python
{
    "learning_rate": 1e-4,            # 학습률
    "buffer_size": 100000,            # 리플레이 버퍼 크기
    "learning_starts": 1000,          # 학습 시작 스텝
    "batch_size": 32,                 # 배치 크기
    "tau": 1.0,                       # 타겟 네트워크 업데이트 비율
    "gamma": 0.99,                    # 할인율
    "train_freq": 4,                  # 학습 빈도
    "target_update_interval": 1000,   # 타겟 업데이트 주기
    "exploration_fraction": 0.1,      # 탐험 비율
    "exploration_final_eps": 0.05     # 최종 엡실론
}
```

---

## 🔍 직접 구현 vs SB3 비교

| 특징 | 직접 구현 | SB3 |
|-----|----------|-----|
| **장점** | 완전한 제어, 커스터마이징 자유 | 검증된 구현, 빠른 개발 |
| **단점** | 버그 가능성, 시간 소요 | 커스터마이징 제약 |
| **학습 속도** | 최적화 필요 | 최적화됨 |
| **안정성** | 구현에 따라 다름 | 높음 |
| **문서화** | 자체 작성 필요 | 풍부함 |
| **커뮤니티** | 없음 | 활발함 |

### 사용 추천

**직접 구현 사용:**
- 새로운 알고리즘 실험
- 특수한 리워드 함수
- 연구 목적

**SB3 사용:**
- 빠른 프로토타이핑
- 벤치마크 비교
- 프로덕션 배포

---

## 🐛 문제 해결

### SB3 설치 오류

```bash
# gymnasium 버전 충돌 시
pip install "gymnasium>=0.28.0"

# PyTorch 버전 확인
pip install torch>=1.11.0
```

### 메모리 부족

```python
# 버퍼 크기 줄이기
model = create_sb3_model(
    env,
    algorithm="DQN",
    buffer_size=10000  # 기본 100000에서 감소
)
```

### 학습이 불안정할 때

```python
# 학습률 감소
model = create_sb3_model(
    env,
    algorithm="PPO",
    learning_rate=1e-5  # 3e-4에서 감소
)
```

---

## 📚 참고 자료

- [SB3 공식 문서](https://stable-baselines3.readthedocs.io/)
- [SB3 GitHub](https://github.com/DLR-RM/stable-baselines3)
- [RL Baselines3 Zoo](https://github.com/DLR-RM/rl-baselines3-zoo) (사전 학습된 모델)
- [SB3 Contrib](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib) (추가 알고리즘)

---

## 📝 예제 코드

전체 예제는 [`examples/example_sb3_usage.py`](../examples/example_sb3_usage.py)를 참고하세요.

```bash
python examples/example_sb3_usage.py
```

---

**작성일**: 2025-10-07
**버전**: 1.0.0
