"""
Stable-Baselines3 사용 예제

SB3 알고리즘을 사용한 트레이딩 에이전트 학습
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
from trading_env import TradingConfig, TradingEnvironment
from models import SB3_AVAILABLE, create_sb3_model

logging.basicConfig(level=logging.INFO)


def check_sb3_installation():
    """SB3 설치 확인"""
    print("=== Stable-Baselines3 설치 확인 ===\n")

    if SB3_AVAILABLE:
        print("✅ Stable-Baselines3가 설치되어 있습니다.")
        try:
            import stable_baselines3 as sb3
            print(f"버전: {sb3.__version__}\n")
        except:
            print()
    else:
        print("❌ Stable-Baselines3가 설치되어 있지 않습니다.")
        print("\n설치 방법:")
        print("  pip install stable-baselines3")
        print("  또는")
        print("  uv add stable-baselines3")
        print()
        return False

    return True


def example_ppo_training():
    """PPO 알고리즘 사용 예제"""
    print("=== PPO 학습 예제 ===\n")

    # 설정
    config = TradingConfig(
        initial_balance=1000000,
        lookback_window=30,
        model_type="sb3_ppo",  # SB3 PPO 모델
        learning_rate=3e-4
    )

    # 환경 생성
    env = TradingEnvironment(config, market="KRW-BTC")
    print("✅ 트레이딩 환경 생성 완료")

    # SB3 모델 생성
    model = create_sb3_model(
        env=env,
        algorithm="PPO",
        learning_rate=3e-4,
        use_recommended_params=True
    )
    print("✅ PPO 모델 생성 완료")

    # 학습
    print("\n학습 시작 (10,000 timesteps)...")
    model.train_step(total_timesteps=10000)
    print("✅ 학습 완료")

    # 모델 저장
    save_path = "models/saved/sb3_ppo_example"
    model.save_model(save_path)
    print(f"✅ 모델 저장: {save_path}")

    return model


def example_compare_algorithms():
    """여러 SB3 알고리즘 비교 예제"""
    print("\n=== SB3 알고리즘 비교 ===\n")

    algorithms = ["PPO", "A2C", "DQN"]
    results = {}

    config = TradingConfig(
        initial_balance=1000000,
        lookback_window=30
    )

    for algo in algorithms:
        print(f"\n{algo} 학습 중...")

        # 환경 생성 (각 알고리즘마다 새로운 환경)
        env = TradingEnvironment(config, market="KRW-BTC")

        # 모델 생성
        model = create_sb3_model(
            env=env,
            algorithm=algo,
            use_recommended_params=True
        )

        # 짧은 학습 (비교용)
        model.train_step(total_timesteps=5000)

        # 결과 저장
        results[algo] = {
            "model": model,
            "algorithm": algo
        }

        print(f"✅ {algo} 학습 완료")

    # 결과 출력
    print("\n" + "="*50)
    print("학습 결과 요약")
    print("="*50)
    for algo, result in results.items():
        print(f"{algo}: 학습 완료 (5000 timesteps)")

    return results


def example_evaluation():
    """학습된 모델 평가 예제"""
    print("\n=== 모델 평가 예제 ===\n")

    # 환경 생성
    config = TradingConfig(
        initial_balance=1000000,
        lookback_window=30
    )
    env = TradingEnvironment(config, market="KRW-BTC")

    # 모델 생성 및 학습
    model = create_sb3_model(env=env, algorithm="PPO")
    model.train_step(total_timesteps=5000)

    # 평가 (deterministic=True)
    print("평가 시작...")
    obs, _ = env.reset()
    total_reward = 0
    steps = 0

    for _ in range(100):  # 100 스텝 평가
        action = model.forward(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1

        if terminated or truncated:
            break

    print(f"\n평가 완료:")
    print(f"  총 스텝: {steps}")
    print(f"  총 보상: {total_reward:.4f}")
    print(f"  평균 보상: {total_reward/steps:.4f}")
    print(f"  최종 자본: {info['total_value']:,.0f} KRW")


def example_with_callback():
    """콜백 사용 예제"""
    print("\n=== 콜백 사용 예제 ===\n")

    try:
        from models.sb3_wrapper import TradingCallback

        config = TradingConfig(initial_balance=1000000)
        env = TradingEnvironment(config, market="KRW-BTC")

        # 콜백 설정
        callback = TradingCallback(
            save_freq=1000,
            save_path="models/saved/",
            name_prefix="ppo_callback",
            verbose=1
        )

        # 모델 생성
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv

        vec_env = DummyVecEnv([lambda: env])
        model_sb3 = PPO("MlpPolicy", vec_env, verbose=1)

        # 콜백과 함께 학습
        print("콜백과 함께 학습 시작...")
        model_sb3.learn(total_timesteps=5000, callback=callback)
        print("✅ 학습 완료")

    except ImportError:
        print("❌ 콜백 예제는 SB3가 설치되어 있어야 합니다.")


def main():
    """메인 함수"""
    print("\n" + "="*60)
    print(" "*15 + "SB3 사용 예제")
    print("="*60 + "\n")

    # SB3 설치 확인
    if not check_sb3_installation():
        return

    try:
        # 예제 1: PPO 학습
        example_ppo_training()

        # 예제 2: 알고리즘 비교 (optional, 시간이 오래 걸림)
        # example_compare_algorithms()

        # 예제 3: 모델 평가
        # example_evaluation()

        # 예제 4: 콜백 사용
        # example_with_callback()

        print("\n" + "="*60)
        print("✅ 모든 예제 실행 완료!")
        print("="*60)

    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
