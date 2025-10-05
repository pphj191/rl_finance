"""
간단한 강화학습 환경 테스트
"""

import sys
import traceback

print("=== 강화학습 트레이딩 시스템 테스트 ===\n")

try:
    # 1. 모듈 import 테스트
    print("1. 모듈 import 테스트...")
    from trading_env import TradingEnvironment
    from trading_env.rl_env import TradingConfig, ActionSpace
    print("✅ 모듈 import 성공!\n")

    # 2. 설정 생성 테스트
    print("2. 설정 생성 테스트...")
    config = TradingConfig(
        initial_balance=1000000,
        lookback_window=10,  # 작은 윈도우로 테스트
        model_type="dqn"
    )
    print("✅ Config 생성 성공!\n")

    # 3. 환경 생성 테스트
    print("3. 환경 생성 테스트...")
    env = TradingEnvironment(config, market="KRW-BTC")
    print("✅ Environment 생성 성공!\n")

    # 4. 환경 리셋 테스트
    print("4. 환경 리셋 테스트...")
    obs, info = env.reset()
    print(f"✅ Environment reset 성공!")
    print(f"   관측값 크기: {obs.shape}")
    print(f"   관측 공간: {env.observation_space}")
    print(f"   액션 공간: {env.action_space}\n")

    # 5. 액션 마스킹 테스트
    print("5. 액션 마스킹 테스트...")
    action_mask = env.get_action_mask()
    print(f"✅ 액션 마스크: {action_mask}")
    print(f"   유효한 액션: {[ActionSpace.get_action_names()[i] for i, valid in enumerate(action_mask) if valid]}\n")

    # 6. 몇 스텝 실행 테스트
    print("6. 환경 스텝 실행 테스트...")
    for step in range(3):
        # 유효한 액션 중 랜덤 선택
        valid_actions = [i for i, valid in enumerate(action_mask) if valid]
        if valid_actions:
            action = valid_actions[0]  # 첫 번째 유효한 액션 선택
        else:
            action = ActionSpace.HOLD
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        print(f"   Step {step}: Action={ActionSpace.get_action_names()[action]}")
        print(f"            Reward={reward:.4f}")
        print(f"            Balance={info['balance']:.2f}")
        print(f"            Position={info['position']:.6f}")
        print(f"            Total Value={info['total_value']:.2f}")
        
        if terminated or truncated:
            print("   환경 종료")
            break
        
        # 다음 스텝을 위한 액션 마스크 업데이트
        action_mask = env.get_action_mask()

    print("\n✅ 모든 테스트 통과!")
    print("\n=== 다음 단계 ===")
    print("1. DQN 에이전트 학습: python dqn_agent.py")
    print("2. 백테스팅 실행: python backtesting.py")

except Exception as e:
    print(f"❌ 오류 발생: {e}")
    print("\n상세 오류:")
    traceback.print_exc()
    
    print("\n=== 문제 해결 방법 ===")
    print("1. 패키지 설치 확인: pip install -r requirements.txt")
    print("2. API 키 설정 확인: .env 파일의 UPBIT_ACCESS_KEY, UPBIT_SECRET_KEY")
    print("3. 인터넷 연결 확인")
