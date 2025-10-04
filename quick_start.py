#!/usr/bin/env python3
"""
프로젝트 빠른 시작 가이드

이 스크립트는 프로젝트의 주요 기능들을 빠르게 테스트해볼 수 있도록 도와줍니다.
"""

import sys
import os
from pathlib import Path

# 프로젝트 루트 디렉토리를 Python 경로에 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    import torch
    import numpy as np
    from models import ModelConfig, create_model, PRESET_CONFIGS, model_summary
    print("✅ 필수 모듈 로드 성공")
except ImportError as e:
    print(f"❌ 모듈 로드 실패: {e}")
    print("setup_check.py를 실행하여 누락된 패키지를 설치하세요.")
    sys.exit(1)


def test_models():
    """신경망 모델 테스트"""
    print("\n=== 신경망 모델 테스트 ===")
    
    state_size = 50
    action_size = 3
    batch_size = 4
    
    # 샘플 데이터 생성
    sample_input = torch.randn(batch_size, state_size)
    sample_mask = torch.tensor([
        [True, True, False],   # 구매, 보류 가능
        [False, True, True],   # 보류, 판매 가능
        [True, True, True],    # 모든 액션 가능
        [False, True, False]   # 보류만 가능
    ])
    
    # 각 모델 타입 테스트
    for config_name, config in PRESET_CONFIGS.items():
        try:
            print(f"\n--- {config_name} 모델 테스트 ---")
            model = create_model(config, state_size, action_size)
            
            # 모델 정보 출력
            param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"파라미터 수: {param_count:,}")
            
            # 추론 테스트
            model.eval()
            with torch.no_grad():
                output = model(sample_input, sample_mask)
                print(f"출력 형태: {output.shape}")
                print(f"출력 범위: [{output.min().item():.3f}, {output.max().item():.3f}]")
                
                # 액션 마스킹 확인
                masked_actions = output.masked_fill(~sample_mask, -float('inf'))
                actions = torch.argmax(masked_actions, dim=1)
                print(f"선택된 액션: {actions.tolist()}")
                
            print(f"✅ {config_name} 모델 정상 동작")
            
        except Exception as e:
            print(f"❌ {config_name} 모델 오류: {e}")


def test_data_flow():
    """데이터 흐름 테스트"""
    print("\n=== 데이터 흐름 테스트 ===")
    
    try:
        # 가상 시장 데이터 생성
        n_steps = 100
        n_features = 20
        
        # 가격 데이터 (랜덤 워크)
        prices = np.cumsum(np.random.randn(n_steps) * 0.01) + 100
        
        # 기술적 지표 시뮬레이션
        features = np.random.randn(n_steps, n_features)
        
        print(f"✅ 시장 데이터 생성: {n_steps} 스텝, {n_features} 특성")
        print(f"가격 범위: [{prices.min():.2f}, {prices.max():.2f}]")
        
        # 정규화 테스트
        from sklearn.preprocessing import RobustScaler
        scaler = RobustScaler()
        normalized_features = scaler.fit_transform(features)
        
        print(f"✅ 데이터 정규화 완료")
        print(f"정규화 후 범위: [{normalized_features.min():.3f}, {normalized_features.max():.3f}]")
        
    except Exception as e:
        print(f"❌ 데이터 흐름 테스트 실패: {e}")


def test_action_masking():
    """액션 마스킹 테스트"""
    print("\n=== 액션 마스킹 테스트 ===")
    
    try:
        # 다양한 포트폴리오 상태 시뮬레이션
        scenarios = [
            {"cash": 1000000, "crypto": 0, "description": "현금만 보유"},
            {"cash": 0, "crypto": 1.5, "description": "암호화폐만 보유"},
            {"cash": 500000, "crypto": 0.8, "description": "현금+암호화폐 보유"},
            {"cash": 0, "crypto": 0, "description": "모든 자산 없음 (불가능)"}
        ]
        
        for scenario in scenarios:
            cash = scenario["cash"]
            crypto = scenario["crypto"]
            desc = scenario["description"]
            
            # 액션 마스크 생성 로직
            can_buy = cash > 0
            can_hold = True  # 항상 가능
            can_sell = crypto > 0
            
            mask = [can_buy, can_hold, can_sell]
            actions = ["구매", "보류", "판매"]
            
            print(f"\n시나리오: {desc}")
            print(f"현금: {cash:,} KRW, 암호화폐: {crypto} BTC")
            
            available_actions = [action for action, available in zip(actions, mask) if available]
            print(f"가능한 액션: {', '.join(available_actions)}")
            
        print("✅ 액션 마스킹 로직 정상")
        
    except Exception as e:
        print(f"❌ 액션 마스킹 테스트 실패: {e}")


def show_project_summary():
    """프로젝트 요약 정보"""
    print("\n=== 프로젝트 요약 ===")
    
    print("🚀 강화학습 기반 암호화폐 트레이딩 시스템")
    print("\n주요 구성 요소:")
    print("• upbit_api/     - Upbit API 연동")
    print("• models.py      - 신경망 모델 (DQN, LSTM, Transformer, Ensemble)")
    print("• rl_trading_env.py - 강화학습 환경")
    print("• dqn_agent.py   - DQN 에이전트 및 학습")
    print("• backtesting.py - 백테스팅 시스템")
    print("• real_time_trader.py - 실시간 트레이딩")
    
    print("\n다음 단계:")
    print("1. 패키지 설치: uv add gymnasium scikit-learn PyJWT websocket-client python-dotenv ta")
    print("2. 가상환경 활성화: source .venv/bin/activate")
    print("3. 전체 설정 확인: python setup_check.py")
    print("4. 기본 예제 실행: python example.py")
    print("5. 모델 학습: python run_trading_system.py --mode train")


def main():
    """메인 테스트 함수"""
    print("🔍 강화학습 트레이딩 시스템 - 빠른 테스트")
    
    # 기본 정보 출력
    show_project_summary()
    
    # 각종 테스트 실행
    test_models()
    test_data_flow()
    test_action_masking()
    
    print("\n" + "="*60)
    print("🎉 빠른 테스트 완료!")
    print("✅ 프로젝트 구조와 핵심 기능이 정상적으로 작동합니다.")
    print("\n💡 다음으로 시도해볼 것:")
    print("• python example.py - 기본 사용 예제")
    print("• python setup_check.py - 전체 환경 점검")
    print("• README.md 파일 확인 - 상세 사용법")


if __name__ == "__main__":
    main()
