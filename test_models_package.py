"""
Models Package Test Script

새로 분리된 models 패키지를 테스트합니다.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    # 1. 기본 import 테스트
    print("=== Models Package Import 테스트 ===")
    
    from models import ModelConfig, PRESET_CONFIGS
    print("✅ ModelConfig, PRESET_CONFIGS 가져오기 성공")
    
    from models import DQNModel, LSTMModel, TransformerModel, EnsembleModel
    print("✅ 모델 클래스들 가져오기 성공")
    
    from models import create_model, count_parameters, model_summary
    print("✅ 팩토리 함수들 가져오기 성공")
    
    # 2. 설정 테스트
    print("\n=== 모델 설정 테스트 ===")
    config = ModelConfig(model_type="dqn", hidden_size=256)
    print(f"✅ 기본 설정 생성: {config.model_type}, hidden_size={config.hidden_size}")
    
    preset_config = PRESET_CONFIGS["medium_dqn"]
    print(f"✅ 프리셋 설정: {preset_config.model_type}, hidden_size={preset_config.hidden_size}")
    
    # 3. 모델 생성 테스트
    print("\n=== 모델 생성 테스트 ===")
    state_size = 50
    action_size = 3
    
    # DQN 모델
    dqn_model = create_model(PRESET_CONFIGS["small_dqn"], state_size, action_size)
    dqn_params = count_parameters(dqn_model)
    print(f"✅ DQN 모델 생성: {dqn_params:,} 파라미터")
    
    # LSTM 모델
    lstm_model = create_model(PRESET_CONFIGS["small_lstm"], state_size, action_size)
    lstm_params = count_parameters(lstm_model)
    print(f"✅ LSTM 모델 생성: {lstm_params:,} 파라미터")
    
    # Transformer 모델
    transformer_model = create_model(PRESET_CONFIGS["small_transformer"], state_size, action_size)
    transformer_params = count_parameters(transformer_model)
    print(f"✅ Transformer 모델 생성: {transformer_params:,} 파라미터")
    
    # 4. 모델 추론 테스트
    print("\n=== 모델 추론 테스트 ===")
    import torch
    
    # 샘플 입력
    sample_input = torch.randn(2, state_size)  # 배치 크기 2
    
    # DQN 추론
    with torch.no_grad():
        dqn_output = dqn_model(sample_input)
        print(f"✅ DQN 출력 형태: {dqn_output.shape}")
    
    # LSTM 추론
    with torch.no_grad():
        lstm_output = lstm_model(sample_input)
        print(f"✅ LSTM 출력 형태: {lstm_output.shape}")
    
    # Transformer 추론
    with torch.no_grad():
        transformer_output = transformer_model(sample_input)
        print(f"✅ Transformer 출력 형태: {transformer_output.shape}")
    
    # 5. 모델 요약 테스트
    print("\n=== 모델 요약 테스트 ===")
    summary = model_summary(dqn_model, (state_size,))
    print("✅ 모델 요약 생성 성공")
    print(summary)
    
    print("\n🎉 모든 테스트 통과!")
    print("models 패키지가 성공적으로 분리되었습니다!")
    
except Exception as e:
    print(f"❌ 테스트 실패: {e}")
    import traceback
    traceback.print_exc()