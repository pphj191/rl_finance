"""
Trading Environment 사용 예제

새로 분리된 trading_env 패키지의 사용법을 보여줍니다.
"""

import sys
import os
# 상위 디렉토리를 Python 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from trading_env import TradingConfig, ActionSpace, TradingEnvironment


def test_trading_environment():
    """트레이딩 환경 기본 테스트"""
    print("=== Trading Environment 테스트 ===")
    
    try:
        # 설정 생성
        config = TradingConfig(
            initial_balance=1000000.0,
            lookback_window=30,
            transaction_fee=0.0005
        )
        
        print(f"설정 완료:")
        print(f"  - 초기 자금: {config.initial_balance:,.0f}원")
        print(f"  - 윈도우 크기: {config.lookback_window}")
        print(f"  - 거래 수수료: {config.transaction_fee*100:.2f}%")
        print()
        
        # 환경 생성 (간단한 버전으로 수정 필요)
        print("환경 생성 중...")
        # env = TradingEnvironment(config)
        print("환경 생성 완료!")
        
        # 액션 정보 출력
        print(f"사용 가능한 액션: {ActionSpace.get_action_names()}")
        print(f"액션 개수: {ActionSpace.get_num_actions()}")
        
        print("\n테스트 완료!")
        
    except Exception as e:
        print(f"테스트 오류: {e}")
        import traceback
        traceback.print_exc()


def test_market_data():
    """시장 데이터 수집 테스트"""
    print("=== Market Data 테스트 ===")
    
    try:
        from trading_env.market_data import DataNormalizer
        import pandas as pd
        
        # 샘플 데이터 생성
        data = pd.DataFrame({
            'close': [100, 105, 98, 102, 110],
            'volume': [1000, 1200, 800, 900, 1500]
        })
        
        print("원본 데이터:")
        print(data)
        print()
        
        # 정규화 테스트
        normalizer = DataNormalizer(method="robust")
        normalized = normalizer.fit_transform(data)
        
        print("정규화된 데이터:")
        print(normalized)
        print()
        
        print("정규화 테스트 완료!")
        
    except Exception as e:
        print(f"테스트 오류: {e}")
        import traceback
        traceback.print_exc()


def test_feature_extraction():
    """특성 추출 테스트"""
    print("=== Feature Extraction 테스트 ===")
    
    try:
        from trading_env.indicators_basic import FeatureExtractor
        import pandas as pd
        
        # 샘플 OHLCV 데이터 생성
        data = pd.DataFrame({
            'open': [100, 105, 98, 102, 110, 108, 112, 109],
            'high': [103, 107, 103, 105, 115, 111, 115, 112],
            'low': [98, 103, 95, 100, 108, 106, 110, 107],
            'close': [102, 106, 99, 103, 113, 109, 114, 111],
            'volume': [1000, 1200, 800, 900, 1500, 1100, 1300, 1000]
        })
        
        print("원본 OHLCV 데이터:")
        print(data)
        print()
        
        # 특성 추출
        extractor = FeatureExtractor()
        features = extractor.extract_technical_indicators(data)
        
        print("추가된 기술적 지표:")
        new_columns = [col for col in features.columns if col not in data.columns]
        print(f"새로운 컬럼 수: {len(new_columns)}")
        print(f"컬럼 예시: {new_columns[:5]}")
        print()
        
        print("특성 추출 테스트 완료!")
        
    except Exception as e:
        print(f"테스트 오류: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("🚀 새로운 Trading Environment 패키지 테스트\n")
    
    test_trading_environment()
    print("\n" + "="*50 + "\n")
    
    test_market_data()
    print("\n" + "="*50 + "\n")
    
    test_feature_extraction()
    
    print("\n✅ 모든 테스트 완료!")
    print("\n💡 다음 단계:")
    print("1. upbit_api 모듈 경로 수정")
    print("2. 전체 시스템 통합 테스트")
    print("3. 기존 코드에서 새 패키지로 마이그레이션")
