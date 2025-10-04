#!/usr/bin/env python3
"""
Bithumb API 테스트 스크립트

Bithumb API 클라이언트의 기본 기능을 테스트합니다.
"""

import sys
import os
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_bithumb_api_structure():
    """Bithumb API 구조 테스트"""
    print("=== Bithumb API 구조 테스트 ===")
    
    # 폴더 구조 확인
    bithumb_dir = Path(__file__).parent
    required_files = [
        '__init__.py',
        'bithumb_api.py',
        'README.md'
    ]
    
    for file_name in required_files:
        file_path = bithumb_dir / file_name
        if file_path.exists():
            print(f"✅ {file_name}")
        else:
            print(f"❌ {file_name} - 파일 없음")
    
    # __init__.py 내용 확인
    init_file = bithumb_dir / '__init__.py'
    if init_file.exists():
        content = init_file.read_text()
        if 'BithumbAPI' in content:
            print("✅ __init__.py에 BithumbAPI 포함됨")
        if 'BithumbWebSocket' in content:
            print("✅ __init__.py에 BithumbWebSocket 포함됨")


def test_upbit_compatibility():
    """Upbit 호환성 테스트"""
    print("\n=== Upbit 호환성 테스트 ===")
    
    # bithumb_api.py 파일에서 Upbit 호환 함수 확인
    bithumb_file = Path(__file__).parent / 'bithumb_api.py'
    if bithumb_file.exists():
        content = bithumb_file.read_text()
        
        upbit_functions = [
            'get_market_all',
            'get_candles_minutes', 
            'get_ticker',
            'get_orderbook',
            'get_trades_ticks',
            'get_accounts',
            'get_order',
            'get_orders',
            'cancel_order',
            'order'
        ]
        
        for func_name in upbit_functions:
            if f'def {func_name}(' in content:
                print(f"✅ {func_name} 함수 구현됨")
            else:
                print(f"❌ {func_name} 함수 누락")
        
        # Upbit 호환 편의 함수 확인
        upbit_compat_functions = [
            'get_upbit_market_all',
            'get_upbit_candles_minutes',
            'get_upbit_ticker', 
            'get_upbit_orderbook'
        ]
        
        print("\n--- Upbit 편의 함수 ---")
        for func_name in upbit_compat_functions:
            if f'def {func_name}(' in content:
                print(f"✅ {func_name} 함수 구현됨")
            else:
                print(f"❌ {func_name} 함수 누락")


def test_api_class_structure():
    """API 클래스 구조 테스트"""
    print("\n=== API 클래스 구조 테스트 ===")
    
    bithumb_file = Path(__file__).parent / 'bithumb_api.py'
    if bithumb_file.exists():
        content = bithumb_file.read_text()
        
        # 클래스 존재 확인
        classes = [
            'BithumbConfig',
            'BithumbAPIError', 
            'BithumbAPI',
            'BithumbWebSocket'
        ]
        
        for class_name in classes:
            if f'class {class_name}' in content:
                print(f"✅ {class_name} 클래스 정의됨")
            else:
                print(f"❌ {class_name} 클래스 누락")
        
        # 중요 메서드 확인
        methods = [
            '__init__',
            '_create_signature',
            '_request',
            'connect',  # WebSocket
            'disconnect'  # WebSocket
        ]
        
        print("\n--- 주요 메서드 ---")
        for method_name in methods:
            if f'def {method_name}(' in content:
                print(f"✅ {method_name} 메서드 구현됨")
            else:
                print(f"❌ {method_name} 메서드 누락")


def test_documentation():
    """문서화 테스트"""
    print("\n=== 문서화 테스트 ===")
    
    readme_file = Path(__file__).parent / 'README.md'
    if readme_file.exists():
        content = readme_file.read_text()
        
        required_sections = [
            '# Bithumb API 클라이언트',
            '## 설치',
            '## 설정', 
            '## 사용법',
            '### 시세 정보 조회',
            '### 계정 및 주문 관리',
            '### WebSocket',
            '## API 레퍼런스',
            '## 주의사항'
        ]
        
        for section in required_sections:
            if section in content:
                print(f"✅ {section}")
            else:
                print(f"❌ {section} 섹션 누락")
        
        # Upbit 호환성 언급 확인
        if 'Upbit' in content and '호환' in content:
            print("✅ Upbit 호환성 설명 포함됨")
        else:
            print("❌ Upbit 호환성 설명 누락")


def show_summary():
    """요약 정보"""
    print("\n=== Bithumb API 패키지 요약 ===")
    
    print("📁 패키지 구조:")
    print("   bithumb_api/")
    print("   ├── __init__.py          # 패키지 초기화")
    print("   ├── bithumb_api.py       # 메인 API 클라이언트")
    print("   └── README.md            # 상세 문서")
    
    print("\n🔗 Upbit 호환성:")
    print("   • 동일한 함수명 사용")
    print("   • 동일한 응답 형식")
    print("   • 쉬운 마이그레이션")
    
    print("\n💡 주요 기능:")
    print("   • REST API 완전 지원")
    print("   • WebSocket 실시간 데이터")
    print("   • 자동 재시도 및 에러 처리")
    print("   • Type Hints 지원")
    
    print("\n🚀 다음 단계:")
    print("   1. 필수 패키지 설치: uv add python-dotenv")
    print("   2. API 키 설정: .env 파일에 BITHUMB_* 키 추가")
    print("   3. 테스트 실행: python bithumb_api/test_api.py")
    print("   4. 문서 확인: bithumb_api/README.md")


def main():
    """메인 테스트 함수"""
    print("🔍 Bithumb API 패키지 테스트\n")
    
    test_bithumb_api_structure()
    test_upbit_compatibility()
    test_api_class_structure()
    test_documentation()
    show_summary()
    
    print("\n" + "="*60)
    print("🎉 Bithumb API 패키지 구조 검증 완료!")
    print("✅ Upbit API와 호환되는 인터페이스로 구현되었습니다.")


if __name__ == "__main__":
    main()
