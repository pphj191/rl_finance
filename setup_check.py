#!/usr/bin/env python3
"""
프로젝트 구조 확인 및 기본 테스트
"""

import os
import sys

def print_project_structure():
    """프로젝트 구조 출력"""
    print("📁 프로젝트 구조:")
    print("rl/")
    print("├── 📂 upbit_api/           # Upbit API 패키지")
    print("│   ├── __init__.py")
    print("│   ├── upbit_api.py        # Upbit API 클래스")
    print("│   └── README.md           # API 문서")
    print("├── rl_trading_env.py       # 강화학습 환경")
    print("├── dqn_agent.py           # DQN 에이전트")
    print("├── backtesting.py         # 백테스팅")
    print("├── real_time_trader.py    # 실시간 트레이딩")
    print("├── run_trading_system.py  # 통합 실행 스크립트")
    print("├── example.py             # 기본 예제")
    print("├── advanced_example.py    # 고급 예제")
    print("├── test.py               # 종합 테스트")
    print("├── quick_test.py         # 빠른 테스트")
    print("├── README.md             # 프로젝트 문서")
    print("└── .env                  # 환경 설정")
    print()

#!/usr/bin/env python3
"""
프로젝트 설정 및 구조 확인 스크립트

전체 프로젝트의 구조와 의존성을 검증합니다.
"""

import os
import sys
import importlib
from pathlib import Path


def check_python_version():
    """Python 버전 확인"""
    print("=== Python 환경 확인 ===")
    version = sys.version_info
    print(f"Python 버전: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 이상이 필요합니다.")
        return False
    else:
        print("✅ Python 버전 적합")
        return True


def check_virtual_environment():
    """가상환경 확인"""
    print("=== 가상환경 확인 ===")
    
    # .venv 폴더 확인
    venv_path = Path(".venv")
    if venv_path.exists():
        print("✅ .venv 폴더 발견")
    else:
        print("⚠️  .venv 폴더가 없습니다. 가상환경을 생성하세요:")
        print("   python -m venv .venv")
        print("   source .venv/bin/activate  # macOS/Linux")
        print("   .venv\\\\Scripts\\\\activate     # Windows")
    
    # 가상환경 활성화 여부 확인
    virtual_env = os.environ.get('VIRTUAL_ENV')
    if virtual_env:
        print(f"✅ 가상환경 활성화됨: {virtual_env}")
        return True
    else:
        print("⚠️  가상환경이 활성화되지 않았습니다.")
        return False


def check_required_packages():
    """필수 패키지 확인"""
    print("=== 필수 패키지 확인 ===")
    
    required_packages = [
        'torch',
        'gymnasium', 
        'numpy',
        'pandas',
        'scikit-learn',
        'matplotlib',
        'seaborn',
        'requests',
        'PyJWT',
        'websocket-client',
        'python-dotenv',
        'ta'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            importlib.import_module(package.replace('-', '_'))
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - 설치 필요")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"⚠️  누락된 패키지: {', '.join(missing_packages)}")
        print("uv를 사용하여 설치하세요:")
        print(f"uv add {' '.join(missing_packages)}")
        return False
    else:
        print("✅ 모든 필수 패키지 설치됨")
        return True


def check_project_structure():
    """프로젝트 구조 확인"""
    print("=== 프로젝트 구조 확인 ===")
    
    required_files = [
        'upbit_api/__init__.py',
        'upbit_api/upbit_api.py',
        'upbit_api/README.md',
        'bithumb_api/__init__.py',
        'bithumb_api/bithumb_api.py', 
        'bithumb_api/README.md',
        'rl_trading_env.py',
        'models.py',
        'dqn_agent.py',
        'backtesting.py',
        'real_time_trader.py',
        'run_trading_system.py',
        'README.md'
    ]
    
    required_dirs = [
        'upbit_api/',
        'bithumb_api/',
        'models/'
    ]
    
    # 디렉토리 확인
    for dir_path in required_dirs:
        if Path(dir_path).exists():
            print(f"✅ {dir_path}")
        else:
            print(f"❌ {dir_path} - 디렉토리 없음")
            if dir_path == 'models/':
                print("   models/ 디렉토리를 생성합니다...")
                Path(dir_path).mkdir(exist_ok=True)
                print("   ✅ models/ 디렉토리 생성됨")
    
    # 파일 확인
    missing_files = []
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - 파일 없음")
            missing_files.append(file_path)
    
    if missing_files:
        print(f"⚠️  누락된 파일: {len(missing_files)}개")
        return False
    else:
        print("✅ 프로젝트 구조 정상")
        return True


def check_env_file():
    """환경 설정 파일 확인"""
    print("=== 환경 설정 확인 ===")
    
    env_file = Path('.env')
    if env_file.exists():
        print("✅ .env 파일 존재")
        
        # .env 파일 내용 확인 (보안상 키 값은 표시하지 않음)
        try:
            with open('.env', 'r') as f:
                content = f.read()
                
                # Upbit API 키 확인
                if 'UPBIT_ACCESS_KEY' in content:
                    print("✅ UPBIT_ACCESS_KEY 설정됨")
                else:
                    print("❌ UPBIT_ACCESS_KEY 누락")
                
                if 'UPBIT_SECRET_KEY' in content:
                    print("✅ UPBIT_SECRET_KEY 설정됨")
                else:
                    print("❌ UPBIT_SECRET_KEY 누락")
                
                # Bithumb API 키 확인
                if 'BITHUMB_ACCESS_KEY' in content:
                    print("✅ BITHUMB_ACCESS_KEY 설정됨")
                else:
                    print("ℹ️  BITHUMB_ACCESS_KEY 미설정 (선택사항)")
                
                if 'BITHUMB_SECRET_KEY' in content:
                    print("✅ BITHUMB_SECRET_KEY 설정됨")
                else:
                    print("ℹ️  BITHUMB_SECRET_KEY 미설정 (선택사항)")
                    
        except Exception as e:
            print(f"⚠️  .env 파일 읽기 오류: {e}")
    else:
        print("❌ .env 파일 없음")
        print("   .env 파일을 생성하고 다음 내용을 추가하세요:")
        print("   # Upbit API")
        print("   UPBIT_ACCESS_KEY=your_upbit_access_key_here")
        print("   UPBIT_SECRET_KEY=your_upbit_secret_key_here")
        print("   # Bithumb API (선택사항)")
        print("   BITHUMB_ACCESS_KEY=your_bithumb_access_key_here")
        print("   BITHUMB_SECRET_KEY=your_bithumb_secret_key_here")
        return False
    
    return True


def check_module_imports():
    """모듈 import 확인"""
    print("=== 모듈 Import 확인 ===")
    
    modules_to_test = [
        ('upbit_api', 'UpbitAPI'),
        ('bithumb_api', 'BithumbAPI'),
        ('models', 'create_model'),
        ('rl_trading_env', 'TradingEnvironment'),
        ('dqn_agent', 'DQNAgent'),
        ('backtesting', 'Backtester'),
        ('real_time_trader', 'RealTimeTrader')
    ]
    
    success_count = 0
    for module_name, class_name in modules_to_test:
        try:
            module = importlib.import_module(module_name)
            if hasattr(module, class_name):
                print(f"✅ {module_name}.{class_name}")
                success_count += 1
            else:
                print(f"❌ {module_name}.{class_name} - 클래스/함수 없음")
        except ImportError as e:
            print(f"❌ {module_name} - Import 오류: {e}")
        except Exception as e:
            print(f"⚠️  {module_name} - 기타 오류: {e}")
    
    total_modules = len(modules_to_test)
    print(f"✅ {success_count}/{total_modules} 모듈 정상 동작")
    return success_count == total_modules


def check_gpu_availability():
    """GPU 사용 가능 여부 확인"""
    print("=== GPU 환경 확인 ===")
    
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA 사용 가능 (GPU 개수: {torch.cuda.device_count()})")
            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                print(f"   GPU {i}: {gpu_name}")
        else:
            print("ℹ️  CUDA 사용 불가 - CPU 모드로 실행됩니다")
        
        if torch.backends.mps.is_available():
            print("✅ Apple Metal Performance Shaders (MPS) 사용 가능")
        
    except ImportError:
        print("❌ PyTorch 설치되지 않음")


def run_setup_check():
    """메인 확인 함수"""
    print("🔍 강화학습 트레이딩 시스템 - 프로젝트 설정 확인\n")
    
    checks = [
        check_python_version(),
        check_virtual_environment(), 
        check_required_packages(),
        check_project_structure(),
        check_env_file(),
        check_module_imports()
    ]
    
    # GPU 확인 (선택사항)
    check_gpu_availability()
    
    print("\n" + "="*50)
    passed_checks = sum(checks)
    total_checks = len(checks)
    
    if passed_checks == total_checks:
        print("🎉 모든 확인 항목 통과!")
        print("✅ 프로젝트 설정이 완료되었습니다.")
        print("\n다음 명령어로 시스템을 시작할 수 있습니다:")
        print("   python quick_test.py")
        print("   python example.py")
        return True
    else:
        print(f"⚠️  {total_checks - passed_checks}개 항목에 문제가 있습니다.")
        print("위의 지시사항을 따라 문제를 해결한 후 다시 실행하세요.")
        return False


if __name__ == "__main__":
    success = run_setup_check()
    sys.exit(0 if success else 1)

def show_usage():
    """사용법 안내"""
    print("🚀 사용법:")
    print("1. 모델 학습:")
    print("   python run_trading_system.py --mode train --episodes 1000")
    print()
    print("2. 백테스팅:")
    print("   python run_trading_system.py --mode backtest --model models/best_model.pth")
    print()
    print("3. 실시간 트레이딩:")
    print("   python run_trading_system.py --mode live --model models/best_model.pth --duration 60")
    print()
    print("4. 기본 예제:")
    print("   python example.py")
    print()
    print("5. 고급 예제:")
    print("   python advanced_example.py")
    print()

def main():
    print("=" * 60)
    print("🤖 강화학습 기반 암호화폐 트레이딩 시스템")
    print("=" * 60)
    print()
    show_usage()
    
    print("⚠️  주의사항:")
    print("- 실시간 트레이딩은 실제 자금이 사용됩니다")
    print("- .env 파일에 Upbit API 키를 설정해주세요")
    print("- 백테스팅으로 충분히 검증 후 실거래를 시작하세요")
    print()
    print("🎯 모든 TODO 항목이 완료되었습니다!")

if __name__ == "__main__":
    main()
