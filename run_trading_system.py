#!/usr/bin/env python3
"""
강화학습 트레이딩 시스템 - 통합 테스트 및 실행 스크립트

사용법:
    python run_trading_system.py --mode [train|backtest|live]
"""

import argparse
import os
import sys
import logging
from datetime import datetime, timedelta
import json

# 프로젝트 루트 디렉토리를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rl_trading_env import TradingEnvironment, TradingConfig
from dqn_agent import DQNAgent, TradingTrainer
from backtesting import Backtester
from real_time_trader import RealTimeTrader, RiskConfig
from upbit_api import UpbitAPI


def setup_logging():
    """로깅 설정"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'trading_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
            logging.StreamHandler()
        ]
    )


def train_model(config: TradingConfig, episodes: int = 1000, save_path: str = "models"):
    """모델 학습"""
    print("=== 모델 학습 시작 ===")
    
    # 트레이너 생성
    trainer = TradingTrainer(config, market="KRW-BTC", device="cpu")
    
    # 모델 저장 디렉토리 생성
    os.makedirs(save_path, exist_ok=True)
    
    # 학습 실행
    training_results = trainer.train(episodes)
    
    print("=== 학습 완료 ===")
    print(f"최고 성과: {training_results.get('best_performance', 0):.2f}")
    print(f"모델 저장 경로: {save_path}")
    
    return training_results


def run_backtest(config: TradingConfig, model_path: str, start_date: str, end_date: str):
    """백테스팅 실행"""
    print("=== 백테스팅 시작 ===")
    
    # 환경 생성
    env = TradingEnvironment(config)
    obs, _ = env.reset()
    state_size = len(obs)
    
    # 에이전트 생성
    agent = DQNAgent(config, state_size)
    
    # 모델 로드
    if os.path.exists(model_path):
        agent.load_model(model_path)
        print(f"모델 로드 완료: {model_path}")
    else:
        print(f"모델 파일을 찾을 수 없습니다: {model_path}")
        return None
    
    # 백테스터 생성 및 실행
    backtester = Backtester(config)
    results = backtester.run_backtest(
        agent=agent,
        env=env,
        start_date=start_date,
        end_date=end_date
    )
    
    # 결과 출력
    print("=== 백테스팅 결과 ===")
    print(f"총 수익률: {results.total_return:.4f}")
    print(f"연간 수익률: {results.annual_return:.4f}")
    print(f"샤프 비율: {results.sharpe_ratio:.4f}")
    print(f"최대 낙폭: {results.max_drawdown:.4f}")
    print(f"승률: {results.win_rate:.4f}")
    print(f"총 거래 수: {results.total_trades}")
    
    # 결과 저장
    results_path = f"backtest_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    results_dict = {
        'total_return': results.total_return,
        'annual_return': results.annual_return,
        'sharpe_ratio': results.sharpe_ratio,
        'max_drawdown': results.max_drawdown,
        'win_rate': results.win_rate,
        'profit_factor': results.profit_factor,
        'total_trades': results.total_trades,
        'daily_returns': results.daily_returns[:100],  # 처음 100개만
        'equity_curve': results.equity_curve[:100]     # 처음 100개만
    }
    
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results_dict, f, indent=2, ensure_ascii=False)
    
    print(f"결과 저장: {results_path}")
    return results


def run_live_trading(config: TradingConfig, risk_config: RiskConfig, 
                    model_path: str, market: str = "KRW-BTC", duration_minutes: int = 60):
    """실시간 트레이딩 실행"""
    print("=== 실시간 트레이딩 시작 ===")
    print(f"시장: {market}")
    print(f"실행 시간: {duration_minutes}분")
    print("주의: 실제 돈이 거래됩니다!")
    
    # 확인 메시지
    confirm = input("실시간 트레이딩을 시작하시겠습니까? (yes/no): ")
    if confirm.lower() != 'yes':
        print("트레이딩 취소됨")
        return
    
    # 실시간 트레이더 생성
    trader = RealTimeTrader(
        config=config,
        risk_config=risk_config,
        model_path=model_path,
        market=market
    )
    
    try:
        # 트레이딩 시작
        trader.start_trading()
        
        # 지정된 시간 동안 실행
        import time
        time.sleep(duration_minutes * 60)
        
        # 최종 성과 보고서
        report = trader.get_performance_report()
        print("=== 최종 성과 보고서 ===")
        print(json.dumps(report, indent=2, ensure_ascii=False))
        
        # 보고서 저장
        report_path = f"trading_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"보고서 저장: {report_path}")
        
    except KeyboardInterrupt:
        print("\n사용자 중단")
    finally:
        trader.stop_trading()
        print("트레이딩 중지됨")


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="강화학습 트레이딩 시스템")
    parser.add_argument("--mode", choices=["train", "backtest", "live"], required=True,
                       help="실행 모드")
    parser.add_argument("--config", default="configs/default_config.json",
                       help="설정 파일 경로")
    parser.add_argument("--model", default="models/best_model.pth",
                       help="모델 파일 경로")
    parser.add_argument("--episodes", type=int, default=1000,
                       help="학습 에피소드 수")
    parser.add_argument("--start-date", default="2024-01-01",
                       help="백테스팅 시작 날짜 (YYYY-MM-DD)")
    parser.add_argument("--end-date", default="2024-12-31",
                       help="백테스팅 종료 날짜 (YYYY-MM-DD)")
    parser.add_argument("--market", default="KRW-BTC",
                       help="거래 마켓")
    parser.add_argument("--duration", type=int, default=60,
                       help="실시간 트레이딩 지속 시간 (분)")
    
    args = parser.parse_args()
    
    # 로깅 설정
    setup_logging()
    
    # 기본 설정
    config = TradingConfig(
        model_type="dqn",
        hidden_size=256,
        learning_rate=0.001,
        batch_size=32,
        memory_size=10000
    )
    
    # 실행 모드별 처리
    if args.mode == "train":
        train_model(config, args.episodes, "models")
    
    elif args.mode == "backtest":
        run_backtest(config, args.model, args.start_date, args.end_date)
    
    elif args.mode == "live":
        risk_config = RiskConfig(
            max_position_size=0.1,
            stop_loss_pct=0.05,
            take_profit_pct=0.1,
            max_daily_trades=10
        )
        run_live_trading(config, risk_config, args.model, args.market, args.duration)


if __name__ == "__main__":
    main()
