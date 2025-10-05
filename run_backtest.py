#!/usr/bin/env python3
"""
백테스팅 실행 스크립트

학습된 강화학습 모델의 성능을 과거 데이터로 검증합니다.

사용법:
    python run_backtest.py --model models/best_model.pth
    python run_backtest.py --model models/best_model.pth --start 2024-01-01 --end 2024-12-31
    python run_backtest.py --model models/best_model.pth --benchmark

최종 업데이트: 2025-10-05 23:35:00
"""

import argparse
import os
import sys
import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 프로젝트 루트 디렉토리를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from trading_env import TradingEnvironment, TradingConfig, ActionSpace
from rl_agent import RLAgent
from core.backtesting_engine import BacktestEngine
from core.performance_metrics import PerformanceMetrics
from core.visualization import TradingVisualizer


def setup_logging(log_dir: str = "logs") -> None:
    """로깅 설정"""
    os.makedirs(log_dir, exist_ok=True)
    
    log_filename = f'backtest_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    log_path = os.path.join(log_dir, log_filename)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )


def load_model(model_path: str, config: TradingConfig, state_size: int) -> RLAgent:
    """모델 로드
    
    Args:
        model_path: 모델 파일 경로
        config: 트레이딩 설정
        state_size: 상태 공간 크기
        
    Returns:
        로드된 에이전트
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")
    
    agent = RLAgent(config, state_size)
    agent.load_model(model_path)
    
    logging.info(f"모델 로드 완료: {model_path}")
    return agent


def run_backtest(
    model_path: str,
    config: TradingConfig,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    market: str = "KRW-BTC",
    include_benchmark: bool = True,
    save_dir: str = "results"
) -> Dict:
    """백테스팅 실행
    
    Args:
        model_path: 모델 파일 경로
        config: 트레이딩 설정
        start_date: 시작 날짜 (YYYY-MM-DD)
        end_date: 종료 날짜 (YYYY-MM-DD)
        market: 거래 마켓
        include_benchmark: 벤치마크 포함 여부
        save_dir: 결과 저장 디렉토리
        
    Returns:
        백테스팅 결과 딕셔너리
    """
    logging.info("=" * 60)
    logging.info("백테스팅 시작")
    logging.info("=" * 60)
    logging.info(f"모델: {model_path}")
    logging.info(f"마켓: {market}")
    logging.info(f"기간: {start_date or '최근'} ~ {end_date or '최근'}")
    
    # 결과 저장 디렉토리 생성
    os.makedirs(save_dir, exist_ok=True)
    
    try:
        # 환경 생성
        env = TradingEnvironment(config, market=market)
        obs, _ = env.reset()
        state_size = len(obs)
        
        # 모델 로드
        agent = load_model(model_path, config, state_size)
        
        # 백테스팅 엔진 생성
        engine = BacktestEngine(config)
        
        # 백테스트 실행
        logging.info("백테스트 실행 중...")
        result = engine.run(
            agent=agent,
            env=env,
            start_date=start_date,
            end_date=end_date
        )
        
        # 성과 지표 계산
        metrics = PerformanceMetrics()
        performance = metrics.calculate_all(result)
        
        # 벤치마크 비교
        benchmark_comparison = None
        if include_benchmark:
            logging.info("벤치마크 생성 중...")
            benchmark_result = engine.run_benchmark(env)
            benchmark_comparison = metrics.compare_with_benchmark(
                result, 
                benchmark_result
            )
        
        # 결과 출력
        print_results(performance, benchmark_comparison)
        
        # 시각화
        visualizer = TradingVisualizer()
        fig_path = os.path.join(save_dir, f"backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")

        visualizer.plot_backtest_results(
            result=result,
            benchmark=benchmark_result if include_benchmark else None,
            save_path=fig_path
        )
        
        logging.info(f"시각화 저장: {fig_path}")
        
        # 결과 저장
        results_dict = {
            'model_path': model_path,
            'market': market,
            'start_date': start_date,
            'end_date': end_date,
            'performance': performance,
            'benchmark_comparison': benchmark_comparison,
            'timestamp': datetime.now().isoformat()
        }
        
        results_path = os.path.join(save_dir, f"backtest_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results_dict, f, indent=2, ensure_ascii=False, default=str)
        
        logging.info(f"결과 저장: {results_path}")
        
        logging.info("=" * 60)
        logging.info("백테스팅 완료")
        logging.info("=" * 60)
        
        return results_dict
        
    except Exception as e:
        logging.error(f"백테스팅 중 오류 발생: {e}")
        import traceback
        logging.error(traceback.format_exc())
        raise


def print_results(performance: Dict, benchmark_comparison: Optional[Dict] = None):
    """결과 출력
    
    Args:
        performance: 성과 지표
        benchmark_comparison: 벤치마크 비교 결과
    """
    print("\n" + "=" * 60)
    print("백테스팅 결과")
    print("=" * 60)
    
    print("\n📊 수익률 지표:")
    print(f"  총 수익률:        {performance['total_return']:>10.2%}")
    print(f"  연환산 수익률:    {performance['annual_return']:>10.2%}")
    print(f"  최대 낙폭:        {performance['max_drawdown']:>10.2%}")
    
    print("\n📈 리스크 지표:")
    print(f"  샤프 비율:        {performance['sharpe_ratio']:>10.2f}")
    print(f"  변동성:           {performance.get('volatility', 0):>10.2%}")
    print(f"  Sortino 비율:     {performance.get('sortino_ratio', 0):>10.2f}")
    
    print("\n💰 거래 지표:")
    print(f"  총 거래 수:       {performance['total_trades']:>10d}")
    print(f"  승률:             {performance['win_rate']:>10.2%}")
    print(f"  Profit Factor:    {performance['profit_factor']:>10.2f}")
    print(f"  평균 거래 수익:   {performance.get('avg_trade_return', 0):>10.2%}")
    
    if benchmark_comparison:
        print("\n🎯 벤치마크 비교:")
        print(f"  에이전트 수익률:  {benchmark_comparison['agent_return']:>10.2%}")
        print(f"  벤치마크 수익률:  {benchmark_comparison['benchmark_return']:>10.2%}")
        print(f"  초과 수익률:      {benchmark_comparison['excess_return']:>10.2%}")
        print(f"  상대 성과:        {benchmark_comparison['outperformance']:>10.2%}")
    
    print("=" * 60)


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(
        description="백테스팅 실행",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예제:
  # 기본 백테스팅
  python run_backtest.py --model models/best_model.pth
  
  # 특정 기간 백테스팅
  python run_backtest.py --model models/best_model.pth --start 2024-01-01 --end 2024-12-31
  
  # 벤치마크 포함
  python run_backtest.py --model models/best_model.pth --benchmark
  
  # 특정 마켓에서 백테스팅
  python run_backtest.py --model models/best_model.pth --market KRW-ETH
        """
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="checkpoints/dqn_best.pth",
        help="모델 파일 경로 (기본: checkpoints/dqn_best.pth)"
    )
    parser.add_argument(
        "--start",
        type=str,
        default=None,
        help="시작 날짜 (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="종료 날짜 (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--market",
        type=str,
        default="KRW-BTC",
        help="거래 마켓 (기본: KRW-BTC)"
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="벤치마크 포함"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="설정 파일 경로 (JSON)"
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="results/backtests",
        help="결과 저장 디렉토리 (기본: results/backtests)"
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs",
        help="로그 파일 디렉토리 (기본: logs)"
    )
    
    args = parser.parse_args()
    
    # 로깅 설정
    setup_logging(args.log_dir)
    
    # 설정 로드
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        config = TradingConfig(**config_dict)
    else:
        config = TradingConfig()
    
    # 백테스팅 실행
    run_backtest(
        model_path=args.model,
        config=config,
        start_date=args.start,
        end_date=args.end,
        market=args.market,
        include_benchmark=args.benchmark,
        save_dir=args.save_dir
    )


if __name__ == "__main__":
    main()
