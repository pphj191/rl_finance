#!/usr/bin/env python3
"""
실시간 트레이딩 실행 스크립트

학습된 강화학습 모델로 실시간 트레이딩을 수행합니다.

⚠️  주의: 실제 돈이 거래됩니다!

사용법:
    python run_realtime_trading.py --model models/best_model.pth --duration 60
    python run_realtime_trading.py --model models/best_model.pth --market KRW-ETH --dry-run
    python run_realtime_trading.py --model models/best_model.pth --config configs/risk_config.json

최종 업데이트: 2025-10-05 23:40:00
"""

import argparse
import os
import sys
import logging
import json
import time
import signal
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional
import threading

# 프로젝트 루트 디렉토리를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from trading_env import TradingConfig
from rl_agent import RLAgent
from upbit_api import UpbitAPI
from core.realtime_trader import RealtimeTrader, RiskConfig, TradingMonitor


# 전역 변수로 트레이더 인스턴스 관리
_trader: Optional[RealtimeTrader] = None


def setup_logging(log_dir: str = "logs") -> None:
    """로깅 설정"""
    os.makedirs(log_dir, exist_ok=True)
    
    log_filename = f'realtime_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    log_path = os.path.join(log_dir, log_filename)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )
    
    logging.info(f"로그 파일: {log_path}")


def load_risk_config(config_path: Optional[str] = None) -> RiskConfig:
    """리스크 설정 로드
    
    Args:
        config_path: 리스크 설정 파일 경로
        
    Returns:
        RiskConfig 객체
    """
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        
        risk_config = RiskConfig(**config_dict)
        logging.info(f"리스크 설정 로드: {config_path}")
    else:
        # 기본 리스크 설정 (보수적)
        risk_config = RiskConfig(
            max_position_size=0.05,      # 총 자산의 5%
            stop_loss_pct=0.03,          # 3% 손절
            take_profit_pct=0.07,        # 7% 익절
            max_daily_trades=5,          # 일일 최대 5회 거래
            min_trade_interval=600,      # 최소 10분 간격
            max_drawdown_pct=0.15        # 최대 15% 낙폭
        )
        logging.info("기본 리스크 설정 사용 (보수적)")
    
    return risk_config


def signal_handler(signum, frame):
    """시그널 핸들러 (Ctrl+C 처리)"""
    global _trader
    
    logging.info("\n프로그램 종료 신호 수신")
    
    if _trader:
        _trader.stop_trading()
    
    sys.exit(0)


def run_realtime_trading(
    model_path: str,
    config: TradingConfig,
    risk_config: RiskConfig,
    market: str = "KRW-BTC",
    duration_minutes: int = 60,
    dry_run: bool = False,
    update_interval: int = 60,
    save_dir: str = "results/realtime"
) -> Dict:
    """실시간 트레이딩 실행
    
    Args:
        model_path: 모델 파일 경로
        config: 트레이딩 설정
        risk_config: 리스크 관리 설정
        market: 거래 마켓
        duration_minutes: 실행 시간 (분)
        dry_run: 모의 거래 모드
        update_interval: 업데이트 간격 (초)
        save_dir: 결과 저장 디렉토리
        
    Returns:
        트레이딩 결과 딕셔너리
    """
    global _trader
    
    logging.info("=" * 60)
    logging.info("실시간 트레이딩 시작")
    logging.info("=" * 60)
    logging.info(f"모델: {model_path}")
    logging.info(f"마켓: {market}")
    logging.info(f"실행 시간: {duration_minutes}분")
    logging.info(f"업데이트 간격: {update_interval}초")
    logging.info(f"모드: {'모의 거래 (Dry Run)' if dry_run else '⚠️  실제 거래'}")
    
    # 결과 저장 디렉토리 생성
    os.makedirs(save_dir, exist_ok=True)
    
    # 실제 거래 확인
    if not dry_run:
        print("\n⚠️  경고: 실제 돈으로 거래가 실행됩니다!")
        print("=" * 60)
        confirm = input("실시간 트레이딩을 시작하시겠습니까? (yes/no): ")
        
        if confirm.lower() != 'yes':
            logging.info("사용자에 의해 트레이딩 취소됨")
            return {'status': 'cancelled'}
    
    try:
        # 트레이더 생성
        _trader = RealtimeTrader(
            config=config,
            risk_config=risk_config,
            model_path=model_path,
            market=market,
            dry_run=dry_run
        )
        
        # 시그널 핸들러 등록
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # 트레이딩 시작
        logging.info("트레이딩 시작...")
        _trader.start_trading(update_interval=update_interval)
        
        # 모니터링
        monitor = TradingMonitor(_trader)
        monitor.start()
        
        # 지정된 시간 동안 실행
        end_time = time.time() + (duration_minutes * 60)
        
        while time.time() < end_time and _trader.is_trading:
            time.sleep(10)  # 10초마다 체크
            
            # 진행 상황 출력
            remaining_minutes = (end_time - time.time()) / 60
            if remaining_minutes > 0:
                logging.info(f"남은 시간: {remaining_minutes:.1f}분")
        
        # 트레이딩 중지
        _trader.stop_trading()
        monitor.stop()
        
        # 성과 보고서 생성
        report = _trader.get_performance_report()
        
        # 결과 출력
        print_trading_report(report)
        
        # 보고서 저장
        report_path = os.path.join(
            save_dir, 
            f"trading_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        
        logging.info(f"보고서 저장: {report_path}")
        
        logging.info("=" * 60)
        logging.info("실시간 트레이딩 완료")
        logging.info("=" * 60)
        
        return report
        
    except KeyboardInterrupt:
        logging.info("\n사용자에 의해 트레이딩 중단됨")
        
        if _trader:
            _trader.stop_trading()
        
        return {'status': 'interrupted'}
        
    except Exception as e:
        logging.error(f"트레이딩 중 오류 발생: {e}")
        import traceback
        logging.error(traceback.format_exc())
        
        if _trader:
            _trader.stop_trading()
        
        raise


def print_trading_report(report: Dict):
    """트레이딩 보고서 출력
    
    Args:
        report: 트레이딩 보고서 딕셔너리
    """
    print("\n" + "=" * 60)
    print("실시간 트레이딩 성과 보고서")
    print("=" * 60)
    
    print("\n📊 수익 현황:")
    print(f"  초기 자산:        {report.get('initial_balance', 0):>15,.0f}원")
    print(f"  최종 자산:        {report.get('final_balance', 0):>15,.0f}원")
    print(f"  수익/손실:        {report.get('total_pnl', 0):>15,.0f}원")
    print(f"  수익률:           {report.get('return_pct', 0):>14.2f}%")
    
    print("\n💰 거래 통계:")
    print(f"  총 거래 수:       {report.get('total_trades', 0):>15d}")
    print(f"  매수 거래:        {report.get('buy_trades', 0):>15d}")
    print(f"  매도 거래:        {report.get('sell_trades', 0):>15d}")
    print(f"  승리 거래:        {report.get('winning_trades', 0):>15d}")
    print(f"  패배 거래:        {report.get('losing_trades', 0):>15d}")
    print(f"  승률:             {report.get('win_rate', 0):>14.2f}%")
    
    print("\n⚠️  리스크 지표:")
    print(f"  최대 낙폭:        {report.get('max_drawdown', 0):>14.2f}%")
    print(f"  손절 실행:        {report.get('stop_loss_triggered', 0):>15d}회")
    print(f"  익절 실행:        {report.get('take_profit_triggered', 0):>15d}회")
    
    print("\n⏱️  실행 정보:")
    print(f"  시작 시간:        {report.get('start_time', 'N/A')}")
    print(f"  종료 시간:        {report.get('end_time', 'N/A')}")
    print(f"  실행 시간:        {report.get('duration_minutes', 0):.1f}분")
    
    print("=" * 60)


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(
        description="실시간 트레이딩 실행",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예제:
  # 모의 거래 (Dry Run)
  python run_realtime_trading.py --model models/best_model.pth --dry-run
  
  # 실제 거래 (1시간)
  python run_realtime_trading.py --model models/best_model.pth --duration 60
  
  # 특정 마켓에서 거래
  python run_realtime_trading.py --model models/best_model.pth --market KRW-ETH
  
  # 리스크 설정 파일 사용
  python run_realtime_trading.py --model models/best_model.pth --risk-config configs/risk.json
        """
    )
    
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="모델 파일 경로"
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=60,
        help="실행 시간 (분) (기본: 60)"
    )
    parser.add_argument(
        "--market",
        type=str,
        default="KRW-BTC",
        help="거래 마켓 (기본: KRW-BTC)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="모의 거래 모드 (실제 주문 없음)"
    )
    parser.add_argument(
        "--update-interval",
        type=int,
        default=60,
        help="업데이트 간격 (초) (기본: 60)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="트레이딩 설정 파일 경로 (JSON)"
    )
    parser.add_argument(
        "--risk-config",
        type=str,
        default=None,
        help="리스크 설정 파일 경로 (JSON)"
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="results/realtime",
        help="결과 저장 디렉토리 (기본: results/realtime)"
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
    
    # 트레이딩 설정 로드
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        config = TradingConfig(**config_dict)
    else:
        config = TradingConfig()
    
    # 리스크 설정 로드
    risk_config = load_risk_config(args.risk_config)
    
    # 실시간 트레이딩 실행
    run_realtime_trading(
        model_path=args.model,
        config=config,
        risk_config=risk_config,
        market=args.market,
        duration_minutes=args.duration,
        dry_run=args.dry_run,
        update_interval=args.update_interval,
        save_dir=args.save_dir
    )


if __name__ == "__main__":
    main()
