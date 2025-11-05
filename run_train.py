#!/usr/bin/env python3
"""
강화학습 모델 학습 스크립트

강화학습 에이전트를 학습하고 모델을 저장합니다.

사용법:
    python run_train.py --episodes 1000 --market KRW-BTC
    python run_train.py --config configs/dqn_config.json
    python run_train.py --resume models/checkpoint.pth

최종 업데이트: 2025-10-05 23:30:00
"""

import argparse
import os
import sys
import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional
import torch
import numpy as np

# 프로젝트 루트 디렉토리를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from trading_env import TradingEnvironment, TradingConfig
from rl_agent import RLAgent, TradingTrainer
from models import create_model


def setup_logging(log_dir: str = "logs") -> None:
    """로깅 설정
    
    Args:
        log_dir: 로그 파일 저장 디렉토리
    """
    os.makedirs(log_dir, exist_ok=True)
    
    log_filename = f'train_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
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


def load_config(config_path: Optional[str] = None) -> TradingConfig:
    """설정 파일 로드
    
    Args:
        config_path: 설정 파일 경로
        
    Returns:
        TradingConfig 객체
    """
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        
        # TradingConfig로 변환
        config = TradingConfig(**config_dict)
        logging.info(f"설정 파일 로드: {config_path}")
    else:
        # 기본 설정
        config = TradingConfig(
            model_type="dqn",
            hidden_size=256,
            learning_rate=0.001,
            batch_size=32,
            memory_size=10000,
            epsilon_start=1.0,
            epsilon_end=0.01,
            epsilon_decay=0.995,
            target_update=100
        )
        logging.info("기본 설정 사용")
    
    return config


def save_config(config: TradingConfig, save_dir: str) -> None:
    """설정 저장
    
    Args:
        config: TradingConfig 객체
        save_dir: 저장 디렉토리
    """
    config_path = os.path.join(save_dir, "train_config.json")
    
    # dataclass를 dict로 변환
    config_dict = {
        'model_type': config.model_type,
        'hidden_size': config.hidden_size,
        'num_layers': config.num_layers,
        'learning_rate': config.learning_rate,
        'batch_size': config.batch_size,
        'memory_size': config.memory_size,
        'epsilon_start': config.epsilon_start,
        'epsilon_end': config.epsilon_end,
        'epsilon_decay': config.epsilon_decay,
        'target_update': config.target_update,
        'initial_balance': config.initial_balance,
        'transaction_fee': config.transaction_fee,
        'lookback_window': config.lookback_window
    }
    
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)
    
    logging.info(f"설정 저장: {config_path}")


def train_model(
    config: TradingConfig,
    episodes: int = 1000,
    save_dir: str = "models/saved",
    market: str = "KRW-BTC",
    resume_path: Optional[str] = None,
    save_interval: int = 100,
    eval_interval: int = 50,
    db_path: Optional[str] = None
) -> Dict:
    """모델 학습

    Args:
        config: 트레이딩 설정
        episodes: 학습 에피소드 수
        save_dir: 모델 저장 디렉토리
        market: 거래 마켓
        resume_path: 재개할 모델 경로
        save_interval: 모델 저장 간격
        eval_interval: 평가 간격
        db_path: SQLite 데이터베이스 경로 (None이면 Upbit API 사용)

    Returns:
        학습 결과 딕셔너리
    """
    logging.info("=" * 50)
    logging.info("모델 학습 시작")
    logging.info("=" * 50)
    logging.info(f"에피소드: {episodes}")
    logging.info(f"마켓: {market}")
    logging.info(f"모델 타입: {config.model_type}")
    logging.info(f"학습률: {config.learning_rate}")
    logging.info(f"배치 크기: {config.batch_size}")
    if db_path:
        logging.info(f"데이터 소스: SQLite ({db_path})")
    else:
        logging.info(f"데이터 소스: Upbit API (실시간)")

    # 저장 디렉토리 생성
    os.makedirs(save_dir, exist_ok=True)
    save_config(config, save_dir)

    # 디바이스 설정
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"디바이스: {device}")

    try:
        # 모드 결정
        training_mode = "offline" if db_path else "realtime"

        # 트레이너 생성 (SQLite 데이터 또는 API 데이터 사용)
        trainer = TradingTrainer(
            config=config,
            market=market,
            device=device,
            db_path=db_path,  # SQLite 데이터베이스 경로 전달
            mode=training_mode,
            cache_enabled=True
        )

        # 체크포인트에서 재개
        if resume_path and os.path.exists(resume_path):
            trainer.agent.load_model(resume_path)
            logging.info(f"체크포인트 로드: {resume_path}")

        # 학습 실행 (올바른 파라미터 사용)
        training_results = trainer.train(
            num_episodes=episodes,
            save_frequency=save_interval,
            eval_frequency=eval_interval
        )
        
        # 최종 모델 저장
        final_model_path = os.path.join(save_dir, "final_model.pth")
        trainer.agent.save_model(final_model_path)
        logging.info(f"최종 모델 저장: {final_model_path}")
        
        # 최고 성능 모델 저장
        if 'best_model_state' in training_results:
            best_model_path = os.path.join(save_dir, "best_model.pth")
            torch.save(training_results['best_model_state'], best_model_path)
            logging.info(f"최고 성능 모델 저장: {best_model_path}")
        
        # 학습 결과 저장
        results_path = os.path.join(save_dir, "training_results.json")
        results_dict = {
            'episodes': episodes,
            'best_reward': training_results.get('best_reward', 0),
            'avg_reward': training_results.get('avg_reward', 0),
            'final_epsilon': training_results.get('final_epsilon', 0),
            'training_time': training_results.get('training_time', 0),
            'timestamp': datetime.now().isoformat()
        }
        
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results_dict, f, indent=2, ensure_ascii=False)
        
        logging.info("=" * 50)
        logging.info("학습 완료")
        logging.info("=" * 50)
        logging.info(f"최고 보상: {training_results.get('best_reward', 0):.2f}")
        logging.info(f"평균 보상: {training_results.get('avg_reward', 0):.2f}")
        logging.info(f"최종 탐험률: {training_results.get('final_epsilon', 0):.4f}")
        logging.info(f"학습 시간: {training_results.get('training_time', 0):.2f}초")
        logging.info(f"모델 저장 경로: {save_dir}")
        
        return training_results
        
    except KeyboardInterrupt:
        logging.info("\n사용자에 의해 학습 중단됨")
        
        # 중단 시점 모델 저장
        interrupted_path = os.path.join(save_dir, f"interrupted_ep{trainer.episode}.pth")
        trainer.agent.save_model(interrupted_path)
        logging.info(f"중단 시점 모델 저장: {interrupted_path}")
        
        return {'status': 'interrupted', 'episode': trainer.episode}
        
    except Exception as e:
        logging.error(f"학습 중 오류 발생: {e}")
        import traceback
        logging.error(traceback.format_exc())
        raise


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(
        description="강화학습 모델 학습",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예제:
  # 기본 학습 (1000 에피소드)
  python run_train.py
  
  # 특정 마켓에서 학습
  python run_train.py --market KRW-ETH --episodes 2000
  
  # 설정 파일 사용
  python run_train.py --config configs/dqn_config.json
  
  # 체크포인트에서 재개
  python run_train.py --resume models/checkpoint_ep500.pth
  
  # GPU 사용 (자동 감지)
  python run_train.py --episodes 5000
        """
    )
    
    parser.add_argument(
        "--episodes", 
        type=int, 
        default=1000,
        help="학습 에피소드 수 (기본: 1000)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="설정 파일 경로 (JSON)"
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="models/saved",
        help="모델 저장 디렉토리 (기본: models/saved)"
    )
    parser.add_argument(
        "--market",
        type=str,
        default="KRW-BTC",
        help="거래 마켓 (기본: KRW-BTC)"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="재개할 모델 체크포인트 경로"
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=100,
        help="모델 저장 간격 (에피소드) (기본: 100)"
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=50,
        help="평가 간격 (에피소드) (기본: 50)"
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs",
        help="로그 파일 디렉토리 (기본: logs)"
    )
    parser.add_argument(
        "--db",
        type=str,
        default=None,
        help="SQLite 데이터베이스 경로 (지정 시 Upbit API 대신 SQLite 사용)"
    )
    parser.add_argument(
        "--collect-data",
        action="store_true",
        help="학습 전에 데이터를 수집하여 SQLite에 저장"
    )
    parser.add_argument(
        "--data-count",
        type=int,
        default=1000,
        help="수집할 데이터 개수 (기본: 1000)"
    )

    args = parser.parse_args()

    # 로깅 설정
    setup_logging(args.log_dir)

    # 데이터 수집 + 특성 추출 (필요 시)
    if args.collect_data:
        from trading_env.env_pipeline import prepare_offline_data
        db_path = args.db or "data/market_data.db"
        days = args.data_count // (24 * 60)  # 1분봉 기준 일수 계산
        if days < 1:
            days = 7  # 최소 7일

        logging.info(f"오프라인 데이터 준비 시작: {args.market} ({days}일)")
        prepare_offline_data(
            market=args.market,
            days=days,
            db_path=db_path
        )
        logging.info("데이터 준비 완료")

    # 설정 로드
    config = load_config(args.config)

    # 학습 실행
    train_model(
        config=config,
        episodes=args.episodes,
        save_dir=args.model_dir,
        market=args.market,
        resume_path=args.resume,
        save_interval=args.save_interval,
        eval_interval=args.eval_interval,
        db_path=args.db  # SQLite 데이터베이스 경로 전달
    )


if __name__ == "__main__":
    main()
