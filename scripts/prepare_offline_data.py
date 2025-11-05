#!/usr/bin/env python3
"""
오프라인 학습용 데이터 준비 스크립트

Upbit 데이터 수집 → 기술적 지표 계산 → 특성 추출 → SQLite 저장
"""

import argparse
import logging
import sys
from pathlib import Path

# 프로젝트 루트를 path에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from trading_env.env_pipeline import prepare_offline_data


def main():
    parser = argparse.ArgumentParser(
        description="오프라인 학습용 데이터 준비",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예제:
  # 기본 (7일 데이터)
  python scripts/prepare_offline_data.py --market KRW-BTC

  # 30일 데이터
  python scripts/prepare_offline_data.py --market KRW-BTC --days 30

  # 여러 마켓 준비
  python scripts/prepare_offline_data.py --market KRW-BTC --days 30
  python scripts/prepare_offline_data.py --market KRW-ETH --days 30
        """
    )

    parser.add_argument(
        "--market",
        type=str,
        default="KRW-BTC",
        help="마켓 코드 (기본: KRW-BTC)"
    )
    parser.add_argument(
        "--days",
        type=int,
        default=7,
        help="수집할 일수 (기본: 7일)"
    )
    parser.add_argument(
        "--db",
        type=str,
        default="data/market_data.db",
        help="SQLite 데이터베이스 경로 (기본: data/market_data.db)"
    )
    parser.add_argument(
        "--normalization",
        type=str,
        default="robust",
        choices=["standard", "minmax", "robust"],
        help="정규화 방법 (기본: robust)"
    )
    parser.add_argument(
        "--no-ssl",
        action="store_true",
        help="SSL 특성 제외"
    )

    args = parser.parse_args()

    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # 데이터 준비 실행
    prepare_offline_data(
        market=args.market,
        days=args.days,
        db_path=args.db,
        normalization_method=args.normalization,
        include_ssl=not args.no_ssl
    )


if __name__ == "__main__":
    main()
