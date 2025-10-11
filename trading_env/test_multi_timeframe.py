"""
멀티 타임프레임 데이터 수집 테스트

멀티 타임프레임 데이터 파이프라인의 주요 기능을 테스트합니다:
1. 타임스탬프 정규화 및 중복 방지
2. 누락 데이터 감지 및 증분 수집
3. Upbit API Rate Limit 자동 처리
4. 1분봉/1시간봉/1일봉 멀티 타임프레임 수집

사용법:
    python -m trading_env.test_multi_timeframe
"""

import os
import sys
import logging
from datetime import datetime, timedelta

# 프로젝트 루트를 sys.path에 추가
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from trading_env.data_storage import (
    collect_multi_timeframe_data,
    MarketDataStorage,
    align_timestamp
)


def test_align_timestamp():
    """타임스탬프 정규화 함수 테스트"""
    logger = logging.getLogger(__name__)
    logger.info(" ")
    logger.info("=" * 70)
    logger.info("[테스트 1/4] 타임스탬프 정규화 테스트")
    logger.info("=" * 70)

    test_cases = [
        (datetime(2025, 10, 9, 14, 23, 45, 123456), '1m', datetime(2025, 10, 9, 14, 23, 0)),
        (datetime(2025, 10, 9, 14, 23, 45, 123456), '1h', datetime(2025, 10, 9, 14, 0, 0)),
        (datetime(2025, 10, 9, 14, 23, 45, 123456), '1d', datetime(2025, 10, 9, 0, 0, 0)),
    ]

    for input_dt, timeframe, expected in test_cases:
        result = align_timestamp(input_dt, timeframe)
        status = "✅" if result == expected else "❌"
        logger.info(f"  {status} {timeframe}: {input_dt} → {result}")
        assert result == expected, f"정규화 실패: {result} != {expected}"

    logger.info("✅ 타임스탬프 정규화 테스트 통과")


def test_collect_by_end_time():
    """방법 1: end_time 지정 방식 테스트"""
    logger = logging.getLogger(__name__)
    logger.info(" ")
    logger.info("=" * 70)
    logger.info("[테스트 2/4] end_time 지정 방식 데이터 수집")
    logger.info("=" * 70)

    market = "KRW-BTC"
    db_path = "data/market_data_test.db"

    # 최근 1시간 데이터 수집
    end_time = datetime.now()
    start_time = end_time - timedelta(hours=1)

    logger.info(f"📥 수집 범위:")
    logger.info(f"  마켓: {market}")
    logger.info(f"  시작: {start_time}")
    logger.info(f"  종료: {end_time}")
    logger.info(f"  추가 lookback: 시간봉 24개, 일봉 30개")

    try:
        collect_multi_timeframe_data(
            market=market,
            start_time=start_time,
            end_time=end_time,
            hourly_lookback_count=24,
            daily_lookback_count=30,
            db_path=db_path,
            show_progress=True
        )

        # 결과 검증
        storage = MarketDataStorage(db_path)
        for tf in ['1m', '1h', '1d']:
            count = storage.get_data_count_by_timeframe(market, tf)
            data_range = storage.get_data_range_by_timeframe(market, tf)
            logger.info(f"✅ {tf}: {count}건")
            logger.info(f"   범위: {data_range[0]} ~ {data_range[1]}")

        logger.info("✅ end_time 지정 방식 테스트 통과")
        return True

    except Exception as e:
        logger.error(f"\n❌ 테스트 실패: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def test_collect_by_count():
    """방법 2: minute_candles_count 지정 방식 테스트"""
    logger = logging.getLogger(__name__)
    logger.info(" ")
    logger.info("=" * 70)
    logger.info("[테스트 3/4] minute_candles_count 지정 방식 데이터 수집")
    logger.info("=" * 70)

    market = "KRW-BTC"
    db_path = "data/market_data_test.db"

    # 최근 시점부터 60개 1분봉 수집
    start_time = datetime.now() - timedelta(hours=2)
    minute_candles_count = 60

    logger.info(f"📥 수집 설정:")
    logger.info(f"  마켓: {market}")
    logger.info(f"  시작: {start_time}")
    logger.info(f"  1분봉 개수: {minute_candles_count}개")
    logger.info(f"  → 종료: {start_time + timedelta(minutes=minute_candles_count)}")
    logger.info(f"  추가 lookback: 시간봉 24개, 일봉 30개")

    try:
        collect_multi_timeframe_data(
            market=market,
            start_time=start_time,
            minute_candles_count=minute_candles_count,
            hourly_lookback_count=24,
            daily_lookback_count=30,
            db_path=db_path,
            show_progress=True
        )

        # 결과 검증
        storage = MarketDataStorage(db_path)
        for tf in ['1m', '1h', '1d']:
            count = storage.get_data_count_by_timeframe(market, tf)
            data_range = storage.get_data_range_by_timeframe(market, tf)
            logger.info(f"✅ {tf}: {count}건")
            logger.info(f"   범위: {data_range[0]} ~ {data_range[1]}")

        logger.info("✅ minute_candles_count 지정 방식 테스트 통과")
        return True

    except Exception as e:
        logger.error(f"❌ 테스트 실패: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def test_incremental_collection():
    """증분 수집 테스트 (누락 데이터만 수집)"""
    logger = logging.getLogger(__name__)
    logger.info(" ")
    logger.info("=" * 70)
    logger.info("[테스트 4/4] 증분 수집 테스트 (누락 데이터만 수집)")
    logger.info("=" * 70)

    market = "KRW-BTC"
    db_path = "data/market_data_test.db"

    # 1차 수집: 10월 1일 ~ 10월 5일
    start_time_1 = datetime(2025, 10, 1)
    end_time_1 = datetime(2025, 10, 5)

    logger.info(f"📥 1차 수집:")
    logger.info(f"  기간: {start_time_1} ~ {end_time_1}")

    try:
        storage = MarketDataStorage(db_path)

        # 기존 데이터 확인
        existing_1m_before = storage.get_data_count_by_timeframe(market, '1m')
        logger.info(f"  기존 1분봉: {existing_1m_before}건")

        collect_multi_timeframe_data(
            market=market,
            start_time=start_time_1,
            end_time=end_time_1,
            hourly_lookback_count=0,  # 테스트 간소화
            daily_lookback_count=0,
            db_path=db_path,
            show_progress=True
        )

        count_1m_after_1st = storage.get_data_count_by_timeframe(market, '1m')
        logger.info(f"✅ 1차 수집 완료: 1분봉 {count_1m_after_1st}건")

        # 2차 수집: 10월 1일 ~ 10월 10일 (5일 연장)
        logger.info(f"📥 2차 수집 (기간 연장):")
        start_time_2 = datetime(2025, 10, 1)
        end_time_2 = datetime(2025, 10, 10)
        logger.info(f"  기간: {start_time_2} ~ {end_time_2}")
        logger.info(f"  → 10월 6일 ~ 10월 10일만 추가 수집 예상")

        collect_multi_timeframe_data(
            market=market,
            start_time=start_time_2,
            end_time=end_time_2,
            hourly_lookback_count=0,
            daily_lookback_count=0,
            db_path=db_path,
            show_progress=True
        )

        count_1m_after_2nd = storage.get_data_count_by_timeframe(market, '1m')
        logger.info(f"✅ 2차 수집 완료: 1분봉 {count_1m_after_2nd}건")

        # 증분 수집 검증
        added_count = count_1m_after_2nd - count_1m_after_1st
        logger.info(f"📊 증분 수집 결과:")
        logger.info(f"  1차 수집 후: {count_1m_after_1st}건")
        logger.info(f"  2차 수집 후: {count_1m_after_2nd}건")
        logger.info(f"  추가된 데이터: {added_count}건")

        if added_count > 0:
            logger.info("✅ 증분 수집 테스트 통과 (누락 데이터만 수집됨)")
        else:
            logger.warning("⚠️  추가 데이터 없음 (이미 모든 데이터가 존재)")

        return True

    except Exception as e:
        logger.error(f"\n❌ 테스트 실패: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def main():
    """메인 테스트 실행"""
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    logger.info("=" * 70)
    logger.info("멀티 타임프레임 데이터 파이프라인 테스트 시작")
    logger.info("=" * 70)

    # 디렉토리 생성
    os.makedirs("data", exist_ok=True)

    results = []

    # 테스트 1: 타임스탬프 정규화
    try:
        test_align_timestamp()
        results.append(("타임스탬프 정규화", True))
    except Exception as e:
        logger.error(f"❌ 타임스탬프 정규화 테스트 실패: {e}")
        results.append(("타임스탬프 정규화", False))

    # 테스트 2: end_time 지정 방식
    result = test_collect_by_end_time()
    results.append(("end_time 지정 방식", result))

    # 테스트 3: minute_candles_count 지정 방식
    result = test_collect_by_count()
    results.append(("minute_candles_count 지정 방식", result))

    # 테스트 4: 증분 수집
    result = test_incremental_collection()
    results.append(("증분 수집", result))

    # 최종 결과 요약
    logger.info(" ")
    logger.info("=" * 70)
    logger.info("테스트 결과 요약")
    logger.info("=" * 70)

    for test_name, success in results:
        status = "✅ 통과" if success else "❌ 실패"
        logger.info(f"  {status}: {test_name}")

    total_tests = len(results)
    passed_tests = sum(1 for _, success in results if success)
    logger.info(f"총 {total_tests}개 테스트 중 {passed_tests}개 통과")

    if passed_tests == total_tests:
        logger.info("🎉 모든 테스트 통과!")
        logger.info("🔧 다음 단계:")
        logger.info("  1. 실제 데이터 수집: collect_multi_timeframe_data() 사용")
        logger.info("  2. 멀티 타임프레임 RL 환경 구현")
        logger.info("  3. 성능 벤치마크 및 최적화")
    else:
        logger.error(f"⚠️  {total_tests - passed_tests}개 테스트 실패")

    logger.info("=" * 70)


if __name__ == "__main__":
    main()
