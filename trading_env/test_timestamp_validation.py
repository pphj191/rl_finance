#!/usr/bin/env python3
"""
타임스탬프 검증 테스트 스크립트

Upbit API의 timezone 처리 및 타임스탬프 정규화를 검증합니다.
"""

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# 프로젝트 루트를 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from trading_env.market_data import UpbitDataCollector
from trading_env.data_storage import align_timestamp, collect_multi_timeframe_data
import sqlite3


def test_upbit_api_timezone():
    """Upbit API가 반환하는 타임스탬프의 timezone 확인"""
    print("=" * 80)
    print("1. Upbit API Timezone 테스트")
    print("=" * 80)

    # UpbitDataCollector는 market과 interval을 생성자에서 받지 않음
    from upbit_api.upbit_api import UpbitAPI

    upbit = UpbitAPI()

    # 최근 1분봉 1개만 가져오기
    candles = upbit.get_candles_minutes(market="KRW-BTC", unit=1, count=1)


    if not candles:
        print("❌ 데이터를 가져올 수 없습니다.")
        return False

    candle = candles[0]

    print(f"\n📊 원본 데이터 (Upbit API 응답):")
    print(f"  market: {candle.get('market')}")
    print(f"  candle_date_time_utc: {candle.get('candle_date_time_utc')}")
    print(f"  candle_date_time_kst: {candle.get('candle_date_time_kst')}")
    print(f"  opening_price: {candle.get('opening_price')}")
    print(f"  high_price: {candle.get('high_price')}")
    print(f"  low_price: {candle.get('low_price')}")
    print(f"  trade_price: {candle.get('trade_price')}")

    # UTC 시간 파싱
    utc_str = candle['candle_date_time_utc']
    # Upbit API는 "2025-10-12T02:00:00" 형식 반환
    utc_time = datetime.fromisoformat(utc_str.replace('Z', ''))

    # KST 시간 파싱
    kst_str = candle['candle_date_time_kst']
    kst_time = datetime.fromisoformat(kst_str)

    print(f"\n🕐 파싱된 시간:")
    print(f"  UTC: {utc_time}")
    print(f"  KST: {kst_time}")
    print(f"  차이: {(kst_time - utc_time).total_seconds() / 3600}시간")

    # 현재 시간
    now_kst = datetime.now()

    print(f"\n⏰ 현재 시간:")
    print(f"  현재 (KST): {now_kst}")

    # 미래 시간 체크
    is_future_kst = kst_time > now_kst

    # UTC-KST 차이가 정확히 9시간인지 확인
    time_diff_hours = (kst_time - utc_time).total_seconds() / 3600
    is_correct_diff = abs(time_diff_hours - 9.0) < 0.01

    print(f"\n✅ 검증 결과:")
    print(f"  KST 시간이 미래인가? {is_future_kst} {'❌ 문제!' if is_future_kst else '✓'}")
    print(f"  UTC-KST 차이가 9시간인가? {is_correct_diff} {'✓' if is_correct_diff else f'❌ 실제: {time_diff_hours:.1f}시간'}")

    # 현재 시간과 차이 (5분 이내가 정상)
    time_diff = abs((now_kst - kst_time).total_seconds())
    is_reasonable = time_diff < 300  # 5분
    print(f"  현재 시간과 차이: {time_diff:.0f}초 {'✓' if is_reasonable else '⚠️ 5분 이상 차이'}")

    return not is_future_kst and is_correct_diff and is_reasonable


def test_timestamp_alignment():
    """타임스탬프 정규화 함수 테스트"""
    print("\n" + "=" * 80)
    print("2. 타임스탬프 정규화 테스트")
    print("=" * 80)

    test_cases = [
        ("1m", datetime(2025, 10, 12, 14, 23, 45, 123456)),
        ("1h", datetime(2025, 10, 12, 14, 23, 45, 123456)),
        ("1d", datetime(2025, 10, 12, 14, 23, 45, 123456)),
    ]

    print("\n원본 시간: 2025-10-12 14:23:45.123456\n")

    all_passed = True
    for timeframe, dt in test_cases:
        aligned = align_timestamp(dt, timeframe)

        expected = {
            "1m": datetime(2025, 10, 12, 14, 23, 0),
            "1h": datetime(2025, 10, 12, 14, 0, 0),
            "1d": datetime(2025, 10, 12, 0, 0, 0),
        }[timeframe]

        passed = aligned == expected
        all_passed &= passed

        print(f"  {timeframe}: {aligned} {'✓' if passed else '❌ 예상: ' + str(expected)}")

    return all_passed


def test_collected_data_validation(market="KRW-BTC", db_path="data/test_timestamp.db"):
    """실제 수집된 데이터의 타임스탬프 검증"""
    print("\n" + "=" * 80)
    print("3. 실제 수집 데이터 검증 테스트")
    print("=" * 80)

    # 테스트용 DB에 최근 10분치 데이터 수집
    now = datetime.now()
    start_time = now - timedelta(minutes=10)

    print(f"\n📥 데이터 수집 중...")
    print(f"  마켓: {market}")
    print(f"  기간: {start_time} ~ {now}")

    try:
        collect_multi_timeframe_data(
            market=market,
            start_time=start_time,
            minute_candles_count=10,
            hourly_lookback_count=0,
            daily_lookback_count=0,
            db_path=db_path,
            show_progress=False
        )

        # DB에서 데이터 검증
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        issues = []

        for timeframe in ['1m', '1h', '1d']:
            cursor.execute(f"""
                SELECT timestamp, open, high, low, close
                FROM ohlcv_{timeframe}
                WHERE market = ?
                ORDER BY timestamp DESC
                LIMIT 10
            """, (market,))

            rows = cursor.fetchall()

            print(f"\n📊 {timeframe} 테이블 (최근 10개):")

            if not rows:
                print(f"  (데이터 없음)")
                continue

            for row in rows[:3]:  # 최근 3개만 출력
                ts_str, open_p, high_p, low_p, close_p = row

                # SQLite에서 읽은 timestamp는 문자열일 수도, datetime 객체일 수도 있음
                if isinstance(ts_str, str):
                    ts = datetime.fromisoformat(ts_str)
                else:
                    ts = ts_str

                # 미래 시간 체크
                is_future = ts > now

                # 타임스탬프 정규화 체크
                expected_aligned = align_timestamp(ts, timeframe)
                is_aligned = ts == expected_aligned

                status = "✓"
                if is_future:
                    status = "❌ 미래 시간!"
                    issues.append(f"{timeframe}: {ts_str}은 미래 시간입니다.")
                elif not is_aligned:
                    status = f"⚠️  정규화 안됨 (예상: {expected_aligned})"
                    issues.append(f"{timeframe}: {ts_str}이 정규화되지 않았습니다.")

                print(f"    {ts_str} | O:{open_p:>10,.0f} H:{high_p:>10,.0f} L:{low_p:>10,.0f} C:{close_p:>10,.0f} {status}")

        conn.close()

        print(f"\n✅ 검증 완료:")
        if issues:
            print(f"  ❌ {len(issues)}개 문제 발견:")
            for issue in issues:
                print(f"    - {issue}")
            return False
        else:
            print(f"  ✓ 모든 타임스탬프가 유효합니다.")
            return True

    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_timezone_aware_vs_naive():
    """Timezone-aware vs naive datetime 비교"""
    print("\n" + "=" * 80)
    print("4. Timezone Aware vs Naive 비교")
    print("=" * 80)

    # Naive datetime (timezone 정보 없음)
    naive_dt = datetime.now()

    # Timezone-aware datetime (KST)
    kst = timezone(timedelta(hours=9))
    aware_kst = datetime.now(kst)

    # Timezone-aware datetime (UTC)
    aware_utc = datetime.now(timezone.utc)

    print(f"\n  Naive (로컬): {naive_dt} (tzinfo: {naive_dt.tzinfo})")
    print(f"  Aware (KST):  {aware_kst} (tzinfo: {aware_kst.tzinfo})")
    print(f"  Aware (UTC):  {aware_utc} (tzinfo: {aware_utc.tzinfo})")

    print(f"\n💡 권장 사항:")
    print(f"  - Upbit API에서 받은 UTC 시간을 그대로 사용")
    print(f"  - 저장 시 timezone 정보 제거 (naive datetime으로 통일)")
    print(f"  - 비교 시에는 같은 timezone으로 변환 후 비교")

    return True


def main():
    """모든 테스트 실행"""
    print("\n🧪 타임스탬프 검증 테스트 시작\n")

    results = []

    # 테스트 1: Upbit API timezone
    try:
        results.append(("Upbit API Timezone", test_upbit_api_timezone()))
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        results.append(("Upbit API Timezone", False))

    # 테스트 2: 타임스탬프 정규화
    try:
        results.append(("Timestamp Alignment", test_timestamp_alignment()))
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        results.append(("Timestamp Alignment", False))

    # 테스트 3: Timezone aware vs naive
    try:
        results.append(("Timezone Aware vs Naive", test_timezone_aware_vs_naive()))
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        results.append(("Timezone Aware vs Naive", False))

    # 테스트 4: 실제 수집 데이터 검증
    try:
        results.append(("Collected Data Validation", test_collected_data_validation()))
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Collected Data Validation", False))

    # 결과 요약
    print("\n" + "=" * 80)
    print("📊 테스트 결과 요약")
    print("=" * 80)

    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status} | {test_name}")

    all_passed = all(result[1] for result in results)

    print("\n" + "=" * 80)
    if all_passed:
        print("🎉 모든 테스트 통과!")
    else:
        print("⚠️  일부 테스트 실패. 위 결과를 확인하세요.")
    print("=" * 80)

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
