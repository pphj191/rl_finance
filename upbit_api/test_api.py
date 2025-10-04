"""
종합 테스트 스크립트

Upbit API의 모든 기능을 테스트합니다.
"""

from upbit_api import UpbitAPI
import time
import json

from upbit_api import UpbitAPI
import time


def test_public_api():
    """공개 API 테스트 (인증 불필요)"""
    print("=== 공개 API 테스트 ===")
    
    upbit = UpbitAPI()
    
    # 1. 마켓 목록 조회
    print("1. 마켓 목록 조회")
    markets = upbit.get_markets()
    krw_markets = [m for m in markets if m['market'].startswith('KRW-')]
    print(f"총 KRW 마켓 개수: {len(krw_markets)}")
    
    # 상위 10개 코인 현재가
    print("\n2. 상위 10개 코인 현재가")
    top_markets = [m['market'] for m in krw_markets[:10]]
    tickers = upbit.get_ticker(top_markets)
    
    # 거래대금 순으로 정렬
    tickers.sort(key=lambda x: float(x['acc_trade_price_24h']), reverse=True)
    
    print("순위  코인       현재가        24h 변화율   24h 거래대금")
    print("-" * 60)
    for i, ticker in enumerate(tickers[:10], 1):
        market = ticker['market'].replace('KRW-', '')
        price = ticker['trade_price']
        change_rate = ticker['change_rate'] * 100
        volume = ticker['acc_trade_price_24h']
        
        print(f"{i:2d}   {market:8s} {price:12,} {change_rate:+6.2f}%  {float(volume)/100000000:8.1f}억")
    
    # 3. 비트코인 캔들 데이터
    print("\n3. 비트코인 1시간 캔들 (최근 12시간)")
    candles = upbit.get_candles_minutes("KRW-BTC", unit=60, count=12)
    
    print("시간      시가        고가        저가        종가        거래량")
    print("-" * 70)
    for candle in reversed(candles):  # 시간순 정렬
        time_str = candle['candle_date_time_kst'][11:16]  # HH:MM만 추출
        open_price = candle['opening_price']
        high_price = candle['high_price']
        low_price = candle['low_price']
        close_price = candle['trade_price']
        volume = candle['candle_acc_trade_volume']
        
        print(f"{time_str}  {open_price:10,} {high_price:10,} {low_price:10,} {close_price:10,} {volume:10.4f}")
    
    # 4. 호가 분석
    print("\n4. 인기 코인 호가 스프레드 분석")
    top_5_markets = [ticker['market'] for ticker in tickers[:5]]
    orderbooks = upbit.get_orderbook(top_5_markets)
    
    print("코인      매수호가      매도호가      스프레드    스프레드%")
    print("-" * 60)
    for ob in orderbooks:
        market = ob['market'].replace('KRW-', '')
        best_bid = ob['orderbook_units'][0]['bid_price']
        best_ask = ob['orderbook_units'][0]['ask_price']
        spread = best_ask - best_bid
        spread_pct = (spread / best_bid) * 100
        
        print(f"{market:8s} {best_bid:10,} {best_ask:10,} {spread:8,} {spread_pct:8.3f}%")


def test_technical_analysis():
    """간단한 기술적 분석 예제"""
    print("\n=== 기술적 분석 예제 ===")
    
    upbit = UpbitAPI()
    
    # 비트코인 일봉 데이터로 이동평균 계산
    candles = upbit.get_candles_days("KRW-BTC", count=20)
    prices = [float(candle['trade_price']) for candle in candles]
    
    # 5일, 20일 이동평균
    ma5 = sum(prices[:5]) / 5
    ma20 = sum(prices) / 20
    current_price = prices[0]
    
    print(f"비트코인 기술적 분석:")
    print(f"현재가:    {current_price:,} KRW")
    print(f"5일 평균:  {ma5:,.0f} KRW")
    print(f"20일 평균: {ma20:,.0f} KRW")
    
    # 단순 신호 판단
    if current_price > ma5 > ma20:
        signal = "🟢 강한 상승 추세"
    elif current_price > ma5:
        signal = "🟡 약한 상승 추세"
    elif current_price < ma5 < ma20:
        signal = "🔴 강한 하락 추세"
    else:
        signal = "🟡 약한 하락 추세"
    
    print(f"신호:      {signal}")


def test_market_monitoring():
    """시장 모니터링 예제"""
    print("\n=== 시장 모니터링 (30초) ===")
    
    upbit = UpbitAPI()
    watch_list = ["KRW-BTC", "KRW-ETH", "KRW-XRP", "KRW-ADA", "KRW-SOL"]
    
    print("코인 가격 변화 모니터링 시작...")
    print("코인      현재가        변화율   시간")
    print("-" * 45)
    
    start_time = time.time()
    last_check = 0
    
    while time.time() - start_time < 30:  # 30초간 모니터링
        current_time = time.time()
        
        if current_time - last_check >= 5:  # 5초마다 체크
            try:
                tickers = upbit.get_ticker(watch_list)
                current_time_str = time.strftime("%H:%M:%S")
                
                for ticker in tickers:
                    market = ticker['market'].replace('KRW-', '')
                    price = ticker['trade_price']
                    change_rate = ticker['change_rate'] * 100
                    
                    print(f"{market:8s} {price:12,} {change_rate:+6.2f}% {current_time_str}")
                
                print("-" * 45)
                
            except Exception as e:
                print(f"오류 발생: {e}")
            
            last_check = current_time
        
        time.sleep(1)
    
    print("모니터링 완료")


def performance_test():
    """성능 테스트"""
    print("\n=== API 성능 테스트 ===")
    
    upbit = UpbitAPI()
    
    # 단일 요청 테스트
    start_time = time.time()
    ticker = upbit.get_ticker("KRW-BTC")
    single_time = time.time() - start_time
    print(f"단일 현재가 조회: {single_time:.3f}초")
    
    # 다중 요청 테스트
    start_time = time.time()
    markets = ["KRW-BTC", "KRW-ETH", "KRW-XRP", "KRW-ADA", "KRW-SOL"]
    tickers = upbit.get_ticker(markets)
    multi_time = time.time() - start_time
    print(f"5개 코인 현재가 조회: {multi_time:.3f}초")
    
    # 캔들 데이터 조회 테스트
    start_time = time.time()
    candles = upbit.get_candles_days("KRW-BTC", count=200)
    candle_time = time.time() - start_time
    print(f"200일 캔들 데이터: {candle_time:.3f}초")
    
    print(f"평균 응답 시간: {(single_time + multi_time + candle_time) / 3:.3f}초")


if __name__ == "__main__":
    try:
        # 1. 공개 API 테스트
        test_public_api()
        
        # 2. 기술적 분석 예제
        test_technical_analysis()
        
        # 3. 성능 테스트
        performance_test()
        
        # 4. 실시간 모니터링 (사용자 선택)
        print("\n실시간 모니터링을 시작하시겠습니까? (y/n): ", end="")
        choice = input().lower()
        if choice == 'y':
            test_market_monitoring()
        
        print("\n테스트 완료!")
        
    except KeyboardInterrupt:
        print("\n사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"\n오류 발생: {e}")
