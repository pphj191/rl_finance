"""
기본 사용 예제

이 예제는 Upbit API의 기본 사용법을 보여줍니다.
"""

from upbit_api import UpbitAPI, UpbitWebSocket
import time
import json

from upbit_api import UpbitAPI, UpbitWebSocket
import time
import json


def main():
    """메인 함수"""
    
    # API 클라이언트 초기화
    # .env 파일에서 자동으로 API 키를 로드합니다
    upbit = UpbitAPI()
    
    print("=== Upbit API 사용 예제 ===\n")
    
    # 1. 시세 정보 조회 (인증 불필요)
    print("1. 마켓 코드 목록 조회")
    try:
        markets = upbit.get_markets()
        krw_markets = [market for market in markets if market['market'].startswith('KRW-')]
        print(f"KRW 마켓 개수: {len(krw_markets)}")
        print(f"처음 5개 마켓: {[m['market'] for m in krw_markets[:5]]}")
    except Exception as e:
        print(f"오류: {e}")
    print()
    
    # 2. 현재가 정보 조회
    print("2. 비트코인 현재가 조회")
    try:
        ticker = upbit.get_ticker("KRW-BTC")
        if ticker:
            btc_price = ticker[0]['trade_price']
            print(f"BTC 현재가: {btc_price:,} KRW")
            print(f"24시간 변화율: {ticker[0]['change_rate'] * 100:.2f}%")
    except Exception as e:
        print(f"오류: {e}")
    print()
    
    # 3. 여러 코인 현재가 조회
    print("3. 여러 코인 현재가 조회")
    try:
        tickers = upbit.get_ticker(["KRW-BTC", "KRW-ETH", "KRW-XRP"])
        for ticker in tickers:
            market = ticker['market']
            price = ticker['trade_price']
            change_rate = ticker['change_rate'] * 100
            print(f"{market}: {price:,} KRW ({change_rate:+.2f}%)")
    except Exception as e:
        print(f"오류: {e}")
    print()
    
    # 4. 캔들 데이터 조회
    print("4. 비트코인 일봉 데이터 조회 (최근 5일)")
    try:
        candles = upbit.get_candles_days("KRW-BTC", count=5)
        for candle in candles:
            date = candle['candle_date_time_kst'][:10]
            open_price = candle['opening_price']
            high_price = candle['high_price']
            low_price = candle['low_price']
            close_price = candle['trade_price']
            print(f"{date}: 시가 {open_price:,}, 고가 {high_price:,}, 저가 {low_price:,}, 종가 {close_price:,}")
    except Exception as e:
        print(f"오류: {e}")
    print()
    
    # 5. 호가 정보 조회
    print("5. 비트코인 호가 정보 조회")
    try:
        orderbook = upbit.get_orderbook("KRW-BTC")
        if orderbook:
            ob = orderbook[0]
            print("매도 호가 (상위 3개):")
            for ask in ob['orderbook_units'][:3]:
                print(f"  가격: {ask['ask_price']:,}, 수량: {ask['ask_size']:.4f}")
            
            print("매수 호가 (상위 3개):")
            for bid in ob['orderbook_units'][:3]:
                print(f"  가격: {bid['bid_price']:,}, 수량: {bid['bid_size']:.4f}")
    except Exception as e:
        print(f"오류: {e}")
    print()
    
    # 6. 계좌 정보 조회 (인증 필요)
    print("6. 계좌 정보 조회 (API 키 필요)")
    try:
        accounts = upbit.get_accounts()
        print(f"보유 자산 개수: {len(accounts)}")
        for account in accounts:
            currency = account['currency']
            balance = float(account['balance'])
            locked = float(account['locked'])
            if balance > 0 or locked > 0:
                print(f"{currency}: 사용가능 {balance:.4f}, 주문중 {locked:.4f}")
    except Exception as e:
        print(f"오류: {e}")
    print()
    
    # 7. 주문 가능 정보 조회
    print("7. 비트코인 주문 가능 정보 조회")
    try:
        order_chance = upbit.get_orders_chance("KRW-BTC")
        market_info = order_chance['market']
        bid_account = order_chance['bid_account']
        ask_account = order_chance['ask_account']
        
        print(f"마켓: {market_info['id']}")
        print(f"매수 가능 금액: {float(bid_account['balance']):,.0f} KRW")
        print(f"매도 가능 수량: {float(ask_account['balance']):.8f} BTC")
    except Exception as e:
        print(f"오류: {e}")
    print()
    
    # 8. 체결 대기 주문 조회
    print("8. 체결 대기 주문 조회")
    try:
        open_orders = upbit.get_orders_open()
        print(f"체결 대기 주문 개수: {len(open_orders)}")
        for order in open_orders[:3]:  # 최대 3개만 표시
            market = order['market']
            side = "매수" if order['side'] == 'bid' else "매도"
            price = float(order['price'])
            volume = float(order['volume'])
            print(f"{market} {side}: {price:,} KRW, {volume:.8f}")
    except Exception as e:
        print(f"오류: {e}")
    print()


def websocket_example():
    """WebSocket 사용 예제"""
    print("\n=== WebSocket 실시간 데이터 수신 예제 ===")
    
    # 실시간 현재가 콜백 함수
    def on_ticker(data):
        market = data.get('code', 'Unknown')
        price = data.get('trade_price', 0)
        change_rate = data.get('change_rate', 0) * 100
        print(f"[실시간] {market}: {price:,} KRW ({change_rate:+.2f}%)")
    
    # 실시간 체결 콜백 함수
    def on_trade(data):
        market = data.get('code', 'Unknown')
        price = data.get('trade_price', 0)
        volume = data.get('trade_volume', 0)
        ask_bid = "매도" if data.get('ask_bid') == 'ASK' else "매수"
        print(f"[체결] {market} {ask_bid}: {price:,} KRW, {volume:.8f}")
    
    try:
        # WebSocket 클라이언트 생성
        ws_client = UpbitWebSocket()
        
        # WebSocket 연결
        ws_client.connect()
        time.sleep(1)  # 연결 대기
        
        # 현재가 정보 구독
        print("비트코인, 이더리움 현재가 실시간 수신 시작...")
        ws_client.subscribe_ticker(["KRW-BTC", "KRW-ETH"], on_ticker)
        
        # 5초 후 체결 정보도 구독
        time.sleep(5)
        print("체결 정보 실시간 수신 시작...")
        ws_client.subscribe_trade(["KRW-BTC"], on_trade)
        
        # 10초간 데이터 수신
        print("10초간 실시간 데이터 수신...")
        time.sleep(10)
        
        # 연결 종료
        ws_client.disconnect()
        print("WebSocket 연결 종료")
        
    except Exception as e:
        print(f"WebSocket 오류: {e}")


def trading_example():
    """주문 예제 (실제 주문은 주석 처리됨)"""
    print("\n=== 주문 예제 (테스트용 - 실제 주문 안됨) ===")
    
    upbit = UpbitAPI()
    
    try:
        # 현재가 조회
        current_price = upbit.get_current_price("KRW-BTC")
        print(f"BTC 현재가: {current_price:,} KRW")
        
        # 시장가 매수 예제 (주석 처리됨)
        print("\n시장가 매수 예제:")
        print(f"# 5,000원으로 BTC 시장가 매수")
        print(f"# result = upbit.buy_market_order('KRW-BTC', '5000')")
        
        # 지정가 매수 예제 (주석 처리됨)
        buy_price = current_price * 0.99  # 현재가보다 1% 낮은 가격
        print(f"\n지정가 매수 예제:")
        print(f"# {buy_price:,.0f}원에 0.001 BTC 지정가 매수")
        print(f"# result = upbit.buy_limit_order('KRW-BTC', '0.001', '{buy_price:.0f}')")
        
        # 지정가 매도 예제 (주석 처리됨)
        sell_price = current_price * 1.01  # 현재가보다 1% 높은 가격
        print(f"\n지정가 매도 예제:")
        print(f"# {sell_price:,.0f}원에 0.001 BTC 지정가 매도")
        print(f"# result = upbit.sell_limit_order('KRW-BTC', '0.001', '{sell_price:.0f}')")
        
        print("\n주의: 실제 거래를 원하시면 주석을 해제하고 금액을 조정하세요.")
        
    except Exception as e:
        print(f"오류: {e}")


if __name__ == "__main__":
    # 기본 API 예제 실행
    main()
    
    # WebSocket 예제 실행 (주석 해제하여 사용)
    # websocket_example()
    
    # 주문 예제 실행
    trading_example()
    
    print("\n예제 실행 완료!")
