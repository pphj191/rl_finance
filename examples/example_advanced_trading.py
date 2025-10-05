"""
고급 사용 예제

이 예제는 Upbit API의 고급 사용법을 보여줍니다.
- 실시간 데이터 수집 및 분석
- 자동 매매 전략 구현
- 리스크 관리
"""

import sys
import os
# 상위 디렉토리를 Python 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from upbit_api import UpbitAPI, UpbitWebSocket
import time
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import threading
import queue

from upbit_api import UpbitAPI, UpbitWebSocket
import time
import statistics
from datetime import datetime, timedelta
from typing import List, Dict, Any


class TradingBot:
    """간단한 자동매매 봇 예제"""
    
    def __init__(self):
        self.upbit = UpbitAPI()
        self.is_running = False
        self.position = None  # 'long', 'short', None
        
    def get_moving_average(self, market: str, period: int, unit: int = 1) -> float:
        """이동평균 계산"""
        try:
            candles = self.upbit.get_candles_minutes(market, unit=unit, count=period)
            prices = [float(candle['trade_price']) for candle in candles]
            return statistics.mean(prices)
        except Exception as e:
            print(f"이동평균 계산 오류: {e}")
            return 0
    
    def get_rsi(self, market: str, period: int = 14, unit: int = 1) -> float:
        """RSI 계산"""
        try:
            candles = self.upbit.get_candles_minutes(market, unit=unit, count=period + 1)
            prices = [float(candle['trade_price']) for candle in candles]
            
            if len(prices) < period + 1:
                return 50
            
            # 가격 변화 계산
            deltas = [prices[i] - prices[i + 1] for i in range(len(prices) - 1)]
            
            # 상승/하락 분리
            gains = [delta if delta > 0 else 0 for delta in deltas]
            losses = [-delta if delta < 0 else 0 for delta in deltas]
            
            # 평균 계산
            avg_gain = statistics.mean(gains[:period])
            avg_loss = statistics.mean(losses[:period])
            
            if avg_loss == 0:
                return 100
            
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
            return rsi
        except Exception as e:
            print(f"RSI 계산 오류: {e}")
            return 50
    
    def simple_ma_strategy(self, market: str = "KRW-BTC", 
                          short_period: int = 10, long_period: int = 20):
        """간단한 이동평균 교차 전략"""
        print(f"=== {market} 이동평균 교차 전략 시작 ===")
        print(f"단기 MA: {short_period}분, 장기 MA: {long_period}분")
        
        try:
            # 현재 잔고 확인
            balance = self.upbit.get_balance('KRW')
            if balance:
                krw_balance = float(balance['balance'])
                print(f"현재 KRW 잔고: {krw_balance:,.0f}원")
            
            # 이동평균 계산
            short_ma = self.get_moving_average(market, short_period)
            long_ma = self.get_moving_average(market, long_period)
            current_price = self.upbit.get_current_price(market)
            
            print(f"현재가: {current_price:,.0f}원")
            print(f"단기 MA({short_period}): {short_ma:,.0f}원")
            print(f"장기 MA({long_period}): {long_ma:,.0f}원")
            
            # 매수 신호: 단기 MA가 장기 MA 위에 있고, 현재가가 단기 MA 위에 있을 때
            if short_ma > long_ma and current_price > short_ma:
                print("🟢 매수 신호 감지!")
                print("(실제 주문은 실행되지 않습니다)")
                # 실제 주문 시:
                # result = self.upbit.buy_market_order(market, '10000')  # 1만원 매수
                
            # 매도 신호: 단기 MA가 장기 MA 아래에 있고, 현재가가 단기 MA 아래에 있을 때
            elif short_ma < long_ma and current_price < short_ma:
                print("🔴 매도 신호 감지!")
                print("(실제 주문은 실행되지 않습니다)")
                # 실제 주문 시:
                # btc_balance = self.upbit.get_balance('BTC')
                # if btc_balance and float(btc_balance['balance']) > 0:
                #     result = self.upbit.sell_market_order(market, btc_balance['balance'])
            
            else:
                print("📊 신호 없음 (대기)")
                
        except Exception as e:
            print(f"전략 실행 오류: {e}")
    
    def rsi_strategy(self, market: str = "KRW-BTC", 
                    oversold: int = 30, overbought: int = 70):
        """RSI 기반 전략"""
        print(f"\n=== {market} RSI 전략 ===")
        print(f"과매도: {oversold}, 과매수: {overbought}")
        
        try:
            rsi = self.get_rsi(market)
            current_price = self.upbit.get_current_price(market)
            
            print(f"현재가: {current_price:,.0f}원")
            print(f"RSI: {rsi:.2f}")
            
            if rsi < oversold:
                print(f"🟢 과매도 구간 (RSI: {rsi:.2f}) - 매수 고려")
            elif rsi > overbought:
                print(f"🔴 과매수 구간 (RSI: {rsi:.2f}) - 매도 고려")
            else:
                print(f"📊 중립 구간 (RSI: {rsi:.2f}) - 대기")
                
        except Exception as e:
            print(f"RSI 전략 오류: {e}")


class PortfolioManager:
    """포트폴리오 관리 클래스"""
    
    def __init__(self):
        self.upbit = UpbitAPI()
    
    def get_portfolio_value(self) -> Dict[str, Any]:
        """포트폴리오 총 가치 계산"""
        try:
            accounts = self.upbit.get_accounts()
            total_krw_value = 0
            portfolio = {}
            
            for account in accounts:
                currency = account['currency']
                balance = float(account['balance'])
                locked = float(account['locked'])
                total_balance = balance + locked
                
                if total_balance > 0:
                    if currency == 'KRW':
                        krw_value = total_balance
                    else:
                        # 다른 화폐는 KRW 가치로 변환
                        try:
                            market = f"KRW-{currency}"
                            ticker = self.upbit.get_ticker(market)
                            if ticker:
                                current_price = float(ticker[0]['trade_price'])
                                krw_value = total_balance * current_price
                            else:
                                krw_value = 0
                        except:
                            krw_value = 0
                    
                    portfolio[currency] = {
                        'balance': balance,
                        'locked': locked,
                        'total': total_balance,
                        'krw_value': krw_value
                    }
                    total_krw_value += krw_value
            
            return {
                'total_value': total_krw_value,
                'assets': portfolio
            }
            
        except Exception as e:
            print(f"포트폴리오 조회 오류: {e}")
            return {'total_value': 0, 'assets': {}}
    
    def show_portfolio(self):
        """포트폴리오 현황 출력"""
        print("=== 포트폴리오 현황 ===")
        
        portfolio = self.get_portfolio_value()
        total_value = portfolio['total_value']
        
        print(f"총 자산 가치: {total_value:,.0f} KRW")
        print("\n자산별 현황:")
        
        for currency, asset in portfolio['assets'].items():
            percentage = (asset['krw_value'] / total_value * 100) if total_value > 0 else 0
            print(f"{currency}: {asset['total']:.8f} "
                  f"({asset['krw_value']:,.0f} KRW, {percentage:.1f}%)")
    
    def rebalance_suggestion(self, target_weights: Dict[str, float]):
        """리밸런싱 제안"""
        print("\n=== 리밸런싱 제안 ===")
        
        portfolio = self.get_portfolio_value()
        total_value = portfolio['total_value']
        
        if total_value == 0:
            print("포트폴리오가 비어있습니다.")
            return
        
        print(f"목표 비중: {target_weights}")
        print("\n현재 vs 목표:")
        
        for currency, target_weight in target_weights.items():
            current_value = portfolio['assets'].get(currency, {}).get('krw_value', 0)
            current_weight = current_value / total_value
            target_value = total_value * target_weight
            diff_value = target_value - current_value
            
            print(f"{currency}:")
            print(f"  현재: {current_weight:.1%} ({current_value:,.0f} KRW)")
            print(f"  목표: {target_weight:.1%} ({target_value:,.0f} KRW)")
            print(f"  차이: {diff_value:+,.0f} KRW")
            
            if abs(diff_value) > total_value * 0.05:  # 5% 이상 차이
                action = "매수" if diff_value > 0 else "매도"
                print(f"  제안: {abs(diff_value):,.0f}원 {action}")


class PriceMonitor:
    """가격 모니터링 클래스"""
    
    def __init__(self):
        self.upbit = UpbitAPI()
        self.alerts = []
    
    def add_price_alert(self, market: str, target_price: float, 
                       condition: str = "above"):
        """가격 알림 추가"""
        alert = {
            'market': market,
            'target_price': target_price,
            'condition': condition,  # 'above' or 'below'
            'triggered': False
        }
        self.alerts.append(alert)
        print(f"알림 추가: {market} 가격이 {target_price:,}원 {condition}")
    
    def check_alerts(self):
        """알림 확인"""
        for alert in self.alerts:
            if alert['triggered']:
                continue
                
            try:
                current_price = self.upbit.get_current_price(alert['market'])
                
                if (alert['condition'] == 'above' and current_price >= alert['target_price']) or \
                   (alert['condition'] == 'below' and current_price <= alert['target_price']):
                    
                    print(f"🚨 가격 알림: {alert['market']} "
                          f"{current_price:,}원 ({alert['condition']} {alert['target_price']:,}원)")
                    alert['triggered'] = True
                    
            except Exception as e:
                print(f"알림 확인 오류: {e}")
    
    def monitor_prices(self, markets: List[str], duration: int = 60):
        """지정된 시간동안 가격 모니터링"""
        print(f"=== {duration}초간 가격 모니터링 시작 ===")
        
        start_time = time.time()
        last_check = 0
        
        while time.time() - start_time < duration:
            current_time = time.time()
            
            # 5초마다 체크
            if current_time - last_check >= 5:
                print(f"\n[{datetime.now().strftime('%H:%M:%S')}] 현재가:")
                
                for market in markets:
                    try:
                        price = self.upbit.get_current_price(market)
                        print(f"  {market}: {price:,}원")
                    except Exception as e:
                        print(f"  {market}: 조회 실패 ({e})")
                
                # 알림 확인
                self.check_alerts()
                
                last_check = current_time
            
            time.sleep(1)
        
        print("\n모니터링 종료")


def advanced_examples():
    """고급 사용 예제 실행"""
    
    print("=== Upbit API 고급 사용 예제 ===\n")
    
    # 1. 트레이딩 봇 예제
    print("1. 트레이딩 봇 전략 예제")
    bot = TradingBot()
    bot.simple_ma_strategy("KRW-BTC", 5, 10)
    bot.rsi_strategy("KRW-BTC")
    
    # 2. 포트폴리오 관리 예제
    print("\n2. 포트폴리오 관리 예제")
    portfolio_manager = PortfolioManager()
    portfolio_manager.show_portfolio()
    
    # 리밸런싱 제안 (60% KRW, 30% BTC, 10% ETH)
    target_weights = {
        'KRW': 0.6,
        'BTC': 0.3,
        'ETH': 0.1
    }
    portfolio_manager.rebalance_suggestion(target_weights)
    
    # 3. 가격 모니터링 예제
    print("\n3. 가격 모니터링 예제")
    monitor = PriceMonitor()
    
    # 비트코인 가격 알림 설정 (예: 현재가 ±5%)
    try:
        current_btc_price = monitor.upbit.get_current_price("KRW-BTC")
        upper_alert = current_btc_price * 1.05
        lower_alert = current_btc_price * 0.95
        
        monitor.add_price_alert("KRW-BTC", upper_alert, "above")
        monitor.add_price_alert("KRW-BTC", lower_alert, "below")
        
        # 10초간 모니터링 (실제로는 더 긴 시간 사용)
        monitor.monitor_prices(["KRW-BTC", "KRW-ETH"], 10)
        
    except Exception as e:
        print(f"모니터링 설정 오류: {e}")


def real_time_trading_simulation():
    """실시간 거래 시뮬레이션"""
    print("\n=== 실시간 거래 시뮬레이션 ===")
    
    # 가상의 자산 (시뮬레이션용)
    simulation_balance = {
        'KRW': 1000000,  # 100만원
        'BTC': 0,
        'ETH': 0
    }
    
    def on_ticker(data):
        """실시간 가격 데이터로 간단한 거래 로직"""
        market = data.get('code')
        price = data.get('trade_price')
        change_rate = data.get('change_rate', 0)
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] "
              f"{market}: {price:,}원 ({change_rate*100:+.2f}%)")
        
        # 간단한 거래 로직 (시뮬레이션)
        if market == 'KRW-BTC' and change_rate > 0.01:  # 1% 상승 시
            print("  🟢 상승 추세 - 매수 신호")
        elif market == 'KRW-BTC' and change_rate < -0.01:  # 1% 하락 시
            print("  🔴 하락 추세 - 매도 신호")
    
    try:
        # WebSocket 연결 및 구독
        ws_client = UpbitWebSocket()
        ws_client.connect()
        time.sleep(1)
        
        print("실시간 데이터 수신 시작... (10초)")
        ws_client.subscribe_ticker(['KRW-BTC', 'KRW-ETH'], on_ticker)
        
        time.sleep(10)
        ws_client.disconnect()
        
    except Exception as e:
        print(f"실시간 시뮬레이션 오류: {e}")


if __name__ == "__main__":
    # 고급 예제 실행
    advanced_examples()
    
    # 실시간 거래 시뮬레이션 (주석 해제하여 사용)
    # real_time_trading_simulation()
    
    print("\n고급 예제 실행 완료!")
    print("\n주의: 실제 거래 시에는 충분한 테스트와 리스크 관리가 필요합니다.")
