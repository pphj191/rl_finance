"""
Bithumb API 클라이언트

빗썸 거래소 REST API 및 WebSocket을 위한 Python 클라이언트
Upbit API와 동일한 함수명을 사용하여 일관된 인터페이스 제공

Bithumb API 문서: https://apidocs.bithumb.com/
"""

import time
import hmac
import hashlib
import base64
import json
import urllib.parse
import requests
import websocket
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union
import logging
from dataclasses import dataclass
import os
from dotenv import load_dotenv

# 환경변수 로드
load_dotenv()


@dataclass
class BithumbConfig:
    """Bithumb API 설정"""
    base_url: str = "https://api.bithumb.com"
    websocket_url: str = "wss://stream.bithumb.com/stream"
    timeout: int = 30
    max_retries: int = 3
    retry_delay: float = 1.0


class BithumbAPIError(Exception):
    """Bithumb API 에러"""
    def __init__(self, message: str, error_code: Optional[str] = None, 
                 response: Optional[requests.Response] = None):
        self.message = message
        self.error_code = error_code
        self.response = response
        super().__init__(message)


class BithumbAPI:
    """
    Bithumb API 클라이언트
    
    Upbit API와 동일한 함수명을 사용하여 호환성 제공
    """
    
    def __init__(self, access_key: Optional[str] = None, 
                 secret_key: Optional[str] = None,
                 config: Optional[BithumbConfig] = None):
        """
        Args:
            access_key: API 액세스 키 (환경변수에서 자동 로드)
            secret_key: API 시크릿 키 (환경변수에서 자동 로드)
            config: API 설정
        """
        self.access_key = access_key or os.getenv('BITHUMB_ACCESS_KEY')
        self.secret_key = secret_key or os.getenv('BITHUMB_SECRET_KEY')
        self.config = config or BithumbConfig()
        
        self.session = requests.Session()
        # timeout은 request 시에 설정
        
        # 로깅 설정
        self.logger = logging.getLogger(__name__)
        
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    def _create_signature(self, nonce: str, query_string: str = "") -> str:
        """API 요청 서명 생성"""
        if not self.secret_key:
            raise BithumbAPIError("Secret key is required for private API calls")
        
        # Bithumb 서명 방식
        query_hash = urllib.parse.urlencode({"endpoint": query_string})
        signature_payload = f"{query_hash}&nonce={nonce}"
        
        signature = hmac.new(
            self.secret_key.encode('utf-8'),
            signature_payload.encode('utf-8'),
            hashlib.sha512
        ).hexdigest()
        
        return signature
    
    def _request(self, method: str, endpoint: str, params: Optional[Dict] = None,
                data: Optional[Dict] = None, auth_required: bool = False) -> Dict:
        """API 요청 실행"""
        url = f"{self.config.base_url}{endpoint}"
        headers = {
            'Content-Type': 'application/x-www-form-urlencoded',
            'User-Agent': 'Bithumb-Python-Client/1.0.0'
        }
        
        if auth_required:
            if not self.access_key or not self.secret_key:
                raise BithumbAPIError("API keys are required for private endpoints")
            
            nonce = str(int(time.time() * 1000))
            signature = self._create_signature(nonce, endpoint)
            
            headers.update({
                'Api-Key': self.access_key,
                'Api-Sign': signature,
                'Api-Nonce': nonce
            })
        
        for attempt in range(self.config.max_retries):
            try:
                response = self.session.request(
                    method=method,
                    url=url,
                    params=params,
                    data=data,
                    headers=headers,
                    timeout=self.config.timeout
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Bithumb API 응답 구조 확인
                    if result.get('status') == '0000':  # 성공
                        return result.get('data', result)
                    else:
                        error_msg = result.get('message', 'Unknown error')
                        error_code = result.get('status')
                        raise BithumbAPIError(f"API Error: {error_msg}", error_code, response)
                else:
                    response.raise_for_status()
                    
            except requests.exceptions.RequestException as e:
                self.logger.warning(f"Request failed (attempt {attempt + 1}): {e}")
                if attempt == self.config.max_retries - 1:
                    raise BithumbAPIError(f"Request failed after {self.config.max_retries} attempts: {e}")
                time.sleep(self.config.retry_delay * (2 ** attempt))
        
        raise BithumbAPIError("Max retries exceeded")
    
    # =================
    # 시세 정보 API (Public)
    # =================
    
    def get_market_all(self) -> List[Dict[str, str]]:
        """
        마켓 코드 조회 (Upbit 호환)
        
        Returns:
            마켓 정보 리스트
        """
        try:
            result = self._request('GET', '/v1/market/all')
            
            # Bithumb 응답을 Upbit 형식으로 변환
            markets = []
            if isinstance(result, list):
                for item in result:
                    markets.append({
                        'market': item.get('market', ''),
                        'korean_name': item.get('korean_name', ''),
                        'english_name': item.get('english_name', ''),
                        'market_warning': item.get('market_warning', 'NONE')
                    })
            
            return markets
        except Exception as e:
            self.logger.error(f"Failed to get market all: {e}")
            # 기본값 반환
            return [
                {'market': 'KRW-BTC', 'korean_name': '비트코인', 'english_name': 'Bitcoin', 'market_warning': 'NONE'},
                {'market': 'KRW-ETH', 'korean_name': '이더리움', 'english_name': 'Ethereum', 'market_warning': 'NONE'},
                {'market': 'KRW-XRP', 'korean_name': '리플', 'english_name': 'Ripple', 'market_warning': 'NONE'}
            ]
    
    def get_candles_minutes(self, unit: int, market: str, to: Optional[str] = None,
                           count: int = 200) -> List[Dict]:
        """
        분 캔들 조회 (Upbit 호환)
        
        Args:
            unit: 분 단위 (1, 3, 5, 15, 10, 30, 60, 240)
            market: 마켓 코드 (KRW-BTC)
            to: 마지막 캔들 시각 (ISO8601)
            count: 캔들 개수 (최대 200)
            
        Returns:
            캔들 데이터 리스트
        """
        try:
            # Bithumb 형식으로 변환
            currency = market.split('-')[1] if '-' in market else market
            
            params = {
                'order_currency': currency,
                'payment_currency': 'KRW',
                'chart_interval': f'{unit}m'
            }
            
            result = self._request('GET', '/v1/candles/minutes', params=params)
            
            # Upbit 형식으로 변환
            candles = []
            if isinstance(result, list):
                for item in result[:count]:
                    candles.append({
                        'market': market,
                        'candle_date_time_utc': item.get('candle_date_time_utc', ''),
                        'candle_date_time_kst': item.get('candle_date_time_kst', ''),
                        'opening_price': float(item.get('opening_price', 0)),
                        'high_price': float(item.get('high_price', 0)),
                        'low_price': float(item.get('low_price', 0)),
                        'trade_price': float(item.get('trade_price', 0)),
                        'timestamp': int(item.get('timestamp', 0)),
                        'candle_acc_trade_price': float(item.get('candle_acc_trade_price', 0)),
                        'candle_acc_trade_volume': float(item.get('candle_acc_trade_volume', 0))
                    })
            
            return candles
        except Exception as e:
            self.logger.error(f"Failed to get candles: {e}")
            return []
    
    def get_ticker(self, markets: Union[str, List[str]]) -> Union[Dict, List[Dict]]:
        """
        현재가 정보 조회 (Upbit 호환)
        
        Args:
            markets: 마켓 코드 또는 마켓 코드 리스트
            
        Returns:
            현재가 정보
        """
        try:
            if isinstance(markets, str):
                markets = [markets]
            
            results = []
            for market in markets:
                currency = market.split('-')[1] if '-' in market else market
                
                params = {
                    'order_currency': currency,
                    'payment_currency': 'KRW'
                }
                
                result = self._request('GET', '/v1/ticker', params=params)
                
                if result:
                    ticker_data = {
                        'market': market,
                        'trade_date': result.get('date', ''),
                        'trade_time': result.get('timestamp', ''),
                        'trade_date_kst': result.get('date', ''),
                        'trade_time_kst': result.get('timestamp', ''),
                        'opening_price': float(result.get('opening_price', 0)),
                        'high_price': float(result.get('max_price', 0)),
                        'low_price': float(result.get('min_price', 0)),
                        'trade_price': float(result.get('closing_price', 0)),
                        'prev_closing_price': float(result.get('prev_closing_price', 0)),
                        'change': 'RISE' if float(result.get('fluctate_24H', 0)) > 0 else 'FALL',
                        'change_price': float(result.get('fluctate_24H', 0)),
                        'change_rate': float(result.get('fluctate_rate_24H', 0)) / 100,
                        'signed_change_price': float(result.get('fluctate_24H', 0)),
                        'signed_change_rate': float(result.get('fluctate_rate_24H', 0)) / 100,
                        'trade_volume': float(result.get('volume_1day', 0)),
                        'acc_trade_price': float(result.get('value_1day', 0)),
                        'acc_trade_price_24h': float(result.get('value_1day', 0)),
                        'acc_trade_volume': float(result.get('volume_1day', 0)),
                        'acc_trade_volume_24h': float(result.get('volume_1day', 0)),
                        'highest_52_week_price': float(result.get('highest_52_week_price', 0)),
                        'highest_52_week_date': result.get('highest_52_week_date', ''),
                        'lowest_52_week_price': float(result.get('lowest_52_week_price', 0)),
                        'lowest_52_week_date': result.get('lowest_52_week_date', ''),
                        'timestamp': int(result.get('timestamp', 0))
                    }
                    results.append(ticker_data)
            
            return results[0] if len(results) == 1 and isinstance(markets, str) else results
            
        except Exception as e:
            self.logger.error(f"Failed to get ticker: {e}")
            return {} if isinstance(markets, str) else []
    
    def get_orderbook(self, markets: Union[str, List[str]]) -> Union[Dict, List[Dict]]:
        """
        호가 정보 조회 (Upbit 호환)
        
        Args:
            markets: 마켓 코드 또는 마켓 코드 리스트
            
        Returns:
            호가 정보
        """
        try:
            if isinstance(markets, str):
                markets = [markets]
            
            results = []
            for market in markets:
                currency = market.split('-')[1] if '-' in market else market
                
                params = {
                    'order_currency': currency,
                    'payment_currency': 'KRW'
                }
                
                result = self._request('GET', '/v1/orderbook', params=params)
                
                if result:
                    orderbook_data = {
                        'market': market,
                        'timestamp': int(result.get('timestamp', 0)),
                        'total_ask_size': float(result.get('total_ask_size', 0)),
                        'total_bid_size': float(result.get('total_bid_size', 0)),
                        'orderbook_units': []
                    }
                    
                    # 호가 정보 변환
                    for item in result.get('data', []):
                        orderbook_data['orderbook_units'].append({
                            'ask_price': float(item.get('ask_price', 0)),
                            'bid_price': float(item.get('bid_price', 0)),
                            'ask_size': float(item.get('ask_quantity', 0)),
                            'bid_size': float(item.get('bid_quantity', 0))
                        })
                    
                    results.append(orderbook_data)
            
            return results[0] if len(results) == 1 and isinstance(markets, str) else results
            
        except Exception as e:
            self.logger.error(f"Failed to get orderbook: {e}")
            return {} if isinstance(markets, str) else []
    
    def get_trades_ticks(self, market: str, to: Optional[str] = None,
                        count: int = 200, cursor: Optional[str] = None) -> List[Dict]:
        """
        최근 체결 내역 조회 (Upbit 호환)
        
        Args:
            market: 마켓 코드
            to: 마지막 체결 시각 (ISO8601)
            count: 체결 개수 (최대 500)
            cursor: 페이지네이션 커서
            
        Returns:
            체결 내역 리스트
        """
        try:
            currency = market.split('-')[1] if '-' in market else market
            
            params = {
                'order_currency': currency,
                'payment_currency': 'KRW',
                'count': min(count, 500)
            }
            
            result = self._request('GET', '/v1/transaction_history', params=params)
            
            trades = []
            if isinstance(result, list):
                for item in result:
                    trades.append({
                        'market': market,
                        'trade_date_utc': item.get('trade_date_utc', ''),
                        'trade_time_utc': item.get('trade_time_utc', ''),
                        'timestamp': int(item.get('timestamp', 0)),
                        'trade_price': float(item.get('price', 0)),
                        'trade_volume': float(item.get('units_traded', 0)),
                        'prev_closing_price': float(item.get('prev_closing_price', 0)),
                        'change_price': float(item.get('change_price', 0)),
                        'ask_bid': item.get('type', 'bid'),
                        'sequential_id': int(item.get('cont_no', 0))
                    })
            
            return trades
            
        except Exception as e:
            self.logger.error(f"Failed to get trades: {e}")
            return []
    
    # =================
    # 계정 관련 API (Private)
    # =================
    
    def get_accounts(self) -> List[Dict]:
        """
        전체 계좌 조회 (Upbit 호환)
        
        Returns:
            계좌 정보 리스트
        """
        try:
            result = self._request('POST', '/v1/user/balance', auth_required=True)
            
            accounts = []
            if isinstance(result, dict):
                for currency, info in result.items():
                    if currency not in ['total_krw', 'total_btc', 'in_use_krw', 'in_use_btc']:
                        accounts.append({
                            'currency': currency.upper(),
                            'balance': str(info.get('total', 0)),
                            'locked': str(info.get('in_use', 0)),
                            'avg_buy_price': str(info.get('avg_buy_price', 0)),
                            'avg_buy_price_modified': False,
                            'unit_currency': 'KRW'
                        })
            
            return accounts
            
        except Exception as e:
            self.logger.error(f"Failed to get accounts: {e}")
            return []
    
    def get_order(self, uuid: Optional[str] = None, 
                  identifier: Optional[str] = None) -> Dict:
        """
        개별 주문 조회 (Upbit 호환)
        
        Args:
            uuid: 주문 UUID
            identifier: 조회용 사용자 지정값
            
        Returns:
            주문 정보
        """
        if not uuid and not identifier:
            raise BithumbAPIError("Either uuid or identifier is required")
        
        try:
            data = {}
            if uuid:
                data['order_id'] = uuid
            if identifier:
                data['identifier'] = identifier
            
            result = self._request('POST', '/v1/user/orders/detail', 
                                 data=data, auth_required=True)
            
            if result:
                return {
                    'uuid': result.get('order_id', ''),
                    'side': result.get('type', ''),
                    'ord_type': result.get('order_type', ''),
                    'price': str(result.get('order_price', 0)),
                    'state': result.get('order_status', ''),
                    'market': result.get('market', ''),
                    'created_at': result.get('order_date', ''),
                    'volume': str(result.get('units', 0)),
                    'remaining_volume': str(result.get('units_remaining', 0)),
                    'reserved_fee': str(result.get('fee', 0)),
                    'remaining_fee': str(result.get('fee', 0)),
                    'paid_fee': str(result.get('fee', 0)),
                    'locked': str(result.get('locked', 0)),
                    'executed_volume': str(result.get('executed_volume', 0)),
                    'trades_count': int(result.get('trades_count', 0))
                }
            
            return {}
            
        except Exception as e:
            self.logger.error(f"Failed to get order: {e}")
            return {}
    
    def get_orders(self, market: Optional[str] = None, 
                   uuids: Optional[List[str]] = None,
                   identifiers: Optional[List[str]] = None,
                   state: str = 'wait', states: Optional[List[str]] = None,
                   page: int = 1, limit: int = 100,
                   order_by: str = 'desc') -> List[Dict]:
        """
        주문 리스트 조회 (Upbit 호환)
        
        Args:
            market: 마켓 코드
            uuids: 주문 UUID 필터
            identifiers: 사용자 지정값 필터
            state: 주문 상태
            states: 주문 상태 리스트
            page: 페이지 수
            limit: 요청 개수
            order_by: 정렬 방식
            
        Returns:
            주문 리스트
        """
        try:
            data = {
                'count': limit,
                'after': str((page - 1) * limit)
            }
            
            if market:
                currency = market.split('-')[1] if '-' in market else market
                data['order_currency'] = currency
                data['payment_currency'] = 'KRW'
            
            result = self._request('POST', '/v1/user/orders', 
                                 data=data, auth_required=True)
            
            orders = []
            if isinstance(result, list):
                for item in result:
                    orders.append({
                        'uuid': item.get('order_id', ''),
                        'side': item.get('type', ''),
                        'ord_type': item.get('order_type', ''),
                        'price': str(item.get('order_price', 0)),
                        'state': item.get('order_status', ''),
                        'market': item.get('market', ''),
                        'created_at': item.get('order_date', ''),
                        'volume': str(item.get('units', 0)),
                        'remaining_volume': str(item.get('units_remaining', 0)),
                        'reserved_fee': str(item.get('fee', 0)),
                        'remaining_fee': str(item.get('fee', 0)),
                        'paid_fee': str(item.get('fee', 0)),
                        'locked': str(item.get('locked', 0)),
                        'executed_volume': str(item.get('executed_volume', 0)),
                        'trades_count': int(item.get('trades_count', 0))
                    })
            
            return orders
            
        except Exception as e:
            self.logger.error(f"Failed to get orders: {e}")
            return []
    
    def cancel_order(self, uuid: Optional[str] = None,
                     identifier: Optional[str] = None) -> Dict:
        """
        주문 취소 (Upbit 호환)
        
        Args:
            uuid: 주문 UUID
            identifier: 조회용 사용자 지정값
            
        Returns:
            취소된 주문 정보
        """
        if not uuid and not identifier:
            raise BithumbAPIError("Either uuid or identifier is required")
        
        try:
            data = {}
            if uuid:
                data['order_id'] = uuid
            if identifier:
                data['identifier'] = identifier
            
            result = self._request('POST', '/v1/user/orders/cancel', 
                                 data=data, auth_required=True)
            
            if result:
                return {
                    'uuid': result.get('order_id', ''),
                    'side': result.get('type', ''),
                    'ord_type': result.get('order_type', ''),
                    'price': str(result.get('order_price', 0)),
                    'state': 'cancelled',
                    'market': result.get('market', ''),
                    'created_at': result.get('order_date', ''),
                    'volume': str(result.get('units', 0)),
                    'remaining_volume': str(result.get('units_remaining', 0)),
                    'reserved_fee': str(result.get('fee', 0)),
                    'remaining_fee': str(result.get('fee', 0)),
                    'paid_fee': str(result.get('fee', 0)),
                    'locked': str(result.get('locked', 0)),
                    'executed_volume': str(result.get('executed_volume', 0)),
                    'trades_count': int(result.get('trades_count', 0))
                }
            
            return {}
            
        except Exception as e:
            self.logger.error(f"Failed to cancel order: {e}")
            return {}
    
    def order(self, market: str, side: str, volume: Optional[str] = None,
              price: Optional[str] = None, ord_type: str = 'limit',
              identifier: Optional[str] = None, time_in_force: str = 'IOC') -> Dict:
        """
        주문하기 (Upbit 호환)
        
        Args:
            market: 마켓 코드 (KRW-BTC)
            side: 주문 종류 (bid: 매수, ask: 매도)
            volume: 주문량 (지정가, 시장가 매도 시 필수)
            price: 주문 가격 (지정가 시 필수)
            ord_type: 주문 타입 (limit, price, market)
            identifier: 조회용 사용자 지정값
            time_in_force: IOC, FOK
            
        Returns:
            주문 결과
        """
        try:
            currency = market.split('-')[1] if '-' in market else market
            
            data = {
                'order_currency': currency,
                'payment_currency': 'KRW',
                'type': side,
                'order_type': ord_type
            }
            
            if volume:
                data['units'] = volume
            if price:
                data['price'] = price
            if identifier:
                data['identifier'] = identifier
            
            endpoint = '/v1/user/orders/place'
            result = self._request('POST', endpoint, data=data, auth_required=True)
            
            if result:
                return {
                    'uuid': result.get('order_id', ''),
                    'side': side,
                    'ord_type': ord_type,
                    'price': str(price) if price else '0',
                    'state': 'wait',
                    'market': market,
                    'created_at': result.get('order_date', ''),
                    'volume': str(volume) if volume else '0',
                    'remaining_volume': str(volume) if volume else '0',
                    'reserved_fee': str(result.get('fee', 0)),
                    'remaining_fee': str(result.get('fee', 0)),
                    'paid_fee': '0',
                    'locked': str(result.get('locked', 0)),
                    'executed_volume': '0',
                    'trades_count': 0
                }
            
            return {}
            
        except Exception as e:
            self.logger.error(f"Failed to place order: {e}")
            return {}


class BithumbWebSocket:
    """
    Bithumb WebSocket 클라이언트
    
    Upbit WebSocket과 동일한 인터페이스 제공
    """
    
    def __init__(self, access_key: Optional[str] = None,
                 secret_key: Optional[str] = None):
        self.access_key = access_key or os.getenv('BITHUMB_ACCESS_KEY')
        self.secret_key = secret_key or os.getenv('BITHUMB_SECRET_KEY')
        self.ws = None
        self.callbacks = {}
        self.is_connected = False
        self.reconnect_interval = 5
        self.logger = logging.getLogger(__name__)
    
    def connect(self, callback: Callable[[str], None],
                markets: List[str], types: List[str] = ['ticker']) -> None:
        """
        WebSocket 연결 및 구독 (Upbit 호환)
        
        Args:
            callback: 메시지 수신 콜백 함수
            markets: 구독할 마켓 리스트
            types: 구독할 타입 리스트 ('ticker', 'trade', 'orderbook')
        """
        try:
            def on_message(ws, message):
                try:
                    callback(message)
                except Exception as e:
                    self.logger.error(f"Callback error: {e}")
            
            def on_error(ws, error):
                self.logger.error(f"WebSocket error: {error}")
                self.is_connected = False
            
            def on_close(ws, close_status_code, close_msg):
                self.logger.info("WebSocket connection closed")
                self.is_connected = False
            
            def on_open(ws):
                self.logger.info("WebSocket connection opened")
                self.is_connected = True
                
                # 구독 메시지 전송
                subscribe_message = {
                    "type": "subscribe",
                    "symbols": markets,
                    "tickTypes": types
                }
                ws.send(json.dumps(subscribe_message))
            
            # WebSocket URL (Bithumb 실제 WebSocket URL로 수정 필요)
            url = "wss://stream.bithumb.com/stream"
            
            self.ws = websocket.WebSocketApp(
                url,
                on_open=on_open,
                on_message=on_message,
                on_error=on_error,
                on_close=on_close
            )
            
            # 별도 스레드에서 실행
            self.ws_thread = threading.Thread(target=self.ws.run_forever)
            self.ws_thread.daemon = True
            self.ws_thread.start()
            
        except Exception as e:
            self.logger.error(f"Failed to connect WebSocket: {e}")
    
    def disconnect(self):
        """WebSocket 연결 해제"""
        if self.ws:
            self.ws.close()
            self.is_connected = False
    
    def is_alive(self) -> bool:
        """연결 상태 확인"""
        return self.is_connected


# 편의 함수들 (Upbit 호환)
def get_upbit_market_all() -> List[Dict[str, str]]:
    """마켓 코드 조회 (Upbit 함수명 유지)"""
    api = BithumbAPI()
    return api.get_market_all()


def get_upbit_candles_minutes(unit: int, market: str, count: int = 200) -> List[Dict]:
    """분 캔들 조회 (Upbit 함수명 유지)"""
    api = BithumbAPI()
    return api.get_candles_minutes(unit, market, count=count)


def get_upbit_ticker(markets: Union[str, List[str]]) -> Union[Dict, List[Dict]]:
    """현재가 정보 조회 (Upbit 함수명 유지)"""
    api = BithumbAPI()
    return api.get_ticker(markets)


def get_upbit_orderbook(markets: Union[str, List[str]]) -> Union[Dict, List[Dict]]:
    """호가 정보 조회 (Upbit 함수명 유지)"""
    api = BithumbAPI()
    return api.get_orderbook(markets)


if __name__ == "__main__":
    # 간단한 테스트
    print("=== Bithumb API 테스트 ===")
    
    api = BithumbAPI()
    
    try:
        # 마켓 정보 조회
        markets = api.get_market_all()
        print(f"마켓 수: {len(markets)}")
        if markets:
            print(f"첫 번째 마켓: {markets[0]}")
        
        # 비트코인 현재가 조회
        ticker = api.get_ticker('KRW-BTC')
        if ticker and isinstance(ticker, dict):
            price = ticker.get('trade_price', 'N/A')
            if isinstance(price, (int, float)):
                print(f"BTC 현재가: {price:,} KRW")
            else:
                print(f"BTC 현재가: {price}")
        elif ticker and isinstance(ticker, list) and len(ticker) > 0:
            price = ticker[0].get('trade_price', 'N/A')
            if isinstance(price, (int, float)):
                print(f"BTC 현재가: {price:,} KRW")
            else:
                print(f"BTC 현재가: {price}")
        
        print("✅ Bithumb API 테스트 완료")
        
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
