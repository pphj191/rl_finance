"""
Upbit API Python Client

이 모듈은 Upbit 거래소의 REST API와 WebSocket을 사용하여
시세 조회, 주문, 자산 관리 등의 기능을 제공합니다.

Author: Your Name
Date: 2025-09-29
"""

import os
import hashlib
import uuid
import jwt
import requests
import json
import websocket
import threading
import time
from urllib.parse import unquote, urlencode
from typing import Dict, List, Optional, Any, Union
from dotenv import load_dotenv


class UpbitAPI:
    """
    Upbit API 클라이언트 클래스
    
    시세 조회, 주문, 자산 관리 등의 기능을 제공합니다.
    """
    
    def __init__(self, access_key: Optional[str] = None, secret_key: Optional[str] = None):
        """
        UpbitAPI 클라이언트 초기화
        
        Args:
            access_key (str, optional): Upbit API Access Key
            secret_key (str, optional): Upbit API Secret Key
        """
        # .env 파일 로드
        load_dotenv()
        
        self.access_key = access_key or os.getenv('UPBIT_ACCESS_KEY')
        self.secret_key = secret_key or os.getenv('UPBIT_SECRET_KEY')
        self.base_url = os.getenv('UPBIT_BASE_URL', 'https://api.upbit.com')
        self.websocket_url = os.getenv('UPBIT_WEBSOCKET_URL', 'wss://api.upbit.com/websocket/v1')
        
        # 세션 설정
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Upbit Python Client/1.0',
            'Accept': 'application/json'
        })
    
    def _build_query_string(self, params: Dict[str, Any]) -> str:
        """
        쿼리 스트링 생성
        
        Args:
            params (Dict[str, Any]): 쿼리 파라미터
            
        Returns:
            str: 쿼리 스트링
        """
        return unquote(urlencode(params, doseq=True))
    
    def _create_jwt_token(self, query_string: str = "") -> str:
        """
        JWT 토큰 생성
        
        Args:
            query_string (str): 쿼리 스트링
            
        Returns:
            str: JWT 토큰
        """
        if not self.access_key or not self.secret_key:
            raise ValueError("API 키가 설정되지 않았습니다. access_key와 secret_key를 확인해주세요.")
        
        payload = {
            "access_key": self.access_key,
            "nonce": str(uuid.uuid4())
        }
        
        if query_string:
            query_hash = hashlib.sha512(query_string.encode("utf-8")).hexdigest()
            payload["query_hash"] = query_hash
            payload["query_hash_alg"] = "SHA512"
        
        token = jwt.encode(payload, self.secret_key, algorithm="HS512")
        return token if isinstance(token, str) else token.decode('utf-8')
    
    def _make_request(self, method: str, endpoint: str, params: Optional[Dict] = None, 
                     data: Optional[Dict] = None, auth_required: bool = False) -> Dict:
        """
        HTTP 요청 실행
        
        Args:
            method (str): HTTP 메서드 (GET, POST, DELETE)
            endpoint (str): API 엔드포인트
            params (Dict, optional): 쿼리 파라미터
            data (Dict, optional): 요청 본문 데이터
            auth_required (bool): 인증 필요 여부
            
        Returns:
            Dict: API 응답
        """
        url = f"{self.base_url}{endpoint}"
        headers = {}
        
        # 인증이 필요한 경우 JWT 토큰 생성
        if auth_required:
            query_string = ""
            if method.upper() == "GET" and params:
                query_string = self._build_query_string(params)
            elif method.upper() == "POST" and data:
                query_string = self._build_query_string(data)
            
            jwt_token = self._create_jwt_token(query_string)
            headers["Authorization"] = f"Bearer {jwt_token}"
        
        if method.upper() == "POST":
            headers["Content-Type"] = "application/json"
        
        try:
            if method.upper() == "GET":
                response = self.session.get(url, params=params, headers=headers)
            elif method.upper() == "POST":
                response = self.session.post(url, json=data, headers=headers)
            elif method.upper() == "DELETE":
                response = self.session.delete(url, params=params, headers=headers)
            else:
                raise ValueError(f"지원하지 않는 HTTP 메서드: {method}")
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            print(f"API 요청 실패: {e}")
            if hasattr(e, 'response') and e.response is not None:
                try:
                    error_data = e.response.json()
                    print(f"오류 상세: {error_data}")
                except:
                    print(f"응답 코드: {e.response.status_code}")
            raise
    
    # =============================================================
    # 시세 정보 (Quotation) - 인증 불필요
    # =============================================================
    
    def get_markets(self) -> List[Dict]:
        """
        마켓 코드 목록 조회
        
        Returns:
            List[Dict]: 마켓 정보 리스트
        """
        return self._make_request("GET", "/v1/market/all")
    
    def get_candles_minutes(self, market: str, unit: int = 1, count: int = 200, 
                           to: Optional[str] = None) -> List[Dict]:
        """
        분 캔들 조회
        
        Args:
            market (str): 마켓 코드 (예: KRW-BTC)
            unit (int): 분 단위 (1, 3, 5, 15, 10, 30, 60, 240)
            count (int): 캔들 개수 (최대 200)
            to (str, optional): 마지막 캔들 시각 (ISO 8601 형식)
            
        Returns:
            List[Dict]: 캔들 데이터 리스트
        """
        params = {"market": market, "count": count}
        if to:
            params["to"] = to
        
        return self._make_request("GET", f"/v1/candles/minutes/{unit}", params=params)
    
    def get_candles_days(self, market: str, count: int = 200, to: Optional[str] = None, 
                        convertingPriceUnit: Optional[str] = None) -> List[Dict]:
        """
        일 캔들 조회
        
        Args:
            market (str): 마켓 코드
            count (int): 캔들 개수 (최대 200)
            to (str, optional): 마지막 캔들 시각
            convertingPriceUnit (str, optional): 종가 환산 화폐 단위
            
        Returns:
            List[Dict]: 캔들 데이터 리스트
        """
        params = {"market": market, "count": count}
        if to:
            params["to"] = to
        if convertingPriceUnit:
            params["convertingPriceUnit"] = convertingPriceUnit
        
        return self._make_request("GET", "/v1/candles/days", params=params)
    
    def get_candles_weeks(self, market: str, count: int = 200, to: Optional[str] = None) -> List[Dict]:
        """
        주 캔들 조회
        
        Args:
            market (str): 마켓 코드
            count (int): 캔들 개수 (최대 200)
            to (str, optional): 마지막 캔들 시각
            
        Returns:
            List[Dict]: 캔들 데이터 리스트
        """
        params = {"market": market, "count": count}
        if to:
            params["to"] = to
        
        return self._make_request("GET", "/v1/candles/weeks", params=params)
    
    def get_candles_months(self, market: str, count: int = 200, to: Optional[str] = None) -> List[Dict]:
        """
        월 캔들 조회
        
        Args:
            market (str): 마켓 코드
            count (int): 캔들 개수 (최대 200)
            to (str, optional): 마지막 캔들 시각
            
        Returns:
            List[Dict]: 캔들 데이터 리스트
        """
        params = {"market": market, "count": count}
        if to:
            params["to"] = to
        
        return self._make_request("GET", "/v1/candles/months", params=params)
    
    def get_trades_ticks(self, market: str, count: int = 200, to: Optional[str] = None, 
                        cursor: Optional[str] = None, daysAgo: Optional[int] = None) -> List[Dict]:
        """
        최근 체결 내역 조회
        
        Args:
            market (str): 마켓 코드
            count (int): 체결 내역 개수 (최대 500)
            to (str, optional): 마지막 체결 시각
            cursor (str, optional): 페이징 커서
            daysAgo (int, optional): 최근 일 수
            
        Returns:
            List[Dict]: 체결 내역 리스트
        """
        params = {"market": market, "count": count}
        if to:
            params["to"] = to
        if cursor:
            params["cursor"] = cursor
        if daysAgo:
            params["daysAgo"] = daysAgo
        
        return self._make_request("GET", "/v1/trades/ticks", params=params)
    
    def get_ticker(self, markets: Union[str, List[str]]) -> List[Dict]:
        """
        현재가 정보 조회
        
        Args:
            markets (Union[str, List[str]]): 마켓 코드 (문자열 또는 리스트)
            
        Returns:
            List[Dict]: 현재가 정보 리스트
        """
        if isinstance(markets, str):
            markets = [markets]
        
        markets_str = ",".join(markets)
        params = {"markets": markets_str}
        
        return self._make_request("GET", "/v1/ticker", params=params)
    
    def get_orderbook(self, markets: Union[str, List[str]]) -> List[Dict]:
        """
        호가 정보 조회
        
        Args:
            markets (Union[str, List[str]]): 마켓 코드
            
        Returns:
            List[Dict]: 호가 정보 리스트
        """
        if isinstance(markets, str):
            markets = [markets]
        
        markets_str = ",".join(markets)
        params = {"markets": markets_str}
        
        return self._make_request("GET", "/v1/orderbook", params=params)
    
    # =============================================================
    # 자산 관리 (Exchange) - 인증 필요
    # =============================================================
    
    def get_accounts(self) -> List[Dict]:
        """
        전체 계좌 조회
        
        Returns:
            List[Dict]: 계좌 정보 리스트
        """
        return self._make_request("GET", "/v1/accounts", auth_required=True)
    
    # =============================================================
    # 주문 관리 (Orders) - 인증 필요
    # =============================================================
    
    def get_orders_chance(self, market: str) -> Dict:
        """
        주문 가능 정보 조회
        
        Args:
            market (str): 마켓 코드
            
        Returns:
            Dict: 주문 가능 정보
        """
        params = {"market": market}
        return self._make_request("GET", "/v1/orders/chance", params=params, auth_required=True)
    
    def get_orders(self, market: Optional[str] = None, uuids: Optional[List[str]] = None, 
                  identifiers: Optional[List[str]] = None, state: str = "wait", 
                  states: Optional[List[str]] = None, page: int = 1, limit: int = 100, 
                  order_by: str = "desc") -> List[Dict]:
        """
        주문 리스트 조회
        
        Args:
            market (str, optional): 마켓 코드
            uuids (List[str], optional): 주문 UUID 리스트
            identifiers (List[str], optional): 주문 식별자 리스트
            state (str): 주문 상태
            states (List[str], optional): 주문 상태 리스트
            page (int): 페이지 번호
            limit (int): 요청 개수
            order_by (str): 정렬 방식
            
        Returns:
            List[Dict]: 주문 리스트
        """
        params = {"state": state, "page": page, "limit": limit, "order_by": order_by}
        
        if market:
            params["market"] = market
        if uuids:
            params["uuids[]"] = uuids
        if identifiers:
            params["identifiers[]"] = identifiers
        if states:
            params["states[]"] = states
        
        return self._make_request("GET", "/v1/orders", params=params, auth_required=True)
    
    def get_orders_open(self, market: str = None, states: List[str] = None, 
                       page: int = 1, limit: int = 100, order_by: str = "desc") -> List[Dict]:
        """
        체결 대기 주문 조회
        
        Args:
            market (str, optional): 마켓 코드
            states (List[str], optional): 주문 상태 리스트
            page (int): 페이지 번호
            limit (int): 요청 개수
            order_by (str): 정렬 방식
            
        Returns:
            List[Dict]: 체결 대기 주문 리스트
        """
        params = {"page": page, "limit": limit, "order_by": order_by}
        
        if market:
            params["market"] = market
        if states:
            params["states[]"] = states
        
        return self._make_request("GET", "/v1/orders/open", params=params, auth_required=True)
    
    def get_orders_closed(self, market: str = None, states: List[str] = None, 
                         start_time: str = None, end_time: str = None, 
                         page: int = 1, limit: int = 100, order_by: str = "desc") -> List[Dict]:
        """
        체결 완료 주문 조회
        
        Args:
            market (str, optional): 마켓 코드
            states (List[str], optional): 주문 상태 리스트
            start_time (str, optional): 조회 시작 시간
            end_time (str, optional): 조회 종료 시간
            page (int): 페이지 번호
            limit (int): 요청 개수
            order_by (str): 정렬 방식
            
        Returns:
            List[Dict]: 체결 완료 주문 리스트
        """
        params = {"page": page, "limit": limit, "order_by": order_by}
        
        if market:
            params["market"] = market
        if states:
            params["states[]"] = states
        if start_time:
            params["start_time"] = start_time
        if end_time:
            params["end_time"] = end_time
        
        return self._make_request("GET", "/v1/orders/closed", params=params, auth_required=True)
    
    def get_order(self, uuid: str = None, identifier: str = None) -> Dict:
        """
        개별 주문 조회
        
        Args:
            uuid (str, optional): 주문 UUID
            identifier (str, optional): 주문 식별자
            
        Returns:
            Dict: 주문 정보
        """
        params = {}
        if uuid:
            params["uuid"] = uuid
        if identifier:
            params["identifier"] = identifier
        
        if not params:
            raise ValueError("uuid 또는 identifier 중 하나는 필수입니다.")
        
        return self._make_request("GET", "/v1/order", params=params, auth_required=True)
    
    def create_order(self, market: str, side: str, ord_type: str, 
                    volume: str = None, price: str = None, 
                    identifier: str = None, time_in_force: str = None) -> Dict:
        """
        주문하기
        
        Args:
            market (str): 마켓 코드
            side (str): 주문 종류 (bid: 매수, ask: 매도)
            ord_type (str): 주문 타입 (limit: 지정가, price: 시장가 매수, market: 시장가 매도)
            volume (str, optional): 주문 수량
            price (str, optional): 주문 가격
            identifier (str, optional): 조회용 사용자 지정값
            time_in_force (str, optional): IOC, FOK 주문
            
        Returns:
            Dict: 주문 결과
        """
        data = {
            "market": market,
            "side": side,
            "ord_type": ord_type
        }
        
        if volume:
            data["volume"] = volume
        if price:
            data["price"] = price
        if identifier:
            data["identifier"] = identifier
        if time_in_force:
            data["time_in_force"] = time_in_force
        
        return self._make_request("POST", "/v1/orders", data=data, auth_required=True)
    
    def cancel_order(self, uuid: str = None, identifier: str = None) -> Dict:
        """
        주문 취소
        
        Args:
            uuid (str, optional): 취소할 주문의 UUID
            identifier (str, optional): 취소할 주문의 식별자
            
        Returns:
            Dict: 주문 취소 결과
        """
        params = {}
        if uuid:
            params["uuid"] = uuid
        if identifier:
            params["identifier"] = identifier
        
        if not params:
            raise ValueError("uuid 또는 identifier 중 하나는 필수입니다.")
        
        return self._make_request("DELETE", "/v1/order", params=params, auth_required=True)
    
    def cancel_orders(self, uuids: List[str] = None, identifiers: List[str] = None) -> List[Dict]:
        """
        주문 일괄 취소
        
        Args:
            uuids (List[str], optional): 취소할 주문의 UUID 리스트
            identifiers (List[str], optional): 취소할 주문의 식별자 리스트
            
        Returns:
            List[Dict]: 주문 취소 결과 리스트
        """
        params = {}
        if uuids:
            params["uuids[]"] = uuids
        if identifiers:
            params["identifiers[]"] = identifiers
        
        if not params:
            raise ValueError("uuids 또는 identifiers 중 하나는 필수입니다.")
        
        return self._make_request("DELETE", "/v1/orders", params=params, auth_required=True)
    
    # =============================================================
    # 편의 메서드
    # =============================================================
    
    def buy_market_order(self, market: str, price: str) -> Dict:
        """
        시장가 매수 주문
        
        Args:
            market (str): 마켓 코드
            price (str): 매수 금액
            
        Returns:
            Dict: 주문 결과
        """
        return self.create_order(market=market, side="bid", ord_type="price", price=price)
    
    def sell_market_order(self, market: str, volume: str) -> Dict:
        """
        시장가 매도 주문
        
        Args:
            market (str): 마켓 코드
            volume (str): 매도 수량
            
        Returns:
            Dict: 주문 결과
        """
        return self.create_order(market=market, side="ask", ord_type="market", volume=volume)
    
    def buy_limit_order(self, market: str, volume: str, price: str) -> Dict:
        """
        지정가 매수 주문
        
        Args:
            market (str): 마켓 코드
            volume (str): 주문 수량
            price (str): 주문 가격
            
        Returns:
            Dict: 주문 결과
        """
        return self.create_order(market=market, side="bid", ord_type="limit", volume=volume, price=price)
    
    def sell_limit_order(self, market: str, volume: str, price: str) -> Dict:
        """
        지정가 매도 주문
        
        Args:
            market (str): 마켓 코드
            volume (str): 주문 수량
            price (str): 주문 가격
            
        Returns:
            Dict: 주문 결과
        """
        return self.create_order(market=market, side="ask", ord_type="limit", volume=volume, price=price)
    
    def get_balance(self, currency: str = None) -> Union[Dict, List[Dict]]:
        """
        잔고 조회
        
        Args:
            currency (str, optional): 통화 코드 (없으면 전체 조회)
            
        Returns:
            Union[Dict, List[Dict]]: 잔고 정보
        """
        accounts = self.get_accounts()
        
        if currency:
            for account in accounts:
                if account['currency'] == currency:
                    return account
            return None
        
        return accounts
    
    def get_current_price(self, market: str) -> float:
        """
        현재가 조회
        
        Args:
            market (str): 마켓 코드
            
        Returns:
            float: 현재가
        """
        ticker = self.get_ticker(market)[0]
        return float(ticker['trade_price'])


    # =============================================================
    # 주문 (Orders) - 인증 필요
    # =============================================================
    
    def place_buy_order(self, market: str, price: Optional[float] = None, volume: Optional[float] = None, 
                       ord_type: str = "limit") -> Dict:
        """
        매수 주문
        
        Args:
            market (str): 마켓 코드 (예: KRW-BTC)
            price (float): 주문 가격 (지정가 주문시)
            volume (float): 주문 수량 (지정가 주문시)
            ord_type (str): 주문 타입 (limit: 지정가, market: 시장가)
            
        Returns:
            Dict: 주문 결과
        """
        data = {
            "market": market,
            "side": "bid",
            "ord_type": ord_type
        }
        
        if ord_type == "limit":
            if price is None or volume is None:
                raise ValueError("지정가 주문시 가격과 수량이 필요합니다")
            data["price"] = str(price)
            data["volume"] = str(volume)
        elif ord_type == "price":
            if price is None:
                raise ValueError("시장가 매수시 주문 금액이 필요합니다")
            data["price"] = str(price)
        
        return self._make_request("POST", "/v1/orders", data=data, auth_required=True)
    
    def place_sell_order(self, market: str, price: Optional[float] = None, volume: Optional[float] = None, 
                        ord_type: str = "limit") -> Dict:
        """
        매도 주문
        
        Args:
            market (str): 마켓 코드
            price (float): 주문 가격 (지정가 주문시)
            volume (float): 주문 수량
            ord_type (str): 주문 타입 (limit: 지정가, market: 시장가)
            
        Returns:
            Dict: 주문 결과
        """
        if volume is None:
            raise ValueError("매도시 수량이 필요합니다")
        
        data = {
            "market": market,
            "side": "ask",
            "ord_type": ord_type,
            "volume": str(volume)
        }
        
        if ord_type == "limit":
            if price is None:
                raise ValueError("지정가 주문시 가격이 필요합니다")
            data["price"] = str(price)
        
        return self._make_request("POST", "/v1/orders", data=data, auth_required=True)
    
    def get_orders_simple(self, market: Optional[str] = None, state: str = "wait", 
                  page: int = 1, limit: int = 100) -> List[Dict]:
        """
        주문 목록 간단 조회
        
        Args:
            market (str, optional): 마켓 코드
            state (str): 주문 상태 (wait, done, cancel)
            page (int): 페이지 번호
            limit (int): 요청 개수 (최대 100)
            
        Returns:
            List[Dict]: 주문 목록
        """
        params = {
            "state": state,
            "page": page,
            "limit": limit
        }
        
        if market:
            params["market"] = market
        
        result = self._make_request("GET", "/v1/orders", params=params, auth_required=True)
        return result if isinstance(result, list) else [result]
    
    def cancel_order_simple(self, uuid: Optional[str] = None, identifier: Optional[str] = None) -> Dict:
        """
        주문 취소
        
        Args:
            uuid (str, optional): 주문 UUID
            identifier (str, optional): 조회용 사용자 지정값
            
        Returns:
            Dict: 취소 결과
        """
        if not uuid and not identifier:
            raise ValueError("UUID 또는 identifier 중 하나는 필요합니다")
        
        params = {}
        if uuid:
            params["uuid"] = uuid
        if identifier:
            params["identifier"] = identifier
        
        return self._make_request("DELETE", "/v1/order", params=params, auth_required=True)
    
    def get_order_simple(self, uuid: Optional[str] = None, identifier: Optional[str] = None) -> Dict:
        """
        개별 주문 조회
        
        Args:
            uuid (str, optional): 주문 UUID
            identifier (str, optional): 조회용 사용자 지정값
            
        Returns:
            Dict: 주문 정보
        """
        if not uuid and not identifier:
            raise ValueError("UUID 또는 identifier 중 하나는 필요합니다")
        
        params = {}
        if uuid:
            params["uuid"] = uuid
        if identifier:
            params["identifier"] = identifier
        
        return self._make_request("GET", "/v1/order", params=params, auth_required=True)


class UpbitWebSocket:
    """
    Upbit WebSocket 클라이언트 클래스
    
    실시간 시세 데이터 수신을 위한 WebSocket 연결을 관리합니다.
    """
    
    def __init__(self, access_key: Optional[str] = None, secret_key: Optional[str] = None):
        """
        UpbitWebSocket 클라이언트 초기화
        
        Args:
            access_key (str, optional): Upbit API Access Key
            secret_key (str, optional): Upbit API Secret Key
        """
        load_dotenv()
        
        self.access_key = access_key or os.getenv('UPBIT_ACCESS_KEY')
        self.secret_key = secret_key or os.getenv('UPBIT_SECRET_KEY')
        self.websocket_url = os.getenv('UPBIT_WEBSOCKET_URL', 'wss://api.upbit.com/websocket/v1')
        
        self.ws = None
        self.callbacks = {}
        self.running = False
    
    def _create_jwt_token(self) -> str:
        """
        WebSocket 연결용 JWT 토큰 생성
        
        Returns:
            str: JWT 토큰
        """
        if not self.access_key or not self.secret_key:
            raise ValueError("API 키가 설정되지 않았습니다. access_key와 secret_key를 확인해주세요.")
        
        payload = {
            "access_key": self.access_key,
            "nonce": str(uuid.uuid4())
        }
        
        token = jwt.encode(payload, self.secret_key, algorithm="HS512")
        return token if isinstance(token, str) else token.decode('utf-8')
    
    def on_message(self, ws, message):
        """WebSocket 메시지 수신 콜백"""
        try:
            data = json.loads(message.decode('utf-8'))
            msg_type = data.get('type', 'unknown')
            
            if msg_type in self.callbacks:
                self.callbacks[msg_type](data)
        except Exception as e:
            print(f"메시지 처리 오류: {e}")
    
    def on_error(self, ws, error):
        """WebSocket 오류 콜백"""
        print(f"WebSocket 오류: {error}")
    
    def on_close(self, ws, close_status_code, close_msg):
        """WebSocket 연결 종료 콜백"""
        print("WebSocket 연결이 종료되었습니다.")
        self.running = False
    
    def on_open(self, ws):
        """WebSocket 연결 시작 콜백"""
        print("WebSocket 연결이 시작되었습니다.")
        self.running = True
    
    def connect(self, private: bool = False):
        """
        WebSocket 연결
        
        Args:
            private (bool): Private 채널 사용 여부
        """
        url = self.websocket_url
        headers = {}
        
        if private:
            if url.endswith('/v1'):
                url += '/private'
            else:
                url += '/private'
            
            jwt_token = self._create_jwt_token()
            headers["Authorization"] = f"Bearer {jwt_token}"
        
        self.ws = websocket.WebSocketApp(
            url,
            header=headers,
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close,
            on_open=self.on_open
        )
        
        # 별도 스레드에서 실행
        self.ws_thread = threading.Thread(target=self.ws.run_forever)
        self.ws_thread.daemon = True
        self.ws_thread.start()
    
    def subscribe_ticker(self, markets: List[str], callback=None):
        """
        현재가 정보 구독
        
        Args:
            markets (List[str]): 마켓 코드 리스트
            callback: 데이터 수신 콜백 함수
        """
        if callback:
            self.callbacks['ticker'] = callback
        
        subscription = [
            {"ticket": str(uuid.uuid4())},
            {"type": "ticker", "codes": markets}
        ]
        
        if self.ws and self.running:
            self.ws.send(json.dumps(subscription))
    
    def subscribe_orderbook(self, markets: List[str], callback=None):
        """
        호가 정보 구독
        
        Args:
            markets (List[str]): 마켓 코드 리스트
            callback: 데이터 수신 콜백 함수
        """
        if callback:
            self.callbacks['orderbook'] = callback
        
        subscription = [
            {"ticket": str(uuid.uuid4())},
            {"type": "orderbook", "codes": markets}
        ]
        
        if self.ws and self.running:
            self.ws.send(json.dumps(subscription))
    
    def subscribe_trade(self, markets: List[str], callback=None):
        """
        체결 정보 구독
        
        Args:
            markets (List[str]): 마켓 코드 리스트
            callback: 데이터 수신 콜백 함수
        """
        if callback:
            self.callbacks['trade'] = callback
        
        subscription = [
            {"ticket": str(uuid.uuid4())},
            {"type": "trade", "codes": markets}
        ]
        
        if self.ws and self.running:
            self.ws.send(json.dumps(subscription))
    
    def disconnect(self):
        """WebSocket 연결 종료"""
        if self.ws:
            self.ws.close()
            self.running = False
