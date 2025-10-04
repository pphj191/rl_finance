# Bithumb API 클라이언트

빗썸 거래소 REST API 및 WebSocket을 위한 Python 클라이언트 라이브러리입니다.
Upbit API와 동일한 함수명과 인터페이스를 제공하여 일관된 사용 경험을 제공합니다.

## 🌟 주요 특징

- **Upbit 호환**: Upbit API와 동일한 함수명 사용
- **완전한 API 커버리지**: 시세, 계정, 주문 모든 API 지원
- **WebSocket 지원**: 실시간 데이터 스트림
- **자동 재시도**: 네트워크 오류 시 자동 재시도
- **Type Hints**: 완전한 타입 힌트 지원
- **에러 처리**: 상세한 에러 정보 제공

## 📦 설치

```bash
# 의존성 패키지
uv add requests websocket-client python-dotenv
```

## 🔧 설정

### 환경 변수 설정

`.env` 파일을 생성하고 Bithumb API 키를 설정하세요:

```env
BITHUMB_ACCESS_KEY=your_access_key_here
BITHUMB_SECRET_KEY=your_secret_key_here
```

### API 키 발급

1. [빗썸 홈페이지](https://www.bithumb.com/) 접속
2. 로그인 후 "MY > API 관리" 메뉴 이동
3. API 키 발급 (거래용 권한 필요 시 별도 설정)

## 💡 사용법

### 기본 사용

```python
from bithumb_api import BithumbAPI

# API 클라이언트 생성
api = BithumbAPI()

# 또는 직접 키 입력
api = BithumbAPI(
    access_key="your_access_key",
    secret_key="your_secret_key"
)
```

### 시세 정보 조회 (Public API)

```python
# 마켓 코드 조회
markets = api.get_market_all()
print(f"마켓 수: {len(markets)}")

# 현재가 정보
ticker = api.get_ticker('KRW-BTC')
print(f"BTC 현재가: {ticker['trade_price']:,} KRW")

# 여러 마켓 현재가
tickers = api.get_ticker(['KRW-BTC', 'KRW-ETH'])
for ticker in tickers:
    print(f"{ticker['market']}: {ticker['trade_price']:,} KRW")

# 분 캔들 조회
candles = api.get_candles_minutes(unit=1, market='KRW-BTC', count=10)
print(f"최근 10개 1분 캔들: {len(candles)}개")

# 호가 정보
orderbook = api.get_orderbook('KRW-BTC')
print(f"매수 호가: {orderbook['orderbook_units'][0]['bid_price']:,}")
print(f"매도 호가: {orderbook['orderbook_units'][0]['ask_price']:,}")

# 최근 체결 내역
trades = api.get_trades_ticks('KRW-BTC', count=5)
print(f"최근 5개 체결: {len(trades)}개")
```

### 계정 및 주문 관리 (Private API)

```python
# 계좌 조회
accounts = api.get_accounts()
for account in accounts:
    if float(account['balance']) > 0:
        print(f"{account['currency']}: {account['balance']}")

# 지정가 매수 주문
order_result = api.order(
    market='KRW-BTC',
    side='bid',          # bid: 매수, ask: 매도
    volume='0.001',      # 주문량
    price='50000000',    # 주문가격
    ord_type='limit'     # limit: 지정가
)
print(f"주문 UUID: {order_result['uuid']}")

# 주문 조회
order_info = api.get_order(uuid=order_result['uuid'])
print(f"주문 상태: {order_info['state']}")

# 주문 취소
cancel_result = api.cancel_order(uuid=order_result['uuid'])
print(f"취소 완료: {cancel_result['uuid']}")

# 주문 리스트 조회
orders = api.get_orders(market='KRW-BTC', state='wait')
print(f"대기 중인 주문: {len(orders)}개")
```

### WebSocket 실시간 데이터

```python
from bithumb_api import BithumbWebSocket
import json

def handle_message(message):
    """메시지 처리 함수"""
    try:
        data = json.loads(message)
        if data.get('type') == 'ticker':
            print(f"실시간 현재가: {data.get('trade_price', 0):,} KRW")
    except:
        pass

# WebSocket 연결
ws = BithumbWebSocket()
ws.connect(
    callback=handle_message,
    markets=['KRW-BTC'],
    types=['ticker']
)

# 연결 상태 확인
if ws.is_alive():
    print("WebSocket 연결됨")

# 연결 해제
# ws.disconnect()
```

## 🔍 API 레퍼런스

### BithumbAPI 클래스

#### 생성자
```python
BithumbAPI(access_key=None, secret_key=None, config=None)
```

#### 시세 정보 메서드

| 메서드 | 설명 | Upbit 호환 |
|--------|------|-----------|
| `get_market_all()` | 마켓 코드 조회 | ✅ |
| `get_candles_minutes(unit, market, to, count)` | 분 캔들 조회 | ✅ |
| `get_ticker(markets)` | 현재가 정보 | ✅ |
| `get_orderbook(markets)` | 호가 정보 | ✅ |
| `get_trades_ticks(market, to, count, cursor)` | 체결 내역 | ✅ |

#### 계정 관리 메서드

| 메서드 | 설명 | Upbit 호환 |
|--------|------|-----------|
| `get_accounts()` | 전체 계좌 조회 | ✅ |
| `get_order(uuid, identifier)` | 개별 주문 조회 | ✅ |
| `get_orders(market, state, page, limit)` | 주문 리스트 조회 | ✅ |
| `cancel_order(uuid, identifier)` | 주문 취소 | ✅ |
| `order(market, side, volume, price, ord_type)` | 주문하기 | ✅ |

### 편의 함수들

Upbit 함수명을 그대로 사용하는 편의 함수들:

```python
from bithumb_api import (
    get_upbit_market_all,
    get_upbit_candles_minutes,
    get_upbit_ticker,
    get_upbit_orderbook
)

# Upbit과 동일한 함수명으로 사용 가능
markets = get_upbit_market_all()
ticker = get_upbit_ticker('KRW-BTC')
```

## 📊 응답 데이터 형식

모든 응답은 Upbit API와 동일한 형식으로 변환됩니다.

### 현재가 정보 (Ticker)

```python
{
    'market': 'KRW-BTC',
    'trade_date': '20240101',
    'trade_time': '123000',
    'trade_price': 50000000.0,
    'change': 'RISE',
    'change_price': 1000000.0,
    'change_rate': 0.02,
    'prev_closing_price': 49000000.0,
    'acc_trade_volume': 100.5,
    'acc_trade_price': 5000000000.0,
    'highest_52_week_price': 80000000.0,
    'lowest_52_week_price': 30000000.0,
    'timestamp': 1640995200000
}
```

### 호가 정보 (Orderbook)

```python
{
    'market': 'KRW-BTC',
    'timestamp': 1640995200000,
    'total_ask_size': 10.0,
    'total_bid_size': 15.0,
    'orderbook_units': [
        {
            'ask_price': 50010000.0,
            'bid_price': 49990000.0,
            'ask_size': 0.5,
            'bid_size': 0.8
        }
    ]
}
```

## ⚠️ 주의사항

### API 제한사항

- **요청 제한**: Bithumb API 호출 제한을 준수하세요
- **시세 API**: 1초당 10회 제한
- **거래 API**: 1초당 5회 제한

### 에러 처리

```python
from bithumb_api import BithumbAPIError

try:
    result = api.get_ticker('INVALID-MARKET')
except BithumbAPIError as e:
    print(f"API 에러: {e.message}")
    print(f"에러 코드: {e.error_code}")
except Exception as e:
    print(f"기타 에러: {e}")
```

### 보안

- API 키를 코드에 직접 입력하지 마세요
- `.env` 파일을 Git에 커밋하지 마세요
- 거래 권한은 필요한 경우에만 부여하세요

## 🔄 Upbit에서 Bithumb으로 마이그레이션

기존 Upbit 코드를 거의 수정 없이 사용할 수 있습니다:

```python
# 기존 Upbit 코드
from upbit_api import UpbitAPI
upbit = UpbitAPI()
ticker = upbit.get_ticker('KRW-BTC')

# Bithumb 코드 (동일한 인터페이스)
from bithumb_api import BithumbAPI
bithumb = BithumbAPI()
ticker = bithumb.get_ticker('KRW-BTC')  # 동일한 함수명
```

## 🧪 테스트

```python
# API 연결 테스트
from bithumb_api import BithumbAPI

api = BithumbAPI()

try:
    markets = api.get_market_all()
    print(f"✅ API 연결 성공: {len(markets)}개 마켓")
except Exception as e:
    print(f"❌ API 연결 실패: {e}")
```

## 📚 관련 문서

- [Bithumb API 공식 문서](https://apidocs.bithumb.com/)
- [Bithumb 거래소](https://www.bithumb.com/)
- [Upbit API 호환성 가이드](../upbit_api/README.md)

## 🤝 기여

버그 리포트나 기능 제안은 GitHub Issues에 등록해 주세요.

## 📄 라이선스

MIT License

---

**⚡ 빗썸과 함께하는 안전한 거래! ⚡**
