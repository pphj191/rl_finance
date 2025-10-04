# Upbit API Python Client

Upbit 거래소의 REST API와 WebSocket을 사용하여 시세 조회, 주문, 자산 관리 등의 기능을 제공하는 Python 클라이언트입니다.

## 주요 기능

### 시세 정보 (인증 불필요)
- 마켓 코드 목록 조회
- 캔들 데이터 조회 (분/일/주/월)
- 현재가 정보 조회
- 호가 정보 조회
- 최근 체결 내역 조회

### 자산 및 주문 관리 (인증 필요)
- 계좌 정보 조회
- 주문 가능 정보 조회
- 주문하기 (시장가/지정가 매수/매도)
- 주문 취소
- 주문 내역 조회

### 실시간 데이터 (WebSocket)
- 실시간 현재가 정보
- 실시간 호가 정보
- 실시간 체결 정보

## 설치

1. 필요한 패키지 설치:
```bash
pip install -r requirements.txt
```

2. 환경 변수 설정:
`.env.example` 파일을 `.env`로 복사하고 API 키를 입력합니다.

```bash
cp .env.example .env
```

`.env` 파일에서 다음 항목을 수정합니다:
```
UPBIT_ACCESS_KEY=your_actual_access_key
UPBIT_SECRET_KEY=your_actual_secret_key
```

## API 키 발급 방법

1. [Upbit](https://upbit.com) 웹사이트에 로그인
2. 마이페이지 > Open API 관리 메뉴 이동
3. Open API 키 발급
4. 필요한 권한 설정:
   - 자산 조회
   - 주문 조회
   - 주문 등록/취소 (거래 시 필요)
5. IP 주소 등록 (최대 10개)

## 사용 예제

### 기본 사용법

```python
from upbit_api import UpbitAPI

# API 클라이언트 초기화 (.env에서 자동 로드)
upbit = UpbitAPI()

# 또는 직접 API 키 지정
# upbit = UpbitAPI(access_key="your_key", secret_key="your_secret")

# 현재가 조회
ticker = upbit.get_ticker("KRW-BTC")
print(f"BTC 현재가: {ticker[0]['trade_price']:,} KRW")

# 계좌 정보 조회 (인증 필요)
accounts = upbit.get_accounts()
for account in accounts:
    if float(account['balance']) > 0:
        print(f"{account['currency']}: {account['balance']}")
```

### 주문 예제

```python
# 시장가 매수 (5,000원)
result = upbit.buy_market_order('KRW-BTC', '5000')

# 지정가 매수 (0.001 BTC를 50,000,000원에)
result = upbit.buy_limit_order('KRW-BTC', '0.001', '50000000')

# 지정가 매도 (0.001 BTC를 60,000,000원에)
result = upbit.sell_limit_order('KRW-BTC', '0.001', '60000000')

# 주문 취소
upbit.cancel_order(uuid=result['uuid'])
```

### WebSocket 실시간 데이터

```python
from upbit_api import UpbitWebSocket

# 실시간 현재가 콜백
def on_ticker(data):
    print(f"{data['code']}: {data['trade_price']:,} KRW")

# WebSocket 클라이언트 생성
ws_client = UpbitWebSocket()
ws_client.connect()

# 실시간 현재가 구독
ws_client.subscribe_ticker(['KRW-BTC', 'KRW-ETH'], on_ticker)
```

### 전체 예제 실행

```bash
python example.py
```

## API 클래스 주요 메서드

### UpbitAPI

#### 시세 정보
- `get_markets()`: 마켓 코드 목록
- `get_ticker(markets)`: 현재가 정보
- `get_orderbook(markets)`: 호가 정보
- `get_candles_minutes(market, unit, count)`: 분 캔들
- `get_candles_days(market, count)`: 일 캔들
- `get_trades_ticks(market, count)`: 최근 체결

#### 계좌 및 주문
- `get_accounts()`: 계좌 정보
- `get_orders_chance(market)`: 주문 가능 정보
- `create_order(market, side, ord_type, volume, price)`: 주문
- `cancel_order(uuid)`: 주문 취소
- `get_orders_open()`: 체결 대기 주문
- `get_orders_closed()`: 체결 완료 주문

#### 편의 메서드
- `buy_market_order(market, price)`: 시장가 매수
- `sell_market_order(market, volume)`: 시장가 매도
- `buy_limit_order(market, volume, price)`: 지정가 매수
- `sell_limit_order(market, volume, price)`: 지정가 매도
- `get_balance(currency)`: 특정 통화 잔고
- `get_current_price(market)`: 현재가 조회

### UpbitWebSocket

- `connect(private=False)`: WebSocket 연결
- `subscribe_ticker(markets, callback)`: 현재가 구독
- `subscribe_orderbook(markets, callback)`: 호가 구독
- `subscribe_trade(markets, callback)`: 체결 구독
- `disconnect()`: 연결 종료

## 주의사항

1. **API 키 보안**: API 키는 절대 공개하지 마세요. `.env` 파일을 버전 관리에 포함하지 마세요.

2. **요청 제한**: Upbit API는 요청 횟수 제한이 있습니다. 과도한 요청을 피하세요.

3. **테스트**: 실제 거래 전에 충분히 테스트하세요. 소액으로 먼저 테스트해보는 것을 권장합니다.

4. **IP 등록**: API 키에 등록된 IP에서만 API를 사용할 수 있습니다.

5. **권한 설정**: 필요한 권한만 설정하세요. 불필요한 권한은 보안 위험을 증가시킵니다.

## 오류 처리

API 호출 시 발생할 수 있는 주요 오류:

- **인증 오류**: API 키가 잘못되었거나 IP가 등록되지 않음
- **권한 오류**: API 키에 필요한 권한이 없음
- **요청 제한**: 요청 횟수 제한 초과
- **잘못된 파라미터**: 마켓 코드나 주문 정보가 잘못됨

## 라이선스

이 프로젝트는 MIT 라이선스 하에 제공됩니다.

## 참고 자료

- [Upbit 개발자 센터](https://docs.upbit.com/kr)
- [Upbit API 문서](https://docs.upbit.com/kr/reference)
- [Upbit Open API 이용약관](https://upbit.com/open_api_agreement)
