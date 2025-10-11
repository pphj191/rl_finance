# Upbit API 리팩토링 TODO

## 파일 분리 (Mixin 기반)

### 목표
- `upbit_api.py`의 거대한 단일 클래스를 mixin 패턴을 활용하여 여러 파일로 분리
- 코드 유지보수성 향상 및 관심사 분리
- 각 mixin이 특정 도메인 기능 담당

### 현재 구조 분석
- `UpbitAPI`: 1000줄 이상의 거대한 클래스
- `UpbitWebSocket`: WebSocket 관련 기능
- 모든 기능이 하나의 파일에 집중

### 계획된 파일 구조
```
upbit_api/
├── __init__.py
├── base.py              # BaseAPI (공통 기능)
├── quotation.py         # QuotationMixin (시세 정보)
├── exchange.py          # ExchangeMixin (자산 관리)
├── orders.py            # OrdersMixin (주문 관리)
├── websocket.py         # WebSocket 기능
├── mixins.py            # Mixin 조합 클래스
└── upbit_api.py         # 기존 파일 (호환성 유지)
```

### TODO 항목

#### 1. BaseAPI 클래스 분리
- [ ] `base.py` 생성
- [ ] 공통 초기화 로직 이동 (`__init__`, `_setup_logging`)
- [ ] Rate limit 관련 메서드 이동 (`_wait_for_rate_limit`)
- [ ] HTTP 요청 공통 로직 이동 (`_make_request`)
- [ ] JWT 토큰 생성 이동 (`_create_jwt_token`)
- [ ] 쿼리 스트링 생성 이동 (`_build_query_string`)

#### 2. QuotationMixin 생성
- [ ] `quotation.py` 생성
- [ ] 시세 정보 관련 메서드 이동:
  - `get_markets()`
  - `get_candles_*()` (모든 캔들 관련)
  - `get_trades_ticks()`
  - `get_ticker()`
  - `get_orderbook()`
  - `get_current_price()`

#### 3. ExchangeMixin 생성
- [ ] `exchange.py` 생성
- [ ] 자산 관리 관련 메서드 이동:
  - `get_accounts()`
  - `get_balance()`

#### 4. OrdersMixin 생성
- [ ] `orders.py` 생성
- [ ] 주문 관리 관련 메서드 이동:
  - `get_orders_chance()`
  - `get_orders()`, `get_orders_open()`, `get_orders_closed()`
  - `get_order()`, `get_orders_simple()`, `get_order_simple()`
  - `create_order()`, `cancel_order()`, `cancel_orders()`
  - `place_buy_order()`, `place_sell_order()`
  - `cancel_order_simple()`
  - 편의 메서드들 (`buy_market_order`, `sell_market_order`, etc.)

#### 5. WebSocket 기능 분리
- [ ] `websocket.py` 생성
- [ ] `UpbitWebSocket` 클래스 이동
- [ ] WebSocket 관련 메서드들 이동

#### 6. Mixin 조합 클래스 생성
- [ ] `mixins.py` 생성
- [ ] `UpbitAPI` 클래스 생성 (모든 mixin 상속)
- [ ] 다중 상속을 통한 기능 조합

#### 7. 호환성 유지
- [ ] `__init__.py`에서 `UpbitAPI` import 유지
- [ ] 기존 `upbit_api.py`는 mixin 기반 클래스로 변경
- [ ] 하위 호환성 보장

#### 8. 테스트 및 검증
- [ ] 각 mixin 별 단위 테스트
- [ ] 통합 테스트 (기존 코드와 동일한 동작 확인)
- [ ] import 및 사용성 검증
- [ ] 문서 업데이트

### 우선순위
1. BaseAPI 분리 (핵심 인프라)
2. QuotationMixin 분리 (가장 간단한 도메인)
3. OrdersMixin 분리 (가장 복잡한 도메인)
4. ExchangeMixin 분리
5. WebSocket 분리
6. Mixin 조합 및 호환성 유지

### 예상 이점
- 코드 가독성 향상
- 유지보수 용이성
- 기능 확장성 향상
- 테스트 용이성
- 관심사 분리

### 주의사항
- Rate limit 공유 상태 유지 (클래스 변수)
- 로깅 설정 일관성 유지
- API 키 및 설정 공유
- 기존 사용자 코드 호환성 보장