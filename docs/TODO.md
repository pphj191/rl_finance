# TODO 목록

> **최종 업데이트**: 2025년 10월 04일 22:27

## 🔴 우선순위 최고 (즉시 실행 필요)

### 1. 데이터 수집 및 저장 시스템 구축
- [ ] **SQLite 데이터베이스 설계**
  - [ ] 스키마 설계 (orderbook, trades, candles 테이블)
  - [ ] 인덱스 최적화
  - [ ] 데이터 무결성 제약조건 설정

- [ ] **실시간 데이터 수집 모듈 구현**
  - [ ] WebSocket 기반 orderbook 데이터 수집 (과거 조회 불가능하므로 실시간 저장 필수)
  - [ ] 거래 체결 데이터 수집
  - [ ] 캔들 데이터 수집 및 저장
  - [ ] 데이터 정합성 검증 로직

- [ ] **데이터 관리 유틸리티**
  - [ ] 데이터 백업 및 복구 기능
  - [ ] 오래된 데이터 정리 (cleanup) 기능
  - [ ] 데이터 조회 및 분석 헬퍼 함수

- [ ] **데이터 수집 스케줄러**
  - [ ] 24시간 연속 수집 시스템
  - [ ] 에러 복구 및 재연결 로직
  - [ ] 수집 상태 모니터링

### 2. 대용량 파일 분리 작업

#### models.py (504라인) → models/ 패키지 분리 ✅ **완료**
- [x] **models.py 백업 생성** (backup/models_backup.py)
- [x] **models.py 구조 분석** (DQN, LSTM, Transformer, Ensemble 모델 확인)
- [x] **models/ 패키지 디렉토리 생성** (__init__.py 포함)
- [x] **base_model.py 분리** (ModelConfig, 기본 설정 클래스)
- [x] **dqn.py 분리** (DQNModel 클래스)
- [x] **lstm.py 분리** (LSTMModel 클래스)
- [x] **transformer.py 분리** (TransformerModel, PositionalEncoding 클래스)
- [x] **ensemble.py 분리** (EnsembleModel 클래스)
- [x] **factory.py 완성** (create_model, 유틸리티 함수들)
- [x] **패키지 통합 및 테스트** (모든 모델 정상 동작 확인)

#### 🔄 현재 진행 중
- [ ] **원본 models.py 대체** (새 패키지로 교체)
- [ ] **프로젝트 전체 import 경로 수정** (models.py → models 패키지)

#### run_backtesting.py (509라인) → 기능별 분리
- [ ] **run_backtesting.py 분석** (구조 분석 및 분리 계획 수립)
- [ ] **백테스팅 핵심 로직 분리** (core/backtesting_engine.py)
- [ ] **성과 지표 계산 분리** (core/performance_metrics.py)
- [ ] **시각화 기능 분리** (core/visualization.py)
- [ ] **실행 스크립트 유지** (run_backtesting.py는 간단한 실행부만)

### 3. 패키지 통합 및 import 경로 수정
- [ ] **trading_env 패키지 import 수정**
  - [ ] 기존 `rl_trading_env` 사용 코드 찾기
  - [ ] 새 `trading_env` 패키지로 변경
  - [ ] 의존성 검증 및 테스트

- [ ] **API 패키지 경로 통일**
  - [ ] `upbit_api.upbit_api` → `upbit_api` 간소화
  - [ ] `bithumb_api.bithumb_api` → `bithumb_api` 간소화

## 🟡 우선순위 높음 (이번 주 완료)

### 4. 중간 크기 파일 최적화
- [ ] **run_real_time_trader.py (480라인)** 리팩토링
  - [ ] 실시간 데이터 수집 로직 분리
  - [ ] 거래 실행 로직 분리
  - [ ] 위험 관리 로직 분리

- [ ] **dqn_agent.py (444라인)** 개선
  - [ ] 에이전트 기본 클래스 분리
  - [ ] 메모리 관리 로직 분리
  - [ ] 학습 알고리즘 개선

### 5. 테스트 및 예제 완성
- [ ] **core 기능 테스트 작성**
  - [ ] `tests/test_models.py`
  - [ ] `tests/test_backtesting.py`
  - [ ] `tests/test_trading_env.py`
  - [ ] `tests/test_data_collector.py` (데이터 수집 테스트)

- [ ] **사용 예제 추가**
  - [ ] `examples/example_model_training.py`
  - [ ] `examples/example_backtesting_simple.py`
  - [ ] `examples/example_real_trading.py`
  - [ ] `examples/example_data_collection.py` (데이터 수집 예제)

### 6. 문서화 완성
- [ ] **API 문서 자동 생성**
  - [ ] docstring 표준화
  - [ ] Sphinx 설정 추가
  - [ ] `docs/` 폴더 활용

- [ ] **사용자 가이드 작성**
  - [ ] 설치 가이드
  - [ ] 빠른 시작 가이드
  - [ ] 고급 사용법 가이드
  - [ ] 데이터 수집 및 관리 가이드

## 🟢 우선순위 중간 (장기 계획)

### 7. 성능 최적화
- [ ] **메모리 사용량 최적화**
  - [ ] 대용량 데이터 스트리밍 처리
  - [ ] 메모리 누수 방지
  - [ ] 가비지 컬렉션 최적화
  - [ ] SQLite 연결 풀링 최적화

- [ ] **처리 속도 개선**
  - [ ] 멀티프로세싱 백테스팅
  - [ ] GPU 가속 모델 학습
  - [ ] 병렬 데이터 처리
  - [ ] SQLite 쿼리 최적화

### 8. 새로운 기능 추가
- [ ] **추가 거래소 지원**
  - [ ] 바이낸스 API 연동
  - [ ] 코인원 API 연동
  - [ ] 다중 거래소 통합 인터페이스

- [ ] **고급 기능 개발**
  - [ ] 실시간 알림 시스템
  - [ ] 웹 대시보드 개발
  - [ ] 포트폴리오 관리 시스템
  - [ ] 데이터 분석 대시보드

### 9. 운영 및 보안
- [ ] **보안 강화**
  - [ ] API 키 암호화 저장
  - [ ] 로그 민감정보 마스킹
  - [ ] 권한별 접근 제어

- [ ] **모니터링 시스템**
  - [ ] 실시간 성능 모니터링
  - [ ] 오류 로깅 시스템
  - [ ] 알림 및 경고 시스템
  - [ ] 데이터 수집 상태 모니터링

## 📊 작업 진행률 추적

### 완료된 작업 ✅
- ✅ **파일 구조 정리** (trading_env 패키지 분리)
- ✅ **디렉토리 구조 생성** (8개 전용 폴더)
- ✅ **파일 명명 규칙 통일** (run_ 접두사 적용)
- ✅ **개발 지침 문서화** (INSTRUCTIONS.md 완성)
- ✅ **models.py 패키지 분리** (models/ 패키지 완성)

### 진행중인 작업 🔄
- 🔄 **models.py 대체 작업** (새 패키지로 교체)
- 🔄 **import 경로 정리** (프로젝트 전체 수정)
- 🔄 **run_backtesting.py 분석** (분리 계획 수립)

### 대기중인 작업 ⏳
- ⏳ **데이터 수집 시스템 구축** (SQLite 기반)
- ⏳ **백테스팅 파일 분리**
- ⏳ **중간 크기 파일 최적화**
- ⏳ **테스트 코드 작성**
- ⏳ **문서화 자동화**

**현재 진행률: 50% (5/10 주요 작업 완료)**

## 🗓️ 단계별 실행 계획

### 📋 1단계: 즉시 실행 (이번 주 목표) 🔴
1. **데이터 수집 시스템 구축**
   - SQLite 데이터베이스 스키마 설계
   - WebSocket 기반 orderbook 실시간 수집
   - 데이터 저장 및 관리 로직 구현

2. **models.py 대체 완료**
   - 새 models 패키지로 원본 models.py 교체
   - 프로젝트 전체 import 경로 수정
   - 동작 확인 및 검증

3. **run_backtesting.py 분석 시작**
   - 509라인 파일 구조 분석
   - 핵심 로직과 실행 스크립트 구분
   - 분리 계획 수립

### 📋 2단계: 이번 달 완료 (10월 전체) 🟡
1. **대용량 파일 분리 완료**
   - `models/` 패키지 분리 작업 완료
   - `run_backtesting.py` 기능별 분리 완료
   - 모든 import 경로 수정 완료

2. **중간 크기 파일 최적화**
   - `run_real_time_trader.py` 리팩토링
   - `dqn_agent.py` 개선 작업
   - 코드 품질 향상

3. **기본 테스트 작성**
   - 핵심 기능 단위 테스트
   - 데이터 수집 테스트
   - 통합 테스트 기본 틀 구성

### 📋 3단계: 장기 계획 (11월 이후) 🟢
1. **문서화 및 가이드 완성**
   - API 문서 자동 생성
   - 사용자 가이드 작성
   - 예제 코드 보완

2. **성능 최적화 및 새 기능**
   - 메모리 사용량 최적화
   - 추가 거래소 지원 검토
   - 실시간 모니터링 시스템 설계

3. **운영 환경 준비**
   - 보안 강화 방안 적용
   - 배포 자동화 시스템 구축
   - 사용자 피드백 수집 시스템

---

**이 문서는 프로젝트와 함께 지속적으로 업데이트됩니다.**
