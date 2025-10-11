# Reward Function Design Guide

> **작성일**: 2025-10-07
> **목적**: RL 에이전트의 리워드 함수 설계 및 비교 분석

---

## 📋 목차

1. [현재 리워드 시스템의 문제점](#현재-리워드-시스템의-문제점)
2. [리워드 설계 원칙](#리워드-설계-원칙)
3. [제안된 리워드 함수들](#제안된-리워드-함수들)
4. [구현 계획](#구현-계획)
5. [실험 및 평가](#실험-및-평가)

---

## 현재 리워드 시스템의 문제점

### 현재 구현 (rl_env.py:262-269)
```python
# 포트폴리오 가치 변화율
prev_total_value = self.total_value
# ... 액션 실행 ...
self.total_value = self.balance + self.position * current_price

if prev_total_value > 0:
    reward = (self.total_value - prev_total_value) / prev_total_value
else:
    reward = (self.total_value - prev_total_value) / self.config.initial_balance
```

### 문제점 분석

#### 1. **매도 회피 문제 (Sell Aversion)**
- **현상**: 에이전트가 매도를 거의 하지 않음
- **원인**:
  - 수수료(0.05%)로 인한 즉각적인 손실
  - 매도 후 기회비용에 대한 고려 없음
  - 가격 상승 시 보유만으로도 양의 리워드 획득

**예시**:
```
BTC 가격: 50,000,000원
보유 중: 0.01 BTC

# 스텝 N: HOLD
가격 1% 상승 → 50,500,000원
total_value: 505,000 → 510,050
reward = (510,050 - 505,000) / 505,000 = 0.01 (1% 리워드)

# 스텝 N+1: SELL 고려
매도 시: 수수료 0.05%
balance = 0.01 * 50,500,000 * 0.9995 = 504,747
reward = (504,747 - 510,050) / 510,050 = -0.0104 (-1.04% 리워드)

→ 매도하면 마이너스 리워드! 매도 안 하는게 이득
```

#### 2. **단기 가격 변동에 과도한 민감성**
- 매 스텝마다 가격 변화만 반영
- 장기 트렌드 무시
- 노이즈에 취약

#### 3. **리스크 무시**
- 변동성에 대한 페널티 없음
- 최대 낙폭(MDD) 고려 안 함
- 과도한 거래에 대한 페널티 없음

#### 4. **희소 리워드 (Sparse Reward)**
- HOLD 시 가격 변동만 리워드
- BUY/SELL의 질(quality) 평가 없음
- 타이밍 좋은 거래에 대한 보상 부족

---

## 리워드 설계 원칙

### 1. **매도 인센티브 (Sell Incentive)**
✅ 수익 실현 시 보너스 제공
✅ 적절한 손절에 대한 최소 페널티
✅ 과도한 보유에 대한 페널티

### 2. **위험 조정 수익률 (Risk-Adjusted Returns)**
✅ 변동성 고려
✅ MDD 페널티
✅ 샤프 비율 기반 보상

### 3. **행동 품질 평가 (Action Quality)**
✅ 좋은 타이밍 매수/매도 보너스
✅ 나쁜 타이밍 페널티
✅ 과도한 거래 억제

### 4. **장기 목표 정렬 (Long-term Alignment)**
✅ 에피소드 전체 수익률 중시
✅ 단기 노이즈 무시
✅ 일관된 전략 유도

---

## 제안된 리워드 함수들

### 리워드 1: 매도 인센티브 추가 (Sell Incentive)

```python
def reward_with_sell_incentive(self, action, prev_total_value, current_total_value):
    """
    기본 수익률 + 매도 인센티브
    """
    # 기본 수익률
    base_reward = (current_total_value - prev_total_value) / prev_total_value

    # 매도 인센티브
    if action == ActionSpace.SELL and self.last_buy_price > 0:
        # 수익 실현 보너스
        profit_rate = (self.sell_price - self.last_buy_price) / self.last_buy_price

        if profit_rate > 0:
            # 수익 매도: 큰 보너스
            sell_bonus = profit_rate * 0.5  # 수익의 50%를 추가 리워드
        elif profit_rate > -0.02:
            # 소액 손실 매도: 작은 보너스 (빠른 손절 장려)
            sell_bonus = 0.005
        else:
            # 큰 손실 매도: 작은 페널티
            sell_bonus = profit_rate * 0.1

        base_reward += sell_bonus

    # 장기 보유 페널티 (30 스텝 이상 보유 시)
    if self.position > 0 and self.hold_duration > 30:
        hold_penalty = -0.0001 * self.hold_duration
        base_reward += hold_penalty

    return base_reward
```

**장점**:
- ✅ 매도 시 수익 실현 보너스
- ✅ 손절 시 최소 페널티
- ✅ 과도한 보유 억제

**단점**:
- ⚠️ 파라미터 튜닝 필요
- ⚠️ 짧은 에피소드에서는 효과 제한적

---

### 리워드 2: 위험 조정 수익률 (Risk-Adjusted Return)

```python
def reward_risk_adjusted(self, action, prev_total_value, current_total_value, window=20):
    """
    샤프 비율 기반 리워드
    """
    # 포트폴리오 수익률
    portfolio_return = (current_total_value - prev_total_value) / prev_total_value

    # 최근 수익률 변동성 계산
    self.recent_returns.append(portfolio_return)
    if len(self.recent_returns) > window:
        self.recent_returns.pop(0)

    if len(self.recent_returns) >= 2:
        returns_std = np.std(self.recent_returns)
        # 샤프 비율 스타일 리워드 (무위험 수익률 = 0)
        reward = portfolio_return / (returns_std + 1e-6)
    else:
        reward = portfolio_return

    # MDD 페널티
    if current_total_value < self.peak_value:
        drawdown = (self.peak_value - current_total_value) / self.peak_value
        reward -= drawdown * 0.5  # MDD 페널티
    else:
        self.peak_value = current_total_value

    return reward
```

**장점**:
- ✅ 변동성 낮은 안정적 수익 선호
- ✅ 최대 낙폭 최소화
- ✅ 위험 관리 학습

**단점**:
- ⚠️ 초기 학습 불안정
- ⚠️ 계산 복잡도 증가

---

### 리워드 3: 벤치마크 대비 초과 수익 (Benchmark Excess Return)

```python
def reward_benchmark_excess(self, action, prev_total_value, current_total_value):
    """
    Buy & Hold 대비 초과 수익
    """
    # 포트폴리오 수익률
    portfolio_return = (current_total_value - prev_total_value) / prev_total_value

    # 벤치마크 수익률 (단순 보유)
    benchmark_return = (self.current_price - self.prev_price) / self.prev_price

    # 초과 수익
    excess_return = portfolio_return - benchmark_return

    return excess_return
```

**장점**:
- ✅ 단순 보유보다 나은 전략 학습
- ✅ 절대 수익이 아닌 상대 성과 중시
- ✅ 시장 상황 무관하게 학습

**단점**:
- ⚠️ 음의 리워드 빈번 (초기 학습 어려움)
- ⚠️ 벤치마크 선택에 따라 결과 변동

---

### 리워드 4: 행동 품질 기반 (Action Quality-Based)

```python
def reward_action_quality(self, action, prev_total_value, current_total_value):
    """
    매수/매도 타이밍 품질 평가
    """
    base_reward = (current_total_value - prev_total_value) / prev_total_value

    # 가격 추세 계산 (최근 5스텝 기울기)
    price_trend = (self.current_price - self.price_5_steps_ago) / self.price_5_steps_ago

    if action == ActionSpace.BUY:
        if price_trend < -0.01:  # 하락 추세에서 매수 → 좋은 타이밍
            base_reward += 0.01
        elif price_trend > 0.02:  # 급등 후 매수 → 나쁜 타이밍
            base_reward -= 0.01

    elif action == ActionSpace.SELL:
        if price_trend > 0.02:  # 상승 추세에서 매도 → 좋은 타이밍
            base_reward += 0.01
        elif price_trend < -0.01:  # 하락 추세에서 매도 → 늦은 매도
            base_reward -= 0.005

    # 과도한 거래 페널티
    self.trade_count += (action != ActionSpace.HOLD)
    if self.trade_count > 50:  # 50회 이상 거래 시
        base_reward -= 0.0001 * self.trade_count

    return base_reward
```

**장점**:
- ✅ 좋은 타이밍 학습
- ✅ 과도한 거래 억제
- ✅ 해석 가능성

**단점**:
- ⚠️ 추세 판단 로직 필요
- ⚠️ 하이퍼파라미터 많음

---

### 리워드 5: 복합 리워드 (Hybrid Reward)

```python
def reward_hybrid(self, action, prev_total_value, current_total_value):
    """
    여러 리워드 요소를 가중 결합
    """
    # 1. 기본 수익률 (가중치: 0.5)
    base_reward = (current_total_value - prev_total_value) / prev_total_value

    # 2. 매도 인센티브 (가중치: 0.2)
    sell_incentive = self._calculate_sell_incentive(action)

    # 3. 위험 조정 (가중치: 0.2)
    risk_penalty = self._calculate_risk_penalty()

    # 4. 행동 품질 (가중치: 0.1)
    action_quality = self._calculate_action_quality(action)

    # 가중 결합
    reward = (
        0.5 * base_reward +
        0.2 * sell_incentive +
        0.2 * risk_penalty +
        0.1 * action_quality
    )

    return reward
```

**장점**:
- ✅ 다양한 목표 균형
- ✅ 가중치로 조절 가능
- ✅ 안정적 학습

**단점**:
- ⚠️ 복잡도 증가
- ⚠️ 가중치 튜닝 필요

---

### 리워드 6: 에피소드 종료 시점 보상 (Terminal Reward)

```python
def reward_terminal(self, action, prev_total_value, current_total_value, done):
    """
    매 스텝: 작은 리워드
    에피소드 종료: 큰 리워드
    """
    # 스텝 리워드 (0.1 가중치)
    step_reward = (current_total_value - prev_total_value) / prev_total_value * 0.1

    # 에피소드 종료 리워드 (10배 가중치)
    terminal_reward = 0
    if done:
        final_return = (current_total_value - self.initial_balance) / self.initial_balance
        terminal_reward = final_return * 10

        # 거래 횟수 보너스/페널티
        if self.trade_count < 5:
            terminal_reward -= 0.1  # 너무 적은 거래
        elif self.trade_count > 100:
            terminal_reward -= 0.1  # 너무 많은 거래

    return step_reward + terminal_reward
```

**장점**:
- ✅ 장기 목표 집중
- ✅ 단기 노이즈 무시
- ✅ 에피소드 전체 최적화

**단점**:
- ⚠️ 학습 초기 느림 (희소 리워드)
- ⚠️ 크레딧 할당 문제

---

## 구현 계획

### Phase 1: 리워드 함수 모듈 생성
- [ ] `trading_env/reward_functions.py` 생성
- [ ] 6가지 리워드 함수 구현
- [ ] 리워드 선택 인터페이스 구현

```python
# reward_functions.py
class RewardFunction:
    def __init__(self, reward_type='basic'):
        self.reward_type = reward_type

    def calculate(self, env, action, prev_total_value):
        if self.reward_type == 'basic':
            return self.reward_basic(...)
        elif self.reward_type == 'sell_incentive':
            return self.reward_with_sell_incentive(...)
        # ...
```

### Phase 2: 환경 통합
- [ ] `rl_env.py` 수정
- [ ] 리워드 함수를 config로 선택 가능하게
- [ ] 필요한 상태 변수 추가 (hold_duration, trade_count 등)

```python
# TradingConfig
@dataclass
class TradingConfig:
    # ...
    reward_type: str = "basic"  # 'basic', 'sell_incentive', 'risk_adjusted', ...
```

### Phase 3: 실험 및 비교
- [ ] 각 리워드 함수별 학습 실행
- [ ] 성과 지표 비교:
  - 최종 수익률
  - 샤프 비율
  - 최대 낙폭
  - 매수/매도 빈도
  - 승률
- [ ] 결과 시각화 및 보고서 작성

### Phase 4: 하이퍼파라미터 튜닝
- [ ] 최적 리워드 함수 선정
- [ ] 가중치/임계값 튜닝
- [ ] 교차 검증

---

## 실험 및 평가

### 실험 설정
```python
# 공통 설정
config = TradingConfig(
    initial_balance=1000000,
    lookback_window=60,
    model_type="dqn",
    learning_rate=1e-4,
    batch_size=32,
    epsilon_decay=0.995
)

# 리워드 함수별 학습
reward_types = [
    'basic',
    'sell_incentive',
    'risk_adjusted',
    'benchmark_excess',
    'action_quality',
    'hybrid',
    'terminal'
]

for reward_type in reward_types:
    config.reward_type = reward_type
    trainer = TradingTrainer(config, market="KRW-BTC")
    results = trainer.train(num_episodes=1000)
    # 결과 저장 및 비교
```

### 평가 지표

| 지표 | 설명 | 목표 |
|-----|------|-----|
| **총 수익률** | (최종 자본 - 초기 자본) / 초기 자본 | 최대화 |
| **샤프 비율** | 평균 수익 / 변동성 | 최대화 |
| **최대 낙폭** | 최고점 대비 최대 하락폭 | 최소화 |
| **매수 횟수** | 에피소드당 평균 매수 횟수 | 적정 수준 |
| **매도 횟수** | 에피소드당 평균 매도 횟수 | 적정 수준 (>0) |
| **승률** | 수익 거래 / 전체 거래 | >50% |
| **학습 안정성** | 리워드 분산 | 낮을수록 좋음 |

### 예상 결과

| 리워드 함수 | 수익률 예상 | 매도 빈도 예상 | 학습 난이도 |
|-----------|----------|------------|-----------|
| Basic | 중간 | 낮음 ⚠️ | 쉬움 |
| Sell Incentive | 높음 | 높음 ✅ | 쉬움 |
| Risk-Adjusted | 중간 | 중간 | 어려움 |
| Benchmark Excess | 높음 | 높음 | 중간 |
| Action Quality | 높음 | 중간 | 중간 |
| Hybrid | 높음 | 높음 ✅ | 중간 |
| Terminal | 높음 | 중간 | 어려움 |

---

## 다음 단계

### 즉시 실행
1. [ ] `reward_functions.py` 구현
2. [ ] `rl_env.py` 통합
3. [ ] 간단한 테스트 (10 에피소드)

### 단기 (1주일)
4. [ ] 전체 실험 실행 (1000 에피소드)
5. [ ] 결과 비교 및 분석
6. [ ] 최적 리워드 함수 선정

### 장기 (1개월)
7. [ ] 하이퍼파라미터 최적화
8. [ ] 다양한 시장 환경에서 테스트
9. [ ] 논문 작성 (선택)

---

**작성자 노트**:
이 문서는 리워드 설계의 청사진입니다. 실제 구현 및 실험을 통해 지속적으로 업데이트될 예정입니다.
