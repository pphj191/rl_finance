"""
트레이딩 전략 모듈

다양한 트레이딩 전략을 구현하고 테스트합니다.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class Signal(Enum):
    """트레이딩 시그널"""
    BUY = 1
    SELL = -1
    HOLD = 0


@dataclass
class TradeSignal:
    """트레이드 시그널 정보"""
    timestamp: pd.Timestamp
    signal: Signal
    price: float
    strength: float  # 시그널 강도 (0~1)
    reason: str  # 시그널 발생 이유


class BaseStrategy:
    """전략 베이스 클래스"""

    def __init__(self, name: str):
        self.name = name

    def generate_signals(self, df: pd.DataFrame) -> List[TradeSignal]:
        """
        트레이딩 시그널 생성

        Args:
            df: OHLCV + 지표 데이터프레임

        Returns:
            트레이드 시그널 리스트
        """
        raise NotImplementedError

    def calculate_position_size(
        self,
        balance: float,
        price: float,
        signal_strength: float
    ) -> float:
        """
        포지션 크기 계산

        Args:
            balance: 현재 잔고
            price: 현재 가격
            signal_strength: 시그널 강도

        Returns:
            매수/매도 금액
        """
        # 기본: 시그널 강도에 비례한 포지션
        return balance * signal_strength


class PullbackStrategy(BaseStrategy):
    """눌림목 매매 전략"""

    def __init__(
        self,
        pullback_threshold: float = 60,
        trend_threshold: float = 30,
        volume_confirm: bool = True
    ):
        """
        Args:
            pullback_threshold: 눌림목 지수 임계값
            trend_threshold: 추세 일관성 임계값
            volume_confirm: 거래량 확인 여부
        """
        super().__init__("Pullback Strategy")
        self.pullback_threshold = pullback_threshold
        self.trend_threshold = trend_threshold
        self.volume_confirm = volume_confirm

    def generate_signals(self, df: pd.DataFrame) -> List[TradeSignal]:
        signals = []

        for i in range(1, len(df)):
            row = df.iloc[i]
            prev_row = df.iloc[i-1]

            # 매수 조건
            # 1. 눌림목 지수 높음
            # 2. 상승 추세 확인
            # 3. 거래량 증가 (옵션)
            if (row['pullback_index'] > self.pullback_threshold and
                row['trend_consistency'] > self.trend_threshold):

                volume_ok = True
                if self.volume_confirm and 'volume' in df.columns:
                    vol_ma = df['volume'].rolling(20).mean().iloc[i]
                    volume_ok = row['volume'] > vol_ma

                if volume_ok:
                    strength = min(row['pullback_index'] / 100, 1.0)
                    signals.append(TradeSignal(
                        timestamp=row.name if isinstance(row.name, pd.Timestamp) else pd.Timestamp.now(),
                        signal=Signal.BUY,
                        price=row['close'],
                        strength=strength,
                        reason=f"Pullback: {row['pullback_index']:.1f}, Trend: {row['trend_consistency']:.1f}"
                    ))

            # 매도 조건
            # 1. 눌림목 지수 급락
            # 2. 추세 반전
            if (prev_row['pullback_index'] > self.pullback_threshold and
                row['pullback_index'] < self.pullback_threshold * 0.5):

                signals.append(TradeSignal(
                    timestamp=row.name if isinstance(row.name, pd.Timestamp) else pd.Timestamp.now(),
                    signal=Signal.SELL,
                    price=row['close'],
                    strength=0.8,
                    reason=f"Pullback declined: {row['pullback_index']:.1f}"
                ))

            # 추세 반전 매도
            if row['trend_consistency'] < -self.trend_threshold:
                signals.append(TradeSignal(
                    timestamp=row.name if isinstance(row.name, pd.Timestamp) else pd.Timestamp.now(),
                    signal=Signal.SELL,
                    price=row['close'],
                    strength=1.0,
                    reason=f"Trend reversal: {row['trend_consistency']:.1f}"
                ))

        return signals


class BreakoutStrategy(BaseStrategy):
    """변동성 돌파 전략"""

    def __init__(
        self,
        breakout_threshold: float = 70,
        support_threshold: float = 50
    ):
        super().__init__("Breakout Strategy")
        self.breakout_threshold = breakout_threshold
        self.support_threshold = support_threshold

    def generate_signals(self, df: pd.DataFrame) -> List[TradeSignal]:
        signals = []

        for i in range(1, len(df)):
            row = df.iloc[i]
            prev_row = df.iloc[i-1]

            # 매수: 돌파 확률 높고 지지선 강함
            if (row['breakout_probability'] > self.breakout_threshold and
                row['support_strength'] > self.support_threshold):

                # 실제 가격 상승 확인
                if row['close'] > prev_row['close']:
                    strength = row['breakout_probability'] / 100
                    signals.append(TradeSignal(
                        timestamp=row.name if isinstance(row.name, pd.Timestamp) else pd.Timestamp.now(),
                        signal=Signal.BUY,
                        price=row['close'],
                        strength=strength,
                        reason=f"Breakout: {row['breakout_probability']:.1f}, Support: {row['support_strength']:.1f}"
                    ))

            # 매도: 저항선 강하고 가격 하락
            if (row['resistance_strength'] > self.support_threshold and
                row['close'] < prev_row['close']):

                signals.append(TradeSignal(
                    timestamp=row.name if isinstance(row.name, pd.Timestamp) else pd.Timestamp.now(),
                    signal=Signal.SELL,
                    price=row['close'],
                    strength=0.8,
                    reason=f"Resistance hit: {row['resistance_strength']:.1f}"
                ))

        return signals


class HybridStrategy(BaseStrategy):
    """복합 전략 (눌림목 + 돌파)"""

    def __init__(self):
        super().__init__("Hybrid Strategy")
        self.pullback_strategy = PullbackStrategy()
        self.breakout_strategy = BreakoutStrategy()

    def generate_signals(self, df: pd.DataFrame) -> List[TradeSignal]:
        # 두 전략의 시그널 통합
        pullback_signals = self.pullback_strategy.generate_signals(df)
        breakout_signals = self.breakout_strategy.generate_signals(df)

        # 시그널 병합 및 중복 제거
        all_signals = pullback_signals + breakout_signals

        # 같은 타임스탬프의 시그널은 강도가 높은 것 선택
        signal_dict = {}
        for signal in all_signals:
            key = signal.timestamp
            if key not in signal_dict or signal.strength > signal_dict[key].strength:
                signal_dict[key] = signal

        return list(signal_dict.values())


def backtest_strategy(
    df: pd.DataFrame,
    strategy: BaseStrategy,
    initial_balance: float = 1000000,
    fee_rate: float = 0.0005
) -> Dict:
    """
    전략 백테스팅 (간단한 버전)

    Args:
        df: OHLCV + 지표 데이터프레임
        strategy: 테스트할 전략
        initial_balance: 초기 자본
        fee_rate: 거래 수수료율

    Returns:
        백테스팅 결과 딕셔너리
    """
    signals = strategy.generate_signals(df)

    balance = initial_balance
    position = 0
    trades = []

    for signal in signals:
        if signal.signal == Signal.BUY and balance > 0:
            # 매수
            amount = balance * signal.strength
            fee = amount * fee_rate
            coins = (amount - fee) / signal.price

            position += coins
            balance -= amount

            trades.append({
                'timestamp': signal.timestamp,
                'type': 'BUY',
                'price': signal.price,
                'amount': amount,
                'coins': coins,
                'reason': signal.reason
            })

        elif signal.signal == Signal.SELL and position > 0:
            # 매도
            amount = position * signal.price
            fee = amount * fee_rate

            balance += (amount - fee)

            trades.append({
                'timestamp': signal.timestamp,
                'type': 'SELL',
                'price': signal.price,
                'amount': amount,
                'coins': position,
                'reason': signal.reason
            })

            position = 0

    # 최종 청산
    final_price = df['close'].iloc[-1]
    final_value = balance + position * final_price

    return {
        'initial_balance': initial_balance,
        'final_value': final_value,
        'profit': final_value - initial_balance,
        'profit_rate': (final_value - initial_balance) / initial_balance * 100,
        'num_trades': len(trades),
        'trades': trades
    }


# 테스트 코드
if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from trading_env.indicators_custom import add_custom_indicators

    # 샘플 데이터 생성
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=200, freq='1H')

    price = 50000
    prices = []
    volumes = []

    for i in range(200):
        # 상승 추세 + 주기적 눌림목
        if i % 40 in range(5, 10):  # 눌림목
            price *= np.random.uniform(0.97, 0.99)
            vol = np.random.randint(2000, 4000)
        else:
            price *= np.random.uniform(1.0, 1.005)
            vol = np.random.randint(1000, 2000)

        prices.append(price)
        volumes.append(vol)

    df = pd.DataFrame({
        'open': prices,
        'high': [p * 1.01 for p in prices],
        'low': [p * 0.99 for p in prices],
        'close': prices,
        'volume': volumes
    }, index=dates)

    # 커스텀 지표 추가
    df = add_custom_indicators(df)

    print("=== 트레이딩 전략 테스트 ===\n")

    # 눌림목 전략 테스트
    pullback_strategy = PullbackStrategy()
    result = backtest_strategy(df, pullback_strategy)

    print(f"전략: {pullback_strategy.name}")
    print(f"초기 자본: {result['initial_balance']:,.0f} KRW")
    print(f"최종 자본: {result['final_value']:,.0f} KRW")
    print(f"수익: {result['profit']:,.0f} KRW ({result['profit_rate']:.2f}%)")
    print(f"거래 횟수: {result['num_trades']}")
    print(f"\n최근 5개 거래:")
    for trade in result['trades'][-5:]:
        print(f"  {trade['timestamp']}: {trade['type']} @ {trade['price']:,.0f} - {trade['reason']}")
