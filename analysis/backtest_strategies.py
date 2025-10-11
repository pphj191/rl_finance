"""
전략 백테스팅 및 성과 분석 모듈
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
from datetime import datetime
import os


class BacktestEngine:
    """백테스팅 엔진"""

    def __init__(
        self,
        initial_balance: float = 1000000,
        fee_rate: float = 0.0005,
        slippage: float = 0.001
    ):
        """
        Args:
            initial_balance: 초기 자본
            fee_rate: 거래 수수료율 (0.05% 기본)
            slippage: 슬리피지 (0.1% 기본)
        """
        self.initial_balance = initial_balance
        self.fee_rate = fee_rate
        self.slippage = slippage

        self.reset()

    def reset(self):
        """백테스트 상태 초기화"""
        self.balance = self.initial_balance
        self.position = 0.0
        self.trades = []
        self.portfolio_values = []
        self.timestamps = []

    def execute_trade(
        self,
        signal: str,
        price: float,
        timestamp: pd.Timestamp,
        amount_ratio: float = 1.0,
        reason: str = ""
    ):
        """
        거래 실행

        Args:
            signal: 'BUY' or 'SELL'
            price: 거래 가격
            timestamp: 거래 시간
            amount_ratio: 거래 비율 (0~1)
            reason: 거래 사유
        """
        if signal == 'BUY' and self.balance > 0:
            # 슬리피지 적용 (매수 시 불리하게)
            actual_price = price * (1 + self.slippage)

            # 거래 금액
            trade_amount = self.balance * amount_ratio
            fee = trade_amount * self.fee_rate
            coins = (trade_amount - fee) / actual_price

            self.position += coins
            self.balance -= trade_amount

            self.trades.append({
                'timestamp': timestamp,
                'type': 'BUY',
                'price': actual_price,
                'amount': trade_amount,
                'coins': coins,
                'fee': fee,
                'balance_after': self.balance,
                'position_after': self.position,
                'reason': reason
            })

        elif signal == 'SELL' and self.position > 0:
            # 슬리피지 적용 (매도 시 불리하게)
            actual_price = price * (1 - self.slippage)

            # 거래 금액
            coins_to_sell = self.position * amount_ratio
            trade_amount = coins_to_sell * actual_price
            fee = trade_amount * self.fee_rate

            self.balance += (trade_amount - fee)
            self.position -= coins_to_sell

            self.trades.append({
                'timestamp': timestamp,
                'type': 'SELL',
                'price': actual_price,
                'amount': trade_amount,
                'coins': coins_to_sell,
                'fee': fee,
                'balance_after': self.balance,
                'position_after': self.position,
                'reason': reason
            })

    def record_portfolio_value(self, timestamp: pd.Timestamp, current_price: float):
        """포트폴리오 가치 기록"""
        total_value = self.balance + self.position * current_price
        self.portfolio_values.append(total_value)
        self.timestamps.append(timestamp)

    def get_metrics(self) -> Dict:
        """성과 지표 계산"""
        if not self.portfolio_values:
            return {}

        values = np.array(self.portfolio_values)
        returns = np.diff(values) / values[:-1]

        final_value = values[-1]
        total_return = (final_value - self.initial_balance) / self.initial_balance * 100

        # 승률
        winning_trades = [t for t in self.trades if
                         (t['type'] == 'SELL' and
                          any(prev['type'] == 'BUY' and prev['price'] < t['price']
                              for prev in self.trades[:self.trades.index(t)]))]
        win_rate = len(winning_trades) / len(self.trades) * 100 if self.trades else 0

        # MDD (Maximum Drawdown)
        cummax = np.maximum.accumulate(values)
        drawdowns = (values - cummax) / cummax * 100
        mdd = abs(drawdowns.min()) if len(drawdowns) > 0 else 0

        # Sharpe Ratio (연율화, 무위험수익률 0 가정)
        if len(returns) > 1:
            sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(365 * 24)  # 시간봉 기준
        else:
            sharpe = 0

        return {
            'initial_balance': self.initial_balance,
            'final_value': final_value,
            'total_profit': final_value - self.initial_balance,
            'total_return': total_return,
            'num_trades': len(self.trades),
            'win_rate': win_rate,
            'max_drawdown': mdd,
            'sharpe_ratio': sharpe,
            'total_fees': sum(t['fee'] for t in self.trades)
        }

    def plot_results(self, save_path: Optional[str] = None):
        """백테스팅 결과 시각화"""
        if not self.portfolio_values:
            print("백테스팅 데이터가 없습니다.")
            return

        fig, axes = plt.subplots(3, 1, figsize=(15, 12))

        # 1. 포트폴리오 가치
        axes[0].plot(self.timestamps, self.portfolio_values, 'b-', linewidth=2)
        axes[0].axhline(y=self.initial_balance, color='r', linestyle='--', alpha=0.5, label='Initial')
        axes[0].set_ylabel('Portfolio Value (KRW)')
        axes[0].set_title('Portfolio Value Over Time')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # 매수/매도 포인트 표시
        for trade in self.trades:
            color = 'green' if trade['type'] == 'BUY' else 'red'
            marker = '^' if trade['type'] == 'BUY' else 'v'
            idx = self.timestamps.index(trade['timestamp']) if trade['timestamp'] in self.timestamps else None
            if idx:
                axes[0].scatter(trade['timestamp'], self.portfolio_values[idx],
                              c=color, marker=marker, s=100, zorder=5)

        # 2. 수익률
        returns = np.diff(self.portfolio_values) / np.array(self.portfolio_values[:-1]) * 100
        axes[1].plot(self.timestamps[1:], returns, 'g-', alpha=0.7)
        axes[1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
        axes[1].set_ylabel('Return (%)')
        axes[1].set_title('Period Returns')
        axes[1].grid(True, alpha=0.3)

        # 3. 누적 수익률
        cumulative_returns = (np.array(self.portfolio_values) / self.initial_balance - 1) * 100
        axes[2].plot(self.timestamps, cumulative_returns, 'purple', linewidth=2)
        axes[2].axhline(y=0, color='r', linestyle='--', alpha=0.5)
        axes[2].fill_between(self.timestamps, 0, cumulative_returns, alpha=0.3)
        axes[2].set_ylabel('Cumulative Return (%)')
        axes[2].set_xlabel('Time')
        axes[2].set_title('Cumulative Returns')
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"그래프 저장: {save_path}")

        plt.show()

    def print_summary(self):
        """백테스팅 결과 요약 출력"""
        metrics = self.get_metrics()

        print("\n" + "="*50)
        print("백테스팅 결과 요약")
        print("="*50)
        print(f"초기 자본:      {metrics['initial_balance']:>15,.0f} KRW")
        print(f"최종 자본:      {metrics['final_value']:>15,.0f} KRW")
        print(f"총 수익:        {metrics['total_profit']:>15,.0f} KRW")
        print(f"수익률:         {metrics['total_return']:>15.2f} %")
        print(f"거래 횟수:      {metrics['num_trades']:>15} 회")
        print(f"승률:           {metrics['win_rate']:>15.2f} %")
        print(f"최대 낙폭:      {metrics['max_drawdown']:>15.2f} %")
        print(f"샤프 비율:      {metrics['sharpe_ratio']:>15.2f}")
        print(f"총 수수료:      {metrics['total_fees']:>15,.0f} KRW")
        print("="*50 + "\n")


def compare_strategies(
    df: pd.DataFrame,
    strategies: List,
    save_dir: Optional[str] = None
) -> pd.DataFrame:
    """
    여러 전략 비교

    Args:
        df: OHLCV + 지표 데이터
        strategies: 전략 리스트
        save_dir: 결과 저장 디렉토리

    Returns:
        전략별 성과 비교 데이터프레임
    """
    results = []

    for strategy in strategies:
        engine = BacktestEngine()

        # 전략 시그널 생성
        from .strategies import backtest_strategy
        result = backtest_strategy(df, strategy)

        results.append({
            'strategy': strategy.name,
            'final_value': result['final_value'],
            'profit': result['profit'],
            'profit_rate': result['profit_rate'],
            'num_trades': result['num_trades']
        })

    comparison_df = pd.DataFrame(results)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        csv_path = os.path.join(save_dir, 'strategy_comparison.csv')
        comparison_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"비교 결과 저장: {csv_path}")

    return comparison_df


# 테스트 코드
if __name__ == "__main__":
    print("=== 백테스팅 엔진 테스트 ===\n")

    # 샘플 데이터
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=100, freq='1H')
    prices = [50000 * (1.01 ** i) * (1 + np.random.randn() * 0.02) for i in range(100)]

    engine = BacktestEngine(initial_balance=1000000)

    # 임의 거래 시뮬레이션
    for i, (ts, price) in enumerate(zip(dates, prices)):
        if i % 20 == 10:  # 매수
            engine.execute_trade('BUY', price, ts, amount_ratio=0.8, reason="Test buy")
        elif i % 20 == 15:  # 매도
            engine.execute_trade('SELL', price, ts, amount_ratio=1.0, reason="Test sell")

        engine.record_portfolio_value(ts, price)

    # 결과 출력
    engine.print_summary()
    engine.plot_results()
