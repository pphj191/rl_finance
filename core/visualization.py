"""
트레이딩 결과 시각화 모듈

백테스트 및 실시간 트레이딩 결과를 차트로 시각화합니다.

최종 업데이트: 2025-10-05 23:55:00
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime

from core.backtesting_engine import BacktestResult


class TradingVisualizer:
    """트레이딩 결과 시각화"""
    
    def __init__(self, style: str = 'seaborn-v0_8-darkgrid'):
        """
        Args:
            style: matplotlib 스타일
        """
        self.style = style
        sns.set_palette("husl")
    
    def plot_backtest_results(
        self,
        result: BacktestResult,
        benchmark: Optional[BacktestResult] = None,
        save_path: Optional[str] = None
    ) -> None:
        """백테스트 결과 전체 시각화
        
        Args:
            result: 백테스트 결과
            benchmark: 벤치마크 결과 (선택사항)
            save_path: 저장 경로 (선택사항)
        """
        fig, axes = plt.subplots(3, 2, figsize=(16, 12))
        fig.suptitle('Backtest Results', fontsize=16, fontweight='bold')
        
        # 1. 포트폴리오 가치 변화
        self._plot_portfolio_value(axes[0, 0], result, benchmark)
        
        # 2. 일일 수익률 분포
        self._plot_daily_returns_distribution(axes[0, 1], result)
        
        # 3. 낙폭 차트
        self._plot_drawdown(axes[1, 0], result)
        
        # 4. 거래 포인트
        self._plot_trades(axes[1, 1], result)
        
        # 5. 월별 수익률
        self._plot_monthly_returns(axes[2, 0], result)
        
        # 6. 성과 지표 요약
        self._plot_metrics_summary(axes[2, 1], result, benchmark)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"차트가 저장되었습니다: {save_path}")
        
        plt.show()
    
    def _plot_portfolio_value(
        self,
        ax: plt.Axes,
        result: BacktestResult,
        benchmark: Optional[BacktestResult] = None
    ) -> None:
        """포트폴리오 가치 변화 차트"""
        ax.plot(result.portfolio_values, label='Agent', linewidth=2)
        
        if benchmark:
            ax.plot(benchmark.portfolio_values, label='Benchmark', linewidth=2, linestyle='--', alpha=0.7)
        
        ax.set_title('Portfolio Value Over Time', fontweight='bold')
        ax.set_xlabel('Days')
        ax.set_ylabel('Portfolio Value')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_daily_returns_distribution(self, ax: plt.Axes, result: BacktestResult) -> None:
        """일일 수익률 분포 차트"""
        returns = np.array(result.daily_returns) * 100
        
        ax.hist(returns, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        ax.axvline(np.mean(returns), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(returns):.2f}%')
        ax.axvline(np.median(returns), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(returns):.2f}%')
        
        ax.set_title('Daily Returns Distribution', fontweight='bold')
        ax.set_xlabel('Daily Return (%)')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_drawdown(self, ax: plt.Axes, result: BacktestResult) -> None:
        """낙폭 차트"""
        portfolio = np.array(result.portfolio_values)
        cummax = np.maximum.accumulate(portfolio)
        drawdown = (portfolio - cummax) / cummax * 100
        
        ax.fill_between(range(len(drawdown)), drawdown, 0, alpha=0.3, color='red')
        ax.plot(drawdown, color='red', linewidth=1)
        ax.axhline(result.max_drawdown * 100, color='darkred', linestyle='--', linewidth=2, label=f'Max DD: {result.max_drawdown*100:.2f}%')
        
        ax.set_title('Drawdown Over Time', fontweight='bold')
        ax.set_xlabel('Days')
        ax.set_ylabel('Drawdown (%)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_trades(self, ax: plt.Axes, result: BacktestResult) -> None:
        """거래 포인트 차트"""
        if not result.trades:
            ax.text(0.5, 0.5, 'No trades executed', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Trading Points', fontweight='bold')
            return
        
        trades_df = pd.DataFrame(result.trades)
        
        # 포트폴리오 가치
        ax.plot(result.portfolio_values, label='Portfolio Value', color='blue', alpha=0.3)
        
        # 매수 포인트
        buy_trades = trades_df[trades_df['action'] == 'BUY']
        if not buy_trades.empty:
            ax.scatter(buy_trades.index, [result.portfolio_values[i] for i in buy_trades.index], 
                      color='green', marker='^', s=100, label='Buy', zorder=5)
        
        # 매도 포인트
        sell_trades = trades_df[trades_df['action'] == 'SELL']
        if not sell_trades.empty:
            ax.scatter(sell_trades.index, [result.portfolio_values[i] for i in sell_trades.index],
                      color='red', marker='v', s=100, label='Sell', zorder=5)
        
        ax.set_title('Trading Points', fontweight='bold')
        ax.set_xlabel('Days')
        ax.set_ylabel('Portfolio Value')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_monthly_returns(self, ax: plt.Axes, result: BacktestResult) -> None:
        """월별 수익률 차트"""
        if len(result.daily_returns) < 20:
            ax.text(0.5, 0.5, 'Insufficient data for monthly returns', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Monthly Returns', fontweight='bold')
            return
        
        # 월별 수익률 계산 (단순화: 20일 = 1개월)
        returns = np.array(result.daily_returns)
        n_months = len(returns) // 20
        monthly_returns = []
        
        for i in range(n_months):
            month_returns = returns[i*20:(i+1)*20]
            monthly_return = (1 + month_returns).prod() - 1
            monthly_returns.append(monthly_return * 100)
        
        colors = ['green' if r > 0 else 'red' for r in monthly_returns]
        ax.bar(range(len(monthly_returns)), monthly_returns, color=colors, alpha=0.7)
        ax.axhline(0, color='black', linewidth=0.8)
        
        ax.set_title('Monthly Returns', fontweight='bold')
        ax.set_xlabel('Month')
        ax.set_ylabel('Return (%)')
        ax.grid(True, alpha=0.3, axis='y')
    
    def _plot_metrics_summary(
        self,
        ax: plt.Axes,
        result: BacktestResult,
        benchmark: Optional[BacktestResult] = None
    ) -> None:
        """성과 지표 요약 텍스트"""
        ax.axis('off')
        
        metrics_text = f"""
        Performance Metrics
        ═══════════════════════════════════
        Total Return:        {result.total_return*100:>8.2f}%
        Annual Return:       {result.annual_return*100:>8.2f}%
        Sharpe Ratio:        {result.sharpe_ratio:>8.2f}
        Max Drawdown:        {result.max_drawdown*100:>8.2f}%
        Volatility:          {result.volatility*100:>8.2f}%
        
        Trading Statistics
        ═══════════════════════════════════
        Total Trades:        {result.total_trades:>8}
        Win Rate:            {result.win_rate*100:>8.2f}%
        Profit Factor:       {result.profit_factor:>8.2f}
        """
        
        if benchmark:
            metrics_text += f"""
        
        vs. Benchmark
        ═══════════════════════════════════
        Excess Return:       {(result.total_return - benchmark.total_return)*100:>8.2f}%
        Sharpe Diff:         {result.sharpe_ratio - benchmark.sharpe_ratio:>8.2f}
        """
        
        ax.text(0.1, 0.9, metrics_text, transform=ax.transAxes, 
               fontfamily='monospace', fontsize=10, verticalalignment='top')
    
    def plot_realtime_performance(
        self,
        portfolio_values: List[float],
        trades: List[Dict],
        save_path: Optional[str] = None
    ) -> None:
        """실시간 트레이딩 성과 시각화
        
        Args:
            portfolio_values: 포트폴리오 가치 리스트
            trades: 거래 리스트
            save_path: 저장 경로 (선택사항)
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Real-Time Trading Performance', fontsize=16, fontweight='bold')
        
        # 1. 포트폴리오 가치
        axes[0, 0].plot(portfolio_values, linewidth=2, color='blue')
        axes[0, 0].set_title('Portfolio Value', fontweight='bold')
        axes[0, 0].set_xlabel('Time Steps')
        axes[0, 0].set_ylabel('Value')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 수익률
        returns = [(portfolio_values[i] - portfolio_values[0]) / portfolio_values[0] * 100 
                  for i in range(len(portfolio_values))]
        axes[0, 1].plot(returns, linewidth=2, color='green')
        axes[0, 1].axhline(0, color='red', linestyle='--', linewidth=1)
        axes[0, 1].set_title('Cumulative Return', fontweight='bold')
        axes[0, 1].set_xlabel('Time Steps')
        axes[0, 1].set_ylabel('Return (%)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 거래 현황
        if trades:
            trades_df = pd.DataFrame(trades)
            action_counts = trades_df['action'].value_counts()
            axes[1, 0].bar(action_counts.index, action_counts.values, alpha=0.7)
            axes[1, 0].set_title('Trade Actions', fontweight='bold')
            axes[1, 0].set_xlabel('Action')
            axes[1, 0].set_ylabel('Count')
            axes[1, 0].grid(True, alpha=0.3, axis='y')
        else:
            axes[1, 0].text(0.5, 0.5, 'No trades yet', ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('Trade Actions', fontweight='bold')
        
        # 4. 거래 포인트
        axes[1, 1].plot(portfolio_values, linewidth=2, color='blue', alpha=0.3)
        
        if trades:
            trades_df = pd.DataFrame(trades)
            for idx, trade in trades_df.iterrows():
                color = 'green' if trade['action'] == 'BUY' else 'red'
                marker = '^' if trade['action'] == 'BUY' else 'v'
                axes[1, 1].scatter(idx, portfolio_values[min(idx, len(portfolio_values)-1)],
                                 color=color, marker=marker, s=100, zorder=5)
        
        axes[1, 1].set_title('Trading Points', fontweight='bold')
        axes[1, 1].set_xlabel('Time Steps')
        axes[1, 1].set_ylabel('Portfolio Value')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"차트가 저장되었습니다: {save_path}")
        
        plt.show()
    
    def plot_comparison(
        self,
        results: Dict[str, BacktestResult],
        save_path: Optional[str] = None
    ) -> None:
        """여러 결과 비교 시각화
        
        Args:
            results: {이름: 결과} 딕셔너리
            save_path: 저장 경로 (선택사항)
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Strategy Comparison', fontsize=16, fontweight='bold')
        
        # 1. 포트폴리오 가치 비교
        for name, result in results.items():
            axes[0, 0].plot(result.portfolio_values, label=name, linewidth=2)
        axes[0, 0].set_title('Portfolio Value Comparison', fontweight='bold')
        axes[0, 0].set_xlabel('Days')
        axes[0, 0].set_ylabel('Portfolio Value')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 성과 지표 비교
        metrics = ['total_return', 'sharpe_ratio', 'max_drawdown', 'win_rate']
        metric_labels = ['Total Return', 'Sharpe Ratio', 'Max Drawdown', 'Win Rate']
        
        for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
            values = [getattr(result, metric) * (100 if metric in ['total_return', 'max_drawdown', 'win_rate'] else 1) 
                     for result in results.values()]
            
            ax_idx = (i // 2, i % 2) if i > 0 else (0, 1)
            if i == 0:
                ax_idx = (0, 1)
            elif i == 1:
                ax_idx = (1, 0)
            elif i == 2:
                ax_idx = (1, 1)
            else:
                continue
            
            axes[ax_idx].bar(results.keys(), values, alpha=0.7)
            axes[ax_idx].set_title(label, fontweight='bold')
            axes[ax_idx].set_ylabel('Value' + (' (%)' if metric in ['total_return', 'max_drawdown', 'win_rate'] else ''))
            axes[ax_idx].grid(True, alpha=0.3, axis='y')
            axes[ax_idx].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"차트가 저장되었습니다: {save_path}")
        
        plt.show()
