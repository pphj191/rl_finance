"""
백테스팅 및 실시간 트레이딩 시스템

TODO:
1. 백테스팅
   - ✅ 백테스팅 환경
   - ✅ 성과 지표 계산
   - ✅ 벤치마크 비교
   - ⬜ 리스크 지표

2. 실시간 트레이딩
   - ⬜ 실시간 데이터 수집
   - ⬜ 모델 추론
   - ⬜ 주문 실행
   - ⬜ 포지션 관리

3. 성과 분석
   - ✅ 수익률 분석
   - ✅ 거래 분석
   - ✅ 시각화
   - ⬜ 리포트 생성
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import json
import time
import threading
import os
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

from rl_trading_env import TradingEnvironment, TradingConfig, ActionSpace
from dqn_agent import DQNAgent
from upbit_api import UpbitAPI


@dataclass
class BacktestResult:
    """백테스트 결과"""
    total_return: float
    annual_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_trades: int
    equity_curve: List[float]
    trades: List[Dict]
    daily_returns: List[float]


class Backtester:
    """백테스팅 시스템"""
    
    def __init__(self, config: TradingConfig):
        self.config = config
        self.results = {}
    
    def run_backtest(self, agent: DQNAgent, env: TradingEnvironment,
                    start_date: Optional[str] = None, 
                    end_date: Optional[str] = None) -> BacktestResult:
        """백테스트 실행"""
        
        print("백테스트 시작...")
        
        # 환경 초기화
        state, _ = env.reset()
        
        # 트레이딩 기록
        equity_curve = [env.total_value]
        trades = []
        daily_returns = []
        
        total_steps = 0
        previous_action = ActionSpace.HOLD
        trade_start_value = env.total_value
        
        while True:
            # 액션 선택 (평가 모드)
            action_mask = env.get_action_mask()
            action = agent.select_action(state, action_mask, training=False)
            
            # 환경 스텝
            next_state, reward, terminated, truncated, info = env.step(action)
            
            # 거래 기록
            if action != ActionSpace.HOLD:
                trade_info = {
                    'step': total_steps,
                    'action': ActionSpace.get_action_names()[action],
                    'price': info['current_price'],
                    'balance': info['balance'],
                    'position': info['position'],
                    'total_value': info['total_value'],
                    'timestamp': datetime.now().isoformat()
                }
                trades.append(trade_info)
            
            # 포트폴리오 가치 기록
            equity_curve.append(info['total_value'])
            
            # 일일 수익률 계산 (매 스텝을 일일로 가정)
            if len(equity_curve) > 1:
                daily_return = (equity_curve[-1] - equity_curve[-2]) / equity_curve[-2]
                daily_returns.append(daily_return)
            
            state = next_state
            previous_action = action
            total_steps += 1
            
            if terminated or truncated:
                break
        
        # 성과 지표 계산
        initial_value = equity_curve[0]
        final_value = equity_curve[-1]
        
        total_return = (final_value - initial_value) / initial_value
        
        # 연환산 수익률 (252 거래일 기준)
        trading_days = len(equity_curve)
        if trading_days > 0:
            annual_return = (final_value / initial_value) ** (252 / trading_days) - 1
        else:
            annual_return = 0
        
        # 샤프 비율
        if len(daily_returns) > 1 and np.std(daily_returns) > 0:
            sharpe_ratio = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)
        else:
            sharpe_ratio = 0
        
        # 최대 낙폭
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (np.array(equity_curve) - peak) / peak
        max_drawdown = np.min(drawdown)
        
        # 승률 계산
        profitable_trades = 0
        total_trades = len(trades)
        
        if total_trades > 0:
            trade_returns = []
            for i in range(1, len(trades)):
                if trades[i]['action'] == 'SELL' and trades[i-1]['action'] == 'BUY':
                    trade_return = (trades[i]['total_value'] - trades[i-1]['total_value']) / trades[i-1]['total_value']
                    trade_returns.append(trade_return)
                    if trade_return > 0:
                        profitable_trades += 1
            
            win_rate = profitable_trades / len(trade_returns) if trade_returns else 0
            
            # 이익 팩터
            profits = sum([r for r in trade_returns if r > 0])
            losses = abs(sum([r for r in trade_returns if r < 0]))
            profit_factor = profits / losses if losses > 0 else float('inf')
        else:
            win_rate = 0
            profit_factor = 0
        
        result = BacktestResult(
            total_return=total_return,
            annual_return=annual_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_trades=total_trades,
            equity_curve=equity_curve,
            trades=trades,
            daily_returns=daily_returns
        )
        
        print(f"백테스트 완료: 총 수익률 {total_return:.2%}, 샤프 비율 {sharpe_ratio:.2f}")
        
        return result
    
    def compare_with_benchmark(self, result: BacktestResult, 
                              benchmark_data: List[float]) -> Dict[str, float]:
        """벤치마크와 비교"""
        
        if len(benchmark_data) != len(result.equity_curve):
            # 길이 맞추기
            min_len = min(len(benchmark_data), len(result.equity_curve))
            benchmark_data = benchmark_data[:min_len]
            equity_curve = result.equity_curve[:min_len]
        else:
            equity_curve = result.equity_curve
        
        # 벤치마크 수익률
        benchmark_return = (benchmark_data[-1] - benchmark_data[0]) / benchmark_data[0]
        
        # 초과 수익률
        excess_return = result.total_return - benchmark_return
        
        # 벤치마크 대비 성능
        outperformance = (result.total_return + 1) / (benchmark_return + 1) - 1
        
        return {
            'benchmark_return': benchmark_return,
            'excess_return': excess_return,
            'outperformance': outperformance,
            'agent_return': result.total_return
        }
    
    def plot_results(self, result: BacktestResult, 
                    benchmark_data: Optional[List[float]] = None,
                    save_path: Optional[str] = None):
        """결과 시각화"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 포트폴리오 가치 곡선
        axes[0, 0].plot(result.equity_curve, label='Portfolio Value', linewidth=2)
        if benchmark_data:
            # 벤치마크를 같은 스케일로 조정
            benchmark_normalized = np.array(benchmark_data) / benchmark_data[0] * result.equity_curve[0]
            axes[0, 0].plot(benchmark_normalized, label='Benchmark (Buy & Hold)', 
                           linewidth=2, alpha=0.7)
        axes[0, 0].set_title('Portfolio Value Over Time')
        axes[0, 0].set_xlabel('Time Steps')
        axes[0, 0].set_ylabel('Portfolio Value (KRW)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 일일 수익률 분포
        if result.daily_returns:
            axes[0, 1].hist(result.daily_returns, bins=50, alpha=0.7, edgecolor='black')
            axes[0, 1].axvline(x=np.mean(result.daily_returns), color='red', 
                              linestyle='--', label=f'Mean: {np.mean(result.daily_returns):.4f}')
            axes[0, 1].set_title('Daily Returns Distribution')
            axes[0, 1].set_xlabel('Daily Return')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # 낙폭 곡선
        peak = np.maximum.accumulate(result.equity_curve)
        drawdown = (np.array(result.equity_curve) - peak) / peak * 100
        axes[1, 0].fill_between(range(len(drawdown)), drawdown, 0, 
                               alpha=0.3, color='red', label='Drawdown')
        axes[1, 0].plot(drawdown, color='red', linewidth=1)
        axes[1, 0].set_title('Drawdown (%)')
        axes[1, 0].set_xlabel('Time Steps')
        axes[1, 0].set_ylabel('Drawdown (%)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 성과 지표 요약
        metrics_text = f"""
        Total Return: {result.total_return:.2%}
        Annual Return: {result.annual_return:.2%}
        Sharpe Ratio: {result.sharpe_ratio:.2f}
        Max Drawdown: {result.max_drawdown:.2%}
        Win Rate: {result.win_rate:.2%}
        Total Trades: {result.total_trades}
        """
        
        axes[1, 1].text(0.1, 0.5, metrics_text, transform=axes[1, 1].transAxes,
                        fontsize=12, verticalalignment='center',
                        bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        axes[1, 1].set_title('Performance Metrics')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"백테스트 결과 저장: {save_path}")
        
        plt.show()


class RealTimeTrader:
    """실시간 트레이딩 시스템"""
    
    def __init__(self, config: TradingConfig, model_path: str, market: str = "KRW-BTC"):
        self.config = config
        self.market = market
        self.upbit = UpbitAPI()
        
        # 모델 로드
        self.agent = DQNAgent(config, state_size=100)  # 임시 크기
        self.agent.load_model(model_path)
        
        # 거래 상태
        self.is_trading = False
        self.current_position = 0.0
        self.current_balance = 0.0
        
        # 데이터 수집기
        self.data_history = []
        
    def start_trading(self, update_interval: int = 60):
        """실시간 트레이딩 시작"""
        print(f"실시간 트레이딩 시작: {self.market}")
        
        self.is_trading = True
        
        # 백그라운드에서 트레이딩 실행
        trading_thread = threading.Thread(target=self._trading_loop, 
                                         args=(update_interval,))
        trading_thread.daemon = True
        trading_thread.start()
        
        return trading_thread
    
    def stop_trading(self):
        """실시간 트레이딩 중지"""
        print("실시간 트레이딩 중지")
        self.is_trading = False
    
    def _trading_loop(self, update_interval: int):
        """트레이딩 루프"""
        
        while self.is_trading:
            try:
                # 현재 상태 수집
                current_state = self._get_current_state()
                
                if current_state is not None:
                    # 액션 예측
                    action_mask = self._get_action_mask()
                    action = self.agent.select_action(current_state, action_mask, training=False)
                    
                    # 액션 실행
                    self._execute_action(action)
                    
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] "
                          f"Action: {ActionSpace.get_action_names()[action]}, "
                          f"Balance: {self.current_balance:.0f}, "
                          f"Position: {self.current_position:.6f}")
                
                # 대기
                time.sleep(update_interval)
                
            except Exception as e:
                print(f"트레이딩 루프 오류: {e}")
                time.sleep(5)  # 오류 시 5초 대기
    
    def _get_current_state(self) -> Optional[np.ndarray]:
        """현재 상태 수집"""
        try:
            # TODO: 실제 환경에서 상태 수집 구현
            # 현재는 더미 데이터 반환
            return np.random.randn(100).astype(np.float32)
            
        except Exception as e:
            print(f"상태 수집 오류: {e}")
            return None
    
    def _get_action_mask(self) -> np.ndarray:
        """액션 마스크 생성"""
        mask = np.ones(ActionSpace.get_num_actions(), dtype=bool)
        
        # 현재 잔고와 포지션에 따라 마스킹
        current_price = self._get_current_price()
        
        if current_price is None:
            return mask
        
        # 매도 불가능한 경우
        if self.current_position <= 0:
            mask[ActionSpace.SELL] = False
        
        # 매수 불가능한 경우
        min_order_amount = current_price * (1 + self.config.transaction_fee)
        if self.current_balance < min_order_amount:
            mask[ActionSpace.BUY] = False
        
        return mask
    
    def _get_current_price(self) -> Optional[float]:
        """현재 가격 조회"""
        try:
            ticker = self.upbit.get_ticker(self.market)[0]
            return float(ticker['trade_price'])
        except Exception as e:
            print(f"가격 조회 오류: {e}")
            return None
    
    def _execute_action(self, action: int):
        """액션 실행"""
        current_price = self._get_current_price()
        
        if current_price is None:
            return
        
        try:
            if action == ActionSpace.BUY and self.current_balance > 0:
                # 매수 실행 (시뮬레이션)
                print(f"매수 신호: {current_price:,}원")
                # 실제 환경에서는 self.upbit.buy_market_order() 사용
                
            elif action == ActionSpace.SELL and self.current_position > 0:
                # 매도 실행 (시뮬레이션)
                print(f"매도 신호: {current_price:,}원")
                # 실제 환경에서는 self.upbit.sell_market_order() 사용
                
        except Exception as e:
            print(f"주문 실행 오류: {e}")


def run_backtest_example():
    """백테스트 예제 실행"""
    
    print("=== 백테스트 예제 ===")
    
    # 설정
    config = TradingConfig(
        initial_balance=1000000,
        lookback_window=30,
        model_type="dqn"
    )
    
    try:
        # 환경 및 에이전트 생성
        env = TradingEnvironment(config)
        obs, _ = env.reset()
        agent = DQNAgent(config, len(obs))
        
        # 백테스터 생성
        backtester = Backtester(config)
        
        # 백테스트 실행
        result = backtester.run_backtest(agent, env)
        
        # 벤치마크 데이터 (Buy & Hold 전략)
        initial_price = env._get_current_price()
        benchmark_data = []
        env.reset()
        
        # 벤치마크 생성 (단순 보유 전략)
        while True:
            current_price = env._get_current_price()
            benchmark_value = config.initial_balance * (current_price / initial_price)
            benchmark_data.append(benchmark_value)
            
            _, _, terminated, truncated, _ = env.step(ActionSpace.HOLD)
            if terminated or truncated:
                break
        
        # 벤치마크 비교
        comparison = backtester.compare_with_benchmark(result, benchmark_data)
        
        print("\n=== 백테스트 결과 ===")
        print(f"총 수익률: {result.total_return:.2%}")
        print(f"연환산 수익률: {result.annual_return:.2%}")
        print(f"샤프 비율: {result.sharpe_ratio:.2f}")
        print(f"최대 낙폭: {result.max_drawdown:.2%}")
        print(f"승률: {result.win_rate:.2%}")
        print(f"총 거래 횟수: {result.total_trades}")
        
        print("\n=== 벤치마크 비교 ===")
        print(f"에이전트 수익률: {comparison['agent_return']:.2%}")
        print(f"벤치마크 수익률: {comparison['benchmark_return']:.2%}")
        print(f"초과 수익률: {comparison['excess_return']:.2%}")
        
        # 결과 시각화
        backtester.plot_results(result, benchmark_data, "backtest_results.png")
        
    except Exception as e:
        print(f"백테스트 실행 오류: {e}")
        import traceback
        traceback.print_exc()


def run_realtime_trading_example():
    """실시간 트레이딩 예제 (시뮬레이션)"""
    
    print("=== 실시간 트레이딩 시뮬레이션 ===")
    
    config = TradingConfig()
    
    try:
        # 더미 모델 파일 생성 (실제로는 학습된 모델 사용)
        model_path = "models/final_model.pth"
        if not os.path.exists(model_path):
            print("학습된 모델이 없습니다. 먼저 DQN 학습을 실행하세요.")
            return
        
        # 실시간 트레이더 생성
        trader = RealTimeTrader(config, model_path)
        
        # 트레이딩 시작 (30초간 시뮬레이션)
        thread = trader.start_trading(update_interval=5)
        
        # 30초 후 중지
        time.sleep(30)
        trader.stop_trading()
        
        print("실시간 트레이딩 시뮬레이션 완료")
        
    except Exception as e:
        print(f"실시간 트레이딩 오류: {e}")


if __name__ == "__main__":
    # 백테스트 예제 실행
    run_backtest_example()
    
    # 실시간 트레이딩 예제 (주석 해제하여 사용)
    # run_realtime_trading_example()
