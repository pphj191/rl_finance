"""
백테스팅 엔진 모듈

과거 데이터를 사용하여 트레이딩 전략의 성과를 평가합니다.

최종 업데이트: 2025-10-05 23:45:00
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict

from trading_env import TradingEnvironment, TradingConfig, ActionSpace
from rl_agent import RLAgent


@dataclass
class BacktestResult:
    """백테스트 결과"""
    # 포트폴리오 정보
    equity_curve: List[float]
    daily_returns: List[float]
    
    # 거래 기록
    trades: List[Dict]
    positions: List[Dict]
    
    # 기본 지표
    total_return: float
    annual_return: float
    max_drawdown: float
    
    # 거래 지표
    total_trades: int
    win_rate: float
    profit_factor: float
    
    # 리스크 지표
    sharpe_ratio: float
    volatility: float
    
    # 메타 정보
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    initial_balance: float = 0
    final_balance: float = 0
    
    def to_dict(self) -> Dict:
        """딕셔너리로 변환"""
        return asdict(self)


class BacktestEngine:
    """백테스팅 엔진"""
    
    def __init__(self, config: TradingConfig):
        """
        Args:
            config: 트레이딩 설정
        """
        self.config = config
        self.results = None
    
    def run(
        self,
        agent: RLAgent,
        env: TradingEnvironment,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        verbose: bool = True
    ) -> BacktestResult:
        """백테스트 실행
        
        Args:
            agent: 트레이딩 에이전트
            env: 트레이딩 환경
            start_date: 시작 날짜
            end_date: 종료 날짜
            verbose: 진행 상황 출력 여부
            
        Returns:
            BacktestResult 객체
        """
        if verbose:
            print("백테스트 실행 중...")
        
        # 환경 초기화
        state, _ = env.reset()
        
        # 기록 변수
        equity_curve = [env.total_value]
        trades = []
        positions = []
        daily_returns = []
        
        initial_balance = env.total_value
        step = 0
        previous_action = ActionSpace.HOLD
        
        # 백테스트 루프
        while True:
            # 액션 선택 (평가 모드)
            action_mask = env.get_action_mask()
            action = agent.select_action(state, action_mask, training=False)
            
            # 환경 스텝
            next_state, reward, terminated, truncated, info = env.step(action)
            
            # 거래 기록
            if action != ActionSpace.HOLD:
                trade = {
                    'step': step,
                    'action': ActionSpace.get_action_names()[action],
                    'price': info['current_price'],
                    'balance': info['balance'],
                    'position': info['position'],
                    'total_value': info['total_value'],
                    'timestamp': datetime.now().isoformat()
                }
                trades.append(trade)
            
            # 포지션 기록
            if info['position'] > 0:
                position = {
                    'step': step,
                    'quantity': info['position'],
                    'price': info['current_price'],
                    'value': info['position'] * info['current_price']
                }
                positions.append(position)
            
            # 포트폴리오 가치 기록
            equity_curve.append(info['total_value'])
            
            # 일일 수익률 계산
            if len(equity_curve) > 1:
                daily_return = (equity_curve[-1] - equity_curve[-2]) / equity_curve[-2]
                daily_returns.append(daily_return)
            
            # 다음 스텝
            state = next_state
            previous_action = action
            step += 1
            
            # 진행 상황 출력
            if verbose and step % 100 == 0:
                print(f"Step {step}: 총 가치 = {info['total_value']:,.0f}원")
            
            # 종료 조건
            if terminated or truncated:
                break
        
        if verbose:
            print(f"백테스트 완료: 총 {step} 스텝")
        
        # 결과 생성
        final_balance = equity_curve[-1]
        
        result = BacktestResult(
            equity_curve=equity_curve,
            daily_returns=daily_returns,
            trades=trades,
            positions=positions,
            total_return=(final_balance - initial_balance) / initial_balance,
            annual_return=self._calculate_annual_return(initial_balance, final_balance, len(equity_curve)),
            max_drawdown=self._calculate_max_drawdown(equity_curve),
            total_trades=len(trades),
            win_rate=self._calculate_win_rate(trades),
            profit_factor=self._calculate_profit_factor(trades),
            sharpe_ratio=self._calculate_sharpe_ratio(daily_returns),
            volatility=np.std(daily_returns) if daily_returns else 0.0,
            start_date=start_date,
            end_date=end_date,
            initial_balance=initial_balance,
            final_balance=final_balance
        )
        
        self.results = result
        return result
    
    def run_benchmark(self, env: TradingEnvironment, verbose: bool = False) -> BacktestResult:
        """벤치마크 (Buy & Hold) 실행
        
        Args:
            env: 트레이딩 환경
            verbose: 진행 상황 출력 여부
            
        Returns:
            BacktestResult 객체
        """
        if verbose:
            print("벤치마크 (Buy & Hold) 실행 중...")
        
        # 환경 초기화
        env.reset()
        
        initial_price = env._get_current_price()
        initial_balance = env.balance
        
        # 전체 매수
        env.step(ActionSpace.BUY)
        
        equity_curve = [env.total_value]
        daily_returns = []
        
        # Hold 유지
        while True:
            _, _, terminated, truncated, info = env.step(ActionSpace.HOLD)
            
            equity_curve.append(info['total_value'])
            
            if len(equity_curve) > 1:
                daily_return = (equity_curve[-1] - equity_curve[-2]) / equity_curve[-2]
                daily_returns.append(daily_return)
            
            if terminated or truncated:
                break
        
        final_balance = equity_curve[-1]
        
        result = BacktestResult(
            equity_curve=equity_curve,
            daily_returns=daily_returns,
            trades=[{'action': 'BUY', 'step': 0}, {'action': 'SELL', 'step': len(equity_curve)-1}],
            positions=[],
            total_return=(final_balance - initial_balance) / initial_balance,
            annual_return=self._calculate_annual_return(initial_balance, final_balance, len(equity_curve)),
            max_drawdown=self._calculate_max_drawdown(equity_curve),
            total_trades=2,
            win_rate=1.0 if final_balance > initial_balance else 0.0,
            profit_factor=float('inf') if final_balance > initial_balance else 0.0,
            sharpe_ratio=self._calculate_sharpe_ratio(daily_returns),
            volatility=np.std(daily_returns) if daily_returns else 0.0,
            initial_balance=initial_balance,
            final_balance=final_balance
        )
        
        return result
    
    def _calculate_annual_return(self, initial: float, final: float, steps: int) -> float:
        """연환산 수익률 계산"""
        if steps == 0 or initial == 0:
            return 0.0
        
        # 252 거래일 기준
        return (final / initial) ** (252 / steps) - 1
    
    def _calculate_max_drawdown(self, equity_curve: List[float]) -> float:
        """최대 낙폭 계산"""
        if not equity_curve:
            return 0.0
        
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (np.array(equity_curve) - peak) / peak
        return float(np.min(drawdown))
    
    def _calculate_win_rate(self, trades: List[Dict]) -> float:
        """승률 계산"""
        if len(trades) < 2:
            return 0.0
        
        # 매수-매도 쌍 찾기
        profitable_trades = 0
        total_pairs = 0
        
        for i in range(1, len(trades)):
            if trades[i]['action'] == 'SELL' and trades[i-1]['action'] == 'BUY':
                if trades[i]['total_value'] > trades[i-1]['total_value']:
                    profitable_trades += 1
                total_pairs += 1
        
        return profitable_trades / total_pairs if total_pairs > 0 else 0.0
    
    def _calculate_profit_factor(self, trades: List[Dict]) -> float:
        """Profit Factor 계산"""
        if len(trades) < 2:
            return 0.0
        
        profits = []
        losses = []
        
        for i in range(1, len(trades)):
            if trades[i]['action'] == 'SELL' and trades[i-1]['action'] == 'BUY':
                pnl = trades[i]['total_value'] - trades[i-1]['total_value']
                if pnl > 0:
                    profits.append(pnl)
                else:
                    losses.append(abs(pnl))
        
        total_profit = sum(profits) if profits else 0
        total_loss = sum(losses) if losses else 0
        
        return total_profit / total_loss if total_loss > 0 else float('inf')
    
    def _calculate_sharpe_ratio(self, daily_returns: List[float], risk_free_rate: float = 0.02) -> float:
        """샤프 비율 계산"""
        if not daily_returns or len(daily_returns) < 2:
            return 0.0
        
        returns = np.array(daily_returns)
        excess_returns = returns - (risk_free_rate / 252)  # 일일 무위험 수익률
        
        if np.std(excess_returns) == 0:
            return 0.0
        
        sharpe = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
        return float(sharpe)
