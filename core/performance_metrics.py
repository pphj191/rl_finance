"""
성과 지표 계산 모듈

트레이딩 성과를 다양한 지표로 평가합니다.

최종 업데이트: 2025-10-05 23:50:00
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from dataclasses import dataclass

from core.backtesting_engine import BacktestResult


class PerformanceMetrics:
    """성과 지표 계산기"""
    
    def __init__(self, risk_free_rate: float = 0.02):
        """
        Args:
            risk_free_rate: 무위험 수익률 (연환산)
        """
        self.risk_free_rate = risk_free_rate
    
    def calculate_all(self, result: BacktestResult) -> Dict:
        """모든 성과 지표 계산
        
        Args:
            result: 백테스트 결과
            
        Returns:
            성과 지표 딕셔너리
        """
        metrics = {
            # 기본 수익률 지표
            'total_return': result.total_return,
            'annual_return': result.annual_return,
            'max_drawdown': result.max_drawdown,
            
            # 리스크 조정 지표
            'sharpe_ratio': result.sharpe_ratio,
            'sortino_ratio': self.calculate_sortino_ratio(result.daily_returns),
            'calmar_ratio': self.calculate_calmar_ratio(result.annual_return, result.max_drawdown),
            
            # 변동성 지표
            'volatility': result.volatility,
            'downside_deviation': self.calculate_downside_deviation(result.daily_returns),
            
            # 거래 지표
            'total_trades': result.total_trades,
            'win_rate': result.win_rate,
            'profit_factor': result.profit_factor,
            'avg_trade_return': self.calculate_avg_trade_return(result.trades),
            
            # 추가 지표
            'max_consecutive_wins': self.calculate_max_consecutive_wins(result.trades),
            'max_consecutive_losses': self.calculate_max_consecutive_losses(result.trades),
            'recovery_factor': self.calculate_recovery_factor(result.total_return, result.max_drawdown),
            'profit_to_drawdown_ratio': self.calculate_profit_to_drawdown_ratio(result.total_return, result.max_drawdown)
        }
        
        return metrics
    
    def calculate_sortino_ratio(self, daily_returns: List[float]) -> float:
        """Sortino 비율 계산 (하방 리스크만 고려)
        
        Args:
            daily_returns: 일일 수익률 리스트
            
        Returns:
            Sortino 비율
        """
        if not daily_returns or len(daily_returns) < 2:
            return 0.0
        
        returns = np.array(daily_returns)
        excess_returns = returns - (self.risk_free_rate / 252)
        
        # 하방 편차 (음수 수익률만 고려)
        downside_returns = returns[returns < 0]
        downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 0.0
        
        if downside_std == 0:
            return 0.0
        
        sortino = np.mean(excess_returns) / downside_std * np.sqrt(252)
        return float(sortino)
    
    def calculate_calmar_ratio(self, annual_return: float, max_drawdown: float) -> float:
        """Calmar 비율 계산 (수익률 / 최대낙폭)
        
        Args:
            annual_return: 연환산 수익률
            max_drawdown: 최대 낙폭
            
        Returns:
            Calmar 비율
        """
        if max_drawdown == 0:
            return 0.0
        
        return annual_return / abs(max_drawdown)
    
    def calculate_downside_deviation(self, daily_returns: List[float]) -> float:
        """하방 편차 계산
        
        Args:
            daily_returns: 일일 수익률 리스트
            
        Returns:
            하방 편차
        """
        if not daily_returns:
            return 0.0
        
        returns = np.array(daily_returns)
        downside_returns = returns[returns < 0]
        
        if len(downside_returns) == 0:
            return 0.0
        
        return float(np.std(downside_returns))
    
    def calculate_avg_trade_return(self, trades: List[Dict]) -> float:
        """평균 거래 수익률 계산
        
        Args:
            trades: 거래 리스트
            
        Returns:
            평균 거래 수익률
        """
        if len(trades) < 2:
            return 0.0
        
        trade_returns = []
        
        for i in range(1, len(trades)):
            if trades[i]['action'] == 'SELL' and trades[i-1]['action'] == 'BUY':
                ret = (trades[i]['total_value'] - trades[i-1]['total_value']) / trades[i-1]['total_value']
                trade_returns.append(ret)
        
        return float(np.mean(trade_returns)) if trade_returns else 0.0
    
    def calculate_max_consecutive_wins(self, trades: List[Dict]) -> int:
        """최대 연속 승리 횟수 계산
        
        Args:
            trades: 거래 리스트
            
        Returns:
            최대 연속 승리 횟수
        """
        if len(trades) < 2:
            return 0
        
        max_wins = 0
        current_wins = 0
        
        for i in range(1, len(trades)):
            if trades[i]['action'] == 'SELL' and trades[i-1]['action'] == 'BUY':
                if trades[i]['total_value'] > trades[i-1]['total_value']:
                    current_wins += 1
                    max_wins = max(max_wins, current_wins)
                else:
                    current_wins = 0
        
        return max_wins
    
    def calculate_max_consecutive_losses(self, trades: List[Dict]) -> int:
        """최대 연속 손실 횟수 계산
        
        Args:
            trades: 거래 리스트
            
        Returns:
            최대 연속 손실 횟수
        """
        if len(trades) < 2:
            return 0
        
        max_losses = 0
        current_losses = 0
        
        for i in range(1, len(trades)):
            if trades[i]['action'] == 'SELL' and trades[i-1]['action'] == 'BUY':
                if trades[i]['total_value'] < trades[i-1]['total_value']:
                    current_losses += 1
                    max_losses = max(max_losses, current_losses)
                else:
                    current_losses = 0
        
        return max_losses
    
    def calculate_recovery_factor(self, total_return: float, max_drawdown: float) -> float:
        """회복 계수 계산
        
        Args:
            total_return: 총 수익률
            max_drawdown: 최대 낙폭
            
        Returns:
            회복 계수
        """
        if max_drawdown == 0:
            return 0.0
        
        return total_return / abs(max_drawdown)
    
    def calculate_profit_to_drawdown_ratio(self, total_return: float, max_drawdown: float) -> float:
        """수익-낙폭 비율 계산
        
        Args:
            total_return: 총 수익률
            max_drawdown: 최대 낙폭
            
        Returns:
            수익-낙폭 비율
        """
        if max_drawdown == 0:
            return 0.0
        
        return (1 + total_return) / (1 + abs(max_drawdown))
    
    def compare_with_benchmark(self, result: BacktestResult, benchmark: BacktestResult) -> Dict:
        """벤치마크와 비교
        
        Args:
            result: 에이전트 결과
            benchmark: 벤치마크 결과
            
        Returns:
            비교 지표 딕셔너리
        """
        comparison = {
            'agent_return': result.total_return,
            'benchmark_return': benchmark.total_return,
            'excess_return': result.total_return - benchmark.total_return,
            'outperformance': (result.total_return + 1) / (benchmark.total_return + 1) - 1,
            
            # 리스크 비교
            'agent_sharpe': result.sharpe_ratio,
            'benchmark_sharpe': benchmark.sharpe_ratio,
            'sharpe_diff': result.sharpe_ratio - benchmark.sharpe_ratio,
            
            # 변동성 비교
            'agent_volatility': result.volatility,
            'benchmark_volatility': benchmark.volatility,
            'volatility_diff': result.volatility - benchmark.volatility,
            
            # 낙폭 비교
            'agent_max_drawdown': result.max_drawdown,
            'benchmark_max_drawdown': benchmark.max_drawdown,
            'drawdown_diff': result.max_drawdown - benchmark.max_drawdown,
            
            # 정보 비율 (Information Ratio)
            'information_ratio': self.calculate_information_ratio(result, benchmark)
        }
        
        return comparison
    
    def calculate_information_ratio(self, result: BacktestResult, benchmark: BacktestResult) -> float:
        """정보 비율 계산
        
        Args:
            result: 에이전트 결과
            benchmark: 벤치마크 결과
            
        Returns:
            정보 비율
        """
        if not result.daily_returns or not benchmark.daily_returns:
            return 0.0
        
        # 길이 맞추기
        min_len = min(len(result.daily_returns), len(benchmark.daily_returns))
        agent_returns = np.array(result.daily_returns[:min_len])
        benchmark_returns = np.array(benchmark.daily_returns[:min_len])
        
        # 초과 수익률
        excess_returns = agent_returns - benchmark_returns
        
        # 추적 오차 (Tracking Error)
        tracking_error = np.std(excess_returns)
        
        if tracking_error == 0:
            return 0.0
        
        information_ratio = np.mean(excess_returns) / tracking_error * np.sqrt(252)
        return float(information_ratio)
