"""
실시간 트레이딩 엔진 모듈

실시간으로 RL 에이전트를 사용하여 트레이딩을 수행합니다.

최종 업데이트: 2025-10-06 00:00:00
"""

import time
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

from trading_env.trading_env import TradingEnvironment
from models.rl_agent import RLAgent
from api.upbit_api import UpbitAPI


@dataclass
class RiskConfig:
    """리스크 관리 설정"""
    stop_loss: float = 0.05  # 손절 비율 (5%)
    take_profit: float = 0.15  # 익절 비율 (15%)
    max_position_size: float = 0.3  # 최대 포지션 크기 (30%)
    max_daily_loss: float = 0.10  # 최대 일일 손실 (10%)
    trailing_stop: float = 0.03  # 트레일링 스탑 (3%)


class RiskManager:
    """리스크 관리자"""
    
    def __init__(self, config: RiskConfig):
        self.config = config
        self.entry_price: Optional[float] = None
        self.highest_price: Optional[float] = None
        self.daily_start_value: float = 0.0
        self.daily_losses: List[float] = []
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def set_entry(self, price: float, portfolio_value: float) -> None:
        """진입 가격 설정
        
        Args:
            price: 진입 가격
            portfolio_value: 현재 포트폴리오 가치
        """
        self.entry_price = price
        self.highest_price = price
        self.daily_start_value = portfolio_value
        self.logger.info(f"포지션 진입: {price:,.0f}원")
    
    def update_highest_price(self, current_price: float) -> None:
        """최고가 업데이트
        
        Args:
            current_price: 현재 가격
        """
        if self.highest_price is None or current_price > self.highest_price:
            self.highest_price = current_price
    
    def check_stop_loss(self, current_price: float) -> bool:
        """손절 확인
        
        Args:
            current_price: 현재 가격
            
        Returns:
            손절 필요 여부
        """
        if self.entry_price is None:
            return False
        
        loss_ratio = (current_price - self.entry_price) / self.entry_price
        
        if loss_ratio <= -self.config.stop_loss:
            self.logger.warning(f"손절 발동: {loss_ratio*100:.2f}%")
            return True
        
        return False
    
    def check_take_profit(self, current_price: float) -> bool:
        """익절 확인
        
        Args:
            current_price: 현재 가격
            
        Returns:
            익절 필요 여부
        """
        if self.entry_price is None:
            return False
        
        profit_ratio = (current_price - self.entry_price) / self.entry_price
        
        if profit_ratio >= self.config.take_profit:
            self.logger.info(f"익절 발동: {profit_ratio*100:.2f}%")
            return True
        
        return False
    
    def check_trailing_stop(self, current_price: float) -> bool:
        """트레일링 스탑 확인
        
        Args:
            current_price: 현재 가격
            
        Returns:
            트레일링 스탑 발동 여부
        """
        if self.highest_price is None:
            return False
        
        self.update_highest_price(current_price)
        
        drop_from_high = (self.highest_price - current_price) / self.highest_price
        
        if drop_from_high >= self.config.trailing_stop:
            self.logger.info(f"트레일링 스탑 발동: 최고가 대비 -{drop_from_high*100:.2f}%")
            return True
        
        return False
    
    def check_daily_loss_limit(self, current_value: float) -> bool:
        """일일 손실 한도 확인
        
        Args:
            current_value: 현재 포트폴리오 가치
            
        Returns:
            일일 손실 한도 초과 여부
        """
        if self.daily_start_value == 0:
            return False
        
        daily_loss = (current_value - self.daily_start_value) / self.daily_start_value
        
        if daily_loss <= -self.config.max_daily_loss:
            self.logger.error(f"일일 손실 한도 초과: {daily_loss*100:.2f}%")
            return True
        
        return False
    
    def reset_position(self) -> None:
        """포지션 초기화"""
        self.entry_price = None
        self.highest_price = None
        self.logger.info("포지션 초기화")


class TradingMonitor:
    """트레이딩 모니터"""
    
    def __init__(self):
        self.trades: List[Dict] = []
        self.portfolio_values: List[float] = []
        self.start_time = datetime.now()
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def record_trade(self, trade: Dict) -> None:
        """거래 기록
        
        Args:
            trade: 거래 정보
        """
        trade['timestamp'] = datetime.now()
        self.trades.append(trade)
        self.logger.info(f"거래 기록: {trade['action']} @ {trade['price']:,.0f}원")
    
    def record_portfolio_value(self, value: float) -> None:
        """포트폴리오 가치 기록
        
        Args:
            value: 포트폴리오 가치
        """
        self.portfolio_values.append(value)
    
    def get_statistics(self) -> Dict:
        """통계 정보 반환
        
        Returns:
            통계 딕셔너리
        """
        if not self.portfolio_values:
            return {}
        
        initial_value = self.portfolio_values[0]
        current_value = self.portfolio_values[-1]
        total_return = (current_value - initial_value) / initial_value
        
        # 거래 통계
        buy_trades = sum(1 for t in self.trades if t['action'] == 'BUY')
        sell_trades = sum(1 for t in self.trades if t['action'] == 'SELL')
        
        # 승률 계산
        winning_trades = 0
        total_completed_trades = 0
        
        for i in range(len(self.trades) - 1):
            if self.trades[i]['action'] == 'BUY' and self.trades[i+1]['action'] == 'SELL':
                total_completed_trades += 1
                if self.trades[i+1]['total_value'] > self.trades[i]['total_value']:
                    winning_trades += 1
        
        win_rate = winning_trades / total_completed_trades if total_completed_trades > 0 else 0.0
        
        # 운영 시간
        runtime = (datetime.now() - self.start_time).total_seconds() / 3600
        
        return {
            'runtime_hours': runtime,
            'total_return': total_return,
            'current_value': current_value,
            'initial_value': initial_value,
            'total_trades': len(self.trades),
            'buy_trades': buy_trades,
            'sell_trades': sell_trades,
            'win_rate': win_rate,
            'completed_trades': total_completed_trades
        }


class RealtimeTrader:
    """실시간 트레이더"""
    
    def __init__(
        self,
        env: TradingEnvironment,
        agent: RLAgent,
        api: UpbitAPI,
        risk_config: Optional[RiskConfig] = None,
        dry_run: bool = True
    ):
        """
        Args:
            env: 트레이딩 환경
            agent: RL 에이전트
            api: Upbit API
            risk_config: 리스크 설정
            dry_run: 모의 거래 모드
        """
        self.env = env
        self.agent = agent
        self.api = api
        self.dry_run = dry_run
        
        self.risk_manager = RiskManager(risk_config or RiskConfig())
        self.monitor = TradingMonitor()
        
        self.logger = logging.getLogger(self.__class__.__name__)
        self.running = False
    
    def start(self, duration_hours: Optional[float] = None, update_interval: int = 60) -> None:
        """트레이딩 시작
        
        Args:
            duration_hours: 실행 시간 (시간), None이면 무한정 실행
            update_interval: 업데이트 간격 (초)
        """
        self.running = True
        self.logger.info("실시간 트레이딩 시작")
        
        if self.dry_run:
            self.logger.warning("⚠️ 모의 거래 모드로 실행 중입니다.")
        else:
            self.logger.warning("🔴 실거래 모드로 실행 중입니다!")
        
        start_time = time.time()
        state = self.env.reset()
        
        # 초기 포트폴리오 값 설정
        initial_value = self.env.cash + self.env.position * self.env.current_price
        self.risk_manager.daily_start_value = initial_value
        self.monitor.record_portfolio_value(initial_value)
        
        try:
            while self.running:
                # 종료 시간 체크
                if duration_hours and (time.time() - start_time) / 3600 >= duration_hours:
                    self.logger.info(f"지정된 시간({duration_hours}시간) 경과, 트레이딩 종료")
                    break
                
                # 에이전트 행동 선택
                action = self.agent.select_action(state, training=False)
                
                # 환경 업데이트 (실시간 데이터 가져오기)
                current_price = self._get_current_price()
                
                # 리스크 관리 체크
                if self.env.position > 0:
                    # 손절/익절/트레일링 스탑 체크
                    if self.risk_manager.check_stop_loss(current_price):
                        action = 2  # SELL
                        self.logger.warning("리스크 관리: 손절 실행")
                    elif self.risk_manager.check_take_profit(current_price):
                        action = 2  # SELL
                        self.logger.info("리스크 관리: 익절 실행")
                    elif self.risk_manager.check_trailing_stop(current_price):
                        action = 2  # SELL
                        self.logger.info("리스크 관리: 트레일링 스탑 실행")
                
                # 일일 손실 한도 체크
                portfolio_value = self.env.cash + self.env.position * current_price
                if self.risk_manager.check_daily_loss_limit(portfolio_value):
                    self.logger.error("일일 손실 한도 초과, 모든 포지션 청산")
                    if self.env.position > 0:
                        action = 2  # SELL
                    self.running = False
                
                # 액션 실행
                next_state, reward, done, info = self.env.step(action)
                
                # 거래 기록
                if action != 0:  # HOLD가 아닌 경우
                    trade = {
                        'action': self.env.action_space.actions[action],
                        'price': current_price,
                        'position': self.env.position,
                        'cash': self.env.cash,
                        'total_value': portfolio_value
                    }
                    self.monitor.record_trade(trade)
                    
                    # 실거래 실행
                    if not self.dry_run:
                        self._execute_real_trade(action, current_price)
                    
                    # 포지션 진입 시 리스크 관리 설정
                    if action == 1 and self.env.position > 0:  # BUY
                        self.risk_manager.set_entry(current_price, portfolio_value)
                    elif action == 2 and self.env.position == 0:  # SELL (포지션 청산)
                        self.risk_manager.reset_position()
                
                # 포트폴리오 가치 기록
                self.monitor.record_portfolio_value(portfolio_value)
                
                # 상태 업데이트
                state = next_state
                
                # 진행 상황 출력
                if len(self.monitor.portfolio_values) % 10 == 0:
                    self._print_status()
                
                # 대기
                time.sleep(update_interval)
                
        except KeyboardInterrupt:
            self.logger.info("사용자에 의해 중단되었습니다.")
        except Exception as e:
            self.logger.error(f"오류 발생: {e}", exc_info=True)
        finally:
            self.stop()
    
    def stop(self) -> None:
        """트레이딩 중지"""
        self.running = False
        self.logger.info("트레이딩 종료")
        
        # 최종 통계 출력
        self._print_final_statistics()
    
    def _get_current_price(self) -> float:
        """현재 시장 가격 가져오기
        
        Returns:
            현재 가격
        """
        try:
            ticker = self.api.get_ticker(self.env.market)
            return ticker['trade_price']
        except Exception as e:
            self.logger.error(f"가격 조회 실패: {e}")
            return self.env.current_price
    
    def _execute_real_trade(self, action: int, price: float) -> None:
        """실거래 실행
        
        Args:
            action: 행동 (1: BUY, 2: SELL)
            price: 가격
        """
        try:
            if action == 1:  # BUY
                # 매수 주문
                volume = (self.env.cash * 0.99) / price  # 수수료 고려
                order = self.api.buy_market_order(self.env.market, volume)
                self.logger.info(f"매수 주문 실행: {order}")
            
            elif action == 2:  # SELL
                # 매도 주문
                order = self.api.sell_market_order(self.env.market, self.env.position)
                self.logger.info(f"매도 주문 실행: {order}")
        
        except Exception as e:
            self.logger.error(f"실거래 실행 실패: {e}", exc_info=True)
    
    def _print_status(self) -> None:
        """현재 상태 출력"""
        stats = self.monitor.get_statistics()
        
        if not stats:
            return
        
        self.logger.info("=" * 50)
        self.logger.info(f"운영 시간: {stats['runtime_hours']:.2f}시간")
        self.logger.info(f"총 수익률: {stats['total_return']*100:.2f}%")
        self.logger.info(f"현재 가치: {stats['current_value']:,.0f}원")
        self.logger.info(f"총 거래: {stats['total_trades']}회 (매수: {stats['buy_trades']}, 매도: {stats['sell_trades']})")
        self.logger.info(f"승률: {stats['win_rate']*100:.2f}%")
        self.logger.info("=" * 50)
    
    def _print_final_statistics(self) -> None:
        """최종 통계 출력"""
        stats = self.monitor.get_statistics()
        
        if not stats:
            self.logger.info("거래 기록이 없습니다.")
            return
        
        self.logger.info("\n" + "=" * 60)
        self.logger.info("최종 트레이딩 결과")
        self.logger.info("=" * 60)
        self.logger.info(f"총 운영 시간:     {stats['runtime_hours']:.2f}시간")
        self.logger.info(f"초기 자산:        {stats['initial_value']:,.0f}원")
        self.logger.info(f"최종 자산:        {stats['current_value']:,.0f}원")
        self.logger.info(f"총 수익:          {(stats['current_value'] - stats['initial_value']):,.0f}원")
        self.logger.info(f"총 수익률:        {stats['total_return']*100:.2f}%")
        self.logger.info(f"총 거래 횟수:     {stats['total_trades']}회")
        self.logger.info(f"  - 매수:         {stats['buy_trades']}회")
        self.logger.info(f"  - 매도:         {stats['sell_trades']}회")
        self.logger.info(f"완료된 거래:      {stats['completed_trades']}회")
        self.logger.info(f"승률:             {stats['win_rate']*100:.2f}%")
        self.logger.info("=" * 60)
