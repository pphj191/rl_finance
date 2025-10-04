"""
실시간 트레이딩 시스템

실제 트레이딩:
- ✅ 실시간 데이터 수집
- ✅ 모델 추론
- ✅ 주문 실행
- ✅ 리스크 관리
"""

import time
import threading
import queue
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd
import torch
from dataclasses import dataclass, asdict
import json

from upbit_api import UpbitAPI
from rl_trading_env import TradingEnvironment, TradingConfig, ActionSpace, FeatureExtractor
from dqn_agent import DQNAgent


@dataclass
class RiskConfig:
    """리스크 관리 설정"""
    max_position_size: float = 0.1  # 최대 포지션 크기 (총 자산 대비)
    stop_loss_pct: float = 0.05     # 손절 비율 (5%)
    take_profit_pct: float = 0.1    # 익절 비율 (10%)
    max_daily_trades: int = 10      # 일일 최대 거래 수
    min_trade_interval: int = 300   # 최소 거래 간격 (초)
    max_drawdown_pct: float = 0.2   # 최대 낙폭 (20%)


@dataclass
class TradeSignal:
    """거래 신호"""
    timestamp: datetime
    action: int
    confidence: float
    price: float
    position_size: float
    reason: str


@dataclass
class Position:
    """포지션 정보"""
    symbol: str
    quantity: float
    avg_price: float
    current_price: float
    unrealized_pnl: float
    entry_time: datetime
    
    @property
    def market_value(self) -> float:
        return self.quantity * self.current_price
    
    @property
    def pnl_pct(self) -> float:
        if self.avg_price == 0:
            return 0.0
        return (self.current_price - self.avg_price) / self.avg_price


class RiskManager:
    """리스크 관리자"""
    
    def __init__(self, config: RiskConfig):
        self.config = config
        self.daily_trades = 0
        self.last_trade_time = None
        self.peak_portfolio_value = 0.0
        self.trade_history = []
    
    def check_risk_limits(self, signal: TradeSignal, portfolio_value: float, 
                         current_position: Optional[Position] = None) -> bool:
        """리스크 한계 체크"""
        
        # 일일 거래 수 제한
        if self.daily_trades >= self.config.max_daily_trades:
            logging.warning(f"일일 최대 거래 수 초과: {self.daily_trades}")
            return False
        
        # 최소 거래 간격 체크
        if (self.last_trade_time and 
            (signal.timestamp - self.last_trade_time).seconds < self.config.min_trade_interval):
            logging.warning("최소 거래 간격 미달")
            return False
        
        # 포지션 크기 제한
        max_position_value = portfolio_value * self.config.max_position_size
        if signal.position_size * signal.price > max_position_value:
            logging.warning(f"포지션 크기 초과: {signal.position_size * signal.price} > {max_position_value}")
            return False
        
        # 손절/익절 체크
        if current_position:
            if signal.action == ActionSpace.SELL:
                pnl_pct = current_position.pnl_pct
                if pnl_pct <= -self.config.stop_loss_pct:
                    logging.info(f"손절 실행: {pnl_pct:.2%}")
                    return True
                elif pnl_pct >= self.config.take_profit_pct:
                    logging.info(f"익절 실행: {pnl_pct:.2%}")
                    return True
        
        # 최대 낙폭 체크
        if portfolio_value > 0:
            self.peak_portfolio_value = max(self.peak_portfolio_value, portfolio_value)
            current_drawdown = (self.peak_portfolio_value - portfolio_value) / self.peak_portfolio_value
            if current_drawdown > self.config.max_drawdown_pct:
                logging.error(f"최대 낙폭 초과: {current_drawdown:.2%}")
                return False
        
        return True
    
    def record_trade(self, signal: TradeSignal):
        """거래 기록"""
        self.daily_trades += 1
        self.last_trade_time = signal.timestamp
        self.trade_history.append(asdict(signal))
    
    def reset_daily_limits(self):
        """일일 제한 리셋"""
        self.daily_trades = 0


class RealTimeTrader:
    """실시간 트레이더"""
    
    def __init__(self, config: TradingConfig, risk_config: RiskConfig, 
                 model_path: str, market: str = "KRW-BTC"):
        self.config = config
        self.risk_config = risk_config
        self.market = market
        
        # API 및 환경 초기화
        self.upbit = UpbitAPI()
        self.feature_extractor = FeatureExtractor()
        self.risk_manager = RiskManager(risk_config)
        
        # 모델 로드
        self.agent = DQNAgent(config, state_size=100)  # 실제 크기는 데이터에 따라 조정
        self.agent.load_model(model_path)
        
        # 데이터 버퍼
        self.data_buffer = queue.Queue(maxsize=1000)
        self.price_history = []
        
        # 상태 관리
        self.is_running = False
        self.current_position = None
        self.portfolio_value = 0.0
        self.cash_balance = 0.0
        
        # 스레드
        self.data_thread = None
        self.trading_thread = None
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def start_trading(self):
        """트레이딩 시작"""
        self.logger.info("실시간 트레이딩 시작")
        self.is_running = True
        
        # 초기 포트폴리오 상태 확인
        self._update_portfolio_status()
        
        # 데이터 수집 스레드 시작
        self.data_thread = threading.Thread(target=self._data_collection_loop)
        self.data_thread.start()
        
        # 트레이딩 스레드 시작
        self.trading_thread = threading.Thread(target=self._trading_loop)
        self.trading_thread.start()
    
    def stop_trading(self):
        """트레이딩 중지"""
        self.logger.info("실시간 트레이딩 중지")
        self.is_running = False
        
        if self.data_thread:
            self.data_thread.join()
        if self.trading_thread:
            self.trading_thread.join()
    
    def _data_collection_loop(self):
        """데이터 수집 루프"""
        while self.is_running:
            try:
                # 현재 시장 데이터 수집
                market_data = self._collect_market_data()
                if market_data:
                    self.data_buffer.put(market_data)
                
                time.sleep(5)  # 5초마다 데이터 수집
                
            except Exception as e:
                self.logger.error(f"데이터 수집 오류: {e}")
                time.sleep(10)
    
    def _trading_loop(self):
        """트레이딩 루프"""
        while self.is_running:
            try:
                # 데이터 버퍼에서 최신 데이터 가져오기
                if not self.data_buffer.empty():
                    market_data = self.data_buffer.get()
                    
                    # 거래 신호 생성
                    signal = self._generate_trade_signal(market_data)
                    
                    if signal:
                        # 리스크 체크
                        if self.risk_manager.check_risk_limits(
                            signal, self.portfolio_value, self.current_position
                        ):
                            # 주문 실행
                            self._execute_trade(signal)
                
                time.sleep(10)  # 10초마다 트레이딩 로직 실행
                
            except Exception as e:
                self.logger.error(f"트레이딩 루프 오류: {e}")
                time.sleep(30)
    
    def _collect_market_data(self) -> Optional[Dict[str, Any]]:
        """시장 데이터 수집"""
        try:
            # 현재가 정보
            ticker = self.upbit.get_ticker(self.market)[0]
            
            # 최근 캔들 데이터
            candles = self.upbit.get_candles_minutes(self.market, unit=1, count=100)
            
            # DataFrame 변환 및 기술적 지표 계산
            df = pd.DataFrame(candles)
            df['timestamp'] = pd.to_datetime(df['candle_date_time_kst'])
            df = df.rename(columns={
                'opening_price': 'open',
                'high_price': 'high',
                'low_price': 'low',
                'trade_price': 'close',
                'candle_acc_trade_volume': 'volume'
            })
            
            # 기술적 지표 및 SSL 특성 추가
            df = self.feature_extractor.extract_technical_indicators(df)
            df = self.feature_extractor.extract_ssl_features(df)
            
            return {
                'timestamp': datetime.now(),
                'current_price': float(ticker['trade_price']),
                'volume': float(ticker['acc_trade_volume_24h']),
                'change_rate': float(ticker['change_rate']),
                'features': df.iloc[-1].fillna(0).values,  # 최신 특성 벡터
                'market_data': ticker
            }
            
        except Exception as e:
            self.logger.error(f"시장 데이터 수집 실패: {e}")
            return None
    
    def _generate_trade_signal(self, market_data: Dict[str, Any]) -> Optional[TradeSignal]:
        """거래 신호 생성"""
        try:
            # 포트폴리오 상태 업데이트
            self._update_portfolio_status()
            
            # 상태 벡터 구성
            features = market_data['features']
            portfolio_features = self._get_portfolio_features()
            state = np.concatenate([features, portfolio_features])
            
            # 액션 마스킹
            action_mask = self._get_action_mask()
            
            # 모델 예측
            action = self.agent.select_action(state, action_mask)  # epsilon 제거
            
            # 신뢰도 계산 (Q값 기반)
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                q_values = self.agent.q_network(state_tensor)
                confidence = torch.softmax(q_values, dim=1).max().item()
            
            # 신호 생성
            if action != ActionSpace.HOLD and confidence > 0.6:  # 최소 신뢰도 임계값
                position_size = self._calculate_position_size(action, market_data['current_price'])
                
                return TradeSignal(
                    timestamp=market_data['timestamp'],
                    action=action,
                    confidence=confidence,
                    price=market_data['current_price'],
                    position_size=position_size,
                    reason=f"Model prediction with {confidence:.2%} confidence"
                )
            
            return None
            
        except Exception as e:
            self.logger.error(f"신호 생성 실패: {e}")
            return None
    
    def _execute_trade(self, signal: TradeSignal):
        """주문 실행"""
        try:
            if signal.action == ActionSpace.BUY:
                # 매수 주문
                order_result = self.upbit.place_buy_order(
                    market=self.market,
                    price=signal.price,
                    volume=signal.position_size
                )
                
                if order_result:
                    self.logger.info(f"매수 주문 실행: {signal.position_size} at {signal.price}")
                    self._update_position_after_buy(signal)
                
            elif signal.action == ActionSpace.SELL and self.current_position:
                # 매도 주문
                order_result = self.upbit.place_sell_order(
                    market=self.market,
                    volume=self.current_position.quantity
                )
                
                if order_result:
                    self.logger.info(f"매도 주문 실행: {self.current_position.quantity} at {signal.price}")
                    self._update_position_after_sell(signal)
            
            # 거래 기록
            self.risk_manager.record_trade(signal)
            
        except Exception as e:
            self.logger.error(f"주문 실행 실패: {e}")
    
    def _update_portfolio_status(self):
        """포트폴리오 상태 업데이트"""
        try:
            # 계좌 잔고 조회
            accounts = self.upbit.get_accounts()
            
            krw_account = next((acc for acc in accounts if acc['currency'] == 'KRW'), None)
            crypto_account = next((acc for acc in accounts if acc['currency'] == self.market.split('-')[1]), None)
            
            self.cash_balance = float(krw_account['balance']) if krw_account else 0.0
            
            if crypto_account and float(crypto_account['balance']) > 0:
                # 현재 포지션 업데이트
                current_price = self.upbit.get_ticker(self.market)[0]['trade_price']
                self.current_position = Position(
                    symbol=self.market,
                    quantity=float(crypto_account['balance']),
                    avg_price=float(crypto_account['avg_buy_price']),
                    current_price=float(current_price),
                    unrealized_pnl=(float(current_price) - float(crypto_account['avg_buy_price'])) * float(crypto_account['balance']),
                    entry_time=datetime.now()  # 실제로는 거래 시간을 기록해야 함
                )
                
                self.portfolio_value = self.cash_balance + self.current_position.market_value
            else:
                self.current_position = None
                self.portfolio_value = self.cash_balance
                
        except Exception as e:
            self.logger.error(f"포트폴리오 상태 업데이트 실패: {e}")
    
    def _get_portfolio_features(self) -> np.ndarray:
        """포트폴리오 특성 벡터"""
        if self.current_position:
            return np.array([
                self.current_position.quantity,
                self.current_position.avg_price,
                self.current_position.pnl_pct,
                self.cash_balance / self.portfolio_value if self.portfolio_value > 0 else 1.0
            ])
        else:
            return np.array([0.0, 0.0, 0.0, 1.0])
    
    def _get_action_mask(self) -> np.ndarray:
        """액션 마스킹"""
        mask = np.array([True, True, True])  # [HOLD, BUY, SELL]
        
        # 포지션이 없으면 매도 불가
        if not self.current_position:
            mask[ActionSpace.SELL] = False
        
        # 현금이 없으면 매수 불가
        if self.cash_balance < 5000:  # 최소 주문 금액
            mask[ActionSpace.BUY] = False
        
        return mask
    
    def _calculate_position_size(self, action: int, price: float) -> float:
        """포지션 크기 계산"""
        if action == ActionSpace.BUY:
            # 사용 가능한 현금의 일정 비율로 매수
            max_investment = self.cash_balance * 0.95  # 수수료 고려
            max_quantity = max_investment / price
            
            # 리스크 관리 적용
            risk_adjusted_investment = self.portfolio_value * self.risk_config.max_position_size
            risk_adjusted_quantity = risk_adjusted_investment / price
            
            return min(max_quantity, risk_adjusted_quantity)
        
        return 0.0
    
    def _update_position_after_buy(self, signal: TradeSignal):
        """매수 후 포지션 업데이트"""
        # 실제로는 주문 체결 결과를 확인해야 함
        self.cash_balance -= signal.position_size * signal.price
    
    def _update_position_after_sell(self, signal: TradeSignal):
        """매도 후 포지션 업데이트"""
        # 실제로는 주문 체결 결과를 확인해야 함
        if self.current_position:
            self.cash_balance += self.current_position.quantity * signal.price
            self.current_position = None
    
    def get_performance_report(self) -> Dict[str, Any]:
        """성과 보고서"""
        return {
            'portfolio_value': self.portfolio_value,
            'cash_balance': self.cash_balance,
            'current_position': asdict(self.current_position) if self.current_position else None,
            'daily_trades': self.risk_manager.daily_trades,
            'trade_history': self.risk_manager.trade_history[-10:],  # 최근 10개 거래
            'peak_portfolio_value': self.risk_manager.peak_portfolio_value,
            'timestamp': datetime.now().isoformat()
        }


if __name__ == "__main__":
    # 설정
    trading_config = TradingConfig(
        model_type="dqn",
        hidden_size=256,
        learning_rate=0.001
    )
    
    risk_config = RiskConfig(
        max_position_size=0.1,
        stop_loss_pct=0.05,
        take_profit_pct=0.1,
        max_daily_trades=5
    )
    
    # 실시간 트레이더 생성
    trader = RealTimeTrader(
        config=trading_config,
        risk_config=risk_config,
        model_path="models/best_model.pth",  # 학습된 모델 경로
        market="KRW-BTC"
    )
    
    try:
        # 트레이딩 시작
        trader.start_trading()
        
        # 30분 동안 실행
        time.sleep(1800)
        
        # 성과 보고서 출력
        report = trader.get_performance_report()
        print(json.dumps(report, indent=2, ensure_ascii=False))
        
    except KeyboardInterrupt:
        print("사용자 중단")
    finally:
        trader.stop_trading()
