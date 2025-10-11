"""
강화학습 에이전트 및 학습 코드

다양한 모델(DQN, LSTM, Transformer, Ensemble)을 지원하는 범용 RL 에이전트

TODO:
1. 학습 구현
   - ✅ RL 에이전트 클래스
   - ✅ 경험 재생 버퍼
   - ✅ 학습 루프
   - ✅ 타겟 네트워크 업데이트
   - ✅ 엡실론 그리디 정책

2. 모델 구조
   - ✅ DQN 기본 모델
   - ✅ LSTM/GRU 기반 모델
   - ✅ Transformer 기반 모델
   - ✅ 앙상블 모델

3. 백테스팅
   - ✅ 백테스팅 환경
   - ✅ 성과 지표 계산
   - ✅ 시각화

4. 모델 저장/로드
   - ✅ 체크포인트 저장
   - ✅ 최적 모델 선택
   - ✅ 모델 로드 및 추론
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
import random
from collections import deque, namedtuple
from typing import List, Tuple, Optional, Dict, Any
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import json
import logging
from trading_env import TradingEnvironment, TradingConfig, ActionSpace
from models import create_model, ModelConfig

# 경험 튜플 정의
Experience = namedtuple('Experience', 
                       ['state', 'action', 'reward', 'next_state', 'done', 'action_mask'])


class ReplayBuffer:
    """경험 재생 버퍼"""
    
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, experience: Experience):
        """경험 추가"""
        self.buffer.append(experience)
    
    def sample(self, batch_size: int) -> List[Experience]:
        """배치 샘플링"""
        return random.sample(self.buffer, batch_size)
    
    def __len__(self) -> int:
        return len(self.buffer)


class RLAgent:
    """강화학습 에이전트 (DQN, LSTM, Transformer, Ensemble 지원)"""
    
    def __init__(self, config: TradingConfig, state_size: int, 
                 device: str = "cpu"):
        self.config = config
        self.state_size = state_size
        self.action_size = ActionSpace.get_num_actions()
        self.device = device
        
        # ModelConfig로 변환
        model_config = ModelConfig(
            model_type=config.model_type,
            hidden_size=config.hidden_size,
            num_layers=getattr(config, 'num_layers', 3),
            dropout=getattr(config, 'dropout', 0.3),
            sequence_length=getattr(config, 'sequence_length', 60),
            d_model=getattr(config, 'd_model', 256),
            nhead=getattr(config, 'nhead', 8),
            num_encoder_layers=getattr(config, 'num_encoder_layers', 6)
        )
        
        # 신경망 초기화
        self.q_network = create_model(model_config, state_size, self.action_size).to(device)
        self.target_network = create_model(model_config, state_size, self.action_size).to(device)
        self.optimizer = optim.Adam(self.q_network.parameters(), 
                                   lr=config.learning_rate)
        
        # 타겟 네트워크를 메인 네트워크로 초기화
        self.update_target_network()
        
        # 경험 재생 버퍼
        self.memory = ReplayBuffer(config.memory_size)
        
        # 탐험 파라미터
        self.epsilon = config.epsilon_start
        self.epsilon_end = config.epsilon_end
        self.epsilon_decay = config.epsilon_decay
        
        # 학습 상태
        self.steps_done = 0
        self.episode_rewards = []
        self.losses = []
    
    def select_action(self, state: np.ndarray, action_mask: np.ndarray, 
                     training: bool = True) -> int:
        """액션 선택 (엡실론 그리디)"""
        if training and random.random() < self.epsilon:
            # 탐험: 유효한 액션 중 랜덤 선택
            valid_actions = np.where(action_mask)[0]
            return random.choice(valid_actions)
        else:
            # 활용: Q값이 가장 높은 유효한 액션 선택
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.q_network(state_tensor)
                
                # 액션 마스킹 적용
                masked_q_values = q_values.clone()
                masked_q_values[0, ~action_mask] = float('-inf')
                
                return masked_q_values.argmax().item()
    
    def store_experience(self, state: np.ndarray, action: int, reward: float,
                        next_state: np.ndarray, done: bool, action_mask: np.ndarray):
        """경험 저장"""
        experience = Experience(state, action, reward, next_state, done, action_mask)
        self.memory.push(experience)
    
    def train_step(self) -> float:
        """한 스텝 학습"""
        if len(self.memory) < self.config.batch_size:
            return 0.0
        
        # 배치 샘플링
        experiences = self.memory.sample(self.config.batch_size)
        
        states = torch.FloatTensor([e.state for e in experiences]).to(self.device)
        actions = torch.LongTensor([e.action for e in experiences]).to(self.device)
        rewards = torch.FloatTensor([e.reward for e in experiences]).to(self.device)
        next_states = torch.FloatTensor([e.next_state for e in experiences]).to(self.device)
        dones = torch.BoolTensor([e.done for e in experiences]).to(self.device)
        
        # 현재 Q값
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # 타겟 Q값
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (0.99 * next_q_values * ~dones)
        
        # 손실 계산
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        # 역전파
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # 엡실론 감소
        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay
        
        return loss.item()
    
    def update_target_network(self):
        """타겟 네트워크 업데이트"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def save_model(self, filepath: str):
        """모델 저장"""
        torch.save({
            'model_state_dict': self.q_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps_done': self.steps_done,
            'episode_rewards': self.episode_rewards,
            'losses': self.losses,
            'config': self.config.__dict__
        }, filepath)
    
    def load_model(self, filepath: str):
        """모델 로드"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.steps_done = checkpoint['steps_done']
        self.episode_rewards = checkpoint['episode_rewards']
        self.losses = checkpoint['losses']
        self.update_target_network()


class TradingTrainer:
    """트레이딩 에이전트 학습기"""

    def __init__(
        self,
        config: TradingConfig,
        market: str = "KRW-BTC",
        device: str = "cpu",
        data: Optional[pd.DataFrame] = None,
        db_path: Optional[str] = None,
        mode: str = "offline",
        cache_enabled: bool = True
    ):
        """
        Args:
            config: 트레이딩 설정
            market: 마켓 코드
            device: 학습 장치 (cpu/cuda)
            data: 미리 준비된 데이터 (DataFrame)
            db_path: SQLite 데이터베이스 경로
            mode: "offline" (SQLite만) | "realtime" (캐시+계산)
            cache_enabled: 캐시 사용 여부
        """
        self.config = config
        self.market = market
        self.device = device
        self.mode = mode

        # 환경 초기화 (데이터 소스 지정)
        self.env = TradingEnvironment(
            config, market,
            data=data,
            db_path=db_path,
            mode=mode,
            cache_enabled=cache_enabled
        )

        # 에이전트 초기화
        obs, _ = self.env.reset()
        self.agent = RLAgent(config, len(obs), device)

        # 로깅 설정
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # 결과 저장 디렉토리
        self.save_dir = "models/saved"
        os.makedirs(self.save_dir, exist_ok=True)

        # 시각화 저장 디렉토리
        self.viz_dir = "results/visualizations"
        os.makedirs(self.viz_dir, exist_ok=True)
    
    def train(self, num_episodes: int = 1000,
              save_frequency: int = 100,
              eval_frequency: int = 50) -> Dict[str, List[float]]:
        """학습 실행"""

        self.logger.info(f"학습 시작: {num_episodes} 에피소드")

        episode_rewards = []
        episode_losses = []
        episode_profits = []

        for episode in range(num_episodes):
            state, _ = self.env.reset()
            total_reward = 0
            total_loss = 0
            step_count = 0
            initial_value = self.env.total_value

            # 에피소드별 액션 및 리워드 추적
            episode_actions = []
            episode_step_rewards = []
            episode_prices = []
            episode_balances = []
            episode_positions = []
            episode_action_names = []

            while True:
                # 액션 마스크 가져오기
                action_mask = self.env.get_action_mask()

                # 액션 선택
                action = self.agent.select_action(state, action_mask, training=True)

                # 환경 스텝
                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated

                # 액션 및 리워드 추적 (info에서 실제 실행된 액션 이름 가져오기)
                episode_actions.append(action)
                episode_action_names.append(info['action'])  # 실제 실행된 액션 이름
                episode_step_rewards.append(reward)
                episode_prices.append(info['current_price'])
                episode_balances.append(info['balance'])
                episode_positions.append(info['position'])

                # 경험 저장
                self.agent.store_experience(state, action, reward, next_state, done, action_mask)

                # 학습
                loss = self.agent.train_step()
                total_loss += loss
                total_reward += reward
                step_count += 1

                state = next_state

                if done:
                    break
            
            # 타겟 네트워크 업데이트
            if episode % self.config.target_update == 0:
                self.agent.update_target_network()

            # 에피소드 결과 기록
            final_value = self.env.total_value
            profit_rate = (final_value - initial_value) / initial_value * 100

            episode_rewards.append(total_reward)
            episode_losses.append(total_loss / max(step_count, 1))
            episode_profits.append(profit_rate)

            # 에피소드 시각화 (주기적으로 저장)
            if episode % save_frequency == 0 and episode > 0:
                viz_path = os.path.join(self.viz_dir, f"episode_{episode}.png")
                self._plot_episode_actions(
                    episode_actions, episode_action_names, episode_step_rewards,
                    episode_prices, episode_balances, episode_positions, episode, viz_path
                )

            # 로깅
            if episode % 10 == 0:
                avg_reward = np.mean(episode_rewards[-10:])
                avg_profit = np.mean(episode_profits[-10:])
                self.logger.info(
                    f"Episode {episode}: "
                    f"Avg Reward={avg_reward:.4f}, "
                    f"Avg Profit={avg_profit:.2f}%, "
                    f"Epsilon={self.agent.epsilon:.3f}, "
                    f"Steps={step_count}"
                )
            
            # 모델 저장
            if episode % save_frequency == 0 and episode > 0:
                model_path = os.path.join(self.save_dir, f"model_episode_{episode}.pth")
                self.agent.save_model(model_path)
                self.logger.info(f"모델 저장: {model_path}")
            
            # 평가 실행
            if episode % eval_frequency == 0 and episode > 0:
                eval_result = self.evaluate(num_episodes=5)
                self.logger.info(f"평가 결과: {eval_result}")
        
        # 최종 모델 저장
        final_model_path = os.path.join(self.save_dir, "final_model.pth")
        self.agent.save_model(final_model_path)
        
        # 학습 결과 저장
        results = {
            'episode_rewards': episode_rewards,
            'episode_losses': episode_losses,
            'episode_profits': episode_profits
        }
        
        results_path = os.path.join(self.save_dir, "training_results.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        return results
    
    def _plot_episode_actions(self, actions: List[int], action_names: List[str],
                              rewards: List[float], prices: List[float],
                              balances: List[float], positions: List[float],
                              episode: int, save_path: str):
        """에피소드 액션 및 리워드 시각화"""
        fig, axes = plt.subplots(4, 1, figsize=(15, 12))

        steps = list(range(len(actions)))

        # 1. 가격 및 액션 표시
        axes[0].plot(steps, prices, 'b-', alpha=0.5, linewidth=1.5, label='Price')
        axes[0].set_ylabel('Price (KRW)', color='b', fontsize=10)
        axes[0].tick_params(axis='y', labelcolor='b')

        # Buy/Sell 액션 표시 (action_names를 사용해서 실제 실행된 액션 표시)
        buy_steps = [i for i, name in enumerate(action_names) if 'BUY' in name]
        sell_steps = [i for i, name in enumerate(action_names) if 'SELL' in name]

        if buy_steps:
            axes[0].scatter([steps[i] for i in buy_steps],
                          [prices[i] for i in buy_steps],
                          c='green', marker='^', s=150, label=f'Buy ({len(buy_steps)})',
                          zorder=5, edgecolors='darkgreen', linewidth=1.5)
        if sell_steps:
            axes[0].scatter([steps[i] for i in sell_steps],
                          [prices[i] for i in sell_steps],
                          c='red', marker='v', s=150, label=f'Sell ({len(sell_steps)})',
                          zorder=5, edgecolors='darkred', linewidth=1.5)

        axes[0].set_title(f'Episode {episode} - Trading Actions', fontsize=12, fontweight='bold')
        axes[0].legend(loc='upper left', fontsize=10)
        axes[0].grid(True, alpha=0.3)

        # 2. 스텝별 리워드
        axes[1].plot(steps, rewards, 'g-', alpha=0.7, linewidth=1.5)
        axes[1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
        axes[1].set_ylabel('Reward', fontsize=10)
        axes[1].set_title('Step Rewards', fontsize=11, fontweight='bold')
        axes[1].grid(True, alpha=0.3)

        # 양수/음수 영역 색칠
        axes[1].fill_between(steps, 0, rewards, where=[r >= 0 for r in rewards],
                            color='green', alpha=0.2, interpolate=True)
        axes[1].fill_between(steps, 0, rewards, where=[r < 0 for r in rewards],
                            color='red', alpha=0.2, interpolate=True)

        # 3. 잔고 추이
        axes[2].plot(steps, balances, 'orange', alpha=0.8, linewidth=2, label='Balance')
        axes[2].fill_between(steps, 0, balances, alpha=0.2, color='orange')
        axes[2].set_ylabel('Balance (KRW)', fontsize=10)
        axes[2].set_title('Balance Over Time', fontsize=11, fontweight='bold')

        # 잔고 통계 표시
        avg_balance = np.mean(balances)
        axes[2].axhline(y=avg_balance, color='blue', linestyle='--', alpha=0.5,
                       label=f'Avg: {avg_balance:,.0f}')
        axes[2].legend(fontsize=9)
        axes[2].grid(True, alpha=0.3)
        axes[2].ticklabel_format(style='plain', axis='y')

        # 4. 포지션 추이
        axes[3].plot(steps, positions, 'purple', alpha=0.8, linewidth=2, label='Position')
        axes[3].fill_between(steps, 0, positions, alpha=0.2, color='purple')
        axes[3].set_ylabel('Position (Coin)', fontsize=10)
        axes[3].set_xlabel('Step', fontsize=10)
        axes[3].set_title('Position Over Time', fontsize=11, fontweight='bold')

        # 포지션 통계 표시
        avg_position = np.mean(positions)
        axes[3].axhline(y=avg_position, color='blue', linestyle='--', alpha=0.5,
                       label=f'Avg: {avg_position:.4f}')
        axes[3].legend(fontsize=9)
        axes[3].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()

        # 디버깅 정보 출력
        num_buys = len(buy_steps)
        num_sells = len(sell_steps)
        num_holds = len(action_names) - num_buys - num_sells

        self.logger.info(f"시각화 저장: {save_path}")
        self.logger.info(f"  액션 통계: Buy={num_buys}, Sell={num_sells}, Hold={num_holds}")
        self.logger.info(f"  잔고 범위: {min(balances):,.0f} ~ {max(balances):,.0f}")
        self.logger.info(f"  포지션 범위: {min(positions):.6f} ~ {max(positions):.6f}")

    def evaluate(self, num_episodes: int = 10) -> Dict[str, float]:
        """모델 평가"""
        eval_rewards = []
        eval_profits = []
        
        for _ in range(num_episodes):
            state, _ = self.env.reset()
            total_reward = 0
            initial_value = self.env.total_value
            
            while True:
                action_mask = self.env.get_action_mask()
                action = self.agent.select_action(state, action_mask, training=False)
                
                state, reward, terminated, truncated, _ = self.env.step(action)
                total_reward += reward
                
                if terminated or truncated:
                    break
            
            final_value = self.env.total_value
            profit_rate = (final_value - initial_value) / initial_value * 100
            
            eval_rewards.append(total_reward)
            eval_profits.append(profit_rate)
        
        return {
            'avg_reward': float(np.mean(eval_rewards)),
            'std_reward': float(np.std(eval_rewards)),
            'avg_profit': float(np.mean(eval_profits)),
            'std_profit': float(np.std(eval_profits)),
            'max_profit': float(np.max(eval_profits)),
            'min_profit': float(np.min(eval_profits))
        }
    
    def plot_training_results(self, results: Dict[str, List[float]], 
                            save_path: Optional[str] = None):
        """학습 결과 시각화"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 에피소드 보상
        axes[0, 0].plot(results['episode_rewards'])
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Total Reward')
        
        # 손실
        axes[0, 1].plot(results['episode_losses'])
        axes[0, 1].set_title('Training Loss')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Loss')
        
        # 수익률
        axes[1, 0].plot(results['episode_profits'])
        axes[1, 0].set_title('Episode Profit Rate (%)')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Profit Rate (%)')
        axes[1, 0].axhline(y=0, color='r', linestyle='--', alpha=0.5)
        
        # 이동평균 수익률
        window = 50
        if len(results['episode_profits']) >= window:
            moving_avg = np.convolve(results['episode_profits'], 
                                   np.ones(window)/window, mode='valid')
            axes[1, 1].plot(range(window-1, len(results['episode_profits'])), moving_avg)
            axes[1, 1].set_title(f'Moving Average Profit Rate ({window} episodes)')
            axes[1, 1].set_xlabel('Episode')
            axes[1, 1].set_ylabel('Profit Rate (%)')
            axes[1, 1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"그래프 저장: {save_path}")
        
        plt.show()


def main():
    """메인 실행 함수"""
    # 설정
    config = TradingConfig(
        initial_balance=1000000,  # 100만원
        lookback_window=30,       # 30분 윈도우
        model_type="dqn",
        learning_rate=1e-4,
        batch_size=32,
        epsilon_decay=0.995
    )
    
    # GPU 사용 가능 여부 확인
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"사용 장치: {device}")
    
    try:
        # 트레이너 생성
        trainer = TradingTrainer(config, market="KRW-BTC", device=device)
        
        print("=== 강화학습 트레이딩 에이전트 학습 시작 ===")
        
        # 짧은 테스트 학습
        results = trainer.train(num_episodes=50, save_frequency=25, eval_frequency=10)
        
        # 결과 시각화
        plot_path = os.path.join(trainer.save_dir, "training_plots.png")
        trainer.plot_training_results(results, plot_path)
        
        # 최종 평가
        final_eval = trainer.evaluate(num_episodes=10)
        print("\n=== 최종 평가 결과 ===")
        for key, value in final_eval.items():
            print(f"{key}: {value:.4f}")
        
        print("\n학습 완료!")
        
    except Exception as e:
        print(f"학습 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
