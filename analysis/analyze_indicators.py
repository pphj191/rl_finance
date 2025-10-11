"""
지표 분석 및 시각화 모듈
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional, Dict
import os
from scipy import stats


def plot_indicator_distribution(
    df: pd.DataFrame,
    indicators: List[str],
    save_path: Optional[str] = None
):
    """
    지표 분포 시각화

    Args:
        df: 지표가 포함된 데이터프레임
        indicators: 시각화할 지표 리스트
        save_path: 저장 경로
    """
    n_indicators = len(indicators)
    fig, axes = plt.subplots(n_indicators, 2, figsize=(15, 5*n_indicators))

    if n_indicators == 1:
        axes = axes.reshape(1, -1)

    for i, indicator in enumerate(indicators):
        if indicator not in df.columns:
            print(f"Warning: {indicator} not found in dataframe")
            continue

        data = df[indicator].dropna()

        # 히스토그램
        axes[i, 0].hist(data, bins=50, edgecolor='black', alpha=0.7)
        axes[i, 0].set_xlabel(indicator)
        axes[i, 0].set_ylabel('Frequency')
        axes[i, 0].set_title(f'{indicator} Distribution')
        axes[i, 0].grid(True, alpha=0.3)

        # 박스플롯
        axes[i, 1].boxplot(data, vert=True)
        axes[i, 1].set_ylabel(indicator)
        axes[i, 1].set_title(f'{indicator} Box Plot')
        axes[i, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"지표 분포 그래프 저장: {save_path}")

    plt.show()


def plot_indicator_correlation(
    df: pd.DataFrame,
    indicators: List[str],
    save_path: Optional[str] = None
):
    """
    지표 간 상관관계 분석

    Args:
        df: 지표가 포함된 데이터프레임
        indicators: 분석할 지표 리스트
        save_path: 저장 경로
    """
    # 지표만 추출
    indicator_df = df[indicators].dropna()

    # 상관계수 계산
    corr_matrix = indicator_df.corr()

    # 히트맵
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, square=True, linewidths=1)
    plt.title('Indicator Correlation Matrix')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"상관관계 히트맵 저장: {save_path}")

    plt.show()

    return corr_matrix


def analyze_indicator_vs_returns(
    df: pd.DataFrame,
    indicator: str,
    price_col: str = 'close',
    forward_periods: int = 5,
    bins: int = 10,
    save_path: Optional[str] = None
) -> Dict:
    """
    지표 값과 미래 수익률 관계 분석

    Args:
        df: 데이터프레임
        indicator: 분석할 지표
        price_col: 가격 컬럼
        forward_periods: 미래 수익률 계산 기간
        bins: 지표 구간 수
        save_path: 저장 경로

    Returns:
        분석 결과 딕셔너리
    """
    # 미래 수익률 계산
    df['forward_return'] = df[price_col].shift(-forward_periods) / df[price_col] - 1
    df['forward_return'] = df['forward_return'] * 100  # 퍼센트로 변환

    # NaN 제거
    analysis_df = df[[indicator, 'forward_return']].dropna()

    # 지표 값을 구간으로 나누기
    analysis_df['indicator_bin'] = pd.qcut(
        analysis_df[indicator],
        q=bins,
        labels=[f'Q{i+1}' for i in range(bins)],
        duplicates='drop'
    )

    # 구간별 평균 수익률
    bin_returns = analysis_df.groupby('indicator_bin')['forward_return'].agg(['mean', 'std', 'count'])

    # 시각화
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 1. 구간별 평균 수익률
    bin_returns['mean'].plot(kind='bar', ax=axes[0, 0], color='steelblue', alpha=0.7)
    axes[0, 0].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    axes[0, 0].set_xlabel(f'{indicator} Quantile')
    axes[0, 0].set_ylabel(f'Avg Forward Return (%, {forward_periods}p)')
    axes[0, 0].set_title(f'{indicator} vs Forward Returns')
    axes[0, 0].grid(True, alpha=0.3)

    # 2. 산점도
    sample_size = min(1000, len(analysis_df))
    sample = analysis_df.sample(sample_size)
    axes[0, 1].scatter(sample[indicator], sample['forward_return'],
                      alpha=0.3, s=10)

    # 추세선
    z = np.polyfit(sample[indicator], sample['forward_return'], 1)
    p = np.poly1d(z)
    x_trend = np.linspace(sample[indicator].min(), sample[indicator].max(), 100)
    axes[0, 1].plot(x_trend, p(x_trend), "r--", alpha=0.8, linewidth=2)

    axes[0, 1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes[0, 1].set_xlabel(indicator)
    axes[0, 1].set_ylabel(f'Forward Return (%, {forward_periods}p)')
    axes[0, 1].set_title('Scatter Plot with Trend Line')
    axes[0, 1].grid(True, alpha=0.3)

    # 3. 구간별 승률
    analysis_df['is_profitable'] = analysis_df['forward_return'] > 0
    win_rate = analysis_df.groupby('indicator_bin')['is_profitable'].mean() * 100

    win_rate.plot(kind='bar', ax=axes[1, 0], color='green', alpha=0.7)
    axes[1, 0].axhline(y=50, color='r', linestyle='--', alpha=0.5, label='50% baseline')
    axes[1, 0].set_xlabel(f'{indicator} Quantile')
    axes[1, 0].set_ylabel('Win Rate (%)')
    axes[1, 0].set_title('Win Rate by Indicator Range')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 4. 통계 요약
    axes[1, 1].axis('off')

    # 상관계수
    correlation = analysis_df[indicator].corr(analysis_df['forward_return'])

    # t-검정 (상위 20% vs 하위 20%)
    top_20 = analysis_df.nlargest(int(len(analysis_df)*0.2), indicator)['forward_return']
    bottom_20 = analysis_df.nsmallest(int(len(analysis_df)*0.2), indicator)['forward_return']
    t_stat, p_value = stats.ttest_ind(top_20, bottom_20)

    summary_text = f"""
    Statistical Summary
    {'='*40}

    Indicator: {indicator}
    Forward Period: {forward_periods}
    Sample Size: {len(analysis_df):,}

    Correlation: {correlation:.4f}

    Top 20% Avg Return: {top_20.mean():.2f}%
    Bottom 20% Avg Return: {bottom_20.mean():.2f}%

    T-Statistic: {t_stat:.4f}
    P-Value: {p_value:.4f}

    Significance: {'YES' if p_value < 0.05 else 'NO'}
    (at 95% confidence level)
    """

    axes[1, 1].text(0.1, 0.5, summary_text, fontsize=11,
                   family='monospace', verticalalignment='center')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"지표 분석 그래프 저장: {save_path}")

    plt.show()

    return {
        'correlation': correlation,
        't_statistic': t_stat,
        'p_value': p_value,
        'significant': p_value < 0.05,
        'bin_returns': bin_returns,
        'win_rate': win_rate
    }


def plot_indicator_timeseries(
    df: pd.DataFrame,
    indicators: List[str],
    price_col: str = 'close',
    save_path: Optional[str] = None
):
    """
    지표 시계열 시각화 (가격과 함께)

    Args:
        df: 데이터프레임
        indicators: 시각화할 지표 리스트
        price_col: 가격 컬럼
        save_path: 저장 경로
    """
    n_indicators = len(indicators)
    fig, axes = plt.subplots(n_indicators + 1, 1, figsize=(15, 4*(n_indicators+1)))

    if n_indicators == 0:
        axes = [axes]
    else:
        axes = axes.flatten()

    # 가격 차트
    axes[0].plot(df.index, df[price_col], 'b-', linewidth=1.5, label='Price')
    axes[0].set_ylabel('Price')
    axes[0].set_title('Price Chart')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 지표 차트들
    for i, indicator in enumerate(indicators, 1):
        if indicator not in df.columns:
            print(f"Warning: {indicator} not found in dataframe")
            continue

        axes[i].plot(df.index, df[indicator], 'g-', linewidth=1.5, label=indicator)
        axes[i].set_ylabel(indicator)
        axes[i].set_title(f'{indicator} Over Time')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)

    axes[-1].set_xlabel('Time')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"시계열 그래프 저장: {save_path}")

    plt.show()


# 테스트 코드
if __name__ == "__main__":
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from trading_env.indicators_custom import add_custom_indicators

    print("=== 지표 분석 테스트 ===\n")

    # 샘플 데이터 생성
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=500, freq='1H')

    price = 50000
    prices = []
    for i in range(500):
        price *= (1 + np.random.randn() * 0.01)
        prices.append(price)

    df = pd.DataFrame({
        'open': prices,
        'high': [p * 1.01 for p in prices],
        'low': [p * 0.99 for p in prices],
        'close': prices,
        'volume': np.random.randint(1000, 5000, 500)
    }, index=dates)

    # 커스텀 지표 추가
    df = add_custom_indicators(df)

    # 테스트
    indicators = ['pullback_index', 'trend_consistency', 'breakout_probability']

    print("1. 지표 분포 분석")
    plot_indicator_distribution(df, indicators)

    print("\n2. 지표 상관관계 분석")
    plot_indicator_correlation(df, indicators)

    print("\n3. 지표 vs 수익률 분석 (눌림목 지수)")
    analyze_indicator_vs_returns(df, 'pullback_index', forward_periods=10)

    print("\n4. 지표 시계열")
    plot_indicator_timeseries(df, indicators[:2])
