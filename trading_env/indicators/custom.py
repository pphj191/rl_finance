"""
커스텀 기술적 지표 모듈

개인적으로 개발한 트레이딩 지표들을 구현합니다.
- 눌림목 지수
- 기타 커스텀 지표들
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple


class CustomIndicators:
    """커스텀 기술적 지표 계산 클래스"""

    @staticmethod
    def pullback_index(
        df: pd.DataFrame,
        price_col: str = 'close',
        high_col: str = 'high',
        low_col: str = 'low',
        lookback: int = 20,
        pullback_threshold: float = 0.02
    ) -> pd.Series:
        """
        눌림목 지수 계산

        눌림목: 상승 추세에서 일시적으로 하락했다가 다시 상승하는 패턴

        지수 계산 방식:
        1. 최근 고점 대비 하락폭 계산
        2. 하락 후 반등 강도 측정
        3. 거래량 변화 고려
        4. 0~100 사이로 정규화 (높을수록 눌림목 가능성 높음)

        Args:
            df: OHLCV 데이터프레임
            price_col: 종가 컬럼명
            high_col: 고가 컬럼명
            low_col: 저가 컬럼명
            lookback: 눌림목 판단 기간
            pullback_threshold: 눌림목 인정 최소 하락률 (기본 2%)

        Returns:
            눌림목 지수 (0~100)
        """
        result = pd.Series(0.0, index=df.index)

        # 최근 고점
        rolling_high = df[high_col].rolling(window=lookback).max()

        # 고점 대비 현재가 하락률
        pullback_pct = (rolling_high - df[price_col]) / rolling_high * 100

        # 최근 저점
        rolling_low = df[low_col].rolling(window=lookback).min()

        # 저점 대비 반등률
        rebound_pct = (df[price_col] - rolling_low) / rolling_low * 100

        # 가격 모멘텀 (단기 이동평균 기울기)
        ma_short = df[price_col].rolling(window=5).mean()
        momentum = ma_short.pct_change(periods=3) * 100

        # 거래량 변화 (있는 경우)
        if 'volume' in df.columns:
            vol_ma = df['volume'].rolling(window=lookback).mean()
            vol_ratio = df['volume'] / vol_ma
        else:
            vol_ratio = pd.Series(1.0, index=df.index)

        # 눌림목 지수 계산
        # 1. 적정한 하락폭 (2~10%)
        pullback_score = np.where(
            (pullback_pct >= pullback_threshold) & (pullback_pct <= 10),
            (pullback_pct / 10) * 40,  # 최대 40점
            0
        )

        # 2. 반등 강도
        rebound_score = np.clip(rebound_pct * 2, 0, 30)  # 최대 30점

        # 3. 모멘텀 (양수면 가산점)
        momentum_score = np.clip(momentum * 2, -10, 20)  # -10~20점

        # 4. 거래량 (평균 이상이면 가산점)
        volume_score = np.clip((vol_ratio - 1) * 10, 0, 10)  # 최대 10점

        # 총합
        total_score = pullback_score + rebound_score + momentum_score + volume_score

        # 0~100 범위로 클리핑
        result = pd.Series(np.clip(total_score, 0, 100), index=df.index)

        return result

    @staticmethod
    def support_resistance_strength(
        df: pd.DataFrame,
        price_col: str = 'close',
        high_col: str = 'high',
        low_col: str = 'low',
        lookback: int = 50,
        tolerance: float = 0.01
    ) -> Tuple[pd.Series, pd.Series]:
        """
        지지/저항 강도 지수

        Args:
            df: OHLCV 데이터프레임
            price_col: 종가 컬럼명
            high_col: 고가 컬럼명
            low_col: 저가 컬럼명
            lookback: 분석 기간
            tolerance: 가격 오차 허용 범위 (1% 기본)

        Returns:
            (지지선 강도, 저항선 강도) 튜플
        """
        support_strength = pd.Series(0.0, index=df.index)
        resistance_strength = pd.Series(0.0, index=df.index)

        for i in range(lookback, len(df)):
            current_price = df[price_col].iloc[i]
            window = df.iloc[i-lookback:i]

            # 지지선: 현재가 근처에서 반등한 횟수
            support_touches = 0
            for j in range(len(window)-1):
                if window[low_col].iloc[j] <= current_price * (1 + tolerance):
                    if window[price_col].iloc[j+1] > window[low_col].iloc[j]:
                        support_touches += 1

            # 저항선: 현재가 근처에서 하락한 횟수
            resistance_touches = 0
            for j in range(len(window)-1):
                if window[high_col].iloc[j] >= current_price * (1 - tolerance):
                    if window[price_col].iloc[j+1] < window[high_col].iloc[j]:
                        resistance_touches += 1

            support_strength.iloc[i] = min(support_touches * 10, 100)
            resistance_strength.iloc[i] = min(resistance_touches * 10, 100)

        return support_strength, resistance_strength

    @staticmethod
    def trend_consistency(
        df: pd.DataFrame,
        price_col: str = 'close',
        window: int = 20
    ) -> pd.Series:
        """
        추세 일관성 지수

        가격이 일정한 방향으로 꾸준히 움직이는지 측정

        Args:
            df: OHLCV 데이터프레임
            price_col: 종가 컬럼명
            window: 분석 윈도우

        Returns:
            추세 일관성 지수 (-100~100, 양수: 상승 일관성, 음수: 하락 일관성)
        """
        # 가격 변화율
        returns = df[price_col].pct_change()

        # 윈도우 내 양/음 수익률 비율
        positive_ratio = returns.rolling(window=window).apply(
            lambda x: (x > 0).sum() / len(x)
        )

        # 평균 수익률 방향
        mean_return = returns.rolling(window=window).mean()

        # 일관성 = 방향성 비율 * 평균 수익률 부호
        consistency = (positive_ratio - 0.5) * 200 * np.sign(mean_return)

        return pd.Series(consistency, index=df.index)

    @staticmethod
    def volatility_breakout_probability(
        df: pd.DataFrame,
        price_col: str = 'close',
        high_col: str = 'high',
        low_col: str = 'low',
        window: int = 20,
        atr_period: int = 14
    ) -> pd.Series:
        """
        변동성 돌파 확률

        좁은 박스권에서 큰 움직임이 나올 가능성 측정

        Args:
            df: OHLCV 데이터프레임
            price_col: 종가 컬럼명
            high_col: 고가 컬럼명
            low_col: 저가 컬럼명
            window: 박스권 판단 기간
            atr_period: ATR 계산 기간

        Returns:
            돌파 확률 (0~100)
        """
        # True Range 계산
        high_low = df[high_col] - df[low_col]
        high_close = np.abs(df[high_col] - df[price_col].shift())
        low_close = np.abs(df[low_col] - df[price_col].shift())

        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(window=atr_period).mean()

        # 최근 변동성 대비 현재 변동성
        recent_volatility = df[price_col].rolling(window=window).std()
        volatility_ratio = recent_volatility / atr

        # 가격 범위 압축 정도
        range_compression = 1 / (volatility_ratio + 1e-8)

        # 확률로 변환 (압축될수록 돌파 확률 높음)
        probability = np.clip(range_compression * 50, 0, 100)

        return pd.Series(probability, index=df.index)


def add_custom_indicators(
    df: pd.DataFrame,
    pullback: bool = True,
    support_resistance: bool = True,
    trend: bool = True,
    volatility: bool = True
) -> pd.DataFrame:
    """
    데이터프레임에 커스텀 지표 일괄 추가

    Args:
        df: OHLCV 데이터프레임
        pullback: 눌림목 지수 추가 여부
        support_resistance: 지지/저항 강도 추가 여부
        trend: 추세 일관성 추가 여부
        volatility: 변동성 돌파 확률 추가 여부

    Returns:
        지표가 추가된 데이터프레임
    """
    result = df.copy()

    if pullback:
        result['pullback_index'] = CustomIndicators.pullback_index(df)

    if support_resistance:
        support, resistance = CustomIndicators.support_resistance_strength(df)
        result['support_strength'] = support
        result['resistance_strength'] = resistance

    if trend:
        result['trend_consistency'] = CustomIndicators.trend_consistency(df)

    if volatility:
        result['breakout_probability'] = CustomIndicators.volatility_breakout_probability(df)

    return result


# 테스트 코드
if __name__ == "__main__":
    # 샘플 데이터 생성
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=100, freq='1H')

    # 상승 추세 + 눌림목 패턴 시뮬레이션
    price = 50000
    prices = []
    for i in range(100):
        if 30 <= i <= 35:  # 눌림목 구간
            price *= 0.98
        else:
            price *= 1.002
        prices.append(price)

    df = pd.DataFrame({
        'timestamp': dates,
        'open': prices,
        'high': [p * 1.01 for p in prices],
        'low': [p * 0.99 for p in prices],
        'close': prices,
        'volume': np.random.randint(1000, 5000, 100)
    })

    # 커스텀 지표 추가
    df_with_indicators = add_custom_indicators(df)

    print("=== 커스텀 지표 테스트 ===")
    print("\n최근 10개 데이터:")
    print(df_with_indicators[['close', 'pullback_index', 'support_strength',
                               'trend_consistency', 'breakout_probability']].tail(10))

    print("\n눌림목 구간 (30~35) 데이터:")
    print(df_with_indicators[['close', 'pullback_index']].iloc[28:38])
