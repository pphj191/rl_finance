"""
데이터 파이프라인 테스트 스크립트

1. Upbit에서 데이터 수집
2. SQLite에 원본 데이터 저장
3. 기술적 지표 계산 (indicators_basic, indicators_custom, indicators_ssl)
4. 계산된 지표를 SQLite에 저장
5. 저장된 데이터 로드 및 검증

사용법:
    python -m trading_env.test_data_pipeline
"""

import os
import sys
import logging
import hashlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.font_manager as fm

# 한글 폰트 설정
plt.rcParams['font.family'] = 'AppleGothic' if sys.platform == 'darwin' else 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지

# 프로젝트 루트를 sys.path에 추가
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from trading_env.data_storage import MarketDataStorage
from trading_env.indicators_basic import FeatureExtractor
from trading_env.indicators_custom import CustomIndicators, add_custom_indicators
from trading_env.indicators_ssl import SSLFeatureExtractor, SSLConfig
from trading_env.market_data import UpbitDataCollector


def main():
    """데이터 파이프라인 테스트 메인 함수"""

    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    # 설정
    MARKET = "KRW-BTC"
    DATA_COUNT = 500  # 500개 캔들 (약 8시간, 1분봉 기준)
    DB_PATH = "data/market_data.db"
    VIZ_DIR = "results/data_pipeline_viz"

    # 디렉토리 생성
    os.makedirs("data", exist_ok=True)
    os.makedirs(VIZ_DIR, exist_ok=True)

    logger.info("=" * 70)
    logger.info("데이터 파이프라인 테스트 시작")
    logger.info("=" * 70)

    # ========================================================================
    # 1단계: Upbit에서 데이터 수집
    # ========================================================================
    logger.info(f"\n[1/5] Upbit API에서 {MARKET} 데이터 수집 중...")
    logger.info(f"      수집할 캔들 개수: {DATA_COUNT}개 (1분봉)")

    try:
        collector = UpbitDataCollector(market=MARKET)
        raw_data = collector.get_historical_data(count=DATA_COUNT, unit=1)

        if raw_data is None or raw_data.empty:
            raise ValueError("데이터 수집 실패")

        logger.info(f"✅ 수집 완료: {len(raw_data)}개 레코드")
        logger.info(f"   데이터 범위: {raw_data.index[0]} ~ {raw_data.index[-1]}")
        logger.info(f"   컬럼: {list(raw_data.columns)}")
        logger.info(f"\n   샘플 데이터 (최근 3개):")
        logger.info(f"\n{raw_data.head(3).to_string()}")

        # 시각화 1: 원본 OHLCV 데이터
        logger.info(f"\n📊 원본 데이터 시각화 중...")
        fig, axes = plt.subplots(2, 1, figsize=(14, 8))

        # 캔들스틱 차트 (간단한 버전)
        ax1 = axes[0]
        ax1.plot(raw_data.index, raw_data['close'], label='Close', color='blue', linewidth=1)
        ax1.fill_between(raw_data.index, raw_data['low'], raw_data['high'], alpha=0.2, color='gray', label='High-Low Range')
        ax1.set_title(f'{MARKET} 가격 차트 (최근 {DATA_COUNT}개 캔들)', fontsize=14, fontweight='bold')
        ax1.set_ylabel('가격 (KRW)', fontsize=11)
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)

        # 거래량 차트
        ax2 = axes[1]
        colors = ['red' if raw_data['close'].iloc[i] < raw_data['open'].iloc[i] else 'green'
                 for i in range(len(raw_data))]
        ax2.bar(raw_data.index, raw_data['volume'], color=colors, alpha=0.6, width=0.0007)
        ax2.set_title('거래량', fontsize=12, fontweight='bold')
        ax2.set_ylabel('거래량', fontsize=11)
        ax2.set_xlabel('시간', fontsize=11)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        viz_path_1 = f"{VIZ_DIR}/01_raw_data.png"
        plt.savefig(viz_path_1, dpi=100, bbox_inches='tight')
        plt.close()
        logger.info(f"   ✅ 저장: {viz_path_1}")

    except Exception as e:
        logger.error(f"❌ 데이터 수집 실패: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise

    # ========================================================================
    # 2단계: SQLite에 원본 OHLCV 데이터 저장
    # ========================================================================
    logger.info(f"\n[2/5] SQLite에 원본 데이터 저장 중... ({DB_PATH})")

    try:
        storage = MarketDataStorage(db_path=DB_PATH)

        # 기존 데이터 확인
        existing_data = storage.load_ohlcv_data(market=MARKET)

        if existing_data is not None and not existing_data.empty:
            logger.info(f"   ⚠️  기존 데이터 발견: {len(existing_data)}개 레코드")
            logger.info(f"   기존 데이터를 모두 삭제하고 새로 저장합니다...")

            # TODO: 현재는 기존 데이터를 삭제하고 새로 저장하는 방식을 사용하고 있습니다.
            # 향후 다음과 같이 개선이 필요합니다:
            # 1. INSERT OR REPLACE 방식으로 중복 데이터는 업데이트
            # 2. 새로운 타임스탬프만 추가 (중복 제거)
            # 3. data_storage.py의 save_ohlcv_data()에 if_exists='replace' 옵션 추가
            # 4. 또는 UPSERT (INSERT ... ON CONFLICT DO UPDATE) SQL 사용
            # 현재 방식은 테스트 목적으로는 적합하지만, 프로덕션에서는 비효율적입니다.

            # 기존 데이터 삭제 (테스트 목적)
            import sqlite3
            with sqlite3.connect(DB_PATH) as conn:
                conn.execute("DELETE FROM ohlcv_data WHERE market = ?", (MARKET,))
                conn.execute("DELETE FROM processed_data WHERE market = ?", (MARKET,))
                conn.commit()

            logger.info(f"   ✅ 기존 데이터 삭제 완료")

        # 새 데이터 저장
        storage.save_ohlcv_data(market=MARKET, data=raw_data)
        logger.info(f"✅ 원본 데이터 저장 완료 ({len(raw_data)}개)")

    except Exception as e:
        logger.error(f"❌ 데이터 저장 실패: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise

    # ========================================================================
    # 3단계: 기술적 지표 계산 (basic + custom + ssl)
    # ========================================================================
    logger.info(f"\n[3/5] 기술적 지표 계산 중...")

    try:
        # 3-1. 기본 지표 계산 (indicators_basic.py)
        logger.info(f"\n   [3-1] indicators_basic 계산 중...")
        logger.info(f"         - SMA, EMA, RSI, MACD, Bollinger Bands, Stochastic, ATR, ADX, OBV")

        feature_extractor = FeatureExtractor()
        data_with_basic = feature_extractor.extract_technical_indicators(raw_data)

        logger.info(f"   ✅ 기본 지표 계산 완료: {len(data_with_basic.columns)}개 컬럼")
        logger.info(f"      추가된 컬럼: {len(data_with_basic.columns) - len(raw_data.columns)}개")

        # 3-2. 커스텀 지표 추가 (indicators_custom.py)
        logger.info(f"\n   [3-2] indicators_custom 계산 중...")
        logger.info(f"         - 눌림목 지수, 지지/저항 강도, 추세 일관성, 변동성 돌파 확률")

        data_with_custom = add_custom_indicators(data_with_basic)

        logger.info(f"   ✅ 커스텀 지표 추가 완료: {len(data_with_custom.columns)}개 컬럼")
        logger.info(f"      추가된 지표: pullback_index, support_strength, resistance_strength, "
                   f"trend_consistency, breakout_probability")

        # 3-3. SSL 특성 추가 (indicators_ssl.py) - 선택 사항
        INCLUDE_SSL = False  # SSL은 계산 시간이 오래 걸리므로 기본적으로 비활성화

        if INCLUDE_SSL:
            logger.info(f"\n   [3-3] indicators_ssl 계산 중...")
            logger.info(f"         - Self-Supervised Learning 기반 특성 추출")

            try:
                ssl_config = SSLConfig(
                    hidden_dim=64,  # 테스트용으로 작게 설정
                    num_epochs=10   # 테스트용으로 적게 설정
                )
                ssl_extractor = SSLFeatureExtractor(config=ssl_config)

                # SSL 특성 추출 (학습 필요)
                data_with_all = ssl_extractor.fit_transform(data_with_custom)

                logger.info(f"   ✅ SSL 특성 추가 완료: {len(data_with_all.columns)}개 컬럼")
            except Exception as e:
                logger.warning(f"   ⚠️  SSL 특성 계산 실패 (스킵): {e}")
                data_with_all = data_with_custom
        else:
            logger.info(f"\n   [3-3] indicators_ssl 계산 스킵 (INCLUDE_SSL=False)")
            logger.info(f"         SSL 특성을 포함하려면 INCLUDE_SSL=True로 설정하세요.")
            data_with_all = data_with_custom

        # 데이터 샘플 출력
        logger.info(f"\n   계산된 지표 샘플 (최근 5개):")
        sample_cols = ['close', 'sma_20', 'rsi', 'macd', 'pullback_index',
                      'support_strength', 'trend_consistency']
        available_cols = [col for col in sample_cols if col in data_with_all.columns]
        logger.info(f"\n{data_with_all[available_cols].tail(5).to_string()}")

        # 시각화 2: 기본 지표들
        logger.info(f"\n📊 기본 지표 시각화 중...")
        fig = plt.figure(figsize=(16, 12))
        gs = gridspec.GridSpec(4, 2, figure=fig)

        # 1. 가격 + 이동평균
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(data_with_all.index, data_with_all['close'], label='Close', color='black', linewidth=1.5)
        if 'sma_5' in data_with_all.columns:
            ax1.plot(data_with_all.index, data_with_all['sma_5'], label='SMA 5', color='blue', alpha=0.7)
        if 'sma_20' in data_with_all.columns:
            ax1.plot(data_with_all.index, data_with_all['sma_20'], label='SMA 20', color='orange', alpha=0.7)
        if 'sma_60' in data_with_all.columns:
            ax1.plot(data_with_all.index, data_with_all['sma_60'], label='SMA 60', color='red', alpha=0.7)
        ax1.set_title('가격 + 이동평균', fontsize=12, fontweight='bold')
        ax1.set_ylabel('가격 (KRW)', fontsize=10)
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)

        # 2. 볼린저 밴드
        ax2 = fig.add_subplot(gs[1, :])
        ax2.plot(data_with_all.index, data_with_all['close'], label='Close', color='black', linewidth=1.5)
        if 'bb_upper' in data_with_all.columns:
            ax2.plot(data_with_all.index, data_with_all['bb_upper'], label='BB Upper', color='red', alpha=0.5, linestyle='--')
            ax2.plot(data_with_all.index, data_with_all['bb_middle'], label='BB Middle', color='blue', alpha=0.5)
            ax2.plot(data_with_all.index, data_with_all['bb_lower'], label='BB Lower', color='green', alpha=0.5, linestyle='--')
            ax2.fill_between(data_with_all.index, data_with_all['bb_lower'], data_with_all['bb_upper'],
                            alpha=0.1, color='gray')
        ax2.set_title('볼린저 밴드', fontsize=12, fontweight='bold')
        ax2.set_ylabel('가격 (KRW)', fontsize=10)
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)

        # 3. RSI
        ax3 = fig.add_subplot(gs[2, 0])
        if 'rsi' in data_with_all.columns:
            ax3.plot(data_with_all.index, data_with_all['rsi'], label='RSI', color='purple', linewidth=1.5)
            ax3.axhline(y=70, color='red', linestyle='--', alpha=0.5, label='과매수(70)')
            ax3.axhline(y=30, color='green', linestyle='--', alpha=0.5, label='과매도(30)')
            ax3.fill_between(data_with_all.index, 30, 70, alpha=0.1, color='gray')
        ax3.set_title('RSI (Relative Strength Index)', fontsize=12, fontweight='bold')
        ax3.set_ylabel('RSI', fontsize=10)
        ax3.set_ylim(0, 100)
        ax3.legend(loc='best')
        ax3.grid(True, alpha=0.3)

        # 4. MACD
        ax4 = fig.add_subplot(gs[2, 1])
        if 'macd' in data_with_all.columns:
            ax4.plot(data_with_all.index, data_with_all['macd'], label='MACD', color='blue', linewidth=1.5)
            ax4.plot(data_with_all.index, data_with_all['macd_signal'], label='Signal', color='red', linewidth=1.5)
            if 'macd_histogram' in data_with_all.columns:
                colors = ['green' if val > 0 else 'red' for val in data_with_all['macd_histogram']]
                ax4.bar(data_with_all.index, data_with_all['macd_histogram'], color=colors, alpha=0.3, label='Histogram')
            ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax4.set_title('MACD', fontsize=12, fontweight='bold')
        ax4.set_ylabel('MACD', fontsize=10)
        ax4.legend(loc='best')
        ax4.grid(True, alpha=0.3)

        # 5. 눌림목 지수
        ax5 = fig.add_subplot(gs[3, 0])
        if 'pullback_index' in data_with_all.columns:
            ax5.plot(data_with_all.index, data_with_all['pullback_index'], label='Pullback Index',
                    color='orange', linewidth=1.5)
            ax5.axhline(y=70, color='red', linestyle='--', alpha=0.5, label='매수 신호(70)')
            ax5.fill_between(data_with_all.index, 70, 100, alpha=0.1, color='green')
        ax5.set_title('눌림목 지수', fontsize=12, fontweight='bold')
        ax5.set_ylabel('Index', fontsize=10)
        ax5.set_ylim(0, 100)
        ax5.legend(loc='best')
        ax5.grid(True, alpha=0.3)

        # 6. 추세 일관성
        ax6 = fig.add_subplot(gs[3, 1])
        if 'trend_consistency' in data_with_all.columns:
            ax6.plot(data_with_all.index, data_with_all['trend_consistency'], label='Trend Consistency',
                    color='teal', linewidth=1.5)
            ax6.axhline(y=0.7, color='green', linestyle='--', alpha=0.5, label='강한 추세(0.7)')
            ax6.axhline(y=-0.7, color='red', linestyle='--', alpha=0.5, label='강한 하락(-0.7)')
            ax6.fill_between(data_with_all.index, -1, 1, alpha=0.05, color='gray')
        ax6.set_title('추세 일관성', fontsize=12, fontweight='bold')
        ax6.set_ylabel('Consistency', fontsize=10)
        ax6.set_ylim(-1, 1)
        ax6.legend(loc='best')
        ax6.grid(True, alpha=0.3)

        plt.tight_layout()
        viz_path_2 = f"{VIZ_DIR}/02_indicators.png"
        plt.savefig(viz_path_2, dpi=100, bbox_inches='tight')
        plt.close()
        logger.info(f"   ✅ 저장: {viz_path_2}")

    except Exception as e:
        logger.error(f"❌ 지표 계산 실패: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise

    # ========================================================================
    # 4단계: 계산된 지표를 SQLite에 저장
    # ========================================================================
    logger.info(f"\n[4/5] 계산된 지표를 SQLite에 저장 중...")

    try:
        # 특성 벡터 추출
        feature_vector, feature_names = feature_extractor.get_feature_vector(data_with_all)

        # 설정 해시 생성
        config_hash = hashlib.md5("robust_False".encode()).hexdigest()

        # SQLite에 저장
        storage.save_processed_data(
            market=MARKET,
            data=data_with_all,
            feature_vector=feature_vector,
            feature_names=feature_names,
            normalization_method="robust",
            normalization_params={"method": "robust", "include_ssl": False},
            config_hash=config_hash
        )

        logger.info(f"✅ 계산된 지표 저장 완료")
        logger.info(f"   특성 벡터 차원: {feature_vector.shape}")
        logger.info(f"   특성 이름 개수: {len(feature_names)}")

    except Exception as e:
        logger.error(f"❌ 지표 저장 실패: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise

    # ========================================================================
    # 5단계: SQLite에서 데이터 로드 및 검증
    # ========================================================================
    logger.info(f"\n[5/5] SQLite에서 데이터 로드 및 검증 중...")

    try:
        # 원본 데이터 로드
        loaded_raw = storage.load_ohlcv_data(market=MARKET)
        logger.info(f"✅ 원본 데이터 로드: {len(loaded_raw)}개 레코드")

        # 처리된 데이터 로드
        loaded_processed = storage.load_processed_data(
            market=MARKET,
            config_hash=config_hash
        )
        logger.info(f"✅ 처리된 데이터 로드: {len(loaded_processed)}개 레코드, "
                   f"{len(loaded_processed.columns)}개 컬럼")

        # 데이터 무결성 검증
        assert len(loaded_raw) == len(raw_data), "원본 데이터 개수 불일치"
        assert len(loaded_processed) == len(data_with_all), "처리된 데이터 개수 불일치"

        logger.info(f"✅ 데이터 무결성 검증 완료")

    except Exception as e:
        logger.error(f"❌ 데이터 로드/검증 실패: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise

    # ========================================================================
    # 최종 결과 요약
    # ========================================================================
    logger.info("=" * 70)
    logger.info("데이터 파이프라인 테스트 완료!")
    logger.info("=" * 70)
    logger.info(f"✅ 시장: {MARKET}")
    logger.info(f"✅ 데이터베이스: {DB_PATH}")
    logger.info(f"✅ 원본 데이터: {len(loaded_raw)}개 레코드")
    logger.info(f"✅ 처리된 데이터: {len(loaded_processed)}개 레코드, {len(loaded_processed.columns)}개 컬럼")
    logger.info(f"✅ 데이터 범위: {loaded_processed.index[0]} ~ {loaded_processed.index[-1]}")

    logger.info(f"\n📊 데이터 통계:")
    logger.info(f"  - 가격 범위: {raw_data['close'].min():,.0f} ~ {raw_data['close'].max():,.0f} KRW")
    logger.info(f"  - 평균 거래량: {raw_data['volume'].mean():,.2f}")
    logger.info(f"  - 총 데이터 크기: {len(loaded_processed) * len(loaded_processed.columns):,} 데이터 포인트")

    logger.info(f"\n📈 계산된 지표:")
    logger.info(f"  - 기본 지표: SMA, EMA, RSI, MACD, Bollinger Bands, Stochastic, ATR, ADX, OBV")
    logger.info(f"  - 커스텀 지표: pullback_index, support_strength, resistance_strength, "
               f"trend_consistency, breakout_probability")

    # 시각화 3: 지표 상관관계 히트맵
    logger.info(f"\n📊 [시각화 3/4] 지표 상관관계 분석 및 시각화 중...")
    try:
        import numpy as np

        # 주요 지표만 선택
        corr_cols = ['close', 'volume', 'rsi', 'macd', 'bb_width', 'pullback_index',
                    'support_strength', 'resistance_strength', 'trend_consistency']
        available_corr_cols = [col for col in corr_cols if col in loaded_processed.columns]

        logger.info(f"   사용 가능한 상관관계 지표: {available_corr_cols}")

        if len(available_corr_cols) > 2:
            logger.info(f"   상관관계 계산 중... ({len(available_corr_cols)}개 지표)")
            corr_data = loaded_processed[available_corr_cols].corr()

            fig, ax = plt.subplots(figsize=(12, 10))
            im = ax.imshow(corr_data, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)

            # 축 설정
            ax.set_xticks(range(len(available_corr_cols)))
            ax.set_yticks(range(len(available_corr_cols)))
            ax.set_xticklabels(available_corr_cols, rotation=45, ha='right')
            ax.set_yticklabels(available_corr_cols)

            # 값 표시
            for i in range(len(available_corr_cols)):
                for j in range(len(available_corr_cols)):
                    text = ax.text(j, i, f'{corr_data.iloc[i, j]:.2f}',
                                 ha="center", va="center", color="black", fontsize=9)

            ax.set_title('지표 간 상관관계 히트맵', fontsize=14, fontweight='bold', pad=20)
            plt.colorbar(im, ax=ax, label='상관계수')
            plt.tight_layout()

            viz_path_3 = f"{VIZ_DIR}/03_correlation_heatmap.png"
            logger.info(f"   히트맵 저장 중: {viz_path_3}")
            plt.savefig(viz_path_3, dpi=100, bbox_inches='tight')
            plt.close()
            logger.info(f"   ✅ 저장 완료: {viz_path_3}")
        else:
            logger.warning(f"   ⚠️  상관관계 계산 스킵: 지표가 충분하지 않음 ({len(available_corr_cols)}개)")
    except Exception as e:
        logger.error(f"   ❌ 상관관계 시각화 실패: {e}")
        import traceback
        logger.error(traceback.format_exc())

    # 시각화 4: 지표 분포 히스토그램
    logger.info(f"\n📊 [시각화 4/4] 지표 분포 히스토그램 생성 중...")
    try:
        hist_cols = ['rsi', 'pullback_index', 'support_strength', 'trend_consistency']
        available_hist_cols = [col for col in hist_cols if col in loaded_processed.columns]

        logger.info(f"   사용 가능한 분포 지표: {available_hist_cols}")

        if len(available_hist_cols) > 0:
            logger.info(f"   히스토그램 생성 중... ({len(available_hist_cols)}개 지표)")
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            axes = axes.flatten()

            for idx, col in enumerate(available_hist_cols[:4]):
                ax = axes[idx]
                data = loaded_processed[col].dropna()
                ax.hist(data, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
                ax.axvline(data.mean(), color='red', linestyle='--', linewidth=2, label=f'평균: {data.mean():.2f}')
                ax.axvline(data.median(), color='green', linestyle='--', linewidth=2, label=f'중앙값: {data.median():.2f}')
                ax.set_title(f'{col} 분포', fontsize=12, fontweight='bold')
                ax.set_xlabel('값', fontsize=10)
                ax.set_ylabel('빈도', fontsize=10)
                ax.legend()
                ax.grid(True, alpha=0.3)

            # 남은 서브플롯 숨기기
            for idx in range(len(available_hist_cols), 4):
                axes[idx].axis('off')

            plt.tight_layout()
            viz_path_4 = f"{VIZ_DIR}/04_indicator_distributions.png"
            logger.info(f"   히스토그램 저장 중: {viz_path_4}")
            plt.savefig(viz_path_4, dpi=100, bbox_inches='tight')
            plt.close()
            logger.info(f"   ✅ 저장 완료: {viz_path_4}")
        else:
            logger.warning(f"   ⚠️  히스토그램 생성 스킵: 지표가 없음")
    except Exception as e:
        logger.error(f"   ❌ 분포 히스토그램 생성 실패: {e}")
        import traceback
        logger.error(traceback.format_exc())

    logger.info("\n🎯 다음 단계:")
    logger.info(f"  1. 개별 지표 테스트: python -m trading_env.indicators_basic")
    logger.info(f"  2. 커스텀 지표 테스트: python -m trading_env.indicators_custom")
    logger.info(f"  3. SSL 특성 테스트: python -m trading_env.indicators_ssl")
    logger.info(f"  4. RL 학습 시작: python run_train.py --db {DB_PATH}")

    logger.info(f"\n📁 생성된 시각화 파일:")
    # 실제로 생성된 파일만 확인하여 표시
    viz_files = [
        (f"{VIZ_DIR}/01_raw_data.png", "원본 OHLCV 데이터"),
        (f"{VIZ_DIR}/02_indicators.png", "기술적 지표"),
        (f"{VIZ_DIR}/03_correlation_heatmap.png", "상관관계"),
        (f"{VIZ_DIR}/04_indicator_distributions.png", "지표 분포")
    ]

    for file_path, description in viz_files:
        if os.path.exists(file_path):
            logger.info(f"  ✅ {file_path} ({description})")
        else:
            logger.info(f"  ❌ {file_path} ({description}) - 생성되지 않음")

    logger.info("=" * 70)


if __name__ == "__main__":
    main()
