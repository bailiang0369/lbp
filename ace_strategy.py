#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ace_strategy.py
===============
ACE (Adaptive / Asymmetric Clustering Edge) 生产级量化策略引擎 - 纯 Python 独立完整版。

完全脱离 Jupyter Notebook 与 Colab 路径依赖，提供：
1. 原生纯 Polars 技术指标计算库 (MACD, Stochastic Oscillator 等)。
2. 多尺度形态滑动窗口标准化与末端强化提取 (Min-Max Pattern & Recency Weighting)。
3. Faiss 向量聚类引擎 (支持 CPU/GPU 自适应，修复原 Notebook 中价格/指标特征覆盖的 Bug)。
4. 训练集无泄露 Edge 计算与经验贝叶斯平滑收缩 (过滤小样本噪声簇)。
5. 两种严谨的无前视（Point-in-Time）置信度阈值确定方案：
   - 方案 A: 样本内非对称分位数校准法 (In-Sample Calibration)
   - 方案 B: 因果滚动历史窗口自适应分位数法 (Rolling Causal Quantiles)
6. 完整的时序回测绩效报告输出与命令行运行接口。
"""

import sys
import gc
import os
import argparse
import numpy as np
import polars as pl
import faiss
from typing import Dict, Tuple, Optional


# ==========================================
# 1. 默认多尺度配置定义
# ==========================================
DEFAULT_HYBRID_CONFIG = {
    'v_short_10': {'window': 10, 'n_clusters': 100, 'repeat': 0},
    'short_15':   {'window': 15, 'n_clusters': 120, 'repeat': 10},
    'short_30':   {'window': 30, 'n_clusters': 150, 'repeat': 15},
    'mid_60':     {'window': 60, 'n_clusters': 300, 'repeat': 15},
    'long_120':    {'window': 120, 'n_clusters': 600, 'repeat': 15}
}


# ==========================================
# 2. 原生指标计算引擎 (替代 utils.pl_exp)
# ==========================================
def calculate_indicators(df: pl.DataFrame) -> pl.DataFrame:
    """
    使用纯 Polars 表达式高效计算 MACD 与不同周期的随机指标 (Stochastic Oscillator)。
    """
    # 确保价格列存在
    if 'close' not in df.columns:
        raise ValueError("数据中缺少 'close' 收盘价列！")

    # 若无 high/low 列，则用 close 兜底
    high_col = pl.col('high') if 'high' in df.columns else pl.col('close')
    low_col = pl.col('low') if 'low' in df.columns else pl.col('close')
    close_col = pl.col('close')

    # MACD (12, 26, 9)
    ema12 = close_col.ewm_mean(span=12)
    ema26 = close_col.ewm_mean(span=26)
    macd_line = (ema12 - ema26).alias('macd')
    macd_signal = macd_line.ewm_mean(span=9).alias('macd_signal')
    macd_hist = (macd_line - macd_signal).alias('macd_hist')

    # Stochastic Oscillator (K=29, Smooth=3) 与 (K=69, Smooth=3)
    def make_stoch_expr(k_period: int, smooth_d: int = 3, suffix: str = ""):
        lowest_low = low_col.rolling_min(window_size=k_period)
        highest_high = high_col.rolling_max(window_size=k_period)
        diff = highest_high - lowest_low
        # 避免除以零
        safe_diff = pl.when(diff == 0).then(1e-9).otherwise(diff)
        stoch_k_raw = (close_col - lowest_low) / safe_diff * 100.0
        stoch_k_smooth = stoch_k_raw.rolling_mean(window_size=smooth_d).alias(f'stoch_k_{k_period}_{smooth_d}{suffix}')
        return stoch_k_smooth

    stoch29 = make_stoch_expr(29, 3)
    stoch69 = make_stoch_expr(69, 3)

    return df.with_columns([
        macd_line,
        macd_signal,
        macd_hist,
        stoch29,
        stoch69
    ])


# ==========================================
# 3. 生产级无泄露 Faiss 多尺度聚类引擎
# ==========================================
def run_hybrid_multi_scale_clustering_leakproof(
    train_data: np.ndarray,
    test_data: np.ndarray,
    config: Dict[str, dict],
    prefix: str = 'price',
    is_indicator: bool = False,
    end_k: int = 3,
    use_gpu: bool = False
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    执行严格无泄露的多尺度形态聚类：
    1. 特征切片与局部归一化 (Min-Max 标准化消除基数，只提取形态)。
    2. 末端强化机制 (End-point emphasis) 将近期走势特征重复堆叠，放大距离权重。
    3. 仅在训练集拟合 Faiss K-Means，测试集仅执行向量检索。
    4. 分别使用独立的前缀命名（如 cluster_price_ 与 cluster_stoch_），彻底修复原 notebook 的列名覆盖 bug。
    """
    tr_ids_dict = {}
    te_ids_dict = {}

    for name, cfg in config.items():
        w = cfg['window']
        n_c = cfg['n_clusters']
        r_count = cfg['repeat']

        print(f"[{prefix.upper()}] 正在处理聚类尺度: {name} (Window={w}, Clusters={n_c}, Repeat={r_count})...")

        def extract_features(data_array: np.ndarray) -> np.ndarray:
            shape = (len(data_array) - w + 1, w)
            strides = (data_array.strides[0], data_array.strides[0])
            windows = np.lib.stride_tricks.as_strided(data_array, shape=shape, strides=strides)

            if is_indicator:
                norm = windows / 100.0
            else:
                w_min = windows.min(1, keepdims=True)
                w_max = windows.max(1, keepdims=True)
                diff = w_max - w_min
                norm = (windows - w_min) / np.where(diff == 0, 1e-9, diff)

            # 末端形态强化加权
            if r_count > 0:
                feat = np.hstack([norm] + [norm[:, -end_k:]] * r_count)
            else:
                feat = norm

            return np.ascontiguousarray(feat, dtype='float32')

        tr_feat = extract_features(train_data)
        te_feat = extract_features(test_data)

        # 严格防泄露：仅在训练集训练 K-Means
        gpu_flag = use_gpu and hasattr(faiss, 'StandardGpuResources')
        km = faiss.Kmeans(tr_feat.shape[1], n_c, niter=40, gpu=gpu_flag, verbose=False)
        km.train(tr_feat)

        _, tr_ids = km.index.search(tr_feat, 1)
        _, te_ids = km.index.search(te_feat, 1)

        col_name = f'cluster_{prefix}_{name}'
        tr_ids_dict[col_name] = np.concatenate([np.full(w - 1, -1, dtype=np.int32), tr_ids.flatten()])
        te_ids_dict[col_name] = np.concatenate([np.full(w - 1, -1, dtype=np.int32), te_ids.flatten()])

        del tr_feat, te_feat
        gc.collect()

    return tr_ids_dict, te_ids_dict


# ==========================================
# 4. 统计平滑与无前视阈值计算器
# ==========================================
class NonLookaheadThresholdEngine:
    """
    无前视阈值计算与信号生成引擎：
    - 支持经验贝叶斯收缩（解决小样本簇虚高 Edge 带来的假信号）
    - 方案 A: 样本内非对称分位数校准法 (In-Sample Calibration)
    - 方案 B: 因果滚动历史窗口自适应分位数法 (Rolling Causal Quantiles)
    """
    @staticmethod
    def compute_shrunk_edges(train_df: pl.DataFrame, cluster_cols: list, prior_weight: float = 30.0) -> pl.DataFrame:
        """
        在训练集上计算每个簇的 Edge，并施加经验贝叶斯收缩：
        shrunk_edge = (K + M * 0.5) / (N + M) - 0.5
        """
        mappings = []
        for col in cluster_cols:
            grp = train_df.filter(pl.col(col) != -1).group_by(col).agg([
                pl.len().alias('count'),
                pl.col('target').cast(pl.Int32).sum().alias('wins')
            ])
            # 贝叶斯后验收缩
            grp = grp.with_columns(
                ((pl.col('wins') + prior_weight * 0.5) / (pl.col('count') + prior_weight) - 0.5).alias(f'edge_{col}')
            ).select([col, f'edge_{col}'])
            mappings.append((col, grp))
        return mappings

    @staticmethod
    def calibrate_in_sample_quantiles(train_scores: np.ndarray, target_cov: float = 0.10) -> Tuple[float, float]:
        """
        基于训练集（纯历史）计算多头与空头的双边独立阈值（完全无前视）。
        """
        valid = train_scores[np.isfinite(train_scores)]
        half_cov = target_cov / 2.0
        long_thresh = float(np.percentile(valid, (1.0 - half_cov) * 100))
        short_thresh = float(np.percentile(valid, half_cov * 100))
        return long_thresh, short_thresh

    @staticmethod
    def generate_rolling_causal_signals(
        ace_scores: np.ndarray,
        init_buffer: np.ndarray,
        window_size: int = 1440 * 14,
        target_cov: float = 0.10
    ) -> np.ndarray:
        """
        基于过去固定回溯窗口（例如14天）的因果滚动历史，动态自适应阈值并生成信号。
        """
        n = len(ace_scores)
        signals = np.zeros(n, dtype=np.int32)
        half_cov = target_cov / 2.0

        hist_arr = np.array(init_buffer[-window_size:], dtype=np.float32)
        long_th, short_th = np.nan, np.nan

        for i in range(n):
            curr = ace_scores[i]
            # 每小时（60分钟）更新一次滚动分位数
            if (i % 60 == 0 or np.isnan(long_th)) and len(hist_arr) >= 100:
                long_th = np.percentile(hist_arr, (1.0 - half_cov) * 100)
                short_th = np.percentile(hist_arr, half_cov * 100)

            if not np.isnan(long_th):
                if curr >= long_th and curr > 0:
                    signals[i] = 1
                elif curr <= short_th and curr < 0:
                    signals[i] = -1

            if len(hist_arr) >= window_size:
                hist_arr = np.roll(hist_arr, -1)
                hist_arr[-1] = curr
            else:
                hist_arr = np.append(hist_arr, curr)

        return signals


# ==========================================
# 5. 真实时序绩效评估报告生成器
# ==========================================
def evaluate_performance(df: pl.DataFrame, signal_col: str, title: str):
    """
    对因果信号进行详细的无偏回测指标评估与多空分布统计。
    """
    print(f"\n==================== {title} ====================")
    clean = df.filter(pl.col('target').is_not_null() & pl.col(signal_col).is_not_null())
    total_len = len(clean)

    sig_active = clean.filter(pl.col(signal_col) != 0)
    sig_count = len(sig_active)

    if sig_count == 0:
        print("未触发任何信号。")
        return

    cov_rate = sig_count / total_len
    # 准确率判定
    acc = sig_active.select(
        acc = pl.when(pl.col(signal_col) == 1)
                .then(pl.col('target') == 1)
                .otherwise(pl.col('target') == 0)
                .mean()
    ).item()

    longs = sig_active.filter(pl.col(signal_col) == 1)
    shorts = sig_active.filter(pl.col(signal_col) == -1)

    long_acc = longs.select((pl.col('target') == 1).mean()).item() if len(longs) > 0 else 0.0
    short_acc = shorts.select((pl.col('target') == 0).mean()).item() if len(shorts) > 0 else 0.0

    print(f"有效评估样本数 : {total_len:,}")
    print(f"触发信号总数   : {sig_count:,} (实际覆盖率: {cov_rate:.2%})")
    print(f"日均信号频次   : {sig_count / (total_len / 1440):.1f} 次/天")
    print(f"总体预测胜率   : {acc:.2%}")
    print(f"  ├─ 做多信号  : {len(longs):,} 条 (占比 {len(longs)/sig_count:.1%}), 胜率: {long_acc:.2%}")
    print(f"  └─ 做空信号  : {len(shorts):,} 条 (占比 {len(shorts)/sig_count:.1%}), 胜率: {short_acc:.2%}")


# ==========================================
# 6. 主策略执行流 (Pipeline)
# ==========================================
def run_ace_pipeline(df: pl.DataFrame, config: dict, target_period: int = 15, split_ratio: float = 0.8):
    print(">>> 启动 ACE 生产级无前视量化策略引擎...")

    # 1. 目标标签生成: 15分钟后收盘价是否上涨
    print(f"1. 生成预测目标 (T+{target_period}m 收益方向)...")
    df = df.with_columns(
        target = (pl.col('close').shift(-target_period) > pl.col('close'))
    )

    # 2. 技术指标计算
    print("2. 计算原生技术指标 (MACD, Stochastic 29 & 69)...")
    df = calculate_indicators(df)

    # 3. 严格因果时序切分
    split_idx = int(len(df) * split_ratio)
    train_df = df.slice(0, split_idx)
    test_df = df.slice(split_idx)
    print(f"3. 时序切分完成: 训练集 {len(train_df):,} 样本, 测试集 {len(test_df):,} 样本")

    tr_prices = train_df['close'].to_numpy().astype(np.float32)
    te_prices = test_df['close'].to_numpy().astype(np.float32)

    tr_k = np.nan_to_num(train_df['stoch_k_29_3'].to_numpy(), nan=50.0).astype(np.float32)
    te_k = np.nan_to_num(test_df['stoch_k_29_3'].to_numpy(), nan=50.0).astype(np.float32)

    # 4. 执行无泄露 Faiss 聚类 (修复键名覆盖 bug)
    print("4. 执行价格序列与随机指标序列的多尺度聚类表征...")
    tr_p_ids, te_p_ids = run_hybrid_multi_scale_clustering_leakproof(tr_prices, te_prices, config, prefix='price')
    tr_s_ids, te_s_ids = run_hybrid_multi_scale_clustering_leakproof(tr_k, te_k, config, prefix='stoch', is_indicator=True)

    # 合并特征列
    train_df = train_df.with_columns([pl.Series(k, v) for k, v in {**tr_p_ids, **tr_s_ids}.items()])
    test_df = test_df.with_columns([pl.Series(k, v) for k, v in {**te_p_ids, **te_s_ids}.items()])

    # 5. 计算贝叶斯平滑 Edge 并映射
    print("5. 仅在训练集上统计簇 Edge (采用贝叶斯后验收缩)...")
    cluster_cols = [c for c in train_df.columns if c.startswith('cluster_')]
    mappings = NonLookaheadThresholdEngine.compute_shrunk_edges(train_df, cluster_cols, prior_weight=30.0)

    for col, mapping_df in mappings:
        train_df = train_df.join(mapping_df, on=col, how='left')
        test_df = test_df.join(mapping_df, on=col, how='left')

    # 6. 计算多尺度共振 ACE 得分
    edge_cols = [f'edge_{c}' for c in cluster_cols]
    train_df = train_df.with_columns(ace_score = pl.sum_horizontal([pl.col(c).fill_null(0) for c in edge_cols]))
    test_df = test_df.with_columns(ace_score = pl.sum_horizontal([pl.col(c).fill_null(0) for c in edge_cols]))

    # 7. 方案 A: 纯样本内非对称分位数阈值确定
    print("6. [方案 A] 纯样本内分位数校准 (无前视静态阈值)...")
    tr_scores = train_df['ace_score'].to_numpy()
    long_th, short_th = NonLookaheadThresholdEngine.calibrate_in_sample_quantiles(tr_scores, target_cov=0.10)
    print(f"   -> 训练集校准得出: 做多阈值 >= {long_th:.4f}, 做空阈值 <= {short_th:.4f}")

    test_df = test_df.with_columns(
        sig_insample = pl.when(pl.col('ace_score') >= long_th).then(1)
                        .when(pl.col('ace_score') <= short_th).then(-1)
                        .otherwise(0)
    )
    evaluate_performance(test_df, 'sig_insample', '方案 A: 样本内非对称分位数校准法 (完全无前视)')

    # 8. 方案 B: 因果滚动历史窗口自适应分位数法
    print("7. [方案 B] 因果滚动历史分位数自适应法 (流式逐行无前视)...")
    te_scores = test_df['ace_score'].to_numpy()
    sig_rolling = NonLookaheadThresholdEngine.generate_rolling_causal_signals(
        te_scores,
        init_buffer=tr_scores,
        window_size=1440 * 14,
        target_cov=0.10
    )
    test_df = test_df.with_columns(sig_rolling = pl.Series('sig_rolling', sig_rolling))
    evaluate_performance(test_df, 'sig_rolling', '方案 B: 因果滚动历史窗口自适应法 (动态 Point-in-Time)')

    print("\n>>> 策略执行完毕！")
    return test_df


# ==========================================
# 7. CLI 入口与自包含演示
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="ACE 生产级量化策略引擎 - 纯 Python 版")
    parser.add_argument("--data_path", type=str, default="", help="真实 K 线数据文件路径 (CSV 或 Parquet)")
    parser.add_argument("--rows", type=int, default=10000, help="演示运行的行数 (用于无文件时的基准演示)")
    args = parser.parse_args()

    if args.data_path and os.path.exists(args.data_path):
        print(f"正在从本地文件加载真实行情数据: {args.data_path}")
        if args.data_path.endswith('.parquet'):
            df = pl.read_parquet(args.data_path)
        else:
            df = pl.read_csv(args.data_path)
    else:
        print("未指定真实数据路径或文件不存在，正在生成高仿真几何布朗运动 + 波动聚集时序以供验证...")
        np.random.seed(42)
        n = args.rows
        # 构造具厚尾特征的仿真价格时序
        rets = np.random.standard_t(df=5, size=n) * 0.002
        prices = 2000.0 * np.exp(np.cumsum(rets))
        highs = prices * (1.0 + np.abs(np.random.randn(n)) * 0.001)
        lows = prices * (1.0 - np.abs(np.random.randn(n)) * 0.001)

        df = pl.DataFrame({
            'ts': np.arange(n),
            'open': prices,
            'high': highs,
            'low': lows,
            'close': prices
        })

    # 运行策略流水线 (针对演示调整快速聚类配置)
    quick_config = {
        'v_short_10': {'window': 10, 'n_clusters': 20, 'repeat': 0},
        'short_15':   {'window': 15, 'n_clusters': 30, 'repeat': 5},
        'short_30':   {'window': 30, 'n_clusters': 40, 'repeat': 5}
    } if len(df) < 50000 else DEFAULT_HYBRID_CONFIG

    run_ace_pipeline(df, quick_config)


if __name__ == '__main__':
    main()
