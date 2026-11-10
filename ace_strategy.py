#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ace_strategy.py
===============
ACE (Adaptive / Asymmetric Clustering Edge) 生产级量化策略引擎 - 纯 Python 独立完整版。

特性：
1. 原生纯 Polars 技术指标计算库：
   - 健壮支持现货与合约命名（high / spot_high / contract_high 等），彻底消除指标塌陷。
   - 在全量时序上先因果计算历史指标，杜绝冷启动与截断窗口缺失。
2. 严格杜绝切分点边界泄露与时序跳跃：
   - 训练集移除最后 target_period 根标签（防止未来信息渗入）；
   - 测试集在提取滑动特征时无缝衔接切分点前 w-1 根真实历史，保证时序严格连续。
3. 高效高性能因果滚动分位数计算（基于循环环形缓冲 Ring Buffer，消除 O(N) 拷贝）。
4. 两种严谨的无前视置信度阈值确定方案 (样本内分位数校准法 & 因果滚动历史分位数自适应法)。
5. 完整支持单文件、多文件通配符、目录加载，或一键自动抓取 GitHub release 全量 Binance 数据。
"""

import sys
import gc
import os
import glob
import argparse
import numpy as np
import polars as pl
import faiss
from typing import Dict, Tuple, List, Optional


DEFAULT_HYBRID_CONFIG = {
    'v_short_10': {'window': 10, 'n_clusters': 100, 'repeat': 0},
    'short_15':   {'window': 15, 'n_clusters': 120, 'repeat': 10},
    'short_30':   {'window': 30, 'n_clusters': 150, 'repeat': 15},
    'mid_60':     {'window': 60, 'n_clusters': 300, 'repeat': 15},
    'long_120':    {'window': 120, 'n_clusters': 600, 'repeat': 15}
}


# ==========================================
# 1. 原生指标计算引擎 (健壮支持现货与合约)
# ==========================================
def calculate_indicators(df: pl.DataFrame) -> pl.DataFrame:
    """
    使用纯 Polars 表达式计算 MACD、随机指标及大周期趋势线 (EMA 200)。
    健壮检测收盘价、最高价、最低价（支持现货与合约命名）。
    """
    cols = df.columns
    # 查找 close 列
    close_name = 'close'
    for c in ['close', 'spot_close', 'contract_close', 'close_spot', 'close_contract']:
        if c in cols:
            close_name = c
            break

    # 查找 high 列
    high_name = close_name
    for c in ['high', 'spot_high', 'contract_high', 'high_spot', 'high_contract']:
        if c in cols:
            high_name = c
            break

    # 查找 low 列
    low_name = close_name
    for c in ['low', 'spot_low', 'contract_low', 'low_spot', 'low_contract']:
        if c in cols:
            low_name = c
            break

    close_col = pl.col(close_name)
    high_col = pl.col(high_name)
    low_col = pl.col(low_name)

    # MACD (12, 26, 9)
    ema12 = close_col.ewm_mean(span=12)
    ema26 = close_col.ewm_mean(span=26)
    macd_line = (ema12 - ema26).alias('macd')
    macd_signal = macd_line.ewm_mean(span=9).alias('macd_signal')
    macd_hist = (macd_line - macd_signal).alias('macd_hist')

    # 大周期趋势过滤器基准 (EMA 200)
    ema200 = close_col.ewm_mean(span=200).alias('trend_ema200')

    # Stochastic Oscillator
    def make_stoch_expr(k_period: int, smooth_d: int = 3, suffix: str = ""):
        lowest_low = low_col.rolling_min(window_size=k_period)
        highest_high = high_col.rolling_max(window_size=k_period)
        diff = highest_high - lowest_low
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
        ema200,
        stoch29,
        stoch69
    ])


# ==========================================
# 2. 生产级无泄露 Faiss 多尺度聚类引擎 (无缝时序衔接)
# ==========================================
def run_hybrid_multi_scale_clustering_leakproof(
    train_data: np.ndarray,
    test_data: np.ndarray,
    pre_test_buffer: np.ndarray,
    config: Dict[str, dict],
    prefix: str = 'price',
    is_indicator: bool = False,
    end_k: int = 3,
    use_gpu: bool = False
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    执行无泄露多尺度聚类：
    1. pre_test_buffer 为紧邻测试集起点的最后 max_w 个真实历史点，确保测试集滑动窗口无 15 分钟时序断裂。
    2. 严格仅在训练集特征上拟合 Faiss K-Means，测试集仅执行向量映射。
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

            if r_count > 0:
                feat = np.hstack([norm] + [norm[:, -end_k:]] * r_count)
            else:
                feat = norm

            return np.ascontiguousarray(feat, dtype='float32')

        # 训练集特征：前 w-1 个点用 -1 填充以与训练行对齐
        tr_feat = extract_features(train_data)

        # 测试集特征：前置衔接切分点前紧邻的真实 w-1 个点，输出长度严格等于 len(test_data)
        te_buffered = np.concatenate([pre_test_buffer[-(w - 1):], test_data])
        te_feat = extract_features(te_buffered)

        gpu_flag = use_gpu and hasattr(faiss, 'StandardGpuResources')
        km = faiss.Kmeans(tr_feat.shape[1], n_c, niter=30, gpu=gpu_flag, verbose=False)
        km.train(tr_feat)

        _, tr_ids = km.index.search(tr_feat, 1)
        _, te_ids = km.index.search(te_feat, 1)

        col_name = f'cluster_{prefix}_{name}'
        tr_ids_dict[col_name] = np.concatenate([np.full(w - 1, -1, dtype=np.int32), tr_ids.flatten()])
        te_ids_dict[col_name] = te_ids.flatten().astype(np.int32)

        del tr_feat, te_feat
        gc.collect()

    return tr_ids_dict, te_ids_dict


# ==========================================
# 3. 统计平滑与无前视阈值计算器 (含高效环形缓冲)
# ==========================================
class NonLookaheadThresholdEngine:
    @staticmethod
    def compute_shrunk_edges(train_df: pl.DataFrame, cluster_cols: list, prior_weight: float = 20.0) -> list:
        mappings = []
        for col in cluster_cols:
            grp = train_df.filter((pl.col(col) != -1) & pl.col('target').is_not_null()).group_by(col).agg([
                pl.len().alias('count'),
                pl.col('target').cast(pl.Int32).sum().alias('wins')
            ])
            grp = grp.with_columns(
                ((pl.col('wins') + prior_weight * 0.5) / (pl.col('count') + prior_weight) - 0.5).alias(f'edge_{col}')
            ).select([col, f'edge_{col}'])
            mappings.append((col, grp))
        return mappings

    @staticmethod
    def calibrate_in_sample_quantiles(train_scores: np.ndarray, target_cov: float = 0.10) -> Tuple[float, float]:
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
        基于 O(1) 环形缓冲区的高性能因果滚动历史分位数自适应计算器。
        彻底消除 600,000 次 np.roll 带来的巨大内存重分配瓶颈。
        """
        n = len(ace_scores)
        signals = np.zeros(n, dtype=np.int32)
        half_cov = target_cov / 2.0

        # 初始化环形缓冲
        hist_buffer = np.array(init_buffer[-window_size:], dtype=np.float32)
        buf_len = len(hist_buffer)
        buf_ptr = 0

        long_th, short_th = np.nan, np.nan

        for i in range(n):
            curr = ace_scores[i]

            # 每 60 步（每小时）或初始计算时更新分位数
            if (i % 60 == 0 or np.isnan(long_th)) and buf_len >= 100:
                long_th = np.percentile(hist_buffer[:buf_len], (1.0 - half_cov) * 100)
                short_th = np.percentile(hist_buffer[:buf_len], half_cov * 100)

            if not np.isnan(long_th):
                if curr >= long_th and curr > 0:
                    signals[i] = 1
                elif curr <= short_th and curr < 0:
                    signals[i] = -1

            # 环形插入更新 (O(1))
            if buf_len < window_size:
                hist_buffer = np.append(hist_buffer, curr)
                buf_len += 1
            else:
                hist_buffer[buf_ptr] = curr
                buf_ptr = (buf_ptr + 1) % window_size

        return signals

    @staticmethod
    def apply_winrate_enhancement_filters(
        df: pl.DataFrame,
        base_signal_col: str,
        edge_cols: list,
        min_consensus_ratio: float = 0.67,
        enable_trend_filter: bool = False
    ) -> pl.Series:
        sigs = df[base_signal_col].to_numpy().copy()

        if enable_trend_filter and 'trend_ema200' in df.columns:
            closes = df['close'].to_numpy()
            ema200s = df['trend_ema200'].to_numpy()
            sigs[(sigs == 1) & (closes < ema200s)] = 0
            sigs[(sigs == -1) & (closes > ema200s)] = 0

        if min_consensus_ratio > 0 and len(edge_cols) > 0:
            edge_mat = df.select([pl.col(c).fill_null(0) for c in edge_cols]).to_numpy()
            pos_count = np.sum(edge_mat > 0, axis=1)
            neg_count = np.sum(edge_mat < 0, axis=1)
            total_scales = edge_mat.shape[1]

            pos_ratio = pos_count / total_scales
            neg_ratio = neg_count / total_scales

            sigs[(sigs == 1) & (pos_ratio < min_consensus_ratio)] = 0
            sigs[(sigs == -1) & (neg_ratio < min_consensus_ratio)] = 0

        return pl.Series(f'{base_signal_col}_enhanced', sigs)


# ==========================================
# 4. 真实时序绩效评估报告生成器
# ==========================================
def evaluate_performance(df: pl.DataFrame, signal_col: str, title: str):
    print(f"\n==================== {title} ====================")
    clean = df.filter(pl.col('target').is_not_null() & pl.col(signal_col).is_not_null())
    total_len = len(clean)

    sig_active = clean.filter(pl.col(signal_col) != 0)
    sig_count = len(sig_active)

    if sig_count == 0:
        print("未触发任何信号。")
        return

    cov_rate = sig_count / total_len
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
# 5. 主策略执行流 (Pipeline)
# ==========================================
def run_ace_pipeline(df: pl.DataFrame, config: dict, target_period: int = 15, split_ratio: float = 0.8):
    print(">>> 启动 ACE 生产级无前视量化策略引擎...")

    # 1. 在全量时序上先计算技术指标（纯回顾性指标，无未来泄露，确保测试集起点拥有完整历史均线平滑）
    print("1. 计算原生技术指标 (MACD, Stochastic 29 & 69, Trend EMA 200)...")
    df = calculate_indicators(df)

    # 2. 严格时序物理切分
    split_idx = int(len(df) * split_ratio)

    # 切分训练集与测试集
    # 为保证严格因果无泄露，训练集剔除最后 target_period 根标签（防止其未来 15m 价格跨入测试集）
    train_full = df.slice(0, split_idx).with_columns(
        target = (pl.col('close').shift(-target_period) > pl.col('close'))
    )
    train_df = train_full.slice(0, split_idx - target_period)

    test_df = df.slice(split_idx).with_columns(
        target = (pl.col('close').shift(-target_period) > pl.col('close'))
    )

    print(f"2. 严格时序物理切分: 训练集有效样本 {len(train_df):,} 行, 测试集样本 {len(test_df):,} 行")

    # 提取数组及紧邻测试集起点的历史缓冲（包含被剔除标签但价格连续的最后 15 根 K 线）
    tr_prices = train_df['close'].to_numpy().astype(np.float32)
    te_prices = test_df['close'].to_numpy().astype(np.float32)
    pre_te_prices = df.slice(split_idx - 200, 200)['close'].to_numpy().astype(np.float32)

    tr_k = np.nan_to_num(train_df['stoch_k_29_3'].to_numpy(), nan=50.0).astype(np.float32)
    te_k = np.nan_to_num(test_df['stoch_k_29_3'].to_numpy(), nan=50.0).astype(np.float32)
    pre_te_k = np.nan_to_num(df.slice(split_idx - 200, 200)['stoch_k_29_3'].to_numpy(), nan=50.0).astype(np.float32)

    # 3. 执行无泄露 Faiss 聚类 (无缝衔接连续历史特征)
    print("3. 执行价格序列与随机指标序列的多尺度聚类表征...")
    tr_p_ids, te_p_ids = run_hybrid_multi_scale_clustering_leakproof(
        tr_prices, te_prices, pre_te_prices, config, prefix='price'
    )
    tr_s_ids, te_s_ids = run_hybrid_multi_scale_clustering_leakproof(
        tr_k, te_k, pre_te_k, config, prefix='stoch', is_indicator=True
    )

    train_df = train_df.with_columns([pl.Series(k, v) for k, v in {**tr_p_ids, **tr_s_ids}.items()])
    test_df = test_df.with_columns([pl.Series(k, v) for k, v in {**te_p_ids, **te_s_ids}.items()])

    # 4. 计算贝叶斯平滑 Edge
    print("4. 仅在训练集上统计簇 Edge (采用贝叶斯后验收缩)...")
    cluster_cols = [c for c in train_df.columns if c.startswith('cluster_')]
    mappings = NonLookaheadThresholdEngine.compute_shrunk_edges(train_df, cluster_cols, prior_weight=20.0)

    for col, mapping_df in mappings:
        train_df = train_df.join(mapping_df, on=col, how='left')
        test_df = test_df.join(mapping_df, on=col, how='left')

    # 5. 计算 ACE 得分
    edge_cols = [f'edge_{c}' for c in cluster_cols]
    train_df = train_df.with_columns(ace_score = pl.sum_horizontal([pl.col(c).fill_null(0) for c in edge_cols]))
    test_df = test_df.with_columns(ace_score = pl.sum_horizontal([pl.col(c).fill_null(0) for c in edge_cols]))

    # 6. 方案 A: 样本内非对称分位数基准 (Top 10% 覆盖率)
    print("5. [模式 1 - 基准标准模式] 样本内分位数校准 (覆盖率 10%)...")
    tr_scores = train_df['ace_score'].to_numpy()
    long_th, short_th = NonLookaheadThresholdEngine.calibrate_in_sample_quantiles(tr_scores, target_cov=0.10)
    print(f"   -> 训练集校准基准阈值: 做多 >= {long_th:.4f}, 做空 <= {short_th:.4f}")

    test_df = test_df.with_columns(
        sig_std = pl.when(pl.col('ace_score') >= long_th).then(1)
                    .when(pl.col('ace_score') <= short_th).then(-1)
                    .otherwise(0)
    )
    evaluate_performance(test_df, 'sig_std', '模式 1: 样本内分位数校准 (覆盖率 10%)')

    # 7. 方案 B: 因果滚动自适应分位数模式
    print("\n6. [模式 2 - 因果滚动自适应模式] 滚动历史自适应分位数 (覆盖率 10%)...")
    te_scores = test_df['ace_score'].to_numpy()
    sig_rolling = NonLookaheadThresholdEngine.generate_rolling_causal_signals(
        te_scores,
        init_buffer=tr_scores,
        window_size=1440 * 14,
        target_cov=0.10
    )
    test_df = test_df.with_columns(sig_rolling = pl.Series('sig_rolling', sig_rolling))
    evaluate_performance(test_df, 'sig_rolling', '模式 2: 因果滚动历史分位数 (动态自适应 10%)')

    # 8. 【胜率提升模式 3】：多尺度一致性共振强化 (Consensus Filter >= 67%)
    print("\n7. [模式 3 - 胜率增强模式] Top 8% 覆盖率 + 多尺度一致性共振过滤 (Consensus >= 67%)...")
    long_th_8, short_th_8 = NonLookaheadThresholdEngine.calibrate_in_sample_quantiles(tr_scores, target_cov=0.08)
    test_df = test_df.with_columns(
        sig_8 = pl.when(pl.col('ace_score') >= long_th_8).then(1)
                  .when(pl.col('ace_score') <= short_th_8).then(-1)
                  .otherwise(0)
    )
    sig_8_enh = NonLookaheadThresholdEngine.apply_winrate_enhancement_filters(
        test_df,
        base_signal_col='sig_8',
        edge_cols=edge_cols,
        min_consensus_ratio=0.67,
        enable_trend_filter=False
    )
    test_df = test_df.with_columns(sig_8_enh = sig_8_enh)
    evaluate_performance(test_df, 'sig_8_enh', '🔥 模式 3: 胜率增强模式 (覆盖率 8% + 67%多尺度共振)')

    # 9. 【胜率提升模式 4】：极致确信度 (Top 2% 强力信号)
    print("\n8. [模式 4 - 极致胜率模式] Top 2% 极致分位截断...")
    long_th_2, short_th_2 = NonLookaheadThresholdEngine.calibrate_in_sample_quantiles(tr_scores, target_cov=0.02)
    test_df = test_df.with_columns(
        sig_top2 = pl.when(pl.col('ace_score') >= long_th_2).then(1)
                     .when(pl.col('ace_score') <= short_th_2).then(-1)
                     .otherwise(0)
    )
    evaluate_performance(test_df, 'sig_top2', '🚀 模式 4: 极致胜率模式 (Top 2% 超强置信度)')

    print("\n>>> 策略全模式执行完毕！")
    return test_df


# ==========================================
# 6. 多文件/目录/全量数据载入工具
# ==========================================
def load_dataset(data_path: str, use_spot: bool = False) -> pl.DataFrame:
    """
    加载数据，支持单文件、通配符或目录，并健壮支持现货与合约。
    """
    if os.path.isdir(data_path):
        parquet_files = sorted(glob.glob(os.path.join(data_path, "*.parquet")))
        if not parquet_files:
            raise FileNotFoundError(f"目录 {data_path} 中未找到任何 parquet 文件！")
        print(f"正在读取目录中 {len(parquet_files)} 个 parquet 文件...")
        dfs = [pl.read_parquet(f) for f in parquet_files]
        df = pl.concat(dfs)
    elif '*' in data_path:
        matched = sorted(glob.glob(data_path))
        if not matched:
            raise FileNotFoundError(f"通配符 {data_path} 未匹配到任何文件！")
        print(f"正在读取匹配的 {len(matched)} 个文件...")
        dfs = [pl.read_parquet(f) if f.endswith('.parquet') else pl.read_csv(f) for f in matched]
        df = pl.concat(dfs)
    else:
        if data_path.endswith('.parquet'):
            df = pl.read_parquet(data_path)
        else:
            df = pl.read_csv(data_path)

    cols = df.columns
    # 时间戳标准化
    for c in ['close_time', 'open_time', 'timestamp', 'time']:
        if c in cols and 'ts' not in cols:
            df = df.with_columns(ts = pl.col(c))
            break

    # 现货或合约收盘价支持
    if use_spot and 'spot_close' in cols:
        print("指定使用现货价格序列作为主行情")
        df = df.with_columns([
            pl.col('spot_open').alias('open'),
            pl.col('spot_high').alias('high'),
            pl.col('spot_low').alias('low'),
            pl.col('spot_close').alias('close')
        ])
    else:
        if 'contract_close' in cols and 'close' not in cols:
            print("指定使用合约价格序列作为主行情")
            df = df.with_columns([
                pl.col('contract_open').alias('open'),
                pl.col('contract_high').alias('high'),
                pl.col('contract_low').alias('low'),
                pl.col('contract_close').alias('close')
            ])

    if 'ts' in df.columns:
        df = df.unique(subset=['ts']).sort('ts')

    return df


def main():
    parser = argparse.ArgumentParser(description="ACE 生产级量化策略引擎 - 纯 Python 版")
    parser.add_argument("--data_path", type=str, default="", help="真实 K 线数据文件路径、目录或通配符 (CSV 或 Parquet)")
    parser.add_argument("--rows", type=int, default=10000, help="演示运行的行数 (用于无文件时的基准演示)")
    parser.add_argument("--fetch_bn", action="store_true", help="自动从 GitHub Release 下载合并 Binance 全量数据")
    parser.add_argument("--spot", action="store_true", help="是否选用现货价格（默认合约价格）")
    args = parser.parse_args()

    if args.fetch_bn:
        from fetch_bn_data import download_and_combine_bn_klines
        target_file = "/app/bn_ethusdt_1m_all.parquet"
        target_file = download_and_combine_bn_klines(merged_output_path=target_file, use_spot=args.spot)
        df = load_dataset(target_file, use_spot=args.spot)
    elif args.data_path:
        print(f"正在加载指定数据: {args.data_path}")
        df = load_dataset(args.data_path, use_spot=args.spot)
    else:
        default_bn = "/tmp/eth_1m_all.parquet"
        if os.path.exists(default_bn):
            print(f"检测到本地全量 Binance 历史数据，直接加载: {default_bn}")
            df = load_dataset(default_bn, use_spot=args.spot)
        else:
            print("未指定数据且无本地数据，生成高仿真几何布朗运动 + 波动聚集时序...")
            np.random.seed(42)
            n = args.rows
            rets = np.random.standard_t(df=5, size=n) * 0.002
            prices = 2000.0 * np.exp(np.cumsum(rets))
            df = pl.DataFrame({
                'ts': np.arange(n),
                'open': prices,
                'high': prices * (1.0 + np.abs(np.random.randn(n)) * 0.001),
                'low': prices * (1.0 - np.abs(np.random.randn(n)) * 0.001),
                'close': prices
            })

    run_ace_pipeline(df, DEFAULT_HYBRID_CONFIG)


if __name__ == '__main__':
    main()
