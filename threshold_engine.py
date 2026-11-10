"""
threshold_engine.py
===================
量化研究与生产级实现：无前视偏差（No Lookahead Bias）的置信度阈值确定与回测引擎。

包含五种无前视阈值计算器：
1. InSampleQuantileCalibrator: 样本内分位数校准与双边非对称阈值
2. RollingQuantileCalibrator: 因果滑动历史窗口动态分位数阈值
3. EmpiricalBayesEdgeProcessor: 经验贝叶斯收缩与二项统计显著性检验（从源头过滤虚假Edge）
4. MultiScaleConsensusFilter: 多尺度方向广度一致性复合过滤器
5. PointInTimeBacktester: 严格逐时间步（Point-in-Time）流式信号生成与回测评估
"""

import numpy as np
import polars as pl
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional


class EmpiricalBayesEdgeProcessor:
    """
    经验贝叶斯 Edge 处理器与统计显著性检验
    解决原始代码中长周期簇因样本量少、胜率虚高导致过拟合与虚假共振的问题。
    """
    def __init__(self, prior_weight: float = 30.0, z_crit: float = 1.96):
        """
        :param prior_weight: 等效先验样本量 M。当样本量 N 较小时，胜率会被强烈收缩到先验 0.5。
        :param z_crit: 统计显著性 Z 临界值（默认 1.96 对应 95% 置信度，置 0 则不进行硬截断）。
        """
        self.prior_weight = prior_weight
        self.z_crit = z_crit

    def shrink_edges(self, cluster_series: np.ndarray, target_series: np.ndarray) -> Dict[int, float]:
        """
        根据训练集样本，计算经过贝叶斯收缩和显著性过滤后的 Edge。
        """
        unique_clusters = np.unique(cluster_series)
        edge_map = {}

        for c in unique_clusters:
            if c == -1:
                edge_map[c] = 0.0
                continue
            mask = (cluster_series == c)
            n = np.sum(mask)
            if n == 0:
                edge_map[c] = 0.0
                continue

            k = np.sum(target_series[mask])
            raw_win_rate = k / n

            # 1. 经验贝叶斯后验收缩 (Laplace / Beta-Binomial Smoothing)
            shrunk_win_rate = (k + self.prior_weight * 0.5) / (n + self.prior_weight)
            shrunk_edge = shrunk_win_rate - 0.5

            # 2. 统计显著性 Z 检验: H0: p = 0.5
            z_score = (raw_win_rate - 0.5) / np.sqrt(0.25 / n)
            if abs(z_score) < self.z_crit and self.z_crit > 0:
                # 样本量不足以证明真实非随机 Edge，置为 0
                shrunk_edge = 0.0

            edge_map[c] = float(shrunk_edge)

        return edge_map


class InSampleQuantileCalibrator:
    """
    方案一：基于训练集历史分布的静态分位数校准
    完全杜绝测试集后验排序，采用双边独立分位数以解决多空非对称性问题。
    """
    def __init__(self, target_coverage: float = 0.10):
        """
        :param target_coverage: 目标整体覆盖率（例如 10%）。双边各占 target_coverage / 2。
        """
        self.target_coverage = target_coverage
        self.long_threshold: float = 0.0
        self.short_threshold: float = 0.0

    def fit(self, train_ace_scores: np.ndarray):
        """
        仅在训练集上拟合阈值！
        """
        valid_scores = train_ace_scores[np.isfinite(train_ace_scores)]
        half_cov = self.target_coverage / 2.0

        # 上分位数（多头）：取 top (half_cov)
        self.long_threshold = float(np.percentile(valid_scores, (1.0 - half_cov) * 100))
        # 下分位数（空头）：取 bottom (half_cov)
        self.short_threshold = float(np.percentile(valid_scores, half_cov * 100))

        # 保护机制：如果多头阈值<=0，则强制要求>0；空头阈值>=0，则强制要求<0
        if self.long_threshold <= 0:
            pos_scores = valid_scores[valid_scores > 0]
            if len(pos_scores) > 0:
                self.long_threshold = float(np.percentile(pos_scores, (1.0 - self.target_coverage) * 100))
            else:
                self.long_threshold = 0.01

        if self.short_threshold >= 0:
            neg_scores = valid_scores[valid_scores < 0]
            if len(neg_scores) > 0:
                self.short_threshold = float(np.percentile(neg_scores, self.target_coverage * 100))
            else:
                self.short_threshold = -0.01

        return self

    def predict_signals(self, test_ace_scores: np.ndarray) -> np.ndarray:
        """
        逐行推理：1 为做多，-1 为做空，0 为观望
        """
        signals = np.zeros(len(test_ace_scores), dtype=np.int32)
        signals[test_ace_scores >= self.long_threshold] = 1
        signals[test_ace_scores <= self.short_threshold] = -1
        return signals


class RollingQuantileCalibrator:
    """
    方案二：因果滚动历史窗口自适应分位数（Point-in-Time Rolling Quantiles）
    时刻 t 仅根据 [t - window, t - 1] 的历史 ace_score 分布计算阈值，适应波动率聚集与环境漂移。
    """
    def __init__(self, window_size: int = 1440 * 14, target_coverage: float = 0.10):
        """
        :param window_size: 历史回溯窗口大小（例如 14天 = 1440 * 14 根1分钟K线）
        :param target_coverage: 目标覆盖率
        """
        self.window_size = window_size
        self.target_coverage = target_coverage

    def generate_causal_signals(self, ace_scores: np.ndarray, init_history: Optional[np.ndarray] = None) -> np.ndarray:
        """
        严格因果信号生成：在处理 t 时刻时，只能看到 t 之前的分数。
        """
        n = len(ace_scores)
        signals = np.zeros(n, dtype=np.int32)
        half_cov = self.target_coverage / 2.0

        # 如果提供了训练集末尾作为初识缓冲，则初始化缓冲
        if init_history is not None and len(init_history) > 0:
            buffer = list(init_history[-self.window_size:])
        else:
            buffer = []

        # 为高效计算滚动分位数，可结合定周期刷新或在线滑动窗口
        # 这里为兼顾展示速度与精确度，按批次或滑动更新
        long_thresh = np.nan
        short_thresh = np.nan

        # 预先将初始历史转为 numpy
        hist_arr = np.array(buffer, dtype=np.float32)

        for i in range(n):
            curr_score = ace_scores[i]

            # 每 60 步（每小时）或窗口初次填满时更新一次阈值，避免每分钟重算排序的巨大开销
            if (i % 60 == 0 or np.isnan(long_thresh)) and len(hist_arr) >= 100:
                long_thresh = np.percentile(hist_arr, (1.0 - half_cov) * 100)
                short_thresh = np.percentile(hist_arr, half_cov * 100)

            if not np.isnan(long_thresh):
                if curr_score >= long_thresh and curr_score > 0:
                    signals[i] = 1
                elif curr_score <= short_thresh and curr_score < 0:
                    signals[i] = -1

            # 滚动更新历史缓冲 (FIFO)
            if len(hist_arr) >= self.window_size:
                hist_arr = np.roll(hist_arr, -1)
                hist_arr[-1] = curr_score
            else:
                hist_arr = np.append(hist_arr, curr_score)

        return signals


class MultiScaleConsensusFilter:
    """
    方案四：多尺度共振一致性投票法
    要求不单单总分 ace_score 达标，且各子尺度 Edge 的正负方向一致率必须达到阈值。
    """
    def __init__(self, min_consensus_ratio: float = 0.8):
        """
        :param min_consensus_ratio: 同向尺度占比阈值（例如 5 个尺度中至少 4 个同向，即 80%）
        """
        self.min_consensus_ratio = min_consensus_ratio

    def filter(self, edge_matrix: np.ndarray, base_signals: np.ndarray) -> np.ndarray:
        """
        :param edge_matrix: 形状为 (N, M) 的各尺度 Edge 得分
        :param base_signals: 基础阈值产生的信号 (+1, -1, 0)
        :return: 经过共振广度过滤后的稳健信号
        """
        n, m = edge_matrix.shape
        filtered_signals = base_signals.copy()

        pos_count = np.sum(edge_matrix > 0, axis=1)
        neg_count = np.sum(edge_matrix < 0, axis=1)

        pos_ratio = pos_count / m
        neg_ratio = neg_count / m

        # 多头信号必须满足看多广度
        long_invalid = (base_signals == 1) & (pos_ratio < self.min_consensus_ratio)
        filtered_signals[long_invalid] = 0

        # 空头信号必须满足看空广度
        short_invalid = (base_signals == -1) & (neg_ratio < self.min_consensus_ratio)
        filtered_signals[short_invalid] = 0

        return filtered_signals


def evaluate_causal_performance(signals: np.ndarray, targets: np.ndarray, method_name: str) -> Dict[str, any]:
    """
    对因果信号进行无前视绩效评估
    """
    mask = (signals != 0)
    total_samples = len(signals)
    signal_count = np.sum(mask)

    if signal_count == 0:
        return {
            '方案': method_name,
            '覆盖率': "0.00%",
            '总体准确率': "0.00%",
            '做多信号数': 0,
            '做多准确率': "0.00%",
            '做空信号数': 0,
            '做空准确率': "0.00%"
        }

    cov_rate = signal_count / total_samples

    # 准确率判定：信号为 1 时 target==1 算对；信号为 -1 时 target==0 算对
    is_correct = np.where(signals[mask] == 1, targets[mask] == 1, targets[mask] == 0)
    overall_acc = np.mean(is_correct)

    long_mask = (signals == 1)
    long_count = np.sum(long_mask)
    long_acc = np.mean(targets[long_mask] == 1) if long_count > 0 else 0.0

    short_mask = (signals == -1)
    short_count = np.sum(short_mask)
    short_acc = np.mean(targets[short_mask] == 0) if short_count > 0 else 0.0

    return {
        '方案': method_name,
        '覆盖率': f"{cov_rate:.2%}",
        '总体准确率': f"{overall_acc:.2%}",
        '做多信号数': int(long_count),
        '做多准确率': f"{long_acc:.2%}",
        '做空信号数': int(short_count),
        '做空准确率': f"{short_acc:.2%}"
    }
