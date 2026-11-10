# 《“faissv6_ipynb”的副本.ipynb》深度逻辑剖析与无前视置信度阈值确定方案

---

## 一、 Notebook 整体架构与全链路逻辑深度分析

该 Notebook 实现了一套基于 **Faiss 向量聚类与多尺度共振（ACE, Adaptive Clustering Edge / Asymmetric Clustering Edge）** 的量化时序分类预测策略，标的为 ETH 1分钟 K 线数据，预测目标为未来 15 分钟的价格涨跌。

### 1. 核心流程与数据流图

```
原始行情 (ETH 1m K线)
   │
   ├─► 目标标签生成: target = (close[t+15] > close[t])
   ├─► 技术指标计算: Stoch_K(29, 3), Stoch_K(69, 3), MACD
   │
   ▼
时序切分 (物理无泄露切分: 前 80% Train, 后 20% Test)
   │
   ├──► 价格滑动窗口标准化 (10, 15, 30, 60, 120) + 末端重复加权 (End-point emphasis)
   └──► 随机指标滑动窗口归一化 (10, 15, 30, 60, 120)
   │
   ▼
Faiss K-Means 聚类表征 (严格仅在 Train 上 fit 中心)
   │
   ├──► 训练集每个尺度分配 Cluster ID
   └──► 测试集通过 `km.index.search(1)` 映射到已有 Cluster ID
   │
   ▼
簇级胜率与 Edge 统计 (仅在 Train 上统计)
   │
   └──► Edge = 簇内胜率(target.mean) - 0.5
   │
   ▼
ACE 共振打分 (Point-in-Time 累加)
   │
   └──► ace_score = Σ (各尺度对应的 edge)
   │
   ▼
后验效果评估 (当前代码存在的问题点)
   │
   └──► 将 Test 集全部 ace_score 绝对值全局排序，切片取 Top 1%, 2%, 5%, 10%, 25% 统计胜率
```

### 2. 核心代码模块与技术细节剖析

#### （1）数据与标签构建（Cell 2 & Cell 3）
* **标的与周期**：ETH 1分钟K线，样本总量约 2,980,800 条（跨度近 5 年）。
* **预测目标（Label）**：`target = (close.shift(-15) > close)`，即判断 15 根 K 线后的收盘价是否上涨。
  * 真实分布：上涨样本 1,499,836（50.3%），下跌 1,480,949（49.7%），基本平衡但略偏看涨。

#### （2）多尺度无泄露特征提取与聚类引擎（Cell 5 `run_hybrid_multi_scale_clustering_leakproof`）
* **多尺度窗口配置（`HYBRID_CONFIG`）**：
  * 超短周期：`v_short_10`（窗口10，簇数100，末端重复0）
  * 短周期：`short_15`（窗口15，簇数120，末端重复10）
  * 短周期：`short_30`（窗口30，簇数150，末端重复15）
  * 中周期：`mid_60`（窗口60，簇数300，末端重复15）
  * 长周期：`long_120`（窗口120，簇数600，末端重复15）
* **滑动窗口向量化（Stride Tricks）**：
  * 对于时序数组，利用 `as_strided` 无内存复制生成 `(N - w + 1, w)` 维矩阵。
* **特征标准化（Min-Max / Indicator Normalization）**：
  * 价格序列：每个切片窗口内部做 Min-Max 归一化到 `[0, 1]`，消除绝对价格基数影响，提取纯形态（Shape Pattern）。
  * 指标序列：由于 Stoch 本身有界，直接除以 `100.0`。
* **末端强化（Recency Bias / Attention）**：
  * `norm[:, -end_k:] * repeat`：将窗口最后 3 个时间步的特征重复拼接到特征向量后部（例如 repeat=15 时重复 15 次）。
  * **作用**：让欧氏距离在 Faiss 聚类计算时，大幅向“近期走势”倾斜，强化近期临近反转或突破形态的权重。
* **无泄露训练**：严格只用 `train_feat` 训练 `faiss.Kmeans`，测试集只做推理检索。

#### （3）Edge 映射与共振打分（Cell 8 `run_leakproof_v30_execution`）
* 训练集计算每个簇对 target 的平均值：$\text{Edge}_k = P(\text{target}=1 \mid \text{cluster}=k) - 0.5$。
* 测试集通过 Left Join 匹配各尺度的 Edge。
* **ACE Score 打分公式**：
  $$\text{ace\_score}_t = \sum_{m \in \text{Scales}} \text{Edge}_{m, t}$$
  * 正分（$\text{ace\_score} > 0$）表示多尺度共振看多；
  * 负分（$\text{ace\_score} < 0$）表示多尺度共振看空；
  * 分值的绝对值代表共振强度与置信度。

#### （4）代码中发现的一处隐蔽 Bug
在 Cell 8 的第 3 步中：
```python
train_final = train_df.with_columns([pl.Series(k, v) for k, v in {**tr_p_ids, **tr_s_ids}.items()])
```
由于 `tr_p_ids` 和 `tr_s_ids` 内部的 key 均为 `f'cluster_{name}'`，字典解包 `{**tr_p_ids, **tr_s_ids}` 导致 **指标聚类（Stoch）的列直接覆盖了价格聚类（Price）的列**！在后续的 Edge 计算中，实际只使用了指标聚类，价格聚类完全没有参与打分。这是后续重构时必须修复的缺陷（应分别命名为 `cluster_price_{name}` 与 `cluster_stoch_{name}`）。

---

## 二、 关键问题诊断：现有阈值确定方式为何具有“前视偏差”？

Notebook 中的 Cell 7（`report_real_performance`）和 Cell 10（`analyze_signal_directions`）确定置信度阈值的方法是：
```python
clean_df = df.filter(...)
df_sorted = clean_df.with_columns(abs_s = pl.col('ace_score').abs()).sort('abs_s', descending=True)
top_count = int(total_test_rows * cov_rate) # 如 cov_rate = 0.10
sub = df_sorted.head(top_count)
```

### 为什么这是严重的前视（Lookahead Bias）？
1. **全局未来信息利用**：它是在整个测试集（长达 1 年甚至更久）全部运行完毕后，**以后验（Ex-post）全局排序** 的方式选出绝对值最高的前 10% 样本。
2. **实盘不可行**：在 $T$ 时刻来了一根新的 1 分钟 K 线，计算出一个 `ace_score = 0.18`。在没有未来的情况下，**你根本不知道 0.18 是否属于未来 1 年全局最高的前 10%**。如果后市波动剧烈，0.18 可能只是前 50%；如果后市窄幅震荡，0.18 可能是前 1%。
3. **市场机制漂移（Regime Shift）破坏**：牛市、熊市和震荡市的 ace_score 分布截然不同。后验全局排序会把高波动时段的信号集中选出，造成实盘信号频率极度不均匀（有时连续几天没信号，有时一天产生数百个信号）。

---

## 三、 在没有任何前视的情况下，如何科学确定置信度阈值？

在实盘交易与严格的因果时序系统（Causal Point-in-Time System）中，确定信号置信度阈值有 **5 种层层递进的量化方案**：

```
                    ┌───────────────────────────────────────────────┐
                    │      无前视置信度阈值确定架构 (Point-in-Time)   │
                    └───────────────────────────────────────────────┘
                                           │
         ┌─────────────────────────────────┼─────────────────────────────────┐
         ▼                                 ▼                                 ▼
【方案一：样本内分位数校准】        【方案二：滚动因果历史窗口】        【方案三：统计显著性与贝叶斯】
 - 在训练集确定分位阈值              - 基于过去 N 天滑动窗口             - 假设检验 (Binomial Test)
 - 静态阈值: θ_long, θ_short         - 动态自适应市场波动与偏度          - 经验贝叶斯收缩防过拟合
         │                                 │                                 │
         └─────────────────────────────────┼─────────────────────────────────┘
                                           │
         ┌─────────────────────────────────┴─────────────────────────────────┐
         ▼                                                                   ▼
【方案四：多尺度一致性共振门槛】                                    【方案五：成本与净期望收益门槛】
 - 方向一致性评分 (Consensus)                                       - 胜率 × 赔率 - 摩擦成本 > 0
 - 过滤“单一极端尺度带偏”                                           - 结合 ATR 动态调整
```

---

### 方案一：样本内分位数校准与双边非对称阈值（In-Sample Quantile Calibration）

#### 原理：
所有的分位数与阈值划分，**只能在切分点以前的历史训练集（Train Set）上进行计算**，然后固定到测试集/实盘中使用。
此外，从 Notebook 的输出可以看出：
* 测试集整体基准胜率偏向多头（Target=1 占比 50.56%）；
* 正向平均分值（0.1459）显著大于负向绝对值（0.1294），各尺度强度比普遍在 1.07 ~ 1.18 之间。

因此，**绝不能使用对称的 `|ace_score| > θ`**，否则看涨信号将被过度触发，而看跌信号被严重压制。

#### 数学模型：
在训练集上，分别对正向得分和负向得分确定独立的单边阈值：
1. 设定目标单边覆盖率 $p$（例如希望多头覆盖率 5%，空头覆盖率 5%）：
   $$\theta_{\text{long}} = \text{Quantile}\left(\text{ace\_score}_{\text{train}}, 1 - p\right)$$
   $$\theta_{\text{short}} = \text{Quantile}\left(\text{ace\_score}_{\text{train}}, p\right)$$
2. 实盘触发规则：
   $$\text{Signal}_t = \begin{cases}
   +1 (\text{做多}), & \text{若 } \text{ace\_score}_t \ge \theta_{\text{long}} \\
   -1 (\text{做空}), & \text{若 } \text{ace\_score}_t \le \theta_{\text{short}} \\
   0 (\text{空仓}), & \text{其他}
   \end{cases}$$
* **优点**：完全无前视，计算简单，天然对冲正负分值非对称性。

---

### 方案二：滚动因果历史窗口自适应分位数（Rolling / Expanding Window Quantiles）

#### 原理：
加密货币市场存在显著的波动率聚集和 Regime Shift（如单边大牛市转入漫长阴跌熊市）。固定的静态阈值可能在熊市中完全失效。
可以在 $T$ 时刻，只利用过去 $W$ 个时间步（例如过去 14 天或 30 天，即 $W = 14 \times 1440$ 根 1m K线）的历史 ace_score 动态更新阈值。

#### 数学模型：
在时刻 $t$，仅使用区间 $[t-W, t-1]$ 内的得分集合 $\mathcal{S}_t = \{\text{ace\_score}_\tau\}_{\tau=t-W}^{t-1}$：
$$\theta_{\text{long}, t} = \text{Quantile}(\mathcal{S}_t, 1 - p)$$
$$\theta_{\text{short}, t} = \text{Quantile}(\mathcal{S}_t, p)$$
或者使用标准化 Z-score 动态门槛：
$$\mu_t = \text{mean}(\mathcal{S}_t), \quad \sigma_t = \text{std}(\mathcal{S}_t)$$
$$Z_t = \frac{\text{ace\_score}_t - \mu_t}{\sigma_t}$$
当 $Z_t > +Z_{\text{crit}}$ 时做多，当 $Z_t < -Z_{\text{crit}}$ 时做空。

* **优点**：能够动态适应市场波动率收缩与扩张，保证信号在不同月份中的日均频次稳定。

---

### 方案三：统计显著性检验与经验贝叶斯收缩（Statistical Significance & Empirical Bayes）

#### 原理：
Notebook 中计算簇 Edge 的方式存在一个严重的统计缺陷：**没有考虑簇内样本量 $N$**！
* 假设某个长周期簇只有 10 个样本，刚好 7 次上涨，计算出 $\text{Edge} = 0.7 - 0.5 = +0.20$；
* 另一个高频簇有 5,000 个样本，2,750 次上涨，计算出 $\text{Edge} = 0.55 - 0.5 = +0.05$。
在原代码中，小样本噪声簇的 Edge 高达 0.20，在共振中会产生巨大的虚假信号！

#### 改进方法：
1. **二项分布假设检验门槛（Binomial Significance Filter）**：
   对每个簇进行统计检验（$H_0: p = 0.5$），计算单侧 p 值或 Z 检验统计量：
   $$Z_{\text{cluster}} = \frac{\hat{p} - 0.5}{\sqrt{\frac{0.5 \times 0.5}{N}}}$$
   只有当 $|Z| \ge 1.96$（$p < 0.05$）时，才承认该簇具备真实统计 Edge，否则将其 Edge 置 0！
2. **经验贝叶斯收缩（Empirical Bayes Shrinkage / Laplace Smoothing）**：
   引入贝叶斯先验参数 $M$（等效于伪观测样本数，如 $M=50$），将胜率向无信息先验收缩：
   $$\tilde{p} = \frac{K + M \times 0.5}{N + M}$$
   $$\widetilde{\text{Edge}} = \tilde{p} - 0.5 = \frac{N}{N + M} (\hat{p} - 0.5)$$
   样本量 $N$ 越小，Edge 越被压缩趋近于 0；样本量充分大时，保持原值。这从源头上滤除了高分噪声。

---

### 方案四：多尺度共振一致性投票法（Consensus & Breadth Thresholding）

#### 原理：
原策略的核心思想是 **“ACE 共振”**。但简单的累加可能出现一种假共振：
* 例如 10 个尺度中，有 9 个尺度给出了微弱的负向得分（-0.01），而只有 1 个尺度因为过拟合给出了极大的正向得分（+0.25），相加后 `ace_score = +0.16`（看似强烈看多，但实际上各尺度严重分歧）。
真正的共振必须满足 **“方向广度（Breadth Consensus）”**。

#### 模型设定：
定义时刻 $t$ 的共振广度得分：
$$\text{Positive Count}_t = \sum_{m=1}^{M} \mathbb{I}(\text{Edge}_{m, t} > 0)$$
$$\text{Negative Count}_t = \sum_{m=1}^{M} \mathbb{I}(\text{Edge}_{m, t} < 0)$$
**置信度准入复合条件**：
* 做多信号：$\text{ace\_score}_t \ge \theta_{\text{long}}$ **且** $\text{Positive Count}_t \ge K_{\text{consensus}}$（例如 5 个尺度中至少 4 个尺度同向为正）；
* 做空信号：$\text{ace\_score}_t \le \theta_{\text{short}}$ **且** $\text{Negative Count}_t \ge K_{\text{consensus}}$。

---

### 方案五：结合期望收益与手续费摩擦的经济学门槛（Economic Hurdle）

#### 原理：
胜率 $56\%$ 是否足以开仓，必须取决于盈亏比和手续费摩擦：
在合约交易中，单次开平仓手续费约 $0.05\% \sim 0.08\%$，滑点约 $0.02\%$，双边摩擦约为 $0.12\% \sim 0.15\%$。
如果只是在小波动时段达到 56% 胜率，扣除摩擦后净期望收益可能为负。

#### 模型设定：
结合当前市场的真实波动率（ATR，Average True Range）：
1. 预测在 15 分钟内的预期波动幅度 $\Delta P \approx \alpha \cdot \text{ATR}_{15}$；
2. 期望收益率：
   $$\mathbb{E}[R] = P_{\text{win}} \cdot \mathbb{E}[\text{Win}] - (1 - P_{\text{win}}) \cdot \mathbb{E}[\text{Loss}] - \text{Fee}$$
3. 将 $P_{\text{win}} \approx 0.5 + c \cdot \text{ace\_score}$ 代入，解出使得 $\mathbb{E}[R] > \text{Target Return}$ 的最小置信度阈值：
   $$\theta_{\text{economic}} = \frac{\text{Fee} + \text{Target Return}}{c \cdot \text{ATR}_{15}}$$
波动率较小时，要求更高的 ace_score 才能弥补固定手续费；波动率充分大时，较低的 ace_score 即可触发交易。

---

## 四、 生产实盘最佳实施架构推荐

为了使该 Notebook 中的模型具备生产落地能力，推荐采用 **两层过滤的无前视阈值引擎**：

| 层次 | 模块名称 | 具体作用与规则 |
|---|---|---|
| **底模层** | **贝叶斯收缩 + Bug 修复** | 修复 Price/Stoch 键名覆盖 Bug；对簇 Edge 进行贝叶斯收缩防小样本过拟合。 |
| **一阶过滤** | **多尺度广度一致性** | 要求 5 个时间尺度中至少 4 个尺度同向（$\ge 80\%$ 广度一致）。 |
| **二阶过滤** | **因果滚动分位数阈值** | 基于过去 14 天因果历史计算正/负各自独立的 $90\%$ 分位数作为开仓阈值。 |
| **风控层** | **微观状态与波动率过滤** | 当 ATR 过低（手续费无法覆盖）或处于异常点时不产生信号。 |

上述方案不仅完全根除了前视偏差，还能解决原始代码中“看多看空信号非对称”以及“小样本簇噪声干扰”的问题，使策略在真实流式行情中具备坚实的统计学基础与实战稳定性。
