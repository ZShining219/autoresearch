# 复用审计表（Reuse Audit）

> 2026-10-12 复核更正：本文对 RobustLight“无决策感知训练”的概括不准确。原论文 §3.2 / Algorithm 1 及官方代码包含奖励驱动的 DSI actor/critic 更新与预测历史上的多步训练。该项及后续创新定位请以 [专项诊断](robustlight_diagnosis_20261012.md) 为准；本次未重新核查本文其他文献条目。

调研日期：2026-10-12。所有外部仓库已克隆至 `auto_research/refs/`（浅克隆，仅供阅读审计）。
审计依据为论文具体章节/公式与代码文件路径，非摘要推测。

---

## 1. LMP-GNN（arXiv:2608.21734v2, 2026-08/09）

| 项目 | 内容 |
|---|---|
| 原论文具体研究问题 | 车道级车辆数观测缺失下，如何在**保持已观测量不变、SMP 控制器不变**的前提下，概率重建缺失车道数并追踪其对压力分数→相位选择→闭环交通的影响（§1） |
| 核心技术机制 | A2 变体：12 维车道特征（式5：log(1+Q)、缺失标记 z、log(1+h) 最近观测、min(s,50)/50 staleness、路口可见均值 log(1+q̇_vis)、车道长/限速/宽/朝向 sin-cos/车道序/车道数）→ 线性编码 64 维 → 3 层 lane-movement 消息传递（式7-10：movement 两端车道均值→movement 表示→车道残差更新）→ 共享头输出 μ=ReLU(h+δ)（锚定最近观测的修正量）与 σ（log-var clip [-2,6]）（式11）；63,362 参数 |
| 训练目标 | **仅 masked 高斯 NLL**（式12，只在缺失位置计损失），Adam lr=1e-3，batch 64，grad-clip 5，≤3000 更新，Gudang 验证早停（§4.4）。**确认：没有使用任何决策相关损失** |
| 缺失→控制接口 | 式4：x_i = (1−z_i)q_i + z_i·g_k(μ,σ,s)，观测直通；三条输入规则（式13-15）：Predictive Mean g_μ=μ；Fixed Lane Discount g_fix=max(0, μ−λσ), λ=1；Staleness Gate g_stale=max(0, μ−λ·e^{−s/τ}·σ), τ=2 决策步 |
| SMP 定义 | 式1-2：P_{j,p}(x)=Σ_{(u,d)∈E_{j,p}}(x_u−x_d)，argmax 取并列第一；t_min 最小相位时长（§3.2） |
| 缺失场景 | §5.2/Table IV：random（每决策独立车道）、correlated（进口道成组 movement-start lanes）、sustained outage（单车道 300/600s）、approach blackout（整条进口道全程） |
| 评估协议 | 三层：重建（masked MAE/RMSE、NLL、coverage）→ 决策时诊断（压力分数误差、phase agreement、wrong-phase rate、Spearman 关联）→ 闭环 ATT；matched outage schedule 配对比较（§5.4-5.5） |
| 官方代码 | 论文未提供仓库链接（arXiv v2 未见 Code 链接）；许可证 CC BY-NC-ND 4.0（论文本身） |
| 平台/依赖 | CityFlow 主实验（1s 步长、3600s、10s 决策间隔）；SUMO 仅用于 Cologne1 迁移辅助实验；Python 3.9.6 + PyTorch 2.8.0 + SUMO 1.27.1/TraCI（§5.1） |
| 可直接复用 | SMP 公式语义（式1-2）与 first-max 平局规则；特征工程清单（式5）；缺失场景分类（§5.2）；三层评估协议与 matched-schedule 配对思想；输入规则 g_μ/g_fix/g_stale 作为 E3 对照基线；A2 图结构作为"基于论文的重实现"参考 |
| 需要修改/重实现 | 无官方代码→全部需按论文重实现并标注"重实现/近似基线"； CityFlow→SUMO 移植（论文 §7.3 自述 simulator-dependent，需本地校准）；无 GRU 时序窗——仅最近观测锚点+staleness（这是其模型与我们 M1 的实质差异） |
| 与本研究的重合及差异 | **重合**：缺失观测→SMP 的问题设定、选择性重建接口、三类缺失几何、三层评估。**差异（本研究增量）**：① M1 引入 GRU 历史窗（其仅用 h,s 两个标量摘要历史）；② M2 决策感知训练——其 §7.4 明确把 "decision-aware or joint uncertainty" 列为 future work，未实现；③ M3 是**相位级**风险评分（压力不确定性传播+相位间隔），其规则是**车道级**输入变换且所有相位共享同一折扣；④ 平台 SUMO 原生 |
| 独立复现可行性 | 中。公式完整（式1-15 全部给出），但无代码、无 CityFlow 路网文件；我方按式1-15 在 SUMO 上重实现 A2 类模型，标注 reimplementation |

**关键定位结论**：LMP-GNN 是当前最近邻工作，但其训练目标是纯重建（masked Gaussian NLL），决策机制是三条固定输入规则。本研究 M2/M3 不与其实质贡献重复；M3 必须与其 Fixed Lane Discount / Staleness Gate 做受控对比（E3）。

---

## 2. RobustLight（ICML 2025, PMLR 267:36192-36214）

| 项目 | 内容 |
|---|---|
| 原论文具体研究问题 | RL-TSC 在对抗攻击与缺失数据下的鲁棒性；用扩散模型在**不改动原 TSC 平台**的前提下修复被破坏状态（plug-and-play restoration） |
| 核心技术机制 | 增强扩散模型 + Dynamic State Infilling（DSI）在线训练；WeightedLoss L1/L2；对 state dict 注入高斯/均匀噪声模拟攻击 |
| 官方代码 | 有，github.com/MJLee00/RobustLight（及 Software 链接指向 AdvancedAI-ComplexSystem/SmartCity/robustlight） |
| 关键模块位置 | `inferences/diffusion.py`（DDPM：beta schedule、q_sample、p_sample_loop、recover）；`inferences/rl_dynamic_state_filling.py`（DSI，839 行）；`utils/make_mask_noise.py`（仅 22 行，噪声注入）；`models/maxpressure_agent.py`（CityFlow 版，用 `traffic_movement_pressure_queue` 移动级压力）；`utils/config.py`（MIN_ACTION_TIME=15、YELLOW_TIME=5 等） |
| 输入/输出/依赖 | CityFlow 环境（utils/cityflow_env.py）；Hangzhou 4x4/Jinan 3x4/NewYork 28x7 数据；state dict（lane_num_vehicle、traffic_movement_pressure 等特征） |
| 许可证 | **无 LICENSE 文件** → 保留权利，仅可参考接口设计与思路，不复制代码 |
| 可直接复用 | "修复模块与原控制器解耦"的接口设计（restoration 输出作为 controller 输入、控制器不变）——与 LMP-GNN 的选择性重建接口一致，验证本研究管线设计合理性；`run_maxpressure.py` 的实验编排参考 |
| 需要修改/重实现 | 扩散模型本体按用户要求不默认引入；其压力定义是 movement 级 queue 压力，与本研究的 signed lane-link SMP 不同；CityFlow 依赖 |
| 与本研究的重合及差异 | 重合：状态修复→下游控制的两段式解耦。差异：其针对对抗攻击+缺失、用扩散模型、控制器是 RL；本研究是车道缺失、轻量概率重建、固定 SMP；其无决策感知训练、无相位级风险机制 |
| 独立复现可行性 | 低-中。需 CityFlow+其定制 env；模型训练管线长。本研究仅借鉴解耦接口，不运行其代码 |

---

## 3. MissLight（IJCAI 2023, Mei et al.）

| 项目 | 内容 |
|---|---|
| 原论文具体研究问题 | 真实路网上**部分路口完全没有传感器**（路口级缺失）时的 RL-TSC；两种方案：只补状态、或状态+奖励都补 |
| 核心技术机制 | 状态填补模型（SFM/GraphWN/NN 系列 predictionModel）+ RL 智能体（IDQN/FRAP/MaxPressure 等）；mask_matrix=1 训练掩码、=2 评估掩码；neighbor/non_neighbor 缺失路口模式 |
| 官方代码 | 有，github.com/derekmei233/MissLight（"temporary pre experiments" 自述） |
| 关键模块位置 | `generate_mask.py`+`utils/mask_pos.py::random_mask(num, policy, relation)`——按邻居关系选缺失**路口**；`predictionModel/SFM.py::SFM_predictor.predict(states, phases, relation, mask_pos, mask_matrix, adj_matrix)`——mask_op 填补+reconstruct_data_slice 回写；`metric/maskMetric.py::MSEMetric`（仅缺失位置评 MSE）；`utils/control.py::app1maxp_train/execute`（填补+MaxP 闭环）；`agent/max_pressure_agent.py` |
| 输入/输出/依赖 | CityFlow（utils/control.py::create_world 读 .cfg）；路口级特征（3 通道切片 inter2edge_slice） |
| 许可证 | 无 LICENSE 文件 → 同上处理 |
| 可直接复用 | 缺失场景生成思想（neighbor 相关性建模→迁移为车道级 correlated mask）；masked-only 评估指标（masked MAE）设计；state-inference 与 controller 的接口签名（predict→completed state→controller） |
| 需要修改/重实现 | 其缺失单位是**整个路口**（num∈{1..4} 个路口），本研究是**车道**；其 GraphWN/SFM 面向路口特征，不可直接搬用 |
| 与本研究的重合及差异 | 重合：缺失→填补→控制链路。差异：缺失粒度（路口 vs 车道）、控制器（RL vs SMP）、无概率输出/无决策损失/无相位风险 |
| 独立复现可行性 | 中。代码完整但自述 pre-experiment；CityFlow cfg 体系与 LibSignal 同源 |

---

## 4. GRIN（ICLR 2022, Cini et al.）

| 项目 | 内容 |
|---|---|
| 原论文具体研究问题 | 多元时空序列（传感器网络）缺失值填补：用图结构+序列模型做 graph imputation |
| 核心技术机制 | GRINet=BiGRIL（双向 GRU + 图消息传递 imputation layer）；GCGRUCell=图卷积 GRU 单元（diffusion-conv 门）；输入 (x, mask, u)；impute_only_holes 保证观测位不变 |
| 官方代码 | 有，github.com/Graph-Machine-Learning-Group/grin |
| 关键模块位置 | `lib/nn/layers/gcrnn.py::GCGRUCell`（纯 PyTorch，~40 行，diffusion-conv 门控 GRU 单元，仅依赖本地 spatial_conv.py）；`lib/nn/layers/mpgru.py::MPGRUImputer`（(x,mask)→hidden 编码器-读出头）；`lib/nn/layers/gril.py::BiGRIL`（含 `reverse_tensor` → **双向=用未来信息**）；`lib/nn/models/grin.py::GRINet` |
| 输入/输出/依赖 | x:[B,T,N,C]、mask、外生变量 u、邻接；输出填补值+预测；requirements：torch==1.8、pytorch-lightning==1.4、tensorflow==2.5（还有 fancyimpute 等）——栈很旧 |
| 许可证 | 无 LICENSE 文件 → 参考实现思路 |
| 可直接复用 | **GCGRUCell 结构模式**（扩散卷积做 GRU 门 → 移植为单向变体即可在线使用）；(x,mask) 拼接输入惯例；impute_only_holes 观测位直通语义；masked-only loss 位置 |
| 需要修改/重实现 | BiGRIL 双向→必须改为**单向**（在线决策禁止未来信息，用户明确要求）；整体框架过重（PL+tsl+TF 依赖），不引入框架，只提取单元设计 |
| 与本研究的重合及差异 | 重合：GRU+图+mask 的时空填补主干。差异：其离线全窗双向、交通数据集（METR-LA 等）；本研究因果在线、车道图、下游控制 |
| 独立复现可行性 | 低（依赖栈陈旧、需 tsl 生态）；单元级重实现可行且轻 |

---

## 5. MagiNet（TKDD 2025）

| 项目 | 内容 |
|---|---|
| 原论文具体研究问题 | 时空数据缺失填补：mask-aware 图填补网络 |
| 核心技术机制 | Mask-Aware Spatial-Temporal Attention（空间/时间注意力显式吃 mask m）+ 切比雪夫图卷积（scaled_Laplacian/cheb_polynomial）+ GTU 块 |
| 官方代码 | 有，github.com/JeremyChou28/MagiNet |
| 关键模块位置 | `models/model.py`：Mask_Spatial_Attention(l.108)、Mask_Temporal_Attention(l.145)、Mask_Aware_Spatial_Temporal_Attention(l.195)、Mask_Aware_Spatial_Conv(l.238)、ST_Block(l.353)、MagiNet_submodule(l.472)；`mask.py`（Sinkhorn 掩码工具）；`configs/*.yaml`（METR-LA/PEMS-BAY/Shenzhen/Chengdu/Seattle） |
| 输入/输出/依赖 | torch+scipy；全窗序列+mask+邻接；交通速度数据集 |
| 许可证 | **MIT** → 可直接取代码片段（保留声明） |
| 可直接复用 | mask-aware 注意力的掩码注入方式；图卷积的 chebyshev 实现参考；全窗→决策时截断的适配思路（LMP-GNN 已做过其 decision-time adaptation，可作先例） |
| 需要修改/重实现 | 注意力是全窗非因果 → 决策时版本需限制为历史窗/因果掩码；模型规模远大于 LMP-GNN 类轻量需求 |
| 与本研究的重合及差异 | 重合：mask-aware 时空填补。差异：离线填补任务定位；无控制下游、无不确定性头 |
| 独立复现可行性 | 中。单文件模型、依赖少；但数据集需从其 Google Drive 下载 |

---

## 6. PyEPO（Predict-then-Optimize 工具包）

| 项目 | 内容 |
|---|---|
| 原论文/工具问题 | 预测模型的训练目标直接计入下游优化决策误差（SPO regret），而非仅预测误差（Elmachtoub & Grigas, SPO/SPO+） |
| 核心技术机制 | SPO+ 凸替代损失：L = max_{w∈S}(c−2ĉ)·w + 2ĉ·w*(c) − c·w*(c)；前向解一次扰动问题，反向用子梯度 2(w*(c)−w*(2ĉ−c))（`func/surrogate.py::SPOPlus` l.24、`SPOPlusFunc` l.82） |
| 官方代码 | 有，github.com/khalil-research/PyEPO |
| 关键模块位置 | `pkg/pyepo/func/`：surrogate.py(SPO+)、rank.py(rank-based losses)、blackbox.py、perturbed.py、contrastive.py；`pkg/pyepo/model/omo|grb|ort|mpax/`（knapsack/shortestpath/tsp/vrp 等优化模型，依赖 Gurobi/ORTools/mpax） |
| 输入/输出/依赖 | pred_cost/true_cost/true_sol/true_obj；需 optModel 适配器 + 求解器 |
| 许可证 | **MIT** → 可自由借鉴/引用 |
| 可直接复用 | **SPO+ 的数学形式本身**（不需要其求解器框架）：本研究决策是有限相位集上的 argmax，"LP" 枚举即可；SMP 压力对 x 是线性的 → SPO+ 闭式成立（详见 M2 设计推导） |
| 需要修改/重实现 | 其 optModel 面向 generic LP/MIP（背包/最短路），接入成本高于收益 → 仅借公式实现 ~30 行自定义损失 |
| 与本研究的重合及差异 | 重合：决策感知训练思想正是 M2 内核。差异：无 TSC 语境、无缺失观测；其 SPO+ 的 w* 需要可解优化问题，本研究退化为离散 argmax |
| 独立复现可行性 | 高（pip 装包即可），但本项目不引入为依赖 |

---

## 7. LibSignal（DaRL-LibSignal + sal0-h SUMO 维护分支）

| 项目 | 内容 |
|---|---|
| 原工具问题 | TSC 统一基准框架：Gym 风格接口 + CityFlow/SUMO/OpenEngine 世界 + 经典/RL 智能体 |
| 核心技术机制 | World（仿真器适配）+ Intersection（相位/lanelink/黄灯）+ Agent + Generator（特征）+ Registry 配置 |
| 官方代码 | 有；主线 github.com/DaRL-LibSignal/LibSignal，SUMO 维护分支 github.com/sal0-h/LibSignal |
| 关键模块位置 | `agent/maxpressure.py::MaxPressureAgent.get_action`（l.108-130：lvc[start]−lvc[end] 对 phase_available_lanelinks 求和、first-max、t_min 最短相位）——**SMP 参考实现**；`world/world_sumo.py`：Intersection(l.29-168：`trafficlight.getControlledLinks`→`phase_available_lanelinks`（state=='G'或's' 的 link）、`create_yellows`（l.339-360：相位对间自动生成 'r' 黄灯串）、`pseudo_step`（l.232-256：黄灯期内不换相）)；指标 `get_average_travel_time`(l.780)、`get_cur_throughput`(l.862)、`get_lane_queue_length`(l.805)、`get_pressure`(l.682) |
| 输入/输出/依赖 | sumo.cfg（net/route/tll）、SUMO_HOME、traci/libsumo；主线 world_sumo.py 953 行；sal0-h 版 1607 行（扩展了事件/ghost 车辆等，结构更重） |
| 许可证 | 两仓均无 LICENSE 文件 → 保留权利；作为语义参考移植，注明出处，不整段复制 |
| 可直接复用 | SMP 语义（phase_available_lanelinks 构造、压力求和、first-max、t_min）；黄灯自动生成逻辑设计；观测/指标口径（lane_count、ATT、throughput、queue、delay）；sumo1x1 冒烟场景思路 |
| 需要修改/重实现 | 整体框架（Registry/config/CityFlow 残留）对本研究过重 → 实现聚焦版 SUMO 世界封装，移植其 Intersection/相位/黄灯/指标语义；需在 lane_count 观测上叠加 mask/重建层（原生接口无此概念） |
| 与本研究的重合及差异 | 重合：仿真器适配与控制器语义来源。差异：LibSignal 本身无缺失观测/重建/不确定性概念 |
| 独立复现可行性 | 高（pip eclipse-sumo+traci 已在本机验证）；本研究采用"语义移植+轻量重写"而非框架继承 |

---

## 查重结论（decision-aware recovery + phase-level risk 组合）

检索结论（2026-10 时点）：**未发现"决策感知状态恢复 + 相位级风险决策 + Max-Pressure/SUMO"的相同组合**。最近邻及区分：

| 工作 | 与本研究差异 |
|---|---|
| LMP-GNN (2026) | 最近邻。纯重建训练（masked NLL）+车道级固定输入规则；其 §7.4 自列 "decision-aware or joint uncertainty / phase-level uncertainty" 为 future work → 本研究正是沿此方向的受控验证 |
| MissLight (IJCAI'23) | 路口级缺失 + RL；无概率/决策损失 |
| RobustLight (ICML'25) | 扩散修复 + RL；接口解耦但训练非决策感知 |
| DiffLight (NeurIPS'24) | 离线 RL + 扩散统一填补与决策（reward-conditioned）→ 是"填补×决策耦合"的另一种范式，但非 SMP、非决策感知重建损失、非相位风险选择 |
| MDRC (arXiv 2609.27528, 2026-09) | 元扩散状态恢复防御（DDIM+Reptile），恢复误差上界；下游是 RL 控制器；无相位级风险/决策感知训练 |
| UCATSC (arXiv 2602.07784) | 视觉部分可观测下的不确定性约束决策层（信念状态+反事实 rollout+安全约束）；非检测器缺失、非 SMP |
| CV-MP (2505.05258) | 部分网联车渗透率下的 MP；缺失模型不同 |
| ST-GIN / SATORIS-N | 交通数据填补（含 UQ），纯离线无控制 |

**定位声明**：本研究是对 LMP-GNN 显式遗留方向（decision-aware training + phase-level uncertainty）的受控实证跟进 + SUMO 平台化 + GRU 时序建模增项。不宣称新问题；若 E0 问题不成立或 H1–H4 被否定，如实记录。
