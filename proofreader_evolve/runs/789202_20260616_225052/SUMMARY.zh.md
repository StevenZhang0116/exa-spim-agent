# 进化运行总结 — `789202_20260616_225052`

- **运行 id：** `789202_20260616_225052`
- **大脑 id：** `789202`
- **代数：** 10
- **被接受：** 1（第 3 代）
- **held-out Edge Accuracy 净增益：** 基线 `71.45397` → 最终 `71.52989` = **+0.07593**（百分点）
- **智能体总成本：** **$59.10**

本次运行学到了一个单一改进 —— 用于 split-repair 合并的**距离相关共线性下限**
（distance-dependent colinearity floor），在第 3 代被接受。其余九代在 held-out
上的得分全部恰好为 +0.000 并被回退。净增益非常小（约 +0.076 个百分点的
Edge Accuracy）。

## 轨迹

held-out Edge Accuracy 是选择指标。父代基准线是上一个被接受策略的 held-out 得分；
只有当一代超过该基准线（并通过 no-new-merge / over-split 门控）时才会被接受。

| 代 | 尝试内容 | Train EdgeAcc | Held-out EdgeAcc | 相对父代 Δ | 结果 |
|----:|---------------|--------------:|-----------------:|------------:|--------|
| 1 | 在 split-repair 接受路径上加半径（口径）连续性 guard | 87.52734 | 71.45397 | +0.00000 | 已回退 |
| 2 | 图像 gap-connectivity guard（跨 gap 的荧光信号） | 87.52734 | 71.45397 | +0.00000 | 已回退 |
| 3 | **距离相关共线性下限**（0.94/0.97/0.992 对应 2/4/8 µm 区段） | 87.52734 | 71.52989 | **+0.07593** | **已接受** |
| 4 | 在接受评分上加节点度数（node-degree）惩罚 | 87.52734 | 71.52989 | +0.00000 | 已回退 |
| 5 | 新增远区段：8–15 µm 上下限 0.997 | 87.52734 | 71.52989 | +0.00000 | 已回退 |
| 6 | 用 branch-tangent（多边）共线性替代单边 | 87.52734 | 71.52989 | +0.00000 | 已回退 |
| 7 | 首次启用 `split_label`（MergeSite）merge-repair 路径 + 接受门控 | 87.52734 | 71.52989 | +0.00000 | 已回退 |
| 8 | `ENUM_PARAMS` 枚举杠杆（暴露更多 SplitSite） | 87.52734 | 71.52989 | +0.00000 | 已回退 |
| 9 | 组合翻转（`tip_to_shaft=False`，去掉 `split_max_sites`） | 87.52734 | 71.52989 | +0.00000 | 已回退 |
| 10 | proximity-override OR 分支（扩大召回的接受） | 87.52734 | 71.52989 | +0.00000 | 已回退 |

注：十行的 train Edge Accuracy 完全相同（`87.52734`）—— collector 对每一代报告
相同的 train primary。

## 学到的策略（最终被接受，第 3 代）

一个没有 `split_label` 的**共线 split-repair**（colinear split-repair）策略。对每个
候选 SplitSite，仅当跨 gap 的最坏情况共线性（`min(cos_in, cos_out)`，经
`_colinearity_score`）越过一个**随 gap 增大而收紧**的下限（`_required_cos`）时，
才合并两个 fragment 标签（`merge_labels`）。锚点：`gen03/heuristics.accepted.py`。

| 参数 | 取值 | 存在原因 |
|-----------|------:|---------------|
| `GAP_TIGHT_UM` | **2.0** | 极近区段的边界。 |
| `MIN_COLINEAR_COS` | **0.94** | `gap ≤ 2.0 µm` 的下限；最宽松，因为极近的配对几乎必然属于同一 neurite。 |
| `GAP_THRESHOLD_UM` | **4.0** | 中区段的边界。 |
| `COS_NEAR` | **0.97** | `2.0 < gap ≤ 4.0 µm` 的下限；比 seed 更紧，以剔除抬高 % Split Edges 并造成一次错误融合的边缘性中 gap 配对。 |
| `GAP_FAR_UM` | **8.0** | 新远区段的边界。 |
| `COS_FAR` | **0.992** | `4.0 < gap ≤ 8.0 µm` 的下限；仅允许近乎完美的延续，以桥接 seed 的 4 µm 盒子从未触及的更长的真实 gap（驱动增益的减少 omit 的合并）。超过 `GAP_FAR_UM` 则拒绝。 |
| `TANGENT_WALK_UM` | **6.0** | 用于共线性测试时测量局部切线方向的 arm 长度。 |

通俗地说：用一个倾斜边界取代 seed 的单一矩形接受盒子
（`gap ≤ 4 AND cos ≥ 0.94`），它**既移除**边缘性的中 gap 编辑，**又加入**
高精度的远 gap 编辑 —— 这是对所提议编辑集的真实改动，而非一个 fail-open 的过滤器。

## 失败教会了什么

change log 与 orchestrator 诊断都汇聚到一个教训：**在这个 held-out 集合上，得分
几乎不动，而只有加入真正高精度的合并才曾经撬动过它。** 具体而言：

- **第 1 代（半径连续性 guard）和第 2 代（图像 gap-connectivity guard）：**
  两者都是*fail-open 的接受侧过滤器* —— 在仅几何（geometry-only）的默认运行中
  它们从未触发，因此未改变所提议的编辑集（+0.000）。教训：一个只会拒绝的 guard
  若从不触发就无济于事；获胜的第 3 代规则则是在两个方向上*改变*编辑集。
- **第 4–6 代（度数惩罚、8–15 µm 远区段、branch-tangent 共线性）：**
  对第 3 代机制的各种变体，未产生 held-out 增量 —— 在一个低敏感度的 held-out
  集合上，运行已收敛到第 3 代的最优点。
- **第 7 代（`split_label` merge-repair）和第 8 代（`ENUM_PARAMS` 召回杠杆）：**
  这是首批结构上全新的尝试 —— 一种全新的编辑类型与一个更宽的候选流 —— 仍得 +0.000。
  fitness 增益完全由减少 omit 的边（split-repair）承载，而 merge-repair 没有找到
  净正收益的切割。
- **第 9–10 代（组合翻转、proximity-override OR 分支）：** 到这一步已连续九个候选
  得分恰为 +0.000；orchestrator 指出本次运行"在一个低敏感度 held-out 集合上收敛于
  gen3 最优点"，且"加入真实的 accept 是唯一曾经撬动得分的方向"。两个翻转都未对
  held-out 编辑集产生足以登记的改变。

## 注意事项（Caveats）

- **held-out 每一代都被复用于选择。** 最终的 `71.52989` 是一个*选择*指标，而非对
  泛化能力的无偏估计；它正是该循环直接优化所针对的得分。
- **增益很小。** 仅来自单独一代被接受的 +0.07593 个百分点 Edge Accuracy，其余九代
  全部平于 +0.000。此处 held-out 集合低敏感度：结构上截然不同的尝试（一种新编辑
  类型、一个更宽的候选流）完全未登记任何变化。
- **诊断文本是 orchestrator 旁白，而非 subagent 推理。** collector 的 `diagnosis`
  行是 orchestrator 在核验 subagent 的工作（"Let me verify the import..."、
  "The subagent made the change..."）。关于每个 guard *为何*存在的可信、平实的
  说明在 **`gen03/rules.accepted.md` 中的 Change log**，本总结即据此撰写。
- **没有 merge-repair 处于生效状态。** 尽管 harness 加入了 `split_label`
  MergeSite 路径（第 7 代），它被回退；被接受的策略只发出 `merge_labels`。

## 指引（相对于运行目录）

- 最终被接受的 heuristics：`gen03/heuristics.accepted.py`
- 最终被接受的 rules / change log：`gen03/rules.accepted.md`
- 账本（Ledger）：`ledger.jsonl`
- orchestrator 尝试日志：`attempts.md`
