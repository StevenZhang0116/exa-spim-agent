# AutoDiscovery 跨报告汇总摘要

## 来源报告与覆盖范围

| 报告（run） | 来源数据集 | 条目数 |
| --- | --- | --- |
| `ground-truth-error-annotations-revised-version_2026-06-17` | 789202 | 20 |
| `run-4--ground-truth-error-annotations-revised-version_2026-06-20` | 789202 | 20 |
| `run-5--ground-truth-error-annotations-revised-version_2026-06-25` | 794495 | 20 |

- 已纳入的报告数（`n_files`）：3
- 已纳入的条目总数（`n_entries_total`）：60
- 合并去重后的不同发现数：24
- 仅出现在单一报告中的发现数：12

### 综述

在这三次运行中，占主导且被反复印证的整体图景是一小组稳健的空间/拓扑误差特征：(1) **合并位点位于稠密、多分叉的邻域中**（局部 fragment 密度高、branch-node 密度高）——分别在 10 µm、15 µm、体积密度以及作为 branch-density ROC 特征下重新测量，且在每个脑上都 GENERALIZES；(2) **合并 segment 是巨大、过度生长、不对称的"失控"标签**——确认其 cable 比非合并 segment 大约 ~15–29×，与单个主神经元呈不对称重叠（中位数 1.0），且 cable length / node count 可作为一个廉价且可泛化的合并先验；(3) **split 误差集中在 branch points 附近以及纤细的远端/终端 tips 处，并在空间上聚集成"误差区"**——由许多距离/测地/拓扑检验发现，全部在方向上 GENERALIZING（尽管有几个是超大 n / 效应量微不足道的情形，在 neuron-cluster 校正下被 WEAKEN）；(4) **split gaps 极小（≈4–7 µm，~99% < 6.5 µm），而 inter-neuron gaps 更大，因此一个 ~6.5–6.84 µm 的邻近阈值加上角度/共线性先验可安全重连 ~85–86% 的 splits**；以及 (5) **omit 误差具有突发性/连续性，并集中于远端终端分支**，由所有脑上的 Markov-transition 和 run-length 分析确认。

最重要的**独有/新增**发现来自 run-4 和 run-5：run-4 的近乎完美的邻近分类器（ROC-AUC 0.9979，6.84 µm 阈值）、angular-inertia 延续启发式（ROC-AUC 0.9322）、对 86% 的 splits 的 A* 图修复，以及两个被驳斥的 Z-orientation/anisotropy 假设；run-5 全新的 crossing-fiber 合并几何（>45°）、双峰 split-gap GMM（现已正式经过模型选择）、不对称 merge-overlap 剪枝洞见、radius-matching 的 precision/recall 权衡，以及将 ~96% 的 omit gaps 归因于 100 µm min-cable-length 过滤器。

最清晰的**跨报告分歧**集中在 Z 轴效应以及 branch-order/distal 梯度上。run-4 的 "z-alignment / Z-dominant orientation 并非风险驱动因素"（#2/#3）带有 DOES-NOT-GENERALIZE / PARTIAL——同一检验在另外两个脑上变得在*相反*方向上强显著。run-4 的 "split risk 随 branch order 上升"（#139）被 OVERTURNED（符号翻转，neuron-cluster permutation 不显著）。2026-06-17 报告的 "omissions 偏远端 / 更靠近 leaves" 这一发现仅为 PARTIAL——在数据集 794491 上方向翻转——而类别型的 terminal-vs-internal omit-rate 重述则干净地 GENERALIZES。若干 "splits/omits 与 merges 共聚类" 以及 "splits 在 merges 附近级联" 的论断在按边来看是显著的，但一旦以 neuron 作为分析单位就被 OVERTURNED 或降级，而 run-5 的 split→omit 局部共现（#42）DOES-NOT-GENERALIZE（在两个额外脑上均符号翻转 + 不显著）。

---

## 独有与新增发现

### 1. (Priority 0.507 · Surprise 0.690) 一个 ~6.84 µm 的欧氏阈值可安全地自动重连绝大多数 split fragments。
- **来源：** run-4--...2026-06-20 (id 30) —— 唯一发现此邻近分类器结果的报告。
- **结论：** 检验了仅凭空间邻近性能否将真实 split gaps 与 inter-neuron gaps 区分开来。在 20 µm 范围内的 6,805 个真实 split gaps 与 4,189 个 inter-neuron gaps 上，分布几乎完全不重叠（真实 splits 峰值约 ~4.5 µm；inter-neuron gaps 很少低于 7 µm）。以 gap distance 作为二元分类器给出 ROC-AUC 0.9979，F1 最优阈值 6.84 µm 达到 F1 = 0.9945。较大的正向 surprisal（+0.690，Leaning False → Leaning True）反映出该 loop 此前倾向于反对仅靠邻近性修复，但数据强力验证了它是一个安全、有效的启发式。
- **沿用判定：** Reproduction REPRODUCED（精确匹配）；Generalization GENERALIZES（ds_794491 ROC-AUC=0.9889，thr=6.48 µm，F1=0.9711；ds_794495 ROC-AUC=0.9953，thr=6.83 µm，F1=0.9878）；Verdict SOUND。
- **为何独有/新增：** 未出现在 2026-06-17 报告中；首次出现在较新的 run-4 中，作为其排名第一、最高 surprise 的结果。（Run-5 的 id 63 印证了*gap-size*这一事实，已折叠进 Corroborated #4；ROC-AUC 分类器构建为 run-4 独有。）
- **Caveats：** 假阳性的 "inter-neuron gaps" 来自模拟中的 GT 标签而非真实合并，因此这是一个部署层面的注意事项；F1 最优阈值是在样本内选取的（轻微乐观偏差，鉴于 AUC=0.9979 可忽略）。

### 2. (Priority 0.409 · Surprise 0.471) 合并发生在 crossing fibers 处，其交叉角度显著大于 45°。
- **来源：** run-5--...2026-06-25 (id 15) —— 唯一发现此结果的报告；最新的运行。
- **结论：** 检验了一个合并位点处两个 GT 神经元之间的 3D 交叉角度是否超过 45°。在 12 个有效合并位点上，交叉角分布的均值为 65.97°、中位数为 74.93°（聚集在 65°–90°）；针对 45° 的 one-sample Wilcoxon signed-rank 检验给出 statistic 69.00，p = 0.0081，拒绝原假设。正向 surprisal（+0.471）是 run-5 中最大的信念偏移，将一个 Uncertain 先验提升至 Likely True——合并主要是一个 crossing-fiber 几何问题。
- **沿用判定：** Reproduction REPRODUCED（精确匹配）；Generalization PARTIAL（ds_794491 中位数 54.30°，stat=223.00，p=5.37e-02，n=25 —— 方向相同但失去显著性；ds_789202 产生 0 个有效合并位点，无法检验）；Verdict WEAK。
- **为何独有/新增：** 首次出现在最新的 run-5；在两个更早的报告中不存在任何 Z-angle/crossing-fiber 合并几何假设。
- **Caveats：** 严重检验力不足（n=12）；45° 基准是一个任意的参照；两个额外数据集中只有一个产出了可检验的位点，且那一个也只是边缘显著（p≈0.054）。

### 3. (Priority 0.275 · Surprise 0.300) Split gaps 呈双峰：极小的单点 dropout gaps 加上较大的遗漏段 gaps。
- **来源：** run-5--...2026-06-25 (id 12) —— 唯一对 split-gap 模态进行建模的报告。
- **结论：** 在 7,988 个 split gaps 上的 2-component Gaussian Mixture Model 发现 Component 1（占 splits 的 ~63.6%）均值约 ~0.0 µm（位移可忽略），Component 2（~36.4%）均值约 ~159.2 µm（std ~259.0 µm，大段遗漏），意味着连接器应使用两个搜索半径。校正后的模型选择检验正式确认了双峰性：BIC(k=1)−BIC(k=2)=114,816.7，likelihood-ratio 114,843.7，Hartigan dip statistic=0.4128，bootstrap p=0.0020。
- **沿用判定：** Reproduction REPRODUCED（从一次记录在案的 sandbox FAILED 运行中恢复）；Generalization GENERALIZES（dip test 在两个额外脑上均拒绝单峰性，p=0.0020；large-gap 均值在 117–329 µm 之间变化）；Verdict WEAK → Post-correction verdict UPHELD。
- **为何独有/新增：** 首次出现在最新的 run-5；更早的报告均未对 split-gap 距离拟合混合模型。
- **Caveats：** GMM component 分配依赖于模型；Component 1 均值约 ~0.0 µm、std ~0.001 µm 可能是一个近乎退化的尖峰。原始分析未报告模型选择检验（校正前双峰性是由构造方式所断言的）。

### 4. (Priority 0.275 · Surprise 0.300) 合并的 segments 高度不对称——大多是单个主神经元的次要分支（overlap 中位数 1.0）。
- **来源：** run-5--...2026-06-25 (id 22) —— 唯一测量合并重叠不对称性的报告。
- **结论：** 在 98 个合并 segments 上，与主神经元的平均 overlap ratio 为 0.9129（中位数 1.0）；检验均值超过 0.5 的 one-sample 检验给出 t = 24.87，p = 3.58e-44。均衡的 50/50 融合极为罕见，因此合并校正是在剪除一个次要的偶然分支，而非分割一个均衡的 segment。校正后的非参数检验（Wilcoxon W=4656.0，p=1.87e-19；sign test p=1.26e-29；中位数 1.0，bootstrap CI [1.0, 1.0]）确认了相同结论。
- **沿用判定：** Reproduction REPRODUCED；Generalization GENERALIZES（ds_794491 均值 0.8558，t=17.18，p=1.72e-31；ds_789202 均值 0.9704，t=43.12，p=8.84e-49）；Verdict MINOR → Post-correction verdict UPHELD。
- **为何独有/新增：** 首次出现在最新的 run-5；这是对合并修复的一个可操作表述（"剪枝，而非分割"），在两个更早的报告中不存在。
- **Caveats：** 头条结论的来源缓存为单一数据集；原始 one-sample t-test 对一个堆积在 1.0 上限处的 [0,1] 有界比率而言并不合适（检验选择错误，已在不改变结论的情况下校正）。

### 5. (Priority 0.275 · Surprise 0.300) radius-matching 约束将假阳性合并削减约 ~56%，但使修复 split 的能力减半。
- **来源：** run-5--...2026-06-25 (id 47) —— 唯一检验此 thickness-constraint 权衡的报告。
- **结论：** 在 < 15 µm 邻近 split-correction 启发式上增加一个叶节点 `node_radius` 差异 < 25% 的约束，将假阳性合并减少了 55.63%（151→67 对，超过 40% 的目标），但仅保留了基线 split-resolving 能力的 43.95%（1,513→665 个真实 splits 被解决）。这是一个 precision/recall 权衡，而非免费收益。
- **沿用判定：** Reproduction REPRODUCED；Generalization GENERALIZES（ds_794491 47→15 对 = 减少 68.09%，保留 37.14%；ds_789202 22→7 对 = 减少 68.18%，保留 35.87%）；Verdict WEAK。
- **为何独有/新增：** 首次出现在最新的 run-5；这是更早报告中不存在的一项明确的安全性/效力权衡研究。
- **Caveats：** 没有推断统计量或 CI；主要指标依赖于很小的对数（151/67，下降至 47/15 和 22/7）。次级 "canonical merges" 指标明显是噪声（origin 33.33%，794491 −100.00%，789202 200.00%），已被搁置。

### 6. (Priority 0.275 · Surprise 0.300) ~96% 的 omit gaps 低于 100 µm min-cable-length 过滤器——主导 omit 的驱动因素是工具的短 fragment 过滤器，而非 U-Net 失败。
- **来源：** run-5--...2026-06-25 (id 69) —— 唯一将 omits 归因于构建期过滤器的报告。
- **结论：** 在 4,145 段连续遗漏段（总计 122,590 µm）中，3,971 段（95.80%）短于 100 µm，占总遗漏 cable 的 85,498 µm（69.74%）。这表明缓存中可配置的 `min_cable_length` 预处理过滤器——而非大尺度分割失败——是主导的 omit 驱动因素，对 pipeline 配置有直接影响。
- **沿用判定：** Reproduction REPRODUCED；Generalization GENERALIZES（ds_794491 98.62% 的段 / 91.38% 的 cable；ds_789202 91.73% / 55.94%）；Verdict MINOR。
- **为何独有/新增：** 首次出现在最新的 run-5；这是一个 pipeline 配置层面的洞见（相对于模型行为），在更早的报告中不存在。
- **Caveats：** 因果归因过度——100 µm 截断本身就是数据集自身的构建参数，因此发现大多数 gaps 低于它部分上是循环论证的；该实验测量的是 gap-length 比例，而非这些 gaps 在其他情况下是否会被重建。

### 7. (Priority 0.265 · Surprise 0.414) split-error 风险随 centrifugal branch order（距 soma 的拓扑深度）上升——OVERTURNED。
- **来源：** run-4--...2026-06-20 (id 139) —— 唯一检验 branch order vs split risk 的报告。
- **结论：** 在 1.4M 条边上的 Logistic regression 发现 branch order 是 split 误差的一个显著正向预测因子（coef = 0.0194，p < 0.001；split rates 在 order 53 附近飙升至 >3%）。正向 surprisal（+0.414）反映了该 loop 向上修正。然而校正显示该效应是一个 artifact：cluster-robust z 从 16.4 塌缩至 1.990（p = 0.04662），且 neuron-level permutation Spearman rho = 0.2308 不显著（p = 0.4697）。
- **沿用判定：** Reproduction REPRODUCED；Generalization DOES-NOT-GENERALIZE（coef 在 ds_794491 上翻转为 −0.0157、在 ds_794495 上为 −0.0063，二者均在*相反*方向显著）；Verdict MAJOR → Post-correction verdict OVERTURNED。
- **为何独有/新增：** 只有 run-4 检验了 centrifugal branch order；首次也是唯一一次出现。
- **Caveats：** `norm_thickness` 协变量方差为零并被剔除，因此 "独立于 thickness" 的条款在结构上无法检验；edge 非独立性夸大了原始 p；在任何脑上，该结论在正确的 neuron-level 检验下都不成立。

### 8. (Priority 0.253 · Surprise 0.284) Angular inertia（~153° vs ~90°）能可靠识别跨越 split 的真实延续方向。
- **来源：** run-4--...2026-06-20 (id 33) —— 唯一将跨 split 的方向延续作为分类器来检验的报告。
- **结论：** 在 13,582 个 split 配置中，真实延续平均为 152.96°，而假候选为 90.17°；分布是区分明显的（KS = 0.746，p ≈ 0），角度对齐以 ROC-AUC 0.9322 进行判别。方向惯性是 agentic proofreader 桥接 splits 的一个强局部启发式。
- **沿用判定：** Reproduction REPRODUCED（精确匹配）；Generalization GENERALIZES（ds_794491 ROC-AUC=0.9281；ds_794495 ROC-AUC=0.9365）；Verdict SOUND。
- **为何独有/新增：** 只有 run-4 将延续角度构造为 ROC 分类器。（与 Corroborated #11 中 2026-06-17 的 endpoint-cosine 发现相关但不同，后者测量的是 anti-parallel 端点对齐，而非 continuation-vs-false-candidate 判别。）
- **Caveats：** 两个角度样本按 split node 配对，而 KS 假设独立性（在 KS=0.746 时无关紧要）；假候选是来自另一神经元的最近节点，是一个合理的保守比较对象。

### 9. (Priority 0.253 · Surprise 0.284) 在 fragments 图上的 A* path-finding 修复了 86% 的 splits 且未引入合并。
- **来源：** run-4--...2026-06-20 (id 39) —— 唯一检验 graph A* split 修复的报告。
- **结论：** 对急剧偏折（>45°）和半径突变施加惩罚的 Graph A* 搜索，连接了 6,805 个目标 splits 中的 5,881 个（86.42% 成功率，远超 40% 阈值），且未跨入不同的 GT 神经元；模拟修复将 edge accuracy 从 78.71% 提升至 79.13%（+0.42%）。
- **沿用判定：** Reproduction REPRODUCED（精确匹配）；Generalization GENERALIZES（ds_794491 84.72%；ds_794495 89.71%，始终为 ~85–90%）；Verdict MINOR。
- **为何独有/新增：** 只有 run-4 实现了 A* 修复模拟；其他报告中不存在。（注意：此 id 39 是 run-4 的 A* 修复，与 2026-06-17 的 id 39 不同，后者是一个 KDTree-sever 合并解决启发式——见 Corroborated #12。）
- **Caveats：** "未引入合并" 是在模拟内部针对 GT 标签裁定的，而非真实分割，因此对部署而言该无合并保证被过度声称；5,000-node 的 A* 扩展上限和 10 µm 的 merge-validity radius 是任意的；净 accuracy 增益（+0.42%）较为微小。

### 10. (Priority 0.253 · Surprise 0.284) omit 误差在极端 Z 深度处比在中心深度处发生概率约高 ~2.4×——OVERTURNED。
- **来源：** run-4--...2026-06-20 (id 45) —— 唯一检验 axial-depth（位置）omit 过量的报告。
- **结论：** 极端 Z 节点（顶部/底部 10%）的 omit rate 为 4.44%（2,283/51,369），而中心为 1.94%（11,342/585,909）；Cochran-Mantel-Haenszel 检验给出合并 OR 2.3561，p ≈ 0。正向 surprisal（+0.284）暗示轴向极端处的信号丢失。校正将其推翻：neuron-cluster permutation OR = 0.5320，p = 0.6607（不显著）。
- **沿用判定：** Reproduction REPRODUCED；Generalization PARTIAL（ds_794495 OR=3.1672，p≈0，更强；ds_794491 OR=1.0149，p=7.996e-01，消失）；Verdict MAJOR → Post-correction verdict OVERTURNED。
- **为何独有/新增：** 只有 run-4 检验了 Z *深度/位置*（与 Corroborated #13 中的 Z *orientation* 假设不同）。首次也是唯一一次出现。
- **Caveats：** CMH 的 "控制脑" 是虚幻的（每个 origin pkl 只有一个脑）；"extreme Z" 是按体积 min-max 归一化的，混淆了光学深度与 FOV 边界 artifacts；node 非独立性夸大了原始 p。

### 11. (Priority 0.253 · Surprise 0.284) 短 omission gaps 是被同一 segment 桥接的内部 dropouts；长 gaps 是真实的终止。
- **来源：** run-4--...2026-06-20 (id 58) —— 唯一将 omits 划分为 bridged 与 broken 两类的报告。
- **结论：** 在所分析的 omit paths 中，307 个为 "bridged"（两侧 flanking segment 相同），4,298 个为 "broken"；bridged gaps 远短于（mean 18.71 µm，median 13.55 µm）broken gaps（mean 42.54 µm，median 20.16 µm），由 one-sided Mann-Whitney U 检验显著（p = 1.95e-19）。短 omits 大多是 artifactual 的内部 dropouts。
- **沿用判定：** Reproduction REPRODUCED（精确匹配）；Generalization GENERALIZES（ds_794491 bridged 10.99 µm vs broken 15.23 µm，p=2.75e-09；ds_794495 12.17 µm vs 16.24 µm，p=1.20e-12）；Verdict SOUND。
- **为何独有/新增：** 只有 run-4 区分了 bridged 与 broken omit paths；其他地方不存在。
- **Caveats：** Bridged paths（n=307，~7% 的 omit paths）是一个不大的少数；分析单位是一个连通的 omit-path 组件（确实独立），因此 MW 独立性成立。

### 12. (Priority 0.324 · Surprise 0.351) split-gap 配对遵循一个非线性的 distance-vs-angle 权衡（仅 origin）——不泛化。
- **来源：** ...2026-06-17 (id 47) —— 唯一拟合 distance×angle 交互作用的报告。
- **结论：** 在 1,072 个 true-split 正例 vs 142 个 false-merge 负例上的 Logistic regression 发现了一个显著的 distance × angle 交互作用（coef = −5.96，p = 0.014；LLR p = 1.626e-17）：短 gaps（< 8 µm）容许急转，而长 gaps（~15 µm）要求共线性。在 origin 脑上，permutation test（p = 3.5964e-02）和 stratified bootstrap CI [−12.3429, −1.1075] 保持了该交互作用显著。
- **沿用判定：** Reproduction REPRODUCED；Generalization DOES-NOT-GENERALIZE（交互作用塌缩：ds_794491 x3=−0.3416，p=0.800；ds_794495 x3=−1.4540，p=0.565，拟合未收敛）；Verdict MAJOR → Post-correction verdict UPHELD on origin，Corrected generalization DOES-NOT-GENERALIZE。
- **为何独有/新增：** 只有 2026-06-17 报告拟合了此交互模型。（其底层工程动机——固定半径连接不够充分——在 Corroborated #6 中被单独印证。）
- **Caveats：** 头条结论依赖于一个在严重类不平衡（1072:142）下的单一边缘交互系数（p=0.014）；pseudo-R² 仅 0.093；µm 阈值是从拟合边界读取的，未经独立验证；该非线性是 origin 脑所特有的。

---

## 已印证发现（合并整理）

### 13. (Priority 0.323 · Surprise 0.795) Z 轴 neurite orientation/anisotropy 并非拓扑误差的一致驱动因素——数据集特定的零结果。
- **来源：** run-4--...2026-06-20 (id 27，Z-alignment mixed-effects logistic；id 21，Z-dominant-edge chi-square) —— 2 条目，1 份报告。
- **结论：** run-4 检验了 anisotropy 假设的两种形式。Z-alignment：在 20,000 条边子集上的 Bayesian mixed-effects logistic 给出 coef = −0.066，p = 0.058，OR 0.80（不显著，略为负）；强烈的负向 surprisal（−0.795）标记了一个被数据所否定、却曾被笃信的假设。Z-dominant edges：误差率几乎相同（3.70% Z-dom vs 3.63% XY-dom；χ² = 3.53，p = 0.0602）。校正后的 neuron-cluster permutation 检验确认在 origin 上没有可检测到的效应（z-alignment p = 0.5721；Z-dominant cluster-permutation p = 0.8136，RR≈1.02）。
- **一致之处：** 两个条目在 origin 上得出相同结论——相对于 anisotropic 轴的取向在 origin 脑上不是一个实质性的误差驱动因素。
- **分歧之处：** 二者都不泛化，且方式互相冲突。对于 Z-dominant edges（id 21），同一检验在额外脑上变得在*相反*方向上强显著：ds_794491 χ²=166.1，p=5.21e-38（Z 更高）；ds_794495 χ²=395.8，p=4.62e-88（Z 更低）。对于 z-alignment（id 27），系数符号不稳定（−/+/−），且 ds_794495 显著（p=6.75e-06，一个下降）。因此 "无 anisotropy 效应" 这一零结果是数据集特定的，而非一个稳定的属性。
- **沿用判定：** Reproduction REPRODUCED（两者，精确匹配）；Verdict MAJOR（两者）；Post-correction verdict UPHELD 作为 origin 上的零结果，但 Generalization DOES-NOT-GENERALIZE（id 21）/ PARTIAL（id 27，id 21 校正后）。
- **Caveats：** 两个原始分析都犯了 absence-of-evidence/evidence-of-absence 谬误；id 27 报告的 p 构造有误（VB 后验被当作 Wald z），且在 20k 子样本上拟合；edge 非独立性夸大了二者；应重述为 "无可检测效应"，而非证明其不存在。

### 14. (Priority 0.287 · Surprise 0.307) 合并位点所处的局部 neurite 邻域显著比正确/对照区域更稠密。
- **来源：** ...2026-06-17 (id 3，10 µm radius；id 23，15 µm radius；id 81，volumetric nodes/µm³；id 49，distance to fragment branch point；id 53，branch-node density per 100 µm)，run-4--...2026-06-20 (id 84，local branch density ROC) —— 6 条目，2 份报告。
- **结论（canonical = id 3，SOUND，最高置信度）：** 合并位点在 10 µm radius 内平均有 7.03 个 fragment nodes，而对照为 4.39，由 Mann-Whitney U 高度显著（U = 3931.0，p = 9.0e-15，n=67/67）。同样的拥挤特征在 15 µm 处重现（11.09 vs 6.33 nodes，U=4100.5，p=8.95e-17 —— id 23），表现为体积密度（0.001678 vs 0.001083 nodes/µm³，Welch t=8.93，p=2.8e-14 —— id 81）、表现为到 fragment branch points 的邻近度（median 4.48 µm vs 179.76 µm，p=2.2e-17 —— id 49）、表现为每段 cable 的 branch-node 密度（0.1078 vs 0.0568 branches/100 µm，~1.9×，p=2.9e-25 —— id 53），以及表现为一个 local-branch-density 分类器（1.13 vs 0.04 branches/15 µm，paired t=13.35，ROC-AUC=0.9236 —— run-4 id 84）。
- **一致之处：** 全部六个都同意合并位点位于拥挤、多分叉的 neuropil 中，且全部 2-for-2 GENERALIZE（如 id 3：794491 p=2.3e-17，794495 p=1.1e-16；id 53 效应在额外脑上增长至 ~3.0×；run-4 id 84 ROC-AUC 在各脑间 0.88–0.92）。基于密度的形式（id 3/23/81）和基于 branch-density 的形式（id 49/53/84）是同一个 "稠密、多分叉环境导致合并" 论断的不同测量方式。
- **分歧之处：** 无——全部为正，全部 GENERALIZE。
- **沿用判定：** 该 cluster 中的最佳/最保守者—— Verdict SOUND（id 3，id 23，id 53）；id 81 WEAK（参数检验选错）→ Post-correction UPHELD；id 49 MINOR（对照未设种子）；run-4 id 84 MINOR（样本内 AUC）。
- **Caveats：** 多数情形下正类样本很小（n≈64–105 个合并位点/segments）；id 81 对计数数据使用了 t-test（已校正为 MW，结论不变）；id 49 的对照未设种子并被限制为非 branch 节点，夸大了差距（rerun 时基线值 DIVERGED，结论成立）；id 84 的 ROC-AUC 是样本内的。

### 15. (Priority 0.287 · Surprise 0.307) 合并 segments 是巨大的 "失控" 过度生长标签，比非合并 segment 大一个数量级。
- **来源：** run-4--...2026-06-20 (id 43，log-cable Welch t)，run-5--...2026-06-25 (id 34，cable ratio + ROC；id 55，corrected-mapping cable MWU；id 56，node count + cable MWU)，以及 id 24（super-merge cable，已推翻）—— 5 条目，2 份报告。
- **结论（canonical = run-5 id 56 / run-4 id 43，SOUND）：** 引起合并的 segments 覆盖的 cable 远多于非合并 segment：run-4 id 43 发现 64 个合并 segments 平均 ~15,449 µm（median 3,099 µm），而 8,273 个非合并 segment 为 ~532 µm（median 102 µm）（~29× 均值，log-lengths 上 Welch t = 16.54，p = 7.32e-25）。run-5 以校正映射确认：id 34（98 个合并 segments 平均 19,040 µm vs 1,221 µm，15.6× ratio，ROC-AUC=0.869）、id 55（median 4,604.78 µm vs 160.14 µm，U=3.53e7，p=0.0）、id 56（median 791 vs 51 nodes，U=368,540，p=1.04e-38）。Cable length / node count 是一个廉价、灵敏的合并先验。
- **一致之处：** 全部汇聚于 "合并是大规模过度生长的标签，而非局部小波动"，在所有脑上 GENERALIZING（id 43 在全部三个脑上 t≥16；id 34 ratios 8.3–17.6×；id 56 p≤1.04e-38）。id 24（super-merges ≥3 神经元覆盖的 cable 多于 2-neuron merges）是同一主题的一个更细的子论断。
- **分歧之处：** id 24 的*显著性*未能成立：它在 rerun 时 DIVERGED（p 0.0059 → 0.2222），并在 per-neuron 构造下被 OVERTURNED（U=8.0，p=0.2222，n_super=1）；一般性的 "合并很巨大" 论断不受影响，但具体的 super-merge-vs-2-neuron 比较样本太少，无法下结论。
- **沿用判定：** Verdict SOUND（id 43，id 34，id 55，id 56）；id 24 MAJOR → Post-correction verdict OVERTURNED。
- **Caveats：** 合并类别样本小（n=64–98）且右偏（均值对离群值敏感；中位数更稳健）；run-5 的非合并基线依赖于 fragment-exclusion/mapping 的选择；id 24 依赖于 n=1–3 个 super-merges。

### 16. (Priority 0.287 · Surprise 0.307) split 误差集中在 branch points 处/附近（split rate 升高，且 split edges 更靠近 branch nodes）。
- **来源：** ...2026-06-17 (id 10，branch-edge split rate ~3.4×；id 13，geodesic distance to branch；id 19，µm distance to GT branch；id 72，topological distance to GT branch)，run-4--...2026-06-20 (id 64，geodesic distance to branch)，run-5--...2026-06-25 (id 3，≤15 µm branch-proximal RR=1.59；id 33，distance to branch node) —— 7 条目，3 份报告。
- **结论（canonical = ...2026-06-17 id 10，SOUND）：** Branching edges 的 split 远多于 linear edges—— 1.62%（248/15,298）vs 0.47%（6,557/1,393,747），~3.4×，χ² = 414.5，p = 3.9e-92。连续距离形式与之一致：split edges 在测地上更靠近 branch points（id 13/72：median ~196 vs 380 µm），在 µm 上（id 19：median 121 vs 222 µm），且 run-5 的 branch-proximal edges 带有 RR=1.59（0.90% vs 0.57%，p=4.90e-34）。所有三份报告独立地重新发现 bifurcations 是系统性的 split-failure 位点。
- **一致之处：** 所有条目在每个脑上方向都 GENERALIZES（id 10 ratio 衰减至 ~2× 但保持 p≤3.5e-29；run-5 id 3 RR 在额外脑上增长至 2.58–3.24×；id 19/72 保持 2-for-2）。
- **分歧之处：** 方向上无分歧。幅度/稳健性有差异：超大 n 的距离形式（id 13、id 72、run-4 id 64、run-5 id 33）在校正下被标记为 effect-size-trivial—— run-4 id 64 WEAKENED（origin 上 neuron-cluster permutation p=0.096，额外脑上 PARTIAL），run-5 id 33 WEAKENED（Cliff's delta 0.07，origin/794491 上 cluster CI 包含 0）。类别型 rate 形式（id 10，run-5 id 3）保持 SOUND。
- **沿用判定：** 最佳 = SOUND（id 10，id 19，id 72，run-5 id 3）；较弱 = MINOR（id 13 —— 记录在案的 double-count bug；run-4 id 64 → WEAKENED）；run-5 id 33 WEAK → WEAKENED。
- **Caveats：** Edge pseudo-replication 夸大了 per-edge p-values；id 13 和 id 65（见 #18）有一个记录在案的 double-counting bug（DIVERGED，结论完好）；距离效应在 794491 上显著缩小。

### 17. (Priority 0.287 · Surprise 0.307) split 误差在空间上聚集成局部化的 "误差区"（一个 split 提高了附近出现 splits 的几率）。
- **来源：** ...2026-06-17 (id 55，KS vs random null)，run-4--...2026-06-20 (id 37，30 µm 内 split-neighbors)，run-5--...2026-06-25 (id 26，Monte Carlo NN distance；id 40，KS NN distance；id 45，Monte Carlo NN replication) —— 5 条目，3 份报告。
- **结论（canonical = run-5 id 26，SOUND）：** 观测到的 inter-split nearest-neighbor 距离在每个脑上都远低于随机零模型—— run-5 id 26：观测 140.30 µm vs 随机 320.79 µm（paired t=−15.10，p=5.76e-12，n=19 神经元）。2026-06-17 id 55（观测 median 25.58 µm vs 随机 236.98 µm，KS D=0.4539）、run-4 id 37（30 µm 内 0.97 vs 0.10 个 split-neighbors）以及 run-5 id 40（KS=0.5635）/ id 45（Monte Carlo replication，t=−14.93）全部确认 splits 是聚集而非散布的。
- **一致之处：** 全部五个在方向上 GENERALIZE（观测值 ≈ 各脑随机值的一半）；run-4 id 37 在 neuron-cluster permutation 下 UPHELD（p=0.0002，Cliff's delta 0.49）。
- **分歧之处：** 无——普遍为正且稳健。Run-5 id 45 明确是 id 26 在相同神经元上的复现。
- **沿用判定：** Verdict SOUND（run-5 id 26，id 40，id 45）；...2026-06-17 id 55 MINOR（因果 "cascade" 措辞过度解读了关联）；run-4 id 37 MINOR → Post-correction UPHELD。
- **Caveats：** "Cascade"/"一个 split 导致另一个" 是因果过度声称——检验表明的是聚集性，这与一个共享的潜在原因（局部图像质量）相符；split edges 非独立（已由 run-4 id 37 的 neuron-cluster 校正处理）。

### 18. (Priority 0.287 · Surprise 0.307) split 误差偏好纤细的远端 / 终端 processes（distance-to-leaf 比正确 edges 更短）。
- **来源：** ...2026-06-17 (id 60，mean distance-to-leaf 939 vs 1283 µm；id 65，median 417 vs 670 µm thickness proxy)，run-5--...2026-06-25 (id 38，terminal vs internal split rate；id 70，monotonic split rate vs distance-to-leaf) —— 4 条目，2 份报告。
- **结论（canonical = ...2026-06-17 id 60，SOUND）：** split edges 比正确 edges 更靠近 terminal leaves—— mean 938.60 µm vs 1,282.69 µm（p=2.2e-134，n=6,805/1,109,034）；id 65 将其重述为 thickness proxy（median 417.22 vs 670.25 µm）。run-5 确认 terminal compartments 比 internal 更易 split（id 38：0.65% terminal vs 0.54% internal，paired t=3.93，p=9.71e-04），且 split rate 随到 leaf tip 的距离单调下降（id 70：< 50 µm 时 0.92% → > 200 µm 时 0.53%，logistic coef=−0.1361，p<0.001）。
- **一致之处：** 方向在各条目间 GENERALIZES（id 60 保持 2-for-2；run-5 id 38 在两个额外脑上同方向 p<0.05）。
- **分歧之处：** 幅度/稳健性有差异。Run-5 id 70 为 PARTIAL—— 正式 logistic 斜率在 ds_794491 上不显著（coef=−0.0095，p=0.409）且校正后 WEAKENED（origin 上 GEE z 从 16.4→2.87），尽管 per-neuron Spearman 的下降趋势在各处都一致。id 65 有一个记录在案的 double-counting bug（DIVERGED，结论完好）。
- **沿用判定：** 最佳 = SOUND（id 60）；id 65 MINOR（double-count bug + 间接 thickness proxy）；run-5 id 38 WEAK（绝对差距极小 ~0.1 pp）；run-5 id 70 MAJOR → Post-correction WEAKENED，PARTIAL generalization。
- **Caveats：** distance-to-leaf 是口径的一个*间接*proxy，而非测量得到的 radius；超大 n 夸大显著性（应读中位数/效应量）；run-5 id 70 预期的 GEE 失败并回退到了 pooled logit，错误处理了 per-neuron 非独立性。

### 19. (Priority 0.287 · Surprise 0.307) omit 误差具有突发性/连续性——集中在连续段中，而非孤立的被丢弃 edges。
- **来源：** ...2026-06-17 (id 48，Markov transition，89% conditional / 28× ratio)，run-5--...2026-06-25 (id 27，run-length transition matrix；id 50，100% zero-distance omit neighbor) —— 3 条目，2 份报告。
- **结论（canonical = run-5 id 27，SOUND）：** omits 以黏连的 runs 出现。run-5 id 27：基线 omit 概率 2.23% 在以相邻 omit 为条件时跃升至 84.55%（Wilcoxon W=0.0，p=3.81e-06；Omit→Omit transition 0.849，mean run length 6.91 edges，max 339）。2026-06-17 id 48 发现相同结果，marginal 3.17% 上升至 conditional 89.08%（ratio 28.09，≫ 假设的 3×）。run-5 id 50（100% 的 omit edges 都有一个 zero-distance omit neighbor）是退化但已校正的版本，经一次 within-neuron permutation UPHELD，显示出超越 run-structure artifact 的真实聚集性（mean per-neuron gap −113.21 µm，p=4.9998e-05）。
- **一致之处：** 全部 GENERALIZE（id 48 在额外脑上 ratio 18.6–40.4×；id 27 在所有脑上 conditional ~80–87%；id 50 校正后聚集性在全部三个脑上显著）。
- **分歧之处：** 结论无分歧。
- **沿用判定：** Verdict SOUND（run-5 id 27）；...2026-06-17 id 48 MINOR（pseudo-replicated χ²）→ Post-correction UPHELD（PARTIAL：校正后的 permutation 仅在 794491 上完成，p=9.99e-04；origin/794495 超时但描述性 ratios 巨大）；run-5 id 50 MAJOR（退化的 point-mass MWU）→ Post-correction UPHELD。
- **Caveats：** 高条件概率对于发生在连通 runs 中的误差而言部分上是定义性的；校正后的检验（edge/neuron permutation）确认了超越该 artifact 的聚集性；id 48 的原始 χ² 是 pseudo-replicated 的，其校正后的 permutation 在两个脑上超时。

### 20. (Priority 0.287 · Surprise 0.307) omit 误差在 branch points 处 / 在 terminal（远端）分支上比在 linear/internal cable 上更频繁。
- **来源：** ...2026-06-17 (id 11，topological distance-to-leaf；id 64，terminal vs internal omit rate)，run-4--...2026-06-20 (id 32，branch vs linear node omit rate；id 85，terminal vs internal omit rate) —— 4 条目，2 份报告。
- **结论（canonical = run-4 id 32 / ...2026-06-17 id 64，SOUND/MINOR）：** omits 集中在复杂/远端拓扑处。Branch nodes 的 omit rate 约 ~11.95%，而 linear nodes 约 ~2.76%（>4×，χ²=1567.69，p<0.0001 —— run-4 id 32）。Terminal edges 的 omit rate 为 4.38% vs internal 2.41%（~1.8×，χ²=4189.94，p≈0 —— id 64 和 run-4 id 85，数字相同）。id 11 发现 omit edges 在拓扑上更靠近 leaves（mean 247.18 vs 325.08 steps）。
- **一致之处：** 类别型 rate 形式（id 32，id 64，run-4 id 85）干净地 GENERALIZE（terminal/branch omit rate 在所有脑上更高；ratios 衰减至 ~1.3–3.8× 但保持高度显著）。
- **分歧之处：** id 11（连续 distance-to-leaf）为 PARTIAL—— 方向在 ds_794491 上*翻转*（omit mean 159.76 > correct 138.76，one-sided p=1.0），尽管它在 origin 和 794495 上成立。因此 "omits 偏远端" 论断作为类别型 terminal-vs-internal rate 是稳健的，但作为连续 distance-to-leaf 指标则不稳定。
- **沿用判定：** 最佳 = SOUND（id 64）；run-4 id 32 MINOR，run-4 id 85 MINOR；id 11 WEAK（超大 n、pseudo-replication、在 794491 上方向翻转）。
- **Caveats：** Edge/node pseudo-replication 夸大了类别型 p-values（在这些效应量下无关紧要）；id 11 的距离是拓扑步数，而非微米，且在一个额外脑上出现完全的方向翻转。

### 21. (Priority 0.287 · Surprise 0.307) 每个 neuron 的 split rates 与 omit rates 呈正相关（一种共享的失败倾向）。
- **来源：** ...2026-06-17 (id 30) —— 1 份报告，但此处作为一个跨数据集已印证的发现而非独有发现列出，因为它在两个额外脑上都强烈 GENERALIZES。
- **结论：** 在 12 个神经元上，splits/mm 预测 omit rate，Pearson r = 0.65（p = 0.022），Spearman ρ = 0.88（p = 0.00015），OLS R² = 42.2%。校正后的 Spearman permutation（p = 4.5999e-04，bootstrap CI [0.5620, 0.9857]）稳健。在连续性上失败的神经元也倾向于完全遗漏结构。
- **一致之处：** GENERALIZES—— ds_794491 ρ=0.9833（p=3.9999e-05），ds_794495 ρ=0.7404（p=4.7999e-04）；边缘显著的 origin Pearson 在两个额外脑上显著增强。
- **分歧之处：** 在本次运行内的各数据集间无分歧；没有其他报告检验过 split–omit 的 per-neuron 相关性，因此它同样可以归入独有发现。之所以放在此处，是因为其多数据集印证才是要点所在。
- **沿用判定：** Reproduction REPRODUCED；Generalization GENERALIZES；Verdict WEAK → Post-correction verdict UPHELD。
- **Caveats：** 仅 12 个神经元；原始 Pearson/OLS 违反了正态性（Jarque-Bera p=0.00036）且存在一个高杠杆点—— Spearman 才是可信的头条；"共享机制" 的措辞是相关性无法分离出来的解释性过度声称。

### 22. (Priority 0.287 · Surprise 0.307) inter-segment split gaps 极小（≈4–20 µm；~99% 低于 6.5 µm），远超内部 fragment edge lengths——促使采用紧凑的修复搜索半径。
- **来源：** ...2026-06-17 (id 35，median gap 19.7 µm vs 95th-pct internal 5.8 µm)，run-4--...2026-06-20 (id 63，100% 的 split gaps < 15 µm，99th ~6.5 µm)，run-5--...2026-06-25 (id 35，86.77% 的 splits 涉及一个 sub-100 µm micro-fragment) —— 3 条目，3 份报告。
- **结论（canonical = run-4 id 63，SOUND）：** split gaps 一律很小——全部 6,805 个 split gaps < 15 µm，聚集在 3–5 µm，95th/99th 百分位为 ~5.85 µm 和 ~6.50 µm，因此一个 ~6.5 µm 的搜索半径能捕获 ~99% 的 splits。2026-06-17 id 35 以对比方式表述同一事实：median inter-segment gap 19.67 µm vs 95th-pct intra-segment edge length 5.77 µm（p≈0），因此一个宽到足以容纳真实 gaps 的半径会远超内部 edges，单凭固定半径会过度连接。run-5 id 35 补充了 86.77% 的 splits 涉及一个 sub-100 µm micro-fragment。
- **一致之处：** 全部 GENERALIZE（id 63：99th 在全部三个脑上 ~6.4–6.5 µm；id 35-17：median gap 各处均为 18–30 µm；run-5 id 35：75–87% 的 micro-fragment 涉及度）。这是独有发现 #1 的 6.84 µm 分类器的描述性基础。
- **分歧之处：** 无。
- **沿用判定：** Verdict SOUND（run-4 id 63，...2026-06-17 id 35）；run-5 id 35 SOUND。
- **Caveats：** 2026-06-17 id 35 将一个中位数（gaps）与一个 95th 百分位（internal edges）作比较——这是一种刻意保守的表述，而非同类对同类的分布检验；run-5 id 35 的 100 µm micro-fragment 截断是一个选定的阈值。

### 23. (Priority 0.287 · Surprise 0.307) split gaps 处对立的端点彼此指向（anti-parallel，cosine −0.68 vs −0.51）。
- **来源：** ...2026-06-17 (id 67) —— 1 份报告；在本次运行内跨数据集得到印证，且在概念上与 run-4 的 angular-inertia 发现（#8）相关。
- **结论：** 在 525 个 split endpoint pairs vs 2,689 个空间相邻的对照 pairs 上，split endpoints 的 mean cosine similarity 为 −0.6847（更 anti-parallel），而对照为 −0.5112，由 Mann-Whitney U 显著（p = 1.1e-08）。跨 gap 的几何共线性是提出 split-joins 的一个 GT-free 特征。
- **一致之处：** GENERALIZES—— ds_794491 −0.5932 vs −0.3334（p=1.4e-10）；ds_794495 −0.7908 vs −0.5273（p=8.6e-20）。run-4 的 id 33 angular-inertia 结果（#8）是同一 "方向连续性桥接 splits" 思想的更强、分类器形式的表达，但测量的是 continuation-vs-false-candidate 角度而非 endpoint cosine，因此保持分开。
- **分歧之处：** 无。
- **沿用判定：** Reproduction REPRODUCED；Generalization GENERALIZES；Verdict SOUND。
- **Caveats：** 对照分布的方差大得多（std 0.534 vs 0.358），因此分布有大量重叠——共线性是一个有用的先验，而非一个完美的独立分类器。

### 24. (Priority 0.287 · Surprise 0.307) 局部 tortuosity / curvature 在 splits 附近统计上更高，但作为独立预测因子较弱。
- **来源：** run-4--...2026-06-20 (id 59，10-edge window，point-biserial r=0.0335；id 73，5-hop window)，run-5--...2026-06-25 (id 36，10 µm window) —— 3 条目，2 份报告。
- **结论（canonical = run-4 id 59，MINOR，包含效应量）：** split edges 比正确 edges 更曲折，但效应极小。run-4 id 59：median 1.1115 vs 1.0764，在 >1M edges 上 p≈0，point-biserial r = 0.0335（curvature 解释了 ~0.1% 的方差）。id 73 在 5-hop window 上复现（median 1.1132 vs 1.0801，p=1.66e-276）。run-5 id 36：mean 1.0905 vs 1.0541（t=14.27，p≈0），Cliff's delta 0.2096 —— "统计上真实，实践上微弱"。
- **一致之处：** 全部在方向上 GENERALIZE（在所有脑上 r≈0.03–0.06 / Cliff's delta 0.19–0.26）；run-4 id 59 和 id 73 在 neuron-cluster permutation 下 UPHELD（p=0.002 和 p=0.0008），效应确认为小。
- **分歧之处：** 无——全部同意该效应一致但小；所有报告都正确地将其谨慎表述为一个较弱的独立预测因子。
- **沿用判定：** Verdict MINOR（run-4 id 59，id 73；run-5 id 36）→ Post-correction WEAKENED（run-5 id 36）/ UPHELD-weak（run-4 id 59，id 73）。
- **Caveats：** 显著性由超大 n 驱动；edges 非独立；只有极端 tortuosity 的尾部在 splits 中富集；run-4 id 73 和 id 59 共享大部分底层数据，因此它们并非彼此的独立确认。

---

## 因冗余而排除

折叠进上述合并发现的非 canonical 条目：

**并入发现 #14（合并位点位于稠密、多分叉的邻域）：**
- `ground-truth-error-annotations-revised-version_2026-06-17` id 23 → 并入 #14（canonical id 3）
- `ground-truth-error-annotations-revised-version_2026-06-17` id 81 → 并入 #14
- `ground-truth-error-annotations-revised-version_2026-06-17` id 49 → 并入 #14
- `ground-truth-error-annotations-revised-version_2026-06-17` id 53 → 并入 #14
- `run-4--ground-truth-error-annotations-revised-version_2026-06-20` id 84 → 并入 #14

**并入发现 #15（合并 segments 是巨大的过度生长标签）：**
- `run-5--ground-truth-error-annotations-revised-version_2026-06-25` id 34 → 并入 #15（canonical id 56 / run-4 id 43）
- `run-5--ground-truth-error-annotations-revised-version_2026-06-25` id 55 → 并入 #15
- `run-4--ground-truth-error-annotations-revised-version_2026-06-20` id 24 → 并入 #15（super-merge 子论断，已推翻）

**并入发现 #16（splits 在 branch points 处/附近）：**
- `ground-truth-error-annotations-revised-version_2026-06-17` id 13 → 并入 #16（canonical id 10）
- `ground-truth-error-annotations-revised-version_2026-06-17` id 19 → 并入 #16
- `ground-truth-error-annotations-revised-version_2026-06-17` id 72 → 并入 #16
- `run-4--ground-truth-error-annotations-revised-version_2026-06-20` id 64 → 并入 #16
- `run-5--ground-truth-error-annotations-revised-version_2026-06-25` id 33 → 并入 #16

**并入发现 #17（splits 聚集成空间 "误差区"）：**
- `ground-truth-error-annotations-revised-version_2026-06-17` id 55 → 并入 #17（canonical run-5 id 26）
- `run-4--ground-truth-error-annotations-revised-version_2026-06-20` id 37 → 并入 #17
- `run-5--ground-truth-error-annotations-revised-version_2026-06-25` id 40 → 并入 #17
- `run-5--ground-truth-error-annotations-revised-version_2026-06-25` id 45 → 并入 #17

**并入发现 #18（splits 偏好纤细的远端/终端 processes）：**
- `ground-truth-error-annotations-revised-version_2026-06-17` id 65 → 并入 #18（canonical id 60）
- `run-5--ground-truth-error-annotations-revised-version_2026-06-25` id 38 → 并入 #18
- `run-5--ground-truth-error-annotations-revised-version_2026-06-25` id 70 → 并入 #18

**并入发现 #19（omits 具有突发性/连续性）：**
- `ground-truth-error-annotations-revised-version_2026-06-17` id 48 → 并入 #19（canonical run-5 id 27）
- `run-5--ground-truth-error-annotations-revised-version_2026-06-25` id 50 → 并入 #19

**并入发现 #20（omits 在 branch points / terminal 分支处）：**
- `ground-truth-error-annotations-revised-version_2026-06-17` id 11 → 并入 #20（canonical id 64）
- `run-4--ground-truth-error-annotations-revised-version_2026-06-20` id 32 → 并入 #20
- `run-4--ground-truth-error-annotations-revised-version_2026-06-20` id 85 → 并入 #20

**并入发现 #22（split gaps 极小，促使采用紧凑搜索半径）：**
- `ground-truth-error-annotations-revised-version_2026-06-17` id 35 → 并入 #22（canonical run-4 id 63）
- `run-5--ground-truth-error-annotations-revised-version_2026-06-25` id 35 → 并入 #22

**并入发现 #24（tortuosity 为较弱的 split 预测因子）：**
- `run-4--ground-truth-error-annotations-revised-version_2026-06-20` id 73 → 并入 #24（canonical run-4 id 59）
- `run-5--ground-truth-error-annotations-revised-version_2026-06-25` id 36 → 并入 #24

**并入发现 #13（Z-orientation 非一致驱动因素）：**
- `run-4--ground-truth-error-annotations-revised-version_2026-06-20` id 21 → 并入 #13（与 canonical id 27 一起）

**作为 split/omit 共聚类主题的已推翻/不泛化重复项折叠移除（在综述中报告，未提升为独立可信发现）：**
- `run-4--ground-truth-error-annotations-revised-version_2026-06-20` id 36（omit cable 更靠近合并位点）→ 在 neuron-cluster permutation 下 OVERTURNED（p=0.4771）；DOES-NOT-GENERALIZE。未保留为发现。
- `run-4--ground-truth-error-annotations-revised-version_2026-06-20` id 61（split edges 更靠近合并位点）→ OVERTURNED（cluster-permutation p=0.6469，CI 跨越 0）；DOES-NOT-GENERALIZE。未保留为发现。
- `run-5--ground-truth-error-annotations-revised-version_2026-06-25` id 42（split→omit 局部共现）→ DOES-NOT-GENERALIZE（在 794491 上符号翻转，在 789202 上不显著）；Verdict MAJOR。未保留为可信发现。
