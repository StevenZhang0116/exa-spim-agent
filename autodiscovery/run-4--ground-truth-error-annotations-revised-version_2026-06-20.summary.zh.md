# AutoDiscovery 结论摘要 — Run 4

**源文件：** `autodiscovery/run-4--ground-truth-error-annotations-revised-version_2026-06-20.json`（150 个假设）
**排序键（`rank_by`）：** `posterior-surprise`（priority = posterior × |surprisal|）
**已排序：** 148 个假设（`n_ranked`）；2 个因缺失 surprisal 被丢弃。
**此处展示：** 148 个已排序中的前 20 个（`n_returned` = 20，`--top 20`）。
**本次运行的 surprise-magnitude 范围：** 0.0000 到 0.6899（最高 priority score 0.5066）。

## 综合分析

两个最强的信念翻转都与成像轴和一项校对（proofreading）启发式规则有关。优先级最高的单个结果（#1，ID 30）是一次从 *Leaning False* 到 *Leaning True* 的正向翻转：一个严格的欧氏距离阈值（~6.84 µm）能够干净地将真实的 split 间隙与神经元间间隙区分开（ROC-AUC 0.9979，F1 0.9945），验证了循环最初曾怀疑的仅依赖临近性的自动重连接方案。与之相对，两个最强的 *负向* surprise（#2 ID 27 和 #3 ID 21）都推翻了一个被高度确信的假设：各向异性的 Z 轴成像分辨率会驱动 split/omit 错误。两个独立的检验——一个混合效应逻辑回归（p = 0.058，OR 0.80）和一个关于边方向的卡方检验（p = 0.060）——均未达到显著性，将信念从 *Likely True* 降至 *Uncertain*。贯穿始终的主题是：**分割错误更多地由局部拓扑与形态学（分支点、迂曲度、分支阶数、终末/远端纤维、Z 深度极值）所支配，而非由全局成像轴朝向所支配**，并且由此产生的错误模式（紧密的 split 间隙、角度惯性、空间聚集）足够规律，可为自动化校对智能体提供支撑。

## 已排序结论

### 1. (Priority 0.507 · Surprise 0.690) 一个 ~6.84 µm 的欧氏阈值可以安全地自动重连接绝大多数 split 片段。
- **Run：** run-4--ground-truth-error-annotations-revised-version_2026-06-20 · **ID：** 30 · **Belief：** Leaning False → Leaning True (0.2917→0.7344) · **Direction：** Positive
- **Tested：** 是否仅凭空间临近性就能区分真实的 split 间隙（属于同一 GT 神经元的片段端点）与神经元间间隙，从而让工具能在不引发 merge 的情况下自动重连接 split。
- **Conclusion：** 在 20 µm 范围内的 6,805 个真实 split 间隙和 4,189 个神经元间间隙中，两个分布几乎完全不重叠——真实 split 峰值在 ~4.5 µm（2–7 µm 带），神经元间间隙很少低于 7 µm。将间隙距离作为二元分类器，ROC-AUC 为 0.9979，F1 最优阈值 6.84 µm 达到 F1 = 0.9945。较大的正向 surprisal（+0.690）反映出循环曾倾向于反对仅依赖临近性的修复，但数据强烈验证其为一个安全、有效的启发式。
- **Caveats：** 无标注；复核确认实现忠实。结果是数据集内部的，因此对其他体积的泛化未经测试。
- **Reproduction：** REPRODUCED（code: revised-loading）
- **Rerun result：** 记录值 n_true=6805，n_fp=4189，ROC-AUC=0.9979，threshold=6.84 µm，F1=0.9945；重跑 n_true=6805，n_fp=4189，ROC-AUC=0.9979，threshold=6.84 µm，F1=0.9945 → 完全一致。
- **Generalization：** GENERALIZES
- **Across datasets：** origin (789202)：ROC-AUC=0.9979，thr=6.84 µm，F1=0.9945（n_true=6805/n_fp=4189）；ds_794491：ROC-AUC=0.9889，thr=6.48 µm，F1=0.9711（n_true=7847/n_fp=13208）；ds_794495：ROC-AUC=0.9953，thr=6.83 µm，F1=0.9878（n_true=7988/n_fp=8867）。近乎完美的临近性分离以及 ~6.5 µm 的阈值在两个额外数据集上均成立（两者均 exit 0）。
- **Verdict：** SOUND
- **Test：** 将间隙距离作为二元分类器的 ROC-AUC（`sklearn roc_curve/auc`）加上 F1 最优阈值；n_true=6805 个真实 split 间隙，n_fp=4189 个 20 µm 内的神经元间间隙；"ROC AUC: 0.9979 / Optimal Distance Threshold: 6.84 um / Max F1 Score: 0.9945"。
- **Statistical issues：** 无实质问题。这是一个描述性的可分性/分类器指标，而非假设检验，因此独立性/正态性假设不适用。F1 最优阈值是在其报告所用的同一数据上选取的（轻微的样本内乐观偏倚），但 AUC=0.9979 和近乎不相交的分布使得这一点无足轻重；阈值在两个额外体积上均可复现（~6.5 µm）。
- **Logic issues：** 无。结论（仅依赖临近性的自动重连接是安全的）直接来自近乎完美的分离，且 surprisal 符号（+0.690，Leaning False → Leaning True）与证据相符。
- **Verdict rationale：** 效应巨大、精确复现、泛化一致（三个体积上 AUC 0.989–0.998）。唯一的注意事项——假阳性"神经元间间隙"是在仿真中从 GT 标签派生的，而非来自真实 merge——是一个部署层面的注意事项，而非所记录检验的缺陷。

### 2. (Priority 0.323 · Surprise 0.795) Z 轴神经突起对齐并 NOT 显著提高拓扑错误风险。
- **Run：** run-4--ground-truth-error-annotations-revised-version_2026-06-20 · **ID：** 27 · **Belief：** Likely True → Uncertain (0.9167→0.4062) · **Direction：** Negative
- **Tested：** 沿低分辨率 z（成像）轴对齐的神经突起是否比 xy 平面对齐的片段遭受更多 split/omission。
- **Conclusion：** 在 1,160,529 条边（其中 4.44% 带有 split/omit 错误）中，对一个 20,000 条边子集做贝叶斯混合效应逻辑回归发现，z 对齐不显著且略为 *负*（coef = −0.066，p = 0.058；odds ratio 0.80）。强烈的负向 surprisal（−0.795，本次运行的最大幅度）标志着一个被高度确信、却被数据反驳的假设：相对于各向异性轴的朝向并非分割失败的主要驱动因素。
- **Caveats：** 回归是在一个 20,000 条边子样本上拟合的（而非全部 1.16M 条边）；p = 0.058 处于临界，因此该效应是"不显著"而非确凿的零效应。复核未发现实现问题。
- **Reproduction：** REPRODUCED（code: revised-loading）
- **Rerun result：** 记录值 edges=1160529，errors=4.44%，z_align coef=−0.0661（p=0.05823），OR=0.8029；重跑 edges=1160529，errors=4.44%，z_align coef=−0.0661（p=0.05823），OR=0.8029 → 完全一致（相同的 20k 子样本种子）。
- **Generalization：** PARTIAL
- **Across datasets：** *零/不显著* 的主张不能干净地成立。origin (789202)：coef=−0.0661，p=0.05823，OR=0.8029 → "does NOT significantly affect"（临界）。ds_794491：coef=+0.0376，p=0.1747，OR=1.1371 → 不显著但符号 *翻转为正*。ds_794495：coef=−0.1697，p=6.75e-06，OR=0.5632 → 显著 DECREASE（脚本本身打印 "significantly DECREASES the risk"）。因此"无显著 z 对齐效应"在 794491 上成立，但在 794495 上被反驳（显著），且方向在三者间不稳定（−/+/−）。原始的反各向异性结论（z 对齐 NOT 是风险 *增加器*）从未被逆转为风险增加，但具体的"不显著"发现是数据集特异的。
- **Verdict：** MAJOR
- **Test：** 在一个 20,000 条边子样本上的 `BinomialBayesMixedGLM.fit_vb()`（变分贝叶斯混合逻辑回归）；z_align_std Post.Mean=−0.0661，Post.SD=0.0349；随后通过将后验均值/SD 当作频率派 Wald z 来制造 p 值（`z=coef/sd; p=2*(1-Φ(|z|))`）。OR=0.8029。
- **Statistical issues：** (1) 检验构造错误——变分贝叶斯后验不是抽样分布，因此将 Post.Mean/Post.SD 转换为双侧 Wald p 值并非有效的显著性检验；所报告的 p=0.058 没有可辩护的频率派解释。(2) 模型在 1.16M 条边的一个 20k 子样本上拟合，丢弃了 ~98% 的数据，且所选种子的结果在各体积间不稳定。(3) 在一个神经元内共享节点的边不是独立的；逐边似然将它们视为独立（只建模了一个神经元随机效应方差，而非朝向协变量在神经元内的边相关性）。
- **Logic issues：** "未能达到 p<0.05" 被当作"z 轴朝向不是主要驱动因素"的证据——这是缺乏证据/证据缺失的谬误，加之 p 值本身无效，使问题更严重。强烈的负向 surprisal（−0.795，Likely True → Uncertain）是由一个临界的、不可复现的、不当计算的 p 所驱动。
- **Verdict rationale：** 头条数字是一个错误构造的 p 值，被用来支持一个站不住脚的零结论：PARTIAL 泛化，系数符号在 −/+/− 之间翻转，且一个体积显著（p=6.75e-06）。下游主张应从自信的"无各向异性效应"下调为"无一致、有效检验的效应"。
- **Corrected test：** 在 FULL 1.16M 条边（而非 20k 子样本）上的频率派逻辑回归，配合一个神经元级别的 cluster-permutation 检验（神经元 = 独立单位），以替换无效的 VB-后验-当作-Wald-z 的 p。原始方法把变分贝叶斯后验均值/SD 当作抽样分布，这不是有效的显著性检验。
- **Corrected result：** Neuron-cluster permutation Spearman rho(neuron z vs err rate) = −0.1818，双侧 p = 0.5721（NOT significant）；全数据 coef = −0.1203，朴素 Wald p = 2.4e-136，但 cluster-robust Wald p = nan（单簇退化），OR = 0.6707。对比原始无效的 VB-Wald p = 0.05823，OR = 0.8029。修正后的恰当检验确认 NO significant z 对齐效应（原始的零主张得以存续，现在建立在有效的依据上）。
- **Post-correction verdict：** UPHELD —— 原始结论是一个 *零* 结论（"z 对齐不是显著的驱动因素"），而有效的 cluster-permutation 检验与之一致（p = 0.5721，远不显著）。头条零结论成立，但只能作为"无可检测的效应"，而非缺失的证明。
- **Corrected generalization：** PARTIAL —— 794491：permutation p = 0.5545（不显著，与零一致）；794495：permutation p = 0.9962（不显著），但全数据 cluster-robust Wald 显著（p = 2.05e-11，OR = 0.514，一个 *减少*）。因此在 permutation 检验下零结论在三者上均成立，而 cluster-robust GLM 在 794495 上标记出问题——在正确的分析单位下，无效应结论得到广泛支持，但并非完全一致。

### 3. (Priority 0.266 · Surprise 0.568) Z 主导边并未表现出统计显著的 split/omit 错误超额。
- **Run：** run-4--ground-truth-error-annotations-revised-version_2026-06-20 · **ID：** 21 · **Belief：** Likely True → Uncertain (0.8333→0.4688) · **Direction：** Negative
- **Tested：** 硬件驱动的成像各向异性是否使 Z 主导边比 XY 主导边更频繁地失败。
- **Conclusion：** 在 433,243 条 Z 主导边和 975,802 条 XY 主导边中，错误率几乎相同（3.70% vs 3.63%）；卡方检验给出 χ² = 3.53，p = 0.0602，未能拒绝零假设。负向 surprisal（−0.568）佐证了假设 #2：方向性成像偏置不是一个实质性的错误驱动因素，与先验预期相矛盾。
- **Caveats：** p = 0.060 仅略高于阈值——是接近未命中而非干净的零结论；微小的绝对差距（0.07%）无论是否显著在实践上都可忽略。
- **Reproduction：** REPRODUCED（code: revised-loading）
- **Rerun result：** 记录值 Z-dom=433243（3.70%），XY-dom=975802（3.63%），χ²=3.5328，p=6.0165e-02；重跑相同 χ²=3.5328，p=6.0165e-02 → 完全一致。
- **Generalization：** DOES-NOT-GENERALIZE
- **Across datasets：** "无显著差异"（零）结论在两个额外数据集上都失败，它们在 *相反* 方向上变得强烈显著。origin (789202)：Z-dom 3.70% vs XY-dom 3.63%，χ²=3.53，p=6.02e-02（零，Z 略高）。ds_794491：Z-dom 6.41% vs XY-dom 5.50%，χ²=166.1，p=5.21e-38（显著，Z *更高*）。ds_794495：Z-dom 2.24% vs XY-dom 2.86%，χ²=395.8，p=4.62e-88（显著，Z *更低*）。在额外数据集上 *存在* 显著的朝向效应，且其符号不一致——因此 origin 干净的接近零结论不成立。
- **Verdict：** MAJOR
- **Test：** 对 Z 主导 vs XY 主导边 × 错误/无错误的 2×2 表做 Pearson 卡方独立性检验（`scipy.stats.chi2_contingency`）；433,243 条 Z 主导边和 975,802 条 XY 主导边；χ²=3.5328，p=0.0602。
- **Statistical issues：** 独立性假设被违反——行是 1.4M 条单独的边，但相邻边共享节点并沿同一神经突起延伸，因此朝向和错误状态是空间自相关的；卡方将它们视为独立抽样，这既可能膨胀也可能（此处）误述检验结果。0.07 个百分点的绝对率差（3.70% vs 3.63%）无论是否显著在实践上都可忽略。
- **Logic issues：** "p>0.05，未能拒绝零假设" 被升级为 "This refutes the hypothesis that imaging anisotropy creates a substantial directional bias"——从不显著结果肯定零假设，这是经典的缺乏证据错误。该结论还把单一体积的接近未命中过度泛化为关于成像流水线的论断。
- **Verdict rationale：** DOES-NOT-GENERALIZE：在两个额外体积上同一检验在 *相反* 方向上强烈显著（p=5.2e-38 和 p=4.6e-88），直接与"无方向性偏置"矛盾。origin 的临界零结论（也未通过 BH；见统计验证摘要）无法支撑反驳主张，必须下调。
- **Corrected test：** 对 Z−XY 错误率差做神经元级别的 block-permutation 检验（神经元 = 独立单位）加上带 95% CI 的风险率比，替换边独立的 Pearson 卡方（1.4M 条共享节点的边是自相关的，因此卡方夸大了有效 n）。
- **Corrected result：** 风险率比（Z/XY）= 1.0178，95% CI [0.9993, 1.0366]；绝对率差 = 0.065 pp；cluster-permutation 双侧 p = 0.8136（NOT significant）。对比原始 χ² = 3.5328，p = 6.0165e-02。修正后的检验同意 origin 显示无可检测的朝向效应，效应量微不足道（RR≈1.02，CI 贴近 1）。
- **Post-correction verdict：** UPHELD（对于 origin 零结论）—— cluster-aware 检验确认无显著的 Z-vs-XY 差异（p = 0.8136），且效应量可忽略。但报告正确地将其重新表述为缺乏证据，而非无各向异性的证明。
- **Corrected generalization：** PARTIAL —— 794491：cluster-permutation p = 0.09745（不显著，RR = 1.1645 CI [1.138, 1.192]，Z 略高）；794495：cluster-permutation p = 0.0004998（SIGNIFICANT，RR = 0.7852 CI [0.767, 0.804]，Z *更低*）。因此即使在正确的神经元级别检验下，origin/794491 的零结论在 794495 上也不成立，那里存在一个真实显著的相反方向朝向效应——证实该发现是体积特异的，而非一个稳定的零结论。

### 4. (Priority 0.265 · Surprise 0.414) Split 错误风险随离心分支阶数（距胞体的拓扑深度）上升。
- **Run：** run-4--ground-truth-error-annotations-revised-version_2026-06-20 · **ID：** 139 · **Belief：** Leaning False → Leaning True (0.375→0.6406) · **Direction：** Positive
- **Tested：** 逐边的 split 错误概率是否随分支阶数增加，且独立于局部纤维粗细。
- **Conclusion：** 在 1.4M 条边上，逻辑回归发现分支阶数是 split 错误的显著正向预测因子（coef = 0.0194，p < 0.001）；split 率在 ~30 阶之前保持低而稳定，随后在更深分支中飙升并变得波动（在 ~53 阶附近 >3%）。正向 surprisal（+0.414）反映了循环从怀疑深度是否重要向上修正。
- **Caveats：** 粗细协变量（`norm_thickness`）方差为零并被剔除，因此"独立于粗细"这一条款实际上无法被控制——粗细效应无法排除。深阶率因高阶数据稀疏而波动。
- **Reproduction：** REPRODUCED（code: revised-loading）
- **Rerun result：** 记录值 edges=1409045，branch_order coef=0.0194，z=16.369，p<0.001；重跑 edges=1409045，branch_order coef=0.0194，z=16.369，p<0.001（两次都剔除了 norm_thickness）→ 完全一致。
- **Generalization：** DOES-NOT-GENERALIZE
- **Across datasets：** 正向的分支阶数 → split 风险斜率在 BOTH 额外数据集上翻转符号。origin (789202)：branch_order coef=+0.0194，z=16.37，p<0.001（风险随深度上升）。ds_794491：coef=−0.0157，z=−7.66，p<0.001（风险随深度 *下降*）。ds_794495：coef=−0.0063，z=−4.20，p<0.001（风险随深度 *下降*）。三者均显著，但方向在两个额外数据集上均相反，因此"更深的分支 split 更多"的结论是 origin 数据集特异的。
- **Verdict：** MAJOR
- **Test：** 在 1,409,045 条边上对 is_split 关于 branch_order 的 `statsmodels.Logit`（二元逻辑回归）；branch_order coef=0.0194，z=16.369，p<0.001。`norm_thickness` 协变量因方差为零被剔除（"constant radius"）。
- **Statistical issues：** (1) 该假设明确声称效应"独立于局部纤维粗细"成立，但粗细协变量方差为零并被剔除，因此模型并未控制任何东西——"独立于粗细"条款在此代码中结构上不可检验，并未被确认。(2) 边的非独立性：神经元内 1.4M 条边在空间上相关，因此 z=16.4（及 p<0.001）夸大了有效信息；真实标准误更大。(3) 分支阶数由一个仅在分支节点处递增、并选取一个任意胞体（最大半径节点）的 BFS 分配，因此该预测因子本身有噪声。
- **Logic issues：** 结论（"更深的拓扑分支更易 split，独立于粗细"）同时主张一个方向性效应和一个该实验从未隔离的粗细受控机制。所记录的正向 surprisal（+0.414）针对的是一个在两个其他体积上都翻转符号的斜率。
- **Verdict rationale：** DOES-NOT-GENERALIZE —— coef 从 +0.0194 翻转为 −0.0157 和 −0.0063（均显著）在两个额外数据集上，因此"更深的分支 split 更多"是体积特异的；结合被剔除的协变量错误，因果/独立性主张得不到支持，必须下调。
- **Corrected test：** Cluster-robust 逻辑回归（按神经元聚类 SE）加上一个神经元级别的 permutation 交叉验证（每神经元分支阶数 vs split 率的 Spearman），替换边独立的 `statsmodels.Logit`，后者的 z=16.4 把 1.4M 条神经元内自相关的边当作独立处理。
- **Corrected result：** Cluster-robust z = 1.990，p = 0.04662（勉强显著），每 +1 分支阶的 OR = 1.0196，95% CI [1.0003, 1.0393]（下界基本在 1）；神经元级别的 permutation 交叉验证给出 Spearman rho = 0.2308，p = 0.4697（NOT significant）。对比原始 coef = 0.0194，z = 16.369，p < 0.001（被当作压倒性显著）。一旦处理边的非独立性，z 从 16.4 崩塌到 ~2，且 permutation 检验不显著。
- **Post-correction verdict：** OVERTURNED —— 神经元级别的 permutation 检验（最干净的分析单位）不显著（p = 0.4697），且 cluster-robust CI 下界处于 1.0003；那个"压倒性"的 p<0.001 完全是把 1.4M 条相关边当作独立处理的伪影。
- **Corrected generalization：** DOES-NOT-GENERALIZE —— 794491：permutation p = 0.9146（不显著），cluster-robust coef = −0.0157（符号翻转）；794495：permutation p = 0.1708（不显著），cluster-robust coef = −0.0063，z = −0.538，p = 0.5907。在正确的检验下，效应在三者上均不显著，且方向在两个额外数据集上均反转——分支阶数 → split 风险的主张无法存续。

### 5. (Priority 0.253 · Surprise 0.284) 超级 merge（≥3 个融合神经元）所覆盖的 GT 纤维比 2 神经元 merge 不成比例地更多。
- **Run：** run-4--ground-truth-error-annotations-revised-version_2026-06-20 · **ID：** 24 · **Belief：** Leaning True → Likely True (0.7083→0.8906) · **Direction：** Positive
- **Tested：** 融合三个或更多 GT 神经元的 merge 片段是否是覆盖每神经元远多于 2 神经元 merge 的庞大结构。
- **Conclusion：** 在所发现的 merge 中，24 个是 2 神经元，3 个是超级 merge。2 神经元 merge 的覆盖纤维中位数为 6.21 mm，而超级 merge 为 35.19 mm，差异显著（Mann-Whitney U p = 0.0059）；每神经元覆盖从 ~3.1 缩放到 ~11.7 mm/神经元。适度的正向 surprisal（+0.284）将一个本已偏好的信念略微上调。
- **Caveats：** 仅有 3 个超级 merge——一个极小样本，因此幅度估计脆弱。这些步骤比较的是 *总* 纤维，而假设指定的是 *每神经元*；复核认为结论仍成立，但所检验的量与主张略有差异。
- **Reproduction：** DIVERGED（code: revised-loading）
- **Rerun result：** 样本计数与显著性都发生了实质变化。记录值：2 神经元 merge n=24，超级 merge n=3，Mann-Whitney U=0.0，p=5.8861e-03（显著）。重跑：2 神经元 merge n=8，超级 merge n=1，U=0.0，p=2.2222e-01（NOT significant —— p 越过 0.05）。覆盖纤维中位数（6.2054 mm vs 35.1873 mm）不变，但 n 的崩塌（24→8，3→1）使得在所提供的 pkl 上 Mann-Whitney 检验不显著，推翻了所记录的结论。该数据集上的 merge 检测步骤找到的 merge 远少于所记录的运行，因此小样本检验不再达到显著性。
- **Generalization：** PARTIAL
- **Across datasets：** 超级 merge 计数微小且不一致。origin-rerun (789202)：2 神经元 n=8（中位数 6.21 mm）vs 超级 n=1（35.19 mm），U=0.0，p=2.22e-01（不显著）。ds_794491：37 个 2 神经元 merge 但 ZERO 超级 merge → "Not enough data to perform" 该检验（无法评估）。ds_794495：2 神经元 n=24（中位数 5.42 mm）vs 超级 n=2（中位数 168.42 mm），U=0.0，p=6.15e-03（显著，超级 merge 大得多）。方向（超级 merge 覆盖更多纤维）在任何存在 ≥1 个超级 merge 处都成立，且在 794495 上显著，但 origin-rerun 不显著，且 794491 没有超级 merge——样本太匮乏，无法判定稳健效应。
- **Verdict：** MAJOR
- **Test：** 双侧 Mann-Whitney U（`scipy.stats.mannwhitneyu`）比较 2 神经元 merge（n=24）vs 超级 merge（n=3）的覆盖纤维；记录值 U=0.0，p=5.8861e-03；中位数 6.2054 mm vs 35.1873 mm。
- **Statistical issues：** (1) 功效不足——n=3 个超级 merge（且 U=0.0，所有三个超级 merge 都高于所有 2 神经元 merge）是甚至能达到 p<0.01 的最小配置；一个被重新分类的 merge 就能改变结论。复现正好确认了这一点：在所提供的 pkl 上计数崩塌为 n=8 / n=1，p 升至 0.2222（NOT significant）。(2) 构念失配——假设是关于 *每神经元* 纤维的，但代码比较的是每个 merge 标签的 *总* `seg_cable[lab]`；总纤维对超级 merge 机械地更大，仅仅因为它们跨越更多神经元，因此该检验部分地检验了自己的定义。
- **Logic issues：** 结论（"超级 merge 不成比例地封装了更大量的物理 GT 纤维"）被陈述为强支持，但所检验的量（总纤维）不同于所主张的量（每神经元纤维），且推断建立在 3 个点上。正向 surprisal（+0.284）夸大了证据。
- **Verdict rationale：** 在复现上 DIVERGED（p 0.0059 → 0.2222，越过 0.05）且在泛化上 PARTIAL（在 794491 上不可检验，仅在 794495 上显著）。一个在重跑时其中一臂 n=1 就导致显著性蒸发的发现并不可信；下调。
- **Corrected test：** 在 PER-NEURON 纤维（假设实际主张的量）上的 Mann-Whitney U，配合 Cliff's delta 效应量和 bootstrap CI，替换原始在 TOTAL 纤维上的检验（对超级 merge 机械地更大，因为它们跨越更多神经元——该检验部分地检验了自己的定义）。
- **Corrected result：** 在每神经元构念上，n_super = 1 vs n_2neuron = 8，Mann-Whitney U = 8.0，精确双侧 p = 0.2222（NOT significant）；Cliff's delta = 1.0 [1.0, 1.0]，但被标记 "n_super=1 too small for reliable inference"。对比原始（总纤维）U = 0.0，p = 5.8861e-03。在正确的每神经元量和所提供 pkl 的 merge 计数下，该检验不显著。
- **Post-correction verdict：** OVERTURNED —— 修正后的每神经元检验不显著（p = 0.2222）且建立在单个超级 merge 上；原始 p = 0.0059 既来自错误的（总纤维）构念，也来自一个在重跑时崩塌的样本。
- **Corrected generalization：** INCONCLUSIVE / PARTIAL —— 794491：零个超级 merge，"not enough data to perform the test"（INCONCLUSIVE）；794495：每神经元 n_super = 2 vs n_2neuron = 24，U = 46.0，p = 0.02462，Cliff's delta = 0.9167 [0.75, 1.0]（显著，超级更大，但仍 "n_super=2 too small"）。方向（超级 > 2 神经元）在存在 ≥1 个超级 merge 处成立，但在正确的构念下证据太样本匮乏，无法判定稳健效应。

### 6. (Priority 0.253 · Surprise 0.284) Omission 错误在分支点处远比在线性纤维上更频繁。
- **Run：** run-4--ground-truth-error-annotations-revised-version_2026-06-20 · **ID：** 32 · **Belief：** Leaning True → Likely True (0.7083→0.8906) · **Direction：** Positive
- **Tested：** Omit 错误是否在拓扑分支节点（degree > 2）处以高于线性节点（degree = 2）的率发生。
- **Conclusion：** 分支节点的 omission 率约为 ~11.95%，而线性节点约为 ~2.76%——>4× 的差异，χ² = 1567.69，p < 0.0001。这强烈支持在校对期间针对复杂分支交汇处寻找缺失纤维。正向 surprisal（+0.284）。
- **Caveats：** 无标注；复核报告一个忠实、无偏离的实现。
- **Reproduction：** REPRODUCED（code: revised-loading）
- **Rerun result：** 记录值 branch omit=11.9479%，linear omit=2.7602%，χ²=1567.6878，p≈0；重跑相同列联表（606/4466 vs 38610/1360197），χ²=1567.6878，p≈0 → 完全一致。
- **Generalization：** GENERALIZES
- **Across datasets：** 分支节点在所有三个数据集上都显示出比线性节点更高的 omit 率，高度显著。origin (789202)：branch 11.95% vs linear 2.76%，χ²=1567.7，p≈0（~4.3×）。ds_794491：branch 7.54% vs linear 3.41%，χ²=193.6，p=5.13e-44（~2.2×）。ds_794495：branch 6.52% vs linear 1.72%，χ²=974.9，p=5.24e-214（~3.8×）。效应量在 794491 上较小，但方向和显著性在各处都成立。
- **Verdict：** MINOR
- **Test：** 对分支节点（degree>2）vs 线性节点（degree=2）× 被 omit/已标注的 2×2 表做 Pearson 卡方（`scipy.stats.chi2_contingency`）；列联 606/4466（branch）vs 38610/1360197（linear）；χ²=1567.6878，p≈0；omit 率 11.95% vs 2.76%。
- **Statistical issues：** 节点级独立性在技术上被违反（相邻节点共享边），因此 χ² 夸大了有效 n；期望单元计数都很大，因此检验本身是良定的。然而效应量很大（≈4.3× 相对率），因此假设违反不威胁定性结论。
- **Logic issues：** 无重大问题。结论（针对分支交汇处寻找缺失纤维）来自 >4× 的率差；相关性未被过度主张为已证明的模型机制。
- **Verdict rationale：** 效应大、已复现、且 GENERALIZES（三者上 2.2×–4.3×，p≤5e-44）。唯一的瑕疵是节点的非独立性，在此效应量下无足轻重——因此 MINOR 而非 SOUND。

### 7. (Priority 0.253 · Surprise 0.284) 角度惯性（~153° vs ~90°）可靠地识别跨 split 的真实延续。
- **Run：** run-4--ground-truth-error-annotations-revised-version_2026-06-20 · **ID：** 33 · **Belief：** Leaning True → Likely True (0.7083→0.8906) · **Direction：** Positive
- **Tested：** 跨 split 的真实延续路径是否比来自另一神经元的邻近假候选边更接近一条笔直的 180° 线。
- **Conclusion：** 在 13,582 个 split 配置中，真实延续平均为 152.96°，而假候选为 90.17°；分布是不同的（KS = 0.746，p ≈ 0），角度对齐以 ROC-AUC 0.9322 进行判别。因此方向性惯性是一个用于桥接 split 的智能体校对器的强局部启发式。正向 surprisal（+0.284）。
- **Caveats：** 无标注；复核确认无偏离计划。
- **Reproduction：** REPRODUCED（code: revised-loading）
- **Rerun result：** 记录值 n=13582，true=152.96°，false=90.17°，KS=0.7460（p≈0），ROC-AUC=0.9322；重跑相同 → 完全一致。
- **Generalization：** GENERALIZES
- **Across datasets：** 角度惯性在所有三个数据集上几乎一致地判别真 vs 假延续。origin (789202)：true 152.96° vs false 90.17°，KS=0.7460（p≈0），ROC-AUC=0.9322（n=13582）。ds_794491：153.49° vs 90.15°，KS=0.7335，ROC-AUC=0.9281（n=15637）。ds_794495：155.17° vs 90.08°，KS=0.7559，ROC-AUC=0.9365（n=15929）。非常稳定、稳健的启发式。
- **Verdict：** SOUND
- **Test：** 对真延续 vs 假候选角度做双样本 Kolmogorov–Smirnov（`scipy.stats.ks_2samp`）加上 ROC-AUC；n=13,582 个 split 节点配置；KS=0.7460，p≈0；均值 152.96° vs 90.17°；ROC-AUC=0.9322。
- **Statistical issues：** 这两个角度样本按 split 节点配对，而 KS 假设独立样本；在 KS=0.746（近乎完全的分布分离）下，依赖性不影响定性结果，而 ROC-AUC（一个配对友好的排序指标）佐证了它。假候选是来自另一神经元的几何最近节点，这是一个合理、保守的比较对象。
- **Logic issues：** 无。结论（方向性惯性是一个强的局部 split 桥接启发式）正是 63° 的均值差距和 AUC 0.93 所支持的；无因果越界。
- **Verdict rationale：** 效应非常大、精确复现，且在三个体积上 GENERALIZES 且 ROC-AUC 0.93。KS 独立性在此分离度下是次要技术细节——SOUND。

### 8. (Priority 0.253 · Surprise 0.284) 被 omit 的纤维在空间上比正确纤维更靠近 merge 位点聚集。
- **Run：** run-4--ground-truth-error-annotations-revised-version_2026-06-20 · **ID：** 36 · **Belief：** Leaning True → Likely True (0.7083→0.8906) · **Direction：** Positive
- **Tested：** Omit 错误是否在空间上围绕 merge 错误聚集，提示模型在融合主导结构时牺牲了细突起。
- **Conclusion：** 将 49,295 个 omit 节点与长度匹配的 49,295 个正确节点对照 67 个 merge 位点进行比较，到最近 merge 的距离中位数对 omit 节点为 1,812.73 µm，对正确节点为 1,959.88 µm——一个高度显著的差异（Mann-Whitney U p = 1.60e-66）。这支持一种系统性偏置：merge 与被丢弃的相邻突起共同出现。正向 surprisal（+0.284）。
- **Caveats：** 绝对中位数差距（~1,800 µm 中的 ~147 µm）在实践上较小，尽管大样本带来极端显著性。仅有 67 个 merge 位点锚定距离计算。
- **Reproduction：** REPRODUCED（code: revised-loading）
- **Rerun result：** 记录值 67 个 merge 位点，omit 中位数=1812.73 µm，correct 中位数=1959.88 µm，U=1138194675.0，p=1.6036e-66；重跑相同 → 完全一致。
- **Generalization：** PARTIAL
- **Across datasets：** "omit 节点比正确节点更靠近 merge 位点" 在一个额外数据集上成立，在另一个上 *反转*。origin (789202)：omit 中位数 1812.73 µm < correct 1959.88 µm（omit 更近），U=1.14e9，p=1.60e-66（显著）。ds_794495：omit 中位数 1422.53 µm < correct 1670.94 µm（omit 更近），p=9.78e-226（显著，同方向）。ds_794491：omit 中位数 1103.24 µm > correct 842.11 µm（omit *更远*），单侧 p=1.0000（不显著；方向翻转）。因此 omit 与 merge 的空间共聚集在 794495 上成立但在 794491 上不成立。
- **Verdict：** MAJOR
- **Test：** 对 49,295 个 omit 节点 vs 49,295 个长度匹配的正确节点、对照 67 个 merge 位点的到最近 merge 距离做单侧 Mann-Whitney U（`alternative='less'`）；U=1138194675.0，p=1.6036e-66；中位数 1812.73 µm（omit）vs 1959.88 µm（correct）。
- **Statistical issues：** (1) 显著性纯粹由样本量驱动——中位数差异在 ~1,800 µm 基线上约为 ~147 µm（≈8%），一个实践上微不足道的位移，然而每臂 n≈49k 迫使 p=1.6e-66。(2) 49k 个节点不是独立观测：它们沿神经突起聚集，且其距离都测量到相同的 67 个锚点，因此有效 n 远小于 49,295，p 值被大幅夸大。(3) 仅有 67 个 merge 锚点定义了整个距离场。
- **Logic issues：** 结论从一个微小的中位数偏移推断出一个系统性机制（"模型在融合时牺牲了细的相邻突起"）；空间聚集主张越过了 8% 中位数差异所能支撑的范围，且方向甚至不稳定。
- **Verdict rationale：** PARTIAL 泛化——效应在 794491 上 *反转*（omit 节点更远，单侧 p=1.0000）。一个既效应量微不足道/功效过度又在各体积间方向不稳定的发现不应被信任为真实的共聚集规律；下调。
- **Corrected test：** 对 omit−correct 中位数距离差做神经元-cluster permutation 检验（神经元 = 独立单位），加上带 bootstrap CI 的 Cliff's delta，替换把仅参照 67 个锚点的空间聚集节点当作独立处理的 49k 节点 Mann-Whitney。
- **Corrected result：** Cliff's delta = −0.0642，95% CI [−0.101, −0.028]（微不足道，≈−7.5% 中位数位移）；神经元-cluster permutation 单侧 p = 0.4771（NOT significant；观测到的神经元级别中位数差仅 −17.33 µm）。对比原始朴素 U = 1.14e9，p = 1.6036e-66。一旦分析单位是神经元，"omit 更靠近 merge" 效应不显著。
- **Post-correction verdict：** OVERTURNED —— cluster-permutation 检验不显著（p = 0.4771）且效应量微不足道；p = 1.6e-66 纯粹是 n 膨胀伪影。
- **Corrected generalization：** DOES-NOT-GENERALIZE —— 794491：Cliff's delta = +0.253（omit *更远*），permutation p = 0.8108（不显著，方向反转）；794495：Cliff's delta = −0.133，permutation p = 0.5007（尽管朴素 p = 9.78e-226 仍不显著）。在正确的神经元-cluster 检验下，共聚集效应在所有三个体积上均不显著。

### 9. (Priority 0.253 · Surprise 0.284) Split 错误在空间上聚集成局部化的"错误区"。
- **Run：** run-4--ground-truth-error-annotations-revised-version_2026-06-20 · **ID：** 37 · **Belief：** Leaning True → Likely True (0.7083→0.8906) · **Direction：** Positive
- **Tested：** 一条 split 边是否比一条正确边更可能在 30 µm 半径内有另一个 split（局部化的伪影区）。
- **Conclusion：** 在 6,805 条 split 边和匹配的 6,805 条正确边中，split 边在 30 µm 内平均有 0.97 个 split 邻居，而正确边仅 0.10（Mann-Whitney U p ≈ 0）。Split 在区域上聚集，支持将审阅者引导至密集错误热点以批量纠正的校对工作流。正向 surprisal（+0.284）。
- **Caveats：** 无标注；复核确认实现忠实。
- **Reproduction：** REPRODUCED（code: revised-loading）
- **Rerun result：** 记录值 6805 条 split 边，平均 split 邻居 0.97（split）vs 0.10（correct），U=34661344.5，p≈0；重跑相同 → 完全一致。
- **Generalization：** GENERALIZES
- **Across datasets：** split 边在所有三个数据集上都比正确边有远多的邻近 split，全程 p≈0。origin (789202)：30 µm 内 0.97 vs 0.10 个 split 邻居，U=3.47e7，p≈0。ds_794491：1.39 vs 0.28，U=4.68e7，p≈0。ds_794495：1.26 vs 0.11，U=5.11e7，p≈0。Split 聚集成错误区是稳健的。
- **Verdict：** MINOR
- **Test：** 对 6,805 条 split 边 vs 6,805 条匹配正确边在 30 µm 内的 split 邻居计数做单侧 Mann-Whitney U（`alternative='greater'`）；U=34661344.5，p≈0；均值 0.97 vs 0.10 个 split 邻居。
- **Statistical issues：** 邻居计数指标是相对于 split 边 KD-tree 计算的，因此 split 边计入它们所属集合中的 *其他 split*——一个内建的不对称性，膨胀了 split 组计数。在 ~10× 均值差异下结论仍成立，但幅度部分是构造伪影，且 split 边不是相互独立的观测。
- **Logic issues：** 结论（split 形成支持热点路由的局部化错误区）来自近 10× 的对比，且恰当地是操作性的而非机制性的。
- **Verdict rationale：** 效应大、已复现、GENERALIZES（0.97/1.39/1.26 vs 0.10/0.28/0.11）。指标不对称和边的非独立性使其保持在 MINOR 而非 SOUND。
- **Corrected test：** 对 split−correct 平均 split 邻居差距做神经元-cluster permutation 检验（神经元 = 独立单位），加上带 cluster bootstrap CI 的 Cliff's delta，替换逐边 Mann-Whitney（split 边非独立）。
- **Corrected result：** Cliff's delta = 0.4913，cluster bootstrap 95% CI [0.4363, 0.5600]（中到大）；神经元-cluster permutation 单侧 p = 0.0002（SIGNIFICANT；观测到的神经元级别差距 = 0.8144）。对比原始均值 0.97 vs 0.10 个 split 邻居，MW p ≈ 0。聚集效应在 cluster-aware 检验下以可观的效应量存续。
- **Post-correction verdict：** UPHELD —— 在神经元-cluster permutation 检验下显著（p = 0.0002），Cliff's delta 为中到大（0.49，CI 不含 0）；split 聚集的"错误区"发现对独立性修正稳健。
- **Corrected generalization：** GENERALIZES —— 794491：Cliff's delta = 0.5193 [0.460, 0.587]，permutation p = 0.0004（显著）；794495：Cliff's delta = 0.5995 [0.559, 0.641]，permutation p = 0.0002（显著）。在两个额外数据集上以可比/更大的效应量显著。

### 10. (Priority 0.253 · Surprise 0.284) 在片段图上的 A* 寻路在不引发 merge 的情况下修复了 86% 的 split。
- **Run：** run-4--ground-truth-error-annotations-revised-version_2026-06-20 · **ID：** 39 · **Belief：** Leaning True → Likely True (0.7083→0.8906) · **Direction：** Positive
- **Tested：** 惩罚急剧偏离（>45°）和半径骤变的图 A* 搜索能否在不产生新 merge 的情况下在 U-Net 片段图上重连接 >40% 的 split 边。
- **Conclusion：** 智能体在不跨入不同 GT 神经元的情况下连接了 6,805 个目标 split 中的 5,881 个（86.42% 成功），远超 40% 阈值；仿真修复将边精度从 78.71% 提升至 79.13%（+0.42%）。正向 surprisal（+0.284）确认图寻路是一个有效的 split 修复策略。
- **Caveats：** 净精度增益（+0.42%）在数据集规模上是适度的。"未引发 merge" 通过仿真中的 GT 标签判定，而非真实分割，因此现实世界的假 merge 风险可能不同。
- **Reproduction：** REPRODUCED（code: revised-loading）
- **Rerun result：** 记录值 5881/6805 split 已解决（86.42%），精度 78.71%→79.13%（+0.42%）；重跑相同 → 完全一致。
- **Generalization：** GENERALIZES
- **Across datasets：** A* split 修复成功率在所有三个数据集上都远超 40% 阈值。origin (789202)：5881/6805 = 86.42% 已解决，精度 78.71%→79.13%（+0.42%）。ds_794491：6648/7847 = 84.72%，74.77%→75.95%（+1.18%）。ds_794495：7166/7988 = 89.71%，68.55%→69.07%（+0.53%）。成功率一致地在 ~85-90%。
- **Verdict：** MINOR
- **Test：** 描述性仿真——带转向/半径惩罚的片段图 A* 寻路；5,881/6,805 个目标 split 已解决（86.42%）对照 >40% 的假设阈值；仿真边精度 78.71%→79.13%（+0.42%）。无推断统计量。
- **Statistical issues：** 不适用（无假设检验；它是对固定 40% 标准的通过/未通过）。5,000 节点 A* 扩展上限和 `dist_gt<10 µm` merge 有效性半径是任意阈值，影响分子和分母。
- **Logic issues：** "未引发 merge" 是在仿真内部对照 GT 标签裁定的，而非对照真实分割，因此现实世界的假 merge 率实际上未被测量——无 merge 保证相对于部署被过度主张。净精度增益（+0.42%）是适度的，不应被解读为大的质量改进。
- **Verdict rationale：** 86% ≫ 40% 的余量已复现且 GENERALIZES（三者上 85–90%），因此头条成立；MINOR 仅因仿真-vs-真实的 merge 安全性越界和任意搜索上限。

### 11. (Priority 0.253 · Surprise 0.284) Merge 片段是失控的"巨型"组件，远大于非 merge 片段。
- **Run：** run-4--ground-truth-error-annotations-revised-version_2026-06-20 · **ID：** 43 · **Belief：** Leaning True → Likely True (0.7083→0.8906) · **Direction：** Positive
- **Tested：** Merge 片段是否覆盖指数级多于非 merge 片段的 GT 纤维（庞大的过度生长标签 vs 局部小波动）。
- **Conclusion：** 64 个 merge 片段平均 ~15,449 µm 纤维（中位数 ~3,099 µm），而 8,273 个非 merge 片段为 ~532 µm（中位数 ~102 µm）——~29× 的均值差异，在对数变换长度上高度显著（Welch's t = 16.54，p = 7.32e-25）。Merge 由庞大的过度生长标签驱动，而非小的局部波动。正向 surprisal（+0.284）。
- **Caveats：** 仅有 64 个 merge 片段，且分布严重偏斜（均值 ≫ 中位数），因此均值对离群值敏感；对数变换和中位数缓解但不消除这一点。
- **Reproduction：** REPRODUCED（code: revised-loading）
- **Rerun result：** 记录值 64 个 merge vs 8273 个非 merge，均值 15448.96 vs 531.76 µm，Welch t=16.5444，p=7.3232e-25；重跑相同 → 完全一致。
- **Generalization：** GENERALIZES
- **Across datasets：** merge 片段在所有三个数据集上都覆盖远多于非 merge 的纤维，在对数长度上高度显著。origin (789202)：均值 15449 vs 532 µm（中位数 3099 vs 102），Welch t=16.54，p=7.32e-25（n=64）。ds_794491：均值 4457 vs 194 µm（中位数 1330 vs 29），t=29.03，p≈0（n=98）。ds_794495：均值 16277 vs 477 µm（中位数 3226 vs 43），t=24.02，p≈0（n=98）。稳健的巨型 merge 特征。
- **Verdict：** SOUND
- **Test：** 对 log10 变换的片段纤维长度做 Welch 双样本 t 检验（`ttest_ind(equal_var=False)`）；64 个 merge vs 8,273 个非 merge 片段；t=16.5444，p=7.3232e-25；均值 15,449 vs 532 µm，中位数 3,099 vs 102 µm。
- **Statistical issues：** 全程选择恰当——log10 变换处理了严重的右偏，Welch t 处理不等方差/组大小，且片段是自然的独立单位（每个预测标签一个长度），因此独立性假设在此满足（不同于边级检验）。64 个片段的 merge 类是适度的，但 ~29× 的中位数差距远离分辨率极限。
- **Logic issues：** 措辞小瑕——假设说"指数级更大"，而检验展示的是一个大的乘法（对数尺度）差异，而非指数增长律；起作用的结论（merge 是巨型过度生长标签，而非局部波动）得到充分支持。
- **Verdict rationale：** 在独立单位上的正确检验、效应巨大、精确复现，且 GENERALIZES（三者上 t≥16，p≤7e-25）。SOUND。

### 12. (Priority 0.253 · Surprise 0.284) Omit 错误在极端 Z 深度处约比中央深度处高 ~2.4×。
- **Run：** run-4--ground-truth-error-annotations-revised-version_2026-06-20 · **ID：** 45 · **Belief：** Leaning True → Likely True (0.7083→0.8906) · **Direction：** Positive
- **Tested：** Omit 错误是否在成像体积的极端顶/底 Z 坐标处比在中央深度处更频繁（光学衰减/散射）。
- **Conclusion：** 极端 Z 节点（顶/底 10%）有 4.44% 的 omit 率（2,283/51,369），而中央为 1.94%（11,342/585,909）；一个控制脑的 Cochran-Mantel-Haenszel 检验给出合并 OR 2.3561，p ≈ 0。轴向极端降低重建质量，与深度依赖的信号丢失一致。正向 surprisal（+0.284）。注意这涉及 Z *深度/位置*，与 #2/#3 中被反驳的 Z *朝向* 效应不同。
- **Caveats：** Min-max 归一化将"极端"相对于每个体积自身的 Z 范围定义，将真实光学深度与体积边缘边界伪影混淆。复核中无其他标注。
- **Reproduction：** REPRODUCED（code: revised-loading）
- **Rerun result：** 记录值 Extreme 4.44%（2283/51369）vs Center 1.94%（11342/585909），CMH OR=2.3561，p≈0；重跑相同 → 完全一致。
- **Generalization：** PARTIAL
- **Across datasets：** 极端 Z 深度处升高的 omit 率在一个额外数据集上成立，在另一个上消失。origin (789202)：Extreme 4.44% vs Center 1.94%，CMH OR=2.3561，p≈0（显著）。ds_794495：Extreme 5.37% vs Center 1.76%，OR=3.1672，p≈0（显著，甚至更强）。ds_794491：Extreme 4.00% vs Center 3.94%，OR=1.0149，p=7.996e-01（无效应——率基本相等）。因此深度极端 omit 超额泛化到 794495 但不泛化到 794491。
- **Verdict：** MAJOR
- **Test：** 对极端 Z vs 中央 Z × omit、按脑分层做 Cochran–Mantel–Haenszel 检验（`statsmodels StratifiedTable`）；合并 OR=2.3561，p≈0；极端 4.44%（2283/51369）vs 中央 1.94%（11342/585909）。
- **Statistical issues：** (1) CMH 的"控制脑"在 origin 上是虚幻的——pkl 中只有一个脑，因此分层检验退化为一个单一的 2×2 卡方，没有实际的混杂控制。(2) 节点的非独立性再次膨胀有效 n。(3) "极端 Z" 由逐体积 min-max 归一化定义（*观测* z 范围的顶/底 10%），将真实光学深度与体积边缘/边界截断伪影混淆——因此即使 OR=2.36 也可能反映 FOV 边界效应而非衰减。
- **Logic issues：** 结论主张了一个具体的物理原因（"极端深度处的光学衰减或散射降低信号"），而实验无法将其与边界/边缘伪影隔离——一个机制越界。OR=2.36 是一个真实的中等效应，但因果归因得不到支持。
- **Verdict rationale：** PARTIAL 泛化——效应在 794495 上增强（OR=3.17），但在 794491 上 *消失*（OR=1.0149，p=0.80），因此它不是一个稳定属性；结合无真实分层和边界混杂问题，光学衰减主张必须下调。
- **Corrected test：** 对 Extreme-vs-Center odds ratio 做神经元-cluster permutation 检验（神经元 = 独立单位），并在单一 2×2 上做 Woolf 95% CI，替换在 origin 上只有一个分层（无真实混杂控制）且把所有节点当作独立处理的 CMH "按脑分层"。
- **Corrected result：** Woolf OR（Extreme vs Center）= 2.3561，95% CI [2.2504, 2.4668]；但神经元-clustered permutation OR = 0.5320，双侧 p = 0.6607（NOT significant）。对比原始 CMH OR = 2.3561，p ≈ 0。一旦神经元（而非节点）是单位，升高的极端 Z omit 率在 origin 上不显著。
- **Post-correction verdict：** OVERTURNED —— 神经元-cluster permutation 检验不显著（p = 0.6607）；节点级 OR = 2.36 和 p ≈ 0 是把 ~640k 个非独立节点当作独立抽样所驱动，且边界/深度混杂未得到处理。
- **Corrected generalization：** PARTIAL / INCONCLUSIVE —— 794491：神经元 OR = nan，permutation p = 0.0002，但节点级 OR 为 1.0149 [0.905, 1.138]（无真实效应；微小的显著 permutation p 配合 nan clustered OR 不可解读为支持——INCONCLUSIVE）；794495：神经元-clustered OR = 2.2188，permutation p = 0.1364（尽管节点级 OR = 3.17，p ≈ 0 仍 NOT significant）。在正确的神经元级检验下，极端 Z omit 超额在任何体积上均不显著。

### 13. (Priority 0.253 · Surprise 0.284) 短 omission 间隙是被同一片段桥接的内部丢失；长间隙是真实终止。
- **Run：** run-4--ground-truth-error-annotations-revised-version_2026-06-20 · **ID：** 58 · **Belief：** Leaning True → Likely True (0.7083→0.8906) · **Direction：** Positive
- **Tested：** 短 omit 间隙是否两侧都被同一预测片段所夹（内部丢失），而长 omit 代表真正的片段边界。
- **Conclusion：** 在所分析的 omit 路径中，307 个是"已桥接"（相同夹持片段），4,298 个是"已断开"；已桥接间隙短得多（均值 18.71 µm，中位数 13.55 µm）于已断开间隙（均值 42.54 µm，中位数 20.16 µm），显著如此（Mann-Whitney U p = 1.95e-19）。短 omit 主要是伪影性的内部丢失。正向 surprisal（+0.284）。
- **Caveats：** 已桥接路径（n = 307）是 omit 路径的一小部分（~7%），因此关于已桥接类的结论建立在一个适度的子样本上。
- **Reproduction：** REPRODUCED（code: revised-loading）
- **Rerun result：** 记录值 307 个已桥接（中位数 13.55 µm）vs 4298 个已断开（中位数 20.16 µm），U=458554.0，p=1.9488e-19；重跑相同 → 完全一致。
- **Generalization：** GENERALIZES
- **Across datasets：** 已桥接 omit 路径在所有三个数据集上都显著短于已断开。origin (789202)：已桥接中位数 13.55 µm（n=307）vs 已断开 20.16 µm（n=4298），U=458554，p=1.95e-19。ds_794491：已桥接 10.99 µm（n=219）vs 已断开 15.23 µm（n=4353），p=2.75e-09。ds_794495：已桥接 12.17 µm（n=330）vs 已断开 16.24 µm（n=3815），p=1.20e-12。方向和显著性全程成立。
- **Verdict：** SOUND
- **Test：** 对 omit 路径长度做单侧 Mann-Whitney U（`alternative='less'`），已桥接（n=307）vs 已断开（n=4298）；U=458554.0，p=1.9488e-19；中位数 13.55 µm vs 20.16 µm。
- **Statistical issues：** 良定——分析单位是一个连通的 omit 路径组件，不同组件真正独立（不同于逐边检验），因此 MW 独立性成立；非参数 MW 是偏斜长度的正确选择。已桥接类（n=307，~7%）是适度的少数但功效充分。
- **Logic issues：** 无重大问题。结论（短 omit 是被同一片段桥接的内部丢失；长 omit 是真实终止）正是已桥接/已断开长度对比所支持的。
- **Verdict rationale：** 在独立单位上的正确检验、效应清晰、精确复现，且 GENERALIZES（三者上 p≤2.8e-09 且同方向）。SOUND。

### 14. (Priority 0.253 · Surprise 0.284) 高局部迂曲度与更多 split 错误相关（相关性小）。
- **Run：** run-4--ground-truth-error-annotations-revised-version_2026-06-20 · **ID：** 59 · **Belief：** Leaning True → Likely True (0.7083→0.8906) · **Direction：** Positive
- **Tested：** 高局部迂曲度（曲率，在 10 边窗口上）是否提高 split 错误可能性。
- **Conclusion：** 在 6,611 条 split 和 1,091,075 条正确边中，split 边有更高的中位数（1.1115 vs 1.0764）和均值（1.2163 vs 1.1175）迂曲度（Mann-Whitney U p ≈ 0）。效应在统计上压倒性，但点二列相关小（r = 0.0335），因此曲率是一个弱但真实的风险因子。正向 surprisal（+0.284）。
- **Caveats：** 效应量微小（r = 0.0335）；近零 p 值由 >1M 样本驱动，因此曲率本身解释的 split 错误方差很少。
- **Reproduction：** REPRODUCED（code: revised-loading）
- **Rerun result：** 记录值 6611 split / 1091075 correct，中位数 1.1115 vs 1.0764，U=4.5781e+09（p≈0），点二列 r=0.0335；重跑相同 → 完全一致。
- **Generalization：** GENERALIZES
- **Across datasets：** split 边在所有三个数据集上都比正确边更迂曲，p≈0，效应量同样小。origin (789202)：中位数 1.1115（split）vs 1.0764（correct），p≈0，r=0.0335。ds_794491：1.0860 vs 1.0597，p≈0，r=0.0607。ds_794495：1.0718 vs 1.0555，p≈0，r=0.0253。方向和显著性成立；效应保持小（r≈0.03-0.06）如同 origin。
- **Verdict：** MINOR
- **Test：** 对 10 边窗口迂曲度做单侧 Mann-Whitney U（`alternative='greater'`）加点二列相关；6,611 split vs 1,091,075 correct 边；U=4.5781e9，p≈0；点二列 r=0.0335（p≈0）；中位数 1.1115 vs 1.0764。
- **Statistical issues：** p≈0 完全由 >1M 样本量驱动；实际效应微不足道（r=0.0335 → ~0.1% 方差），因此曲率本身几乎解释不了什么。神经元内的边空间自相关，进一步膨胀名义显著性。关键的是，该分析 *报告* 了 r 并明确称效应"小"，因此它没有隐藏效应量问题。
- **Logic issues：** 无——结论被正确地限定为"一个弱但真实的风险因子"，与 r 值相符；无强或因果驱动的主张。
- **Verdict rationale：** 对一个微小但一致、可复现且 GENERALIZES（三者上 r≈0.03–0.06）的效应的诚实报告。显著性由 n 驱动使其保持 MINOR；它不是 FLAWED，因为效应量被披露且结论恰当地弱。
- **Corrected test：** 对 split−correct 中位数迂曲度差距做神经元-cluster permutation 检验（神经元 = 独立单位），加上带 cluster bootstrap CI 的 Cliff's delta，替换逐边 Mann-Whitney，后者的 p≈0 由 >1M 自相关边驱动。
- **Corrected result：** Cliff's delta = 0.2618，cluster bootstrap 95% CI [0.1583, 0.3674]（小但 CI 不含 0）；神经元-cluster permutation 单侧 p = 0.002（SIGNIFICANT；神经元级中位数差距 = 0.0829）。对比原始 MW p ≈ 0，点二列 r = 0.0335。方向和显著性在 cluster 修正下存续，效应确认为小。
- **Post-correction verdict：** UPHELD（弱效应）—— 在神经元-cluster permutation 检验下显著（p = 0.002），Cliff's delta CI 不含 0；原始"弱但真实的风险因子"表述恰好正确，并在正确检验下存续。
- **Corrected generalization：** GENERALIZES —— 794491：Cliff's delta = 0.2698 [0.235, 0.346]，permutation p = 0.0016（显著）；794495：Cliff's delta = 0.1973 [0.123, 0.253]，permutation p = 0.0002（显著）。在两个额外数据集上显著的小效应。

### 15. (Priority 0.253 · Surprise 0.284) Split 边比正确边略微更靠近 merge 位点。
- **Run：** run-4--ground-truth-error-annotations-revised-version_2026-06-20 · **ID：** 61 · **Belief：** Leaning True → Likely True (0.7083→0.8906) · **Direction：** Positive
- **Tested：** Split 边是否在空间上比正确边更靠近最近的 merge 位点（共定位的分割失败区域）。
- **Conclusion：** 在 6,805 条 split 和 1,109,034 条正确边中，split 边比正确边（均值 2,265.14 µm，中位数 1,956.93 µm）更靠近 merge 位点（均值 2,084.76 µm，中位数 1,794.64 µm），高度显著（Mann-Whitney U p = 3.50e-38；Welch's t p = 7.14e-27）。Split 和 merge 在局部化的失败区域共聚集。正向 surprisal（+0.284）。
- **Caveats：** 绝对中位数差异（~1,800 µm 中的 ~162 µm）较小；极端显著性来自 >1M 边样本，因此实际共定位较弱。
- **Reproduction：** REPRODUCED（code: revised-loading）
- **Rerun result：** 记录值 6805 split / 1109034 correct，split 中位数 1794.64 µm vs correct 1956.93 µm，U=3432660112.0（p=3.50e-38），Welch t=−10.7131（p=7.14e-27）；重跑相同 → 完全一致。
- **Generalization：** GENERALIZES
- **Across datasets：** split 边在所有三个数据集上都比正确边更靠近 merge 位点，显著。origin (789202)：split 中位数 1794.64 µm vs correct 1956.93 µm，U 检验 p=3.50e-38，Welch p=7.14e-27。ds_794491：818.04 vs 846.80 µm，p=1.98e-14，Welch p=2.18e-03（显著但差距最小，~29 µm）。ds_794495：1476.84 vs 1672.23 µm，p=2.91e-73，Welch p=4.29e-107。方向和显著性成立；实际差距小（如 caveats 中所标记），在 794491 上最窄。
- **Verdict：** MINOR
- **Test：** 对到最近 merge 位点的距离做单侧 Mann-Whitney U（`alternative='less'`）和单侧 Welch t；6,805 split vs 1,109,034 correct 边；MW p=3.50e-38，Welch t=−10.7131 p=7.14e-27；中位数 1794.64 vs 1956.93 µm。
- **Statistical issues：** 效应实践上微不足道——~1,900 µm 中位数中的 ~162 µm（≈8%）——极端 p 值仅由 >1.1M 正确边产生；边非独立且所有距离都参照同一个小的 merge 位点集，因此有效 n 远小于报告值。caveat 明确标记了小的绝对差距。
- **Logic issues：** 结论（"split 错误和 merge 位点在局部化失败区域空间聚集"）合理但强于一个 8% 中位数偏移所能保证的；聚集是从一个集中趋势的小位移主张的。
- **Verdict rationale：** 在方向/显著性上 GENERALIZES，但效应微小且功效过度（且在 794491 上缩至 ~29 µm）。诚实地加了注意事项，因此 MINOR 而非 MAJOR。
- **Corrected test：** 对 split−correct 中位数到 merge 距离差距做神经元-cluster permutation 检验（神经元 = 独立单位），加上带 cluster bootstrap CI 的 Cliff's delta，替换逐边 Mann-Whitney / Welch t，后者的 p≈0 来自 >1.1M 条全部参照同一小 merge 位点集的非独立边。
- **Corrected result：** Cliff's delta = −0.0740，cluster bootstrap 95% CI [−0.2816, 0.1356]（CI 跨越 0——与无效应不可区分）；神经元-cluster permutation 单侧 p = 0.6469（NOT significant；神经元级差距仅 49.42 µm）。对比原始 MW p = 3.50e-38，Welch p = 7.14e-27。一旦神经元是单位，"split 更靠近 merge" 效应消失。
- **Post-correction verdict：** OVERTURNED —— cluster-permutation 检验不显著（p = 0.6469）且 Cliff's delta CI 跨越 0；极端 p 值完全是非独立边的 n 膨胀伪影。
- **Corrected generalization：** DOES-NOT-GENERALIZE —— 794491：Cliff's delta = −0.0375 [−0.270, 0.150]（跨越 0），permutation p = 0.3913（不显著）；794495：Cliff's delta = −0.0966 [−0.317, 0.075]（跨越 0），permutation p = 0.3611（尽管朴素 p = 2.91e-73 仍不显著）。在正确检验下在所有三个体积上均不显著。

### 16. (Priority 0.253 · Surprise 0.284) Split 间隙一致地小——100% 小于 15 µm，99% 小于 ~6.5 µm。
- **Run：** run-4--ground-truth-error-annotations-revised-version_2026-06-20 · **ID：** 63 · **Belief：** Leaning True → Likely True (0.7083→0.8906) · **Direction：** Positive
- **Tested：** Split 错误是否是小的局部化间隙（>80% 的同神经元断开片段相隔 <15 µm），以确定修复工具的搜索半径。
- **Conclusion：** 所有 6,805 个 split 间隙都落在 15 µm 以下（100%），聚集在 3–5 µm 之间，最大离群值 ~9 µm；第 95/99 百分位为 ~5.85 µm 和 ~6.50 µm。修复智能体可以使用紧凑的 ~6.5 µm 搜索半径来捕获 ~99% 的 split，同时最小化假 merge 风险。正向 surprisal（+0.284）。这佐证了排名第一结果（#1）的 ~6.84 µm 阈值。
- **Caveats：** 无标注；与 ID 30 一致。80% 的主张被远超，表明先前的框定低估了 split 间隙有多紧凑。
- **Reproduction：** REPRODUCED（code: revised-loading）
- **Rerun result：** 记录值 6805 个 split 间隙，100.00% 小于 15 µm，第 95 百分位 ~5.85 µm / 第 99 百分位 ~6.50 µm；重跑相同 → 完全一致。
- **Generalization：** GENERALIZES
- **Across datasets：** split 间隙在所有三个数据集上都一致地紧凑。origin (789202)：100% < 15 µm，第 95 ~5.85 µm，第 99 ~6.50 µm（n=6805）。ds_794491：100% < 15 µm，第 95 ~5.79 µm，第 99 ~6.44 µm（n=7847）。ds_794495：100% < 15 µm，第 95 ~5.80 µm，第 99 ~6.41 µm（n=7988）。~6.5 µm 搜索半径建议在各数据集间基本一致。
- **Verdict：** SOUND
- **Test：** 描述性分布摘要——6,805 个 split 间隙中小于 15 µm 的比例及第 95/99 百分位；"Percentage of split gaps < 15 µm: 100.00%"，第 95 ≈5.85 µm，第 99 ≈6.50 µm。无推断检验。
- **Statistical issues：** 无——这是一个百分位/ECDF 描述，而非检验，因此无分布假设适用。Split 间隙定义（具有不同非零标签的相邻同神经元节点）是自然的，且匹配 #1（id 30）构造。
- **Logic issues：** 假设下限（"超过 80% 小于 15 µm"）被远超（100%），因此该主张轻松成立；~6.5 µm 搜索半径建议直接来自第 99 百分位。
- **Verdict rationale：** 一个纯描述性、无歧义的结果，可复现且 GENERALIZES（三者上 100% <15 µm，第 99 ~6.4–6.5 µm）并佐证 id 30。SOUND。

### 17. (Priority 0.253 · Surprise 0.284) Split 错误在测地线上比正确边更靠近分支点聚集。
- **Run：** run-4--ground-truth-error-annotations-revised-version_2026-06-20 · **ID：** 64 · **Belief：** Leaning True → Likely True (0.7083→0.8906) · **Direction：** Positive
- **Tested：** 到 GT 分支点的临近性是否是 split 错误的风险因子（复杂的局部几何使重建碎片化）。
- **Conclusion：** 在 6,805 条 split 和 1,109,034 条正确边中，到最近分支点的平均测地线距离对 split 为 516.30 µm，对正确边为 710.59 µm（Mann-Whitney U p = 1.32e-197）。Split 集中在分支点附近，且缺乏正确边中看到的长距离离群值，支持一个分支点风险信号。正向 surprisal（+0.284）。
- **Caveats：** 复核字段为 "N/A"（未记录独立审计），因此忠实实现确认比其他条目弱。
- **Reproduction：** REPRODUCED（code: revised-loading）
- **Rerun result：** 记录值 6805 split / 1109034 correct，到分支的平均距离 516.30 µm（split）vs 710.59 µm（correct），U=2979025451.5，p=1.3239e-197；重跑相同 → 完全一致。
- **Generalization：** GENERALIZES
- **Across datasets：** split 边在所有三个数据集上都在测地线上比正确边更靠近分支点，显著。origin (789202)：均值 516.30 µm（split）vs 710.59 µm（correct），p=1.32e-197（差距 ~194 µm）。ds_794491：384.29 vs 389.49 µm，p=5.16e-67（显著但实践上差距微小 ~5 µm）。ds_794495：423.67 vs 506.89 µm，p=2.97e-27（差距 ~83 µm）。方向和显著性成立；效应幅度在 794491 上弱得多。
- **Verdict：** MINOR
- **Test：** 对到最近分支点的测地线（Dijkstra，从所有分支点多源）距离做双侧 Mann-Whitney U；6,805 split vs 1,109,034 correct 边；U=2979025451.5，p=1.3239e-197；均值 516.30 vs 710.59 µm。
- **Statistical issues：** 测地线距离方法实现良好（虚节点多源 Dijkstra），但对比部分由正确边尾部驱动：split 边只是缺乏长距离离群值（max ~8,500 vs ~13,000 µm），因此部分均值差距反映范围截断而非在分支附近的集中。大 n 膨胀显著性，且 1.1M 正确边非独立。记录的 `review` 字段为 "N/A"（无独立审计）。
- **Logic issues：** 结论（分支点临近性是 split 风险因子）合理，但不应被解读为分支点 *导致* split——比较是观察性的，且效应在 794491 上崩塌至 ~5 µm。
- **Verdict rationale：** 在方向/显著性上 GENERALIZES，但仅在三个体积中的两个上有有意义的效应（~194 µm 和 ~83 µm；794491 上 ~5 µm）。设计合理、效应适度/可变、显著性由 n 驱动——MINOR。
- **Corrected test：** 对 split−correct 中位数到分支测地线距离差距做神经元-cluster permutation 检验（神经元 = 独立单位），加上带 cluster bootstrap CI 的 Cliff's delta，替换在 1.1M 非独立边上的逐边 Mann-Whitney。
- **Corrected result：** Cliff's delta = −0.2184，cluster bootstrap 95% CI [−0.3692, −0.0309]（小到中，CI 不含 0）；神经元-cluster permutation 单侧 p = 0.09598（在 0.05 处 NOT significant；神经元级差距 = −116.52 µm）。对比原始 MW p = 1.3239e-197。效应量 CI 不含 0，但 cluster-permutation p 在 origin 上恰好错过显著性。
- **Post-correction verdict：** WEAKENED —— Cliff's delta CI [−0.369, −0.031] 仍表明 split 略微更靠近分支点，但神经元-cluster permutation 检验在 origin 上不再显著（p = 0.096）；p = 1.3e-197 被大幅 n 膨胀。
- **Corrected generalization：** PARTIAL —— 794491：Cliff's delta = −0.1356 [−0.246, −0.005]，permutation p = 0.009198（显著）；794495：Cliff's delta = −0.0762 [−0.150, 0.016]（CI 跨越 0），permutation p = 0.1342（不显著）。在正确检验下仅在两个额外数据集之一上显著，且全程效应量小。

### 18. (Priority 0.253 · Surprise 0.284) Split 错误发生在比正确片段更迂曲的片段上（5 跳窗口）。
- **Run：** run-4--ground-truth-error-annotations-revised-version_2026-06-20 · **ID：** 73 · **Belief：** Leaning True → Likely True (0.7083→0.8906) · **Direction：** Positive
- **Tested：** Split 错误是否落在比笔直片段更高度迂曲（弯曲）的片段上，在 5 跳骨架窗口上测量。
- **Conclusion：** 在 1,109,034 条正确和 6,805 条 split 边中，split 边有更高的中位数（1.1132 vs 1.0801）和均值（1.2062 vs 1.1194）迂曲度，单侧 Mann-Whitney U 显著（p = 1.66e-276）。这在不同窗口大小上复现 ID 59：急转弯系统性地挑战连续追踪。正向 surprisal（+0.284）。
- **Caveats：** 与 ID 59 一样，中位数差异在绝对值上小；极端 p 值反映样本量，而非大的效应幅度。两个迂曲度检验共享大部分底层数据，因此它们不是完全独立的确认。
- **Reproduction：** REPRODUCED（code: revised-loading）
- **Rerun result：** 记录值 1109034 correct / 6805 split，中位数 1.0801 vs 1.1132，U=4714207979.0，p=1.6599e-276；重跑相同 → 完全一致。
- **Generalization：** GENERALIZES
- **Across datasets：** split 边在所有三个数据集上都比正确边更迂曲（5 跳窗口），显著。origin (789202)：split 中位数 1.1132 vs correct 1.0801，p=1.66e-276。ds_794491：1.0854 vs 1.0617，p≈0。ds_794495：1.0735 vs 1.0575，p=6.19e-192。在各数据集间复现 id 59；效应小但在方向和显著性上一致。
- **Verdict：** MINOR
- **Test：** 对 5 跳窗口迂曲度做单侧 Mann-Whitney U（`alternative='greater'`）；1,109,034 correct vs 6,805 split 边；U=4714207979.0，p=1.6599e-276；中位数 1.1132（split）vs 1.0801（correct）。
- **Statistical issues：** 与 id 59 相同的微小效应/功效过度模式——中位数差距 ~0.03，p≈0 来自 >1.1M 边；不同于 id 59，此运行 *未* 报告效应量统计（无点二列 r），因此实际的微小性在分析文本中不那么可见。边非独立。此外，本项和 id 59 共享大部分相同的底层数据，因此它们不是彼此的独立确认。
- **Logic issues：** 结论（"压倒性的统计证据"）倚靠 p 值并省略了 id 59 所包含的效应量注意事项，略微夸大了实践重要性；方向性主张本身正确。
- **Verdict rationale：** 方向可复现且 GENERALIZES，但它是 id 59 的近重复，具有相同的微不足道效应，且呈现时没有效应量限定——MINOR（显著性由 n 驱动，冗余确认）。
- **Corrected test：** 对 split−correct 中位数 5 跳迂曲度差距做神经元-cluster permutation 检验（神经元 = 独立单位），加上带 cluster bootstrap CI 的 Cliff's delta（id 73 省略的效应量），替换逐边 Mann-Whitney，后者的 p≈0 来自 >1.1M 非独立边。
- **Corrected result：** Cliff's delta = 0.2451，cluster bootstrap 95% CI [0.1511, 0.3419]（小但 CI 不含 0）；神经元-cluster permutation 单侧 p = 0.0007998（SIGNIFICANT；神经元级差距 = 0.0725）。对比原始 MW p = 1.6599e-276（未报告效应量）。方向和显著性存续；现报告的效应量小。
- **Post-correction verdict：** UPHELD（弱效应）—— 在神经元-cluster permutation 检验下显著（p = 0.0008），Cliff's delta CI 不含 0；与 id 59 镜像。结论成立，但效应小，正如现添加的效应量明确显示。
- **Corrected generalization：** GENERALIZES —— 794491：Cliff's delta = 0.2455 [0.210, 0.310]，permutation p = 0.0014（显著）；794495：Cliff's delta = 0.1856 [0.139, 0.246]，permutation p = 0.0002（显著）。在两个额外数据集上显著的小效应。

### 19. (Priority 0.253 · Surprise 0.284) Merge 位点携带强的局部分支密度特征（ROC-AUC 0.92）。
- **Run：** run-4--ground-truth-error-annotations-revised-version_2026-06-20 · **ID：** 84 · **Belief：** Leaning True → Likely True (0.7083→0.8906) · **Direction：** Positive
- **Tested：** Merge 位点在 UNet 片段图中是否比同一片段上的非 merge 区域表现出更高的局部分支密度（用于自动解析的几何特征）。
- **Conclusion：** 在 67 个 merge 位点中，15 µm 内的平均局部分支密度为 1.13 个分支，而在同一片段上的匹配对照为 0.04（配对 t = 13.35，p = 2.03e-20；Wilcoxon W = 21.0，p = 1.19e-11），ROC-AUC 0.9236。虚假的局部分支是一个用于自动化校对的高度预测性 merge 特征。正向 surprisal（+0.284）。
- **Caveats：** 仅有 67 个 merge 位点；强 ROC-AUC 建立在一个小的正类上，因此样本外预测能力不确定。
- **Reproduction：** REPRODUCED（code: revised-loading）
- **Rerun result：** 记录值 67 个 merge 位点，平均密度 1.13 vs 0.04，配对 t=13.3482（p=2.0302e-20），Wilcoxon W=21.0（p=1.1942e-11），ROC-AUC=0.9236；重跑相同 → 完全一致（stderr 中仅一个 matplotlib 弃用警告）。
- **Generalization：** GENERALIZES
- **Across datasets：** merge 位点在所有三个数据集上都显示出比对照高得多的局部分支密度，显著，且 ROC-AUC 强。origin (789202)：1.13 vs 0.04 branches/15µm，配对 t=13.35，p=2.03e-20，ROC-AUC=0.9236（n=67）。ds_794491：1.10 vs 0.08，t=12.14，p=2.97e-20，ROC-AUC=0.8825（n=86）。ds_794495：1.02 vs 0.05，t=13.46，p=1.65e-24，ROC-AUC=0.8857（n=105）。AUC 略降（~0.88）但特征稳健。
- **Verdict：** MINOR
- **Test：** 对 67 个 merge 位点 vs 匹配的片段上对照的局部分支密度（15 µm 内的分支）做配对 t 检验和 Wilcoxon 符号秩，加上 ROC-AUC；配对 t=13.3482（p=2.0302e-20），Wilcoxon W=21.0（p=1.1942e-11），ROC-AUC=0.9236；均值 1.13 vs 0.04。
- **Statistical issues：** 配对设计（merge 位点 vs *同一* 片段上的对照）是正确选择，且 Wilcoxon 佐证 t 检验，因此尽管 n=67 显著性仍稳健。唯一的真实问题：ROC-AUC=0.9236 是在用于定义对比的同一 merge/对照点上做样本内计算的（无留出拆分），因此它夸大了真实的样本外预测能力——caveat 标记了小的正类。
- **Logic issues：** 结论（局部分支密度是用于自动校对的预测性 merge 特征）通过引用样本内 AUC 作为"高度预测性"略微越界；*差异* 本身被稳固地确立。
- **Verdict rationale：** 正确的配对检验、效应大、可复现，且 GENERALIZES（AUC 0.88–0.92）。仅因样本内 AUC 乐观和小 n 下调到 MINOR。

### 20. (Priority 0.253 · Surprise 0.284) Omit 错误集中在终末（远端）分支上，约为内部率的 ~1.8×。
- **Run：** run-4--ground-truth-error-annotations-revised-version_2026-06-20 · **ID：** 85 · **Belief：** Leaning True → Likely True (0.7083→0.8906) · **Direction：** Positive
- **Tested：** Omit 错误是否不成比例地命中终末分支（以叶节点结尾的路径）而非内部片段。
- **Conclusion：** 终末边有 4.38% 的 omit 率（23,792/543,477），而内部边为 2.41%（20,898/865,568）——81% 的相对增加，χ² = 4189.94，p < 1e-300。模型难以追踪细的远端末端，因此校对者应优先处理终末分支以补缺失纤维。正向 surprisal（+0.284）。
- **Caveats：** 无标注；复核确认忠实的分类和分析。
- **Reproduction：** REPRODUCED（code: revised-loading）
- **Rerun result：** 记录值 terminal 4.38%（23792/543477）vs internal 2.41%（20898/865568），χ²=4189.9364，p≈0；重跑相同 → 完全一致。
- **Generalization：** GENERALIZES
- **Across datasets：** 终末边在所有三个数据集上都显示出比内部更高的 omit 率，显著。origin (789202)：terminal 4.38% vs internal 2.41%，χ²=4189.9，p≈0（+81% 相对）。ds_794491：terminal 4.94% vs internal 3.82%，χ²=425.4，p=1.63e-94（+29% 相对）。ds_794495：terminal 2.47% vs internal 1.82%，χ²=692.0，p=1.66e-152（+36% 相对）。方向和显著性成立；相对超额在额外数据集上较小。
- **Verdict：** MINOR
- **Test：** 对终末 vs 内部边 × omit/非 omit 做 Pearson 卡方（`scipy.stats.chi2_contingency`）；23792/519685（terminal）vs 20898/844670（internal）；χ²=4189.9364，p≈0；omit 率 4.38% vs 2.41%（+81% 相对）。
- **Statistical issues：** 边非独立（终末路径是连续的边运行），因此 χ² 夸大有效 n；期望单元计数大，因此检验在其他方面良构。+81% 相对超额是一个实质性、非平凡的效应，经受住假设关切。
- **Logic issues：** 无重大问题——结论（优先处理终末/远端分支以补缺失纤维）来自清晰的率差，且不在"模型难以处理细的远端末端"之外过度主张机制。
- **Verdict rationale：** 效应大、已复现、GENERALIZES（三者上 +29% 到 +81% 相对，p≤1.6e-94）。边非独立是唯一问题，在此幅度下无足轻重——MINOR。

## 复现 — 摘要

**所用数据集 pkl：** `/allen/programs/mindscope/workgroups/auto-model/zihan.zhang/exaspim-agent/exa-spim-agent/cache/dataset_cache_789202_mcl100_add.pkl`

**分项（在 n_rerun = 20 中）：** REPRODUCED 19 · DIVERGED 1 · FAILED 0。全部 20 个都在 `revised`-loading 代码上运行（直接 `$RERUN_PKL` 加载）；0 个在记录代码上。全部 exit 0，无超时，无环境失败。

**未复现：**
- **Entry 5 · id 24 — DIVERGED（分析）：** merge 严重度纤维检验。记录值 n=24 个 2 神经元 merge / n=3 个超级 merge，Mann-Whitney U=0.0，p=5.8861e-03（显著）；在所提供 pkl 上重跑仅找到 n=8 / n=1，U=0.0，p=2.2222e-01（不显著——p 越过 0.05）。中位数纤维值（6.21 vs 35.19 mm）不变，但该数据集上远更小的 merge 计数使检验跌破显著性，推翻了所记录的结论。这是一个数据/分析分歧（不同的 merge 检测计数），而非加载或环境失败。

| Entry | id | Verdict | code_source |
|------:|---:|---------|-------------|
| 1 | 30 | REPRODUCED | revised |
| 2 | 27 | REPRODUCED | revised |
| 3 | 21 | REPRODUCED | revised |
| 4 | 139 | REPRODUCED | revised |
| 5 | 24 | DIVERGED | revised |
| 6 | 32 | REPRODUCED | revised |
| 7 | 33 | REPRODUCED | revised |
| 8 | 36 | REPRODUCED | revised |
| 9 | 37 | REPRODUCED | revised |
| 10 | 39 | REPRODUCED | revised |
| 11 | 43 | REPRODUCED | revised |
| 12 | 45 | REPRODUCED | revised |
| 13 | 58 | REPRODUCED | revised |
| 14 | 59 | REPRODUCED | revised |
| 15 | 61 | REPRODUCED | revised |
| 16 | 63 | REPRODUCED | revised |
| 17 | 64 | REPRODUCED | revised |
| 18 | 73 | REPRODUCED | revised |
| 19 | 84 | REPRODUCED | revised |
| 20 | 85 | REPRODUCED | revised |

**加载修订：** 每个展示的记录（全部 20 个）都在 revised-loading 代码上运行，该代码用直接 `$RERUN_PKL` 加载替换了原始数据集搜索。那些记录代码曾通过 glob/硬编码相对路径搜索、否则会错过所提供 pkl 的记录包括 ids 64 和 73（"Found 1 dataset files" / "No dataset files found matching *_add.pkl" 关卡）以及 ids 139、43、73（硬编码 `../data/...` 路径）——现在全部直接加载编排器 pkl。没有记录命中 ENVIRONMENT 失败；运行器被抑制的软件包安装通知（ids 21、32、84）是良性的空操作，而非失败。

## 泛化 — 摘要

**测试的额外数据集（origin = 789202）：** `cache/dataset_cache_794491_mcl100_add.pkl` 和 `cache/dataset_cache_794495_mcl100_add.pkl`。全部 20 个复现的脚本在两个额外 pkl 上都运行到 exit 0（无超时，无加载失败）——因此没有条目在执行依据上是 INCONCLUSIVE。

**分项（在 20 中）：** GENERALIZES 14 · PARTIAL 4 · DOES-NOT-GENERALIZE 2 · INCONCLUSIVE 0。

**未完全泛化：**
- **Entry 3 · id 21 — DOES-NOT-GENERALIZE：** 接近零的 Z 主导 vs XY 主导错误率发现在两个额外数据集上变得强烈 *显著*，方向 *相反*（794491：Z 更高，p=5.21e-38；794495：Z 更低，p=4.62e-88）。Origin 干净的零结论是数据集特异的。
- **Entry 4 · id 139 — DOES-NOT-GENERALIZE：** 分支阶数 → split 风险斜率在两个额外数据集上翻转符号（origin coef=+0.0194；794491 coef=−0.0157；794495 coef=−0.0063），全部显著。"更深的分支 split 更多" 在 origin 之外反转。
- **Entry 2 · id 27 — PARTIAL：** "z 对齐不显著" 主张在 794491 上成立（p=0.175，但符号翻转 +），却在 794495 上被反驳（p=6.75e-06，显著 *减少*）；方向不稳定（−/+/−）。
- **Entry 5 · id 24 — PARTIAL：** 超级 merge 纤维检验样本匮乏——origin-rerun 不显著（p=0.222），794491 有零个超级 merge（不可检验），794495 显著（p=6.15e-03）。方向（超级 > 2 神经元）仅在存在 ≥1 个超级 merge 处成立。
- **Entry 8 · id 36 — PARTIAL：** omit-near-merge 聚集在 794495 上成立（p=9.78e-226），但在 794491 上 *反转*（omit 更远；单侧 p=1.0000）。
- **Entry 12 · id 45 — PARTIAL：** 极端 Z omit 超额在 794495 上成立（OR=3.17，p≈0），但在 794491 上消失（OR=1.0149，p=0.7996）。

**INCONCLUSIVE：** 无。

**综合分析：** 稳健的、与数据集无关的结论是局部拓扑/形态学和校对启发式发现：~6.5 µm split 间隙临近阈值（id 30、63）、角度惯性（id 33）、A* split 修复（id 39）、分支点 omission 超额（id 32）、终末分支 omission 超额（id 85）、迂曲度 → split 风险（id 59、73）、split 空间聚集（id 37）、巨型 merge 纤维（id 43）、桥接-vs-断开 omit 长度（id 58）、merge 分支密度特征（id 84），以及 split-near-merge / split-near-branch 临近性（id 61、64）——全部 14 个在两个额外体积间保持方向 + 显著性（少数效应量较弱，例如 794491 上的 id 17/64）。相比之下，每个与 *全局成像轴 / Z 效应* 或微小样本 merge 统计相关的发现都脆弱：两个各向异性相关结果（id 21、27）和分支阶数斜率（id 139）要么翻转符号，要么在数据集间以不一致方向变得显著，而依赖少数 merge 位点或体积特异 Z 极值的 merge/深度发现（id 24、36、45）仅在两个额外数据集之一上成立。仅有两个额外数据集，证据是适度的——14 个 GENERALIZES 判定有充分支持（两者上一致），但 PARTIAL/DOES-NOT 判定标记出朝向和小样本驱动的结论很可能是体积特异的，而非分割流水线的一般属性。

| Entry | id | Verdict | 依据（一行） |
|------:|---:|---------|------------------|
| 1 | 30 | GENERALIZES | ROC-AUC 0.998/0.989/0.995，三者上 thr ~6.5 µm |
| 2 | 27 | PARTIAL | 非显著在 794491 上成立（符号翻转 +）；794495 显著减少 p=6.75e-06 |
| 3 | 21 | DOES-NOT-GENERALIZE | 两个额外数据集在相反方向上显著（p=5.2e-38 / 4.6e-88） |
| 4 | 139 | DOES-NOT-GENERALIZE | 分支阶数斜率在两个额外数据集上翻转符号（+0.019 → −0.016 / −0.006） |
| 5 | 24 | PARTIAL | 样本匮乏：origin p=0.222，794491 无超级 merge，794495 p=6.15e-03 |
| 6 | 32 | GENERALIZES | 分支 omit > 线性，三者上（χ² 1568/194/975，p≤5e-44） |
| 7 | 33 | GENERALIZES | true ~153° vs false ~90°，三者上 ROC-AUC 0.93 |
| 8 | 36 | PARTIAL | 794495 成立（p=9.8e-226）；794491 反转（p=1.0） |
| 9 | 37 | GENERALIZES | split 邻居超额三者上 p≈0 |
| 10 | 39 | GENERALIZES | A* 解决 86/85/90% 的 split，全部 ≫40% |
| 11 | 43 | GENERALIZES | merge 片段 ~10-30× 更大，t≥16，p≤7e-25 |
| 12 | 45 | PARTIAL | 794495 成立（OR=3.17）；794491 消失（OR=1.01，p=0.80） |
| 13 | 58 | GENERALIZES | 桥接短于断开，三者上（p≤2.8e-09） |
| 14 | 59 | GENERALIZES | split 更迂曲，p≈0，三者上小 r≈0.03-0.06 |
| 15 | 61 | GENERALIZES | split 更靠近 merge，三者上（p≤2e-14） |
| 16 | 63 | GENERALIZES | 100% 间隙 <15 µm，三者上第 99 百分位 ~6.4-6.5 µm |
| 17 | 64 | GENERALIZES | split 更靠近分支，三者上（p≤5e-67），794491 上微小差距 |
| 18 | 73 | GENERALIZES | split 更迂曲（5 跳），三者上（p≤6e-192） |
| 19 | 84 | GENERALIZES | merge 分支密度特征，ROC-AUC 0.92/0.88/0.89 |
| 20 | 85 | GENERALIZES | 终末 omit > 内部，三者上（χ² 4190/425/692） |

## 统计验证 — 摘要

**已审计：** 全部 20 个已排序假设，根据记录的 `code` / `codeOutput` / `analysis` 加上已折叠的复现和泛化数字判断（无实验重跑）。

**Verdict 分项（5 级方案 SOUND | WEAK | MINOR | MAJOR | CRITICAL；20 个各计一次）：**

- **SOUND — 5：** entries **1 (id 30)、7 (id 33)、11 (id 43)、13 (id 58)、16 (id 63)**——干净的设计，效应大/无歧义，完全泛化（30/63 的描述性可分性；33 的近乎不相交角度分布上的 KS；43 的独立片段对数长度上的 Welch t；58 的独立 omit 路径组件上的 Mann-Whitney）。
- **WEAK — 0。**
- **MINOR — 9：** entries **6 (id 32)、9 (id 37)、10 (id 39)、14 (id 59)、15 (id 61)、17 (id 64)、18 (id 73)、19 (id 84)、20 (id 85)**——方向合理、可复现且泛化，但每个携带一个次要问题（边/节点非独立性膨胀显著性、效应量微不足道但功效过度的检验、样本内 ROC-AUC 乐观，或仿真-vs-真实越界）。
- **MAJOR — 6：** entries **2 (id 27)、3 (id 21)、4 (id 139)、5 (id 24)、8 (id 36)、12 (id 45)**——每个都有一个具体的检验错误（下文详述），实质性地削弱了所记录的结论。
- **CRITICAL — 0。**

**确定计数：SOUND 5 · WEAK 0 · MINOR 9 · MAJOR 6 · CRITICAL 0 = 20。**

**六个 MAJOR 发现及其具体检验错误（科学家不应照原样信任的发现）：**

- **Entry 2 · id 27** —— p 值通过将变分贝叶斯后验均值/SD 当作频率派 Wald z（`p=2*(1-Φ(coef/sd))`）制造；VB 后验不是抽样分布，因此所报告的 p=0.058 是无效的显著性检验，且它被用来支持一个零结论（"z 轴不是主要驱动因素"）。修复：重拟合一个恰当的频率派混合 GLM（或报告可信区间，而非 p），在全数据而非 20k 子样本上，并使用 clustered/robust SE。
- **Entry 3 · id 21** —— 在 1.4M 条共享节点（自相关）的边上的卡方独立性，且"未能拒绝（p=0.060）"被升级为"反驳成像各向异性偏置"；DOES-NOT-GENERALIZE（两个额外数据集显著，符号相反）。修复：使用按神经元的 clustered/permutation 检验，且绝不把不显著的 p 解读为无效应的证明。
- **Entry 4 · id 139** —— `norm_thickness` 控制方差为零并被剔除，因此假设的"独立于纤维粗细"条款结构上不可检验；边非独立；斜率在两个额外数据集上翻转符号。修复：在主张粗细独立性之前获得一个真实的粗细协变量（此 pkl 中半径是常数）。
- **Entry 5 · id 24** —— n=3 个超级 merge 的 Mann-Whitney（U=0.0，p=0.0059），在重跑时 DIVERGES 到 n=1 / p=0.222，且检验 *总* 纤维而假设是 *每神经元*。修复：在任何推断之前做每神经元归一化和远更多的超级 merge 实例。
- **Entry 8 · id 36** —— p=1.6e-66 来自 49k 个仅参照 67 个锚点的非独立节点上的 ~8% 中位数偏移（~1,800 µm 中的 147 µm）；方向在 794491 上 *反转*（p=1.0000）。修复：报告效应量、使用独立单位，并视为不泛化。
- **Entry 12 · id 45** —— CMH "控制脑" 在 origin 上只有一个分层（无真实控制）；"极端 Z" 将光学深度与 FOV 边界截断混淆；因果光学衰减主张；效应在 794491 上消失（OR=1.01，p=0.80）。修复：将边界与深度分离并放弃因果语言。

### 多重比较（Benjamini–Hochberg FDR）

从 20 个假设中各收集了单个头条 p 值。三个条目报告 **无推断 p**——1/id 30（ROC-AUC）、10/id 39（成功率 vs 40% 标准）、16/id 63（百分位）是描述性的——剩下 **m = 17 个数值 p 值**。报告 p≈0 的条目被限定在 1e-300；id 139 的 "p<0.001" 录入为 1e-3。

BH 在 α=0.05（秩 k，临界值 k/m·α）：p ≤ k/m·α 的最大秩是 **rank 15（entry 5 / id 24，p=5.886e-03 ≤ 0.0441）**，因此 BH 阈值为 p* = 5.886e-03，且 **17 个中 15 个存续**。

| rank | entry · id | test | p-value | crit (k/m·α) | survives FDR? |
|----:|-----------|------|--------:|-------------:|:-------------:|
| 1 | 6 · 32 | chi-square | ~1e-300 | 0.0029 | YES |
| 2 | 7 · 33 | KS | ~1e-300 | 0.0059 | YES |
| 3 | 9 · 37 | Mann-Whitney | ~1e-300 | 0.0088 | YES |
| 4 | 12 · 45 | CMH | ~1e-300 | 0.0118 | YES |
| 5 | 14 · 59 | Mann-Whitney / point-biserial | ~1e-300 | 0.0147 | YES |
| 6 | 20 · 85 | chi-square | ~1e-300 | 0.0176 | YES |
| 7 | 18 · 73 | Mann-Whitney | 1.66e-276 | 0.0206 | YES |
| 8 | 17 · 64 | Mann-Whitney | 1.32e-197 | 0.0235 | YES |
| 9 | 8 · 36 | Mann-Whitney | 1.60e-66 | 0.0265 | YES |
| 10 | 15 · 61 | Mann-Whitney | 3.50e-38 | 0.0294 | YES |
| 11 | 11 · 43 | Welch t | 7.32e-25 | 0.0324 | YES |
| 12 | 19 · 84 | paired t | 2.03e-20 | 0.0353 | YES |
| 13 | 13 · 58 | Mann-Whitney | 1.95e-19 | 0.0382 | YES |
| 14 | 4 · 139 | logistic (branch_order) | 1.0e-03 | 0.0412 | YES |
| 15 | 5 · 24 | Mann-Whitney | 5.89e-03 | 0.0441 | YES |
| 16 | 2 · 27 | mixed-GLM Wald (invalid) | 5.82e-02 | 0.0471 | no |
| 17 | 3 · 21 | chi-square | 6.02e-02 | 0.0500 | no |

**FDR 解读：** 两个未通过 BH 的发现（entry 2 / id 27 在 p=0.058 和 entry 3 / id 21 在 p=0.060）正是两个各向异性 *零* 结果——它们本已不显著，因此未通过 BH 与其"未能拒绝"状态一致，而非额外问题；它们的实质问题是缺乏证据的逻辑和无效的 Wald 构造（id 27），而非多重性。Entry 5 / id 24（p=5.886e-03）仅名义上通过 BH，且是边缘秩结果；鉴于其 DIVERGED 复现（重跑 p=0.222），尽管它越过了 FDR 线也应被视为不稳健。Entry 4 / id 139 通过 FDR，但存续的斜率在数据集间翻转符号，因此 FDR 存续不挽救其泛化失败。13 个大效应发现（秩 1–13）以巨大余量通过 FDR；对其中若干个（id 36、59、61、64、73、85）而言，存续由样本量而非效应幅度驱动，因此 FDR 存续不应被解读为实践重要性——见各条目效应量注记。

## 已排除（无 surprisal 分数）

辅助程序丢弃了 2 个缺少 surprisal 分数的假设（`n_dropped_missing_surprisal` = 2）：
- run-4--ground-truth-error-annotations-revised-version_2026-06-20 · ID 10
- run-4--ground-truth-error-annotations-revised-version_2026-06-20 · ID 41

## 统计检验修正 — 摘要

**标记需检验修复：** 11 个假设（ids 21、24、27、36、37、45、59、61、64、73、139）。每个都收到一个修正脚本，用恰当的分析单位检验——几乎总是一个 **神经元-cluster permutation 检验**（神经元 = 独立单位）加上一个 **带 bootstrap CI 的 Cliff's delta / rank-biserial 效应量**，或一个 cluster-robust 回归——替换了有缺陷的检验（假定逐边/逐节点独立、无效的 VB-as-Wald p、错误的构念，或无效应量的 n 膨胀 p）。修正结果由驱动程序重新测量（全部 11 个在 origin 和两个额外 pkl 上都运行到 exit 0）。

**修正后分项：UPHELD 5 · WEAKENED 1 · OVERTURNED 5。**

| id | Entry | 原始头条 | 修正头条（正确检验） | 修正后 verdict | 修正后泛化 |
|---:|------:|-------------------|---------------------------------|:------------------------|:-------------------------|
| 27 | 2 | Z 对齐 NOT 显著提高错误风险（无效 VB-Wald p=0.058，OR=0.80） | Neuron-permutation p=0.5721——无显著 z 对齐效应（有效零结论） | **UPHELD**（零） | PARTIAL（perm 零在全部 3 个；cluster-robust GLM 标记 794495 p=2e-11） |
| 21 | 3 | 无显著 Z-vs-XY 错误率差异（χ²=3.53，p=0.060） | RR=1.018 CI[0.999,1.037]，cluster-perm p=0.8136——微不足道，不显著 | **UPHELD**（零，origin） | PARTIAL（794491 perm p=0.097；794495 perm p=5e-4，Z *更低*） |
| 139 | 4 | Split 风险随分支阶数上升（z=16.4，p<0.001） | Neuron-permutation p=0.4697；cluster-robust z=1.99 CI 下界=1.0003 | **OVERTURNED** | DOES-NOT-GENERALIZE（perm n.s. 全部 3 个；两个额外数据集符号翻转） |
| 24 | 5 | 超级 merge 覆盖比 2 神经元 merge 更多纤维（总纤维，U=0，p=0.0059） | 每神经元纤维，U=8，p=0.2222，n_super=1（不可靠） | **OVERTURNED** | INCONCLUSIVE/PARTIAL（794491 无超级 merge；794495 p=0.025） |
| 36 | 8 | Omit 节点更靠近 merge 聚集（U=1.1e9，p=1.6e-66） | Cliff's δ=−0.064 CI[−0.10,−0.03]，neuron-perm p=0.4771 | **OVERTURNED** | DOES-NOT-GENERALIZE（perm n.s. 两个额外数据集；794491 反转） |
| 37 | 9 | Split 边有更多邻近 split（均值 0.97 vs 0.10，MW p≈0） | Cliff's δ=0.49 CI[0.44,0.56]，neuron-perm p=0.0002 | **UPHELD** | GENERALIZES（perm p≤4e-4，两个额外数据集 δ 0.52–0.60） |
| 45 | 12 | 极端 Z omit 超额（CMH OR=2.36，p≈0） | Neuron-clustered OR=0.53，neuron-perm p=0.6607 | **OVERTURNED** | PARTIAL/INCONCLUSIVE（794491 nan OR；794495 perm p=0.136 n.s.） |
| 59 | 14 | 高迂曲度 → 更多 split（MW p≈0，r=0.034） | Cliff's δ=0.26 CI[0.16,0.37]，neuron-perm p=0.002 | **UPHELD**（弱） | GENERALIZES（perm p≤0.0016 两个额外数据集，小 δ） |
| 61 | 15 | Split 边更靠近 merge 位点（MW p=3.5e-38） | Cliff's δ=−0.074 CI[−0.28,0.14]（跨越 0），neuron-perm p=0.6469 | **OVERTURNED** | DOES-NOT-GENERALIZE（perm n.s. 两个额外数据集；CI 跨越 0） |
| 64 | 17 | Split 在分支点附近聚集（MW p=1.3e-197） | Cliff's δ=−0.22 CI[−0.37,−0.03]，neuron-perm p=0.096 | **WEAKENED** | PARTIAL（794491 perm p=0.009；794495 perm p=0.134 n.s.） |
| 73 | 18 | Split 在更迂曲（5 跳）片段上（MW p=1.7e-276） | Cliff's δ=0.25 CI[0.15,0.34]，neuron-perm p=0.0008 | **UPHELD**（弱） | GENERALIZES（perm p≤0.0014 两个额外数据集，小 δ） |

**哪些结论改变（WEAKENED 或 OVERTURNED —— 11 个中 6 个）：**

- **id 139 (entry 4) — OVERTURNED。** 错误检验：在 1.4M 条神经元内自相关边上的边独立逻辑回归（z=16.4）。在 cluster-robust SE + 神经元 permutation 下效应不显著（perm p=0.4697）且在两个额外数据集上翻转符号。
- **id 24 (entry 5) — OVERTURNED。** 错误检验/构念：在 *总* 纤维上的 Mann-Whitney（假设是 *每神经元*），n=3 个超级 merge。n_super=1 的每神经元 MW 给出 p=0.2222。
- **id 36 (entry 8) — OVERTURNED。** 错误检验：49k 节点 Mann-Whitney，把参照 67 个锚点的聚集节点当作独立处理；p=1.6e-66 来自微不足道的 ~7.5% 位移。Neuron-permutation p=0.4771，δ=−0.064。
- **id 45 (entry 12) — OVERTURNED。** 错误检验：在非独立节点上有单一分层（无真实控制）的 CMH。Neuron-clustered permutation OR=0.53，p=0.6607。
- **id 61 (entry 15) — OVERTURNED。** 错误检验：在 1.1M 条参照一个小 merge 集的非独立边上的逐边 MW/Welch；δ CI 跨越 0，neuron-permutation p=0.6469。
- **id 64 (entry 17) — WEAKENED。** 错误检验：在 1.1M 条边上的逐边 MW（p=1.3e-197）。Cliff's δ CI [−0.369,−0.031] 仍不含 0，但 neuron-permutation p=0.096 在 origin 上不再显著。
- **id 21 / 27 — 注记：** 两者都是 *零* 发现，其零结论在正确检验下存续（作为零结论 UPHELD），因此它们的头条方向未改变，但其泛化现在是 PARTIAL，而非 DOES-NOT/PARTIAL 且基于更强的依据——见各条目要点。

**综合分析。** 修正干净地分离了两类发现。(1) **效应驱动的局部形态学发现经受住了正确检验：** split 空间聚集（id 37，δ≈0.49–0.60，perm p≤4e-4）被稳健地 UPHELD，而两个迂曲度 → split 风险发现（id 59、id 73）以其现在暴露的真实 *小* 效应量被 UPHELD（Cliff's δ≈0.18–0.27，CI 不含 0，perm p≤0.002）并泛化。(2) **头条建立在非独立边/节点上 n 膨胀 p 值之上的发现，一旦神经元成为分析单位就崩塌：** 分支阶数 → split 风险（id 139）、omit-near-merge 聚集（id 36）、split-near-merge 聚集（id 61）、极端 Z omit 超额（id 45），以及超级 merge 纤维主张（id 24）全部变得不显著（5 个 OVERTURNED），而 split-near-branch-point（id 64）被 WEAKENED 为一个边缘的、效应存在但不显著的结果。两个各向异性 *零* 主张（id 21、27）在其头条上对检验选择稳健（仍是"无显著效应"），但正确检验暴露出它们的泛化是体积特异的（一个真实的、符号相反的效应出现在 794495 上）。净结果：与校对相关的 *split 的空间聚集* 和 *迂曲度* 信号经受住考验，而大多数 *到锚点距离的共定位* 和 *全局轴 / 分支阶数* 主张是把数百万条相关边当作独立处理的伪影。
