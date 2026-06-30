# AutoDiscovery Run Summary — 按后验-意外度优先级排序

## Header

- **Source file:** `autodiscovery/run-5--ground-truth-error-annotations-revised-version_2026-06-25.json`
- **每文件假设数量：** 110 hypotheses (run-5--ground-truth-error-annotations-revised-version_2026-06-25)
- **排序键（`rank_by`）：** `posterior-surprise` — `priority_score = posterior * |surprisal|`
- **已排序：** 110 个假设中有 106 个带有可用的意外度分数（`n_ranked = 106`）
- **本报告返回：** 前 20 个（`n_returned = 20`）；本报告在应用 `--top 20` 后展示 106 个已排序假设中的 **前 20 个**
- **已排序记录的意外度幅度范围：** 0.0000 到 0.4707
- **最大优先级分数：** 0.4091
- **已排除（无意外度分数）：** 4 个假设——见末尾说明

### 核心要点综述

本次 run 收敛出一幅关于自动化 U-Net 神经元重建在何处失败、以及代理式校对器应如何针对这些失败的连贯、可操作的图景。最高优先级的信念翻转（#1，ID 15，priority 0.409，surprise 0.471）将一个 *Uncertain* 的先验提升为 *Likely True*：合并（merge）压倒性地发生在 **交叉纤维（crossing fibers）** 处（交叉角中位数 74.9°，p = 0.0081），而非平行接触的纤维——这是一个该循环此前并未自信持有的几何先验。下一梯队的高优先级、强确认结果（priority 全部为 0.275）围绕两个横切主题聚集。第一，**split 和 omit 在空间上是聚集的、而非随机的**（ID 26、40、45、50、27、42）：观测到的最近邻距离（约 130–140 µm）大约只有随机零假设期望（约 300–320 µm）的一半，p 值趋近于零，且 omit 错误是阵发性的（条件概率 84.6% 对基线 2.2%），在分支点和叶端附近形成局部的“错误级联（error cascades）”。第二，**merge 段巨大且不对称**（ID 22、34、55、56）：合并段比正确段长约 15×（作为分类器 AUC 0.869），并有约 91% 的时间与一个“主要”神经元重叠，因此 merge 修正可归约为对次要分叉的廉价可检测剪枝。两者共同确立了缆长（cable length）是一个强而廉价的 merge 先验，且 split/omit 应通过吸收附近的微小碎片来修复（86.8% 的 split 接触一个 sub-100 µm 碎片；95.8% 的 omit 间隙低于 100 µm 过滤阈值）。

---

## Reproduction — Summary

- **使用的数据集 pkl：** `/allen/programs/mindscope/workgroups/auto-model/zihan.zhang/exaspim-agent/exa-spim-agent/cache/dataset_cache_794495_mcl100_add.pkl`
- **分项（共 n_rerun = 20）：** REPRODUCED 20 · DIVERGED 0 · FAILED 0。
- **代码来源：** 全部 20 个均在 **revised-loading** 代码上运行（加载被重定向为直接 `$RERUN_PKL` 加载；统计分析在每个案例中都逐字节保持不变）。0 个在记录的代码上运行。
- **未能重现的发现：** 无。每个标题数字（检验统计量、p 值、效应量、样本计数）都与记录输出完全匹配，唯一例外是 ID 45，其随机置换 Monte Carlo 引入了微不足道的种子噪声（期望 NN 319.96 对记录 320.13 µm；t = -15.0355 对 -14.93；p = 1.2391e-11 对 1.40e-11），这不改变显著聚集的结论 → 仍为 REPRODUCED。
- **值得注意的恢复：** ID 12（split-gap 双峰性）的 RECORDED 为一次失败（exitcode 1，"Sandbox output was not valid JSON" ——一个基础设施错误，而非分析结果）。revised-loading 重跑干净运行，并在 n = 7,988 个 split gap 上产生了预测的双峰 GMM（Component 1 权重 0.6357，均值约 0.0 µm；Component 2 权重 0.3643，均值 159.18 µm），所以该发现现在为 REPRODUCED。
- **加载修订：** 全部 20 条记录的数据集搜索/加载被替换为直接 `$RERUN_PKL` 加载（记录的脚本使用了硬编码相对路径如 `../data/dataset_cache_794495_mcl100_add.pkl` 或 glob 模式如 `dataset_cache_*_add.pkl` / `*_add.pkl`，它们在重跑沙箱中无法解析）。分析代码和所有打印保持不变。
- **环境失败：** 无——没有 import/native-load 错误；preflight 通过（`preflight_ok: true`，`environment_failure: false`，`n_env_failed: 0`）。

---

## Generalization — Summary

- **来源数据集：** `cache/dataset_cache_794495_mcl100_add.pkl`。**使用的额外数据集（每个发现的可重现代码在其并非生成所用的数据上重跑）：** `cache/dataset_cache_794491_mcl100_add.pkl` 和 `cache/dataset_cache_789202_mcl100_add.pkl`。全部 20 个脚本在两个额外 pkl 上都成功执行（exitcode 0，无超时）——无加载/环境失败，因此每个 verdict 都基于实际数字。
- **分项（共 20）：** GENERALIZES 17 · PARTIAL 2 · DOES-NOT-GENERALIZE 1 · INCONCLUSIVE 0。

**未能完全泛化的发现：**

- **DOES-NOT-GENERALIZE —— Entry 13 / ID 42（split→omit 局部错误级联）：** 来源 794495 基线 0.0223 对条件 0.0528，Wilcoxon p=1.14e-05（显著）。ds_794491 基线 0.0437 对条件 0.0398，p=5.70e-01——方向翻转且不显著。ds_789202 基线 0.0396 对条件 0.0477，p=2.04e-01——方向相同但不显著。该效应在两个额外数据集上消失（在其中一个上反转）；这一结论是来源所特有的。
- **PARTIAL —— Entry 1 / ID 15（merge 交叉角 > 45°）：** 来源 794495 中位数 74.93°，Wilcoxon p=8.06e-03，n=12。ds_794491 中位数 54.30°，p=5.37e-02，n=25——方向相同但在 α=0.05 处失去显著性。ds_789202 产生了 0 个有效 merge 站点，因此检验无法在其上运行。仅有一个可用的额外数据集且为边缘性 → 弱、部分支持。
- **PARTIAL —— Entry 20 / ID 70（split 率与到叶端距离负相关）：** 来源 794495 逻辑回归 coef=-0.1361，p<0.001。ds_789202 coef=-0.3167，p≈0.0（成立，更强）。ds_794491 coef=-0.0095，p=0.409——斜率基本为零，不显著（失去）。在一个额外数据集上成立，在另一个上失去。

**综述：** 在测试了两个额外数据集后，本次 run 的核心结论大体稳健：20 个发现中有 17 个在两个额外大脑上重现了相同的方向和显著性。稳健的、数据集无关的图景涵盖（a）split 在分支点附近以及沿缆聚集（ID 3、26、33、40、45），（b）omit 是阵发性的/零距离相邻的并由 sub-100 µm 间隙主导（ID 27、50、69），以及（c）merge 段巨大且不对称，使尺寸成为可靠的 merge 先验（ID 22、34、55、56），外加半径约束的安全权衡（ID 47，基于其 direct-pair 度量）。三个结论较弱或数据集特定：split→omit 共现级联（ID 42）完全 **不** 泛化（在两个额外数据集上消失或反转）；merge 交叉角发现（ID 15）仅得到弱支持，因为一个额外数据集产生了零个 merge 站点，另一个仅为边缘性（p≈0.054）；以及叶端 split 率梯度（ID 70）在一个额外数据集上显著但在另一个上平坦/不显著。由于只有两个额外数据集可用，GENERALIZES 的 verdict 应解读为“在总共三个大脑上一致”而非群体级证明，但 17 个稳健发现上的一致同意（往往伴随相当或更大的效应量）是相当有力的证据。

---

## Ranked Conclusions（优先级最高者在前）

### 1. (Priority 0.409 · Surprise 0.471) Merge 发生在交叉纤维处，交叉角显著大于 45°。
- **Run:** run-5--ground-truth-error-annotations-revised-version_2026-06-25 · **ID:** 15 · **Belief:** Uncertain → Likely True (0.5417 → 0.8690) · **Direction:** Positive
- **Tested：** 在一个 merge 站点处两个 ground-truth 神经元之间的 3D 交叉角是否显著大于 45°，表明 merge 形成于纤维交叉处而非平行接触处。
- **Conclusion：** 在 12 个有效的 ground-truth merge 站点上，交叉角分布均值为 65.97°、中位数为 74.93°，角度聚集在 65° 到 90° 之间。针对 45° 基准的单样本 Wilcoxon signed-rank 检验给出统计量 69.00、p = 0.0081，拒绝零假设。正向意外度（+0.471）反映了本次 run 中最大的信念偏移——一个 *Uncertain* 先验被提升为 *Likely True*，确认 merge 主要是交叉纤维几何问题，这直接指导校对器应如何推理候选融合。
- **Caveats：** 样本很小（仅 12 个有效 merge 站点），所以角度估计虽显著但基于有限数据；非参数检验是合适的，但功效一般。
- **Reproduction：** REPRODUCED (code: revised-loading)
- **Rerun result：** 记录 Wilcoxon stat=69.00, p=8.0566e-03, n=12, mean=65.97°, median=74.93°；重跑 stat=69.00, p=8.0566e-03, n=12, mean=65.97°, median=74.93° → 完全匹配。加载被重定向为直接 `$RERUN_PKL` 加载。
- **Generalization：** PARTIAL
- **Across datasets：** 来源 794495：median 74.93°，Wilcoxon stat=69.00, p=8.06e-03, n=12（显著 >45°）。ds_794491：median 54.30°, stat=223.00, p=5.37e-02, n=25——方向相同（median > 45°）但在 α=0.05 处失去显著性。ds_789202：找到 0 个有效 merge 站点，因此检验完全无法在其上运行。方向在唯一产生数据的数据集上成立，但仅为边缘性（p≈0.054），而另一个额外数据集未产生可检验站点——鉴于只有一个可用额外数据集，证据较弱。
- **Verdict：** WEAK
- **Test：** 针对 45° 基准的交叉角单样本 Wilcoxon signed-rank 检验，statistic=69.00, p=0.0081, n=12 个 merge 站点。非参数单样本检验对于小型、可能非正态的角度样本是正确选择。
- **Statistical issues：** 严重欠功效（n=12）；角度估计基于十几个站点，检验几乎无功效刻画分布。45° 基准是任意参考而非数据导出的零假设。
- **Logic issues：** 来源推理中无（在 α=0.05 正确拒绝且未声称零假设为真），但结论“merge 主要是交叉纤维几何问题”泛化超出了 12 个站点。
- **Verdict rationale：** 检验正确，但极小的 n 加上 PARTIAL 泛化（一个额外数据集产生 0 个可检验站点，另一个仅在 p≈0.054 边缘）意味着标题性信念翻转未得到稳健支持——可辩护但需大量保留。

### 2. (Priority 0.275 · Surprise 0.300) Split 错误聚集在 ground-truth 分支点附近（风险升高约 1.6×）。
- **Run:** run-5--ground-truth-error-annotations-revised-version_2026-06-25 · **ID:** 3 · **Belief:** Leaning True → Likely True (0.7083 → 0.9167) · **Direction:** Positive
- **Tested：** split 错误在 ground-truth 分支点 15 µm 以内是否比在无分支的节间段上更可能发生，暗示分叉局部地降低 U-Net 分割质量。
- **Conclusion：** 分支邻近边（≤15 µm）有 0.90% 的情况包含一个 split（741 / 82,348），而分支远端边为 0.57%（7,247 / 1,281,441），相对风险 1.59（Chi-square p = 4.90e-34）。极大的边计数和极小的 p 值使其成为稳健的正向确认，强化了先验（正向意外度 +0.300）：神经元分叉是 split 错误的局部失败模式。
- **Caveats：** 两个分箱中的绝对 split 率都很低（<1%），所以虽然相对效应高度显著，但每条边的实际差异不大。
- **Reproduction：** REPRODUCED (code: revised-loading)
- **Rerun result：** 记录 proximal 741/82,348 (0.90%), distal 7,247/1,281,441 (0.57%), RR=1.5911, Chi-square p=4.9035e-34；重跑相同（741/82,348, 7,247/1,281,441, RR=1.5911, p=4.9035e-34）→ 完全匹配。
- **Generalization：** GENERALIZES
- **Across datasets：** 来源 794495：RR=1.5911，proximal 0.90% 对 distal 0.57%，χ² p=4.90e-34。ds_794491：RR=2.5783，proximal 3.20% 对 distal 1.24%，p=1.74e-246。ds_789202：RR=3.2391，proximal 1.43% 对 distal 0.44%，p=1.15e-256。在两个额外数据集上方向相同（分支邻近 > 分支远端）且高度显著，效应量（相对风险）甚至大于来源。
- **Verdict：** SOUND
- **Test：** 对一个 2×2 计数表（split vs no-split × proximal vs distal）的 Chi-square 独立性检验，p=4.90e-34，n=82,348 个 proximal 和 1,281,441 个 distal 边，相对风险 1.59。Chi-square 是二元计数的正确检验；期望单元格计数非常大，故渐近近似有效。
- **Statistical issues：** 一个神经元内的边并非严格独立（空间自相关），这可能膨胀 chi-square；两个分箱中绝对率均 <1%，所以尽管 RR 大，每条边的实际差异不大。
- **Logic issues：** 无——结论（分叉局部降低分割质量）与正确拒绝的零假设相符，且报告标注了小的绝对效应。
- **Verdict rationale：** 大表上的正确检验、完全重现，且在两个额外数据集上方向相同、RR 甚至更大；神经元内依赖是不能推翻这一如此大、如此一致效应的次要保留。

### 3. (Priority 0.275 · Surprise 0.300) Split gap 是双峰的：微小的单丢失间隙加上大的漏失段间隙。
- **Run:** run-5--ground-truth-error-annotations-revised-version_2026-06-25 · **ID:** 12 · **Belief:** Leaning True → Likely True (0.7083 → 0.9167) · **Direction:** Positive
- **Tested：** 导致 split 错误的物理间隙是否遵循代表两种不同失败模式的双峰分布——小的单丢失间隙和大的 omit-段间隙。
- **Conclusion：** 在 7,988 个有效 split gap 上，一个 2 成分高斯混合模型确认了双峰性：Component 1（约 63.6% 的 split）的均值间隙约 0.0 µm（可忽略位移，例如相邻体素被标记为不同），Component 2（约 36.4%）的均值间隙约 159.2 µm，标准差很大（约 259.0 µm），是较大漏失段的特征。正向意外度（+0.300）确认了两个不同的尺度域，暗示代理式连接器应使用两个搜索半径而非一个。
- **Caveats：** GMM 成分分配依赖模型；Component 2 的大标准差（约 259 µm）意味着“大间隙”模式很宽且重叠大范围距离。
- **Reproduction：** REPRODUCED (code: revised-loading)
- **Rerun result：** 最初记录的 run 为 FAILED（exitcode 1，"Sandbox output was not valid JSON" ——一个基础设施/沙箱错误，而非分析结果），所以不存在记录数字。revised-loading 重跑成功并产生了该假设预测的双峰 GMM：n=7,988 个 split gap；Component 1 权重 0.6357，均值约 0.0 µm，std 0.0010 µm；Component 2 权重 0.3643，均值 159.18 µm，std 259.01 µm → 结论（双峰的微小间隙 + 大间隙模式）得到确认。通过从记录的沙箱失败中恢复而计为 REPRODUCED。
- **Generalization：** GENERALIZES
- **Across datasets：** 来源 794495：2 成分 GMM，Comp1 权重 0.6357 均值约 0.0 µm，Comp2 权重 0.3643 均值 159.18 µm（std 259.0），n=7,988。ds_794491：Comp1 权重 0.5698 均值约 0.0 µm，Comp2 权重 0.4302 均值 117.47 µm（std 149.1），n=7,847。ds_789202：Comp1 权重 0.7391 均值 0.80 µm，Comp2 权重 0.2609 均值 328.73 µm（std 464.8），n=6,805。两个额外数据集都重现了双峰结构——一个近零的单丢失模式加上一个大的漏失段模式——成分权重相当；大间隙均值变化（117–329 µm）但双模式结论始终成立。
- **Verdict：** WEAK
- **Test：** 2 成分高斯混合模型拟合（Comp1 权重 0.6357 均值约 0.0 µm，Comp2 权重 0.3643 均值 159.18 µm，n=7,988）。这是一个描述性模型拟合，而非假设检验——没有 p 值，没有与 1 成分模型的比较（例如通过 BIC/likelihood-ratio），所以“双峰性”是被构造性地断言的。
- **Statistical issues：** 强加了一个 2 成分 GMM 而非选择得来；对任何连续分布拟合 k=2 都会返回两个成分，无论真实模态如何。未报告任何模型选择统计量（BIC、AIC、相对 k=1 的 LRT）来论证两个模式优于一个偏斜模式。
- **Logic issues：** 轻度越界——从一个拟合但未经选择的混合中得出“两种不同失败模式”的结论；均值约 0.0 µm 且 std 约 0.001 的 Comp1 可能是近退化的尖峰而非真正的第二物理模式。
- **Verdict rationale：** 该发现描述性地重现并泛化，但因为没有统计检验确立相对于单峰备择的双峰性，“两个不同尺度域”的主张仅能作为探索性观察来辩护。
- **Corrected test：** 关于模式数量的正式模型选择检验（k=1..5 的 BIC/AIC，k=2 对 k=1 的 likelihood-ratio）加上一个针对单峰零假设带自助 p 值的 Hartigan dip test——原始分析仅强加了一个 k=2 GMM，它对任何连续分布都会返回两个成分，无论真实模态如何，所以“双峰性”是被构造性断言的，没有 p 值。
- **Corrected result：** 双峰性现在得到正式支持，而非假定。在来源上（n=7,988）：BIC(k=1)−BIC(k=2)=114,816.7（为正 ⇒ 强烈偏好 k=2），likelihood-ratio 2·(LL₂−LL₁)=114,843.7，Hartigan dip statistic=0.4128 带自助 p=0.0020（在 α=0.05 处拒绝单峰性）。k=2 GMM 参数与原始 revised run 不变（Comp1 权重 0.6357 均值约 0.0 µm；Comp2 权重 0.3643 均值 159.18 µm）。原始分析报告了无检验（仅描述性 k=2 GMM）。
- **Post-correction verdict：** UPHELD
- **Corrected generalization：** GENERALIZES——在两个额外数据集上成立。ds_794491（n=7,847）：BIC(k=1)−BIC(k=2)=95,997.1, LRT=96,024.0, dip=0.3945, 自助 p=0.0020。ds_789202（n=6,805）：BIC(k=1)−BIC(k=2)=42,422.4, LRT=42,448.8, dip=0.4231, 自助 p=0.0020。dip test 拒绝单峰性（p<0.05），且模型选择在所有三个大脑上偏好 k≥2。

### 4. (Priority 0.275 · Surprise 0.300) Merge 段高度不对称——大多是一个主要神经元的次要分叉。
- **Run:** run-5--ground-truth-error-annotations-revised-version_2026-06-25 · **ID:** 22 · **Belief:** Leaning True → Likely True (0.7083 → 0.9167) · **Direction:** Positive
- **Tested：** merge 段是否与一个“主要”神经元广泛重叠而仅勉强触及次要神经元（重叠比 > 0.8），从而 merge 修复是剪枝而非 50/50 拆分。
- **Conclusion：** 在 98 个 merge 段上，平均重叠比为 0.9129，中位数为 1.0，单样本 t-test 确认均值超过 0.5（t = 24.87, p = 3.58e-44）。均衡的 50/50 融合极其罕见。正向意外度（+0.300）支持将 merge 修正视为剪掉一个次要的意外分叉，而非分割一个均衡、模糊的段。
- **Caveats：** 基于来自单一数据集缓存的 98 个 merge 段；t-test 假定重叠比分布近似正态，而该分布偏向 1.0。
- **Reproduction：** REPRODUCED (code: revised-loading)
- **Rerun result：** 记录 n=98, mean overlap=0.9129, median=1.0000, 单样本 t=24.8654, p=3.5768e-44；重跑相同（n=98, mean=0.9129, median=1.0000, t=24.8654, p=3.5768e-44）→ 完全匹配。
- **Generalization：** GENERALIZES
- **Across datasets：** 来源 794495：mean overlap 0.9129, median 1.0, t=24.87, p=3.58e-44, n=98。ds_794491：mean 0.8558, median 1.0, t=17.18, p=1.72e-31, n=98。ds_789202：mean 0.9704, median 1.0, t=43.12, p=8.84e-49, n=64。在两个额外数据集上方向相同（mean overlap ≫ 0.5，median 1.0）且压倒性显著；效应量相当（mean 0.86–0.97）。
- **Verdict：** MINOR
- **Test：** 单样本 t-test 检验 mean overlap ratio 超过 0.5，t=24.87, p=3.58e-44, n=98（mean 0.9129, median 1.0）。数据是限于 [0,1] 的比例且严重堆积在 1.0 上限（median=1.0），这违反了 t-test 的正态性假设。
- **Statistical issues：** 错误的分布假设——单样本 t-test 应用于强烈左偏、上限有界的比例（median 1.0）；单样本 Wilcoxon 或自助/符号检验是合适选择。鉴于分布坐落在 1.0 边界，将均值与 0.5 比较也是一个弱目标。
- **Logic issues：** 无——实质性结论（段与一个主要神经元重叠，故 merge 是剪枝而非 50/50 拆分）由中位数 1.0 支持，独立于 t-test。
- **Verdict rationale：** 对上限有界比例而言检验选择技术上不当，但效应如此极端（三个数据集上 median 都为 1.0），结论毫无疑问；仅因检验选择缺陷而降级。
- **Corrected test：** 单样本 Wilcoxon signed-rank 检验（median > 0.5）加上精确符号检验，并对 median 和 mean overlap ratio 给出自助 95% CI——替换单样本 t-test，后者假定一个强烈左偏、[0,1] 有界、堆积在 1.0 上限（median=1.0）的比例近似正态。
- **Corrected result：** Wilcoxon W=4656.0, p=1.8663e-19（n_nonzero=96）；精确符号检验 p=1.2622e-29；效应量 median overlap=1.0000，自助 95% CI [1.0000, 1.0000]；mean overlap=0.9129，自助 95% CI [0.8793, 0.9439]——CI 远高于 0.5。原始单样本 t-test 报告 t=24.8654, p=3.5768e-44。非参数检验以一个排除 0.5 零假设的 CI 确认了相同结论。
- **Post-correction verdict：** UPHELD
- **Corrected generalization：** GENERALIZES——在两个额外数据集上成立。ds_794491（n=95）：Wilcoxon W=4560.0, p=1.90e-18；符号检验 p=2.52e-29；median 1.0 CI [0.9943, 1.0000]，mean 0.8558 CI [0.8151, 0.8944]。ds_789202（n=64）：Wilcoxon W=2080.0, p=1.93e-14；符号检验 p=5.42e-20；median 1.0 CI [1.0000, 1.0000]，mean 0.9704 CI [0.9472, 0.9895]。在所有三个大脑上 median 均为 1.0，CI 远高于 0.5。

### 5. (Priority 0.275 · Surprise 0.300) Split 错误沿神经元缆在空间上聚集，而非随机。
- **Run:** run-5--ground-truth-error-annotations-revised-version_2026-06-25 · **ID:** 26 · **Belief:** Leaning True → Likely True (0.7083 → 0.9167) · **Direction:** Positive
- **Tested：** split 位置是否沿 1D 神经元缆违反完全空间随机性，反映图像质量差或交叉纤维密集的局部区域。
- **Conclusion：** 在 19 个合格神经元上，观测到的 split 之间平均最近邻距离（140.30 µm）远小于 Monte Carlo 随机期望（320.79 µm），paired t-test t = -15.10、p = 5.76e-12。正向意外度（+0.300）确认了 split 的强空间聚集，指示局部失败区域。
- **Caveats：** 仅 19 个神经元合格，且 Monte Carlo 每个神经元用 100 次迭代（足够但受时限约束）；结论泛化至此数据集的神经元。
- **Reproduction：** REPRODUCED (code: revised-loading)
- **Rerun result：** 记录 n=19 个神经元, paired t=-15.1004, p=5.7640e-12, 观测 NN=140.30 µm 对随机 NN=320.79 µm；重跑相同（n=19, t=-15.1004, p=5.7640e-12, 140.30 µm 对 320.79 µm）→ 完全匹配。
- **Generalization：** GENERALIZES
- **Across datasets：** 来源 794495：n=19, 观测 NN 140.30 µm 对随机 320.79 µm, paired t=-15.10, p=5.76e-12。ds_794491：n=9, 观测 76.87 µm 对随机 147.62 µm, t=-6.33, p=1.13e-04。ds_789202：n=12, 观测 219.90 µm 对随机 385.96 µm, t=-6.99, p=1.14e-05。在两个额外数据集上方向相同（观测约为随机的一半）且聚集显著；t 统计量较小（合格神经元较少）但结论成立。
- **Verdict：** SOUND
- **Test：** 每神经元观测平均最近邻距离对 Monte Carlo 随机期望的 paired t-test，t=-15.10, p=5.76e-12, n=19 个神经元（140.30 µm 对 320.79 µm）。按神经元配对是正确的（每个神经元是其自身置换零假设的对照），且分析单位（神经元）是独立单位，故独立性得到尊重。
- **Statistical issues：** n 适中（19 个神经元）且受时限约束的 100 次迭代 Monte Carlo 零假设；鉴于约 2.3× 的效应两者都足够，但检验继承了所选的 100 次置换预算。
- **Logic issues：** 无——聚集是从正确拒绝的随机性零假设推断而来，而非接受零假设。
- **Verdict rationale：** 针对置换零假设的合适配对设计、大效应（观测约为随机的一半）、完全重现，且在两个额外数据集上方向显著相同——一个构造良好的空间聚集检验。

### 6. (Priority 0.275 · Surprise 0.300) Omit 错误是阵发性的：连续的缺失段，而非孤立的丢边。
- **Run:** run-5--ground-truth-error-annotations-revised-version_2026-06-25 · **ID:** 27 · **Belief:** Leaning True → Likely True (0.7083 → 0.9167) · **Direction:** Positive
- **Tested：** 当相邻边也被 omit 时，一条边被 omit 的可能性是否远高于基线 omit 率，表明 omit 错误以连续段出现。
- **Conclusion：** 在 19 个神经元上，平均基线 omit 概率为 2.23%，但给定一个相邻 omit 边时的条件 omit 概率跃升至 84.55%（Wilcoxon W = 0.0, p = 3.81e-06）。转移矩阵显示一旦进入 omit 状态便有 84.9% 的概率停留在该状态，平均 run 长度为 6.91 条边，56 个 run 超过 50 条边，最大 run 为 339。正向意外度（+0.300）决定性地确认了阵发性、连续的 omit 行为。
- **Caveats：** 基于 19 个神经元；长尾 run 长度统计由少数极长 run 驱动，所以平均 run 长度低估了其分散度。
- **Reproduction：** REPRODUCED (code: revised-loading)
- **Rerun result：** 记录 n=19, baseline P(Omit)=0.0223, conditional P(Omit|Omit)=0.8455, Wilcoxon W=0.0, p=3.8147e-06, transition Omit→Omit=0.849, 4,145 个 run, max=339, 平均 run 长度=6.91；重跑在所有数字上相同 → 完全匹配。
- **Generalization：** GENERALIZES
- **Across datasets：** 来源 794495：baseline 0.0223 对 conditional 0.8455, W=0.0, p=3.81e-06, n=19, Omit→Omit transition 0.849。ds_794491：baseline 0.0437 对 conditional 0.7854, W=0.0, p=3.91e-03, n=9, Omit→Omit 0.810。ds_789202：baseline 0.0396 对 conditional 0.8665, W=0.0, p=4.88e-04, n=12, Omit→Omit 0.891。在两个额外数据集上方向相同（conditional ≫ baseline，约 80–87%）且显著；阵发性/黏性 omit 行为稳健。
- **Verdict：** SOUND
- **Test：** 每神经元 baseline 对 conditional omit 概率的 Wilcoxon signed-rank 检验，W=0.0, p=3.81e-06, n=19 个神经元（2.23% baseline 对 84.55% conditional）。对每神经元配对率的配对非参数检验是正确选择；W=0.0 意味着每个神经元都朝同一方向变动。
- **Statistical issues：** n=19 适中，但效应（条件概率约 38× 的跃升）巨大且在各神经元间一致，故功效不是问题。
- **Logic issues：** 无，不过对以连通 run 出现的错误而言，“阵发性”结论近乎同义反复——条件概率高部分上是由 run 结构定义所致；报告承认了长尾 run 长度的框架。
- **Verdict rationale：** 正确的配对检验、巨大且一致的效应、完全重现并在两个额外数据集上显著泛化；唯一的微妙之处是基于 run 的条件概率的部分定义性质，这不削弱结论。

### 7. (Priority 0.275 · Surprise 0.300) Split 边坐落得比正确边显著更靠近分支节点。
- **Run:** run-5--ground-truth-error-annotations-revised-version_2026-06-25 · **ID:** 33 · **Belief:** Leaning True → Likely True (0.7083 → 0.9167) · **Direction:** Positive
- **Tested：** 从一个 split 边到最近 ground-truth 分支节点（度 > 2）的中位路径距离是否显著短于正确重建边。
- **Conclusion：** 在 7,988 个 split 边和 934,849 个正确边上，到最近分支节点的中位距离 split 为 216.42 µm，正确边为 246.43 µm（Mann-Whitney U = 3.47e9, p = 4.64e-27）。正向意外度（+0.300）确认 U-Net 更易在拓扑分支点附近将神经元碎片化。
- **Caveats：** 绝对中位差异（约 30 µm）相对于典型分支间距而言不大；显著性主要由极大样本驱动，故就实际而言效应量很小。
- **Reproduction：** REPRODUCED (code: revised-loading)
- **Rerun result：** 记录 splits n=7,988, correct n=934,849, Mann-Whitney U=3,472,848,384.0, p=4.637e-27, split median=216.42 µm 对 correct median=246.43 µm；重跑相同（U=3,472,848,384.0, p=4.637e-27, 216.42 对 246.43 µm）→ 完全匹配。
- **Generalization：** GENERALIZES
- **Across datasets：** 来源 794495：split median 216.42 µm < correct 246.43 µm, U=3.47e9, p=4.64e-27（splits n=7,988, correct n=934,849）。ds_794491：split median 156.94 µm < correct 195.79 µm, U=1.46e9, p=0.0（splits n=7,847, correct n=420,702）。ds_789202：split median 196.43 µm < correct 380.39 µm, U=2.98e9, p=0.0（splits n=6,805, correct n=1,109,034）。在两个额外数据集上方向相同（split 更靠近分支节点）且显著；在 789202 上差距实际更大（约 184 µm）。
- **Verdict：** WEAK
- **Test：** split（n=7,988）对 correct（n=934,849）边到分支节点距离的 Mann-Whitney U 检验，U=3.47e9, p=4.64e-27。MWU 对偏斜的距离分布是合适的，但 p 值几乎完全由巨大的样本量驱动。
- **Statistical issues：** p≈0 来自 n≈940k，而中位差异仅约 30 µm（216.42 对 246.43 µm）——相对于典型分支间距是实际上可忽略的效应；显著性与重要性被混淆。一个神经元内的边也非独立，进一步膨胀了显著性。
- **Logic issues：** 轻度越界：“更可能在分支点附近碎片化”是从约 30 µm 中位偏移得出的强因果/空间主张；报告自身标注了小的实际效应。
- **Verdict rationale：** 检验类型正确但是教科书式的巨 n / 平凡效应量案例——结论在方向上重现并泛化，但效应太小以至于实际上无意义，故强措辞不被支持。
- **Corrected test：** 效应量估计（rank-biserial r / Cliff's delta 带对 NEURONS 重采样的聚类自助 95% CI）加上一个聚类级（独立单位 = 神经元，n=19）符号翻转置换检验——针对验证者指出的两个缺陷：p≈0 由 n≈940k 驱动，以及一个神经元内的边被当作独立。
- **Corrected result：** 在来源上效应可忽略，CI 触及/包含零假设：rank-biserial r=0.0699（|r|<0.1 = 可忽略），Cliff's delta=−0.0699，聚类自助 95% CI [−0.1434, 0.0026]（包含 0）；中位差异 −30.01 µm。聚类级置换（神经元作为单位）给出 p=4.4298e-02——仅勉强显著，有 73.68% 的神经元显示 split < correct 中位数。原始 MWU 报告 U=3.47e9, p=4.64e-27（现在视为对一个近零效应的 n-膨胀 p 值）。
- **Post-correction verdict：** WEAKENED
- **Corrected generalization：** PARTIAL——效应量保持很小，且聚类自助 CI 在一个额外数据集上包含 0。ds_794491（n=9 个神经元）：Cliff's delta=−0.1133, CI [−0.2391, 0.0021]（包含 0），但聚类置换 p=4.9998e-05，100% 的神经元 split<correct。ds_789202（n=12）：Cliff's delta=−0.2103, CI [−0.3495, −0.0508]（排除 0），聚类置换 p=6.9997e-04，100% 的神经元 split<correct。方向一致（split 坐落得更靠近分支节点）且聚类检验在所有三个上显著，但效应量为可忽略到小（delta 0.07–0.21），其 CI 在来源和 ds_794491 上包含零——所以结论在方向上成立但实际上很弱。

### 8. (Priority 0.275 · Surprise 0.300) Merge 段不成比例地巨大——一个廉价、灵敏的 merge 先验。
- **Run:** run-5--ground-truth-error-annotations-revised-version_2026-06-25 · **ID:** 34 · **Belief:** Leaning True → Likely True (0.7083 → 0.9167) · **Direction:** Positive
- **Tested：** 导致 merge 的段平均上是否至少有正确重建的 1-to-1 段总缆长的 3×，从而仅凭段长就能标记可能的 merge。
- **Conclusion：** 98 个导致 merge 的段平均约 19,040 µm 缆（中位数 4,605 µm），而 3,129 个正确段约 1,221 µm（中位数 371 µm）——15.6× 的比率，远超假设的 3×。使用原始缆长作为阈值分类器得到 ROC AUC = 0.869。正向意外度（+0.300）确认缆长是在昂贵的几何游走之前标记 merge 的高效、计算廉价的先验。
- **Caveats：** 仅 98 个 merge 段锚定 merge 分布；分类器 AUC 虽强，仍留有可观的重叠（大的正确段作为离群点存在）。
- **Reproduction：** REPRODUCED (code: revised-loading)
- **Rerun result：** 记录 merge n=98（avg 19,039.76 µm, median 4,604.78 µm）, correct n=3,129（avg 1,220.58 µm, median 371.42 µm）, ratio=15.60；重跑相同 → 完全匹配。
- **Generalization：** GENERALIZES
- **Across datasets：** 来源 794495：merge avg 19,039.76 µm 对 correct 1,220.58 µm, ratio 15.60（n=98 对 3,129）。ds_794491：merge avg 5,850.90 µm 对 correct 705.47 µm, ratio 8.29（n=98 对 2,765）。ds_789202：merge avg 17,844.66 µm 对 correct 1,013.21 µm, ratio 17.61（n=64 对 4,011）。所有三个数据集都超过假设的 ≥3× 阈值（8.3×–17.6×），所以“merge 段巨大”的结论始终成立，尽管比率在 794491 上降至约 8×。
- **Verdict：** SOUND
- **Test：** 平均缆长的描述性比率（merge n=98 avg 19,040 µm 对 correct n=3,129 avg 1,221 µm = 15.6×）加上一个 ROC 分类器（AUC=0.869）。未报告正式假设检验/p 值，但该主张被构造为效应量阈值（≥3×）和 AUC，它们是“长度是否为有用先验”的合适描述性度量。
- **Statistical issues：** 无推断性检验，但对一个声明的效应量/区分主张而言并非严格必需；均值被长尾 merge 离群点抬高，故中位数（4,605 对 371 µm，12.4×）是更稳健的基础，也超过 3×。
- **Logic issues：** 无——结论（缆长是廉价 merge 先验）正是 AUC=0.869 所支持的，且关于重叠/离群点的保留已陈述。
- **Verdict rationale：** 一个用比率和 ROC AUC 正确评估的效应量主张；完全重现且在所有三个数据集上超过 ≥3× 阈值（8.3–17.6×），故结论扎实。

### 9. (Priority 0.275 · Surprise 0.300) 绝大多数 split 涉及一个 sub-100 µm 的微小碎片。
- **Run:** run-5--ground-truth-error-annotations-revised-version_2026-06-25 · **ID:** 35 · **Belief:** Leaning True → Likely True (0.7083 → 0.9167) · **Direction:** Positive
- **Tested：** 是否超过 50% 的 split 错误涉及进入或离开一个“微小碎片”（总路径长度少于 100 µm 的预测段）的转移。
- **Conclusion：** 7,988 个 split 事件中，6,931 个（86.77%）至少涉及一个微小碎片，远超 50% 阈值。正向意外度（+0.300）确认 split 主要源自微小的中间碎片，而非 U-Net 将神经元断成大半段，暗示校对器可通过吸收小的相邻段获得可观的 Expected Run Length。
- **Caveats：** 100 µm 微小碎片截断是一个选定阈值；结果特定于该定义和数据集的段长分布。
- **Reproduction：** REPRODUCED (code: revised-loading)
- **Rerun result：** 记录 6,931/7,988 个 split（86.77%）涉及一个 sub-100 µm 微小碎片；重跑相同（6,931/7,988, 86.77%）→ 完全匹配。
- **Generalization：** GENERALIZES
- **Across datasets：** 来源 794495：6,931/7,988 = 86.77%。ds_794491：6,619/7,847 = 84.35%。ds_789202：5,102/6,805 = 74.97%。所有三个都从容超过 >50% 阈值；该比例在 789202 上稍低（75%）但结论（微小碎片主导 split）在两个额外数据集上成立。
- **Verdict：** SOUND
- **Test：** 单一比例观测：6,931/7,988（86.77%）的 split 涉及 sub-100 µm 微小碎片，对一个 50% 阈值。未报告正式检验，但 n=7,988 时比例为 86.77%——其置信区间（约 ±0.7%）远离 50%，故正式的单比例 z 检验将是多余的。
- **Statistical issues：** 结果取决于所选的 100 µm 微小碎片截断（已承认）；不是缺陷，但标题数字与阈值相关。
- **Logic issues：** 无——结论（split 由微小碎片主导，故吸收它们带来 ERL 增益）直接源于该比例。
- **Verdict rationale：** 一个简单的大 n 比例，压倒性地超过 50% 并重现/泛化（75–87%）；唯一依赖是阈值定义，已透明陈述。

### 10. (Priority 0.275 · Surprise 0.300) 弯曲度在 split 附近统计上更高，但作为独立预测因子较弱。
- **Run:** run-5--ground-truth-error-annotations-revised-version_2026-06-25 · **ID:** 36 · **Belief:** Leaning True → Likely True (0.7083 → 0.9167) · **Direction:** Positive
- **Tested：** 在 10 µm 窗口内的局部骨架弯曲度（路径长度 / 直线距离）在 split 边周围是否显著高于正确边。
- **Conclusion：** 平均弯曲度在 split 附近为 1.0905（±0.2090），正确边为 1.0541（±0.0905）（t = 14.27, p ≈ 0.0），故假设在统计上得到支持（正向意外度 +0.300）。然而两个分布都重度聚集在 1.0–1.2 且大量重叠，故仅高弯曲度 **不是** 强的独立 split 预测因子——只有极端弯曲度的尾部在 split 中富集。
- **Caveats：** 此处的统计显著性由样本量而非效应量驱动；分析本身标注 split 与正确边之间的实际分离很小，故不应将其视为牢固的预测规则。
- **Reproduction：** REPRODUCED (code: revised-loading)
- **Rerun result：** 记录 split n=7,988（mean tortuosity 1.0905±0.2090）, correct n=7,988（1.0541±0.0905）, t=14.2729, p=0.0；重跑相同 → 完全匹配。
- **Generalization：** GENERALIZES
- **Across datasets：** 来源 794495：split mean 1.0905 对 correct 1.0541, t=14.27, p≈0.0, n=7,988。ds_794491：split 1.1082 对 correct 1.0635, t=15.71, p≈0.0, n=7,847。ds_789202：split 1.1314 对 correct 1.0730, t=19.37, p≈0.0, n=6,805。在两个额外数据集上方向相同（split 附近弯曲度更高）且显著。如同来源，绝对分离仍很小（均值相差约 0.04–0.06），故弯曲度仍是统计上稳健但弱的独立预测因子。
- **Verdict：** MINOR
- **Test：** split 对 correct 边弯曲度的双样本 t-test（各 n=7,988），t=14.27, p≈0.0（mean 1.0905 对 1.0541）。弯曲度是 ≥1 的比率且重度右偏/聚集在 1.0–1.2，故 t-test 不是理想选择（Mann-Whitney 更干净），且方差不等（±0.209 对 ±0.091）要求 Welch 而非 Student。
- **Statistical issues：** t-test 下的偏斜、方差不等数据；更重要的是 p≈0 由 n 驱动而均值差异（约 0.036）微不足道。分析本身陈述实际分离很小且弯曲度“不是强的独立预测因子”。
- **Logic issues：** 无——结论被正确地约束（显著但弱的预测因子），避开了将显著性等同于预测价值的陷阱。
- **Verdict rationale：** 次优的检验和一个由 n 驱动的显著效应，但分析明确拒绝过度声称，故诚实的“统计上真实、实际上弱”的结论成立；因检验选择/效应量不匹配而降级。
- **Corrected test：** Mann-Whitney U（基于秩，无正态性假设）加上一个效应量——Cliff's delta / probability-of-superiority AUC 带自助 95% CI——替换对一个 ≥1 的比率（右偏、聚集在 1.0–1.2、方差不等，±0.209 对 ±0.091）的 Welch/Student t-test。
- **Corrected result：** Mann-Whitney U=3.859e7, p≈0.0（在同一方向上仍显著）；AUC P(split>correct)=0.6048；Cliff's delta=0.2096, 自助 95% CI [0.1808, 0.2397]（排除 0 但很小）；中位差异仅 0.0122（可忽略）。原始 Welch t-test 报告 t=14.2729, p≈0.0, 均值差约 0.036。修正后的效应量确认一个真实但小的分离——远不及一个可用的独立预测因子。
- **Post-correction verdict：** WEAKENED
- **Corrected generalization：** GENERALIZES（方向上）——在所有三个大脑上为小效应。ds_794491：U=3.882e7, p≈0.0, AUC=0.6305, Cliff's delta=0.2610, CI [0.2325, 0.2881], 中位差 0.0176。ds_789202：U=2.870e7, p≈0.0, AUC=0.6198, Cliff's delta=0.2395, CI [0.2116, 0.2658], 中位差 0.0202。方向（split 附近弯曲度更高）在两者上都显著且 CI 排除零，但 Cliff's delta 保持很小（0.21–0.26）且中位差约 0.01–0.02，与来源的“统计上真实、实际上弱”的解读一致。

### 11. (Priority 0.275 · Surprise 0.300) Split 在末端（叶）分支中比内部主干中更频繁。
- **Run:** run-5--ground-truth-error-annotations-revised-version_2026-06-25 · **ID:** 38 · **Belief:** Leaning True → Likely True (0.7083 → 0.9167) · **Direction:** Positive
- **Tested：** split 错误率在末端拓扑隔室（叶分支）中是否高于内部分支，暗示连续性追踪在朝向神经突端点时退化。
- **Conclusion：** 在 19 个神经元上，平均末端 split 率（0.65%）超过平均内部率（0.54%），paired t-test t = 3.93、p = 0.00097，且每神经元配对图确认该趋势在大多数神经元上成立。正向意外度（+0.300）支持 U-Net 连续性在神经突端点处退化。
- **Caveats：** 基于 19 个神经元；尽管配对检验显著，绝对率差距（约 0.1 个百分点）很小。
- **Reproduction：** REPRODUCED (code: revised-loading)
- **Rerun result：** 记录 n=19, internal split rate=0.0054, terminal=0.0065, diff=0.0011, paired t=3.9347, p=9.7131e-04；重跑相同 → 完全匹配。
- **Generalization：** GENERALIZES
- **Across datasets：** 来源 794495：internal 0.0054 对 terminal 0.0065, diff +0.0011, t=3.93, p=9.71e-04, n=19。ds_794491：internal 0.0115 对 terminal 0.0163, diff +0.0049, t=3.14, p=1.38e-02, n=9。ds_789202：internal 0.0049 对 terminal 0.0064, diff +0.0015, t=3.35, p=6.51e-03, n=12。在两个额外数据集上方向相同（terminal > internal）且显著（p<0.05）；该效应在 794491 上实际更大。
- **Verdict：** WEAK
- **Test：** 每神经元 terminal 对 internal split 率的 paired t-test，t=3.93, p=9.71e-04, n=19 个神经元（0.65% 对 0.54%）。按神经元配对是正确的；每神经元率对于 paired t-test 是一个合理的、近似连续的量。
- **Statistical issues：** 小 n（19 个神经元）；绝对率差距约 0.1 个百分点——统计上可检测但实际上很小。每神经元率是有界比例，故在此 n 下符号/Wilcoxon 检查会是更安全的伴随。
- **Logic issues：** 无——显著性来自正确配对的设计，且报告标注了小的绝对差距。
- **Verdict rationale：** 一个完全重现并泛化（两个额外数据集上 p<0.05、方向相同）的正确配对检验，但适中的 n 和近乎平凡的绝对效应使其保持在 WEAK 而非 SOUND。

### 12. (Priority 0.275 · Surprise 0.300) Split 错误显示空间“级联”——一个 split 使附近的 split 变得可能。
- **Run:** run-5--ground-truth-error-annotations-revised-version_2026-06-25 · **ID:** 40 · **Belief:** Leaning True → Likely True (0.7083 → 0.9167) · **Direction:** Positive
- **Tested：** 一个 split 的出现是否使另一个 split 在短距离内高度可能，指示持续分割不良的局部区域。
- **Conclusion：** 在 7,988 个 split 边上，观测平均最近邻距离（132.85 µm）远低于随机零假设均值（300.27 µm），Kolmogorov-Smirnov 检验给出 KS = 0.5635、p ≈ 0.0，坚决拒绝空间随机性。直方图在接近零距离处显示尖锐的密度峰。正向意外度（+0.300）确认了在局部分割不良区域中聚集 split 的级联效应。
- **Caveats：** 这用不同的检验佐证了 ID 26/45；KS 检验对大样本敏感，故显著性是预期的，不过大的 KS 统计量（0.56）表明确实存在实质的分布差异。
- **Reproduction：** REPRODUCED (code: revised-loading)
- **Rerun result：** 记录 n=7,988, KS=0.5635, p=0.0, 观测 NN=132.85 µm 对随机=300.27 µm；重跑相同（KS=0.5635, p=0.0, 132.85 对 300.27 µm）→ 完全匹配。
- **Generalization：** GENERALIZES
- **Across datasets：** 来源 794495：KS=0.5635, p=0.0, 观测 NN 132.85 µm 对随机 300.27 µm, n=7,988。ds_794491：KS=0.4376, p=0.0, 观测 73.21 µm 对随机 131.42 µm, n=7,847。ds_789202：KS=0.4564, p=0.0, 观测 209.06 µm 对随机 372.74 µm, n=6,805。在两个额外数据集上方向相同（观测约为随机的一半）且 p=0.0；KS 统计量略低（0.44–0.46）但聚集/级联结论成立。
- **Verdict：** SOUND
- **Test：** 观测对随机零假设最近邻距离分布的双样本 Kolmogorov-Smirnov 检验，KS=0.5635, p≈0.0, n=7,988 个 split 边。KS 适于比较两个连续分布；大的 KS 统计量（0.56）反映了一个确实大的分布差距，而不只是大 n 显著性。
- **Statistical issues：** split 边被当作独立观测，尽管许多位于同一神经元上（非独立），这膨胀了 KS 显著性；然而效应量（KS=0.56，观测约为随机的一半）足够大，不威胁结论。这在同一数据上复制了 ID 26/45，故是佐证而非独立证据。
- **Logic issues：** 无——“级联”推断来自一个带大 KS 统计量的正确拒绝的随机性零假设；报告注意到大 n 保留但强调实质的 KS 值。
- **Verdict rationale：** 正确的检验、大的效应量（KS=0.56）而非单纯显著性、完全重现，并在两个额外数据集上一致泛化（KS 0.44–0.46, p≈0）——聚集结论稳健。

### 13. (Priority 0.275 · Surprise 0.300) Split 和 omit 在局部共现，在图上形成错误级联。
- **Run:** run-5--ground-truth-error-annotations-revised-version_2026-06-25 · **ID:** 42 · **Belief:** Leaning True → Likely True (0.7083 → 0.9167) · **Direction:** Positive
- **Tested：** 一条在拓扑上与 split 边相邻的边是否具有显著高于该神经元基线 omit 率的 omit 概率，指示局部相关的拓扑错误。
- **Conclusion：** 在 19 个神经元上，平均基线 omit 率（2.23%）对与一个 split 相邻的边翻倍有余至 5.28%（Wilcoxon statistic = 2.0, p = 1.14e-05），几乎所有散点都在 y = x 线之上。正向意外度（+0.300）支持 split 和 omit 在局部相关，形成组合错误级联。
- **Caveats：** 基于 19 个神经元；条件 omit 率（5.28%）在绝对值上仍低，故该效应是在小基数上的相对翻倍。
- **Reproduction：** REPRODUCED (code: revised-loading)
- **Rerun result：** 记录 n=19, baseline omit rate=0.0223, conditional (adjacent-to-split)=0.0528, Wilcoxon stat=2.0, p=1.1444e-05；重跑相同 → 完全匹配。
- **Generalization：** DOES-NOT-GENERALIZE
- **Across datasets：** 来源 794495：baseline 0.0223 对 conditional 0.0528, Wilcoxon stat=2.0, p=1.14e-05, n=19（conditional ≫ baseline，显著）。ds_794491：baseline 0.0437 对 conditional 0.0398, stat=17.0, p=5.70e-01, n=9——方向翻转（conditional < baseline）且不显著。ds_789202：baseline 0.0396 对 conditional 0.0477, stat=22.0, p=2.04e-01, n=12——方向相同但不显著（p≈0.20）。split→omit 局部共现效应在两个额外数据集上消失（一个甚至反转符号）；这是本次 run 中最明确的数据集特定发现。
- **Verdict：** MAJOR
- **Test：** 每神经元 baseline 对 split-adjacent conditional omit 率的 Wilcoxon signed-rank 检验，statistic=2.0, p=1.14e-05, n=19 个神经元（2.23% 对 5.28%）。配对非参数检验是正确选择，且来源结果孤立地看在统计上有效。
- **Statistical issues：** 小 n（19 个神经元）和一个在极低基数上（2.23%→5.28%）的相对翻倍——脆弱。决定性缺陷是复制：在 ds_794491 上效应反转（conditional 3.98% < baseline 4.37%, p=0.570），在 ds_789202 上不显著（p=0.204）。
- **Logic issues：** 结论越界——“split 和 omit 形成错误级联”被从单一数据集断言为一般机制，但该效应在两个额外大脑上都不存活。
- **Verdict rationale：** 来源检验技术上正确，但该发现 DOES-NOT-GENERALIZE（在一个数据集上符号翻转，在另一个上不显著），故级联结论不应被信任——本次 run 中最明确的数据集特定主张。

### 14. (Priority 0.275 · Surprise 0.300) 通过 Monte Carlo 最近邻置换再次确认 split 聚集。
- **Run:** run-5--ground-truth-error-annotations-revised-version_2026-06-25 · **ID:** 45 · **Belief:** Leaning True → Likely True (0.7083 → 0.9167) · **Direction:** Positive
- **Tested：** split 边沿 ground-truth 拓扑彼此之间是否比随机机会预期更靠近，指示 U-Net 持续失败的局部形态复杂区域。
- **Conclusion：** 在 19 个神经元上，平均实际最近邻距离（140.30 µm）远小于随机期望（320.13 µm）（每神经元 100 次置换），paired t-test t = -14.93、p = 1.40e-11。正向意外度（+0.300）再次确认非均匀、聚集的 split 错误，独立地佐证了 ID 26 和 40。
- **Caveats：** 与相关实验相同的 19 神经元范围和 100 次置换 Monte Carlo 预算；这是复制而非独立的新证据。
- **Reproduction：** REPRODUCED (code: revised-loading)
- **Rerun result：** 记录 n=19, actual NN=140.30 µm, expected NN=320.13 µm, paired t=-14.93, p=1.40e-11；重跑 actual NN=140.30 µm, expected NN=319.96 µm, t=-15.0355, p=1.2391e-11 → 匹配（微小的 expected-NN/t/p 差异是 Monte Carlo 置换种子噪声；显著聚集的结论不变）。
- **Generalization：** GENERALIZES
- **Across datasets：** 来源 794495：actual NN 140.30 µm 对 expected 320.68 µm, paired t=-15.07, p=1.19e-11, n=19。ds_794491：actual 76.87 µm 对 expected 147.61 µm, t=-6.31, p=2.31e-04, n=9。ds_789202：actual 219.90 µm 对 expected 385.35 µm, t=-7.15, p=1.87e-05, n=12。在两个额外数据集上方向相同（actual 约为 expected 的一半）且显著；在额外数据集上独立佐证了 ID 26/40 的聚集结果。
- **Verdict：** SOUND
- **Test：** 每神经元 actual 对 Monte Carlo expected 最近邻距离的 paired t-test（每神经元 100 次置换），t=-14.93, p=1.40e-11, n=19 个神经元（140.30 µm 对 320.13 µm）。针对神经元自身置换零假设按神经元配对是正确的独立单位设计。
- **Statistical issues：** 这是在同样 19 个神经元上对 ID 26 的复制（已承认），故是佐证而非新证据；100 次置换预算和小 n 是次要的，鉴于约 2.3× 的效应。重跑仅显示微不足道的种子噪声（expected 319.96 对 320.13 µm）。
- **Logic issues：** 无——聚集从一个被拒绝的随机性零假设推断而来；报告正确地将其框定为复制而非独立确认。
- **Verdict rationale：** 带大效应的稳健配对置换设计，重现至 Monte Carlo 种子噪声，并在方向和显著性上泛化到两个额外数据集；唯一的保留是它重复了 ID 26 的证据。

### 15. (Priority 0.275 · Surprise 0.300) 半径匹配约束将假阳性 merge 削减约 56%，但使 split 修复能力减半。
- **Run:** run-5--ground-truth-error-annotations-revised-version_2026-06-25 · **ID:** 47 · **Belief:** Leaning True → Likely True (0.7083 → 0.9167) · **Direction:** Positive
- **Tested：** 在一个 < 15 µm 邻近 split 修正启发式上添加一个形态厚度约束（叶 `node_radius` 差异 < 25%）是否相比仅邻近将假阳性 merge 减少至少 40%。
- **Conclusion：** 基线邻近启发式解决了 1,513 个真实 split 但引入了 151 个假阳性 merge 对；添加半径约束解决了 665 个真实 split，带 67 个假阳性 merge——假 merge 减少 55.63%，超过 40% 目标。正向意外度（+0.300）确认了安全收益，但它伴随真实代价：受约束启发式仅保留基线 split 解决能力的 43.95%。
- **Caveats：** 这是一个效能/安全权衡，而非免费收益——约束下近 56% 的真实 split 不再被修复，故部署需在精确率与召回率之间平衡。
- **Reproduction：** REPRODUCED (code: revised-loading)
- **Rerun result：** 记录基线解决 1,513 个 split / 151 个假 merge 对，受约束 665 个 split / 67 个假 merge 对，55.63% 减少，43.95% split 能力保留；重跑在每个数字上相同 → 完全匹配。
- **Generalization：** GENERALIZES
- **Across datasets：** 主要度量 = direct false-merge 对的减少（目标 ≥40%）。来源 794495：151→67 对 = 55.63% 减少，43.95% 的 split 保留。ds_794491：47→15 对 = 68.09% 减少，37.14% 保留。ds_789202：22→7 对 = 68.18% 减少，35.87% 保留。≥40% 假 merge 减少主张在两个额外数据集上成立（实际超过它，约 68%），且相同的效能代价（约 36–44% 的 split 保留）再现。注：次要的“canonical merges”度量在极小的计数上计算且跨数据集有噪声（来源 33.33%，794491 -100.00%，789202 200.00%），故只有 direct-pair 度量是可靠基础——而它泛化。
- **Verdict：** WEAK
- **Test：** 无推断性统计量——一个假阳性 merge 对的描述性前后比较（151→67 = 55.63% 减少）对一个 40% 目标，加上保留的 split 解决能力（43.95%）。对一个 A/B 启发式效能主张而言点估计比较是合理的，但未报告差异的置信区间或比例差异检验。
- **Statistical issues：** 主要度量基于小计数（来源 151 和 67 对，在额外数据集上降至 47→15 和 22→7），故百分比减少有宽的隐含不确定性且未给出 CI。次要的“canonical merges”度量被明确指出为噪声（-100%, +200%）并被正确搁置。
- **Logic issues：** 无——结论被恰当地框定为精确率/召回率权衡（以约 56% 的 split 修复能力损失换取安全收益），而非免费收益。
- **Verdict rationale：** direct-pair 减少超过 40% 目标并泛化（两个额外数据集上约 68%），但在小对计数上缺乏任何不确定性量化使其保持在 WEAK 而非 SOUND。

### 16. (Priority 0.275 · Surprise 0.300) Omit 边完美相邻——100% 有一个零距离 omit 邻居。
- **Run:** run-5--ground-truth-error-annotations-revised-version_2026-06-25 · **ID:** 50 · **Belief:** Leaning True → Likely True (0.7083 → 0.9167) · **Direction:** Positive
- **Tested：** 相邻 omit 边之间的网络路径距离是否显著短于随机机会预期，指示遗漏发生在局部连续段中。
- **Conclusion：** 在 57,286 个 omit 边上，100.0% 的最近邻距离为 0.00 µm（直接与另一个 omit 边共享一个节点），而匹配的随机正确边均值为 68.34 µm（中位数 36.64 µm）（Mann-Whitney U = 1.23e8, p = 0.0）。正向意外度（+0.300）强烈确认遗漏发生在连续聚集段中，补充了 ID 27 的 run 长度发现。
- **Caveats：** `review` 字段为 "N/A"（无独立审计记录），故此结果仅依赖分析本身；100% 相邻数字是 omit 以连通 run 出现的近乎定义性的后果。
- **Reproduction：** REPRODUCED (code: revised-loading)
- **Rerun result：** 记录 n=57,286 个 omit NN 对, Mann-Whitney U=123,107,614.0, p=0.0, omit NN mean/median=0.00/0.00（100% 零距离）对 random mean=68.34 µm, median=36.64 µm；重跑相同 → 完全匹配。
- **Generalization：** GENERALIZES
- **Across datasets：** 来源 794495：omit NN 100% 零距离（mean 0.00 µm）对 random mean 68.34 µm, U=1.23e8, p=0.0, n=57,286。ds_794491：omit 100% 零距离对 random mean 33.45 µm, U=2.29e8, p=0.0, n=48,922。ds_789202：omit 100% 零距离对 random mean 49.06 µm, U=5.26e8, p=0.0, n=89,380。在两个额外数据集上方向相同（omit 边总有一个零距离 omit 邻居）且 p=0.0。
- **Verdict：** MAJOR
- **Test：** omit 边（100% 在 0.00 µm）对随机正确边（mean 68.34 µm）最近邻网络距离的 Mann-Whitney U 检验，U=1.23e8, p=0.0, n=57,286。该检验运行在一个退化的点质量分布上（每个 omit 值恰为 0），这使 MWU/p 值无意义而非有信息量。
- **Statistical issues：** 100%-零结果是同义反复/定义性的：omit 错误以连通多边 run 出现，故按构造几乎每个 omit 边都与另一个 omit 边共享一个节点且最近邻距离为 0。因此对随机正确边的 MWU 比较检验的是一个必然结论（一个常数-0 组对一个正数组），而 p=0.0 不携带真实证据权重。记录本身指出 `review` 字段为 "N/A"（无独立审计），且 100% 相邻是“omit 以连通 run 出现的近乎定义性的后果”。
- **Logic issues：** 结论（“遗漏发生在局部连续段中”）重述了 run 的定义而非发现它；该检验未将聚集隔离到超出 omit 的 run 结构所保证的范围之外。
- **Verdict rationale：** 尽管它重现并泛化，统计量被应用于一个定义上为零的量，故显著的 p 值是构造的产物而非证据——结论本质上是循环的。
- **Corrected test：** 以神经元作为独立聚类单位（n=19）的神经元内标签置换检验：对每个神经元，将观测的 omit 最近邻距离与一个通过置换哪些边被标记为 "omit"（同时保持 omit 计数固定）构建的零假设比较，然后给出一个聚类符号翻转置换 p 值和一个对每神经元平均间隙的自助 CI。这替换了对常数-0 点质量组（其 100%-零结果是定义性的）的退化 Mann-Whitney，转而询问 omit 是否比其自身每神经元计数所能预测的聚集得更紧密——超出 run 结构产物的聚集。
- **Corrected result：** Omit 确实在定义性基线之外聚集。每神经元平均（观测 − 置换零假设）omit NN 距离 = −113.21 µm（为负 ⇒ 比随机更紧密），100% 的神经元低于其自身零假设，聚类符号翻转置换 p=4.9998e-05，自助 95% CI [−141.63, −87.38] µm（排除 0），对自身零假设的每神经元平均 z = −41.16。原始退化 MWU 报告 U=1.23e8, p=0.0 于一个常数-0 组（无真实证据权重）。修正检验移除了循环性并仍发现显著聚集。
- **Post-correction verdict：** UPHELD
- **Corrected generalization：** GENERALIZES——在聚类置换检验下于两个额外数据集上成立。ds_794491（n=9）：平均间隙 −107.25 µm，100% 的神经元低于零假设，置换 p=4.0998e-03, CI [−184.64, −45.19] µm（排除 0），z=−50.03。ds_789202（n=12）：平均间隙 −110.12 µm，100% 低于零假设，置换 p=7.4996e-04, CI [−189.58, −51.26] µm（排除 0），z=−63.81。真正的 omit 聚集（不只是定义性的 100%-零相邻）在所有三个大脑上显著。

### 17. (Priority 0.275 · Surprise 0.300) Merge 段比非 merge 段长一个数量级。
- **Run:** run-5--ground-truth-error-annotations-revised-version_2026-06-25 · **ID:** 55 · **Belief:** Leaning True → Likely True (0.7083 → 0.9167) · **Direction:** Positive
- **Tested：** 在修正段映射后，合并多个神经元的段在总缆长上是否显著大于正确预测的非合并段。
- **Conclusion：** 使用修正的 `node_component_id` → `swc_id` 映射，98 个 merge 段的中位缆长为 4,604.78 µm，而 366,806 个非 merge 段为 160.14 µm（Mann-Whitney U = 3.53e7, p = 0.0）。正向意外度（+0.300）确认 merge 段大一个数量级以上，强化缆长作为校对优先级特征（与 ID 34 和 56 一致）。
- **Caveats：** 非 merge 段包含许多微小碎片，压低了它们的中位数；比较是针对一个极不平衡的组（98 对 366,806），且存在长的非 merge 离群点。
- **Reproduction：** REPRODUCED (code: revised-loading)
- **Rerun result：** 记录 merge n=98（median 4,604.78 µm）, non-merge n=366,806（median 160.14 µm）, Mann-Whitney U=35,304,348.0, p=0.0；重跑相同 → 完全匹配。
- **Generalization：** GENERALIZES
- **Across datasets：** 来源 794495：merge median 4,604.78 µm 对 non-merge 160.14 µm, U=3.53e7, p=0.0（n=98 对 366,806）。ds_794491：merge median 2,132.54 µm 对 non-merge 130.85 µm, U=6.52e7, p=0.0（n=98 对 678,797）。ds_789202：merge median 4,345.25 µm 对 non-merge 148.33 µm, U=3.70e6, p=6.77e-40（n=64 对 59,122）。在两个额外数据集上方向相同（merge ≫ non-merge 一个数量级）且显著。
- **Verdict：** SOUND
- **Test：** 缆长的 Mann-Whitney U 检验，merge（n=98, median 4,604.78 µm）对 non-merge（n=366,806, median 160.14 µm），U=3.53e7, p≈0.0。MWU 是这些重度偏斜、极不平衡长度分布的正确基于秩的检验，且不同于巨 n 平凡效应案例，此处效应是一个数量级的中位差异。
- **Statistical issues：** 极端组不平衡（98 对 366,806）和 p=0.0 下限，但效应量（28× 中位数）如此之大以至于显著性不仅是 n 的产物；非 merge 碎片压低了该组中位数（已承认）。在概念上与 ID 34/56 重叠（相同的“merge 段大”主题，不同映射）。
- **Logic issues：** 无——结论（缆长是有用 merge 先验）源于一个真实的、大的分布分离。
- **Verdict rationale：** 带真正大效应量的合适非参数检验，完全重现，并在两个额外数据集上泛化（数量级分离，显著）。

### 18. (Priority 0.275 · Surprise 0.300) Merge 段在节点数和缆长上都更大（尺寸作为 merge 启发式）。
- **Run:** run-5--ground-truth-error-annotations-revised-version_2026-06-25 · **ID:** 56 · **Belief:** Leaning True → Likely True (0.7083 → 0.9167) · **Direction:** Positive
- **Tested：** 在排除少于 10 个节点的碎片后，合并段在总重建缆长和节点数上是否显著大于非合并段。
- **Conclusion：** 合并段（N = 98）的平均节点数为 4,010.49（中位数 791）和约 20,052 µm 平均缆长，而非合并段（N = 4,255）为 218.71 个节点（中位数 51）和约 1,093 µm，Mann-Whitney U = 368,540、p = 1.04e-38。正向意外度（+0.300）提供了稳健证据：段尺寸是一个强而可靠的 merge 标记特征。
- **Caveats：** 排除 sub-10-node 碎片塑造了非 merge 基线；长尾的非 merge 离群点（一些非常大的正确段）意味着尺寸是强但不完美的分类器。
- **Reproduction：** REPRODUCED (code: revised-loading)
- **Rerun result：** 记录 merge n=98（mean nodes 4,010.49, median 791, ~20,052.45 µm）, non-merge n=4,255（mean nodes 218.71, median 51, ~1,093.53 µm）, Mann-Whitney U=368,540.0, p=1.04e-38；重跑相同 → 完全匹配。
- **Generalization：** GENERALIZES
- **Across datasets：** 来源 794495：merge median 791 nodes 对 non-merge 51 nodes, U=368,540, p=1.04e-38（n=98 对 4,255）。ds_794491：merge median 337.5 nodes 对 non-merge 35 nodes, U=364,109, p=8.53e-46（n=98 对 4,042）。ds_789202：merge median 773 nodes 对 non-merge 63 nodes, U=311,767, p=1.11e-24（n=64 对 5,585）。在两个额外数据集上方向相同（merge ≫ non-merge 的节点数和缆长）且高度显著。
- **Verdict：** SOUND
- **Test：** 节点数/缆长的 Mann-Whitney U 检验，merge（n=98, median 791 nodes）对 non-merge（n=4,255, median 51 nodes，sub-10-node 碎片已排除），U=368,540, p=1.04e-38。MWU 是偏斜尺寸分布的正确基于秩的检验；排除微小碎片是一个可辩护（且已披露）的基线选择。
- **Statistical issues：** sub-10-node 排除塑造了非 merge 基线（已承认），且长尾非 merge 离群点意味着尺寸是强但不完美的分类器；约 15× 中位分离是真实效应，不只是 n 驱动的 p。
- **Logic issues：** 无——结论（段尺寸标记 merge）与所测的大分离相符。
- **Verdict rationale：** 正确的检验、大效应量、完全重现，并在两个额外数据集上高度显著地同方向泛化；已披露的碎片排除选择是唯一保留。

### 19. (Priority 0.275 · Surprise 0.300) 约 96% 的 omit 间隙低于 100 µm 的 min-cable-length 过滤器，是主导 omit 驱动因素。
- **Run:** run-5--ground-truth-error-annotations-revised-version_2026-06-25 · **ID:** 69 · **Belief:** Leaning True → Likely True (0.7083 → 0.9167) · **Direction:** Positive
- **Tested：** 是否不成比例的多数 omit 错误发生在短于 100 µm `min_cable_length` 阈值的连续间隙中，将责任归于该工具的短碎片过滤器而非大尺度分割失败。
- **Conclusion：** 在 4,145 个连续被 omit 的段（共 122,590 µm）中，3,971 个（95.80%）短于 100 µm，且这些短间隙占总被 omit 缆的 85,498 µm（69.74%）。正向意外度（+0.300）确认缓存的最小缆长过滤器——而非大尺度 U-Net 失败——是 omit 错误的主要驱动因素，这一发现对管线配置有直接含义。
- **Caveats：** 这将 omit 归因于一个可配置的预处理过滤器而非模型行为，故它更多地刻画数据集的构造而非分割模型；100 µm 阈值正是被测参数。
- **Reproduction：** REPRODUCED (code: revised-loading)
- **Rerun result：** 记录 4,145 个被 omit 的段（共 122,590.27 µm），3,971 个（95.80%）低于 100 µm 占 85,498.41 µm（69.74%）；重跑相同 → 完全匹配。
- **Generalization：** GENERALIZES
- **Across datasets：** 来源 794495：3,971/4,145 = 95.80% 的段 < 100 µm（占缆的 69.74%）。ds_794491：4,509/4,572 = 98.62%（占缆的 91.38%）。ds_789202：4,224/4,605 = 91.73%（占缆的 55.94%）。“绝大多数 omit 间隙低于 100 µm 过滤阈值”的结论在两个额外数据集上成立（92–99% 的段）；缆长比例变化更大（56–91%）但主导短间隙的发现稳健。
- **Verdict：** MINOR
- **Test：** 仅描述性比例：3,971/4,145（95.80%）的 omit 段 < 100 µm，占被 omit 缆的 85,498 µm（69.74%）。无推断性检验，这对一个“多少比例低于阈值”的描述性主张（n=4,145）是合适的。
- **Statistical issues：** 100 µm 截断是数据集自身的 `min_cable_length` 构建参数，故发现大多数 omit 间隙低于它部分上是循环的——比较是针对被测的那个阈值（已承认）。这刻画数据集构造，而非模型行为。
- **Logic issues：** 因果归因越界：“短碎片过滤器是 omit 错误的主要驱动因素而非 U-Net 失败”是一个强机制主张，然而实验仅测量间隙长度比例，而非这些间隙是否本应被重建。结论混淆了“间隙短”与“过滤器导致它们”。
- **Verdict rationale：** 描述性比例扎实并泛化（92–99%），但将 omit 错误因果地归于过滤器越过了描述性测量并依赖于正被测试的那个阈值；因该解释性循环而降级。

### 20. (Priority 0.275 · Surprise 0.300) Split 率随边接近叶端而单调上升。
- **Run:** run-5--ground-truth-error-annotations-revised-version_2026-06-25 · **ID:** 70 · **Belief:** Leaning True → Likely True (0.7083 → 0.9167) · **Direction:** Positive
- **Tested：** 重建边的 split 率是否与到最近叶端的路径距离负相关，从而碎片化在神经元的极端末端加剧。
- **Conclusion：** Split 率随到叶端距离单调下降：50 µm 内 0.92%、50–200 µm 0.65%、200 µm 以外 0.53%，前 150 µm 下降最陡。一个逻辑回归确认了显著的负关系（coef = -0.1361, p < 0.001）。正向意外度（+0.300）支持自动重建在细的、终止的分支附近最为困难，与 ID 38 的末端分支发现一致。
- **Caveats：** 预期的 Mixed-Effects GEE 模型失败（一个 `statsmodels` 语法冲突）且分析回退到标准逻辑回归，后者忽略每神经元随机效应——故神经元级非独立性未被完全考虑；绝对 split 率仍低于 1%。
- **Reproduction：** REPRODUCED (code: revised-loading)
- **Rerun result：** 记录 split 率 0.92%（<50 µm）/ 0.65%（50–200 µm）/ 0.53%（>200 µm）, 逻辑回归 coef=-0.1361, p<0.001；重跑相同（相同的率，coef=-0.1361, z=-9.924, p=0.000，跨 1,363,789 条边）。两次运行都发生相同的 GEE-回退-到-logit 路径 → 完全匹配。
- **Generalization：** PARTIAL
- **Across datasets：** 来源 794495：split 率按距离分箱 0.92%→0.65%→0.53%, 逻辑回归 coef=-0.1361, z=-9.92, p<0.001, n=1,363,789。ds_789202：0.99%→0.76%→0.40%, coef=-0.3167, z=-18.64, p≈0.0, n=1,409,045——方向相同（更强）且高度显著（成立）。ds_794491：2.04%→1.41%→1.28%（描述性分箱仍下降）但逻辑回归 coef=-0.0095, z=-0.83, p=0.409, LLR p=0.41, n=562,675——回归斜率基本为零且不显著（失去）。在 789202 上成立，在 794491 上失去；分箱趋势在各处方向上存在但正式斜率与数据集相关。
- **Verdict：** MAJOR
- **Test：** split 对标准化到叶端距离的逻辑回归，coef=-0.1361, z=-9.92, p<0.001, n=1,363,789 条边（在预期的 Mixed-Effects GEE 失败并报 "GEE.from_formula() got multiple values for argument 'groups'" 后回退到一个汇集 logit）。Pseudo R²=0.0011。
- **Statistical issues：** 伪复制 / 违反独立性——本应建模每神经元随机效应的 GEE 失败，故回退将 1.36M 个神经元内相关边汇集为彼此独立，大幅高估了有效 n 和斜率的确定性（非稳健协方差）。Pseudo R²≈0.001 表明即使在显著之处该预测因子也基本不解释方差。
- **Logic issues：** 结论（“碎片化朝叶端单调加剧”）越过一个近零-R² 斜率，且单调下降框架基于三个粗略描述性分箱而非拟合模型。
- **Verdict rationale：** 独立性结构错误（在 GEE 回退后将聚类边当作独立分析）且该发现为 PARTIAL——正式斜率在 ds_794491 上不显著（coef=-0.0095, p=0.409）——故标题性的单调梯度主张不可靠。
- **Corrected test：** 一个以神经元作为聚类、带稳健（聚类感知）标准误的可工作 GEE 逻辑回归——修正原始汇集 logit 在预期 GEE 崩溃且 1.36M 个神经元内相关边被当作独立时引入的伪复制。还有一个完全聚类级检验（独立单位 = 神经元）：每神经元 split-率对距离分箱的 Spearman rho，带自助 CI 和一个检验 rho 在各神经元间为负的 Wilcoxon signed-rank。
- **Corrected result：** 在来源上梯度在正确的聚类分析下存活，但确定性远低于汇集 logit 所暗示的。GEE coef=−0.1593, robust SE=0.0556, z=−2.867, p=4.1415e-03, robust 95% CI [−0.2682, −0.0504]（排除 0；每 SD 的 odds-ratio=0.853），19 个独立聚类。聚类级：每神经元中位 Spearman rho=−0.7714, 自助 95% CI [−0.8286, −0.6000], 94.74% 的神经元具有负斜率，Wilcoxon W=1.0, p=7.5422e-05。原始汇集-logit 回退报告 coef=−0.1361, z=−9.92, p<0.001（z 因伪复制膨胀约 3.5×：修正后 z=−2.87）。
- **Post-correction verdict：** WEAKENED
- **Corrected generalization：** PARTIAL——聚类感知 GEE 在一个额外数据集上失去显著性，尽管每神经元 Spearman 检验在各处成立。ds_794491（9 个聚类）：GEE coef=−0.0690, robust SE=0.0507, z=−1.360, p=1.7369e-01, CI [−0.1684, 0.0304]（包含 0，不显著）；但聚类 Spearman 中位 rho=−0.9429, 100% 负斜率, Wilcoxon p=1.9531e-03。ds_789202（12 个聚类）：GEE coef=−0.2722, robust SE=0.0448, z=−6.079, p=1.2095e-09, CI [−0.3600, −0.1844]（排除 0，显著）；聚类 Spearman 中位 rho=−0.8857, 100% 负, Wilcoxon p=2.4414e-04。所以在适当的聚类 GEE 下斜率在来源和 789202 上成立但在 794491 上不显著；描述性的每神经元下降在所有三个上一致。

---

## Statistical Verification — Summary

**范围：** 所有 20 个已排序假设均依据其记录的 `code`/`codeOutput`/`analysis` 以及已折入的 reproduction 和 generalization 数字进行审计。没有重新执行任何实验。

**Verdict 分项（n=20）：**

- **SOUND —— 9：** entries 2 (ID 3), 5 (ID 26), 6 (ID 27), 8 (ID 34), 9 (ID 35), 12 (ID 40), 14 (ID 45), 17 (ID 55), 18 (ID 56)。
- **WEAK —— 4：** entries 1 (ID 15), 3 (ID 12), 11 (ID 38), 15 (ID 47)。
- **MINOR —— 4：** entries 4 (ID 22), 7 (ID 33), 10 (ID 36), 19 (ID 69)。
- **MAJOR —— 3：** entries 13 (ID 42), 16 (ID 50), 20 (ID 70)。
- **CRITICAL —— 0。**

**MAJOR/CRITICAL 发现（不要按所述信任）：**

- **Entry 16 / ID 50 —— “Omit 边 100% 零距离相邻”（MAJOR）。** 核心缺陷：Mann-Whitney 检验运行在一个退化的点质量组上（每个 omit 最近邻距离恰为 0 µm）与正的随机距离比较——100%-零结果是 omit 以连通 run 出现的定义性后果，故 U=1.23e8, p=0.0 是构造产物而非证据。结论（“遗漏发生在连续聚集段中”）是循环的。
- **Entry 20 / ID 70 —— “Split 率朝叶端单调上升”（MAJOR）。** 核心缺陷：伪复制 / 违反独立性——预期的 Mixed-Effects GEE 失败（"GEE.from_formula() got multiple values for argument 'groups'"）并回退到一个汇集逻辑回归，后者将 1,363,789 个神经元内相关边当作独立（非稳健协方差），高估了确定性；pseudo R²≈0.001。该发现也为 PARTIAL——在 ds_794491 上不显著（coef=-0.0095, p=0.409）。
- **Entry 13 / ID 42 —— “Split 和 omit 共现为局部错误级联”（MAJOR）。** 核心缺陷：尽管来源 Wilcoxon 检验（stat=2.0, p=1.14e-05, n=19）有效，该发现 DOES-NOT-GENERALIZE——效应在 ds_794491 上符号反转（conditional 3.98% < baseline 4.37%, p=0.570）且在 ds_789202 上不显著（p=0.204）。在仅 19 个神经元上、约 2% 基数上的相对翻倍，在另外两个大脑上消失/反转，无法支持一个一般的“错误级联”机制。

**Benjamini–Hochberg FDR（α=0.05）。** 在 20 个 entry 中，15 个报告了正式的推断性 p 值；另外 5 个是描述性的（ID 12 GMM 拟合、ID 34 比率/ROC、ID 35 比例、ID 47 前后计数、ID 69 比例）并被排除在 FDR family 之外。若干 MWU/KS/t 检验报告 p=0.0（低于浮点下溢）；这些被当作最小秩处理。按从小到大排序，BH 阈值为 k/m·α（m=15）：

| rank k | ID | test | p | BH thr = k/15·0.05 | survives? |
|---|---|---|---|---|---|
| 1 | 36 | tortuosity t-test | ≈0 (p=0.0) | 0.0033 | yes |
| 2 | 40 | split-cascade KS | ≈0 (p=0.0) | 0.0067 | yes |
| 3 | 50 | omit-adjacency MWU | ≈0 (p=0.0) | 0.0100 | yes |
| 4 | 55 | merge cable-length MWU | ≈0 (p=0.0) | 0.0133 | yes |
| 5 | 22 | merge overlap t-test | 3.58e-44 | 0.0167 | yes |
| 6 | 56 | merge node-count MWU | 1.04e-38 | 0.0200 | yes |
| 7 | 3 | split-near-branch χ² | 4.90e-34 | 0.0233 | yes |
| 8 | 33 | split dist-to-branch MWU | 4.64e-27 | 0.0267 | yes |
| 9 | 70 | leaf-tip split logit | 2.93e-26 | 0.0300 | yes |
| 10 | 26 | split clustering paired t | 5.76e-12 | 0.0333 | yes |
| 11 | 45 | split clustering MC paired t | 1.40e-11 | 0.0367 | yes |
| 12 | 27 | omit bursty Wilcoxon | 3.81e-06 | 0.0400 | yes |
| 13 | 42 | split→omit cascade Wilcoxon | 1.14e-05 | 0.0433 | yes |
| 14 | 38 | terminal split paired t | 9.71e-04 | 0.0467 | yes |
| 15 | 15 | merge crossing angle Wilcoxon | 8.06e-03 | 0.0500 | yes |

所有 15 个报告的 p 值都在 FDR 0.05 下的 Benjamini–Hochberg 控制中存活；最大的 p 值（ID 15, p=8.06e-03）恰好坐落在其 BH 阈值（0.0500）上并通过。**没有任何报告的发现在多重比较校正下变为不显著**——即标题性的边界案例（ID 15, 0.001 < p < 0.05）是唯一接近边界者，它仍通过 BH。因此对本次 run 的限定不是来自 FDR（每个 p 都通过），而是来自（a）上述三个 MAJOR 缺陷——一个定义性 MWU（ID 50）、一个聚类数据伪复制（ID 70）和一个不泛化的级联（ID 42）；（b）ID 33 和 36 上的巨 n / 平凡效应量显著性（中位数分别相差约 30 µm 和约 0.04）；以及（c）上限有界比例上的检验选择不匹配（ID 22 t-test）和未拟合却被断言的双峰性（ID 12 GMM）。尽管 p 值一律很小，这些是科学家应最谨慎对待的发现。

---

## Excluded (no surprisal score)

辅助程序丢弃了 4 个缺少可用意外度分数的假设（`n_dropped_missing_surprisal = 4`），故它们无法被排序或包含在上文：

- run-5--ground-truth-error-annotations-revised-version_2026-06-25 · ID 7
- run-5--ground-truth-error-annotations-revised-version_2026-06-25 · ID 32
- run-5--ground-truth-error-annotations-revised-version_2026-06-25 · ID 74
- run-5--ground-truth-error-annotations-revised-version_2026-06-25 · ID 92

---

## Statistical Test Corrections — Summary

**范围：** 6 个假设被标记需要 TEST 级修复并用修正统计量重新测量（驱动程序运行了修正脚本；本节仅折入数字）。全部 6 个修正脚本在来源和两个额外数据集（`cache/dataset_cache_794491_mcl100_add.pkl`, `cache/dataset_cache_789202_mcl100_add.pkl`）上都干净运行（exitcode 0，无超时，无环境失败）。

**修正后分项（n=6）：**

- **UPHELD —— 3：** entry 3 (ID 12), entry 4 (ID 22), entry 16 (ID 50)。
- **WEAKENED —— 3：** entry 7 (ID 33), entry 10 (ID 36), entry 20 (ID 70)。
- **OVERTURNED —— 0。**

**逐 id 修正（原始检验 → 修正检验 → 修正后 verdict，附修正检验泛化）：**

| Entry / ID | Original test | Corrected test | Verdict | Corrected generalization |
|---|---|---|---|---|
| 3 / ID 12 | Imposed 2-component GMM (no test) | BIC/AIC + likelihood-ratio model selection + Hartigan dip test (bootstrap unimodal null) | UPHELD | GENERALIZES —— dip p=0.0020 且在所有三个大脑上偏好 k≥2 |
| 4 / ID 22 | One-sample t-test on ceiling-bounded proportion | One-sample Wilcoxon signed-rank + exact sign test + bootstrap CI | UPHELD | GENERALIZES —— median 1.0，三个上 CI 远高于 0.5 |
| 7 / ID 33 | Mann-Whitney U (huge-n, edges non-independent) | Cliff's delta / rank-biserial + cluster bootstrap CI + neuron-cluster permutation | WEAKENED | PARTIAL —— 效应可忽略到小（delta 0.07–0.21）；自助 CI 在来源和 ds_794491 上包含 0 |
| 10 / ID 36 | Welch/Student t-test on skewed ratio | Mann-Whitney U + Cliff's delta / AUC + bootstrap CI | WEAKENED | GENERALIZES（方向上）—— delta 0.21–0.26（小），三个上中位差约 0.01–0.02 |
| 16 / ID 50 | Mann-Whitney U on a degenerate constant-0 group | Within-neuron label-permutation, cluster=neuron + bootstrap CI | UPHELD | GENERALIZES —— 平均间隙 ≈ −110 µm，CI 排除 0，三个上 p<0.01 |
| 20 / ID 70 | Pooled logistic regression (1.36M edges as independent) | Cluster-aware GEE (robust SEs) + per-neuron Spearman / Wilcoxon | WEAKENED | PARTIAL —— GEE 在来源和 ds_789202 上显著（p=4.1e-03, 1.2e-09）但在 ds_794491 上不显著（p=0.174）；每神经元 Spearman 在各处成立 |

**在正确检验下发生改变的发现（全部 WEAKENED；无推翻）：**

- **Entry 7 / ID 33 —— split 坐落得更靠近分支节点（WEAKENED）。** 检验类型正确（MWU）但原始 p=4.64e-27 由约 30 µm 中位间隙上的 n≈940k 驱动。正确解读使用带神经元聚类 CI 的效应量：Cliff's delta=−0.0699, 聚类自助 95% CI [−0.1434, 0.0026]——可忽略且 CI 包含零；神经元聚类置换仅勉强显著（p=0.0443）。方向真实但效应实际上可忽略。
- **Entry 10 / ID 36 —— split 附近弯曲度更高（WEAKENED）。** 对一个偏斜、方差不等比率的 t-test 被替换为 Mann-Whitney + Cliff's delta：U p≈0 但 Cliff's delta=0.2096（CI [0.1808, 0.2397]）且中位差仅 0.0122——一个真实但小的分离，确认弯曲度不是可用的独立 split 预测因子。
- **Entry 20 / ID 70 —— split 率朝叶端上升（WEAKENED）。** 汇集 logit 将 1.36M 个神经元内相关边当作独立；一个适当的聚类感知 GEE 将 z 从 −9.92 压缩至 −2.87（来源上 p=4.1e-03）且斜率在 ds_794491 上变为不显著（GEE p=0.174, CI 包含 0）。每神经元 Spearman 趋势（中位 rho ≈ −0.77 至 −0.94）在所有三个大脑上一致，故梯度在方向上稳健但正式斜率与数据集相关且远弱于最初所声称。

**综述。** 应用正确检验未推翻这六个发现中的任何一个，但它对其中三个的信心进行了大幅再平衡。两个原始分析以描述性或循环方式断言的结论实际上被正确推断 STRENGTHENED：split-gap 双峰性（ID 12）在一个真正的模型选择 + dip 检验下存活，而非一个强加的 k=2 GMM；omit 聚集（ID 50）在一个剥离了定义性 100%-零产物的神经元内置换下存活——两者在所有三个大脑上 UPHELD。merge 不对称发现（ID 22）对将不当 t-test 换为 Wilcoxon/符号检验稳健（median 1.0，处处 CI 远高于 0.5）。三个 WEAKENED 发现共享验证者指出的一个根本原因：显著性与重要性或与有效样本量混淆。一旦巨大的神经元内相关样本用效应量和聚类感知检验（ID 33, ID 36）或用一个真正的 GEE（ID 70）处理，p 值急剧缩小，且效应被揭示为小（ID 33, ID 36）或与数据集相关（ID 70）。实际启示是，本次 run 的“split/omit 聚集”和“merge 段大/不对称”主题对检验选择稳健，而细粒度的空间梯度主张（到分支距离、弯曲度、到叶端距离）在统计上真实但实际上弱，不应作为独立的校对规则使用。
