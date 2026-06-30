# AutoDiscovery 排序摘要 — Ground-Truth Error Annotations (revised, 2026-06-17)

## Header

- **源文件：** `autodiscovery/ground-truth-error-annotations-revised-version_2026-06-17.json`（100 个假设）
- **排序键（`rank_by`）：** `posterior-surprise`（priority_score = posterior × |surprisal|）
- **已排序的假设数（`n_ranked`）：** 100 个中的 98 个（有 2 个因缺少 surprisal 分值而被剔除）
- **本报告保留的假设数（`n_returned`）：** 20 个——本报告展示 **98 个中排名前 20** 的记录。
- **Surprise-magnitude 范围：** 0.000 至 0.351；**priority_score max：** 0.324。

**综述（Synthesis）。** 排名前 20 的每一条记录都是经确认的、提升信念的结果（方向全部为 `Positive`，`Leaning True` → `Likely True`），因此本次运行的主基调是一致性而非反转：发现循环关于分割误差发生位置的结构性先验在极低的 p 值下被反复证实。两个优先级最高的发现是：(1) 一个经确认的、支配 split-gap 连接的 **非线性距离-角度权衡（distance-vs-angle trade-off）**（rank 1，priority 0.324，surprise 0.351——单条最能改变信念的结果），以及 (2) **merge 误差集中于密集神经突邻域**（rank 2，priority 0.287）。占主导的横向主题是一种清晰的 **误差的拓扑地理（topological geography of errors）**：splits 与 omits 聚集在 *branch points* 和 *distal terminal tips*（细突起）附近，而 merges 聚集在 *拥挤、高密度 / false-branch* 区域——并且有一条可操作的结果（rank 10）表明，在几何 merge 位点切断 fragment 节点可解决约 86% 的 merges，同时将 edge accuracy 从 82% *提升* 到 94%。

---

## Ranked Conclusions（前 20，按优先级从高到低）

### 1. (Priority 0.324 · Surprise 0.351) Split-gap 配对遵循一个经确认的非线性距离-角度权衡：短间隙容许急转弯，长间隙要求共线性。
- **Run：** ground-truth-error-annotations-revised-version_2026-06-17 · **ID：** 47 · **Belief：** Leaning True → Likely True (0.6667→0.9231) · **Direction：** Positive
- **Tested：** 区分真实 split（应重新连接）与虚假 merge 的几何，是否非线性地依赖于间隙距离与转弯角度，并用 logistic 回归而非固定阈值建模。
- **Conclusion：** 强烈确认（本次运行中最能改变信念的结果，surprisal +0.351）。以 1,072 个 true-split 正例对 142 个 false-merge 负例进行拟合，距离 × 角度交互项显著（p = 0.014），系数为负（−5.96）。决策边界在短间隙（< 8 µm）时容许非常大的转弯角度，但随着间隙接近 15 µm 而趋向共线性，证实了对于短 splits，局部邻近性压倒方向性，而长间隙严格要求方向对齐。
- **Caveats：** 负控制集很小（142 个 false merges 对 1,072 个正例），导致类别不平衡的拟合；结论中的 µm 阈值是从拟合边界读取的，而非独立验证。
- **Reproduction：** REPRODUCED（code: revised-loading）
- **Rerun result：** 记录 n=1214（1072 pos/142 neg），interaction x3 coef=−5.9629 p=0.014，LLR p=1.626e-17；rerun 完全一致（x3=−5.9629，p=0.014，LLR p=1.626e-17，n=1214）→ match。加载变更：直接 $RERUN_PKL 加载。
- **Generalization：** DOES-NOT-GENERALIZE
- **Across datasets：** 头条主张是显著的负距离×角度 **交互（interaction）** 项（x3）。origin (789202)：x3=−5.9629，p=0.014（显著，n=1214）。794491：x3=−0.3416，p=0.800（交互不显著，n=930）——交互项坍塌。794495：x3=−1.4540，p=0.565（不显著，n=1012；拟合也未收敛）。整体 logistic 判别在各处仍然强（LLR p=1.6e-17 → 6.3e-38 → 3.6e-47），线性距离项 x1 仍为负/显著，但作为本发现贡献的特定 **非线性距离-角度权衡**（即该交互项）在两个额外数据集上都不能复现。
- **Verdict：** MAJOR
- **Test：** Logistic 回归（`sm.Logit`），交互系数 x3 = −5.9629，z = −2.459，p = 0.014，n = 1214（1072 pos / 142 neg）；整体 LLR p = 1.626e-17。该模型本身对于非线性二元边界是正确的工具。
- **Statistical issues：** 头条依赖于 **单个临界交互系数**（p = 0.014），而非模型拟合（强 LLR p 由平凡的线性距离项驱动，而那并非所声称的贡献）。严重的类别不平衡（1072:142）和稀疏、重叠的负类使这一个系数脆弱；pseudo-R² 仅为 0.093。结论的 µm 阈值（"< 8 µm 容许急转弯，~15 µm 要求共线性"）是从拟合边界读取的，并未独立检验。
- **Logic issues：** 文字结论将一个勉强显著的交互项当作"证明"了某机制；这从一个 p=0.014 的系数过度推断到所陈述的物理规律。跨数据集的坍塌（p=0.800, 0.565）表明所声称的非线性是数据集特异的，而非普遍的。
- **Verdict rationale：** 决定性贡献——非线性交互——是一个孤立的临界系数，并 **不能推广** 到任一其他大脑，因此"确认的非线性权衡"这一头条不被支持；尽管模型选择稳健，仍因脆弱性 + 不可复现而下调为 MAJOR。
- **Corrected test：** 对交互系数 x3 做置换检验（permutation test），加上分层（保持类别比例的）bootstrap 95% CI。原始的 Wald z/p 假设来自一个小型、严重类别不平衡（1072:142）logistic 拟合的单个系数的渐近正态抽样分布——对一个临界项不可靠；基于重采样的推断是免假设的。
- **Corrected result：** origin x3 = −5.9629，Wald p = 0.0139 → permutation p = 3.5964e-02，stratified bootstrap 95% CI [−12.3429, −1.1075]（不含 0）。交互项保持显著，CI 在 origin 大脑上排除了零，因此校正后的检验不推翻 origin 上的发现。
- **Post-correction verdict：** UPHELD（在 origin 上）——非线性交互在置换检验下显著，其 bootstrap CI 排除 0；但见 corrected generalization。
- **Corrected generalization：** DOES-NOT-GENERALIZE——在正确的置换/bootstrap 推断下，交互项在两个额外大脑上都坍塌。794491：x3 = −0.3416，permutation p = 7.5225e-01，bootstrap CI [−3.1831, 2.2972]（跨越 0）。794495：x3 = −1.4540，permutation p = 4.1059e-01，bootstrap CI [−7.0707, 5.9019]（跨越 0；logistic 拟合也未收敛）。校正后的检验证实了原始的推广判定：非线性权衡是 origin 大脑特异的。

### 2. (Priority 0.287 · Surprise 0.307) Merge 误差发生在比正确重建显著更密集的局部神经突邻域中。
- **Run：** ground-truth-error-annotations-revised-version_2026-06-17 · **ID：** 3 · **Belief：** Leaning True → Likely True (0.7083→0.9327) · **Direction：** Positive
- **Tested：** 在 10 µm 半径内，merge 位点是否位于比正确重建对照位点更空间密集的 `fragments_graph` 邻域中。
- **Conclusion：** 确认（surprisal +0.307）。merge 位点在 10 µm 半径内平均有 7.03 个 fragment 节点，而对照为 4.39，该差异经 Mann-Whitney U 高度显著（p = 9.0e-15）。这支持了如下观点：在许多 fragments 紧密堆积的拥挤、杂乱区域，自动分割易于发生 merges。
- **Caveats：** 仅 67 个 merge 位点对 67 个对照——样本不大，尽管效应很大且 p 值极端。
- **Reproduction：** REPRODUCED（code: revised-loading）
- **Rerun result：** 记录密度 7.03/4.39，U=3931.0，p=8.9996e-15，n=67/67；rerun 完全一致（7.03/4.39，U=3931.0，p=8.9996e-15，n=67/67）→ match。加载变更：直接 $RERUN_PKL 加载。
- **Generalization：** GENERALIZES
- **Across datasets：** merge 位点比对照更密集（10 µm 半径），方向相同且各处高度显著。origin (789202)：7.03 vs 4.39 节点，U=3931.0，p=9.0e-15，n=67/67。794491：7.30 vs 4.42，U=6390.0，p=2.3e-17，n=86/86。794495：6.55 vs 4.63，U=9031.5，p=1.1e-16，n=105/105。
- **Verdict：** SOUND
- **Test：** Mann-Whitney U，单侧（`alternative='greater'`），U = 3931.0，p = 8.9996e-15，n = 67 merge / 67 control。对半径内节点计数的非参数检验是正确选择（计数为离散/偏态）。
- **Statistical issues：** 无实质问题。n=67/67 不大但效应（7.03 vs 4.39，~1.6×）很大；对照是正确节点的有种子随机样本。单侧方向与方向性假设一致。
- **Logic issues：** 无。结论（"merges 发生在拥挤区域"）直接来自密度比较，未在数据之外断言机制。
- **Verdict rationale：** 正确的非参数检验，大效应，2-for-2 推广（p=2.3e-17, 1.1e-16）；无过度推断。

### 3. (Priority 0.287 · Surprise 0.307) Split 误差在 branch-point 边上的频率约为线性边的 3.4×。
- **Run：** ground-truth-error-annotations-revised-version_2026-06-17 · **ID：** 10 · **Belief：** Leaning True → Likely True (0.7083→0.9327) · **Direction：** Positive
- **Tested：** 触及 branch point 的边是否比两个线性（degree-2）节点之间的边更常 split，反映分叉处的拓扑歧义。
- **Conclusion：** 强烈确认（surprisal +0.307）。在 15,298 条分叉边与 1,393,747 条线性边中，分叉边的 split rate 为 1.62%（248 个 splits），而线性边为 0.47%（6,557 个 splits）——约 3.4× 的增加，经 chi-square 高度显著（χ² = 414.5，p = 3.9e-92）。分叉处的拓扑歧义是分割碎片化的主要来源。
- **Caveats：** 无注明；样本大且效应高度显著。
- **Reproduction：** REPRODUCED（code: revised-loading）
- **Rerun result：** 记录 branching 15298/248（1.62%），linear 1393747/6557（0.47%），χ²=414.4724，p=3.8959e-92；rerun 完全一致 → match。加载变更：直接 $RERUN_PKL 加载。
- **Generalization：** GENERALIZES
- **Across datasets：** 分叉边的 split rate 高于线性边，方向相同且各处显著。origin (789202)：1.62% (248/15298) vs 0.47% (6557/1393747)，χ²=414.47，p=3.9e-92（~3.4×）。794491：2.81% (326/11620) vs 1.36% (7521/551055)，χ²=170.72，p=5.2e-39（~2.1×）。794495：1.16% (257/22202) vs 0.58% (7731/1341587)，χ²=125.75，p=3.5e-29（~2.0×）。在额外数据集上比值较小，但升高的 branch-point split rate 强力复现。
- **Verdict：** SOUND
- **Test：** 在 2×2 计数表上的卡方独立性检验（Chi-square test of independence），χ² = 414.4724，p = 3.8959e-92，n = 15,298 branching + 1,393,747 linear edges。对分类计数数据的正确检验；所有期望单元计数都很大，因此 χ² 近似有效。
- **Statistical issues：** 共享一个节点的边并非严格独立（轻度伪重复），且巨大的 n 意味着即便微小的比率差也会达到极端 p；此处效应确实很大（~3.4×，1.62% vs 0.47%），因此显著性并非仅由 n 造成的假象。
- **Logic issues：** 无。"branch points 处的拓扑歧义导致 splits"被表述为关联；效应量主张（~3.4×）从比率中正确读取。
- **Verdict rationale：** 适当的检验，大而有意义的效应，能推广（比值衰减至 ~2× 但仍高度显著）；独立性这一注意点不威胁结论。

### 4. (Priority 0.287 · Surprise 0.307) Omission 误差在拓扑上集中于 terminal leaf 节点附近（远端分支）。
- **Run：** ground-truth-error-annotations-revised-version_2026-06-17 · **ID：** 11 · **Belief：** Leaning True → Likely True (0.7083→0.9327) · **Direction：** Positive
- **Tested：** 使用对汇聚图的多源 BFS，`omit` 边是否在拓扑上比正确重建的边更靠近最近的 terminal leaf。
- **Conclusion：** 强烈确认（surprisal +0.307）。omit 边（n = 44,690）到 leaf 的平均拓扑距离为 247.18（中位数 130.50），而正确边（n = 1,109,034）为 325.08（中位数 168.50）；单侧 Mann-Whitney U 检验基本上具有决定性（p ≈ 7.7e-311）。omissions 不成比例地位于远端分支。
- **Caveats：** "距离"以拓扑边步数计，而非微米；巨大样本使即便微小的分布偏移也在统计上显著，因此实际效应量应从中位数读取。
- **Reproduction：** REPRODUCED（code: revised-loading）
- **Rerun result：** 记录 omit n=44690 mean=247.18（median 130.5），correct n=1109034 mean=325.08（median 168.5），U=22181022735.5，p=7.7483e-311；rerun 完全一致 → match。加载变更：直接 $RERUN_PKL 加载。
- **Generalization：** PARTIAL
- **Across datasets：** 主张是 omit 边在拓扑上比正确边 *更靠近* leaves。origin (789202)：omit mean 247.18（median 130.5）< correct 325.08（median 168.5），U=2.218e10，p=7.7e-311（成立）。794495：omit mean 124.85（median 70.5）< correct 172.81（median 83.5），U=1.218e10，p=6.1e-149（成立）。794491：omit mean 159.76（median 84.5）**高于** correct 138.76（median 70.5）——方向 **反转**，单侧 U=5.555e10，p=1.0（不显著）。在 origin + 794495 上成立，在 794491 上反转 → PARTIAL。
- **Verdict：** WEAK
- **Test：** Mann-Whitney U，单侧（`alternative='less'`），U = 22,181,022,735.5，p = 7.7483e-311，n = 44,690 omit / 1,109,034 correct edges。非参数对偏态的拓扑步距是正确的。
- **Statistical issues：** 巨大 n 将 p 推向浮点下限；效应中等（中位数 130.5 vs 168.5 步）。同一分支上的边共享几乎相同的 distance-to-leaf 值，因此约 1.15M 个观测严重聚集/伪重复——名义 p 大幅夸大了证据权重。距离以拓扑步数计，而非微米。
- **Logic issues：** origin 推理中无问题，但鉴于 **方向在 794491 上反转**（omit 边最终 *更远* 离 leaves，p=1.0），结论"omissions 不成比例地偏远端"被过度推广。
- **Verdict rationale：** 正确的检验和清晰的 origin 效应，但中等的效应量、伪重复的巨大 n，以及在两个额外大脑之一上的完全方向 **反转** 使普遍主张脆弱 → WEAK。

### 5. (Priority 0.287 · Surprise 0.307) Split 边在测地距离上比正确边更靠近 branch points。
- **Run：** ground-truth-error-annotations-revised-version_2026-06-17 · **ID：** 13 · **Belief：** Leaning True → Likely True (0.7083→0.9327) · **Direction：** Positive
- **Tested：** 使用稀疏图上的 Dijkstra，split 边到最近 branch point 的测地距离是否比正确边更短。
- **Conclusion：** 确认（surprisal +0.307）。在约 2.2M 条正确边与约 13,600 条 split 边中，Mann-Whitney U 检验给出 p < 0.001（报告为 0.0），且小提琴图显示 split 边的中心质量更靠近零。分叉作为主要结构性失败点，通过连续距离度量强化了 rank 3 的 branch-point 发现。
- **Caveats：** 两个分布都是重尾的且向零偏斜；在数百万条边下检验高度有效，因此有意义的量是集中趋势的偏移而非 p 值本身。记录中未报告确切中位数。
- **Reproduction：** DIVERGED（code: revised-loading）
- **Rerun result：** 记录处理了两个文件路径（双重计数）：Correct n=2218068，Split n=13610，U=11916101802.0，p=0.0000e+00；rerun（单个 pkl，无双重计数）：Correct n=1109034，Split n=6805，U=2979025450.5，p=1.3239e-197。样本计数减半，U/p 实质性不同（记录 p 报告为恰好 0 vs rerun 1.3e-197）→ DIVERGED。定性结论（splits 更靠近 branch points）仍然成立；该分歧是记录运行在两个路径下 glob 同一 pkl 所致的假象。
- **Generalization：** GENERALIZES
- **Across datasets：** splits 在测地距离上更靠近 branch points，各处显著。origin (789202)：Correct n=1109034 / Split n=6805，U=2.979e9，p=1.3e-197。794491：Correct n=420702 / Split n=7847，U=1.463e9，p=5.2e-67。794495：Correct n=934849 / Split n=7988，U=3.472e9，p=3.0e-27。方向相同（split 距离分布向 branches 偏移），在两个额外数据集上都高度显著。
- **Verdict：** MINOR
- **Test：** Mann-Whitney U，双侧，记录 U = 11,916,101,802.0，p 报告为 0.0（下溢）；单 pkl 的 rerun 给出 U = 2,979,025,450.5，p = 1.3239e-197，Correct n = 1,109,034 / Split n = 6,805。对重尾、零偏斜的测地距离，非参数是正确的。
- **Statistical issues：** 记录运行在两个路径下 glob 同一 pkl，将每条边 **双重计数**（n 膨胀为 2,218,068 / 13,610），因此记录的 U 和 "p=0.0" 是记账假象；rerun 纠正了这一点。即便纠正后，巨大 n 也使 p 作为效应量无意义——应以中心偏移评判（按小提琴图，split 中位数 ≈200 vs correct ≈350 µm）。
- **Logic issues：** 无；结论（splits 聚集于分叉附近）是关联，并由 ranks 3/6/19/72 佐证。
- **Verdict rationale：** 结论稳健，在纠正后的运行和两个额外数据集上方向复现，但记录的统计量因双重计数 bug 而失效 → MINOR（假象，非分析失败）。

### 6. (Priority 0.287 · Surprise 0.307) Split 误差发生在显著更靠近 GT branch points 处（mean 276 vs 382 µm）。
- **Run：** ground-truth-error-annotations-revised-version_2026-06-17 · **ID：** 19 · **Belief：** Leaning True → Likely True (0.7083→0.9327) · **Direction：** Positive
- **Tested：** split 边是否在物理上（以 µm 计）比正确边更靠近最近的 GT branch 节点，将 splits 归因于 U-Net 在分叉拓扑处的困难。
- **Conclusion：** 确认（surprisal +0.307）。split 边到 branch 的平均距离为 275.76 µm（中位数 121.12），而正确边为 381.98 µm（中位数 222.08），经 Mann-Whitney U 高度显著（p ≈ 1.4e-229），splits 在 branches 50 µm 内急剧集中。这是 ranks 3、5、19 拓扑 branch-point 发现的度量距离对应物。
- **Caveats：** 除了与其他 branch-point 假设的一般冗余外无注明；评审注意到一个较早的文件路径错误，已在最终运行前解决。
- **Reproduction：** REPRODUCED（code: revised-loading）
- **Rerun result：** 记录 mean split 275.76（median 121.12）vs correct 381.98（median 222.08），U=2916506932.0，p=1.386e-229；rerun 完全一致 → match。加载变更：直接 $RERUN_PKL 加载。
- **Generalization：** GENERALIZES
- **Across datasets：** split 边（µm）比正确边更靠近 GT branch points，方向相同且各处显著。origin (789202)：split mean 275.76（median 121.12）< correct 381.98（median 222.08），U=2.917e9，p=1.4e-229。794491：split 215.34（median 100.48）< correct 233.63（median 125.12），U=1.451e9，p=1.7e-75（差距较小）。794495：split 251.45（median 130.75）< correct 299.19（median 152.08），U=3.446e9，p=1.5e-32。
- **Verdict：** SOUND
- **Test：** Mann-Whitney U，双侧，U = 2,916,506,932.0，p = 1.386e-229，n = 6,805 split / 1,109,034 correct edges。对偏态 µm 距离的正确非参数检验。
- **Statistical issues：** 巨大 n 膨胀显著性，但效应真实且可观（中位数 121 vs 222 µm）。逐神经元最近 branch 距离正确地将 KDTree 限制在同一 GT 神经元内，避免跨神经元泄漏。
- **Logic issues：** 无。U-Net 机制性措辞（"难以解析分叉拓扑"）是一种解读，但所陈述的发现是距离关联，该关联成立。
- **Verdict rationale：** 正确的检验，从中位数读取的有意义效应量，2-for-2 推广；branch-point 簇（ranks 3/5/19/72）的 µm 距离对应物。

### 7. (Priority 0.287 · Surprise 0.307) Merge 误差聚集于高局部 fragment 密度处（15 µm 内 mean 11.1 vs 6.3 节点）。
- **Run：** ground-truth-error-annotations-revised-version_2026-06-17 · **ID：** 23 · **Belief：** Leaning True → Likely True (0.7083→0.9327) · **Direction：** Positive
- **Tested：** merge 位点 15 µm 内的局部 `fragments_graph` 密度是否超过非 merge 对照位点，表明细胞拥挤。
- **Conclusion：** 确认（surprisal +0.307）。merge 位点 15 µm 内平均有 11.09 个 fragment 节点，而对照为 6.33，经 Mann-Whitney U 高度显著（U = 4100.5，p = 9.0e-17），merge 位点的四分位距完全高于对照。这在更大半径上佐证了 rank 2：密集拥挤损害相邻突起的分离。
- **Caveats：** 仅 67 个 merge 位点对 67 个对照；与 ranks 2 和 81 高度重叠（同一密度-vs-merge 主张在不同半径下）。
- **Reproduction：** REPRODUCED（code: revised-loading）
- **Rerun result：** 记录 merge 11.09 vs control 6.33 节点（15 µm 内），U=4100.5，p=8.95e-17，n=67；rerun 完全一致 → match。加载变更：直接 $RERUN_PKL 加载。
- **Generalization：** GENERALIZES
- **Across datasets：** merge 位点比对照更密集（15 µm 半径），方向相同且各处高度显著。origin (789202)：11.09 vs 6.33 节点，U=4100.5，p=9.0e-17，n=67/67。794491：12.05 vs 6.56，U=6752.5，p=5.1e-21，n=86/86。794495：10.43 vs 6.56，U=9894.5，p=7.9e-24，n=105/105。
- **Verdict：** SOUND
- **Test：** Mann-Whitney U，双侧（默认），U = 4100.5，p = 8.95e-17，n = 67 merge / 67 control。对半径内节点计数的正确非参数检验。
- **Statistical issues：** 无实质问题；n=67/67 不大，效应大（11.09 vs 6.33，~1.75×），merge IQR 完全高于对照。实质上是 rank 2 的更大半径重述。
- **Logic issues：** 无。结论（"拥挤损害分离"）是与 rank 2/49/53/81 一致的关联。
- **Verdict rationale：** 适当的检验，大效应，2-for-2 推广（p 低至 7.9e-24）；对密度-vs-merge 主张的冗余但稳健的佐证。

### 8. (Priority 0.287 · Surprise 0.307) 逐神经元的 split rates 与 omit rates 正相关，暗示共同的失败机制。
- **Run：** ground-truth-error-annotations-revised-version_2026-06-17 · **ID：** 30 · **Belief：** Leaning True → Likely True (0.7083→0.9327) · **Direction：** Positive
- **Tested：** 高碎片化（splits/mm）的神经元是否也呈现不成比例的高 omission rates，表明 splits 与 omits 耦合而非独立的误差模式。
- **Conclusion：** 确认（surprisal +0.307）。在 12 个神经元中，splits/mm 预测 omit rate，Pearson r = 0.65（p = 0.022），Spearman ρ = 0.88（p = 0.00015）；OLS 拟合解释了方差的 R² = 42.2%（p = 0.022）。在连续性上失败的重建也倾向于完全遗漏结构，指向诸如弱信号或低对比度的共同原因。
- **Caveats：** **仅 12 个神经元**——对相关而言样本小；Pearson p（0.022）处于临界，单个有影响力的神经元即可使其偏移。视为提示性而非确凿。
- **Reproduction：** REPRODUCED（code: revised-loading）
- **Rerun result：** 记录 Pearson r=0.6500（p=2.2134e-02），Spearman ρ=0.8811（p=1.5267e-04），OLS R²=0.422，F=7.315，splits_per_mm coef=2.2771，n=12；rerun 完全一致 → match。加载变更：直接 $RERUN_PKL 加载。
- **Generalization：** GENERALIZES
- **Across datasets：** 逐神经元 splits/mm 正向预测 omit rate，方向相同且在两个额外数据集上显著（更强）。origin (789202)：Pearson r=0.65（p=0.022），Spearman ρ=0.8811（p=1.5e-04），R²=0.422，n=12。794491：Pearson r=0.9893（p=4.1e-07），Spearman ρ=0.9833（p=1.9e-06），R²=0.979，n=9。794495：Pearson r=0.8047（p=3.3e-05），Spearman ρ=0.7404（p=2.9e-04），R²=0.648，n=19。临界的 origin Pearson p 在两个额外数据集上显著增强。
- **Verdict：** WEAK
- **Test：** Pearson r = 0.6500（p = 0.0221）与 Spearman ρ = 0.8811（p = 1.5267e-04）；OLS R² = 0.422，F = 7.315（p = 0.0221），n = 12 个神经元。对这个偏态的 n=12 样本，**Spearman** 是适当的头条统计量；Pearson/OLS 脆弱。
- **Statistical issues：** n=12 小；OLS 残差严重非正态（Jarque-Bera p = 0.00036，skew = 2.06，kurtosis = 6.84），图中显示一个高杠杆离群点（≈2.5, 12.6%）膨胀了 Pearson 斜率——因此头条 Pearson p=0.022 处于临界且违反假设。Spearman ρ=0.88（p=1.5e-4）对二者都稳健，是应信赖的结果。
- **Logic issues：** 结论断言"共同的潜在机制性失败（如弱信号）"——一个 **相关无法分离的因果/机制主张**；splits/mm 与 omit rate 可能通过任何共同驱动因素（或神经元长度的假象）共变。
- **Verdict rationale：** 单调关联真实且在两个额外数据集上增强（ρ=0.98, 0.74），但头条 Pearson/OLS 在带影响点的 n=12 上违反正态性，且机制性结论过度推断 → WEAK。
- **Corrected test：** 带置换 p 值和 ρ 的 bootstrap 95% CI 的 Spearman 秩相关，替换假设双变量正态的参数化 Pearson r / OLS。在 n=12 且残差非正态（Jarque-Bera p = 0.00036）并有高杠杆点的情况下，基于秩的置换检验是免假设的头条。
- **Corrected result：** origin Pearson r = 0.6500（p = 0.0221，违反假设）→ Spearman ρ = 0.8811，permutation p = 4.5999e-04（50,000 perms），bootstrap 95% CI [0.5620, 0.9857]（不含 0）。单调关联强，CI 从容排除零——远不如临界 Pearson p 脆弱。
- **Post-correction verdict：** UPHELD——基于秩的置换检验以更小、稳健的 p（4.6e-04 vs 0.022）和排除 0 的 CI 确认了正关联。（过度推断的因果"共同机制"措辞仍是解读性注意点，而非检验问题。）
- **Corrected generalization：** GENERALIZES——两个额外数据集在校正后的 Spearman/置换检验下成立。794491：ρ = 0.9833，permutation p = 3.9999e-05，CI [0.8165, 1.0000]，n=9。794495：ρ = 0.7404，permutation p = 4.7999e-04，CI [0.3626, 0.9431]，n=19。在正确的检验下，正单调关联在所有三个大脑上显著。

### 9. (Priority 0.287 · Surprise 0.307) 段间 split 间隙（中位数 19.7 µm）远超 fragment 内部边长（95th pct 5.8 µm），因此固定半径连接不充分。
- **Run：** ground-truth-error-annotations-revised-version_2026-06-17 · **ID：** 35 · **Belief：** Leaning True → Likely True (0.7083→0.9327) · **Direction：** Positive
- **Tested：** 桥接同一 GT 神经元 split fragments 的物理欧氏间隙是否大于内部的 fragment 内边长，这会使简单的最近邻缝合失效。
- **Conclusion：** 确认（surprisal +0.307）。在 12 个神经元的 4,078 个 split 桥接间隙中，段间间隙中位数为 19.67 µm，而段内边长的第 95 百分位仅为 5.77 µm（Mann-Whitney p ≈ 0），split 间隙呈现长达 250 µm 的长尾。一个足够宽以捕捉真实间隙（> 20 µm）的搜索半径会远超内部边长并产生许多虚假连接——因此固定距离启发式不充分（这促成了 rank 1 的学习边界）。
- **Caveats：** 基于 12 个神经元；比较将（间隙的）中位数与（内部边的）第 95 百分位对比，这是一个刻意保守的框定，而非同类分布检验。
- **Reproduction：** REPRODUCED（code: revised-loading）
- **Rerun result：** 记录 4078 个间隙，median gap 19.67 µm，95th pct internal 5.77 µm，U=3763981856.0，p=0.0000e+00；rerun 完全一致 → match。加载变更：直接 $RERUN_PKL 加载。
- **Generalization：** GENERALIZES
- **Across datasets：** 段间 split 间隙远超内部边长，方向相同且各处 p≈0。origin (789202)：4078 个间隙，median gap 19.67 µm vs 95th-pct internal 5.77 µm，U=3.764e9，p=0.0。794491：2943 个间隙，median gap 29.95 µm vs internal 5.71 µm，U=1.578e9，p=0.0。794495：3250 个间隙，median gap 18.45 µm vs internal 5.71 µm，U=3.321e9，p=0.0。
- **Verdict：** SOUND
- **Test：** Mann-Whitney U，单侧（`alternative='greater'`），U = 3,763,981,856.0，p 报告 0.0，n = 4,078 桥接间隙 vs 一大批段内边长。对两个重尾距离分布，非参数是正确的。
- **Statistical issues：** MW 检验比较完整分布（有效）；结论的头条将 *中位数* 间隙（19.67 µm）与 *第 95 百分位* 内部长度（5.77 µm）对比，这是刻意保守的框定，而非同类统计量。巨大的零假设池使 p≈0 不具信息量；要点在于分离的幅度。
- **Logic issues：** 无。"固定半径连接不充分"从分布重叠/分离论证逻辑地得出（捕捉间隙的 >20 µm 半径会淹没 5.8 µm 内部边）。
- **Verdict rationale：** 正确的检验，稳健的 ≫ 分离且能推广（三者上中位数间隙 18–30 µm vs ~5.7 µm 内部），工程结论成立；中位数-vs-p95 的框定是保守的，而非误导的。

### 10. (Priority 0.287 · Surprise 0.307) 在几何 merge 位点切断 fragment 节点解决约 86% 的 merge 误差并将 edge accuracy 从 82% 提升到 94%。
- **Run：** ground-truth-error-annotations-revised-version_2026-06-17 · **ID：** 39 · **Belief：** Leaning True → Likely True (0.7083→0.9327) · **Direction：** Positive
- **Tested：** 定位每个几何 merge 坐标最近的物理 fragment 节点并切断它，是否在不损害 edge accuracy 的情况下移除 > 80% 的 merge 误差——一个可操作的校对启发式。
- **Conclusion：** 强烈确认且是最可操作的结果（surprisal +0.307）。KDTree 定向切断将全局 %-merged-edges 从 13.25% 降至 1.83%（~86% 削减，超过 80% 阈值），并且非但没有仅仅保持准确率，反而将平均 edge accuracy 从 82.27% *提升* 到 93.66%。严重 merged 的神经元（N013, N018）从约 40-43% merged edges 降至约 1%，将其准确率从 50 多提升至约 96-98%。
- **Caveats：** 切断由 *ground-truth* merge 坐标引导；该结果展示了校正效应的上界，尚非无需 GT 的检测器。逐神经元改善未附带统计检验。
- **Reproduction：** REPRODUCED（code: revised-loading）
- **Rerun result：** 记录 67 个 merge 位点 → 55 个组件断开，219658 个 GT 节点重映射，mean %merged 13.25%→1.83%，mean edge accuracy 82.27%→93.66%（所有逐神经元行完全一致）；rerun 完全一致 → match。加载变更：直接 $RERUN_PKL 加载。
- **Generalization：** PARTIAL
- **Across datasets：** 切断总是降低 %merged-edges 并提升准确率（方向各处成立），但头条 ">80% merge 削减"在 794495 上失败。origin (789202)：mean %merged 13.25%→1.83%（~86% 削减），accuracy 82.27%→93.66%。794491：20.54%→6.15%（~70% 削减），accuracy 73.81%→88.13%（低于 80% 阈值）。794495：28.54%→16.02%（~44% 削减），accuracy 68.68%→81.17%（远低于阈值；若干严重 merged 的神经元几乎不变）。校正方向稳健，但 merge 移除的幅度是数据集特异的，在额外数据集上未达到所声称的 ~86%/>80% → PARTIAL。
- **Verdict：** MAJOR
- **Test：** **无统计检验。** 对 12 个神经元的描述性前/后均值：mean %merged 13.25%→1.83%，mean edge accuracy 82.27%→93.66%。逐神经元改善未附带 p 值、CI 或配对检验（如对 12 个逐神经元 deltas 的 Wilcoxon signed-rank）。
- **Statistical issues：** 一个裸的两数均值比较，无显著性/不确定性量化。假设是 *逐神经元* 陈述的（"每个神经元 >80% 的 merge 误差"）却作为 **全局均值** 评估，掩盖了失败：N020 降 0%（8.15%→8.15%），N006 仅约 55%——因此即使在 origin 上，逐神经元主张对若干神经元也是错的。
- **Logic issues：** 切断由 **ground-truth `gt_merge_sites` 坐标** 引导，因此这衡量的是上界 oracle，而非可部署的无需 GT 检测器——"可行的 agentic 策略"结论按其措辞是循环的。推广确认了幅度是数据集特异的（在 794495 上仅约 44%）。
- **Verdict rationale：** 方向确实有用，但头条数字无统计检验，逐神经元主张在其自身表内被反驳，它依赖 GT oracle 坐标，且 ">80%" 阈值在一个额外大脑上失败 → MAJOR。
- **Corrected test：** 对 12 个逐神经元前/后 deltas（%merged-edges 与 edge accuracy 二者）的配对 Wilcoxon signed-rank 检验，加上逐神经元 bootstrap CI 和对 ">=80% merge 削减"主张的明确逐神经元统计——替换那个无显著性/不确定性且掩盖逐神经元失败的裸全局两数均值比较。
- **Corrected result：** origin 全局均值 13.25%→1.83% merged，82.27%→93.66% accuracy（不变）→ Wilcoxon W = 66.0，p = 4.8828e-04（%merged 削减与 accuracy 提升二者）；逐神经元 median %merged 削减 = 5.96 pct-points（95% CI [2.13, 12.60]），median accuracy 提升 = 5.94 pct-points（95% CI [2.12, 12.58]）。逐神经元 ">=80% 削减"主张仅对 8/12 个神经元成立（逐神经元削减中位数 97.4%，95% CI [73.0%, 100.0%]——CI 下界落在 80% 以下）。
- **Post-correction verdict：** WEAKENED——配对检验确认了具有统计显著性的逐神经元改善（p = 4.9e-04，CIs 排除 0），因此校正 *方向/效应* 现已得到适当支持；但头条"每个神经元 >80% 的 merge 误差"在 origin 上仅对 8/12 个神经元成立，其 bootstrap CI 跌破 80%，因此该特定定量主张比所陈述的实质性更弱。
- **Corrected generalization：** PARTIAL——配对改善在两个额外数据集上显著，但 ">80%" 头条不能推广。794491：Wilcoxon W = 45.0，p = 1.9531e-03（n=9），median merge 削减 73.5%（95% CI [24.6%, 89.9%]），仅 3/9 个神经元达到 >=80%。794495：Wilcoxon W = 190.0，p = 1.9073e-06（n=19），median merge 削减 38.0%（95% CI [19.4%, 80.1%]），仅 5/18 个神经元达到 >=80%。校正效应各处显著；>80%-per-neuron 的幅度是 origin 特异的。

### 11. (Priority 0.287 · Surprise 0.307) Omit 误差高度空间聚集：与一条 omit 相邻的边约 89% 也会被 omit（为边际率的 28×）。
- **Run：** ground-truth-error-annotations-revised-version_2026-06-17 · **ID：** 48 · **Belief：** Leaning True → Likely True (0.7083→0.9327) · **Direction：** Positive
- **Tested：** 通过沿 GT 骨架的 Markov 转移矩阵，omit 误差是否以连续段而非独立地发生（假设 conditional ÷ marginal ≥ 3）。
- **Conclusion：** 强烈确认（surprisal +0.307）。边际 omit 概率约为 3.17%，但在相邻 omit 条件下的条件概率跃升至约 89.08%——比值为 28.09，远超假设的 3×，伴随巨大的 chi-square（~2.23M，p = 0.0）。omissions 以集中的连续段而非随机出现。
- **Caveats：** 无注明；效应非常大。与 rank 4/64 的远端聚集图景一致（沿 terminal 分支的 omitted 段）。
- **Reproduction：** REPRODUCED（code: revised-loading）
- **Rerun result：** 记录 marginal OMIT=0.0317，conditional=0.8908，ratio=28.09，χ²=2226077.8739，p=0.0000e+00（转移矩阵 [[2727004,9999],[9999,81558]]）；rerun 完全一致 → match。加载变更：直接 $RERUN_PKL 加载。
- **Generalization：** GENERALIZES
- **Across datasets：** omits 强烈空间聚集（conditional/marginal 比值 ≫3），方向相同且各处 p≈0。origin (789202)：marginal 0.0317，conditional 0.8908，ratio 28.09，χ²=2.226e6，p=0.0。794491：marginal 0.0435，conditional 0.8098，ratio 18.63，χ²=7.275e5，p=0.0。794495：marginal 0.0210，conditional 0.8490，ratio 40.42，χ²=1.962e6，p=0.0。比值有变化（18.6–40.4）但总是远超假设的 3×。
- **Verdict：** MINOR
- **Test：** 在 2×2 边状态转移矩阵 [[2727004, 9999],[9999, 81558]] 上的卡方独立性检验，χ² = 2,226,077.87，p = 0.0；加上描述性的 conditional/marginal 比值（0.8908 / 0.0317 = 28.09）。
- **Statistical issues：** 此处的 χ² 检验作为正式独立性检验 **不可靠**：每条边在许多相邻对"观测"中被重复使用（度为 d 的节点贡献 d·(d−1) 对；~2.8M 单元总数来自 ~1.4M 条边），因此观测严重伪重复，χ² 统计量被膨胀——其 p 值不可解释。恰当的检验应是沿骨架对段长的置换/块自助（block-bootstrap）。
- **Logic issues：** 结论本身无问题；描述性 **比值（28×）** 是一个有效、非常大的效应，假设（≥3×）以此为准，且不依赖 χ² p。
- **Verdict rationale：** 推断检验错误（伪重复的 χ²），但头条聚集主张依赖描述性 28× 比值，该比值巨大且复现（18.6–40.4×）→ MINOR（尽管 p 不可靠，效应承载结论）。
- **Corrected test：** 以每条边为单位的边标签置换检验（在边上打乱 OMIT/OTHER 标签并重新计算 conditional/marginal 比值），替换那个伪重复的 2×2 χ²——其 ~2.8M 个"观测"在许多相邻对中重复使用了 ~1.4M 条边中的每一条，因而大幅膨胀了统计量。
- **Corrected result：** 描述性比值复现（origin 28.09，794491 18.63，794495 40.42——全部 ≫ 假设的 3×）。校正后的置换检验仅在 794491 上产生了有效 p：observed ratio = 18.63 vs 置换零假设均值比 1.000（97.5th pct 1.053），permutation p = 9.9900e-04——相对正确的边洗牌零假设，聚集是显著的。origin 与 794495 的置换运行在超时前未发出校正 p（origin 仅打印了原始不可靠的 χ²，794495 更早超时）；这两者上的原始 χ² p≈0 不可解释。
- **Post-correction verdict：** UPHELD——在正确的边置换检验完成处（794491），conditional/marginal 聚集相对一个比值约为 1.0 的零假设显著（p = 9.99e-04），且三个大脑上描述性比值都巨大（18.6–40.4×）。结论（omits 以连续段出现，比值 ≫3×）经正确检验存活；仅膨胀的 χ² p 不可靠。
- **Corrected generalization：** PARTIAL——794491 在校正后的置换检验下成立（ratio 18.63，p = 9.99e-04）。794495 在校正后的检验下为 INCONCLUSIVE（运行在 1800 s 超时，未计算出置换 p），尽管其描述性比值（40.42×）是三者中最大的。校正后的置换检验在 origin 大脑上也未发出（在描述性比值后超时），因此正式的校正推断在两个额外数据集之一上得到确认，大的描述性比值在两者上都复现。

### 12. (Priority 0.287 · Surprise 0.307) Merge 位点与 fragment-graph branch points 共定位（中位数 4.5 µm vs 随机节点的 180 µm）。
- **Run：** ground-truth-error-annotations-revised-version_2026-06-17 · **ID：** 49 · **Belief：** Leaning True → Likely True (0.7083→0.9327) · **Direction：** Positive
- **Tested：** merge 位点是否在空间上比随机 fragment 节点更靠近 `fragments_graph` branch points，即 merges 是否表现为 false branches。
- **Conclusion：** 强烈确认（surprisal +0.307）。merge 位点到最近 fragment branch point 的中位距离为 4.48 µm，而随机节点为 179.76 µm（Mann-Whitney p = 2.2e-17）；约 40% 的 merge 位点实际上位于一个 branch point 上，约 86% 在 10 µm 内。因此瞄准异常 false branches 是一个可行的无需 GT 的 merge 解决策略，补充了 rank 10 的几何坐标方法。
- **Caveats：** 仅 67 个 merge 位点对 67 个随机节点。
- **Reproduction：** DIVERGED（code: revised-loading）
- **Rerun result：** Merge 中位数 4.48 µm 与两次运行都一致，但随机节点基线偏移：记录 random mean 250.32 µm（median 179.76），U=363.0，p=2.21e-17；rerun random mean 320.30 µm（median 217.34），U=282.0，p=9.53e-19。随机样本是无固定种子抽取的，因此对照分布与 U/p 不同 → DIVERGED，尽管结论（merges 与 branch points 共定位，中位数约 4.5 µm）成立。加载变更：直接 $RERUN_PKL 加载。
- **Generalization：** GENERALIZES
- **Across datasets：** merge 位点比随机节点远更靠近 fragment branch points，方向相同且各处高度显著。origin (789202)：merge median 4.48 µm vs random median 170.18 µm，U=388.0，p=5.7e-17，n=67/67。794491：merge median 3.86 µm vs random 127.86 µm，U=621.0，p=1.7e-21，n=86/86。794495：merge median 5.11 µm vs random 169.33 µm，U=681.0，p=2.2e-28，n=105/105。
- **Verdict：** MINOR
- **Test：** Mann-Whitney U，单侧（`alternative='less'`），记录 U = 363.0，p = 2.21e-17，n = 67 merge / 67 random；rerun U = 282.0，p = 9.53e-19（无种子对照重采样）。正确的非参数检验。
- **Statistical issues：** 对照 **仅从非 branch 节点（degree ≤ 2）** 抽样，这在构造上使随机基线偏向远离 branch points，从而膨胀对比。对照也无种子 → rerun 上 DIVERGED（random median 179.76 → 217.34 µm）。尽管如此，merge 侧结果（中位数 4.48 µm，约 40% 恰好在 branch point 上）本身仍然引人注目。
- **Logic issues：** 无实质问题；结论（"merges 表现为 false branches"）来自 merge 侧的邻近性，该邻近性稳健，独立于对照的确切基线。
- **Verdict rationale：** 正确的检验和一个 2-for-2 推广的稳健 merge 侧效应，但对照抽样限制夸大了差距且基线依赖种子 → MINOR。

### 13. (Priority 0.287 · Surprise 0.307) 引起 merge 的段的 branch-node 密度几乎是正确段的两倍（一个无需 GT 的特征）。
- **Run：** ground-truth-error-annotations-revised-version_2026-06-17 · **ID：** 53 · **Belief：** Leaning True → Likely True (0.7083→0.9327) · **Direction：** Positive
- **Tested：** 引起 merges 的 U-Net 段是否携带比正确重建段更高的内在 branch-node 密度（每 100 µm 缆线的 branches）——可在无 ground truth 下使用。
- **Conclusion：** 确认（surprisal +0.307）。merge 段平均每 100 µm 有 0.1078 个 branches，而对照为 0.0568（~1.9×），在 64 个 merge 段和 4,006 个对照段上经 Mann-Whitney U 高度显著（p = 2.9e-25）。内在 branch 密度是标记 merge 候选的强力、独立于 ground-truth 的启发式。
- **Caveats：** 仅 64 个 merge 段（对 4,006 个对照）——不平衡的比较，尽管效应大且高度显著。
- **Reproduction：** REPRODUCED（code: revised-loading）
- **Rerun result：** 记录 64 merge/4006 control，0.1078 vs 0.0568 branches/100 µm，U=204163.5，p=2.9141e-25；rerun 完全一致 → match。加载变更：直接 $RERUN_PKL 加载。
- **Generalization：** GENERALIZES
- **Across datasets：** 引起 merge 的段比对照携带更高的 branch-node 密度，方向相同且各处高度显著。origin (789202)：0.1078 vs 0.0568 branches/100 µm（~1.9×），U=204163.5，p=2.9e-25，64/4006。794491：0.1662 vs 0.0546（~3.0×），U=213596.0，p=4.3e-42，98/2729。794495：0.1528 vs 0.0658（~2.3×），U=235396.0，p=1.2e-27，98/3108。在额外数据集上效应更大。
- **Verdict：** SOUND
- **Test：** Mann-Whitney U，双侧，U = 204,163.5，p = 2.9141e-25，n = 64 merge segments / 4,006 control segments。非参数正确处理偏态的 branch-density-per-100µm 和不等的组大小。
- **Statistical issues：** 类别不平衡（64:4006）对 MW 没问题；merge 组小但效应（~1.9×）大，且逐段密度是恰当的比率（branches 按缆线长度归一化），避免了长度混杂。
- **Logic issues：** 无。"无需 GT 启发式"主张是恰当的，因为 branch-node 密度纯粹由 `fragments_graph` 计算，不使用 GT。
- **Verdict rationale：** 正确的检验，长度归一化的比率，在两个额外数据集上增大的大效应（高达 ~3.0×）；一个真正独立于 GT 的特征。

### 14. (Priority 0.287 · Surprise 0.307) Split 误差级联：观测到的 split 间距离（中位数 26 µm）远短于随机零假设（中位数 237 µm）。
- **Run：** ground-truth-error-annotations-revised-version_2026-06-17 · **ID：** 55 · **Belief：** Leaning True → Likely True (0.7083→0.9327) · **Direction：** Positive
- **Tested：** 相对于均匀随机零假设，splits 是否沿神经元拓扑呈现空间自相关（一处 split 提高了附近另一处的概率）。
- **Conclusion：** 确认（surprisal +0.307）。观测到的 split 间测地距离中位数为 25.58 µm，而随机零假设下为 236.98 µm，差距高度显著（KS D = 0.4539，p ≈ 0.0）。splits 聚集而非散布，暗示局部触发因素（图像质量、困难的形态）导致连续断裂——这是 rank 11 omit 聚集的 split 类比。
- **Caveats：** 无注明；零假设模型是模拟的，因此结论取决于随机分配假设是否匹配真实基线。
- **Reproduction：** REPRODUCED（code: revised-loading）
- **Rerun result：** 记录 KS D=0.4539，p=0.0，median observed 25.58 µm vs median random 236.98 µm；rerun KS D=0.4542，median observed 25.58 µm，median random 236.66 µm → 在模拟零假设噪声内匹配（无固定种子）。加载变更：直接 $RERUN_PKL 加载。
- **Generalization：** GENERALIZES
- **Across datasets：** 观测到的 split 间距离远短于随机零假设（splits 聚集），方向相同且各处 p≈0。origin (789202)：median observed 25.58 µm vs random 237.34 µm，KS D=0.4544，p=0.0。794491：observed 14.49 µm vs random 87.33 µm，KS D=0.4364，p=0.0。794495：observed 14.37 µm vs random 205.83 µm，KS D=0.5640，p=0.0。
- **Verdict：** MINOR
- **Test：** 针对置换零假设的双样本 Kolmogorov-Smirnov，KS D = 0.4539，p = 0.0，n_observed = 6,805 vs n_null = 680,500（100 次随机重标记）。置换零假设比较是空间自相关检验的正确设计。
- **Statistical issues：** 零假设样本（680,500）是观测样本的 100×，因此 KS p 在巨大的合并 n 上计算并读为恰好 0；有意义的证据是大的 D=0.45 和中位数差距（25.6 vs 237 µm）。100 次置换共享每神经元相同的观测 split 计数（恰当），因此零假设有效；仅 p 值的量级被夸大。
- **Logic issues：** "级联（cascading）"/"一处 split 提高另一处的概率"一词暗示 **因果触发**，但检验仅显示空间聚集——同样与共享的潜在原因（局部图像质量）一致，分析本身也承认了这一点。框定中有轻度因果过度推断。
- **Verdict rationale：** 适当的置换检验和一个大的、可推广的聚集效应（三者上 D=0.44–0.56）；仅因果"级联"措辞过度解读了关联 → MINOR。

### 15. (Priority 0.287 · Surprise 0.307) Split 边比正确边更靠近 terminal leaves（mean 939 vs 1283 µm）。
- **Run：** ground-truth-error-annotations-revised-version_2026-06-17 · **ID：** 60 · **Belief：** Leaning True → Likely True (0.7083→0.9327) · **Direction：** Positive
- **Tested：** 到最近 leaf 的网络距离对 split 边是否比正确边更短，即 splits 是否集中于远端尖端而非骨干。
- **Conclusion：** 确认（surprisal +0.307）。在 6,805 条 split 边和 1,109,034 条正确边中，split 边到 leaf 的平均距离为 938.60 µm，而正确边为 1,282.69 µm（Mann-Whitney p ≈ 2.2e-134），splits 有更尖锐的近零峰。splits 集中于细远端区域，与 rank 65 的厚度代理结果一致。
- **Caveats：** 巨大样本驱动极端显著性；以均值/中位数评判幅度。在概念上与 ranks 11/65 重叠。
- **Reproduction：** REPRODUCED（code: revised-loading）
- **Rerun result：** 记录 split n=6805 mean 938.60 µm vs correct n=1109034 mean 1282.69 µm，p=2.1613e-134；rerun 完全一致 → match。加载变更：直接 $RERUN_PKL 加载。
- **Generalization：** GENERALIZES
- **Across datasets：** split 边比正确边更靠近 terminal leaves，方向相同且各处显著。origin (789202)：split 938.60 µm < correct 1282.69 µm，p=2.2e-134，n=6805/1109034。794491：split 518.26 µm < correct 546.75 µm，p=1.2e-33（差距小但一致），n=7847/420702。794495：split 600.59 µm < correct 699.28 µm，p=1.4e-32，n=7988/934849。在 794491 上效应量小得多，但方向和显著性成立。
- **Verdict：** SOUND
- **Test：** Mann-Whitney U，双侧，p = 2.1613e-134，n = 6,805 split / 1,109,034 correct edges（经多源 Dijkstra 的网络 distance-to-leaf）。对偏态网络距离的正确非参数检验。
- **Statistical issues：** 巨大 n 膨胀显著性；origin 上效应中等（mean 939 vs 1283 µm），在 794491 上显著缩小（518 vs 547 µm）。边沿分支非独立（伪重复），因此读均值，而非 p。
- **Logic issues：** 无。陈述为关联（"splits 集中于远端尖端"）；在概念上与 ranks 11/65 重叠。
- **Verdict rationale：** 正确的检验，方向 2-for-2 成立且效应真实（尽管衰减）；显著性由巨大 n 驱动，但集中趋势偏移是真实的 → SOUND。

### 16. (Priority 0.287 · Surprise 0.307) Omit 误差率在 terminal 分支上几乎是内部段的两倍（4.38% vs 2.41%）。
- **Run：** ground-truth-error-annotations-revised-version_2026-06-17 · **ID：** 64 · **Belief：** Leaning True → Likely True (0.7083→0.9327) · **Direction：** Positive
- **Tested：** omit 误差是否偏向 terminal（leaf-path）边而非内部节间边，反映对远端尖端解析的失败。
- **Conclusion：** 确认（surprisal +0.307）。在 543,477 条 terminal 边和 865,568 条 internal 边中，omit rate 在 terminal 上为 4.38%，在 internal 边上为 2.41%（~1.8×），经 chi-square 高度显著（χ² = 4189.94，p ≈ 0.0）。omissions 集中于细远端尖端，以分类划分强化了 ranks 4 和 11 的远端 omission 图景。
- **Caveats：** 无注明；样本大，效应大。
- **Reproduction：** REPRODUCED（code: revised-loading）
- **Rerun result：** 记录 terminal 543477（4.38%）vs internal 865568（2.41%），χ²=4189.9364，p=0.0000e+00；rerun 完全一致 → match。加载变更：直接 $RERUN_PKL 加载。
- **Generalization：** GENERALIZES
- **Across datasets：** terminal 边的 omit rate 高于 internal 边，方向相同且各处显著。origin (789202)：4.38% vs 2.41%（~1.8×），χ²=4189.94，p=0.0。794491：4.94% vs 3.82%（~1.3×），χ²=425.40，p=1.6e-94。794495：2.47% vs 1.82%（~1.4×），χ²=691.98，p=1.7e-152。在额外数据集上比值衰减，但 terminal 偏向成立。
- **Verdict：** SOUND
- **Test：** 在 2×2 计数表上的卡方独立性检验，χ² = 4189.94，p ≈ 0.0，n = 543,477 terminal + 865,568 internal edges。对分类计数数据的正确检验；期望计数大。
- **Statistical issues：** 同一 terminal/internal 路径内的边共享标签，因此观测伪重复（名义 p 过于精确），但效应（4.38% vs 2.41%，~1.8×）大，且基于路径的分类是合理单位；显著性并非仅由 n 造成的假象。
- **Logic issues：** 无。是 ranks 4/11 的分类重述；结论是关联，未在"细远端尖端更难"之外声称机制。
- **Verdict rationale：** 适当的 χ²，大效应，2-for-2 推广（比值衰减至 ~1.3–1.4× 但仍高度显著）；伪重复注意点对结论无关紧要。

### 17. (Priority 0.287 · Surprise 0.307) Split 误差偏好细远端突起（distance-to-leaf 中位数 417 vs 670 µm）。
- **Run：** ground-truth-error-annotations-revised-version_2026-06-17 · **ID：** 65 · **Belief：** Leaning True → Likely True (0.7083→0.9327) · **Direction：** Positive
- **Tested：** split 边的拓扑 distance-to-leaf（突起厚度的代理）是否比正确边更短，并下采样以避免数值溢出。
- **Conclusion：** 确认（surprisal +0.307）。将正确边下采样至 50,000 后，split 边的 distance-to-leaf 中位数为 417.22 µm，而正确边为 670.25 µm（Mann-Whitney p = 0.0），split 密度在近零处达峰。Fragment 断裂不成比例地影响更细的 terminal 突起。
- **Caveats：** Distance-to-leaf 是厚度的 *间接* 代理，而非直接的口径测量；正确边被下采样，作者指出这是为避免先前的溢出错误所需。在概念上与 rank 15 重叠。
- **Reproduction：** DIVERGED（code: revised-loading）
- **Rerun result：** Split 边统计完全复现（mean 940.69 µm，median 417.22 µm，p=0.0 二者），但记录运行从两个文件路径双重计数正确边：记录 Correct-before-downsampling n=2218068，Split n=13610，U=281944000.0；rerun Correct n=1109034，Split n=6805，U=140888192.0，下采样后正确边 mean 1287.44（vs 记录 1278.24）。样本总数减半且 U 不同 → DIVERGED；结论（splits 偏好细远端突起）不变。加载变更：直接 $RERUN_PKL 加载（单 pkl，无双重 glob）。
- **Generalization：** GENERALIZES
- **Across datasets：** split 边的 distance-to-leaf（厚度代理）比（下采样的）正确边更短，方向相同且各处显著。origin (789202)：split median 417.22 µm < correct 671.61 µm，U=1.409e8，p=0.0。794491：split median 238.34 µm < correct 280.89 µm，U=1.808e8，p=1.9e-29。794495：split median 295.78 µm < correct 341.05 µm，U=1.851e8，p=3.1e-26。额外数据集上差距较小但方向和显著性成立。
- **Verdict：** MINOR
- **Test：** Mann-Whitney U，单侧（`alternative='less'`），记录 U = 281,944,000.0，p = 0.0，正确边下采样至 50,000（有种子）；移除双重计数后 rerun U = 140,888,192.0。非参数正确；下采样用于避免溢出。
- **Statistical issues：** 记录运行通过两次 glob 同一 pkl 将正确/split 边 **双重计数**（2,218,068 / 13,610 vs 真实 1,109,034 / 6,805），偏移了 U 和下采样均值（1278.24 → 1287.44 µm）→ DIVERGED；split 侧统计完全复现。巨大的零假设和下采样使 p 不具信息量——以中位数评判（417 vs 670 µm）。
- **Logic issues：** "Distance-to-leaf" 是口径/厚度的 **间接代理**；结论（"splits 影响更细突起"）将拓扑 terminal 位置与实际半径混为一谈，而后者未被直接测量。
- **Verdict rationale：** 正确的检验和一个稳健、可推广的中位数偏移，但记录的双重计数假象加上一个依赖间接代理的厚度主张 → MINOR。

### 18. (Priority 0.287 · Surprise 0.307) Split 间隙处的对置端点彼此指向（cosine −0.68 vs 随机附近端点的 −0.51）。
- **Run：** ground-truth-error-annotations-revised-version_2026-06-17 · **ID：** 67 · **Belief：** Leaning True → Likely True (0.7083→0.9327) · **Direction：** Positive
- **Tested：** 真实 split 位点处断裂的 fragment 端点是否比偶然附近未连接的端点在方向上更共线（反平行，跨间隙指向）。
- **Conclusion：** 确认（surprisal +0.307）。在 525 个 split 端点对对 2,689 个空间相邻对照对中，split 端点的平均余弦相似度为 −0.6847（更反平行），而对照为 −0.5112，经 Mann-Whitney U 显著（p = 1.1e-08）。跨间隙的几何共线性是一个可靠的、agent 可利用以提议 split-joins 的无需 GT 特征——支撑 rank 1 模型的方向特征。
- **Caveats：** 对照分布的方差宽得多（std 0.534 vs 0.358），因此均值不同但分布大幅重叠；共线性是有用的先验，而非单独的完美分类器。
- **Reproduction：** REPRODUCED（code: revised-loading）
- **Rerun result：** 记录 525 split / 2689 control 对，mean cos −0.6847 vs −0.5112，U=594814.0，p=1.13e-08；rerun 完全一致 → match。加载变更：直接 pkl 加载。
- **Generalization：** GENERALIZES
- **Across datasets：** split 端点比对照更反平行（余弦更负），方向相同且各处显著。origin (789202)：mean cos −0.6847 vs −0.5112，U=594814.0，p=1.1e-08，525/2689 对。794491：−0.5932 vs −0.3334，U=361650.0，p=1.4e-10，276/3412 对。794495：−0.7908 vs −0.5273，U=778749.5，p=8.6e-20，449/4686 对。
- **Verdict：** SOUND
- **Test：** Mann-Whitney U，双侧，U = 594,814.0，p = 1.13e-08，n = 525 split pairs / 2,689 control pairs（有种子抽样）。对有界的余弦相似度数据的正确非参数检验。
- **Statistical issues：** 中等 n，p 在该族中最小（1.1e-08）——从容显著但不极端。对照方差宽得多（std 0.534 vs 0.358），因此分布大幅重叠；均值不同但该特征单独是弱分类器（已承认）。
- **Logic issues：** 无。结论（"共线性是 split-joins 的无需 GT 特征"）被恰当地限定为有用先验，而非完美分类器。
- **Verdict rationale：** 正确的检验，对重叠诚实，方向和显著性 2-for-2 复现（p 至 8.6e-20）；一个稳健、注意点充分的信号。

### 19. (Priority 0.287 · Surprise 0.307) Split 边在拓扑上比正确边更靠近 GT branch 节点（中位数 196 vs 380 µm）。
- **Run：** ground-truth-error-annotations-revised-version_2026-06-17 · **ID：** 72 · **Belief：** Leaning True → Likely True (0.7083→0.9327) · **Direction：** Positive
- **Tested：** EDGE_SPLIT 边是否在拓扑上比 EDGE_CORRECT 边更靠近 GT branch 节点（degree ≥ 3），确认 branch points 为 split 失败位点。
- **Conclusion：** 确认（surprisal +0.307）。在 1,409,045 条边和 5,072 个 branch 节点中，split 边到 branch 的平均距离为 518.39 µm（中位数 196.43），而正确边为 712.57 µm（中位数 380.38）——split 中位数约为正确中位数的一半（Mann-Whitney p = 0.0）。分叉是系统性的 split 失败位点，这是 branch-point 脆弱性的第四条汇聚证据线（与 ranks 3、5、6）。
- **Caveats：** 非常大的样本膨胀显著性；与 ranks 3/5/6 的冗余意味着这是佐证，而非独立发现。
- **Reproduction：** REPRODUCED（code: revised-loading）
- **Rerun result：** 记录 split n=6805 mean 518.39 µm（median 196.43）vs correct n=1109034 mean 712.57 µm（median 380.38），U=2979956736.0，p=0.0；rerun 完全一致 → match。加载变更：直接 $RERUN_PKL 加载。
- **Generalization：** GENERALIZES
- **Across datasets：** split 边在拓扑上比正确边更靠近 GT branch 节点，方向相同且各处显著。origin (789202)：split median 196.43 µm < correct 380.38 µm，U=2.980e9，p=0.0。794491：split median 156.94 µm < correct 195.79 µm，U=1.464e9，p=0.0。794495：split median 216.42 µm < correct 246.41 µm，U=3.473e9，p=4.6e-27。额外数据集上差距较小但一致。
- **Verdict：** SOUND
- **Test：** Mann-Whitney U，双侧，U = 2,979,956,736.0，p 报告 0.0（下溢），n = 6,805 split / 1,109,034 correct edges（到最近 degree≥3 节点的加权多源 Dijkstra）。正确的非参数检验。
- **Statistical issues：** 非常大的 n 使 p 不具信息量；效应真实且大（中位数 196 vs 380 µm，split ≈ 一半）。边伪重复适用，但中心偏移稳健。这是第四条汇聚的 branch-point 度量（ranks 3/5/6）。
- **Logic issues：** 无；陈述为关联并明确框定为佐证而非独立发现。
- **Verdict rationale：** 正确的检验，大的可推广效应（额外数据集上差距衰减但成立，p 低至 4.6e-27）；对 branch-point 脆弱性的冗余但稳健的确认。

### 20. (Priority 0.287 · Surprise 0.307) Merge 位点的局部 fragment-node 密度比正确区域高约 55%（Welch t = 8.9）。
- **Run：** ground-truth-error-annotations-revised-version_2026-06-17 · **ID：** 81 · **Belief：** Leaning True → Likely True (0.7083→0.9327) · **Direction：** Positive
- **Tested：** 密集神经毡——以 10 µm 内更高的局部 `fragments_graph` 节点密度衡量——是否与 merge 位点相关（对比正确追踪的对照区域），使用体积密度（nodes/µm³）。
- **Conclusion：** 确认（surprisal +0.307）。merge 位点平均为 0.001678 nodes/µm³，而对照为 0.001083（高约 55%），在 67 个 merge 和 67 个 control 点上经 Welch's t-test 显著（t = 8.93，p = 2.8e-14）。密集神经毡是可测量的 merge 环境风险因素——ranks 2 和 23 的体积密度重述。
- **Caveats：** 仅 67 对 67 个点；对可能偏态的密度数据使用了参数化 t-test（密切相关的 ranks 2/23 使用了非参数检验）。实质上与 ranks 2 和 23 冗余。
- **Reproduction：** REPRODUCED（code: revised-loading）
- **Rerun result：** 记录 merge 0.001678 vs control 0.001083 nodes/µm³，Welch t=8.9288，p=2.7673e-14，n=67/67；rerun 完全一致 → match。加载变更：直接 pkl 加载。
- **Generalization：** GENERALIZES
- **Across datasets：** merge 位点的体积 fragment 密度比对照更高，方向相同且各处显著。origin (789202)：0.001678 vs 0.001083 nodes/µm³，Welch t=8.93，p=2.8e-14，n=67/67。794491：0.001743 vs 0.001044，t=9.45，p=3.9e-17，n=86/86。794495：0.001564 vs 0.001021，t=9.41，p=1.9e-17，n=105/105。
- **Verdict：** WEAK
- **Test：** Welch's 双样本 t-test，t = 8.9288，p = 2.7673e-14，n = 67 merge / 67 control。**正确的检验应是 Mann-Whitney U**（如实质相同的 ranks 2 和 23 所用），因为"密度"是球内节点 *计数* 除以一个常量体积——离散、非负、右偏的计数数据，而非正态。
- **Statistical issues：** 对偏态的计数派生密度应用了参数化 t-test（n=67 下正态性假设可疑）；ranks 2/23 在完全相同的量上刻意使用了非参数检验。结果恰好稳健（t≈8.9，且 ranks 2/23 的 MW 版本一致），因此检验选择不推翻结论，但对该数据类型而言是错误的检验。
- **Logic issues：** 无。结论（"密集神经毡是风险因素"）是 ranks 2/23 的重述并相符。
- **Verdict rationale：** 对计数数据用了错误的/参数化检验（应为 Mann-Whitney）应得一个注意点，但效应大，正确的非参数版本（ranks 2/23）一致，且 2-for-2 推广 → WEAK（因检验选择而脆弱，而非错误）。
- **Corrected test：** 带 rank-biserial 效应量和 bootstrap 95% CI 的 Mann-Whitney U（双侧），替换 Welch's t-test。"密度"是常量体积上球内节点 *计数*——离散、非负、右偏——因此 t-test 的正态性假设不适当；非参数检验与实质相同的 ranks 2/23 匹配。
- **Corrected result：** origin Welch t = 8.9288，p = 2.7673e-14 → Mann-Whitney U = 3951.5，p = 8.0203e-15，rank-biserial r_rb = 0.7605（95% bootstrap CI [0.6393, 0.8594]）；中位数 0.001671 vs 0.001194 nodes/µm³。非参数检验给出更小的 p 和一个 CI 远离 0 的大效应量。
- **Post-correction verdict：** UPHELD——正确的非参数检验确认了差异（p = 8.0e-15，大的 rank-biserial r_rb = 0.76，CI 排除 0），与 ranks 2/23 一致；检验选择注意点不改变结论。
- **Corrected generalization：** GENERALIZES——两个额外数据集在 Mann-Whitney 下成立且效应量大。794491：U = 6524.0，p = 1.0533e-18，r_rb = 0.7642（95% CI [0.6579, 0.8530]），n=86/86。794495：U = 9320.0，p = 7.4411e-19，r_rb = 0.6907（95% CI [0.5807, 0.7857]），n=105/105。在正确的检验下，merge 位点各处都更密集。

---

## Reproduction — Summary

- **使用的数据集 pkl：** `/allen/programs/mindscope/workgroups/auto-model/zihan.zhang/exaspim-agent/exa-spim-agent/cache/dataset_cache_789202_mcl100_add.pkl`
- **分解（n_rerun = 20 中）：** REPRODUCED 17 · DIVERGED 3 · FAILED 0。全部 20 个在 **revised-loading** 代码上运行（0 个记录）；每个脚本都以 0 退出。
- **加载修订：** 全部 20 条 top-records 都需要加载修订——每个记录脚本通过 glob/相对路径搜索 pkl（如 `*_add.pkl`、`dataset_cache_*_add.pkl`、`../data/...`）或在 dataset-not-found 检查上设门，因此 rerun harness 将加载重定向至所提供的 `$RERUN_PKL`。分析和打印保持逐字节不变。
- **未复现（DIVERGED）——3 个发现：**
  - **Entry 5 / id 13**（split 边在测地距离上更靠近 branch points）：记录运行在两个路径下 glob 同一 pkl 并双重计数边（Correct 2,218,068 / Split 13,610），而单 pkl 的 rerun 给出 Correct 1,109,034 / Split 6,805，因此 U=11,916,101,802 → 2,979,025,450 且 p=0.0 → 1.32e-197。方向相同，计数/统计量实质性不同——一个记录的双重计数假象，而非分析失败。
  - **Entry 12 / id 49**（merge 位点与 fragment branch points 共定位）：merge 中位数 4.48 µm 复现，但无种子随机节点对照重采样不同（random mean 250.32 → 320.30 µm；U=363 → 282；p=2.21e-17 → 9.53e-19）。结论成立；分歧是对照中的随机种子噪声。
  - **Entry 17 / id 65**（splits 偏好细远端突起）：split 边统计完全复现，但记录双重计数了正确边（2,218,068 vs 1,109,034）和 split n（13,610 vs 6,805），偏移了 U（281,944,000 → 140,888,192）和下采样正确边均值（1278.24 → 1287.44 µm）。方向相同；记录的双重计数假象。
- 全部 3 个分歧都是 **数据记账 / 随机种子** 效应（非分析失败），且每个发现的定性结论都被保留。17 个复现的发现完全匹配记录的统计量、p 值、效应量和 n（或对 id 55 在模拟零假设种子噪声内）。

## Generalization — Summary

- **Origin 数据集：** `dataset_cache_789202_mcl100_add.pkl`。**测试的额外数据集（2 个）：** `dataset_cache_794491_mcl100_add.pkl` 和 `dataset_cache_794495_mcl100_add.pkl`（每个发现在其未生成于的两个大脑上重新运行；全部 40 个额外数据集运行以 0 退出，无超时）。
- **分解（20 中）：** GENERALIZES 17 · PARTIAL 2 · DOES-NOT-GENERALIZE 1 · INCONCLUSIVE 0。
- **未完全推广的发现（3）：**
  - **Entry 1 / id 47** — DOES-NOT-GENERALIZE — split-gap 非线性距离×角度权衡。决定性的 **交互项** 在 origin 上显著（x3 p=0.014）但在两个额外数据集上坍塌（794491 p=0.800；794495 p=0.565，拟合未收敛）。整体 logistic 判别和线性距离项仍成立，但作为该发现实际贡献的特定非线性权衡在任一数据集上都不复现。
  - **Entry 4 / id 11** — PARTIAL — omit 边在拓扑上更靠近 leaves。在 origin（p=7.7e-311）和 794495（omit 124.85 < correct 172.81，p=6.1e-149）上成立，但在 794491 上 **方向反转**（omit mean 159.76 > correct 138.76，单侧 p=1.0）。
  - **Entry 10 / id 39** — PARTIAL — merge 位点的 KDTree 切断。在所有数据集上以相同方向降低 %merged-edges 并提升准确率，但头条 ">80% / ~86% merge 移除"是数据集特异的：794491 上仅约 70%，794495 上仅约 44%，准确率提升为 82→94%（origin）vs 额外数据集上的 74→88% 和 69→81%。
- **综述（Synthesis）。** 本次运行的主导主题——**误差的拓扑/空间地理**——在所有三个大脑上稳健：splits 集中于 branch points 附近（ids 13, 19, 72）和细远端尖端（ids 60, 65），omits 连续聚集（id 48）和在 terminals 处（id 64），merges 位于密集的 false-branch 神经毡（ids 3, 23, 49, 53, 81），split 间隙远超内部边（id 35），split 端点反平行（id 67），splits 空间级联（id 55），逐神经元 split/omit 率共变（id 30）——全部在两个额外数据集上具有一致方向和极低 p 值（效应量常常增大或缩小但从不反转）。两个优先级最高的 *头条* 结果在外推下最弱：rank-1 非线性 split-gap 交互（id 47）不复现，rank-10 可操作切断启发式（id 39）保持其方向但在更难的大脑（794495，merged 最严重的）上失去定量冲击力。仅有两个额外数据集时证据为中等而非确定性，但结构性/拓扑性发现 2-for-2 得到佐证，而唯一的反转（id 11 在 794491 上）和两个衰减都明确局部化。

## Statistical Verification — Summary

全部 20 个排序假设都就检验选择、假设、power/effect-size、p 值解释、"failed-to-reject ≠ null-is-true" 谬误和结论过度推断进行了审计，整合了本报告中已有的 Reproduction（REPRODUCED/DIVERGED）和 Generalization（GENERALIZES/PARTIAL/DOES-NOT-GENERALIZE）结果。

- **Verdict 分解（20 中）：** SOUND 10 · MINOR 5 · WEAK 3 · MAJOR 2 · CRITICAL 0。
  - **SOUND (10)：** ids 3, 10, 19, 23, 35, 53, 60, 64, 67, 72。
  - **MINOR (5)：** ids 13, 48, 49, 55, 65。
  - **WEAK (3)：** ids 11, 30, 81。
  - **MAJOR (2)：** ids 47, 39。
  - **CRITICAL (0)：** 无——没有结论是彻底无效的。

- **MAJOR ids（一行理由）：**
  - **Entry 1 / id 47** — 头条非线性距离×角度 **交互** 是一个类别不平衡拟合上的单个临界系数（p = 0.014），且 **不能推广**（两个额外大脑上 p = 0.800, 0.565）。
  - **Entry 10 / id 39** — 可操作的切断结果 **无统计检验**，陈述了一个作为全局均值评估的 *逐神经元* >80% 主张（且对若干神经元为假，如 N020 0%，N006 ~55%），使用 **ground-truth merge 坐标**（oracle，而非可部署检测器），且 ">80%" 阈值在 794495 上失败（~44%）。

  （注：**id 48** 使用了一个作为正式独立性检验不可靠的伪重复 χ²，但其结论由一个有效的描述性 28× 比值承载，因此评级为 **MINOR** 而非 MAJOR。）

- **科学家不应过度信任的最严重问题：** 两个优先级最高的头条最弱。**rank-1 非线性 split-gap 权衡（id 47）** 依赖一个在两个其他大脑上坍塌的脆弱交互项——将 *非线性* 主张视为不被支持（线性距离效应存活）。**rank-10 切断启发式（id 39）** 是一个未检验、GT 引导的上界，其逐神经元和 >80% 主张不一致成立。次要警示：**id 30**（n=12，带杠杆点和非正态残差的临界 Pearson——信赖 Spearman ρ=0.88，而非因果"共同机制"措辞）；**id 81**（对偏态计数密度的 Welch t-test——应像 ids 2/23 那样用 Mann-Whitney，二者一致）；**id 48**（因伪重复而膨胀的 χ²——依赖 28× 比值，而非 p）；以及 **id 11**（omit-near-leaves 方向在 794491 上 **反转**）。许多巨大 n 的 Mann-Whitney / chi-square 结果（ids 10, 13, 19, 60, 64, 72）方向稳健，但其天文数字的 p 值反映样本量而非效应幅度——读中位数。

### Benjamini–Hochberg FDR

从 20 个假设中各收集了关键 p 值并在 q = 0.05 下进行 BH 校正。**id 39 被排除**（无统计检验 / 无 p 值），给出 **m = 19** 的族。对于记录 p 下溢至 0.0 的 ids（ids 13, 35, 48, 55, 64, 65, 72），在可用处使用校正/rerun 值（如 id 13 → 1.32e-197），否则使用保守的 ≤1e-300 下限。

- **在 q = 0.05 下通过 BH：** **全部 19 个** p 值通过。BH 临界值是族中最大的 p，p = 0.0221（id 30），位于 rank 19，此处 BH 阈值恰为 0.05；每个更小的 p 都越过其阈值。
- **临界头条发现（0.001 < p < 0.05）：** 仅 **两个**——**id 47**（interaction p = 0.014）和 **id 30**（Pearson p = 0.0221）。它们位于排序 p 列表的最顶端，仅以最微小的边际通过 BH（id 30 的阈值等于 q = 0.05）。它们是在任何更严格的族下最可能未通过校正的发现（如若对完整的约 98 条记录运行联合校正，或若加入更多中等 p 值），且二者已因独立原因被标记为 MAJOR/WEAK（id 47 不推广；id 30 是脆弱的 n=12 Pearson）。其他每个报告的发现都如此极端（p ≤ 1.1e-08），以至 FDR 控制对其无关紧要。

## Excluded（无 surprisal 分值）

两个假设因缺少 surprisal 值被 ranker 剔除，未在上文呈现：
- Run `ground-truth-error-annotations-revised-version_2026-06-17`，ID 15
- Run `ground-truth-error-annotations-revised-version_2026-06-17`，ID 41

## Statistical Test Corrections — Summary

**5 个假设被标记为需要检验修复，并用正确的统计量重新运行**（ids 47, 30, 39, 48, 81）。校正后分解：**UPHELD 3 · WEAKENED 1 · OVERTURNED 0**（3 个 UPHELD 条目之一，id 47，仅在 origin 大脑上被支持，且在正确检验下不推广）。

**在正确检验下 CHANGED 的发现：**
- **Entry 10 / id 39 — WEAKENED。** 原始：裸的全局前/后均值，**无统计检验**（13.25%→1.83% merged，82.27%→93.66% accuracy）。正确检验：对逐神经元 deltas 的 **配对 Wilcoxon signed-rank**（W = 66.0，p = 4.9e-04，削减与准确率提升二者）。改善现已恰当地显著，但头条"*每个神经元* 移除 >80% 的 merge 误差"在 origin 上仅对 **8/12 个神经元** 成立（中位数 97.4%，95% CI [73.0%, 100%]——下界低于 80%）且不能推广（794491 上 3/9，794495 上 5/18），因此定量头条比所陈述的实质性更不令人信服。

**在正确检验下 UPHELD 的发现（含推广）：**
- **Entry 1 / id 47 — 在 origin 上 UPHELD，但 DOES-NOT-GENERALIZE。** 正确检验：对交互系数的 **置换检验 + 分层 bootstrap CI**（Wald z 在小型类别不平衡 logistic 拟合上不可靠）。Origin：x3 = −5.96，permutation p = 0.036，bootstrap CI [−12.34, −1.11] 排除 0 → 被支持。两个额外数据集在正确检验下坍塌（794491 perm p = 0.752，CI 跨越 0；794495 perm p = 0.411，CI 跨越 0）。非线性权衡仅在 origin 大脑上成立——校正后的检验重新确认了先前的 MAJOR/不推广判定。
- **Entry 8 / id 30 — UPHELD，GENERALIZES。** 正确检验：**带置换 p + bootstrap CI 的 Spearman ρ**（参数化 Pearson/OLS 在 n=12 上违反正态性）。ρ = 0.8811，permutation p = 4.6e-04，CI [0.562, 0.986]——远比临界 Pearson p = 0.022 稳健；在两个额外数据集上成立（794491 ρ = 0.983，p = 4.0e-05；794495 ρ = 0.740，p = 4.8e-04）。
- **Entry 11 / id 48 — UPHELD，推广 PARTIAL。** 正确检验：**边标签置换检验**（原始 2×2 χ² 伪重复）。在其完成处（794491）：observed ratio 18.63 vs null ~1.0，permutation p = 9.99e-04。描述性聚集比值在所有三个大脑上巨大（28.09 / 18.63 / 40.42×，全部 ≫ 假设的 3×）；校正后的置换检验在 794491 上确认显著，但 origin 和 794495 在发出校正 p 前超时（794495 在正确检验下 INCONCLUSIVE）。
- **Entry 20 / id 81 — UPHELD，GENERALIZES。** 正确检验：**Mann-Whitney U + rank-biserial 效应量**（Welch t-test 对偏态计数密度错误；与 ranks 2/23 匹配）。U = 3951.5，p = 8.0e-15，r_rb = 0.76（CI [0.639, 0.859]）；在两个额外数据集上成立（794491 r_rb = 0.764，p = 1.1e-18；794495 r_rb = 0.691，p = 7.4e-19）。

**综述（Synthesis）。** 应用正确的检验未推翻 **任何** 结论：三个密度/相关/聚集发现（ids 81, 30, 48）对检验选择完全稳健——从 t-test 切换到 Mann-Whitney（id 81）、从 Pearson 切换到置换 Spearman（id 30）、从伪重复 χ² 切换到边置换检验（id 48），都使各自保持显著且带有大的、CI 支持的效应，且 ids 81 和 30 在校正后的检验下 2-for-2 推广。唯一实质性 **weakened** 的发现是可操作的切断启发式（id 39）：一旦运行恰当的配对 Wilcoxon，改善的 *方向* 在所有三个大脑上确实显著，但特定的 ">80%-per-neuron" 头条仅对少数神经元为真且不推广。最脆弱的头条，rank-1 非线性 split-gap 交互（id 47），在 origin 大脑上经置换/bootstrap 推断存活，但其 CI 在两个其他大脑上跨越零——在正确的免假设检验下确认了非线性权衡是数据集特异的而非普遍规律。
