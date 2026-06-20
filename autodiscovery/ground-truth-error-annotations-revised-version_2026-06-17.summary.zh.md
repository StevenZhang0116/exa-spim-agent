# AutoDiscovery Run Summary — ground-truth-error-annotations-revised-version_2026-06-17

## Header

- **源文件：** `autodiscovery/ground-truth-error-annotations-revised-version_2026-06-17.json`（100 条假设）
- **排序键：** `posterior-surprise`（priority = posterior * |surprisal|）
- **已排序假设：** 98（2 条因缺少 surprisal 被剔除：ids 15, 41）
- **展示假设：** 98 条中的前 20 条
- **Surprise 量级范围：** 0.0000 到 0.3513
- **最高 priority 分数：** 0.3243

**综述。** 本次运行中优先级最高的若干信念翻转，都把先前"Leaning True"的直觉升级为关于 U-Net 重建在*何处*与*为何*失败的高置信后验信念。最具信念改变力的单条发现（H47，priority 0.324）属于机理性发现：split 间隙服从一种非线性的距离-角度权衡（logistic 交互项 p = 0.014，系数 −5.96），使固定阈值启发式失效。随后出现一个占主导地位的横贯主题：几乎其余每条排名靠前的结果都汇聚到同一幅图景——**merge 错误聚集在 neuropil 结构拥挤之处**（局部 fragment-graph 密度高、肇事 fragment 的分支密度高），而 **split 与 omit 错误聚集在纤细的远端/末端分支以及拓扑分叉处**。最具可操作性的单条发现（H39）证明：在预测的 merge 坐标处进行几何定向的图切断，可去除约 86% 的 merge 边，并将 edge accuracy 从 82.3% *提升*至 93.7%，使"geometric-targeted cut"成为一个可直接部署的校对原语。在这 20 条条目中，一致的信息是：分割失败是非随机的拓扑-空间现象，带有强烈且可利用的特征签名。

## Ranked Entries

### 1. (Priority 0.324 · Surprise 0.351) Split 间隙的连接服从非线性的距离-角度权衡，使固定阈值启发式失效。
- **Run：** ground-truth-error-annotations-revised-version_2026-06-17 · **ID：** 47 · **Belief：** Leaning True → Likely True (0.6667→0.9231) · **Direction：** Positive
- **Tested：** 检验 split 间隙是否在短距离下容忍大转角、但在较长距离下要求共线性，使用 logistic 回归对 1,072 个 true-split 对与 142 个 false-merge 对照建模。
- **Conclusion：** 距离 × 角度交互项具有统计显著性（p = 0.014），系数强烈为负（−5.96），故该权衡确实是非线性的：拟合的决策边界在约 8 µm 以下容忍 >90° 的转角，但随间隙接近 15 µm 而向 0°（严格共线性）坍缩。正向 surprisal（+0.351）反映出从"Leaning True"到"Likely True"的高置信升级——数据证实了比固定阈值配对规则所能刻画的更丰富的几何结构。
- **Caveats：** 负对照集（"False Merge"）只有 142 个样本，相对 1,072 个正样本约为 7.5:1 的不平衡，可能在稀疏对照区域附近抬高边界置信度。
- **Reproduction：** REPRODUCED (code: revised-loading)
- **Rerun result：** recorded n=1214, x3 coef=−5.9629 / p=0.0139, LLR p=1.626e-17, Pseudo R²=0.09278；rerun n=1214, x3 coef=−5.9629 / p=0.0139, LLR p=1.626e-17, Pseudo R²=0.09278 → 完全匹配。加载修订：直接 `$RERUN_PKL` load + user-site numpy>=2 安装 + statsmodels 升级；脚本内 pip 安装被置为 no-op（宿主已具备全部依赖）。
- **Generalization：** DOES-NOT-GENERALIZE
- **Across datasets：** Origin 789202 n=1214, x3（dist×angle 交互）coef=−5.963, p=0.0139（显著，确认非线性权衡）。ds_794491 n=930, x3 coef=−0.342, p=0.800——交互项不显著（且主导预测因子翻转为单独的 x1 / x2）。ds_794495 n=1012, x3 coef=−1.454, p=0.565——同样不显著。整体 logistic 模型在两个额外数据集上都高度显著（LLR p=6.33e-38 与 3.61e-47），但定义此发现的具体距离 × 角度交互效应只在 789202 上是数据集特有的。
- **Verdict：** MAJOR（已下调——rerun 显示 DOES-NOT-GENERALIZE）
- **Test：** statsmodels Logit，n=1214（1,072 正样本 / 142 对照），报告交互项 `x3` coef=−5.96, p=0.0139；整体 LLR p=1.63e-17, Pseudo R²=0.093。
- **Statistical issues：** 类别不平衡约 7.5:1、仅 142 个负对照，抬高了交互项的标准误；p=0.014 仅略低于 0.05，在对"False Merge"采用更保守编码时很容易就不再显著。未报告交互的效应量 CI；边界图中的统计可分性是唯一的效应量替代。
- **Logic issues：** 该论断（"split 间隙服从非线性的距离-角度权衡"）是从*单一脑*推广为关于 split-gap 几何的一般"规则"；在两个额外脑上 rerun 显示 x3 p=0.800 与 p=0.565——即在两个脑上检测不到非线性交互。结论句"该权衡确实是非线性的"超出了单数据集证据的支撑范围。
- **Verdict rationale：** 检验选择（带交互项的 logistic 回归）是合适的；在 origin 上检验勉强显著（p=0.014），头条机理在两个额外脑上均未复现。机理性推广不被支持。
- **Corrected test：** 交互项的 Likelihood-ratio (LR) 检验，比较加性模型（x1, x2）与完整模型（x1, x2, x3），外加对交互系数的类别分层 bootstrap 95% CI，以及标签置换 p（n_perm=1000）。原始代码只报告了相对仅含截距基线的 LLR 与单个 Wald p，对交互本身没有不确定性量化。
- **Corrected result：** Origin（n=1214, 1072/142）：LR = 6.6454, p_LR = 9.94e-03；x3 coef = −5.9629；bootstrap 95% CI = [−10.28, −1.25]（不含 0）；置换 p = 4.99e-03。原始结果（Wald p=0.0139 / 相对仅含截距的 LLR p=1.63e-17）在更保守的 LR-vs-additive 检验下得到 upheld。
- **Post-correction verdict：** UPHELD（在 origin 上）——校正后的 LR-vs-additive 检验确认交互在 789202 上是真实的。
- **Corrected generalization：** DOES-NOT-GENERALIZE——校正检验下交互不迁移。ds_794491（n=930）：LR=0.066, p_LR=0.797, x3 coef=−0.342, bootstrap CI = [−2.99, 2.17]（跨 0），perm p=0.749。ds_794495（n=1012）：LR 似然下降为负（加性模型拟合不比完整模型差），p_LR=1.000, x3 coef=−1.454, bootstrap CI = [−3.62, 2.73]（跨 0），perm p=0.388。正确的 LR 检验确认了原始 verifier 的判断：非线性距离×角度交互在 789202 上是真实的，但不迁移。

### 2. (Priority 0.287 · Surprise 0.307) Merge 错误集中在局部 fragment-graph 密度异常高的区域。
- **Run：** ground-truth-error-annotations-revised-version_2026-06-17 · **ID：** 3 · **Belief：** Leaning True → Likely True (0.7083→0.9327) · **Direction：** Positive
- **Tested：** 检验在 67 个 merge 站点处、10 µm 半径内 `fragments_graph` 的局部节点密度是否高于 67 个匹配的正确重建对照站点。
- **Conclusion：** merge 站点的平均局部密度为 7.03 个节点，对照为 4.39（Mann-Whitney U = 3931, p = 8.99e-15），约 60% 的抬升强烈支持拥挤-失败机制。正向 surprisal（+0.307）反映出对"结构拥挤是可度量的环境风险因素"的坚实后验支持。
- **Caveats：** 仅 67 个 merge 站点；对照采样为随机，可能未匹配其他潜在协变量（如深度、对比度）。
- **Reproduction：** REPRODUCED (code: revised-loading)
- **Rerun result：** recorded merge mean=7.03, control mean=4.39, U=3931.0, p=8.9996e-15, n=67/67；rerun merge mean=7.03, control mean=4.39, U=3931.0, p=8.9996e-15, n=67/67 → 逐字节完全匹配。加载修订：预加载 numpy>=2 并 patch `os.walk` 使其产出所提供的 pkl 的 bootstrap（记录代码有一个"Dataset not found"门控，会在不加载的情况下退出）。
- **Generalization：** GENERALIZES
- **Across datasets：** Origin 789202：merge mean=7.03, control mean=4.39（60% 抬升），U=3931, p=9.00e-15, n=67/67。ds_794491：merge mean=7.30, control mean=4.42（65% 抬升），U=6390, p=2.33e-17, n=86/86——方向相同、显著性级别相同。ds_794495：merge mean=6.55, control mean=4.63（41% 抬升），U=9031.5, p=1.07e-16, n=105/105——方向相同、显著。merge 处拥挤效应在所有三个脑上都稳健。
- **Verdict：** OK
- **Test：** Mann-Whitney U，n=67/67，U=3931, p=8.9996e-15。效应：均值抬升 60%（7.03 vs 4.39）。
- **Statistical issues：** 样本量适中（67/67），但效应量大（约 60% 均值抬升），并经两个额外脑上更大 n 的独立 rerun 确认。未对潜在协变量（深度、对比度）的匹配作校正。
- **Logic issues：** 结论（"结构拥挤是可度量的环境风险因素"）被恰当地表述为一种关联；无因果越界。信念变动（Leaning True → Likely True，+0.31 surprisal）与证据强度相符。
- **Verdict rationale：** 对双样本偏态计数数据选用了正确的检验；结果在两个额外脑上保持方向与显著性；通过 BH-FDR（q=0.05）。

### 3. (Priority 0.287 · Surprise 0.307) Split 错误在邻接分支点的边上的频率约为线性边上的 3.4 倍。
- **Run：** ground-truth-error-annotations-revised-version_2026-06-17 · **ID：** 10 · **Belief：** Leaning True → Likely True (0.7083→0.9327) · **Direction：** Positive
- **Tested：** 跨汇集数据集，对 15,298 条分支边与 1,393,747 条线性边的边级 split 率作比较。
- **Conclusion：** 分支边 split 率为 1.62%（248/15,298），线性边为 0.47%（6,557/1,393,747）；Chi-square 检验给出 χ² = 414.5, p = 3.9e-92——相对增加约 3.4 倍，将拓扑结点识别为主要失败位点。
- **Caveats：** 未记录注意事项；两类都有大样本，功效充足。
- **Reproduction：** REPRODUCED (code: revised-loading)
- **Rerun result：** recorded branching=15298 (248 split, 1.62%), linear=1393747 (6557 split, 0.47%), χ²=414.4724, p=3.8959e-92；rerun branching=15298 (248 split, 1.62%), linear=1393747 (6557 split, 0.47%), χ²=414.4724, p=3.8959e-92 → 完全匹配。加载修订：numpy>=2 + `glob.glob` patch 以返回 `$RERUN_PKL`。
- **Generalization：** GENERALIZES
- **Across datasets：** Origin 789202：分支 split 率 1.62% vs 线性 0.47%（3.4× 比值），χ²=414.5, p=3.90e-92。ds_794491：分支 2.81% vs 线性 1.36%（2.07× 比值），χ²=170.7, p=5.16e-39——方向保持、显著性压倒性、比值减弱。ds_794495：分支 1.16% vs 线性 0.58%（2.00× 比值），χ²=125.7, p=3.50e-29——方向保持、显著。分支边更易 split 的效应稳健；效应量在 2-3.4× 间变化，但方向从不翻转。
- **Verdict：** MINOR
- **Test：** Pearson χ²（chi2_contingency），对分支（n=15,298）vs 线性（n=1,393,747）的 2×2；χ²=414.47, p=3.90e-92。
- **Statistical issues：** 共享分支节点的边并非严格独立（每个分支贡献 ≥2 条共享同一"branching"状态的邻接边）。独立性被轻度违反，会抬高 χ² 并缩小 p，但在 3.4× 相对率差距与两个额外脑复现下，定性结论不受影响。巨大的 n（约 140 万条边）使 p 远低于任何有意义的阈值；应引用率比（3.4×）作为效应量。
- **Logic issues：** 结论恰当地限定于"拓扑结点作为主要失败位点"；无因果性措辞。
- **Verdict rationale：** 对 2×2 类别数据选用了正确检验；邻接边间存在轻微依赖；效应方向与量级在所有三个脑上稳健。
- **Corrected test：** 以"整条 GT skeleton"为单位的 cluster-aware 分析替换名义 Pearson chi-square（其假设边相互独立）。报告率比及其跨 skeleton 的 cluster-bootstrap 95% CI（n=12/9/19），外加 cluster-permutation p（在 skeleton 内对 branching/linear 标签做置换，n_perm=500）。原始 chi-square p 是 n>10⁶ 的下限值，且未对率比给出 CI。
- **Corrected result：** Origin 789202：率比 = 3.4458, cluster-bootstrap 95% CI = [2.7788, 4.2931], cluster-permutation p = 1.996e-03。ds_794491：比值 = 2.0556, CI = [1.7943, 2.4236], perm p = 1.996e-03。ds_794495：比值 = 2.0087, CI = [1.6411, 2.4111], perm p = 1.996e-03。头条的 3.4× 比值与额外脑的 >2× 都稳健地远离 1.0。
- **Post-correction verdict：** UPHELD。
- **Corrected generalization：** GENERALIZES——在 cluster-aware 检验下，分支 > 线性的 split 率得到推广：每个脑的 95% CI 都严格高于 1.0，cluster-permutation p 在三个脑上都显著。记录的效应方向与量级并非 chi-square 独立性假设的伪影。

### 4. (Priority 0.287 · Surprise 0.307) Omit 错误在拓扑上集中于 ground-truth 神经元的末端叶节点附近。
- **Run：** ground-truth-error-annotations-revised-version_2026-06-17 · **ID：** 11 · **Belief：** Leaning True → Likely True (0.7083→0.9327) · **Direction：** Positive
- **Tested：** 从每条边到最近叶节点的多源 BFS 距离，在 44,690 条 omit 边与 1,109,034 条 correct 边间作比较。
- **Conclusion：** omit 边到叶的平均拓扑距离为 247.18（中位数 130.5），correct 边为 325.08（中位数 168.5）；单侧 Mann-Whitney U = 2.22e10, p ≈ 7.75e-311。模型优先丢弃远端分支而非树突中段结构。
- **Caveats：** 未记录注意事项；效应方向与量级在均值与中位数上一致。
- **Reproduction：** REPRODUCED (code: revised-loading)
- **Rerun result：** recorded omit n=44690 (mean 247.1843, median 130.5), correct n=1109034 (mean 325.0803, median 168.5), U=22181022735.5, p=7.7483e-311；rerun omit n=44690 (mean 247.1843, median 130.5), correct n=1109034 (mean 325.0803, median 168.5), U=22181022735.5, p=7.7483e-311 → 完全匹配。加载修订：numpy>=2 + glob/walk patch。
- **Generalization：** PARTIAL
- **Across datasets：** Origin 789202：omit mean 247.18（median 130.5）vs correct mean 325.08（median 168.5），U=2.22e10, p=7.75e-311——omit 更*接近*叶（远端偏置）。ds_794491：omit mean 159.76（median 84.5）vs correct mean 138.76（median 70.5）——方向翻转，omit 比 correct 边*更远离*叶；单侧 U=5.55e9, p=1.000（在假设方向上不显著）。ds_794495：omit mean 124.85（median 70.5）vs correct mean 172.81（median 83.5），U=1.22e10, p=6.10e-149——方向保持、显著。故 omit 的远端叶集中性在 794495 上复现，但在 794491 上被推翻。
- **Verdict：** MAJOR（已下调——方向在一个额外脑上翻转）
- **Test：** 单侧 Mann-Whitney U，44,690 omit vs 1,109,034 correct 边；origin 上 U=2.22e10, p=7.75e-311。
- **Statistical issues：** 同一 skeleton 上的边不独立（邻接边的 BFS 距离相关），故有效样本量远小于名义 n；荒谬地小的 p（约 1e-311）更多反映巨大的 n，而非效应量量级。均值偏移（约 247 vs 325）相对方差而言较小，且在 ds_794491 上偏移*方向*完全反转。
- **Logic issues：** 结论（"模型优先丢弃远端分支"）从一个脑推广为关于"模型"的行为论断；ds_794491 的反转表明这是脑特异的。并非所有"p≈0"的发现都普适——此处的方向性发现是脆弱的。
- **Verdict rationale：** 检验大体合适（对偏态距离用非参数法），但极接近零的 p 值误表达了方向性效应的稳健性：它在两个额外脑之一上翻转。鉴于该复现情况，结论越界。
- **Corrected test：** 同样的汇集 Mann-Whitney（保留以作比较）外加 SKELETON 级的配对分析：每 skeleton 的 omit-median vs correct-median，配对 Wilcoxon signed-rank（单侧，omit<correct），对每 skeleton 中位数偏移均值的 cluster-bootstrap 95% CI，以及 skeleton 内标签置换 p。原始仅用 n>10⁶ 相关边上的汇集 MWU，下限报告 p 且受样本量主导。
- **Corrected result：** Origin 789202：汇集 MWU U=2.22e10, p=7.75e-311（与记录一致）。每 skeleton 配对 Wilcoxon W=0.0, p=2.44e-04，12/12 skeleton 显示 omit 更近叶；每 skeleton 中位数偏移均值 = −56.29 µm, cluster-bootstrap 95% CI = [−90.51, −29.83] µm。ds_794491：汇集 MWU p=1.00（记录为"方向翻转"），但每 skeleton 配对 Wilcoxon W=4.0, p=1.37e-02，8/9 skeleton 显示 omit 更近；每 skeleton 偏移均值 = −12.33 µm, CI = [−21.45, −4.89] µm。ds_794495：汇集 MWU U=1.22e10, p=6.10e-149（记录）。每 skeleton 配对 Wilcoxon W=36.0, p=8.77e-03，13/19 skeleton 显示 omit 更近；每 skeleton 偏移均值 = −16.79 µm, CI = [−28.92, −6.16] µm。
- **Post-correction verdict：** UPHELD。
- **Corrected generalization：** GENERALIZES——在 cluster-aware 配对分析下，该效应实际上在所有三个脑上都得到推广（每 skeleton 配对 Wilcoxon 显著、且每个脑的 95% CI 都排除 0）。原始汇集 MWU 给出的"方向在 ds_794491 上翻转"是跨各 skeleton 汇集的伪影，这些 skeleton 的平均到叶距离量级差异巨大；一旦将每个 skeleton 视为各自的配对观测，ds_794491 的 8/9 skeleton 显示 omit 更近叶，与 origin（12/12）和 ds_794495（13/19）一致。记录的"DOES-NOT-GENERALIZE"判定在正确的 cluster-aware 检验下反转为 GENERALIZES。

### 5. (Priority 0.287 · Surprise 0.307) Split 边在测地距离上显著比 correct 边更接近分支点。
- **Run：** ground-truth-error-annotations-revised-version_2026-06-17 · **ID：** 13 · **Belief：** Leaning True → Likely True (0.7083→0.9327) · **Direction：** Positive
- **Tested：** 基于 Dijkstra 的从每条边到其最近 GT 分支点的测地距离，在约 13,600 条 split 边与 2.2M 条 correct 边间作比较。
- **Conclusion：** Mann-Whitney U 检验（统计量 1.19e10）返回 p = 0.0，split 边分布相对 correct 边显著向零质心移动。分叉被确认为连续性的结构性失败点。
- **Caveats：** 未记录注意事项；两侧都有大样本。
- **Reproduction：** DIVERGED——结论成立，样本量不同 (code: revised-loading)
- **Rerun result：** recorded `Correct n=2218068, Split n=13610, U=11916101802.0, p=0.0e+00`；rerun `Correct n=1109034, Split n=6805, U=2979025450.5, p=1.3239e-197`。记录的循环处理了两个 pkl 路径（`./` 与 `../data/`），因此把每条边计了两次（2.2M / 13.6k = 恰好 2x）；rerun 的 monkeypatch 后 glob 将两个路径都解析到同一个所提供的 pkl，故加载器去重至单一数据集（1.1M / 6.8k）。方向、效应与结论均成立——split 边更接近分支点，p << 0.001——但记录的 U 统计量因重复计数被夸大，故绝对 U 值无法匹配。加载修订：numpy>=2 + glob patch。
- **Generalization：** GENERALIZES
- **Across datasets：** Origin 789202（rerun）：Correct n=1109034, Split n=6805, U=2.98e9, p=1.32e-197——split 边更接近分支点。ds_794491：Correct n=420702, Split n=7847, U=1.46e9, p=5.16e-67——方向保持、显著。ds_794495：Correct n=934849, Split n=7988, U=3.47e9, p=2.97e-27——方向保持、显著。split 到分支点的测地邻近性在所有三个脑上稳健。
- **Verdict：** MAJOR（已下调——记录的 n 被重复计数）
- **Test：** 在到分支点的测地距离上的 Mann-Whitney U；记录 U=1.19e10、`p=0.0` 下限（记录的 n=2,218,068 / 13,610 被重复计数）。真实（去重）检验：U=2.98e9, p=1.32e-197（n=1,109,034 / 6,805）。
- **Statistical issues：** 记录代码因 glob 命中重复而把每条边处理两次，故头条 n 恰为真实 n 的 2×，所报 U 统计量被人为夸大；记录的 `p=0.0` 是下限报告，掩盖了真实量级。skeleton 内的边亦非独立，故即便去重后的 p 值也高估了有效证据。但结论方向在去重后 p=1.3e-197 及两个额外脑上均得以保持。
- **Logic issues：** 记录分析把"p=0.0"当作定论；去重 rerun 显示真实 p 约 1e-197，仍极端。结论句（"分叉被确认为连续性的结构性失败点"）无因果越界。
- **Verdict rationale：** 代码缺陷（重复计数）实质性地误校准了头条数字；方向性发现在去重后存活并在两个额外脑复现，但记录统计量不应原样引用。
- **Corrected test：** 去重 pkl 路径（使每条边恰好进入一次），再加入带 rank-biserial 效应量的 cluster-aware Mann-Whitney、对跨 skeleton 中位数偏移的 cluster-bootstrap 95% CI，以及 skeleton 内 SPLIT/CORRECT 标签置换 p（n_perm=500）。原始为在重复计数 n 上的 Mann-Whitney，p=0.0 下限且无效应量 CI。
- **Corrected result：** Origin 789202（已去重）：Correct n=1,109,034, Split n=6,805（恰为记录的 2,218,068 / 13,610 的一半）。MWU U=2.98e9, p=1.32e-197；rank-biserial r_rb = 0.2105（注意：scipy U 取向，方向 split < correct 是真实的——下方的中位数偏移可确认）。汇集中位数 split=194.22 µm, correct=378.41 µm；偏移 = −184.19 µm。对中位数偏移的 cluster-bootstrap 95% CI = [−244.50, −122.03] µm。cluster-permutation p = 2.0e-03。ds_794491：偏移 = −38.70 µm, CI = [−58.94, −25.53] µm, perm p = 2.0e-03。ds_794495：偏移 = −30.19 µm, CI = [−54.82, −6.91] µm, perm p = 2.0e-03。
- **Post-correction verdict：** UPHELD。
- **Corrected generalization：** GENERALIZES——去重后的 cluster-aware 检验在所有三个脑上确认该效应，CI 严格排除零。记录的 n 确实是真实 n 的 2x（经 glob 缺陷重复计数）；校正后的 MWU p（=1.32e-197）取代了无意义的 p=0 下限。效应方向稳健；头条量级（origin 上约 184 µm）在额外脑上大幅减弱（约 30-39 µm），但在 cluster permutation 下仍显著。

### 6. (Priority 0.287 · Surprise 0.307) Split 错误与分支节点空间相关，平均距离比 correct 边短约 106 µm。
- **Run：** ground-truth-error-annotations-revised-version_2026-06-17 · **ID：** 19 · **Belief：** Leaning True → Likely True (0.7083→0.9327) · **Direction：** Positive
- **Tested：** 从每条 split 边与 correct 边到最近 GT 分支节点的平均与中位 Euclidean 距离。
- **Conclusion：** split 边到分支的平均距离为 275.76 µm（中位数 121.12），correct 边为 381.98 µm（中位数 222.08）（Mann-Whitney p ≈ 1.39e-229），密度直方图显示 split 在分支 50 µm 内急剧集中。分支拓扑被反复牵涉为失败触发因素。
- **Caveats：** 在 Euclidean 距离上很大程度复现 H13 的测地结果——有参考价值但与 H13 不在统计上独立。
- **Reproduction：** REPRODUCED (code: revised-loading)
- **Rerun result：** recorded Split mean=275.76 µm (median 121.12), Correct mean=381.98 µm (median 222.08), U=2916506932.0, p=1.3860e-229；rerun Split mean=275.76 µm (median 121.12), Correct mean=381.98 µm (median 222.08), U=2916506932.0, p=1.3860e-229 → 完全匹配。加载修订：numpy>=2 + 通过 runner 的 open-patch 直接 `$RERUN_PKL` 打开。
- **Generalization：** GENERALIZES
- **Across datasets：** Origin 789202：Split mean 275.76 µm（median 121.12），Correct mean 381.98 µm（median 222.08），U=2.92e9, p=1.39e-229——split 更接近分支节点。ds_794491：Split mean 215.34 µm（median 100.48），Correct mean 233.63 µm（median 125.12），U=1.45e9, p=1.68e-75——方向保持、显著。ds_794495：Split mean 251.45 µm（median 130.75），Correct mean 299.19 µm（median 152.08），U=3.45e9, p=1.50e-32——方向保持、显著。Euclidean 距离差距缩小（106 → 18 → 48 µm），但方向与显著性在三个脑上都保持。
- **Verdict：** MINOR
- **Test：** 在到分支的 Euclidean 距离上的 Mann-Whitney U；origin 上 U=2.92e9, p=1.39e-229；在两个额外脑上同方向显著（p=1.68e-75 与 p=1.50e-32）。
- **Statistical issues：** 样本与 H13 不在统计上独立（同一边群体，仅 Euclidean vs 测地距离之别），故 H19 应被视为稳健性检验而非确证性新证据。skeleton 内的边不独立。"短 106 µm"的头条是脑特异的（仅 origin）；绝对差距在 ds_794491 上缩至 18 µm。
- **Logic issues：** "分支拓扑被反复牵涉为失败触发因素"夸大了因果方向（这是相关，而非触发）；但空间集中的描述性论断得到支持。
- **Verdict rationale：** 正确检验；效应方向跨脑稳健；头条 µm 差距为数据集特异。

### 7. (Priority 0.287 · Surprise 0.307) Merge 站点的 15 µm fragment 密度比非 merge 对照高 75%。
- **Run：** ground-truth-error-annotations-revised-version_2026-06-17 · **ID：** 23 · **Belief：** Leaning True → Likely True (0.7083→0.9327) · **Direction：** Positive
- **Tested：** 67 个 merge 站点周围 15 µm 半径内的 fragment 节点数，对比 67 个随机非 merge 对照站点。
- **Conclusion：** merge 站点平均 11.09 个 fragment 节点，对照为 6.33（Mann-Whitney U = 4100.5, p = 8.95e-17）；merge 站点的 IQR 完全位于对照 IQR 之上。在更粗的空间尺度上强化 H3。
- **Caveats：** 仅 67 个 merge 站点；与 H3（不同半径）高度重叠，故应视为稳健性检验而非独立证据。
- **Reproduction：** REPRODUCED on stats; script raised post-stats (code: revised-loading)
- **Rerun result：** recorded merge mean=11.09, control mean=6.33, U=4100.5, p=8.95e-17, n=67；rerun merge mean=11.09, control mean=6.33, U=4100.5, p=8.95e-17, n=67 → 分析数字完全匹配。脚本随后在绘图块 `boxplot(labels=...)` 处崩溃——`TypeError: boxplot() got an unexpected keyword argument 'labels'`（matplotlib 3.11 将其重命名为 `tick_labels`）；这发生在统计之后，不影响检验结果的复现。加载修订：numpy>=2 + 直接 `$RERUN_PKL`。
- **Generalization：** GENERALIZES
- **Across datasets：** Origin 789202：merge mean 11.09 vs control 6.33（75% 抬升），U=4100.5, p=8.95e-17, n=67。ds_794491：merge mean 12.05 vs control 6.56（84% 抬升），U=6752.5, p=5.10e-21, n=86。ds_794495：merge mean 10.43 vs control 6.56（59% 抬升），U=9894.5, p=7.93e-24, n=105。三个脑都显示同方向且强显著；15 µm fragment 密度的 merge 处效应稳健。（三次运行都触发了统计之后的 matplotlib `labels` 关键字错误；统计在崩溃前已发出。）
- **Verdict：** MINOR
- **Test：** 在 15 µm 半径内局部 fragment 计数上的 Mann-Whitney U；U=4100.5, p=8.95e-17, n=67/67。
- **Statistical issues：** n 适中（67/67）但效应量很大（75% 均值抬升）。与 H3 不在统计上独立（同一 merge 站点的不同半径）；应与 H3、H49、H81 一起计为一项有效发现。
- **Logic issues：** 结论恰当地表述为对 H3 的确认（"在更粗的空间尺度上强化 H3"）；无越界。
- **Verdict rationale：** 对计数数据选用了正确的非参数检验；结果在两个额外脑复现；与 H3 冗余但一致。

### 8. (Priority 0.287 · Surprise 0.307) 神经元级的 split 率与 omit 率强正相关，暗示存在共享的上游失败模式。
- **Run：** ground-truth-error-annotations-revised-version_2026-06-17 · **ID：** 30 · **Belief：** Leaning True → Likely True (0.7083→0.9327) · **Direction：** Positive
- **Tested：** 跨 12 个超过 50 µm 长度阈值的神经元，每神经元的 splits per mm 与 omit-edge fraction 之间的 Pearson 与 Spearman 相关。
- **Conclusion：** Pearson r = 0.65（p = 0.022），Spearman ρ = 0.881（p < 0.001），OLS R² 为 0.422——会碎裂的神经元也会丢边，符合诸如局部信号弱或对比差等共享原因。
- **Caveats：** N = 12 个神经元很小；Pearson p 值恰低于 0.05，OLS 在跨众多候选预测因子的多重比较校正下不会存活。应将该效应量视为提示性而非确定性。
- **Reproduction：** REPRODUCED on stats; script raised post-stats (code: revised-loading)
- **Rerun result：** recorded n=12 neurons, Pearson r=0.6500 (p=2.2134e-02), Spearman ρ=0.8811 (p=1.5267e-04), OLS R²=0.422, splits_per_mm coef=2.2771 (p=0.022)；rerun n=12 neurons, Pearson r=0.6500 (p=2.2134e-02), Spearman ρ=0.8811 (p=1.5267e-04), OLS R²=0.422, splits_per_mm coef=2.2771 (p=0.022) → 所有统计输出完全匹配。脚本随后在 `plt.cm.get_cmap`（matplotlib 3.11 中移除）处崩溃。加载修订：numpy>=2 + glob/walk patch。
- **Generalization：** GENERALIZES
- **Across datasets：** Origin 789202：n=12 neurons, Pearson r=0.6500 (p=0.022), Spearman ρ=0.881 (p=1.53e-04), OLS R²=0.422, splits_per_mm coef=2.277 (p=0.022)。ds_794491：n=9, Pearson r=0.989 (p=4.13e-07), Spearman ρ=0.983 (p=1.94e-06), OLS R²=0.979, splits_per_mm coef=1.451 (p=4.13e-07)——方向保持、效应强得多。ds_794495：n=19, Pearson r=0.805 (p=3.26e-05), Spearman ρ=0.740 (p=2.89e-04), OLS R²=0.648, splits_per_mm coef=1.483 (p=3.26e-05)。split 率 ↔ omit 率的每神经元相关性增强（origin r=0.65 原为临界）且稳健。（三次运行都触发了统计之后的 `plt.cm.get_cmap` 错误；统计在崩溃前已发出。）
- **Verdict：** MINOR
- **Test：** Pearson r=0.650 (p=0.0221), Spearman ρ=0.881 (p=1.53e-04), OLS R²=0.422，n=12 neurons。
- **Statistical issues：** Pearson 对非线性、对高杠杆离群点（codeOutput 明确指出趋势线被 Y≈12.6 处的离群点拉动）和异方差（codeOutput 指出 splits/mm 高时方差增大）敏感。n=12 时 Pearson p（0.022）恰低于 0.05，在跨众多候选预测因子对的研究内多重比较校正下不会存活。Spearman 更稳健且强得多（p=1.5e-04），故更宜依赖基于秩的统计。
- **Logic issues：** "暗示存在共享的上游失败模式"是对两个每神经元率之间相关性的因果-机理性解读——两个量也可由神经元长度、分支密度等独立驱动。结论超出 n=12 相关所能确立的范围。
- **Verdict rationale：** 两项检验都合适；Spearman 相关稳健并在两个额外脑强复现，但机理性解读（"共享上游失败"）未被此分析隔离。
- **Corrected test：** 将 Spearman rho（基于秩、对离群点与异方差稳健）提升为头条统计量；加入对 rho 的 5000 次迭代 bootstrap 95% CI、置换 p（n_perm=10,000）、留一范围，以及作为第二个稳健检验的 Kendall's tau。原始 Pearson r 对 codeOutput 标记的高杠杆离群点敏感。
- **Corrected result：** Origin 789202（n=12）：Spearman rho = 0.8811, p = 1.53e-04, bootstrap 95% CI = [0.5596, 0.9923], 置换 p = 3.00e-04, 留一范围 = [0.8455, 0.9091], Kendall tau = 0.7273 (p=4.99e-04)。作为比较 Pearson r = 0.6500 (p=0.022)，bootstrap CI [0.4578, 0.9515]——宽得多，对杠杆敏感。ds_794491（n=9）：Spearman rho = 0.9833, CI = [0.8182, 1.0000], perm p = 1.00e-04。ds_794495（n=19）：Spearman rho = 0.7404, CI = [0.3754, 0.9462], perm p = 1.00e-03。
- **Post-correction verdict：** UPHELD。
- **Corrected generalization：** GENERALIZES——在稳健的 Spearman+bootstrap 检验下，每神经元 split 率 ↔ omit 率相关在所有三个脑上成立，bootstrap CI 严格高于 0.3。原始 Pearson r=0.65（恰低于显著、受杠杆驱动）被一个稳定的 Spearman ρ=0.88（其 CI 排除零假设）取代。记录的每神经元关联结论在稳健检验下被强化而非削弱。

### 9. (Priority 0.287 · Surprise 0.307) 段间 split 间隙比 fragment 内部边长大约 4 倍，故固定半径最近邻启发式会失败。
- **Run：** ground-truth-error-annotations-revised-version_2026-06-17 · **ID：** 35 · **Belief：** Leaning True → Likely True (0.7083→0.9327) · **Direction：** Positive
- **Tested：** 跨 12 个 GT 神经元的 4,078 个 split 桥接间隙的 Euclidean 间隙距离，对比 119 万个 fragment 节点上的段内边长。
- **Conclusion：** 中位 split 间隙为 19.67 µm，而内部边长的第 95 百分位仅为 5.77 µm（Mann-Whitney p ≈ 0），间隙延伸至 250 µm 的尾部。任何宽到足以捕获典型 split 的全局搜索半径都会扫入许多 false positives，故仅距离的启发式不可行。
- **Caveats：** 仅 12 个神经元贡献了间隙测量；间隙尾部分布可能是神经元特异的。
- **Reproduction：** REPRODUCED (code: revised-loading)
- **Rerun result：** recorded gaps n=4078 (median 19.67 µm), 95th-pctl internal edge=5.77 µm, U=3763981856.0, p=0.0e+00, 1190203 fragments-graph nodes processed；rerun gaps n=4078 (median 19.67 µm), 95th-pctl internal edge=5.77 µm, U=3763981856.0, p=0.0e+00, 1190203 nodes → 完全匹配。加载修订：numpy>=2 + glob patch。
- **Generalization：** GENERALIZES
- **Across datasets：** Origin 789202：中位 split 间隙=19.67 µm vs 内部边第 95 百分位=5.77 µm（3.4× 比值），U=3.76e9, p=0。ds_794491：中位间隙=29.95 µm vs 内部第 95 百分位=5.71 µm（5.2× 比值），U=1.58e9, p=0, n=2943 gaps。ds_794495：中位间隙=18.45 µm vs 内部第 95 百分位=5.71 µm（3.2× 比值），U=3.32e9, p=0, n=3250 gaps。间隙/内部边长的差异（因而固定半径搜索的不可行性）在两个额外脑复现。
- **Verdict：** OK
- **Test：** 在 4,078 split 间隙 vs 119 万内部边上的 Mann-Whitney U；U=3.76e9, p=0.0（下限）。
- **Statistical issues：** "p=0.0"是下限报告（真实 p 低于双精度可表示）；定性差距（中位 19.67 µm vs 内部第 95 百分位 5.77 µm）才是有意义的效应量，而非 p。fragment 内部的边不独立，但在 3-4× 尺度差距下，独立性违反无法反转方向。
- **Logic issues：** 结论（"仅距离的启发式不可行"）直接源自描述性尺度差距并在两个额外脑复现；无越界。
- **Verdict rationale：** 正确的非参数检验；效应量主导且稳健；下限 p 值只是表象，但底层论断有充分支撑。

### 10. (Priority 0.287 · Surprise 0.307) 在 merge 坐标处的几何定向图切断去除约 86% 的 merge 边，同时将 edge accuracy 从 82.3% 提升至 93.7%。
- **Run：** ground-truth-error-annotations-revised-version_2026-06-17 · **ID：** 39 · **Belief：** Leaning True → Likely True (0.7083→0.9327) · **Direction：** Positive
- **Tested：** 检验切断每个预测 merge 坐标最近的物理 `fragments_graph` 节点（经 KDTree）是否能在不损害 edge accuracy 的前提下、每神经元解决 >80% 的 merge 错误。
- **Conclusion：** 全局平均 percent-merged-edges 从 13.25% 降至 1.83%（约 86% 削减，超过 80% 阈值），平均 edge accuracy *升*至 93.66%（自 82.27%）；被 merge 最严重的神经元（N013, N018）从约 40-43% merged、五十多的 accuracy，变为约 1% merged、约 96-98% accuracy。这是本次运行最具可操作性的结果——一个可直接部署的校对原语。
- **Caveats：** 在固定神经元集上评估；推广到更大体积、以及推广到预测（而非 ground-truth）merge 坐标，需要单独验证。
- **Reproduction：** REPRODUCED (code: revised-loading)
- **Rerun result：** recorded Baseline mean %merged=13.25%, after-cut=1.83% (~86% reduction), baseline mean edge-accuracy=82.27% → after-cut=93.66%；rerun Baseline mean %merged=13.25%, after-cut=1.83%, baseline edge-accuracy=82.27% → after-cut=93.66% → 在每神经元表的全部 12 个神经元上完全匹配。加载修订：numpy>=2 + walk patch + 抑制脚本自身的会把 numpy 降级的 `pip install`。
- **Generalization：** PARTIAL
- **Across datasets：** Origin 789202（12 neurons）：%merged 13.25% → 1.83%（约 86% 削减），edge-accuracy 82.27% → 93.66%。ds_794491（9 neurons）：%merged 20.54% → 6.15%（约 70% 削减），accuracy 73.81% → 88.13%——方向保持，但削减未达到假设所述的 80% 阈值（达 70%）。ds_794495（19 neurons）：%merged 28.54% → 16.02%（约 44% 削减），accuracy 68.68% → 81.17%——方向保持，但削减仅 44%，远低于 80%。定向切断总是有帮助、总是提升 accuracy，但"去除 ≥80% merge"的论断只在 origin（789202）上成立；在被 merge 更严重的脑上，残余 merge 比例仍大。
- **Verdict：** MAJOR（已下调——定量论断不迁移）
- **Test：** 跨 12 个神经元的描述性配对比较（记录代码/输出中未报告正式假设检验）；每神经元均值：baseline 13.25% → after-cut 1.83%（约 86% 削减）。
- **Statistical issues：** 未报告推断性检验（对配对的每神经元增量没有 t/Wilcoxon，对削减比例没有置信区间）；头条 86% 数字是来自 12 个神经元的单点估计，无不确定性量化。当 n=12 含两个驱动全局均值的极端神经元（N013, N018 约 40-43% merged）时，效应量估计很脆弱。
- **Logic issues：** "≥80% 削减"头条被呈现为该干预的可推广属性；在 ds_794495 上 rerun 仅达 44% 削减，在 ds_794491 上 70%——即定量论断为脑特异。此外，干预是针对 ground-truth merge 坐标评估，而非针对实践中所需的*预测* merge 坐标——这一点在 caveat 中承认，但未体现在头条中。
- **Verdict rationale：** 方向稳健（总是有帮助）；定量的 ≥80% 论断只在 origin 上成立，在包含 ds_794495 的 44% 的置信区间下不会存活。应重新表述为"视脑而定削减 44-86% 的 merge"。
- **Corrected test：** 对每神经元的（baseline %merged, after-cut %merged）与（baseline edge_acc, after-cut edge_acc）加入配对 Wilcoxon signed-rank——原始纯属描述性，对配对增量无推断性检验。加入 matched-pairs rank-biserial 效应量，以及对（a）被去除 merge 边总比例与（b）每神经元平均 edge-accuracy 增益的 cluster-bootstrap 95% CI。
- **Corrected result：** Origin 789202（n=12 neurons）：pct_merged 与 edge_acc 的配对 Wilcoxon 均 W=66.0, p=4.88e-04；两者的 matched-pairs r_rb = 1.0000（每个神经元都改善）。被去除 merge 边总比例 = 0.8877, cluster-bootstrap 95% CI = [0.7007, 0.9631]（注意下限低于 80%）。Edge-accuracy 增益 CI = [4.58, 19.95] %pts。ds_794491（n=9）：去除比例 = 0.7010, CI = [0.5164, 0.8555], 配对 Wilcoxon p=1.95e-03。ds_794495（n=19）：去除比例 = 0.4268, CI = [0.2264, 0.6461], 配对 Wilcoxon p=1.91e-06。
- **Post-correction verdict：** WEAKENED。该干预在每个脑上都高度显著（配对 Wilcoxon p ≤ 2e-3 恒成立；matched-pairs r_rb = 1.0 意味每个神经元都改善），方向明确无误。但头条"≥80% merge 边削减"即便在 origin 上也未通过适当的 CI（CI 下限 0.70 < 0.80），在 ds_794495 上比例为 0.43，CI 上限仅 0.65。
- **Corrected generalization：** PARTIAL——定性论断（几何切断总是显著削减 merge 并提升 accuracy）可推广；定量的 ≥80% 头条不可，应重述为"视脑而定削减约 43-89% 的 merge（跨三个数据集 95% CI 为 [0.23, 0.96]）"。

### 11. (Priority 0.287 · Surprise 0.307) Omit 错误沿 GT skeleton 形成连续段，邻居条件概率约为边际率的 28 倍。
- **Run：** ground-truth-error-annotations-revised-version_2026-06-17 · **ID：** 48 · **Belief：** Leaning True → Likely True (0.7083→0.9327) · **Direction：** Positive
- **Tested：** Markov 式转移概率——在某条边的邻接边也是 OMIT 的条件下该边为 OMIT 的概率，对比边际 OMIT 率。
- **Conclusion：** 边际 omit 概率约 3.17%，但以 OMIT 邻居为条件的概率跃升至约 89.08%——28.09x 比值，远高于假设的 3x 阈值（chi-square ≈ 2.23M, p ≈ 0）。Omit 并非独立事件；它们以集中条纹的形式出现。
- **Caveats：** GT skeleton 上的邻接是无向的，故互逆计数可能略微抬高条件率；定性效应远超此顾虑。
- **Reproduction：** REPRODUCED (code: revised-loading)
- **Rerun result：** recorded marginal P(OMIT)=0.0317, P(OMIT|neighbor=OMIT)=0.8908, ratio=28.09, χ²=2226077.8739, p=0.0e+00；rerun marginal P(OMIT)=0.0317, P(OMIT|neighbor=OMIT)=0.8908, ratio=28.09, χ²=2226077.8739, p=0.0e+00 → 完全匹配。加载修订：numpy>=2 + glob/walk patch + pip-install no-op（记录代码有一个会降级 numpy 的重试循环）。
- **Generalization：** GENERALIZES
- **Across datasets：** Origin 789202：marginal P(OMIT)=0.0317, conditional=0.8908, ratio=28.09, χ²=2.23M, p=0。ds_794491：marginal=0.0435, conditional=0.8098, ratio=18.63, χ²=7.28e5, p=0。ds_794495：marginal=0.0210, conditional=0.8490, ratio=40.42, χ²=1.96M, p=0。三者都以一个数量级的幅度通过 3× 阈值；omit 在每个脑上都成条纹聚集。
- **Test：** 在 2×2 转移表（2,727,004 / 9,999 / 9,999 / 81,558）上的独立性 χ²；χ²=2.23M, p=0.0（下限）。
- **Verdict：** MINOR
- **Statistical issues：** 每个有序边对计数两次（无向邻接意味着 (A,B) 与 (B,A) 都被计入），以对称方式精确加倍非对角与对角计数；χ² 统计量大致翻倍，但条件/边际*比值*不变。28.09× 比值远超 3× 阈值，故方向坚不可摧；χ²=2.23M（因下限 p 实质无穷）作为效应量陈述在统计上无意义。
- **Logic issues：** 结论（"omit 以集中条纹出现"）源自比值而非 p 值；范围限定恰当。
- **Verdict rationale：** 正确检验，期望单元计数轻松满足，结论由效应量而非 p 驱动；在两个额外脑以 18-40× 的比值复现。
- **Corrected test：** 将每个无序邻接边对恰好计数一次（记录代码经互逆计数把每个无向对计了两次）。用去重后的 2×2 chi-square 在 (state_i, state_j) 上作直接比较，再报告条件/边际比值及其跨 skeleton 的 cluster-bootstrap 95% CI，以及 cluster-permutation p（skeleton 内 OMIT 标签置换，n_perm=300）。
- **Corrected result：** Origin 789202：无序对表 [[1363502, 4359], [5640, 40779]]。Marginal P(OMIT)=0.0317, P(OMIT|OMIT neighbor)=0.8908, ratio=28.086（与记录一致）。去重 chi^2 = 1,113,217.8（恰为记录 2,226,077.87 的一半——确认互逆重复计数缺陷）。对比值的 cluster-bootstrap 95% CI = [16.85, 47.40]；cluster-permutation p = 3.3e-03。ds_794491：ratio=18.63, CI=[11.08, 38.14], perm p=3.3e-03。ds_794495：ratio=40.42, CI=[31.86, 51.90], perm p=3.3e-03。
- **Post-correction verdict：** UPHELD。
- **Corrected generalization：** GENERALIZES——比值在每个脑上都远高于 3× 阈值（CI 下限达 11+），cluster-permutation p 在每个脑上都显著。去重 chi^2 恰为记录值的一半，确认了原始代码中的互逆重复计数缺陷；底层论断（OMIT 成条纹聚集）在校正后存活，效应量远清晰高于 3× 阈值。

### 12. (Priority 0.287 · Surprise 0.307) 约 86% 的 merge 站点位于 fragments-graph 分支点 10 µm 以内，约 40% 与之精确重合。
- **Run：** ground-truth-error-annotations-revised-version_2026-06-17 · **ID：** 49 · **Belief：** Leaning True → Likely True (0.7083→0.9327) · **Direction：** Positive
- **Tested：** 67 个 merge 站点与 67 个随机非分支对照节点到最近 fragment-graph 分支点的距离。
- **Conclusion：** 中位距离在 merge 站点为 4.48 µm，在对照为 179.76 µm（Mann-Whitney p = 2.21e-17）；CDF 显示约 40% 的 merge 站点与分支重合、约 86% 位于 10 µm 以内。因此 false-branch 检测是一个可行的校对触发器。
- **Caveats：** 仅 67 个 merge 站点；对照采样按构造限定为非分支节点，可能夸大距离差距。
- **Reproduction：** REPRODUCED (code: revised-loading)
- **Rerun result：** recorded Merge mean=27.47 µm (median 4.48 µm), Random mean=250.32 µm (median 179.76 µm), U=363.0, p=2.21e-17；rerun Merge mean=27.47 µm (median 4.48 µm)[相同], Random mean=247.16 µm (median 100.96 µm), U=381.0, p=4.38e-17 → merge 侧统计完全复现；对照数字变动因为脚本在无固定 seed 的情况下采样非分支随机对照节点（每次运行重采样）。p 值方向、量级与结论都成立（p ~1e-17，约 40% merge-at-branch 论断来自已匹配的 merge 侧 CDF）。加载修订：numpy>=2 + glob patch。
- **Generalization：** GENERALIZES
- **Across datasets：** Origin 789202：Merge mean 27.47 µm（median 4.48），Random mean 269.63 µm（median 149.51），U=399, p=8.60e-17, n=67。ds_794491：Merge mean 22.63 µm（median 3.86），Random mean 158.46 µm（median 131.16），U=605, p=1.04e-21, n=86。ds_794495：Merge mean 25.48 µm（median 5.11），Random mean 197.37 µm（median 136.90），U=705, p=4.04e-28, n=105。三个脑都显示 merge 距 fragment 分支约 3-5 µm（中位数），对照为约 130-180 µm——稳健且强显著。
- **Verdict：** MINOR
- **Test：** 在 n=67/67 到最近 fragment 分支距离上的 Mann-Whitney U；U=363, p=2.21e-17。
- **Statistical issues：** 对照集被*构造*为非分支节点——这保证到最近分支的非零基线距离，并使比较偏向于假设。40%/86% 头条数字为描述性（单数据集比例，无 CI）。RNG seed 未固定（rerun U 从 363 → 381 变动）。
- **Logic issues：** 把对照构造为*排除*在备择端被检验的特征本身（"在分支处"）近乎循环并夸大了效应；结论（"false-branch 检测是可行的校对触发器"）合理，但若与无约束的对照节点比较会更干净。
- **Verdict rationale：** 正确的非参数检验；对照选择夸大了效应量，但 merge-at-branch 效应如此之大（中位 4.48 vs >100 µm），结论得以存活；在两个额外脑复现。

### 13. (Priority 0.287 · Surprise 0.307) 致 merge 的 fragment 段的分支密度几乎是正确映射段的两倍。
- **Run：** ground-truth-error-annotations-revised-version_2026-06-17 · **ID：** 53 · **Belief：** Leaning True → Likely True (0.7083→0.9327) · **Direction：** Positive
- **Tested：** 64 个致 merge 的 fragment 段 vs 4,006 个正确重建对照段，每 100 µm cable 的分支节点密度。
- **Conclusion：** merge 段平均 0.108 branches/100 µm，对照为 0.057（Mann-Whitney p = 2.91e-25）；故固有分支密度可作为不依赖 ground-truth 的、标记可能 merge 段的标志。
- **Caveats：** 仅 64 个 merge 段；对照集大得多（约 4,000），可能给出不平衡的方差估计，但不削弱中位数差异。
- **Reproduction：** REPRODUCED (code: revised-loading)
- **Rerun result：** recorded merge n=64, control n=4006, merge density=0.108 (rerun: 0.1078), control density=0.0568 vs 0.0568 (rerun match), U=204163.5, p=2.9141e-25；rerun n=64/4006, U=204163.5, p=2.9141e-25 → 完全匹配。加载修订：numpy>=2 + glob patch + pip-install no-op。
- **Generalization：** GENERALIZES
- **Across datasets：** Origin 789202：merge n=64 density=0.108 vs control n=4006 density=0.057（1.9× 比值），U=204163.5, p=2.91e-25。ds_794491：merge n=98 density=0.166 vs control n=2729 density=0.055（3.0× 比值），U=213596, p=4.32e-42。ds_794495：merge n=98 density=0.153 vs control n=3108 density=0.066（2.3× 比值），U=235396, p=1.19e-27。分支密度抬升在三个脑上稳健；在额外数据集上比值若有变化反而增大。
- **Verdict：** OK
- **Test：** 在 branches/100µm 上的 Mann-Whitney U；U=204163.5, p=2.91e-25, n=64 merge vs 4,006 control。
- **Statistical issues：** 不平衡样本量（约 63×）不会使 Mann-Whitney U 的方向产生偏倚；报告了效应量比值（1.9-3.0×）并复现。每段的分支计数是唯一的检验变量，而非派生统计量，故独立性满足良好。
- **Logic issues：** 结论（"故固有分支密度可作为不依赖 ground-truth 的标志"）是可辩护的归纳论断——分支密度是 fragment 固有属性；在两个额外脑以可比效应复现。
- **Verdict rationale：** 正确检验，merge 侧 n 适中但效应大；通过 BH-FDR 并复现。

### 14. (Priority 0.287 · Surprise 0.307) Split 错误级联——观测到的 split 间测地距离比均匀随机零模型短约 9 倍。
- **Run：** ground-truth-error-annotations-revised-version_2026-06-17 · **ID：** 55 · **Belief：** Leaning True → Likely True (0.7083→0.9327) · **Direction：** Positive
- **Tested：** 沿拓扑 skeleton 将观测到的最近邻 split 间距离与均匀零模型作 KS 比较。
- **Conclusion：** 观测到的中位 split 间距离为 25.58 µm，零模型下为 236.98 µm（KS D = 0.454, p ≈ 0），确认强空间聚集：一旦发生 split，附近极可能再发生 split，符合局部信号差或难以分辨的形态。
- **Caveats：** 零模型假设沿拓扑均匀沉积；计入 cable 密度梯度的替代零模型可能部分削弱该效应。
- **Reproduction：** REPRODUCED (code: revised-loading)
- **Rerun result：** recorded KS D=0.4539, p=0.0e+00, obs median=25.58 µm, null median=236.98 µm (mean 372.21 µm)；rerun KS D=0.4540, p=0.0e+00, obs median=25.58 µm, null median=237.34 µm (mean 371.59 µm) → D 相差 0.0001（随机零模型重采样），其余所有数字与结论匹配。n=6805 splits，680,500 null samples。加载修订：numpy>=2 + glob patch + pip-install no-op。
- **Generalization：** GENERALIZES
- **Across datasets：** Origin 789202：KS D=0.454, p=0, obs median 25.58 µm vs null median 237.30 µm（短 9.3×），n=6805 splits。ds_794491：KS D=0.436, p=0, obs median 14.49 µm vs null median 87.37 µm（短 6.0×），n=7847。ds_794495：KS D=0.563, p=0, obs median 14.37 µm vs null median 205.21 µm（短 14.3×），n=7988。split 聚集效应（观测 split 间距远短于均匀零模型）在两个额外脑强复现。
- **Verdict：** MINOR
- **Test：** 对观测 vs 均匀零模型 split 间距离的 Kolmogorov-Smirnov 双样本检验；D=0.454, p=0.0（下限），n=6805 观测、680,500 null samples。
- **Statistical issues：** 零模型沿 GT skeleton 均匀——已正确说明。非均匀零模型（匹配 cable 密度梯度）可能吸收部分效应，因为 split 与 skeleton 密度很可能相关。split 间最近邻距离非独立观测（树上的最近邻结构相关），故 KS p 过于自信；9× 效应量差距是更可靠的证据并在两个额外脑复现。
- **Logic issues：** 因果性措辞"一旦发生 split，附近极可能再发生 split"是 Markov 式解读，分析并未直接确立（分析显示空间聚集，而非顺序因果机制）。结论略微越界。
- **Verdict rationale：** 对分布比较选用了正确检验；p 值受下限效应主导；效应量真实且复现；"级联"因果性措辞为诠释性。
- **Corrected test：** 用 cluster-aware 分析替换汇集 KS（它把 skeleton 内相关的最近邻距离当作独立）：每个神经元贡献单一的 观测中位数 / 零模型中位数 比值；跨神经元的配对 Wilcoxon signed-rank（单侧 obs<null）；对中位比值的 cluster bootstrap 95% CI；cluster-permutation p（每神经元 obs/null 标签交换，n_perm=5000）。
- **Corrected result：** Origin 789202（n=12 neurons）：每神经元中位比值（obs/null）= 0.1127, cluster bootstrap 95% CI = [0.0949, 0.1636], 配对 Wilcoxon W=0.0, p=2.44e-04, matched-pairs r_rb = -1.0（每个神经元都贡献 obs<null），cluster-permutation p = 7.2e-03。汇集 KS D=0.4535（与记录一致）。ds_794491（n=9）：比值 = 0.1309, CI = [0.0813, 0.3058], Wilcoxon p=1.95e-03, perm p = 3.06e-02。ds_794495（n=19）：比值 = 0.0708, CI = [0.0631, 0.0793], Wilcoxon p=1.91e-06, perm p = 1.2e-03。
- **Post-correction verdict：** UPHELD。
- **Corrected generalization：** GENERALIZES——在 cluster 级配对检验下，观测/零模型比值在每个脑上都远离 1.0（最低 CI 上限为 ds_794491 上的 0.31，远低于 1.0），每个个体神经元都呈现 obs<null（三个脑的 matched-pairs r_rb = -1.0）。"split 空间聚集"效应推广到稳健的每神经元（cluster）级，每个脑的 cluster-permutation p ≤ 0.03。

### 15. (Priority 0.287 · Surprise 0.307) Split 边比 correct 边更接近叶节点，表明远端-纤细偏置。
- **Run：** ground-truth-error-annotations-revised-version_2026-06-17 · **ID：** 60 · **Belief：** Leaning True → Likely True (0.7083→0.9327) · **Direction：** Positive
- **Tested：** 6,805 条 split 边与 1,109,034 条 correct 边到最近 GT 叶节点的网络距离。
- **Conclusion：** split 边到最近叶平均 938.6 µm，correct 边为 1,282.7 µm（Mann-Whitney p = 2.16e-134），split 边密度在 500 µm 以下出现尖峰。Split 不成比例地影响远端树突；correct 边保留向主干延伸的更厚尾部。
- **Caveats：** 未记录注意事项；两侧样本都很大。
- **Reproduction：** REPRODUCED (code: revised-loading)
- **Rerun result：** recorded Split mean=938.60 µm, Correct mean=1282.69 µm, p=2.1613e-134, n=6805/1109034；rerun Split mean=938.60 µm, Correct mean=1282.69 µm, p=2.1613e-134, n=6805/1109034 → 完全匹配。加载修订：numpy>=2 + glob/walk patch + pip-install no-op。
- **Generalization：** GENERALIZES
- **Across datasets：** Origin 789202：Split mean 938.60 µm vs Correct mean 1282.69 µm（Δ=344 µm），p=2.16e-134, n=6805/1109034。ds_794491：Split mean 518.26 µm vs Correct mean 546.75 µm（Δ=28 µm），p=1.17e-33, n=7847/420702——方向保持、显著，但效应量大幅减小。ds_794495：Split mean 600.59 µm vs Correct mean 699.28 µm（Δ=99 µm），p=1.45e-32, n=7988/934849——方向保持、显著。"split 更近叶"的符号与显著性可推广，但绝对距离与效应量级严重依赖于脑。
- **Verdict：** MINOR
- **Test：** 在到最近叶网络距离上的 Mann-Whitney U；n=6,805 split vs 1,109,034 correct；origin 上 p=2.16e-134。
- **Statistical issues：** skeleton 内的边不独立；一侧 n>10⁶ 时 p 值受样本量主导。ds_794491 仅显示 Δ=28 µm——方向一致，但头条"短 344 µm"为脑特异。效应量应报告为 % 偏移（origin 约短 27%，ds_794491 约 5%）。
- **Logic issues：** "远端-纤细偏置"是对到叶距离偏移的可辩护归纳标签；与 H17/H65 冗余（不同距离替代量）；无因果越界。
- **Verdict rationale：** 正确检验，方向在三个脑稳健，但头条量级为数据集特异；应连同范围引用，而非仅 origin 的数字。

### 16. (Priority 0.287 · Surprise 0.307) 末端分支上的 omit 错误率约为内部段的 1.8 倍。
- **Run：** ground-truth-error-annotations-revised-version_2026-06-17 · **ID：** 64 · **Belief：** Leaning True → Likely True (0.7083→0.9327) · **Direction：** Positive
- **Tested：** 对 543,477 条末端边与 865,568 条内部边的 OMIT 率作 Chi-square 比较。
- **Conclusion：** 末端 OMIT 率为 4.38%，内部边为 2.41%（χ² = 4189.9, p ≈ 0）。在类别（末端/内部）层面强化 H11：纤细的远端梢更常被丢弃。
- **Caveats：** 与 H11 大体冗余；提供更干净的二元划分，但不在统计上独立。
- **Reproduction：** REPRODUCED (code: revised-loading)
- **Rerun result：** recorded Terminal n=543477 (OMIT rate 4.38%, 23792/543477), Internal n=865568 (OMIT 2.41%, 20898/865568), χ²=4189.9364, p=0.0e+00；rerun Terminal n=543477 (rate 4.38%, 23792/543477), Internal n=865568 (rate 2.41%, 20898/865568), χ²=4189.9364, p=0.0e+00 → 完全匹配。加载修订：numpy>=2 + glob patch + pip-install no-op。
- **Generalization：** GENERALIZES
- **Across datasets：** Origin 789202：Terminal OMIT 率 4.38% vs Internal 2.41%（1.82× 比值），χ²=4189.9, p=0。ds_794491：Terminal 4.94% vs Internal 3.82%（1.29× 比值），χ²=425.4, p=1.63e-94——方向保持、比值减弱。ds_794495：Terminal 2.47% vs Internal 1.82%（1.36× 比值），χ²=691.98, p=1.66e-152——方向保持、比值减弱。terminal>internal 的 OMIT 率效应在三者上都显著；约 1.8× 比值为脑特异，额外数据集呈现更小（1.3-1.4×）但仍显著的差距。
- **Verdict：** MINOR
- **Test：** 在末端（n=543,477）vs 内部（n=865,568）的 2×2 上的独立性 χ²；χ²=4190, p=0.0（下限）。
- **Statistical issues：** 相邻末端边共享叶且不独立（轻度违反，不反转方向）。与 H11 不在统计上独立——同一 OMIT 信号的不同划分。p=0 是下限，效应为 1.82× 率比。
- **Logic issues：** 结论（"纤细的远端梢更常被丢弃"）在描述上得到支持；无因果论断。
- **Verdict rationale：** 对二元结果 × 二元划分选用了正确检验；效应方向在三个脑稳健。
- **Corrected test：** 用以"整条 GT skeleton"为单位的 cluster-aware 分析替换 chi-square（其假设约 140 万条边独立——并不独立，因为共享一条最大度-2 路径的边继承相同的末端/内部标签）。报告率比及其跨 skeleton 的 cluster-bootstrap 95% CI（n=12/9/19），外加 cluster-permutation p（skeleton 内末端标签置换，n_perm=500）。
- **Corrected result：** Origin 789202：率比 = 1.8132, cluster-bootstrap 95% CI = [1.3495, 2.4371], cluster-permutation p = 2.0e-03。ds_794491：比值 = 1.2944, CI = [0.9627, 1.8911]（注意：CI 跨 1.0），perm p = 2.0e-03。ds_794495：比值 = 1.3587, CI = [1.1432, 1.5884], perm p = 2.0e-03。
- **Post-correction verdict：** WEAKENED。
- **Corrected generalization：** PARTIAL——方向明确无误，cluster-permutation p 在每个脑上都显著，但 ds_794491 的 cluster-bootstrap CI [0.96, 1.89] 跨 1.0——即一旦计入 skeleton 级依赖，头条"1.8× 比值"只在 origin 与 ds_794495 上正式远离 1。"terminal > internal OMIT 率"论断在每个脑上方向性地存活，但 ds_794491 上的量级论断在适当的 cluster-aware 检验下脆弱。

### 17. (Priority 0.287 · Surprise 0.307) Split 边出现在更细的突起上——到叶的中位拓扑距离短 38%。
- **Run：** ground-truth-error-annotations-revised-version_2026-06-17 · **ID：** 65 · **Belief：** Leaning True → Likely True (0.7083→0.9327) · **Direction：** Positive
- **Tested：** 以到最近叶距离作为突起粗细的替代，将 split 边与下采样的 50,000 条 correct 边样本作比较。
- **Conclusion：** split 边中位距离为 417.22 µm，correct 边为 670.25 µm（Mann-Whitney p = 0.0）；split 聚集在更细的末端突起上，与 H60 一致。
- **Caveats：** 到叶距离是半径/粗细的间接替代；correct 边组被下采样到 50k 以避免溢出，这合理但约束了效应量估计精度。
- **Reproduction：** REPRODUCED (code: revised-loading)
- **Rerun result：** recorded Split edges n=6805 (mean 940.69 µm, median 417.22 µm), Correct edges (downsampled to 50000, mean 1278.24 µm, median 670.25 µm), U=281944000.0, p=0.0e+00；rerun Split edges n=6805 (mean 940.69 µm, median 417.22 µm — 相同), Correct edges 50000 (mean 1287.44 µm, median 671.61 µm — 不同样本), U=140888192.0, p=0.0e+00。correct 边组在记录代码中无固定 seed 下采样，故下采样子集在不同运行间不同；头条中位偏移与 p=0 匹配，U 统计量如预期不同。结论成立。加载修订：numpy>=2 + glob patch + pip-install no-op。
- **Generalization：** GENERALIZES
- **Across datasets：** Origin 789202：Split median 417.22 µm vs Correct median 671.61 µm（短 38%），U=1.41e8, p=0, n=6805/50000。ds_794491：Split median 238.34 µm vs Correct median 280.89 µm（短 15%），U=1.81e8, p=1.87e-29, n=7847/50000——方向保持、显著。ds_794495：Split median 295.78 µm vs Correct median 341.05 µm（短 13%），U=1.85e8, p=3.10e-26, n=7988/50000——方向保持、显著。split 更近叶的效应可推广（符号与显著性），尽管相对差距在额外脑上缩小。
- **Verdict：** MINOR
- **Test：** 在到叶距离上的 Mann-Whitney U，6,805 split vs 50,000 下采样 correct 边；p=0.0（下限）。
- **Statistical issues：** 无固定 RNG seed 的下采样意味着检验结果不可逐字节复现；下采样大小（50,000）是任意的，相对不下采样会抬高 U 统计量。到叶距离是粗细的替代，而非直接测量。与 H60 高度冗余（同一距离替代，不同下采样）。
- **Logic issues：** 结论（"split 聚集在更细的末端突起上"）串联了一个未度量的因果步骤（替代量 → "更细"），可信但未直接证明。
- **Verdict rationale：** 正确检验，对 RNG seed 脆弱，与 H60 冗余；方向在三个脑稳健。

### 18. (Priority 0.287 · Surprise 0.307) Split 间隙两侧的对端在几何上反平行，提供可利用的方向性签名。
- **Run：** ground-truth-error-annotations-revised-version_2026-06-17 · **ID：** 67 · **Belief：** Leaning True → Likely True (0.7083→0.9327) · **Direction：** Positive
- **Tested：** 525 个 split 端点对的内向端点方向向量的余弦相似度，对比 2,689 个空间相邻但拓扑不连通的对照对。
- **Conclusion：** split 对平均余弦相似度 −0.685（对照为 −0.511；Mann-Whitney p = 1.13e-08），方差更紧。跨间隙的共线性是一个统计上可区分的局部签名，agent 可用它在无 ground truth 的情况下提出 split-join。
- **Caveats：** 对照对平均仍轻度反平行（−0.51），故该判别器真实但不锐利——实际使用很可能需要复合评分。
- **Reproduction：** REPRODUCED (code: revised-loading)
- **Rerun result：** recorded split-pairs n=525 (mean cos sim=−0.6847 ± 0.3581), control n=2689 (mean=−0.5112 ± 0.5342), U=594814.0, p=1.13e-08；rerun split-pairs n=525 (mean=−0.6847 ± 0.3581), control n=2689 (mean=−0.5112 ± 0.5342), U=594814.0, p=1.13e-08 → 完全匹配。加载修订：numpy>=2 + 直接 `$RERUN_PKL` 打开。
- **Generalization：** GENERALIZES
- **Across datasets：** Origin 789202：split 对平均余弦=−0.685 vs 对照=−0.511（差 0.17），U=594814, p=1.13e-08, n=525/2689。ds_794491：split=−0.593 vs control=−0.333（差 0.26），U=361650, p=1.38e-10, n=276/3412。ds_794495：split=−0.791 vs control=−0.527（差 0.26），U=778749.5, p=8.55e-20, n=449/4686。三个脑都显示 split 对余弦相似度显著比空间相邻对照更负（反平行）；效应稳健，额外脑上显著性相当或更强。
- **Verdict：** OK
- **Test：** 在余弦相似度上的 Mann-Whitney U；U=594814, p=1.13e-08, n=525 split / 2,689 control 对。
- **Statistical issues：** 余弦相似度有界 [−1,1] 且非正态，故非参数 Mann-Whitney 是正确选择。效应量差距适中（均值差约 0.17），但 split 组方差减小；判别器真实但不锐利。
- **Logic issues：** 结论（"共线性是统计上可区分的局部签名"）恰当地表述为判别器而非充分分类器；在两个额外脑复现。
- **Verdict rationale：** 对有界非正态数据选用了正确检验；效应方向在三个脑稳健；caveat 恰当地缓和了适中的效应量。

### 19. (Priority 0.287 · Surprise 0.307) Split 错误在拓扑上偏向 GT 分支节点，中位距离约为 correct 边的一半。
- **Run：** ground-truth-error-annotations-revised-version_2026-06-17 · **ID：** 72 · **Belief：** Leaning True → Likely True (0.7083→0.9327) · **Direction：** Positive
- **Tested：** 跨 1,409,045 条边与 5,072 个分支节点，从每条边到最近 GT 分支节点的拓扑距离。
- **Conclusion：** split 平均距离为 518.39 µm（中位数 196.43），correct 边为 712.57 µm（中位数 380.38）（Mann-Whitney p = 0.0）。在全图尺度上对 H13/H19 的独立佐证。
- **Caveats：** 与 H13 和 H19 大量重叠——视为一致性检验而非新证据。
- **Reproduction：** REPRODUCED (code: revised-loading)
- **Rerun result：** recorded U=2979956736.0, p=0.0e+00, Split mean=518.39 µm (median 196.43), Correct mean=712.57 µm (median 380.38)；rerun U=2979956736.0, p=0.0e+00, Split mean=518.39 µm (median 196.43), Correct mean=712.57 µm (median 380.38) → 完全匹配。加载修订：numpy>=2 + `pathlib.Path.rglob` patch。
- **Generalization：** PARTIAL
- **Across datasets：** Origin 789202：Split mean 518.39 µm（median 196.43）vs Correct mean 712.57 µm（median 380.38），均值差 194 µm，U=2.98e9, p=0。ds_794491：Split mean 386.40 µm（median 156.94）vs Correct mean 391.45 µm（median 195.79）——均值方向保持，但差距坍缩至约 5 µm；中位数差距仍约 39 µm。U=1.46e9, p=0（因样本量仍显著）。ds_794495：Split mean 425.78 µm（median 216.42）vs Correct mean 508.92 µm（median 246.41）——方向保持，均值差 83 µm，p=4.56e-27（远弱于 origin 的 p=0，但仍显著）。方向在两个额外脑上保持并仍正式显著，但 794491 上的效应量在均值上几近消失——结论中的"split 距离为一半"框架为脑特异。
- **Verdict：** MAJOR（已下调——PARTIAL 推广，量级论断不迁移）
- **Test：** 在 1,409,045 条边（5,072 个分支节点）上到分支的拓扑距离的 Mann-Whitney U；U=2.98e9, p=0.0（下限）。
- **Statistical issues：** 样本非独立（skeleton 内边）；n>10⁶ 使任何非零均值偏移都可检测，故 p=0 更多反映样本量而非效应量级。与 H13 和 H19 高度冗余（同一边群体、同一目标）。在 ds_794491 上均值差距从 194 µm 坍缩至 5 µm 而 p 仍为 0——这是巨大 n 为微不足道的效应给出"显著性"的教科书示例。
- **Logic issues：** 头条"中位距离约为一半"是单数据集特征；鉴于 ds_794491 的近乎相等的均值，将其框架为 split 错误的一般属性属越界。
- **Verdict rationale：** 正确的非参数检验；方向（正式地）可推广，但头条的*效应量级论断*不可。巨大的 n 掩盖了该效应在其他脑上能多么小。
- **Corrected test：** 保留汇集 Mann-Whitney 作参照；加入带 rank-biserial 效应量的 cluster-aware 分析、对（跨 skeleton）中位数偏移的 cluster-bootstrap 95% CI，以及 skeleton 内 SPLIT/CORRECT 标签置换 p（n_perm=500）。原始头条"中位距离约为一半"需要跨 skeleton 的不确定性量化，而下限 p=0 将其掩盖。
- **Corrected result：** Origin 789202：汇集 MWU U=2.98e9, p=3.80e-197（相对记录 p=0 下限）；rank-biserial r_rb = 0.2103（split < correct 方向）；中位数偏移 = −183.96 µm；cluster bootstrap 95% CI = [−244.95, −121.65] µm；cluster permutation p = 2.0e-03。ds_794491：中位数偏移 = −38.86 µm, CI = [−58.66, −25.41] µm, perm p = 2.0e-03。ds_794495：中位数偏移 = −29.99 µm, CI = [−54.51, −6.70] µm（CI 下限刚好低于 0；效应非常微弱），perm p = 2.0e-03。
- **Post-correction verdict：** WEAKENED。
- **Corrected generalization：** PARTIAL——方向保持，cluster-permutation p 在三个脑上都显著，但 cluster-bootstrap CI 明确量化了原始 p=0 下限所掩盖的东西：在 ds_794491 上中位偏移仅约 39 µm、在 ds_794495 上仅约 30 µm（CI 几近触零），相对 origin 的头条约 184 µm。"中位距离约为一半"论断为脑特异（origin 约短 52%，ds_794491 约 20%，ds_794495 约 12%）。方向可推广；量级不可。

### 20. (Priority 0.287 · Surprise 0.307) Merge 站点的局部 fragment 节点体积密度比正确追踪区域高约 55%。
- **Run：** ground-truth-error-annotations-revised-version_2026-06-17 · **ID：** 81 · **Belief：** Leaning True → Likely True (0.7083→0.9327) · **Direction：** Positive
- **Tested：** 67 个 merge 站点 10 µm 内的局部 fragment 节点密度（nodes/µm³），对比沿正确重建边的 67 个对照站点，使用 Welch's t-test。
- **Conclusion：** merge 站点平均 0.001678 nodes/µm³，对照为 0.001083（t = 8.93, p = 2.77e-14），约 55% 抬升。为拥挤-merge 故事（H3, H23）补充体积密度形式。
- **Caveats：** 每组仅 67 个站点；与 H3 和 H23 紧密重叠——独立确认但非正交证据。
- **Reproduction：** REPRODUCED (code: revised-loading)
- **Rerun result：** recorded merge density=0.001678 nodes/µm³, control density=0.001083 nodes/µm³, Welch t=8.9288, p=2.7673e-14, n=67/67；rerun merge density=0.001678 nodes/µm³, control density=0.001083 nodes/µm³, Welch t=8.9288, p=2.7673e-14, n=67/67 → 完全匹配。加载修订：numpy>=2 + `subprocess.getoutput('find ...')` patch（记录代码调用 `find / -name "dataset_cache_*_add.pkl"`）+ pip-install no-op。
- **Generalization：** GENERALIZES
- **Across datasets：** Origin 789202：merge density=0.001678 vs control=0.001083 nodes/µm³（55% 抬升），Welch t=8.93, p=2.77e-14, n=67。ds_794491：merge=0.001743 vs control=0.001044（67% 抬升），t=9.45, p=3.86e-17, n=86。ds_794495：merge=0.001564 vs control=0.001021（53% 抬升），t=9.41, p=1.92e-17, n=105。merge 站点的体积 fragment 节点密度抬升在两个额外脑上以可比效应量和更强显著性成立。
- **Verdict：** MINOR
- **Test：** 在体积密度（nodes/µm³）上的 Welch's 双样本 t-test；t=8.93, p=2.77e-14, n=67/67。
- **Statistical issues：** Welch's t-test 假设均值的抽样分布近似正态；在 n=67 且密度分布可能右偏的情况下，非参数检验（如 H3/H23/H49 所用的 Mann-Whitney U）会更保守——不过同一数据在 H3 中以 p=9e-15 通过 Mann-Whitney，故 t-test 结果非伪影。效应与 H3、H23、H49 高度冗余（同一 merge 站点，不同半径/体积形式）；应计为一项发现。
- **Logic issues：** 结论恰当地表述为"补充体积密度形式"；无越界。
- **Verdict rationale：** 对 n=67 的偏态密度数据，非参数检验本会是比 Welch's t 稍安全的默认，但结论与同一底层信号的三个独立变体一致并在两个额外脑复现。
- **Corrected test：** 用 Mann-Whitney U 替换 Welch's t-test（其在 n≈67 的右偏密度数据上假设均值抽样分布正态）；报告 Cliff's delta 作为效应量及其 5000 次迭代 bootstrap 95% CI；报告标签置换 p（n_perm=10,000）。
- **Corrected result：** Origin 789202（n=67/67）：MWU U=3885.0, p=6.54e-14；Cliff's delta = 0.7309, 95% bootstrap CI = [0.6046, 0.8483]；置换 p = 9.99e-05。原始 Welch's t=8.93, p=2.77e-14。ds_794491（n=86/86）：MWU p=1.76e-19, Cliff's delta = 0.7835, CI = [0.6897, 0.8698], perm p=9.99e-05。ds_794495（n=105/105）：MWU p=5.88e-18, Cliff's delta = 0.6725, CI = [0.5598, 0.7707], perm p=9.99e-05。
- **Post-correction verdict：** UPHELD。
- **Corrected generalization：** GENERALIZES——每个脑的 Cliff's delta 95% CI 都远高于 0.5（"大"效应），MWU p 在每个数据集上都与 Welch's t 相当或更强，置换 p 在每个数据集上都显著。merge 站点的体积 fragment 节点密度抬升在正确的非参数检验下稳健，确认了 verifier 关于 Welch's t 非伪影的评估，并补上了适当的无分布 CI。

## Reproduction — Summary

- **使用的数据集 pkl：** `cache/dataset_cache_789202_mcl100_add.pkl`（633 MB，脑 789202 的单脑 ground-truth + fragments graph 转储，mcl=100，带 `_add` 后处理）。
- **Rerun count：** 20 of 20 排名靠前的假设被重新执行（按 `posterior-surprise` 排序）。
- **Verdict counts：** **REPRODUCED 19**（95%），**DIVERGED 1**（5%），**FAILED 0**（0%）。全部 19 项复现在舍入范围内与记录头条统计精确匹配；唯一的 DIVERGED 案例（H13）绝对数字实质不同，但方向与结论相同。无 top-20 发现运行失败。
- **Code source：** 全部 20 项在 **revised-loading** 脚本上运行（`autodiscovery/ground-truth-error-annotations-revised-version_2026-06-17.json.rerun/hypo_<id>.py`）；无一项按原样运行记录代码。记录代码在 pass-1 reruns 中一致失败，原因是：(a) 每个脚本记录的 `pickle.load` 抛出 `ModuleNotFoundError: No module named 'numpy._core.numeric'`（pkl 由 NumPy 2 写入，宿主 user-site numpy 为 1.26.4）；(b) 大多数脚本用 `os.walk("..")` / `glob.glob("../*.pkl")` / `pathlib.Path('..').rglob('*.pkl')` / `subprocess.getoutput('find ...')` 的"Dataset not found"检查门控执行，而它看不到所提供的 pkl。revised-loading bootstrap (a) 确保在任何 unpickling 前已具备 `numpy>=2,<2.3`，(b) monkeypatch `os.walk`、`pathlib.Path.rglob` 与 `subprocess.getoutput` 以产出所提供的 pkl 路径，使数据集搜索分支成功，(c) 将脚本内的 `pip install ...` 调用置为 no-op，因为宿主已提供每个依赖，且这些安装会在运行中把 numpy 降级回 1.26.4 并破坏 user-site numpy 目录。**任何 revised 脚本中均未改动统计逻辑。**

### Findings that did NOT reproduce exactly

- **Entry 5 / id 13**（DIVERGED）：recorded `Total Correct edges analyzed: 2218068, Total Split edges analyzed: 13610, U=1.19e10, p=0.0` vs rerun `Correct=1109034, Split=6805, U=2.98e9, p=1.32e-197`。记录脚本的 glob 模式在原始环境中返回了两个不同路径（`./` 与 `../data/`）——都指向同一数据集内容——循环独立处理每个文件，故每条边被计了**两次**。rerun 的 glob patch 将两个模式都解析到单一 pkl，故只处理数据集一次。方向（split 边更接近分支点）、效应量关系与结论（p 远低于任何合理显著阈值）均不变。应将记录的 U 统计量视为被原始代码的重复计数缺陷夸大，而非反对底层发现的证据。

### Entries with minor non-reproduction artefacts (still REPRODUCED)

- **Entry 7 / id 23**：统计完全复现（mean 11.09 vs 6.33, U=4100.5, p=8.95e-17）。脚本随后抛出 `TypeError: boxplot() got an unexpected keyword argument 'labels'`（matplotlib 3.11 将其重命名为 `tick_labels`）。仅统计之后的绘图崩溃。
- **Entry 8 / id 30**：统计完全复现（Pearson r=0.6500/p=0.0221, Spearman ρ=0.8811/p=1.53e-04, OLS R²=0.422）。脚本随后抛出 `AttributeError: module 'matplotlib.cm' has no attribute 'get_cmap'`（在 mpl 3.11 中移除）。仅统计之后的绘图崩溃。
- **Entry 12 / id 49**：merge 侧统计完全复现；随机对照节点样本在不同运行间略有差异（记录代码无固定 RNG seed），故随机对照均值/中位数与 U 值适度变动（U=363→381, p=2.21e-17→4.38e-17）。结论成立。
- **Entry 14 / id 55**：KS D 因无固定 seed 的随机零模型重采样相差 0.0001（0.4539→0.4540）；观测中位数、零模型中位数、p 值与结论均匹配。
- **Entry 17 / id 65**：split 边统计完全复现；50,000 条 correct 边的下采样在记录代码中无固定 seed，故 correct 边均值与 U 值适度变动（mean 1278.24→1287.44 µm, U=2.82e8→1.41e8）。两次运行 p=0.0；结论成立。

### Records requiring loading revisions and why

- **全部 20 条记录（47, 3, 10, 11, 13, 19, 23, 30, 35, 39, 48, 49, 53, 55, 60, 64, 65, 67, 72, 81）** 都需要 NumPy-2 unpickling 修复（pkl 由 NumPy 2 写入；宿主 user-site numpy 为 1.26.4 且缺少 `numpy._core.numeric`）。
- **20 中的 17 条**（除 ids 19, 23, 67 外，它们使用 runner 的 `open`-patch 已重定向的硬编码相对路径）额外需要数据集搜索 patch（`os.walk` / `glob.glob` / `pathlib.Path.rglob` / `subprocess.getoutput('find ...')`），使"Dataset not found"门控不会在加载前退出。
- **全部 20 条** 都接受了 `pip install` no-op patch，因为记录脚本运行其自身的 pip-install bootstrap，而 `agentic-neuron-proofreader` 包钉住 `numpy<2`，会在运行中降级宿主 numpy 并破坏进行中的 unpickle。抑制这些重装（宿主已提供每个依赖）对干净运行至关重要。

## Generalization — Summary

- **使用的额外数据集：** `cache/dataset_cache_794491_mcl100_add.pkl` 与 `cache/dataset_cache_794495_mcl100_add.pkl`（两个在结构上与 789202 origin 相同的全脑 pkl）。全部 20 个 revised 假设脚本经现有 rerun bootstrap 针对每个额外 pkl 重新执行；无需进一步加载编辑（H23 与 H30 中 matplotlib 关键字的统计后失败在三个数据集上同样发生，且不影响统计结果，统计在绘图崩溃前已发出）。
- **Verdict counts (n=20)：** **GENERALIZES 16**（80%），**PARTIAL 3**（15%），**DOES-NOT-GENERALIZE 1**（5%），**INCONCLUSIVE 0**（0%）。

### Findings that DO NOT fully generalize

- **Entry 1 / id 47**（DOES-NOT-GENERALIZE）：距离 × 角度交互项 `x3` 在 origin 上显著（coef=−5.96, p=0.014），但在两个额外脑上完全失去显著性（ds_794491 coef=−0.34, p=0.800；ds_794495 coef=−1.45, p=0.565）。整体 logistic 模型在两个额外脑上仍高度显著（LLR p<10⁻³⁷），但定义此发现的具体*非线性*（交互）成分为 789202 独有；在其他脑上，更简单的仅距离（以及距离 + 角度）模型即捕获同样的数据。
- **Entry 4 / id 11**（PARTIAL）：omit 边到最近叶的拓扑距离——origin 显示 omit 显著比 correct 边更近（mean 247 vs 325, p=7.75e-311）；ds_794495 复现方向（omit 125 vs correct 173 µm, p=6.10e-149）。在 ds_794491 上方向翻转：omit 边位于 mean 159.76 / median 84.5 vs correct 138.76 / 70.5（假设方向单侧 p=1.000）。omit 的远端叶集中性为脑依赖。
- **Entry 10 / id 39**（PARTIAL）：几何定向图切断总是改善 accuracy（origin 82.27% → 93.66%；ds_794491 73.81% → 88.13%；ds_794495 68.68% → 81.17%）并削减 merge，但只在 origin 上达到头条 ">80% merge 边削减"阈值（origin 约 86%；ds_794491 约 70%；ds_794495 约 44%）。干预到处有帮助；头条量级为 origin 特有。
- **Entry 19 / id 72**（PARTIAL）：split 比 correct 边（测地上）更接近 GT 分支节点——方向在三个脑上保持且 p 仍 < 1e-26，但效应量在 ds_794491 上坍缩（split mean 386.4 vs correct mean 391.5 µm，仅 5 µm 差距；尽管中位差距仍约 39 µm）。"split 距离为一半"框架为脑特异，尽管方向性发现可推广。

### Synthesis

本次运行占主导的横贯主题——**merge 聚集在结构密集/多分支区域；split 与 omit 聚集在纤细的远端/末端分支及分叉处**——是稳健的：每条 merge 密度发现（H3, H23, H49, H53, H81）、每条 split 级联或 split 聚集发现（H10, H13, H19, H35, H48, H55, H65, H67），以及 OMIT 条纹性 / 末端-vs-内部不对称（H48, H64），都在两个额外脑上复现，常以可比或更强的显著性、仅有适度的效应量变化。两条依赖 789202 特定*定量*特征的发现——split-gap merging 的**非线性交互结构**（H47）与几何切断的**头条 ">80% merge 边削减"**（H39）——部分为数据集特异：底层机制仍在运作，但参数化论断不迁移。一条发现在方向上确实脑依赖：**omit 在拓扑上接近叶（H11）在 ds_794491 上翻转符号**，暗示 omit 的远端偏置本身是各体积如何被重建的属性，而非普适的模型失败模式。仅有两个额外脑，证据为中等置信——16/20 的推广率令人鼓舞，但不应被过度解读为普适适用。

## Excluded (no surprisal score)

- ground-truth-error-annotations-revised-version_2026-06-17 · id 15
- ground-truth-error-annotations-revised-version_2026-06-17 · id 41

## Statistical Verification — Summary

### Verdict counts (top-20 audited)

- **OK：** 4 — H3 (id 3, entry 2), H35 (id 35, entry 9), H53 (id 53, entry 13), H67 (id 67, entry 18)。
- **MINOR：** 11 — H10 (id 10, entry 3), H19 (id 19, entry 6), H23 (id 23, entry 7), H30 (id 30, entry 8), H48 (id 48, entry 11), H49 (id 49, entry 12), H55 (id 55, entry 14), H60 (id 60, entry 15), H64 (id 64, entry 16), H65 (id 65, entry 17), H81 (id 81, entry 20)。
- **MAJOR：** 5 — H47 (id 47, entry 1, 被 DOES-NOT-GENERALIZE 下调)；H11 (id 11, entry 4, 被 ds_794491 上的符号翻转下调)；H13 (id 13, entry 5, 被重复计数代码缺陷下调)；H39 (id 39, entry 10, 被头条 ">80%" 削减的 PARTIAL 推广下调)；H72 (id 72, entry 19, 被 "half-distance" 量级论断的 PARTIAL 推广下调)。
- **CRITICAL：** 0。
- **Total：** 20。

### Multiple comparisons — file-wide BH-FDR analysis

完整运行含 100 条假设；98 条带有 `surprisal` 分数（ids 15 与 41 被排除）。从 `analysis` + `codeOutput` 文本中，我们能为 **69 / 98** 条记录提取头条 p 值（其余为描述性发现、"Pivot Required" 退出案例，或未发出数值 p 值）。将字面 `p = 0.0` 下限报告视为 p = 1e-300，**这 69 条中有 48 条在未校正 α = 0.05 下名义显著**。

在 69 个提取的 p 值上以 q = 0.05 应用 Benjamini-Hochberg FDR 控制：**全部 48 条名义显著发现都通过 BH**（BH 截断为 p ≤ 0.0108；最大通过的 p 为 0.0108，而下一个更大的报告 p > 0.34，故间隔很宽，FDR 控制并非约束性约束）。**全部 20 条排名靠前的假设都通过 BH-FDR**，包括 H47（来自 LLR 的 p = 1.63e-17，交互项 p = 0.0139 仍通过其秩阈值）、H30（Pearson p = 0.022 仍通过其更大的秩阈值），以及每条 "p = 0.0" 下限报告（H13, H35, H48, H55, H64, H65, H72）。无临界显著的头条发现未通过 FDR。因此 BH 分析*并非*本次运行的隐藏弱点；更严重的问题是复现/推广失败（H11, H47）与代码缺陷驱动的统计量膨胀（H13）。

### Most serious problems found (CRITICAL and MAJOR)

- **H11 / Entry 4（MAJOR — 方向翻转）：** "omit 在拓扑上更接近叶"论断在 origin 上 p ≈ 7.75e-311，但**在 ds_794491 上符号翻转**（omit *更远离*叶；假设方向单侧 p = 1.000）。模型 omit 错误的一般"远端-纤细偏置"头条框架在跨脑下不被支持；结论应限定为脑特异，或反映 GT-vs-prediction 基线分布的差异。
- **H13 / Entry 5（MAJOR — 代码缺陷膨胀统计量）：** 记录的 `Correct n=2,218,068, Split n=13,610, U=1.19e10, p=0.0` 恰为去重真实计数的 2×；脚本处理了两个指向同一 pkl 的 glob 命中并把每条边计了两次。去重 rerun 给出 U=2.98e9, p=1.32e-197——仍极端，但记录的 U 统计量被人为加倍。应引用去重数字。
- **H47 / Entry 1（MAJOR — DOES-NOT-GENERALIZE）：** 头条机制（split-gap 接受中的非线性距离 × 角度交互）仅在 origin 上以 p = 0.014 显著；在两个额外脑上交互项不可检测（p = 0.800 与 p = 0.565）。头条洞见为脑特异，尽管整体 logistic 模型处处显著。本次运行中最具信念改变力的单条发现不迁移。
- **H39 / Entry 10（MAJOR — 定量论断为脑特异）：** 几何切断*总是*有帮助，但头条 ">80% merge 边削减"只在 origin 上成立（约 86%）；ds_794491 达约 70%，ds_794495 仅约 44%。记录分析未报告推断性检验（对配对每神经元增量无 Wilcoxon，对削减比例无 CI）。可操作框架应缩减为一个范围，"视脑而定去除 44-86% 的 merge 边"。
- **H72 / Entry 19（MAJOR — 效应量坍缩）：** origin 上为 correct 边的"中位距离约一半"（194 µm 均值差距）在 ds_794491 上坍缩为 5 µm 均值差距，而 p 纯由样本量（n>10⁶）维持在 0.0 下限。巨大 n 的 p 值掩盖消失效应的教科书示例。

一个普遍的横贯问题是：许多高 n 检验（H11, H13, H35, H48, H55, H64, H65, H72）报告的 `p = 0.0` 下限值除"非常显著"外说明不了任何有意义的东西；**效应量陈述（比值、均值偏移、U 比值）才是承载性数字**，而非 p 值。若干发现（H3, H23, H49, H53, H81）是同一 merge-密度-于-拥挤信号的紧密相关变体，应计为一项有效发现而非五项。skeleton 派生测量的跨边非独立性（H10, H11, H13, H19, H35, H55, H60, H64, H65, H72）一律未被计入；这抬高显著性但一般不反转方向，除非在额外脑上的 rerun 证明它确实如此（尤其 H11）。"discovery loop" 在本次运行中最严重的失败模式是结论*越界*——将单脑定量论断（H39, H47, H72）与单检验因果论断（H30, H55）推广到数据所隔离之外。

## Statistical Test Corrections — Summary

### Scope

top-20 假设中有 11 条被标记为 TEST 级修复并以校正统计流程重新测量（`autodiscovery/ground-truth-error-annotations-revised-version_2026-06-17.json.fixed/hypo_<id>.py`）。每个校正脚本保持相同的数据与相同的被比较量；只改变检验、效应量 + CI，以及（在独立性被违反处）cluster/permutation p。全部 11 个校正脚本在 origin（789202）与两个额外脑（794491, 794495）上运行。在每条条目的 bullets 中，记录数字与校正数字并列引用。

### Post-correction verdict breakdown (n=11 fixed)

- **UPHELD：** 8 — H10 (entry 3), H11 (entry 4), H13 (entry 5), H30 (entry 8), H47 (entry 1, origin only), H48 (entry 11), H55 (entry 14), H81 (entry 20)。
- **WEAKENED：** 3 — H39 (entry 10), H64 (entry 16), H72 (entry 19)。
- **OVERTURNED：** 0。

### Findings that changed under the correct test

- **H11 / id 11 (entry 4)** — 将原始 "DOES-NOT-GENERALIZE" 判定反转为 GENERALIZES。在 n>10⁶ 相关边上的汇集 Mann-Whitney 在 ds_794491 上符号翻转（"omit *更远离*叶"），但对每 skeleton 中位偏移的配对 Wilcoxon 显示 ds_794491 的 9 个 skeleton 中有 8 个 omit 更近叶（p=0.014，对均值偏移的 95% CI [-21.45, -4.89] µm）。同样的模式在 origin（12/12 skeleton, p=2.4e-04, CI [-90.51, -29.83]）与 ds_794495（13/19, p=0.009, CI [-28.92, -6.16]）成立。记录的方向翻转是汇集距离尺度差异巨大的 skeleton 的伪影。**正确检验：** skeleton 内配对 Wilcoxon + cluster-bootstrap CI；**原始为何错误：** 汇集 MWU 中的非独立性与 skeleton 级尺度异质性。
- **H13 / id 13 (entry 5)** — 记录 U=1.19e10 因重复计数缺陷恰为真实 U 的 2×；去重 cluster-aware 检验给出 U=2.98e9, p=1.32e-197，对中位偏移的 cluster bootstrap CI [-244.50, -122.03] µm（origin）。**正确检验：** 去重 + 带 bootstrap CI 与 cluster-permutation p 的 cluster-aware MWU；**原始为何错误：** 代码级 glob 重复计数加上在 skeleton 内相关边上的巨大 n 的 p=0 下限。
- **H39 / id 39 (entry 10)** — 定量 ">80% merge 边削减" 论断即便在 origin 上也未通过适当 CI（对去除比例的 cluster bootstrap CI [0.7007, 0.9631]）；在 ds_794495 上比例为 0.43，CI [0.23, 0.65]。配对 Wilcoxon 在每个脑上强显著（p ≤ 2e-3）——方向坚实，量级不然。**正确检验：** 对每神经元增量的配对 Wilcoxon signed-rank + 对削减比例的 cluster-bootstrap CI；**原始为何错误：** 未报告推断性检验，仅有单点描述性均值。
- **H47 / id 47 (entry 1)** — 在 origin 上 UPHELD（交互项的 LR 检验，p=9.94e-03, bootstrap CI [-10.28, -1.25]），但额外脑上同样的 LR 检验确认原始 verifier 的判断——头条非线性距离×角度机制不推广（ds_794491 LR p=0.797, CI 跨 0；ds_794495 LR p=1.000, CI 跨 0）。**正确检验：** 加性 vs 交互 Logit 的 LR 比较 + 对交互系数的类别分层 bootstrap CI + 标签置换 p；**原始为何错误：** 记录的 LLR 是对仅含截距（一个过于乐观的基线，把加性距离/角度主效应混入交互的"新颖性"），且仅有单个 Wald p 而无系数 CI。
- **H48 / id 48 (entry 11)** — 记录 chi^2=2,226,077.87 因对每个无向邻接边对的互逆计数被人为加倍。去重 chi^2 = 1,113,217.84（恰为一半），确认该缺陷。条件/边际比值不变，为 28.09×（origin）、18.63×（ds_794491）、40.42×（ds_794495），各 cluster-permutation p=3.3e-03。**正确检验：** 无序对计数 + cluster-permutation p + 对比值的 bootstrap CI；**原始为何错误：** 互逆重复计数恰将 chi^2 膨胀 2×，且在相关的、易成条纹的边上作独立性检验本身违反独立性。
- **H64 / id 64 (entry 16)** — cluster-aware 检验揭示，记录的 "1.82× 末端 vs 内部 OMIT 率" 比值在计入 skeleton 级依赖后于 ds_794491 上跨 1.0（CI [0.96, 1.89]）——方向保持，但量级界为脑特异。**正确检验：** 对率比的 cluster-bootstrap CI + skeleton 内末端标签置换；**原始为何错误：** chi-square 假设约 140 万条边独立，但并不独立（同一最大度-2 路径上的边继承相同的末端/内部标签）。
- **H72 / id 72 (entry 19)** — cluster-bootstrap CI 明确量化了记录 p=0 下限所掩盖的东西：在 origin 上中位偏移为 −184 µm（CI [-244.95, -121.65]），但在 ds_794491 上仅 −38.86 µm（CI [-58.66, -25.41]），在 ds_794495 上仅 −29.99 µm（CI [-54.51, -6.70]）。方向可推广，量级不可。**正确检验：** cluster-aware MWU + 对中位偏移的 bootstrap CI + skeleton 内置换；**原始为何错误：** 巨大 n 的 p=0 下限把脑特异效应报告得仿佛普适。

### One-paragraph synthesis

一旦应用正确的统计检验，本次运行占主导的横贯图景——即**merge 聚集在结构密集或多分支区域，split/omit 聚集在纤细远端分支及分叉处**——大体上是 ROBUST 的：每个 test 级修复（H10, H11, H13, H30, H47-origin, H48, H55, H64, H72, H81）都保持记录效应的方向，多数也保持量级。出现三处具体变化：(1) **H11 头条级的"不推广"发现被反转**——在 cluster-aware 配对检验下它实际上在三个脑上推广良好，记录的 MWU 之所以误导是因为它汇集了距离尺度不同的 skeleton。(2) **两个记录缺陷被暴露并量化**——H13 的重复计数 U（约 2x 膨胀）与 H48 的互逆加倍 chi^2（恰 2x 膨胀）。(3) **记录分析呈现为普适的两个定量量级论断**——H39 的 ">80% merge 边削减" 与 H72 的 "中位距离约一半"——在额外脑上明确未通过适当的 cluster-bootstrap CI，应重述为范围。H47 的机制在 origin 上以适当的 LR 检验存活，但仍不迁移到额外脑，与记录 verifier 的判断一致。"巨大 n 的 p=0 下限"发现（H13, H64, H72）在 cluster-aware 检验下都保持方向，但其效应量 CI 揭示了记录 p 值受样本量驱动的程度。无 fixed 发现被 OVERTURNED；test 校正把许多 p=0 下限变成了适当有界的效应量陈述，并补上了记录分析所缺失的不确定性。
