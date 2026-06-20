# AutoDiscovery Run 4 — Top-20 排序结论

## 来源

- **文件：** `autodiscovery/run-4--ground-truth-error-annotations-revised-version_2026-06-20.json`（150 条假设）
- **排序依据：** `posterior-surprise`（priority = posterior × |surprisal|）
- **参与排序的假设：** 148（2 条因缺失 surprisal 被排除：IDs 10、41）
- **本报告返回的假设：** 148 条中的 top 20
- **Surprise 量级范围：** 0.0000 – 0.6899
- **观测到的最大 priority 分值：** 0.5066

### 总览综述

优先级最高的单条发现（#1，ID 30）是一次置信度翻转，转而支持仅用欧氏间隙距离（约 6.84 µm 阈值，AUC 0.9979）来自动桥接 split 误差——一个 "Leaning False" 的先验被近乎完美的 ROC 推翻为 "Leaning True"。接下来两条高优先级条目（#2 ID 27、#3 ID 21）是*负向*翻转，否定了一个此前被自信持有的假设：z 轴各向异性曾被广泛认为会驱动 split/omit 误差，但两个独立的统计检验（混合效应 logistic 回归与一个 Chi-square 分割检验，p 均约为 0.06）都未能找到显著的 Z-vs-XY 效应，使置信度从 "Likely True" 坍缩到 "Uncertain"。其余 17 条形成一个紧凑簇（priority 0.253，|surprisal| 0.2841），都是已确认但中等惊讶度的先验：空间/拓扑特征（分支点、终末边、迂曲度、分支密度缠结、merge-omit 共址）反复作为误差类型的高 AUC 判别器出现，而 merging 段被确认为"巨大的"过度生长标签，而非偶发噪点。综合来看，本次运行论证了**局部几何——邻近度、角度、迂曲度、分支密度——是比成像轴更可操作的误差信号**。

---

## 排序条目

### 1. (Priority 0.507 · Surprise 0.690) 仅靠欧氏间隙距离就能近乎完美地区分真实 split 与跨神经元邻居，从而翻转为支持仅基于距离的自动重连。
- **Run：** run-4--ground-truth-error-annotations-revised-version_2026-06-20 · **ID：** 30 · **Belief：** Leaning False → Leaning True (0.2917 → 0.7344) · **Direction：** Positive
- **Tested：** 属于同一 GT 神经元的被 split 预测段端点之间的欧氏间隙，是否明显小于到局部邻域中*不同* GT 神经元碎片的间隙，即仅靠空间邻近度进行自动重连是否安全。
- **Conclusion：** 在 20 µm 内的 6,805 个真实 split 间隙与 4,189 个跨神经元间隙中，真实 split 距离紧密地峰值集中在约 4.5 µm（大多 2–7 µm），而跨神经元间隙极少低于 7 µm。二元分类器 ROC AUC 为 0.9979，F1 最优阈值为 6.84 µm（max F1 = 0.9945），证实仅靠距离即为可行、安全的启发式。强正向 surprisal 反映了先前 "Leaning False" 的假设（担心意外 merge）被推翻。
- **Caveats：** 未记录；样本量很大，判别器基本饱和。
- **Reproduction：** REPRODUCED (code: revised-loading)
- **Rerun result：** 记录值 "Number of true split gaps: 6805, false positive gaps: 4189, ROC AUC: 0.9979, Optimal Distance Threshold: 6.84 um, Max F1 Score: 0.9945"；重跑 "true split gaps: 6805, false positive gaps: 4189, ROC AUC: 0.9979, Optimal Distance Threshold: 6.84 um, Max F1 Score: 0.9945" → 完全一致。修订：直接 $RERUN_PKL 加载 + numpy>=2 + 内置（vendored）scikit-learn，以绕过 numpy-2 与系统 sklearn 构建之间的 ABI 不兼容。
- **Generalization：** GENERALIZES
- **Across datasets：** origin 789202：n_true=6805，n_false=4189，ROC AUC=0.9979，optimal threshold=6.84 µm，max F1=0.9945；794491：n_true=7847，n_false=13208，AUC=0.9889，threshold=6.48 µm，F1=0.9711；794495：n_true=7988，n_false=8867，AUC=0.9953，threshold=6.83 µm，F1=0.9878。真实 split 与跨神经元间隙的仅距离可分性在全部三个脑上都成立（AUC ≥ 0.989，threshold 6.48–6.84 µm，F1 ≥ 0.97）。
- **Verdict：** OK
- **Test：** 在二元分类器（间隙距离）上用 F1 最优阈值选择的 ROC-AUC——适用于标签-vs-距离的判别问题；无参数假设。
- **Statistical issues：**
  - 阈值是通过*在同一数据上*最大化 F1 来选取的，这是轻度的样本内乐观偏差；留出验证会使该阈值主张更稳固，但在 n = 10,994 个间隙上 AUC = 0.9979，且在另两个脑上重现且 AUC ≥ 0.989，过拟合不太可能。
  - 4,189 个"跨神经元"对照在空间上被限定为 20 µm 内——这界定了操作范围，但对于实用启发式问题而言是恰当的。
- **Logic issues：**
  - "适合自动重连"的结论有经验支持，但仍取决于 GT 是否正确以及 20 µm 内什么算作"候选"；部署的 agent 需要防范测试集未涵盖的致密神经毡区域中的假阳性。推广到另两个脑大大缓解了这一担忧。
- **Downgrade based on rerun/extrapolation：** No——在全部三个脑上 GENERALIZES，AUC 一致 ≥ 0.989。

### 2. (Priority 0.323 · Surprise 0.795) 沿 Z 轴排列的神经突并不比沿 XY 排列的更易出错，从而退役一个被自信持有的各向异性假设。
- **Run：** run-4--ground-truth-error-annotations-revised-version_2026-06-20 · **ID：** 27 · **Belief：** Likely True → Uncertain (0.9167 → 0.4062) · **Direction：** Negative
- **Tested：** 用以神经元为随机效应的贝叶斯混合效应 logistic 回归，检验沿低分辨率 Z 成像轴排列的神经突是否比处于高分辨率 XY 平面者承受更多 split/omit 误差。
- **Conclusion：** 在 1,160,529 条有效边（4.44% 误差率）中，一个 20,000 边的回归返回标准化 z-alignment 系数 −0.0661（p = 0.0582，OR ≈ 0.80）。该效应不显著且方向略微*相反*，因此 z 轴朝向不是碎片化的主要驱动因素。较大的负向 surprisal（−0.7954）标志着这是对一个此前近乎确定的先验的重大置信度下降。
- **Caveats：** 该回归使用了 >1.1M 边中的 20k 边子样本；p 值（0.0582）处于边缘，在全数据集下可能跨过显著性，尽管 OR 仍接近 1。
- **Reproduction：** REPRODUCED (code: revised-loading)
- **Rerun result：** 记录值 "Total valid edges loaded: 1160529, errors 51495 (4.44%); Standardized Coefficient (Z-alignment): -0.0661 (p-value = 0.05823); OR = 0.8029"；重跑 "Total valid edges loaded: 1160529, errors 51495 (4.44%); Standardized Coefficient (Z-alignment): -0.0661 (p-value = 0.05823); OR = 0.8029" → 完全一致。修订：直接 $RERUN_PKL 加载 + numpy>=2 + 内置 patsy/statsmodels，以修复 pandas 3 下的 patsy "<StringDtype(na_value=nan)>" 崩溃。
- **Generalization：** PARTIAL
- **Across datasets：** origin 789202：标准化 coef z_align=−0.0661，p=0.0582，OR=0.8029（不显著，略负）→ "Z not a driver"；794491：coef=+0.0376，p=0.1747，OR=1.1371（不显著，符号翻转为正）→ 脚本仍打印 "Z does NOT significantly affect risk"；794495：coef=−0.1697，p=6.75e-06，OR=0.5632（高度显著，负向 → Z-alignment *降低*误差几率，脚本打印 "Z-axis alignment significantly DECREASES the risk"）。该假设的文字结论 "Z is not meaningfully more error-prone than XY" 在三个脑上仍成立（任何脑上都无正向 Z 效应），但底层系数的符号（−0.17、−0.07、+0.04）与显著性变化剧烈，其中 794495 通过发现强保护性效应而与"无效应"先验相矛盾。因此混合效应回归结果并不稳健。
- **Verdict：** MAJOR
- **Test：** 贝叶斯混合效应 logistic 回归（二元误差结局，神经元为随机效应）——原则上对聚类二元数据是正确的，但所选子样本量（1,160,529 边中的 20,000，约 1.7%）是一个随意的截断，浪费了约 98% 的可用数据，正是这导致 p 值停留在 0.058 的边缘。
- **Statistical issues：**
  - **子抽样：** 在 1.16M 边中只用 20k 跑回归是不合理的；标准误无端被抬高。若用全数据集，同样的 OR（约 0.80）几乎必然会跨过 α=0.05，削弱"不显著"的裁决。
  - **"未能拒绝 != 原假设为真"谬误：** "z 轴朝向不是主要驱动因素"的结论是从 p = 0.058 推出的——证据缺失不等于缺失的证据。在 OR ≈ 0.80 下，该检验与一个真实但很小的保护性效应是一致的。
  - 精确 p 值 0.05823 落在"边缘"带内，多重检验惩罚（此处 FDR q=0.05 截断约为 0.044）会将其推出显著范围；即便仅在 top-20 中它也无法通过 BH-FDR。
- **Logic issues：**
  - **置信度翻转超出证据所能支撑：** 先验 "Likely True" → 后验 "Uncertain"（以及较大的负向 surprisal −0.795）完全由一个子样本上的单一边缘 p 值驱动。该假设实际是在不确定的证据上被"退役"的。
  - 推广证实了这一点：在 794495 上回归以*相反*方向强烈拒绝了无效应原假设（OR=0.56，p=6.75e-06），因此"无各向异性"的标题是脑特异的，而非普适的。
- **Downgrade based on rerun/extrapolation：** Yes——PARTIAL 推广（符号与显著性跨脑变化）意味着 "Z is not a driver" 的标题不能被宣称为稳健发现；至少应重述为"在这些脑上无正向 Z 效应，且有一个脑显示显著保护性效应"。
- **Corrected test：** 以神经元为 cluster id 的 cluster-robust GEE logistic 回归，在完整约 1.16M 边上拟合，而非 20k 子样本。GEE 是正确的检验，因为 (a) 它使用每一个数据点而非丢弃 98%，且 (b) 它通过 cluster-robust 标准误考虑了神经元内相关性——正是原始贝叶斯随机效应试图刻画的结构。
- **Corrected result：** origin 789202 标准化 coef(z_align) = -0.1229，SE = 0.0112，z = -10.93，p ≈ 0；每 1 单位 z_align 的 OR = 0.6649，95% CI [0.6180, 0.7154]；n_edges = 1,160,529，n_neurons = 12。并列对比：原始 "coef = -0.0661, p = 0.05823, OR = 0.8029"（20k 子样本）变为 "coef = -0.1229, p ≈ 0, OR = 0.6649 [0.6180, 0.7154]"——即当在全数据上以恰当聚类测量时，该效应是真实的且显著保护性的，而非"无效应"。
- **Corrected generalization：** PARTIAL——origin 789202：OR = 0.6649 [0.6180, 0.7154]，p ≈ 0（Z 显著*保护*）；794491：OR = 1.1198 [0.8117, 1.5448]，p = 0.49（无显著效应，CI 跨越 1）；794495：OR = 0.5141 [0.4254, 0.6213]，p = 5.7e-12（Z 显著保护）。保护性方向在 2/3 脑上一致；在 794491 上效应缺失（CI 穿过 1）。在校正检验下 Z 从不是风险因子。
- **Post-correction verdict：** OVERTURNED——原始文字标题 "Z-aligned neurites are NOT meaningfully more error-prone than XY-aligned ones" 仅作为一个否定主张存活；在 cluster-robust GEE 下*正确*的主张是 "Z-alignment is a significant PROTECTIVE factor (OR ≈ 0.51-0.66) on 2/3 brains"，与先验相反。原始 p=0.058 是子抽样伪影，"Uncertain" 后验也无依据——一旦把 1.16M 边正确聚类，origin 与 794495 上的证据都很强。

### 3. (Priority 0.266 · Surprise 0.568) Z 主导与 XY 主导边的误差率基本相同（3.70% vs 3.63%），独立印证了无各向异性的结果。
- **Run：** run-4--ground-truth-error-annotations-revised-version_2026-06-20 · **ID：** 21 · **Belief：** Likely True → Uncertain (0.8333 → 0.4688) · **Direction：** Negative
- **Tested：** 通过对 Z 主导 vs XY 主导边的 Chi-square 检验，检验拓扑边朝向是否如成像各向异性理论所预测的那样分割 split/omit 误差率。
- **Conclusion：** 433,243 条 Z 主导边（3.70% 误差）vs 975,802 条 XY 主导边（3.63% 误差）得到 χ² = 3.5328，p = 0.0602——未达显著。微小的绝对差异加上边缘 p 值否定了"Z 轴偏置"说法，负向 surprisal（−0.5681）再次代表先前自信的 "Likely True" 信念的实质性坍缩。这印证了条目 #2。
- **Caveats：** p 值刚好略大于 0.05；在如此庞大的样本量下，未能拒绝是有意义的，但无论如何效应（若有）的绝对量级都微不足道。
- **Reproduction：** REPRODUCED (code: revised-loading)
- **Rerun result：** 记录值 "Z-dominant edges: 433243 total, 16027 errors (3.70%); XY-dominant: 975802 total, 35468 errors (3.63%); Chi2 = 3.5328, p = 6.0165e-02"；重跑 "Z-dominant: 433243 total, 16027 errors (3.70%); XY-dominant: 975802 total, 35468 errors (3.63%); Chi2 = 3.5328, p = 6.0165e-02" → 完全一致。修订：直接 $RERUN_PKL 加载 + numpy>=2。
- **Generalization：** DOES-NOT-GENERALIZE
- **Across datasets：** origin 789202：Z 3.70% vs XY 3.63%，χ²=3.53，p=0.060（无显著差异）；794491：Z 6.41% vs XY 5.50%，χ²=166.12，p=5.21e-38（Z 误差率显著更高）；794495：Z 2.24% vs XY 2.86%，χ²=395.76，p=4.62e-88（Z 误差率显著更低）。"无各向异性"结论在两个额外脑上都被拒绝，且方向相反（794491 上 Z 更差，794495 上 Z 更好）。因此 χ² 检验跨脑不稳健——任何 Z-vs-XY 效应的方向都取决于数据集。
- **Verdict：** CRITICAL
- **Test：** 对边朝向（Z 主导 vs XY 主导）× 误差（是/否）的 Chi-square 独立性检验，χ²=3.5328，p=0.0602，n=1,409,045 边——该检验对 2×2 列联表属于正确族系，但它把每条边当作独立观测，而事实上同一神经元/段内的边高度聚类（非独立性）。
- **Statistical issues：**
  - **违反独立性：** 来自同一神经元/碎片的边共享空间、生物学和标注结构。把 1.4M 边当作 i.i.d. 会抬高有效 n 并缩小 p 值（或在此处收窄 H₀ 下的 χ² 分布）；cluster-robust 或混合效应 logistic 检验才是正确方法。
  - **效应量微不足道：** 绝对差异为 3.70% vs 3.63%（相对约 2%），即便显著也无生物学意义。在 n = 1.4M 上 χ² 统计量为 3.53，是一个在任何合理 α 下显著性都无关紧要的教科书例子。
  - **p 刚过 0.05：** 在 BH-FDR 下此排名处的截断要小得多（≈ 0.044）；即便在考虑聚类问题之前，这条假设也会是边缘的未通过者。
- **Logic issues：**
  - **"未能拒绝 = 原假设为真"谬误：** 结论明确读作 "Z-dominant edges do NOT exhibit a significantly higher rate" → "refutes hypothesis that imaging anisotropy creates a substantial directional bias"。刚刚高于 0.05 的 p 不能否定 H₁；它只能未能拒绝 H₀。
  - 推广到两个额外脑以*相反*方向拒绝原假设（794491 上 Z 更差，794495 上 Z 更好），表明无各向异性的结论是脑 789202 特定分布的伪影。
- **Downgrade based on rerun/extrapolation：** Yes——DOES-NOT-GENERALIZE。"无各向异性效应"标题被两个额外脑推翻；这条假设应被撤回，而不应作为置信度下降来报告。
- **Corrected test：** 在正确分析单元上的三个互补检验：(a) 以神经元为 cluster id 的 GEE logistic 回归，(b) cluster-permutation chi-square（Z/XY 标签在*神经元*层面置换，使一个神经元内所有边一起翻转），(c) 对（Z 误差率 – XY 误差率）的逐神经元配对 Wilcoxon signed-rank。三者都尊重原始 i.i.d. chi-square 忽略的神经元内聚类。
- **Corrected result：** origin 789202：GEE OR(Z vs XY) = 1.0184，95% CI [0.9712, 1.0680]，p = 0.45；cluster-permutation p = 0.6304（1000 次置换，12 个簇）；逐神经元 Wilcoxon W = 32，p = 0.62，对每神经元 median(Z − XY) 的 bootstrap 95% CI = [−0.092%, +0.350%]。三者在 origin 上一致：一旦检验尊重神经元层级聚类便无效应。并列对比：原始 "chi² = 3.5328, p = 0.0602 on n = 1,409,045 edges" 本已不显著但把 1.4M 当作有效 n；正确的有效 n ≈ 12 个神经元，校正后 p（0.45–0.63）距离显著更远。
- **Corrected generalization：** DOES-NOT-GENERALIZE——origin 789202：GEE p = 0.45，cluster-perm p = 0.63，Wilcoxon p = 0.62（无效应）；794491：GEE OR = 1.18 [0.98, 1.41]，p = 0.076（NS 但提示正向），cluster-perm p = 0.0020（显著），Wilcoxon p = 0.20（NS）——信号混杂；794495：GEE OR = 0.78 [0.70, 0.87]，p = 5.7e-06（显著负向——Z 误差更少），cluster-perm p = 0.0010（显著），Wilcoxon p = 2.1e-04（显著负向）。origin 显示无效应；794491 在 1/3 检验中提示正向；794495 显示稳健的*保护性*效应。
- **Post-correction verdict：** 在 origin 上 UPHELD（当检验正确处理聚类时无各向异性效应），但文字框架现在站得住脚：不再是 "p ≈ 0.06, fail to reject" 而是 "GEE p = 0.45 with neuron-clustered SE, CI [0.97, 1.07] centred on 1"。推广仍为 DOES-NOT-GENERALIZE，因为两个额外脑显示*相反*方向（794495 上 Z 保护，794491 上 Z 略有风险），且都有强烈的聚类显著证据。

### 4. (Priority 0.265 · Surprise 0.414) 离心分支阶（centrifugal branch order）终究能预测 split 误差，深阶分支误差率飙升超过 3%。
- **Run：** run-4--ground-truth-error-annotations-revised-version_2026-06-20 · **ID：** 139 · **Belief：** Leaning False → Leaning True (0.3750 → 0.6406) · **Direction：** Positive
- **Tested：** 通过对约 1.4M 边的 logistic 回归，检验 split 误差概率是否随离心分支阶（距胞体的拓扑深度）上升，且独立于局部线缆粗细。
- **Conclusion：** 分支阶显著（系数 0.0194，p < 0.001），split 误差率在约 30 阶之前保持低且稳定，随后飙升，在约 50 阶之后变得波动加剧（在约 53 阶附近 >3%）。正向 surprisal（0.4139）反映了一个倾向 False 的先验被数据推翻。
- **Caveats：** `norm_thickness` 在整个数据集上方差为零，不得不被丢弃，因此该假设所提的（"独立于局部线缆粗细"）实际无法检验——粗细在该数据集中只是不是一个可用的预测变量；深阶误差飙升也依赖更小的每箱样本计数和高波动性。
- **Reproduction：** REPRODUCED (code: revised-loading)
- **Rerun result：** 记录值 "branch_order coef 0.0194, p < 0.001, ~1.4M edges, norm_thickness dropped (zero variance)"；重跑 "Total edges processed: 1409045; branch_order coef 0.0194, std err 0.001, z=16.369, P>|z|=0.000; const = -5.6486; 'norm_thickness' has zero variance ... Dropping" → 系数相同、z 统计量相同、丢弃 thickness 行为相同。修订：$RERUN_PKL 替换了一个在重跑沙箱中返回空的 `os.walk('..')` 搜索门 + numpy>=2 bootstrap。
- **Generalization：** DOES-NOT-GENERALIZE
- **Across datasets：** origin 789202：branch_order coef=+0.0194（z=+16.37，p<0.001），覆盖 1,409,045 边（正向 → 更深阶分支更易 split）；794491：coef=−0.0157（z=−7.66，p<0.001），覆盖 562,675 边（负向 → 更深阶分支*更不*易 split）；794495：coef=−0.0063（z=−4.20，p<0.001），覆盖 1,363,789 边（负向，量级更小）。三者都高度显著，但符号在两个额外脑上都翻转。因此"深分支阶增加 split 风险"的主张是 789202 脑特异的；在另两个脑上关系相反。
- **Verdict：** CRITICAL
- **Test：** 在 1,409,045 边上对 split 误差关于离心分支阶的 logistic 回归（coef=+0.0194，z=16.37，p<0.001）——该检验族适合二元结局，但观测是神经元内的边，而模型未用随机效应/未用 cluster-robust SE。
- **Statistical issues：**
  - **违反独立性：** 仅来自一小组神经元的 1.4M 边共享空间和分支结构；标准 logistic 回归把它们当作 i.i.d. 并严重低估标准误。以神经元为随机截距的混合效应模型才合适。
  - **效应量 vs 显著性：** coef = 0.0194（每单位分支阶）→ 每阶比值比 ≈ 1.0196，每单位的增量近乎微不足道。"50 阶后飙升 >3%" 是由稀疏的高阶箱和 logistic 曲线驱动的，而非标题系数。
  - **混杂被丢弃而非控制：** `norm_thickness` 方差为零，因此假设中"独立于线缆粗细"的限定语实际未被检验——它被丢弃了。
  - 在巨大 n 下小效应轻易越过 α=0.001；统计显著性在此几乎不起科学作用。
- **Logic issues：**
  - 结论将关系表述为因果（"更深拓扑分支更易 split"）；回归纯属观察性，无法排除混杂（深度、半径、线缆密度等）。
  - **挑选式举例：** "约 53 阶附近 >3%" 的飙升被强调，但它依赖小 n 的箱，而全局每单位系数很小——可视化夸大了模型实际捕捉到的内容。
  - 推广在两个额外脑上翻转符号，同时仍各自打印 p<0.001。高度显著 + 跨数据集符号反转，是 (i) 真实的脑特异结构或 (ii) 混杂推断的标志，而非稳健的普适规律。
- **Downgrade based on rerun/extrapolation：** Yes——DOES-NOT-GENERALIZE。标题（"深阶分支误差率飙升超过 3%"）在 794491 和 794495 上被反驳（负斜率）；789202 上的正符号是数据集特定的。
- **Corrected test：** 以神经元为 cluster id 的 GEE logistic 回归（cluster-robust SE），将 is_split 关于 branch_order 建模。与原始相同的回归族，但标准误现已针对原始忽略的神经元内聚类做了校正。同时报告神经元层级上 mean（及 max）branch_order 与每神经元 split 率之间的 Spearman 作为聚合层级的辅助检查。
- **Corrected result：** origin 789202：GEE coef(branch_order) = +0.01943（点估计相同），SE = 0.00935（比普通-GLM SE 0.00119 大约 8 倍），z = 2.078，p = 0.0377；每 +1 branch_order 的 OR = 1.0196，95% CI [1.0011, 1.0385]。并列对比：原始普通 GLM "coef = +0.0194, z = 16.37, p < 0.001" 在尊重神经元聚类后降为 "coef = +0.01943, z = 2.08, p = 0.038"——p 值损失约 57 个数量级。辅助的神经元层级 Spearman 不显著（rho = +0.23，p = 0.47）。
- **Corrected generalization：** DOES-NOT-GENERALIZE——origin 789202：OR = 1.0196 [1.001, 1.039]，p = 0.038（勉强显著，CI 几乎穿过 1）；794491：OR = 0.9844 [0.970, 0.999]，p = 0.034（勉强显著，方向相反）；794495：OR = 0.9937 [0.972, 1.016]，p = 0.58（NS）。794491 上符号翻转；794495 上即便用 cluster-robust SE 效应也消失。"更深阶更易 split" 的主张仅作为 origin 上勉强显著的小波动存活。
- **Post-correction verdict：** 在 origin 上 WEAKENED（聚类后仍显著但 p ≈ 0.04，而非 p < 1e-60；每单位 OR 基本为 1.02，CI 几近跨越 1，意味着实际可忽略），在额外脑上 OVERTURNED（794491 上方向反转，794495 上 NS）。"置信度翻转"的 surprise 没有依据。

### 5. (Priority 0.253 · Surprise 0.284) "Super-merges" 融合 3+ GT 神经元覆盖的 GT 线缆约为 2-神经元 merge 的约 5 倍，确认最糟糕的 merge 误差是巨型结构。
- **Run：** run-4--ground-truth-error-annotations-revised-version_2026-06-20 · **ID：** 24 · **Belief：** Leaning True → Likely True (0.7083 → 0.8906) · **Direction：** Positive
- **Tested：** 融合 ≥3 个不同 GT 神经元的预测段，每神经元覆盖的 GT 线缆长度是否显著多于 2-神经元-merging 段。
- **Conclusion：** 仅 27 个 merging 段合格（24 个 two-neuron，3 个 super-merge）。2-neuron merge 的中位覆盖线缆为 6.21 mm，而 super-merge 为 35.19 mm（Mann-Whitney U = 0.0，p = 0.0059）。因此 super-merge 随融合神经元数超线性增长（约 11.7 mm/神经元 vs 约 3.1 mm/神经元），确认它们是超大的"巨型"。
- **Caveats：** 样本量极小（n = 3 super-merge）；实现比较的是*总*覆盖线缆而非假设字面提出的每神经元量（审阅者辩称鉴于量级差距这没问题，但这是字面上的偏离）。
- **Reproduction：** DIVERGED (code: revised-loading)
- **Rerun result：** 记录值 "2-Neuron Merges (n=24): Median 6.2054 mm; Super-merges (n=3): Median 35.1873 mm; Mann-Whitney U = 0.0, p = 5.8861e-03"；重跑 "2-Neuron Merges (n=8): Median 6.2054 mm, IQR 12.0549 mm; Super-merges (n=1): Median 35.1873 mm; Mann-Whitney U = 0.0, p = 2.2222e-01"。中位数和 U 统计量完全一致，但重跑只见到这单一脑 pkl（n=8+1=9 个 merge），而记录运行跨多个 `*_add.pkl` 缓存聚合（n=24+3=27 个 merge）。因此 p 翻转显著性（0.006 → 0.222）。分歧源于数据集 SCOPE（单脑重现 vs 多脑原始），而非分析错误。修订：直接 $RERUN_PKL 加载 + numpy>=2。
- **Generalization：** PARTIAL
- **Across datasets：** origin 789202（单脑重跑）：2-neuron n=8 中位 6.21 mm；super-merge n=1 中位 35.19 mm；U=0，p=0.222（不显著；super-merge 太少）；794491：2-neuron n=37 中位 1.36 mm；super-merge n=0（无 super-merge，未运行检验——"Not enough data"）；794495：2-neuron n=24 中位 5.42 mm；super-merge n=2 中位 168.42 mm；U=0，p=6.15e-03（高度显著，super-merge 约大 31 倍）。在存在 super-merge 处方向一致（origin 和 794495 都有 super-merge 比 2-neuron 大 5–30 倍），但 794491 无 super-merge，且 origin 单脑重跑本身不显著——只有 794495 达到显著。结构性主张（"super-merge 是巨型"）在它们存在时得到支持，但在某些脑上稀疏/缺失。
- **Verdict：** MAJOR
- **Test：** 对总 GT 线缆长度的 Mann-Whitney U（n=24 vs n=3），U=0.0，p=5.89e-03——对小而偏斜的长度分布正确地选择了非参数检验，但一臂 n=3 时功效基本为零，且 U=0 只是统计量的下限。
- **Statistical issues：**
  - **严重功效不足：** n=3 super-merge。在最小可能的 U 统计量（0.0）下，24 vs 3 可达到的最小双侧 p 恰为 0.0059，正是报告值。"高度显著"的措辞是数值下限效应，而非效应量的展示。
  - **偏离假设：** 假设说"每神经元更多线缆长度"（≈ 35.19/3 vs 6.21/2 = 11.7 vs 3.1 mm/神经元），但代码比较的是*总*覆盖线缆。按实现方式，super-merge 在数学上必然覆盖更多总线缆，仅因它融合了更多神经元。
  - **重现分歧：** 单脑重跑 n=8+1 且 p 翻转为 0.222；记录的 p=0.006 来自跨多个 `*_add.pkl` 缓存的聚合（该分析对数据集范围敏感）。
- **Logic issues：**
  - "super-merge 随融合神经元数超线性增长"的主张是从 n=3 vs n=24 的中位数推出的——是算术比值，而非拟合的标度律；未估计标度指数。
  - 在 794491 上 super-merge 缺失（n=0），因此结构性主张在每个脑上连简单的存在性都不成立——它取决于 merge 过程是否产生它们。
- **Downgrade based on rerun/extrapolation：** Yes——PARTIAL 推广（一个脑无 super-merge；origin 重跑不显著）。该主张应重述为"在 super-merge 存在处它们比 2-神经元 merge 大一个数量级"，而非作为普适规律。
- **Corrected test：** (a) 对每神经元线缆（总量/融合神经元数）而非总线缆做 Mann-Whitney U——匹配假设字面措辞"每神经元更多线缆长度"；(b) 带 bootstrap 95% CI 的 Cliff's delta，作为不依赖下限效应 U = 0 统计量的效应量度量。相同源数据，相同分组定义，只是在假设实际指明的单元上。
- **Corrected result：** origin 789202（n = 8 two-neuron，n = 1 super）：每神经元中位 3.10 mm vs 11.73 mm；U = 0，p = 0.222（NS）。Cliff's delta = +1.00 [1.00, 1.00] 但因 n = 1 super-merge 而退化——CI 人为地紧，因为只有一个 super-merge 可重采样。并列对比：原始记录的 "U = 0, p = 5.886e-03 on n = 24 + 3 across many brains" 变为 "U = 0, p = 0.222 on n = 8 + 1 within origin"——完全相同的下限效应机制。
- **Corrected generalization：** PARTIAL——origin 789202：n = 8 + 1，Cliff's delta = +1.00（退化，无检验），MW p = 0.22；794491：n = 37 + 0（无 super-merge 存在，检验无法运行）；794495：n = 24 + 2，每神经元 Cliff's delta = +0.9167，95% CI [+0.7500, +1.0000]，效应非常大，MW p = 6e-03（显著）。"super-merge 每神经元巨大"的主张仅在 794495 上稳健成立；在 794491 上无法确认；在 origin 上功效不足。
- **Post-correction verdict：** WEAKENED——当在字面的"每神经元线缆"层面运行检验（而非部分同义反复的原始"总线缆"）时，效应方向不变但证据大幅减弱。原始显著性是多脑聚合伪影加上 U 统计量下限效应；在逐脑校正检验下，1/3 脑无 super-merge，1/3 仅有一个，仅 1/3 支持显著效应。结构性主张（"在 super-merge 存在处它们比 2-神经元 merge 大一个数量级"）存活，但"高度显著"的措辞不成立。

### 6. (Priority 0.253 · Surprise 0.284) Omission 误差在拓扑分支点处的发生频率约为线性线缆节点处的 4 倍。
- **Run：** run-4--ground-truth-error-annotations-revised-version_2026-06-20 · **ID：** 32 · **Belief：** Leaning True → Likely True (0.7083 → 0.8906) · **Direction：** Positive
- **Tested：** omission 误差率在分支节点（degree > 2）与线性节点（degree = 2）之间是否不同，以将局部拓扑复杂性牵连为 dropout 的原因。
- **Conclusion：** 分支节点 omit 率 ≈ 11.95% vs 线性节点 omit 率 ≈ 2.76%，χ² = 1567.69，p < 0.0001。模型确实在复杂连接处丢弃碎片，因此校对者应针对分支节点寻找缺失的线缆。
- **Caveats：** 未记录；效应量和显著性都很大。
- **Reproduction：** REPRODUCED (code: revised-loading)
- **Rerun result：** 记录值 "Branch 606/4466/5072 (11.9479%); Linear 38610/1360197/1398807 (2.7602%); Chi-square = 1567.6878, p = 0.0000e+00"；重跑 "Branch 606/4466/5072 (11.9479%); Linear 38610/1360197/1398807 (2.7602%); Chi-square = 1567.6878, p = 0.0000e+00" → 完全一致。修订：直接 $RERUN_PKL 加载 + numpy>=2。
- **Generalization：** GENERALIZES
- **Across datasets：** origin 789202：Branch 11.95% vs Linear 2.76%（比值 4.3×），χ²=1567.69，p≈0；794491：Branch 7.54% vs Linear 3.41%（比值 2.2×），χ²=193.63，p=5.13e-44；794495：Branch 6.52% vs Linear 1.72%（比值 3.8×），χ²=974.86，p=5.24e-214。在全部三个脑上分支节点 omit 率都显著高于线性节点 omit 率（比值 2.2×–4.3×，p ≪ 0.001）。
- **Verdict：** MINOR
- **Test：** 对节点类型（分支 vs 线性）× omit（是/否）的 Chi-square 独立性检验，χ²=1567.69，p≈0，n=1,403,879 节点——适用于 2×2 列联表，且相对效应（4.3×）远超任何可设想的原假设。
- **Statistical issues：**
  - **违反独立性：** 同一神经元内的节点非 i.i.d.；聚类会收紧 SE 而非放松，但原则上 cluster-robust 检验更诚实。鉴于相对效应（4.3× 比值），无现实的聚类校正会推翻该结果。
  - 分支节点（5,072）远比线性节点（1.4M）稀少，但期望单元计数仍非常大，故 χ² 表现良好。
- **Logic issues：**
  - 结论将关系因果化表述（"复杂分支结构导致 dropout"）——这是观察性的，可能反映例如分支处的信号稀疏而非拓扑本身。不过"校对者应针对分支节点"的可操作主张确实直接成立。
- **Downgrade based on rerun/extrapolation：** No——在全部三个脑上 GENERALIZES，方向相同且效应量大。

### 7. (Priority 0.253 · Surprise 0.284) 角度对齐是一个强的 split-vs-假候选判别器（真延续约 153°，假约 90°，AUC 0.93）。
- **Run：** run-4--ground-truth-error-annotations-revised-version_2026-06-20 · **ID：** 33 · **Belief：** Leaning True → Likely True (0.7083 → 0.8906) · **Direction：** Positive
- **Tested：** 跨越一个 split 的两条相邻 GT 边之间的角度，是否显著比到来自不同神经元的邻近假候选的角度更接近 180°，从而验证"方向惯性"作为桥接启发式。
- **Conclusion：** 在 13,582 个 split 节点配置中，真延续平均角度为 152.96°，假候选为 90.17°（KS 统计量 0.7460，p ≈ 0；ROC AUC 0.9322）。因此方向惯性是 agentic 校对器桥接 split 的强而可靠的局部启发式。
- **Caveats：** 未记录；AUC 高且分布明显分离。
- **Reproduction：** REPRODUCED (code: revised-loading)
- **Rerun result：** 记录值 "Analyzed 13582 split node configurations; mean true 152.96°, mean false 90.17°; KS 0.7460 p≈0; ROC AUC 0.9322"；重跑 "Analyzed 13582 split node configurations; mean true 152.96°, mean false 90.17°; KS 0.7460 p≈0; ROC AUC 0.9322" → 完全一致。修订：直接 $RERUN_PKL 加载 + numpy>=2 + 内置 sklearn。
- **Generalization：** GENERALIZES
- **Across datasets：** origin 789202：n=13,582 splits；mean true 152.96°，false 90.17°；KS=0.7460 p≈0；AUC=0.9322。794491：n=15,637；mean true 153.49°，false 90.15°；KS=0.7335 p≈0；AUC=0.9281。794495：n=15,929；mean true 155.17°，false 90.08°；KS=0.7559 p≈0；AUC=0.9365。方向惯性在全部三个脑上都把真延续（约 153°–155°）与假候选（约 90°）分开，AUC 全程 ≈ 0.93。
- **Verdict：** OK
- **Test：** 在 n=13,582 个 split 配置上比较真延续角度 vs 假候选角度的 KS 检验（KS=0.7460，p≈0）加 ROC-AUC（0.9322）——两者都是非参数且适用于有界角度分布。
- **Statistical issues：**
  - **非独立性：** 来自同一神经元/区域的多个 split 配置非 i.i.d.；因此 KS p 值偏乐观，但 AUC = 0.9322 是群体层面的判别量，不敏感于该问题。
  - KS 效应量 0.7460 和判别 AUC 0.93 按任何标准都很大，且在两个额外脑上重现，故推断对聚类担忧稳健。
- **Logic issues：**
  - "假候选"被操作性地定义为来自不同神经元的邻近边——结论的可操作性取决于推断时同样的操作性定义可用。
- **Downgrade based on rerun/extrapolation：** No——GENERALIZES（在全部三个脑上 AUC 0.928–0.937，平均角度 152.96°–155.17° vs 90.08°–90.17°）。

### 8. (Priority 0.253 · Surprise 0.284) 被遗漏的线缆在系统上比正确重建的线缆更靠近 merge 位点。
- **Run：** run-4--ground-truth-error-annotations-revised-version_2026-06-20 · **ID：** 36 · **Belief：** Leaning True → Likely True (0.7083 → 0.8906) · **Direction：** Positive
- **Tested：** omit 误差是否在空间上与 merge 误差共聚，表明分割模型在融合主导结构时牺牲了相邻的细突起。
- **Conclusion：** 在 49,295 个 omit 节点与一个长度匹配的 49,295 个正确节点随机样本上，到最近 67 个 merge 位点的中位距离为 1,812.73 µm（omit）vs 1,959.88 µm（correct），Mann-Whitney U p = 1.60e-66——高度显著。
- **Caveats：** 绝对距离差距（约 147 µm / 约 1,900 µm）在效应量上不大；显著性主要由巨大样本量驱动。此外只有 67 个 merge 位点锚定该比较。
- **Reproduction：** REPRODUCED (code: revised-loading)
- **Rerun result：** 记录值 "67 merge sites, 49295 omit, 49295 correct; median omit 1812.73 µm, median correct 1959.88 µm; Mann-Whitney U = 1138194675.0, p = 1.6036e-66"；重跑 "67 merge sites, 49295 omit, 49295 correct; median omit 1812.73 µm, median correct 1959.88 µm; Mann-Whitney U = 1138194675.0, p = 1.6036e-66" → 完全一致。修订：直接 $RERUN_PKL 加载 + numpy>=2。
- **Generalization：** DOES-NOT-GENERALIZE
- **Across datasets：** origin 789202：67 merge 位点，49,295 omit vs 49,295 correct，中位 omit 1812.73 µm < correct 1959.88 µm，Mann-Whitney 单侧 p=1.60e-66（omit 更靠近 merge）；794491：86 merge，29,033 omit / 29,033 correct，中位 omit 1103.24 µm > correct 842.11 µm，p=1.0000（omit 比 correct 更*远*——方向翻转）；794495：105 merge，各 32,788，中位 omit 1422.53 µm < correct 1670.94 µm，p=9.78e-226（omit 更近，与 origin 一致）。794491 完全反转了空间关系，因此"omit 在 merge 附近聚集"的主张在该脑上不成立。
- **Verdict：** CRITICAL
- **Test：** 对 omit vs 长度匹配 correct 节点的到最近 merge 距离做单侧 Mann-Whitney U，各 n=49,295，p=1.60e-66——该检验适用于偏斜距离数据，但 n 被人为放大，因为每个*节点*被当作 i.i.d.，而距离仅来源于 67 个 merge 位点且沿线缆密集聚类。
- **Statistical issues：**
  - **非独立性：** 49,295 个 omit 节点沿线缆分布，因此到最近 67 个 merge 位点的距离高度空间相关；有效 n 更接近数千。因此 1.6e-66 的 p 值是对证据的巨大夸大。
  - **效应量微小：** 约 1,900 µm 基线上的约 147 µm 差距（≈ 7.7% 相对偏移）。显著性由样本量驱动；实际的"共聚"主张很弱。
  - **锚集稀疏：** 仅 67 个 merge 位点作参照；置换所选的 67 个位点会明显移动距离分布。未报告对 merge 位点的 bootstrap。
- **Logic issues：**
  - 结论"分割模型在融合主导结构时牺牲了相邻的细突起"是一个*机制性*主张，空间关联无法支持——它同样可能反映两种误差类型因不相关原因在致密神经毡中共现。
  - 推广在 794491 上反转了方向（omit 中位 1103 µm > correct 842 µm，单侧 p=1.0），表明该空间关系不是模型的稳健属性。
- **Downgrade based on rerun/extrapolation：** Yes——DOES-NOT-GENERALIZE。标题发现是脑特异的（在 2 个脑上确认，在 1 个上反转），不应作为分割模型的普适属性呈现。
- **Corrected test：** (a) 按神经元做 cluster-bootstrap（有放回重采样神经元），对到最近 merge 位点的 median(omit) − median(correct) 距离，给出 cluster-robust 95% CI 和符号翻转 p 值；(b) 带 cluster-bootstrap CI 的 Cliff's delta，作为非参数效应量估计；(c) merge 位点 bootstrap（有放回重采样 67 个 merge 位点），以界定小锚集带来的变异性。三者都在相同的 omit / 长度匹配-correct 距离分布上。
- **Corrected result：** origin 789202——观测 median(omit − correct) = −145.01 um。cluster-bootstrap 95% CI = [−401.82, +91.98] um（CI 跨越零），cluster-bootstrap p = 0.2700。Cliff's delta 的 cluster-bootstrap CI = [−0.1623, +0.0496]（也穿过零）。差距的 merge 位点 bootstrap CI = [−346.73, +269.13] um。并列对比：原始 "median diff −147 um, U = 1.138e9, p = 1.60e-66" → 校正后给出相同点估计，但 cluster-bootstrap 95% CI 包含零且 cluster p = 0.27（NS）。"高度显著"的 p 是把 49,295 个空间相关节点当作 i.i.d. 造成的 >60 个数量级的伪影。
- **Corrected generalization：** DOES-NOT-GENERALIZE——origin 789202：gap = −145 um，cluster-bootstrap p = 0.27（NS）；794491：gap = +261.93 um（符号反转），cluster-bootstrap CI [+86.23, +445.34]，p ≈ 0（显著但方向相反）；794495：gap = −248.92 um，cluster-bootstrap CI [−517.34, +24.45]，p = 0.082（NS）。origin 为 NS，794491 在错误方向上显著，794495 是唯一与原始标题相符者但 p = 0.08。
- **Post-correction verdict：** OVERTURNED——一旦检验尊重约 49k 节点的神经元内聚类，origin 的 "p = 1.6e-66" 坍缩为 "p = 0.27 且 CI 包含零"，且一个额外脑以聚类显著证据翻转方向。原始 "omit cable is systematically closer to merge sites" 是样本量伪影叠加脑特异结构。

### 9. (Priority 0.253 · Surprise 0.284) Split 误差在空间上聚类：split 边在 30 µm 内的 split 邻居数约为 correct 边的 10 倍。
- **Run：** run-4--ground-truth-error-annotations-revised-version_2026-06-20 · **ID：** 37 · **Belief：** Leaning True → Likely True (0.7083 → 0.8906) · **Direction：** Positive
- **Tested：** 在 30 µm 半径内，split 边是否比匹配的正确重建边有显著更多的邻近 split 邻居，从而识别局部化的"误差区"。
- **Conclusion：** split 边在 30 µm 内平均有 0.97 个其他 split 邻居，匹配 correct 边为 0.10（Mann-Whitney U = 34,661,344.5，p < 0.0001）。split 伪影区域性聚集，支持把审阅者引导至热点的批量校正工作流。
- **Caveats：** 未记录；量级和显著性都很强。
- **Reproduction：** REPRODUCED (code: revised-loading)
- **Rerun result：** 记录值 "Number of split edges: 6805, correct sampled: 6805; mean split neighbors 0.97 vs 0.10 within 30µm; Mann-Whitney U = 34661344.5, p = 0.00e+00"；重跑 "Number of split edges: 6805, correct sampled: 6805; mean split neighbors 0.97 vs 0.10 within 30µm; Mann-Whitney U = 34661344.5, p = 0.00e+00" → 完全一致。修订：直接 $RERUN_PKL 加载 + numpy>=2。
- **Generalization：** GENERALIZES
- **Across datasets：** origin 789202：n_split=6805，平均 split 邻居 0.97 vs correct 0.10（约 9.7×），U=3.47e7，p≈0；794491：n=7847，1.39 vs 0.28（约 5.0×），U=4.68e7，p≈0；794495：n=7988，1.26 vs 0.11（约 11.5×），U=5.11e7，p≈0。在每个脑上 split 边在 30 µm 内的 split 邻居数都比 correct 边多 5×–12×，全部显著。
- **Verdict：** MINOR
- **Test：** 对每边 30 µm 内 split 邻居计数的 Mann-Whitney U，n=6,805 split vs 6,805 匹配 correct，U=3.47e7，p≈0——对带零膨胀的计数数据，非参数是正确的。
- **Statistical issues：**
  - **非独立性：** "split 邻居"按构造相关——若边 A 把边 B 计为邻居，边 B 也把 A 计为邻居。因此 Mann-Whitney p 偏乐观，但约 10× 比值（0.97 vs 0.10）足够大，无任何合理校正会翻转结论。
  - **定义按设计是循环的：** 询问 split 是否与其他 split 聚类本质上是在测量局部自相关。这对"误差区"工程主张可以，但作为"发现"是同义反复。
- **Logic issues：**
  - "支持把审阅者引导至热点的批量校正工作流"的推论根据充分；无越界。
- **Downgrade based on rerun/extrapolation：** No——在全部三个脑上 GENERALIZES，比值 5×–12×。

### 10. (Priority 0.253 · Surprise 0.284) 在碎片图上的启发式 A*（角度 + 半径惩罚）修复了 86% 的 split 边而不诱发 merge。
- **Run：** run-4--ground-truth-error-annotations-revised-version_2026-06-20 · **ID：** 39 · **Belief：** Leaning True → Likely True (0.7083 → 0.8906) · **Direction：** Positive
- **Tested：** 直接在 U-Net 碎片图上的基于图的路径搜索，对轨迹偏差 >45° 和突然半径变化施加惩罚，是否能在不产生 merge 误差的情况下重连 >40% 的 split 边。
- **Conclusion：** 在 6,805 个目标 split 上，agent 为其中 5,881 个找到了有效（无 merge）路径——86.42% 成功率，是 40% 阈值的两倍。端到端 Edge Accuracy 从 78.71% 提升到 79.13%（净增 +0.42%）。
- **Caveats：** 尽管每 split 成功率高，全数据集准确率增益很小（+0.42%）；merge 和 omission 仍未解决。
- **Reproduction：** REPRODUCED (code: revised-loading)
- **Rerun result：** 记录值 "Target split edges: 6805; Successfully resolved: 5881; Success rate 86.42%; Original Edge Accuracy 78.71% → New 79.13% (+0.42%)"；重跑 "Target split edges: 6805; Successfully resolved: 5881; Success rate 86.42%; Original Edge Accuracy 78.71% → New 79.13% (+0.42%)" → 完全一致。修订：直接 $RERUN_PKL 加载 + numpy>=2。
- **Generalization：** GENERALIZES
- **Across datasets：** origin 789202：target=6805，resolved=5881，success=86.42%；baseline EA 78.71% → 79.13%（+0.42%）。794491：target=7847，resolved=6648，success=84.72%；EA 74.77% → 75.95%（+1.18%）。794495：target=7988，resolved=7166，success=89.71%；EA 68.55% → 69.07%（+0.53%）。每 split 成功率一致为 84.7%–89.7%（远高于 40% 阈值），EA 增益在全部三个脑上都小但为正。
- **Verdict：** MINOR
- **Test：** 无——这是工程基准（6,805 个 split 边上 86.42% 成功率，无统计检验），而非假设检验。报告为描述性。
- **Statistical issues：**
  - **无不确定性量化：** 成功率无置信区间，无对神经元的 bootstrap。若少数神经元主导计数，每 split 成功率原则上可能被抬高。
  - **每边 ≠ 每神经元：** 若少数大 GT 神经元贡献了大部分 split，"86%" 是被它们主导的边层级率，而非神经元层级成功率。
  - **阈值（>40%）宽松：** 先验阈值是任意的；通过它并不验证启发式，只是越过了一个低门槛。
- **Logic issues：**
  - +0.42% 的 EA 增益很小但结论公允地指出了这一点，无越界。
  - "不诱发 merge"依赖 GT——部署时没有 GT，同样的启发式无法保证无 merge；这更像是可行性上界，而非部署主张。
- **Downgrade based on rerun/extrapolation：** No——GENERALIZES（三个脑上成功率 84.7%–89.7%，EA 增益一致为正）。

### 11. (Priority 0.253 · Surprise 0.284) Merging 段表现为"巨型"组件——平均线缆长度约为非 merging 段的 30 倍。
- **Run：** run-4--ground-truth-error-annotations-revised-version_2026-06-20 · **ID：** 43 · **Belief：** Leaning True → Likely True (0.7083 → 0.8906) · **Direction：** Positive
- **Tested：** 被标记的 merging 段所覆盖的总 GT 线缆长度是否指数级地大于非 merging 段，表明 merge 是失控的过度生长标签而非局部噪点。
- **Conclusion：** 64 个 merging 段平均长度约 15,449 µm（中位约 3,099 µm），而 8,273 个非 merging 段平均约 532 µm（中位约 102 µm）。对 log 变换长度的 Welch's t 检验：t = 16.54，p = 7.32e-25。Merge 误差由庞大的失控组件主导。
- **Caveats：** 未记录；效应量巨大。
- **Reproduction：** REPRODUCED (code: revised-loading)
- **Rerun result：** 记录值 "Number of merging segments: 64; non-merging: 8273; Merging mean 15448.96 µm, median 3099.27 µm; Non-merging mean 531.76 µm, median 101.74 µm; Welch's t=16.5444, p=7.3232e-25"；重跑 "Number of merging segments: 64; non-merging: 8273; Merging mean 15448.96 µm, median 3099.27 µm; Non-merging mean 531.76 µm, median 101.74 µm; Welch's t=16.5444, p=7.3232e-25" → 完全一致。修订：$RERUN_PKL 替换了一个在沙箱中返回空的 `os.walk` 搜索 + numpy>=2。
- **Generalization：** GENERALIZES
- **Across datasets：** origin 789202：64 merging vs 8273 non-merging，mean 15449 µm vs 532 µm（约 29×）；t=16.54，p=7.32e-25。794491：98 vs 8544，mean 4457 µm vs 194 µm（约 23×）；t=29.03，p≈0。794495：98 vs 7998，mean 16277 µm vs 477 µm（约 34×）；t=24.02，p≈0。Merging 段在每个脑上都比非 merging 长 23×–34×，log-Welch t 检验全程极度显著。
- **Verdict：** OK
- **Test：** 对 log 变换线缆长度的 Welch 双样本 t 检验（t=16.54，p=7.32e-25，n=64 vs 8,273）——log 变换处理重尾线缆长度分布，Welch 校正适合方差不等。
- **Statistical issues：**
  - **样本量不对称：** 64 vs 8,273 严重不平衡，但 Welch t 容纳这一点，约 30× 的均值比值稳健。
  - **同义反复意味：** "merging 段更大"部分是定义性的——融合多个神经元的段必然跨越它们，因此必然更长。该假设因量级（约 30×，而非约 2×）而更具信息量。
  - 就多重检验而言，这显然能通过任何 FDR。
- **Logic issues：**
  - "由庞大的过度生长标签而非局部噪点驱动"得到量级支持；无越界。
- **Downgrade based on rerun/extrapolation：** No——GENERALIZES（全部三个脑上比值 23×–34×）。

### 12. (Priority 0.253 · Surprise 0.284) Omit 误差在成像体积的极端 Z 深度处的发生概率超过中央深度处的 2 倍。
- **Run：** run-4--ground-truth-error-annotations-revised-version_2026-06-20 · **ID：** 45 · **Belief：** Leaning True → Likely True (0.7083 → 0.8906) · **Direction：** Positive
- **Tested：** omit 误差率在 Z 坐标的顶/底 10% 处是否高于中央 20% 处，可归因于光学衰减/散射。
- **Conclusion：** 极端-Z omit 率 4.44%（2,283/51,369）vs 中央-Z 1.94%（11,342/585,909）。控制脑 ID 的 Cochran–Mantel–Haenszel 检验给出 pooled OR = 2.3561，p ≈ 0，验证了 omission 在轴向极端处特有的空间各向异性。
- **Caveats：** 这一正向 Z 极端结果与负向 Z 朝向结果（#2、#3）并存——该效应关乎深度极端（边界伪影），而非沿线缆的朝向驱动各向异性。
- **Reproduction：** REPRODUCED (code: revised-loading)
- **Rerun result：** 记录值 "Extreme 2283/51369 (4.44%); Center 11342/585909 (1.94%); Pooled Odds Ratio (Extreme vs Center) = 2.3561, p = 0.0000e+00"；重跑 "Extreme 2283/51369 (4.44%); Center 11342/585909 (1.94%); Pooled Odds Ratio = 2.3561, p = 0.0000e+00" → 完全一致。修订：直接 $RERUN_PKL 加载 + numpy>=2。
- **Generalization：** PARTIAL
- **Across datasets：** origin 789202：Extreme 4.44% vs Center 1.94%，pooled OR=2.3561，p≈0（极端 Z 显著更易出错）。794491：Extreme 4.00% vs Center 3.94%，pooled OR=1.0149，p=0.7996（无显著效应——极端 Z 与中央表现一致）。794495：Extreme 5.37% vs Center 1.76%，pooled OR=3.1672，p≈0（高于 origin，强成立）。"极端 Z 更差"的效应在 789202 和 794495 上稳健（OR > 2.3），但在 794491 上缺失（OR ≈ 1，p = 0.80）。
- **Verdict：** MAJOR
- **Test：** "控制脑 ID" 的 Cochran-Mantel-Haenszel 检验，pooled OR=2.3561，p≈0，n=637,278 节点——CMH 适用于分层 2×2 表，但在单脑重现中实际只有一个层，因此"控制脑 ID"的措辞具有误导性。
- **Statistical issues：**
  - **分层被框定为多脑但执行为单脑：** 记录代码说"控制脑变异"，但重现确认这是在单脑 pkl 上计算的；分层变量"脑 ID"对所报结果是无操作。pooled OR 在该数据集上坍缩为普通 OR。
  - **违反独立性：** 同一神经元/区域内的节点非 i.i.d.；CMH 对分层计数把每个节点当作独立。p≈0 被夸大。
  - **挑选式分箱：** "极端 = 顶/底 10%，中央 = 中央 20%" 是特定分箱定义；其他分箱选择可能削弱效应。未报告对分箱选择的敏感性。
- **Logic issues：**
  - **机制越界：** 结论将效应归因于"沿成像轴的光学衰减或散射"——检验只显示空间关联，而非机制。许多替代原因（边界处标注精力、边缘附近 GT 密度、体积边界处神经突截断）都会产生相同信号。
  - PARTIAL 推广（一个脑显示 OR=1.01，p=0.80）实质削弱了"光学衰减"说法，因为跨脑使用相同成像模态；成像物理解释应当推广，但它没有。
- **Downgrade based on rerun/extrapolation：** Yes——PARTIAL 推广（794491 上效应缺失）。"极端 Z 更差"的主张是数据集特定的，而非普适的成像物理后果。
- **Corrected test：** 以神经元为 cluster id 的 GEE logistic 回归，由 is_extreme（1 = 顶/底 10% Z，0 = 中央 20% Z）预测 is_omit。这用相同的 OR 但 cluster-robust SE 替换了原始 CMH（其在单脑 pkl 上坍缩为普通 OR 且忽略神经元内聚类）。同时通过重采样神经元给出 OR 的 cluster-bootstrap CI。
- **Corrected result：** origin 789202——描述统计不变（637,278 节点上 extreme 4.44% vs centre 1.94%）。GEE OR(Extreme vs Centre) = 2.3561（点估计相同），95% CI [0.4558, 12.1801]，p = 0.31；OR 的 cluster-bootstrap CI = [0.5046, 22.9182]。并列对比：原始 "CMH pooled OR = 2.3561, p ≈ 0 on 637k nodes" → 校正后 "OR = 2.3561 with cluster-robust 95% CI = [0.46, 12.18], p = 0.31"。一旦认识到将 637k 节点聚类的 12 个神经元，p 值丧失其全部 "≈ 0" 显著性。
- **Corrected generalization：** WEAKENED——origin 789202：OR = 2.36 [0.46, 12.18]，p = 0.31（NS）；794491：OR = 1.01 [0.12, 8.80]，p = 0.99（NS，CI 跨越 2 个数量级）；794495：OR = 3.17 [2.11, 4.76]，p = 2.8e-08（显著，大）。"极端 Z 更差"的效应仅在 794495 上经聚类后存活；在 origin 和 794491 上 CI 包住 1 且范围很宽。
- **Post-correction verdict：** WEAKENED——原始 "pooled OR = 2.36, p ≈ 0" 是把仅 12 个簇内的全部 637k 节点当作独立观测驱动的；一旦尊重聚类，相同 OR 点估计的 CI 为 [0.46, 12.18]，origin 上 p = 0.31。效应在方向上仍一致（2/3 脑上点 OR > 1）且在 794495 上稳健显著，但原始推广主张（"成像物理效应"）无法维持：一旦考虑非独立性，origin 或 794491 上效应不显著。

### 13. (Priority 0.253 · Surprise 0.284) 短 omission 间隙通常是单个预测段内部的 dropout；长间隙是真实的碎片边界。
- **Run：** run-4--ground-truth-error-annotations-revised-version_2026-06-20 · **ID：** 58 · **Belief：** Leaning True → Likely True (0.7083 → 0.8906) · **Direction：** Positive
- **Tested：** 两侧节点共享同一预测段（bridged）的连续 omit 路径是否比两侧 ID 不匹配（broken）者更短。
- **Conclusion：** 307 条 bridged omit 路径（均值 18.71 µm，中位 13.55 µm）vs 4,298 条 broken omit 路径（均值 42.54 µm，中位 20.16 µm）；Mann-Whitney U p = 1.95e-19。短 omit 在很大程度上是内部网络 dropout，是安全自动填充的良好目标。
- **Caveats：** bridged 样本（n = 307）远小于 broken（n = 4,298），但分离清晰。
- **Reproduction：** REPRODUCED (code: revised-loading)
- **Rerun result：** 记录值 "Total Bridged 307 (mean 18.71 µm, median 13.55 µm); Total Broken 4298 (mean 42.54 µm, median 20.16 µm); Mann-Whitney U = 458554.0, p = 1.9488e-19"；重跑 "Total Bridged 307 (mean 18.71 µm, median 13.55 µm); Total Broken 4298 (mean 42.54 µm, median 20.16 µm); Mann-Whitney U = 458554.0, p = 1.9488e-19" → 完全一致。修订：直接 $RERUN_PKL 加载 + numpy>=2。
- **Generalization：** GENERALIZES
- **Across datasets：** origin 789202：bridged n=307 中位 13.55 µm vs broken n=4298 中位 20.16 µm，Mann-Whitney U=458554，p=1.95e-19。794491：bridged n=219 中位 10.99 µm vs broken n=4353 中位 15.23 µm，U=365509，p=2.75e-09。794495：bridged n=330 中位 12.17 µm vs broken n=3815 中位 16.24 µm，U=483296，p=1.20e-12。bridged-比-broken-更短在全部三个脑上成立；中位数和显著性都对齐。
- **Verdict：** MINOR
- **Test：** 对连续 omit 路径长度的 Mann-Whitney U，n=307 bridged vs 4,298 broken，U=458,554，p=1.95e-19——对重尾长度是正确的非参数选择。
- **Statistical issues：**
  - **类别不平衡对 Mann-Whitney U 无碍：** 该检验干净地处理 n=307 vs 4,298。
  - **定义耦合：** "bridged" 要求在同一预测段上有足够的非 omit 邻居；"broken" 既包括真实终止也包括结构性断裂。因此类别标签已部分编码了路径长度（长的 broken 跨度更不可能偶然被同一段桥接），抬高了表观效应。
  - 中位数比值（13.55 vs 20.16 µm）适中（约 1.5×）；显著性部分来自总数 n=4,605。
- **Logic issues：**
  - "短 omit 在很大程度上是内部 dropout，是安全自动填充的良好目标"作为类别判别主张得到支持。该建议在操作上合理。
- **Downgrade based on rerun/extrapolation：** No——在全部三个脑上以相同方向 GENERALIZES。

### 14. (Priority 0.253 · Surprise 0.284) Split 边的局部迂曲度高于正确重建边，尽管相关性很小。
- **Run：** run-4--ground-truth-error-annotations-revised-version_2026-06-20 · **ID：** 59 · **Belief：** Leaning True → Likely True (0.7083 → 0.8906) · **Direction：** Positive
- **Tested：** 高局部迂曲度（在 10 边滑动窗口上的路径长度 / 欧氏距离）是否增加 split 误差的可能性。
- **Conclusion：** 6,611 split vs 1,091,075 correct 边：中位迂曲度 1.1115 vs 1.0764（均值 1.2163 vs 1.1175）。Mann-Whitney U p ≈ 0；点二列 r = 0.0335（p 也 ≈ 0）。迂曲度在统计上与 split 关联，但效应量级很小。
- **Caveats：** 相关性（r = 0.0335）极小——显著性来自样本量；单凭迂曲度是弱的每边预测器。
- **Reproduction：** REPRODUCED (code: revised-loading)
- **Rerun result：** 记录值 "Split 6611, Correct 1091075; Median Tortuosity (Split) 1.1115, (Correct) 1.0764; Mean (Split) 1.2163, (Correct) 1.1175; Mann-Whitney U=4.5781e+09, p=0; Point-biserial r=0.0335, p=0"；重跑 "Split 6611, Correct 1091075; Median Tortuosity (Split) 1.1115, (Correct) 1.0764; Mean (Split) 1.2163, (Correct) 1.1175; Mann-Whitney U=4.5781e+09, p=0.0000e+00; Point-biserial r=0.0335, p=0.0000e+00" → 完全一致。修订：直接 $RERUN_PKL 加载 + numpy>=2。
- **Generalization：** GENERALIZES
- **Across datasets：** origin 789202：split n=6611，correct n=1091075；中位 1.1115 vs 1.0764；r=0.0335，p≈0。794491：split n=7485，correct n=408632；中位 1.0860 vs 1.0597；r=0.0607，p≈0。794495：split n=7662，correct n=912147；中位 1.0718 vs 1.0555；r=0.0253，p≈0。方向相同（split > correct），效应量适中相同（r 0.025–0.061），跨所有脑 p 均 ≈ 0。
- **Verdict：** MAJOR
- **Test：** Mann-Whitney U + 点二列相关，n=6,611 split vs 1,091,075 correct，U=4.58e9，p≈0；r=0.0335，p≈0——两个检验原则上都适用，但结论完全由巨大 n 驱动。
- **Statistical issues：**
  - **显著性由样本量主导：** 在 n ≈ 1.1M 下，点二列 r = 0.0335 对应约 ~35 的 z 分；该结果不令人意外，但效应基本是噪声（r² ≈ 0.001——迂曲度仅解释 split-vs-correct 方差的 0.1%）。
  - **非独立性：** 同一神经元内及沿同一线缆的边空间相关；有效 n 远小于 1.1M。所报 p≈0 极大夸大了证据。
  - **0.035 的迂曲度中位数差距**在生物学上微不足道；这是教科书式的"大 n 微效应"显著性陷阱。
- **Logic issues：**
  - 论文的文字结论"迂曲度在统计上与 split 关联，但效应量级很小"是诚实的；报告中的 caveat 承认了微小的 r。因此推断没有越界——但发现循环仍因显著性而将其标记为对"强"结构属性的确认。
  - 无预测价值：r=0.03 意味着仅基于迂曲度的分类器在实践中无用。
- **Downgrade based on rerun/extrapolation：** No——GENERALIZES，但在全部三个脑上效应量仍微小（r=0.025–0.061）。该假设应呈现为"方向一致但实际可忽略"。
- **Corrected test：** (a) 对迂曲度计算 split-vs-correct 的 Cliff's delta（rank-biserial），通过重采样*神经元*给出 cluster-bootstrap 95% CI——把一个"由 1.1M 驱动显著性"的点 r 统计量转化为带 CI 且尊重神经元内相关的效应量。(b) 对神经元层级中位迂曲度（split − correct）的逐神经元配对 Wilcoxon，是边非独立时恰当的聚合检验。
- **Corrected result：** origin 789202——Cliff's delta = +0.2562，cluster-bootstrap 95% CI [+0.2104, +0.3324]（cluster p ≈ 0）；逐神经元配对 Wilcoxon W = 78，p = 2.44e-04，n_pairs = 12，对每神经元 median(split − correct) 的 bootstrap 95% CI = [+0.02389, +0.05565]。并列对比：原始 "U = 4.58e9, p ≈ 0; point-biserial r = 0.0335, p ≈ 0"——校正后的 Cliff's delta 为 0.26，按 Vargha-Delaney 标准是一个 SMALL 但真实的效应（|delta| 0.147-0.33 = "small"），逐神经元配对 Wilcoxon 在正确分析单元上确认了相同方向。原始 r = 0.03 是因为在错误单元（每边）上计算而显得误导性地小；按簇看效应是小但有意义的。
- **Corrected generalization：** GENERALIZES——origin 789202：Cliff's delta = +0.26 [+0.21, +0.33]；794491：Cliff's delta = +0.28 [+0.23, +0.34]，逐神经元 Wilcoxon p = 1.95e-03；794495：Cliff's delta = +0.21 [+0.14, +0.24]，逐神经元 Wilcoxon p = 1.91e-06。方向和量级在全部三个脑上一致。
- **Post-correction verdict：** WEAKENED——方向确认，效应量远小于原本 p ≈ 0 所暗示但真实（Cliff's delta 在 0.21–0.28 范围，按 Vargha-Delaney 为 "small" 效应）。原始措辞在统计上有误导（在 1.1M 边上的点二列 r）；校正后的 Cliff's delta 和逐神经元 Wilcoxon 确认了一个小但稳健的迂曲度-vs-split 效应。

### 15. (Priority 0.253 · Surprise 0.284) Split 边在空间上比 correct 边更靠近 merge 位点（约 170 µm），提示联合失效区。
- **Run：** run-4--ground-truth-error-annotations-revised-version_2026-06-20 · **ID：** 61 · **Belief：** Leaning True → Likely True (0.7083 → 0.8906) · **Direction：** Positive
- **Tested：** 从 split 边到最近 merge 位点的欧氏距离是否小于 correct 边的。
- **Conclusion：** 在 6,805 个 split 和 1,109,034 个 correct 边上，split-到-merge 平均距离 2,084.76 µm（中位 1,794.64 µm）vs correct 2,265.14 µm（中位 1,956.93 µm）。Mann-Whitney U p = 3.50e-38，Welch's t p = 7.14e-27。split 和 merge 在共享失效区中共址。
- **Caveats：** 效应量适中（约 2 mm 基线上约 170 µm 差距）；显著性再次由样本量驱动。
- **Reproduction：** REPRODUCED (code: revised-loading)
- **Rerun result：** 记录值 "Split mean 2084.76 µm (median 1794.64); Correct mean 2265.14 µm (median 1956.93); Mann-Whitney U=3432660112.0, p=3.50e-38; Welch's t=-10.7131, p=7.14e-27"；重跑 "n_split=6805, n_correct=1109034; Split mean 2084.76 µm (median 1794.64); Correct mean 2265.14 µm (median 1956.93); Mann-Whitney U=3432660112.0, p=3.50e-38; Welch's t=-10.7131, p=7.14e-27" → 完全一致。修订：直接 $RERUN_PKL 加载 + numpy>=2。
- **Generalization：** GENERALIZES
- **Across datasets：** origin 789202：split 中位 1794.64 µm vs correct 1956.93 µm，Mann-Whitney p=3.50e-38，t=−10.71，p=7.14e-27。794491：split 818.04 vs correct 846.80 µm，p=1.98e-14，t=−2.85，p=2.18e-03（效应小得多但方向相同）。794495：split 1476.84 vs correct 1672.23 µm，p=2.91e-73，t=−22.28，p=4.29e-107。split-更靠近-merge 在全部三个脑上成立；794491 上差距缩小但方向和显著性保留。
- **Verdict：** MAJOR
- **Test：** Mann-Whitney U + Welch's t（n=6,805 split，1,109,034 correct），p=3.50e-38 / p=7.14e-27——对该数据形态是适当的检验，但结论混淆了统计与实际显著性。
- **Statistical issues：**
  - **效应量小：** 约 2,000 µm 基线上约 170 µm 差距（约 8% 相对偏移）。庞大 n 驱动 p 值，而非效应。
  - **非独立性：** 沿线缆的距离空间相关；把 1.1M 边各自当作独立会抬高证据。对神经元的 bootstrap 会给出温和得多的 p 值。
  - **稀疏锚点：** 仅 67 个 merge 位点锚定该比较；未报告对 merge 位点的置换。
  - **效应在 794491 上实质减弱**（中位差距从约 170 µm 缩到约 29 µm，t 从 −10.71 降到 −2.85）——方向相同但实际是不同的量级体系。
- **Logic issues：**
  - "联合失效区"是检验无法确立的机制框架——它同样可能反映两种误差类型因不相关原因偏好相同的致密神经毡，或任何未建模的共同混杂。
  - 标题"split 误差更靠近 merge 位点约 170 µm"夸大了实际效应，鉴于约 170 µm 相对基线有多小。
- **Downgrade based on rerun/extrapolation：** No——方向上 GENERALIZES，但应强调 794491 上的效应量减弱；报告应补充"量级跨脑实质变化"。
- **Corrected test：** (a) 按神经元对到最近 merge 位点的 median(split − correct) 距离做 cluster-bootstrap，返回 95% CI 和 cluster-robust p 值；(b) 带 cluster-bootstrap 95% CI 的 Cliff's delta；(c) merge 位点 bootstrap（重采样 67 个 merge 锚点）以界定锚集变异性。全部在相同数据上。
- **Corrected result：** origin 789202——观测中位差距（split − correct）= −162.29 um。cluster-bootstrap 95% CI = [−373.03, +15.55] um（CI 穿过零），cluster-p = 0.1067。Cliff's delta 的 cluster-bootstrap CI = [−0.1935, +0.0171]，p = 0.1333。merge 位点 bootstrap CI = [−302.78, +274.89] um。并列对比：原始 "U = 3.43e9, p = 3.5e-38; t = −10.71, p = 7.1e-27" → 校正后 "gap CI 跨越零, p = 0.11; Cliff's delta CI 跨越零, p = 0.13"——cluster 和 merge 位点 bootstrap 都无法在 α = 0.05 拒绝零。
- **Corrected generalization：** DOES-NOT-GENERALIZE（cluster 校正）——origin：gap CI [−373, +16]，p = 0.11；794491：gap CI [−126, +25]，p = 0.18（NS，差距坍缩到 ≈ −29 um）；794495：gap CI [−457, +24]，p = 0.09（NS 但边缘；观测差距 −195 um）。在校正检验下三个脑无一达到 cluster 显著，尽管三者点估计都为负。
- **Post-correction verdict：** OVERTURNED——原始 "split 更靠近 merge 位点 170 µm 且 p = 3.5e-38" 依赖把 1.1M 边当作 i.i.d.；一旦尊重神经元内相关，相同点估计在每个脑上的 95% CI 都包含零。方向一致性（三者均为负）有提示性但不显著；"联合失效区"机制主张不被 cluster 校正检验支持。

### 16. (Priority 0.253 · Surprise 0.284) 实际上 100% 的 GT split 间隙 <15 µm，远低于所设的 80% 阈值。
- **Run：** run-4--ground-truth-error-annotations-revised-version_2026-06-20 · **ID：** 63 · **Belief：** Leaning True → Likely True (0.7083 → 0.8906) · **Direction：** Positive
- **Tested：** 属于同一神经元的断开 GT 段是否有 >80% 被 <15 µm 分隔，以指导路径搜索 agent 的搜索半径。
- **Conclusion：** 在 6,805 个 split 过渡中，100.00% 的间隙落在 15 µm 以下，ECDF 在 3–5 µm 之间陡升。第 95 百分位 ≈ 5.85 µm，第 99 ≈ 6.50 µm。修复 agent 可用紧凑的约 6.5 µm 半径覆盖几乎所有真实 split，同时最小化意外 merge。
- **Caveats：** 未记录；这直接强化条目 #1。
- **Reproduction：** REPRODUCED (code: revised-loading)
- **Rerun result：** 记录值 "Total split gaps analyzed: 6805; % < 15 µm = 100.00%; ~5.85 µm covers 95%; ~6.50 µm covers 99%"；重跑 "Total split gaps analyzed: 6805; % < 15 µm = 100.00%; ~5.85 µm (95%), ~6.50 µm (99%)" → 完全一致。修订：直接 $RERUN_PKL 加载 + numpy>=2。
- **Generalization：** GENERALIZES
- **Across datasets：** origin 789202：6805 间隙，100.00% < 15 µm；95% 由 5.85 µm 覆盖，99% 由 6.50 µm。794491：7847 间隙，100.00% < 15 µm；95% 由 5.79 µm，99% 由 6.44 µm。794495：7988 间隙，100.00% < 15 µm；95% 由 5.80 µm，99% 由 6.41 µm。"100% 在 15 µm 以下"的主张和约 6.5 µm 的 99 百分位阈值在全部三个脑上几乎相同。
- **Verdict：** OK
- **Test：** 在 n=6,805 间隙上的描述性 ECDF 和百分位阈值；未报告正式假设检验。"≥80% 在 15 µm 以下"的主张鉴于 100% 在 15 µm 以下而显然成立。
- **Statistical issues：**
  - **无正式检验：** 未报告第 95/99 百分位的 CI，但在 n=6,805 下这些百分位估计得很紧，且结果近乎相同地推广到两个额外脑，因此这没问题。
  - 假设所设的 80% 阈值远低于观测到的 100%，故检验在一个平凡方向上是单侧的；无功效担忧。
- **Logic issues：**
  - 无——描述与结论相符。推广很引人注目：第 95 百分位在全部三个脑上为 5.79–5.85 µm。
- **Downgrade based on rerun/extrapolation：** No——在全部三个脑上基本相同地 GENERALIZES。

### 17. (Priority 0.253 · Surprise 0.284) Split 误差聚集在拓扑分支点附近（平均测地距离约 516 µm vs correct 边约 711 µm）。
- **Run：** run-4--ground-truth-error-annotations-revised-version_2026-06-20 · **ID：** 64 · **Belief：** Leaning True → Likely True (0.7083 → 0.8906) · **Direction：** Positive
- **Tested：** split 边是否（测地上）比 correct 边更靠近 GT 分支点，从而将局部几何复杂性牵连入碎片化。
- **Conclusion：** 在 6,805 个 split 和 1,109,034 个 correct 边上，到最近分支点的平均测地距离为 516.30 µm（split）vs 710.59 µm（correct），Mann-Whitney U p = 1.32e-197。split 边被紧密约束在分支点区域，缺少 correct 边的长距离尾部。
- **Caveats：** 该假设未提供正式审阅（review 字段 "N/A"），因此实现忠实性未经验证。
- **Reproduction：** REPRODUCED (code: revised-loading)
- **Rerun result：** 记录值 "split mean 516.30 µm, correct mean 710.59 µm; Mann-Whitney U p = 1.32e-197 over 6805 split & 1109034 correct edges"；重跑 "Total split edges evaluated: 6,805; Total correct edges evaluated: 1,109,034; split mean 516.30 µm, correct mean 710.59 µm; Mann-Whitney U=2979025451.5, p=1.3239e-197" → 完全一致。修订：$RERUN_PKL 替换了一个在沙箱中返回空的 `Path.rglob` 搜索 + numpy>=2。
- **Generalization：** GENERALIZES
- **Across datasets：** origin 789202：split mean 516.30 µm vs correct 710.59 µm，覆盖 6805/1109034 边，U=2.98e9，p=1.32e-197。794491：split 384.29 µm vs correct 389.49 µm，覆盖 7847/420702 边，U=1.46e9，p=5.16e-67（效应量小得多——仅约 5 µm 差距——但方向相同，仍高度显著）。794495：split 423.67 µm vs correct 506.89 µm，覆盖 7988/934849 边，U=3.47e9，p=2.97e-27。split-更靠近-分支点在全部三个脑上成立；效应量在 origin 上最大，在 794491 上最弱。
- **Verdict：** MAJOR
- **Test：** 对到最近分支点测地距离的 Mann-Whitney U，n=6,805 split vs 1,109,034 correct，U=2.98e9，p=1.32e-197——对偏斜距离数据是适当的非参数；review 字段为 "N/A"，故实现忠实性未经审计。
- **Statistical issues：**
  - **非独立性：** 一条边到分支点的测地距离与其邻居的距离高度相关；有效 n 远小于 1.1M。
  - **效应量减弱：** 在 794491 上中位差距从约 195 µm 坍缩到约 5 µm，但 p 仍 <1e-66。这是 n 驱动显著性的标志：方向保留，但实际信号在一个脑上几乎消失。
  - **无审阅（review = "N/A"）**——实现忠实性未经审计；实验循环未生成检查器。
- **Logic issues：**
  - "分支点周围的复杂局部几何增加碎片化风险"机制上可信，但观察性检验无法排除替代解释（信号密度、GT 密度、胞体邻近）。
  - 标题夸大："split 边被紧密约束在分支点区域"未能刻画 794491 上高度减弱的效应。
- **Downgrade based on rerun/extrapolation：** No——在全部三个脑上方向上 GENERALIZES，不过报告应附以"效应量跨脑变化约 30×"的限定。
- **Corrected test：** (a) 按神经元对到最近分支点的 median(split − correct) 测地距离和 Cliff's delta 做 cluster-bootstrap；(b) 对神经元层级中位距离的逐神经元配对 Wilcoxon signed-rank。两者都考虑原始 Mann-Whitney 当作 i.i.d. 的神经元内空间自相关。
- **Corrected result：** origin 789202——观测中位差距 = −184.19 um，Cliff's delta = −0.2238，cluster-bootstrap 95% CI = [−0.2847, −0.1346]，cluster-p ≈ 0；逐神经元配对 Wilcoxon W = 0，p = 2.44e-04，对每神经元 median(split − correct) 的 bootstrap CI = [−178.03, −87.09] um。并列对比：原始 "U = 2.98e9, p = 1.32e-197" → 校正后 "Cliff's delta = −0.22 [−0.28, −0.13]（按 Vargha-Delaney 为 small 效应），cluster-p ≈ 0；逐神经元 Wilcoxon p = 2.4e-04"。方向被确认；效应量从"p ≈ 0 样本量伪影"移动到"小但真实"的 Cliff's delta。
- **Corrected generalization：** GENERALIZES——origin 789202：Cliff's delta = −0.22 [−0.28, −0.13]，p ≈ 0；794491：Cliff's delta = ≈ −0.13 [−0.18, −0.07]，差距的 cluster-bootstrap CI = [−56.90, −26.46] um，p ≈ 0；逐神经元 Wilcoxon p = 1.95e-03。794495：Cliff's delta = ≈ −0.08 [−0.13, −0.03]，差距 CI = [−55.85, −9.74] um，p = 0.013；逐神经元 Wilcoxon p = 0.027。在 cluster 校正检验下，效应在全部三个脑上方向一致且显著，效应量从 origin 上的"小"减弱到 794495 上的"可忽略-到-小"。
- **Post-correction verdict：** UPHELD——原始结论（"split 边更靠近分支点"）在全部三个脑上经 cluster 校正后存活，但实际量级现被恰当地量化为 Cliff's delta 0.08–0.22（可忽略-到-小），而非原始 i.i.d. Mann-Whitney 暗示的 "p ≈ 1e-197"。标题措辞应为"小但稳健的空间关联"而非"紧密约束"。

### 18. (Priority 0.253 · Surprise 0.284) 独立再确认：split 边的局部 5-hop 迂曲度显著高于 correct 边。
- **Run：** run-4--ground-truth-error-annotations-revised-version_2026-06-20 · **ID：** 73 · **Belief：** Leaning True → Likely True (0.7083 → 0.8906) · **Direction：** Positive
- **Tested：** 5-hop 局部骨架迂曲度对 split 边是否高于 correct 边（对 #14 的更细窗口复制）。
- **Conclusion：** 1,109,034 correct vs 6,805 split 边：中位迂曲度 1.0801 vs 1.1132（均值 1.1194 vs 1.2062），单侧 Mann-Whitney p = 1.66e-276。U-Net 一贯在急转弯处吃力，但每边效应量再次很小。
- **Caveats：** 与 #14 相同的 caveat：绝对中位差异（约 0.03）不大；结果的显著性依赖巨大样本。这本质上是 #14 的稳健性检查。
- **Reproduction：** REPRODUCED (code: revised-loading)
- **Rerun result：** 记录值 "Median Tortuosity CORRECT 1.080117, SPLIT 1.113188; Mean CORRECT 1.119395, SPLIT 1.206197; Mann-Whitney U=4714207979.0, p=1.6599e-276 over 1109034 correct & 6805 split edges"；重跑 "Processed 1109034 correct edges and 6805 split edges; Median CORRECT 1.080117, SPLIT 1.113188; Mean CORRECT 1.119395, SPLIT 1.206197; Mann-Whitney U=4714207979.0, p=1.6599e-276" → 完全一致。修订：$RERUN_PKL 替换了一个在沙箱中返回空的 `Path.rglob` 搜索 + numpy>=2。
- **Generalization：** GENERALIZES
- **Across datasets：** origin 789202：correct 中位 1.0801，split 中位 1.1132，覆盖 1.11M correct / 6805 split，U=4.71e9，p=1.66e-276。794491：correct 1.0617，split 1.0854，覆盖 420702/7485，U=2.09e9，p≈0。794495：correct 1.0575，split 1.0735，覆盖 912147/7662，U=4.45e9，p=6.19e-192。方向相同（split > correct），量级适中相同（约 0.02–0.03 中位差距），所有 p 都极小。对 #14 的独立复制在所有脑上成立。
- **Verdict：** MAJOR
- **Test：** 对 5-hop 局部迂曲度的单侧 Mann-Whitney U，n=6,805 split vs 1,109,034 correct，U=4.71e9，p=1.66e-276——适当的非参数选择；与 #14 相同的统计担忧（这是其滑动窗口复制）。
- **Statistical issues：**
  - **迂曲度窗口重叠：** 连续边共享其 5-hop 窗口的大部分，因此每边迂曲度值高度自相关。Mann-Whitney 的独立性假设被违反得比 #14 还要严重。
  - **效应量微小：** 迂曲度中位差距约 0.03；均值差距约 0.09。巨大 p 值由 n ~ 1.1M 驱动。
  - **复制不增加证据：** 因为 #18 使用与 #14 相同的数据集和近乎相同的度量，它主要是稳健性检查而非独立确认。将其作为单独发现报告抬高了表观确认数。
- **Logic issues：**
  - "压倒性的统计证据"是误导性措辞：中位差距很小且与 #14 的中位差距无法区分。结论应强调效应的边缘性，而非其 p 值。
- **Downgrade based on rerun/extrapolation：** No——GENERALIZES，但与 #14 相同的 caveat：所有脑上效应量实际可忽略。
- **Corrected test：** (a) 按神经元对 median(split − correct) 5-hop 迂曲度差距和 Cliff's delta 做 cluster-bootstrap——5-hop 滑动窗口重叠 4 条边，故每边迂曲度不独立，Mann-Whitney 的 p 值 1.66e-276 是被抬高的后果；(b) 对神经元层级中位迂曲度（split vs correct）的逐神经元配对 Wilcoxon。
- **Corrected result：** origin 789202——观测中位差距 = +0.0331（与原始 ≈ +0.03 基本相同），Cliff's delta = +0.2498，cluster-bootstrap 95% CI = [+0.1926, +0.3110]，cluster-p ≈ 0；逐神经元配对 Wilcoxon W = 78，p = 2.44e-04，n_pairs = 12。并列对比：原始 "U = 4.71e9, p = 1.66e-276" → 校正后 "Cliff's delta = +0.25 [+0.19, +0.31]（small 效应），cluster-p ≈ 0；逐神经元 Wilcoxon p = 2.4e-04"。方向确认；效应量从"由 n 驱动的显著性"移动到"小但一致"的 Cliff's delta。
- **Corrected generalization：** GENERALIZES——origin 789202：Cliff's delta = +0.25 [+0.19, +0.31]；794491：Cliff's delta = +0.27 [+0.22, +0.32]，逐神经元 Wilcoxon p = 1.95e-03；794495：Cliff's delta = +0.19 [+0.15, +0.24]，逐神经元 Wilcoxon p = 1.91e-06。方向和量级在全部三个脑上一致。
- **Post-correction verdict：** UPHELD——方向确认，Cliff's delta 在全部三个脑上处于 0.19–0.27 范围（小但真实），与 #14（id 59）的校正效应量几乎完全匹配（10-hop 的 Cliff's delta 0.21–0.28）。这是对迂曲度-vs-split 关系的真正再确认，但处于"小效应"量级而非"p ≈ 0"标题；措辞应为"小但稳健"而非"压倒性的统计证据"。

### 19. (Priority 0.253 · Surprise 0.284) Merge 位点携带强的几何"缠结"特征——局部分支密度约为非 merge 对照的 28 倍。
- **Run：** run-4--ground-truth-error-annotations-revised-version_2026-06-20 · **ID：** 84 · **Belief：** Leaning True → Likely True (0.7083 → 0.8906) · **Direction：** Positive
- **Tested：** merge 位点在 U-Net 碎片图中是否比同段上匹配的非 merge 对照位点表现出更高的局部分支密度（15 µm 内）。
- **Conclusion：** 在 67 个 merge 位点上，平均局部分支密度为 1.13 vs 对照的 0.04；配对 t = 13.35（p = 2.03e-20），Wilcoxon W = 21.0（p = 1.19e-11），ROC-AUC 0.9236。虚假的局部分支是自动检测 merge 误差的强几何特征。
- **Caveats：** 仅 67 个 merge 位点；在更大 merge 群体上的可重现性会加强该结果。
- **Reproduction：** REPRODUCED (code: revised-loading)
- **Rerun result：** 记录值 "67 merge sites; density merge 1.13 vs control 0.04; paired t=13.35 p=2.03e-20; Wilcoxon W=21.0 p=1.19e-11; ROC-AUC 0.9236"；重跑 "Analyzed 67 merge sites; merge 1.13 vs control 0.04 branches/15µm; paired t=13.3482 p=2.0302e-20; Wilcoxon W=21.0000 p=1.1942e-11; ROC-AUC 0.9236" → 所有关键数字完全一致。脚本在打印完所有统计量*之后*以退出码 1 结束，因为分析后的 matplotlib `boxplot(..., labels=...)` 关键字在 matplotlib 3.11 中被重命名为 `tick_labels`；这是绘图-API 怪癖，而非重现失败。修订：直接 $RERUN_PKL 加载 + numpy>=2 + 内置 sklearn。
- **Generalization：** GENERALIZES
- **Across datasets：** origin 789202：67 merge 位点；merge 密度 1.13 vs control 0.04；配对 t=13.35，p=2.03e-20；Wilcoxon W=21，p=1.19e-11；AUC=0.9236。794491：86 merge；1.10 vs 0.08；t=12.14，p=2.97e-20；W=75，p=5.47e-13；AUC=0.8825。794495：105 merge；1.02 vs 0.05；t=13.46，p=1.65e-24；W=109，p=7.95e-16；AUC=0.8857。（三次运行在统计量打印后都出现相同的下游 matplotlib boxplot `labels` 崩溃——分析本身在每个脑上都干净地重现。）merge 位点在全部三个脑上都携带密集分支缠结特征，AUC ≥ 0.88。
- **Verdict：** OK
- **Test：** 在 67 个 merge 位点上以同段匹配对照做配对 t 检验（t=13.35，p=2.03e-20）+ Wilcoxon signed-rank（W=21，p=1.19e-11）+ ROC-AUC（0.9236）——两个检验都适用于配对数据，且 Wilcoxon（非参数）针对任何正态性担忧佐证了参数 t 检验。
- **Statistical issues：**
  - **小但充分的 n：** 67 个 merge 位点适中，但效应量巨大（1.13 vs 0.04，约 28× 比值），故功效良好。参数和非参数检验都一致。
  - **同段匹配对照**是吸收神经元层级混杂的合理设计。
  - **同一度量被用作可发现特征：** AUC 0.9236 是样本内-在-对照设计上的量；在原始段上的样本外性能才是相关的部署数字，但结构性主张（merge 位点局部分支密度高）是稳固的。
- **Logic issues：**
  - "强几何特征……可作为强大特征"得到三个脑上 AUC ≥ 0.88 的支持，无越界。
- **Downgrade based on rerun/extrapolation：** No——GENERALIZES（全部三个脑上 AUC 0.88–0.94，配对 t 和 Wilcoxon 都显著）。

### 20. (Priority 0.253 · Surprise 0.284) Omit 误差在终末（叶端）边上的发生概率几乎是内部边的两倍。
- **Run：** run-4--ground-truth-error-annotations-revised-version_2026-06-20 · **ID：** 85 · **Belief：** Leaning True → Likely True (0.7083 → 0.8906) · **Direction：** Positive
- **Tested：** omit 误差是否不成比例地发生在终末分支（以叶结束的路径）上，而非 GT 神经元拓扑的内部段。
- **Conclusion：** 终末边 omit 率 4.38%（23,792 / 543,477）vs 内部边 omit 率 2.41%（20,898 / 865,568）；χ² = 4189.94，p < 1e-300。81% 的相对增幅表明分割模型以约两倍于内部的速率丢失远端细神经突。
- **Caveats：** 未记录；样本量和效应量都很大。
- **Reproduction：** REPRODUCED (code: revised-loading)
- **Rerun result：** 记录值 "Terminal Omit=23792 / Non-Omit=519685 (4.38%); Internal Omit=20898 / Non-Omit=844670 (2.41%); Chi-square=4189.94, p ≈ 0"；重跑 "Terminal Omit=23792 Non-Omit=519685 (4.38%); Internal Omit=20898 Non-Omit=844670 (2.41%); Chi-square=4189.9364, p=0.0000e+00" → 完全一致。修订：直接 $RERUN_PKL 加载 + numpy>=2。
- **Generalization：** GENERALIZES
- **Across datasets：** origin 789202：terminal 4.38% vs internal 2.41%（比值 1.82×），χ²=4189.94，p≈0。794491：terminal 4.94% vs internal 3.82%（比值 1.29×），χ²=425.40，p=1.63e-94。794495：terminal 2.47% vs internal 1.82%（比值 1.36×），χ²=691.98，p=1.66e-152。terminal-omit-率-大于-internal 在全部三个脑上成立，所有 p ≪ 0.001；相对量级约 1.3×–1.8×。
- **Verdict：** MINOR
- **Test：** 对边类型（terminal vs internal）× omit（是/否）的 Chi-square 独立性检验，χ²=4189.94，p<1e-300，n=1,409,045 边——适用于 2×2 列联表；期望单元计数非常大。
- **Statistical issues：**
  - **违反独立性：** 同一神经元内的边非 i.i.d.；神经元层级聚类会收紧 SE。在比值 1.82× 和 χ² ≈ 4200 下，无现实校正会推翻结论。
  - **效应量跨脑减弱：** 比值从 1.82×（origin）降到 1.29×（794491）和 1.36×（794495）。方向一致，但"两倍可能"的标题对三个脑中的两个有所夸大。
- **Logic issues：**
  - "远端细神经突"是未被直接检验的机制性猜想——终末边可能因许多原因被遗漏（信号稀疏、轴突-vs-树突差异、GT 整理伪影）。
- **Downgrade based on rerun/extrapolation：** No——方向上 GENERALIZES。标题"几乎两倍可能"应软化为"视脑而定多 1.3× 到 1.8×"。

---

## Reproduction — Summary

- **使用的数据集 pkl：** `cache/dataset_cache_789202_mcl100_add.pkl`（单脑：789202，mcl 100，带 `_add` 富集）。
- **对 top-20 重排序记录的分解：** **19 REPRODUCED**、**1 DIVERGED**、**0 FAILED**（n_rerun = 20）。全部 20 个裁决都来自 `code_source: revised`（每个脚本都需要加载/环境修订；0 个未经改动地运行记录代码）。
- **未能重现的发现：**
  - 条目 5（id 24，"super-merges fuse 3+ neurons"）：**DIVERGED——数据集范围问题，而非分析 bug。** 记录实验跨多个 `*_add.pkl` 脑缓存聚合（n=24 two-neuron + 3 super-merges，U=0，p=5.89e-03）；重跑只有提供的单脑 pkl（n=8 + 1，U=0，p=2.22e-01）。中位数（6.21 mm vs 35.19 mm）和 U 统计量仍一致——只有样本量因而 p 值不同。pkl 内部的分析是忠实的。
- **应用的加载修订**（`autodiscovery/run-4--ground-truth-error-annotations-revised-version_2026-06-20.json.rerun/hypo_<id>.py`）：
  - **在宿主的 NumPy 1.x 下 NumPy-2 pkl 无法 unpickle**（ids 21、24、27、30、32、33、36、37、39、45、58、59、61、63、84、85）：将一套全新的 `numpy>=2 / scipy / pandas / sklearn / patsy / statsmodels / matplotlib / networkx / psutil / agentic_neuron_proofreader` 栈内置到 `~/.local-numpy2`，将其前置到 `sys.path`，并从路径中移除 `~/.local/lib` 和 `/shared/utils...` site-packages，使新栈胜出。
  - **"Dataset not found" 门**（ids 64、73、139、43）：记录代码用 `Path.rglob`/`os.walk('..')` 搜索 `*_add.pkl`，而重跑助手的 `glob`/`Path.exists` monkeypatch 未覆盖。修订在 bootstrap 中添加 `Path.rglob` 和 `os.walk` monkeypatch，使任何 `.pkl` 搜索都产出单一 `$RERUN_PKL`。
  - 每个脚本中的 **`pip install` 重试循环**被去除（替换为 no-op `subprocess`），以避免运行时按记录重复安装。
  - **sklearn / patsy ABI 不兼容**（ids 30、27）由同一内置 numpy-2 栈解决（sklearn 1.9.0、patsy 1.0.2、statsmodels 0.14.6、pandas 3.0.3）。
- **关于条目 19（id 84）的说明：** 重跑退出码为 1，但所有关键数字（配对 t=13.3482 p=2.03e-20，Wilcoxon W=21 p=1.19e-11，ROC-AUC 0.9236，n=67 merge 位点）都在崩溃*之前*打印，崩溃发生在 `plt.boxplot(..., labels=...)` 绘图调用中——`labels` 在 matplotlib 3.11 中被重命名为 `tick_labels`。分析本身完全重现；只有分析后的绘图渲染被下游库 API 重命名击中，因此计为 REPRODUCED。

---

## Generalization — Summary

- **测试的额外数据集：** `cache/dataset_cache_794491_mcl100_add.pkl` 和 `cache/dataset_cache_794495_mcl100_add.pkl`（都是与 origin 789202 脑具有相同载荷结构的单脑缓存）。
- **top 20 的裁决分解：** **14 GENERALIZES**、**3 PARTIAL**、**3 DOES-NOT-GENERALIZE**、**0 INCONCLUSIVE**。
- **逐条目裁决：** #1 (id 30) GENERALIZES · #2 (id 27) PARTIAL · #3 (id 21) DOES-NOT-GENERALIZE · #4 (id 139) DOES-NOT-GENERALIZE · #5 (id 24) PARTIAL · #6 (id 32) GENERALIZES · #7 (id 33) GENERALIZES · #8 (id 36) DOES-NOT-GENERALIZE · #9 (id 37) GENERALIZES · #10 (id 39) GENERALIZES · #11 (id 43) GENERALIZES · #12 (id 45) PARTIAL · #13 (id 58) GENERALIZES · #14 (id 59) GENERALIZES · #15 (id 61) GENERALIZES · #16 (id 63) GENERALIZES · #17 (id 64) GENERALIZES · #18 (id 73) GENERALIZES · #19 (id 84) GENERALIZES · #20 (id 85) GENERALIZES。

**未完全推广的发现（附一行原因）：**

- **#3 (id 21) — Z 主导 vs XY 主导 Chi-square — DOES-NOT-GENERALIZE。** origin 无显著差异（p=0.060）；794491 显示 Z 误差率显著更高（p=5.2e-38），794495 显示 Z 显著更低（p=4.6e-88）。两个额外脑都以相反方向拒绝了无各向异性说法。
- **#4 (id 139) — 分支阶预测 split（正系数）— DOES-NOT-GENERALIZE。** origin coef=+0.0194（p<0.001）；794491 coef=−0.0157 且 794495 coef=−0.0063（均 p<0.001）。三者都显著，但符号在两个额外脑上都翻转。
- **#8 (id 36) — 被遗漏线缆更靠近 merge 位点 — DOES-NOT-GENERALIZE。** origin 和 794495 确认（omit < correct），但在 794491 上关系反转（omit 1103 µm > correct 842 µm，单侧 p=1.000）。空间共聚跨脑不稳健。
- **#2 (id 27) — Z-alignment 混合效应回归 — PARTIAL。** 三个脑都打印非正 Z 系数（故 "Z not a driver" 标题存活），但标准化 coef 从 +0.0376（794491，p=0.17）到 −0.0661（origin，p=0.058）到 −0.1697（794495，高度显著，p=6.75e-06）变化。794495 发现先验未预料到的强保护性效应。
- **#5 (id 24) — Super-merges 融合 3+ 神经元覆盖更多线缆 — PARTIAL。** origin 单脑重跑不显著（n=8+1，p=0.222）——已标记 DIVERGED；794491 有 0 个 super-merge 故检验无法运行；794495 强确认（n=24+2，p=6.15e-03）。该模式在 super-merge 存在处成立，但它们在某些脑上稀少。
- **#12 (id 45) — 极端-Z omit 率 vs 中央-Z — PARTIAL。** origin OR=2.36（p≈0）和 794495 OR=3.17（p≈0）确认；794491 OR=1.01（p=0.80）完全无效应。体积边缘 omission 是数据集特定的。

**综述。** 在测试了两个额外脑后，局部几何发现——仅距离的 split 桥接（#1）、分支节点 omit 率（#6）、角度对齐（#7）、split 聚类（#9）、A* 修复成功率（#10）、merging 段是巨型（#11）、bridged-vs-broken omit 长度（#13）、迂曲度（#14、#18）、split 到 merge 和分支点的邻近度（#15、#17）、6.5 µm split 间隙阈值（#16）、merge 缠结特征（#19）、终末边 omit 率（#20）——在全部三个脑上干净地复制，看起来像是 U-Net 失效模式的稳健、数据集无关属性。相比之下，每个 Z 轴/各向异性主张（#2、#3、#12）都是脑特异或方向不一致的：origin 上的无各向异性说法实际在另两个脑上翻转为显著不同（且相互矛盾）的 Z 效应，而极端-Z omission 效应在 2/3 脑上存在、在第三个上缺失。分支阶系数（#4）在两个额外脑上翻转符号，故其在 origin 上的正方向很可能是脑 789202 的伪影而非普遍趋势。super-merge 主张（#5）和 merge-omit 空间共聚主张（#8）在一个额外脑上成立但在另一个上失效。鉴于仅有两个额外数据集，关于"稳健"的陈述应读作"在 3 个脑上一致"，这是良好证据但非普适性的确凿证明。

---

## Excluded (no surprisal score)

助手因缺失 surprisal 丢弃了 2 条假设：

- Run `run-4--ground-truth-error-annotations-revised-version_2026-06-20`，ID 10
- Run `run-4--ground-truth-error-annotations-revised-version_2026-06-20`，ID 41

---

## Statistical Verification — Summary

### 逐假设严重度（优先级顺序）

**CRITICAL (3)：**
- **#3 (id 21) — Z 主导 vs XY 主导 Chi-square。** 边缘 p=0.0602 被误读为"否定 Z 各向异性"；"未能拒绝 = 原假设为真"谬误。推广在两个额外脑上以相反方向反转了原假设。
- **#4 (id 139) — 离心分支阶 logistic 回归。** 违反独立性（无神经元随机效应），微小的每单位 OR ≈ 1.02 被 n = 1.4M 抬高，且系数符号在两个额外脑上都翻转同时仍打印 p < 0.001——显著但不稳健的标志。
- **#8 (id 36) — 被遗漏线缆更靠近 merge 位点。** 非独立性 + 约 2 mm 基线上约 147 µm 的小效应 + 794491 上反转（omit 比 correct 更*远*，p=1.000）。机制主张（"模型牺牲相邻细突起"）越界。

**MAJOR (7)：**
- **#2 (id 27) — Z-alignment 混合效应 logreg。** 任意 20k 子样本丢弃 98% 数据并把 p 停在 0.058 边缘；结论 "Z not a driver" 是从 p>0.05 以同样的接受原假设谬误推出的。推广 PARTIAL（符号和显著性变化）。
- **#5 (id 24) — Super-merges 融合 3+ 神经元。** 功效不足（n=3 super-merge），检验比较总线缆而非每神经元线缆（偏离假设），且在 794491 上 super-merge 根本不存在。
- **#12 (id 45) — 极端-Z omit 率。** "按脑 ID 分层"在单脑上是无操作；非独立性低估 p；机制性光学衰减主张被 794491 反驳（OR=1.01，p=0.80）。
- **#14 (id 59) — 用于 split 检测的局部迂曲度。** r=0.0335 是有显著无效应（r² = 0.001）；非独立性；大 n 微效应陷阱。
- **#15 (id 61) — Split 到 merge 位点的距离。** 2 mm 基线上约 170 µm 差距（约 8% 相对偏移）；效应在 794491 上减弱约 6×。
- **#17 (id 64) — Split 到分支点的距离。** 效应在 794491 上减弱约 30×（195 µm → 5 µm 差距），但 p 仍 <1e-66；1.1M 边上的非独立性。review 字段为 "N/A"，故忠实性未经审计。
- **#18 (id 73) — 5-hop 迂曲度复制。** 与 #14 相同的由 n 驱动显著性陷阱；并非真正独立（相同数据集，近乎相同的度量）。报告得如同它增加了证据。

**MINOR (5)：**
- **#6 (id 32) — 分支 vs 线性节点 omit 率。** 大效应（4.3×），p≈0；唯一担忧是边非独立性，不会推翻结果。GENERALIZES。
- **#9 (id 37) — Split 聚类。** 部分同义反复（"split 与 split 聚类"），但 10× 比值真实且推广。
- **#10 (id 39) — A* 修复。** 无统计检验、无成功率 CI 的工程基准。
- **#13 (id 58) — Bridged vs broken omit 长度。** 类别定义已部分编码长度；效应适中（约 1.5× 中位比值）。
- **#20 (id 85) — 终末 vs 内部 omit 率。** 边非独立性；比值在两个额外脑上减弱到 1.3×。

**OK (4)：**
- **#1 (id 30) — 欧氏间隙距离。** 在 11k 间隙上仅距离 AUC 0.9979，在两个额外脑上重现至 ≥0.989；样本内阈值乐观偏差轻微。
- **#7 (id 33) — 角度对齐。** 在 13,582 个 split 上 AUC 0.9322、KS 0.7460，在两个额外脑上重现 AUC 0.93。
- **#11 (id 43) — Merging 段是巨型。** 对 log 线缆长度的 Welch t，跨所有脑约 30× 比值。
- **#16 (id 63) — 100% 的 split 间隙在 15 µm 以下。** 描述性，全部三个脑上近乎相同的第 95 百分位阈值（约 5.8 µm）。
- **#19 (id 84) — Merge 位点分支密度。** 配对 t + Wilcoxon + AUC=0.92，带同段匹配对照。在全部三个脑上重现，AUC ≥ 0.88。

### 多重比较（BH-FDR，q=0.05）覆盖 top-20

- top-排序的 20 个发现中有 18 个报告了可用 p 值（#10 和 #16 报告描述性统计，无 p；这两个被排除在 BH-FDR 之外）。
- 对这 18 个 p 值应用 Benjamini–Hochberg（q=0.05），截断在 k=16，p ≤ k/N·q = 0.0444。
- **存活者（16/18）：** #1（有效 p ≈ 0，基于 AUC）、#4（logreg p<0.001）、#5（Mann-Whitney p=0.0059）、#6（χ²≈0）、#7（KS≈0）、#8（p=1.6e-66）、#9（p≈0）、#11（p=7.3e-25）、#12（p≈0）、#13（p=1.9e-19）、#14（p≈0）、#15（p=3.5e-38）、#17（p=1.3e-197）、#18（p=1.7e-276）、#19（配对 t p=2.0e-20）、#20（χ²≈0）。
- **未存活（2/18）：** **#2 (id 27) p=0.0582 > 0.0444**、**#3 (id 21) p=0.0602 > 0.0444**。两者都是边缘 Z 轴"无效应"结果，其文字结论已被跨脑推广所否定。BH-FDR 结果确认，即便在这个小面板规模下，这些边缘 p 值也无法通过全局校正。
- 对完整 150 假设运行（不仅是 top-20），BH 截断会紧得多；若进一步考虑这类运行典型的约 250 假设发现循环族，在 α=0.05 下预期偶然产生约 12 个虚假显著结果。强效应发现（AUC > 0.9，比值 > 4×，p ≪ 1e-20）对任何合理校正不敏感；边缘者（#2、#3）则不然。

### 横切问题

- **神经元内边/节点的非独立性是普遍存在的。** #3、#4、#6、#8、#9、#14、#15、#17、#18、#20 都在约 1M 边上把它们当作 i.i.d. 运行检验，而实际上同一神经元/线缆/碎片内的边高度相关。考虑聚类后，有效 n 小一到数个数量级。这不会推翻大效应发现（#6 比值 4.3×、#9 比值 10×、#20 比值 1.8×），但它使边缘/小效应发现（#14 r=0.03、#15 约 170 µm 差距、#17 在 794491 上减弱到约 5 µm、#18 中位差距 0.03）比所印 p 值暗示的弱得多。top-20 中无一使用混合效应模型或 cluster-robust SE。
- **"未能拒绝 = 原假设为真"谬误。** #2 和 #3 都把刚高于 0.05 的 p 当作确认无效应；这两者上的先验 → 后验摆动是本次运行"标题"surprise 分值的主要贡献。这是本次运行中最具后果的逻辑错误。
- **由样本量而非效应量驱动的显著性。** #8（约 1M 边上约 7% 相对偏移，p=1.6e-66）、#14（1.1M 边上 r=0.0335，p≈0）、#15（约 8% 相对偏移，p=3.5e-38）、#17（5–195 µm 差距，p=1.3e-197）、#18（Δ中位 0.03，p=1.7e-276）、#20（1.4M 边上 1.3×–1.8× 比值）——如果用对神经元 bootstrap 的 CI 报告，这些都应主张小得多的"效应量"。
- **以推广为验证。** 三个发现（#3、#4、#8）在额外脑上完全反转，另三个（#2、#5、#12）部分反转；一旦考虑额外脑，这些上的原始 surprise / belief 变动便无依据。它们应从"置信度翻转"降级为"脑特异"。
- **机制越界。** #4（"更深阶分支更易受影响"）、#8（"模型在融合时牺牲细突起"）、#12（"深处光学衰减"）、#15（"联合失效区"）、#20（"远端细神经突"）都从纯关联性检验得出因果/机制结论。下游含义对可操作干预设计的重要性大于对所报 surprise 分值。

---

## Statistical Test Corrections — Summary

**范围：** top-20 假设中有 10 个被标记需要测试层级修复，并用 `autodiscovery/run-4--ground-truth-error-annotations-revised-version_2026-06-20.json.fixed/hypo_<id>.py` 中的校正脚本重新测量。其余 10 个条目要么运行了正确检验，要么只有"检验没问题，标题越界"的 caveat。

### 校正后裁决分解

- **UPHELD (4)：** #3 (id 21)、#14 (id 59)、#17 (id 64)、#18 (id 73)。在每个上，原始所宣称的方向在 cluster 校正检验下存活；效应量被恰当量化（神经元层级的 Cliff's delta 或配对 Wilcoxon），是"小但真实"而非原始 i.i.d. 检验抬高的 "p ≈ 0"。#3 仅在 origin 上 UPHELD（尊重神经元簇时无各向异性效应）但 DOES-NOT-GENERALIZE——那部分不变。
- **WEAKENED (4)：** #4 (id 139)、#5 (id 24)、#12 (id 45)、#14 (id 59)。方向存活但效应量远小于原始显著性所暗示，或在 cluster 校正检验下仅在 1/3 脑上显著。（#14 同时列入 UPHELD 和 WEAKENED，因为方向推广但一旦恰当估计效应量很小；详见其条目。）
- **OVERTURNED (3)：** #2 (id 27)、#8 (id 36)、#15 (id 61)。校正检验要么反转符号（#2：在 cluster-robust GEE 下 Z 显著*保护*，与 "Z not a driver" 相反；#8：在 cluster bootstrap 下 794491 上差距*反转*），要么无脑达到 cluster 显著（#15：差距的 cluster-bootstrap CI 在每个脑上都包含零）。

### 表格汇总

| # | id | 原始检验 | 原始标题结果 | 校正检验 | 校正后 origin 结果 | 裁决 |
| --- | --- | --- | --- | --- | --- | --- |
| 2 | 27 | 20k 子样本上的混合效应 logreg | coef = -0.066, p = 0.058, OR = 0.80 | 完整 1.16M 上的 GEE，cluster-robust SE | coef = -0.123, p ≈ 0, OR = 0.66 [0.62, 0.72] | OVERTURNED |
| 3 | 21 | 1.4M 边上的 chi-square（i.i.d.） | chi2 = 3.53, p = 0.060 | GEE + cluster-perm chi2 + 逐神经元 Wilcoxon | OR = 1.02 [0.97, 1.07], all p > 0.45 | UPHELD (origin) |
| 4 | 139 | 1.4M 边上的普通 logreg（i.i.d.） | coef = +0.019, z = 16.37, p < 0.001 | 带神经元簇的 GEE | coef = +0.019, p = 0.038, OR = 1.02 [1.001, 1.04] | WEAKENED |
| 5 | 24 | 对总线缆的 Mann-Whitney U（多脑） | U = 0, p = 5.9e-03 (floor) | 对每神经元线缆的 MW + Cliff's delta + bootstrap CI | origin 上 n=8 vs 1, p = 0.22 (NS) | WEAKENED |
| 8 | 36 | 49k 节点距离上的单侧 MW | U = 1.138e9, p = 1.60e-66 | 按神经元 cluster-bootstrap + Cliff's delta + merge 位点 bootstrap | gap CI [-402, +92] um, p = 0.27 | OVERTURNED |
| 12 | 45 | CMH "按脑分层"（单脑） | OR = 2.36, p ≈ 0 | 带神经元簇的 GEE + OR 的 cluster-bootstrap | OR = 2.36 [0.46, 12.18], p = 0.31 | WEAKENED |
| 14 | 59 | 1.1M 边上的 MW + 点二列 | U = 4.58e9, p ≈ 0; r = 0.034 | cluster-bootstrap Cliff's delta + 逐神经元 Wilcoxon | delta = +0.26 [+0.21, +0.33] | WEAKENED |
| 15 | 61 | 1.1M 边上的单侧 MW + Welch's t | U = 3.43e9, p = 3.5e-38 | 按神经元 cluster-bootstrap + Cliff's delta + merge 位点 bootstrap | gap CI [-373, +16] um, p = 0.11 | OVERTURNED |
| 17 | 64 | 1.1M 边测地距离上的 MW | U = 2.98e9, p = 1.32e-197 | cluster-bootstrap Cliff's delta + 逐神经元 Wilcoxon | delta = -0.22 [-0.28, -0.13], p ≈ 0 | UPHELD |
| 18 | 73 | 1.1M 边 5-hop 迂曲度上的单侧 MW | U = 4.71e9, p = 1.66e-276 | cluster-bootstrap Cliff's delta + 逐神经元 Wilcoxon | delta = +0.25 [+0.19, +0.31], p ≈ 0 | UPHELD |

### 模式

- **非独立性抬高是主导缺陷。** 10 个校正条目中有 8 个（#3、#4、#8、#12、#14、#15、#17、#18）的检验把神经元内相关观测当作 i.i.d.。一旦应用 cluster-robust（GEE）或 cluster-bootstrap（神经元重采样）检验，三个发现（#2、#8、#15）OVERTURN，两个仅在 3 个脑中的 1 个上发现显著效应（#4 在 origin，#12 在 794495），三个（#14、#17、#18）以原始 "p ≈ 0" 框架严重夸大的"小效应"量级保留方向。
- **"未能拒绝 = 原假设为真"谬误与抽样不足叠加。** #2（20k 子样本，p = 0.058）和 #3（i.i.d. chi-square，p = 0.060）是两个原始结论（"Z not a driver"）从 p > 0.05 推出的条目。两者在校正分析下都被翻转——#2 在 origin 和 794495 上变为显著*保护*（Z 降低误差几率），#3 变为"origin 上无效应但 DOES-NOT-GENERALIZE"，一个脑显示保护、一个显示正向 Z 效应。#2 和 #3 上的 "Likely True → Uncertain" surprise 不被数据支持。
- **由样本量驱动的显著性变为效应量诚实。** #14、#17、#18 在 cluster 校正检验下都保留了相同方向，但 Cliff's delta 处于"小"范围（|delta| 0.08–0.28）。原始 p ≈ 0 / p ≈ 1e-200 数字是把约 1M 相关边当作独立的伪影；真实证据跨脑一致但处于"小效应"量级。
- **下限效应和同义反复。** #5 (id 24) 是教科书式的小样本下限效应：n = 3 super-merge 下 U = 0 产生可达到的最小 p 0.006，但仅在 origin 脑上的每神经元重跑给出 p = 0.22；"高度显著"主张仅在 794495（n = 24 + 2 super-merge）上可维持。

### 综述

一旦应用正确检验，本次运行的标题图景在三个重要方面发生转变。**第一，两个置信度翻转的各向异性发现（#2 id 27、#3 id 21）无法作为"无效应"结果维持；** 在 origin 上，#2 在完整 1.16M 边上的 cluster-robust GEE 显示 Z-alignment 显著*保护*（OR = 0.66，p ≈ 0）——恰与先验的 "Likely True → Uncertain" 摆动相反——而 #3 的"未能拒绝"在正确聚类下存活但以相反方向 DOES-NOT-GENERALIZE。**第二，三个"巨大 n、微小效应"发现（#14、#17、#18）变为诚实的"小效应"发现：** 方向和跨脑稳健性存活，但效应量被恰当描述为 Cliff's delta ≈ 0.08–0.28，而非 "p ≈ 1e-200"。**第三，两个空间共聚发现（#8 omit-near-merge、#15 split-near-merge）在 cluster 校正下坍缩：** #15 在每个脑上、#8 在 2/3 脑上，中位差距的 cluster-bootstrap 95% CI 都包含零，且一个脑实际反转方向。关于"联合失效区"和"模型在融合时牺牲细突起"的机制主张不再被样本量诚实的检验支持。在正确检验下稳健 UPHELD 的发现是局部几何/拓扑类——迂曲度（#14、#18）和分支点邻近度（#17）——处于小但真实的效应量。OVERTURNED 或 WEAKENED 的发现集中于 (a) Z 轴各向异性说法和 (b) 空间共聚主张，恰是先前横切分析已为"大 n 微效应显著性"和"未能拒绝谬误"所标记的条目。
