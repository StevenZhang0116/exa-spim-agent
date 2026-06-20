# AutoDiscovery 跨报告合并 — 所有运行

## Header

**已纳入的来源报告**(辅助脚本:`python agentic/collect_summaries.py`,`n_files = 2`,`n_entries_total = 40`):

- `ground-truth-error-annotations-revised-version_2026-06-17.summary.md` — 运行 `ground-truth-error-annotations-revised-version_2026-06-17` — 20 条条目(较旧)。
- `run-4--ground-truth-error-annotations-revised-version_2026-06-20.summary.md` — 运行 `run-4--ground-truth-error-annotations-revised-version_2026-06-20` — 20 条条目(较新)。

**聚类后:** 共保留 26 项独立的科学发现(两份报告共同印证的 14 项 + 仅在一份报告中出现的 12 项)。在那 12 项独有发现中,有 11 项仅出现在较新的 run-4 报告中,1 项仅出现在较旧的 2026-06-17 报告中。

**综合解读。** 两份报告独立地重新发现了同一个主导主题:U-Net 分割失败是非随机的拓扑—空间现象,具体表现为 (a) **merge 错误集中在结构拥挤 / 分支密集的神经网中**,(b) **split 错误集中于拓扑分支点以及远端 / 细小突起上**,以及 (c) **omit 错误集中在末端 / 叶子边上,并形成连续的条带**。两次运行中得到最清晰的共同印证的判别指标是:跨 split 缺口的方向对齐角度(真实情况下 ~153° 对比错误情况下 ~90°,AUC ≈ 0.93)以及 split 缺口距离分布的紧致性(95 分位数 ≈ 5.8 µm,99 分位数 ≈ 6.5 µm)。最重要的独有 / 新发现来自较新的 run-4 报告:**(1) 头条结论的信念翻转,转向仅靠距离即可自动重连(AUC 0.9979,F1 最优阈值 6.84 µm)**(id 30);**(2) Z 轴各向异性假设被否定**(ids 27 和 21,经修正后分别达到 CRITICAL/MAJOR);**(3) "超合并"发现,即融合 3 个及以上神经元的合并所覆盖的 cable 是 2 神经元合并的约 5×**(id 24);以及 **(4) omit 与 split 错误在空间上与 merge 位点共定位**(ids 36 和 61),两者在 cluster-correct 检验下均被 OVERTURNED。跨报告分歧:较旧报告中的 H39 度量的是 **针对 merge 的切断**(~86% merge 移除,准确率 +11pt),而较新报告中的 H39 度量的是 **A\* split 修复**(每个 split 86% 的成功率,准确率 +0.42pt)——同一个 id,不同干预手段,不同效应。较旧报告的 H35 宣称固定半径的 split bridging "不可行"(缺口比内部边大 ~4×),而较新报告的 H30/H63 则宣称仅依靠距离 bridging 对 inter-neuron 控制组而言近乎完美——这一表面矛盾可由控制组的选择来调和(intra-fragment 内部边 vs inter-neuron 邻居)。

---

## 独有与新发现

### 1. (Priority 0.507 · Surprise 0.690) 仅靠欧氏缺口距离就能近乎完美地将真实 split 与 inter-neuron 邻居区分开,使信念转向支持仅靠距离的自动重连。
- **来源:** `run-4--ground-truth-error-annotations-revised-version_2026-06-20` (id 30) — 仅此一份报告发现。最新运行。
- **结论:** 在 6,805 个真实 split 缺口与 4,189 个 20 µm 以内的 inter-neuron 缺口上,真实 split 距离紧密集中在 ~4.5 µm 附近(多数在 2–7 µm),而 inter-neuron 缺口几乎不会低于 7 µm。二分类器 ROC AUC 为 0.9979,F1 最优阈值为 6.84 µm(max F1 = 0.9945),证实仅靠距离就是一个可行、安全的启发式。先验 "Leaning False" (0.2917) 被翻转为 "Leaning True" (0.7344)。在所有三个大脑上都泛化:AUC ≥ 0.9889,阈值 6.48–6.84 µm,F1 ≥ 0.9711。
- **沿用判定:** Reproduction REPRODUCED;Generalization GENERALIZES;Verdict OK。
- **为何独有/新:** 仅靠距离的自动重连在较旧报告中被认为不可行(H35:缺口比内部边大 ~4×)。较新报告将控制组换为 inter-neuron(而非 intra-fragment 内部边)的重新框定,使结论被反转。
- **Caveats:** 样本内进行 F1 最优阈值选择会引入轻度乐观偏差;部署中的智能体需要防范在测试集中未代表的密集神经网区域产生的假阳性。

### 2. (Priority 0.323 · Surprise 0.795) 沿 Z 轴对齐的神经突起并没有比沿 XY 对齐者明显更容易出错——但经修正后的 GEE 翻转结果为 "Z 显著具有保护性"。
- **来源:** `run-4--ground-truth-error-annotations-revised-version_2026-06-20` (id 27) — 仅此一份报告发现。最新运行。
- **结论:** 记录在案的混合效应 logistic 回归在 20,000 条边的子样本上(总计 1,160,529 条)给出标准化 z 对齐系数为 −0.0661,p = 0.0582,OR = 0.8029——不显著且方向上略微"反向",促使信念从 Likely True (0.9167) 降为 Uncertain (0.4062)。在全部 1.16M 条边上做 cluster-robust GEE 修正后,系数为 −0.1229,p ≈ 0,OR = 0.6649 [0.6180, 0.7154]:Z 对齐显著具有保护性,与先验相反。Generalization PARTIAL:origin 789202 有保护性(OR 0.66,p≈0);794495 有保护性(OR 0.51,p=5.7e-12);794491 无显著效应(OR 1.12,CI [0.81, 1.55])。
- **沿用判定:** Reproduction REPRODUCED;Generalization PARTIAL;Verdict MAJOR;Post-correction verdict OVERTURNED。
- **为何独有/新:** 较旧报告未测试 Z 轴各向异性。
- **Caveats:** 原始 20k 子样本浪费了约 98% 可用数据;"未拒绝 = 零假设为真"的谬误导致了原始的负向意外感;经修正后的保护性效应是大脑特异的(在 794491 上不存在)。

### 3. (Priority 0.266 · Surprise 0.568) Z 主导边相对 XY 主导边的错误率本质上相同(3.70% vs 3.63%),独立印证了无各向异性结果——但在额外大脑上符号翻转。
- **来源:** `run-4--ground-truth-error-annotations-revised-version_2026-06-20` (id 21) — 仅此一份报告发现。最新运行。
- **结论:** 433,243 条 Z 主导边(错误率 3.70%)对比 975,802 条 XY 主导边(错误率 3.63%)的卡方检验给出 χ² = 3.5328,p = 0.0602——未达显著。信念从 Likely True (0.8333) 降为 Uncertain (0.4688)。DOES-NOT-GENERALIZE:在 794491 上 Z 错误率显著更高(χ²=166.12,p=5.21e-38),在 794495 上显著更低(χ²=395.76,p=4.62e-88)。修正后的 GEE OR(Z vs XY) = 1.0184 [0.9712, 1.0680],p = 0.45 在 origin 上;UPHELD 仅在 origin 上"无效应",而两个额外大脑指向相反方向。
- **沿用判定:** Reproduction REPRODUCED;Generalization DOES-NOT-GENERALIZE;Verdict CRITICAL;Post-correction verdict UPHELD on origin only。
- **为何独有/新:** 是发现 #2(id 27)的配套发现;较旧报告完全未测试成像轴各向异性。
- **Caveats:** 边缘 p = 0.0602 被误读为反驳 H₁("未拒绝 = 零假设为真"谬误);未通过 BH-FDR(阈值 0.0444);来自同一神经元的边违反独立性,使有效 n 被夸大。

### 4. (Priority 0.265 · Surprise 0.414) 离心分支阶数终究可以预测 split 错误,深阶分支错误率突破 3%——但在两个额外大脑上符号都翻转。
- **来源:** `run-4--ground-truth-error-annotations-revised-version_2026-06-20` (id 139) — 仅此一份报告发现。最新运行。
- **结论:** 对 1,409,045 条边的 logistic 回归给出 branch_order 系数 = +0.0194,z = 16.37,p < 0.001(origin 上,阶数越深越易 split)。Generalization DOES-NOT-GENERALIZE:在 794491 上系数 = −0.0157(p < 0.001),在 794495 上系数 = −0.0063(p < 0.001)——均在相反方向上显著。带 neuron 聚类的修正后 GEE 给出 origin 上系数 = +0.01943,p = 0.038(勉强显著;OR = 1.0196 [1.001, 1.039]),在 794491 上方向相反,p = 0.034;在 794495 上 p = 0.58(NS)。
- **沿用判定:** Reproduction REPRODUCED;Generalization DOES-NOT-GENERALIZE;Verdict CRITICAL;Post-correction verdict WEAKENED on origin and OVERTURNED on extras。
- **为何独有/新:** 较旧报告未分析离心分支阶数。
- **Caveats:** `norm_thickness` 方差为零被剔除,因此"独立于 cable 粗细"的限定条件从未被实测;深阶峰值依赖于稀疏的高阶分箱;机制性因果措辞并不被这种观察性回归所支持。

### 5. (Priority 0.253 · Surprise 0.284) 融合 3 个及以上 GT 神经元的"超合并"覆盖的 GT cable 是 2 神经元合并的 ~5×,印证最严重的 merge 错误是巨型结构。
- **来源:** `run-4--ground-truth-error-annotations-revised-version_2026-06-20` (id 24) — 仅此一份报告发现。最新运行。
- **结论:** 27 个合并 segment 入选(24 个两神经元,3 个超合并)。覆盖 cable 的中位数:2 神经元为 6.21 mm,超合并为 35.19 mm(Mann-Whitney U = 0.0,p = 5.89e-03)。按每神经元计算为 ~11.7 mm/neuron vs ~3.1 mm/neuron。信念从 Leaning True (0.7083) 移至 Likely True (0.8906)。Generalization PARTIAL:794495 强力印证(n=24+2,p=6.15e-03,超合并中位数 168.42 mm);794491 没有超合并(检验无法进行);origin 单脑 rerun 给出 n=8+1,p=0.222(NS,U=0 是地板效应)。
- **沿用判定:** Reproduction DIVERGED(数据集范围,单脑 pkl 仅有 8+1);Generalization PARTIAL;Verdict MAJOR;Post-correction verdict WEAKENED。
- **为何独有/新:** 较旧报告未按融合神经元数对 merge 分组。
- **Caveats:** 功效严重不足(n=3 超合并,U=0 是地板);该检验比较的是总 cable(部分上有同义反复——融合更多神经元在机械上就跨越更多 cable),而非该假设字面上提出的每神经元数量。

### 6. (Priority 0.253 · Surprise 0.284) Omission 错误在拓扑分支点处的发生频率是线性 cable 节点的 ~4×。
- **来源:** `run-4--ground-truth-error-annotations-revised-version_2026-06-20` (id 32) — 仅此一份报告发现。最新运行。
- **结论:** 分支节点 omit 率 11.95%(606/5,072)vs 线性节点 omit 率 2.76%(38,610/1,398,807),χ² = 1567.69,p ≈ 0。模型在复杂结点处明显会丢失 fragment。在所有三个大脑上泛化:794491 7.54% vs 3.41%(比率 2.2×,χ²=193.63,p=5.13e-44);794495 6.52% vs 1.72%(比率 3.8×,χ²=974.86,p=5.24e-214)。信念从 Leaning True (0.7083) 移至 Likely True (0.8906)。
- **沿用判定:** Reproduction REPRODUCED;Generalization GENERALIZES;Verdict MINOR。
- **为何独有/新:** 较旧报告测试的是 OMIT 集中在 LEAVES 附近(H11、H64),而非分支点;SPLIT vs branch (H10) 是不同的错误类型。较新报告的 id 32 是首次对 OMIT vs 分支节点的测量。
- **Caveats:** 同一神经元内的边/节点不是 i.i.d.(未对聚类建模);分支节点 (5,072) 比线性节点 (1.4M) 稀有得多;"复杂分支结构导致脱漏"的因果措辞仅为观察性。

### 7. (Priority 0.253 · Surprise 0.284) 被 omit 掉的 cable 系统性地比正确重建的 cable 更靠近 merge 位点——但在 cluster 修正下被 OVERTURNED。
- **来源:** `run-4--ground-truth-error-annotations-revised-version_2026-06-20` (id 36) — 仅此一份报告发现。最新运行。
- **结论:** 在 49,295 个 omit 节点对比长度匹配的 49,295 个正确节点的随机样本上,到 67 个 merge 位点中最近者的距离中位数为 1,812.73 µm(omit)vs 1,959.88 µm(correct),单侧 Mann-Whitney U = 1.138e9,p = 1.60e-66。信念从 Leaning True (0.7083) 移至 Likely True (0.8906)。Generalization DOES-NOT-GENERALIZE:在 794491 上 omit 中位数 1103.24 µm > correct 842.11 µm(方向翻转,假设方向下 p=1.000);794495 印证(1422.53 vs 1670.94 µm,p=9.78e-226)。修正后的按神经元 cluster-bootstrap:差距 CI [−401.82, +91.98] µm 跨过零,p = 0.27(NS)。
- **沿用判定:** Reproduction REPRODUCED;Generalization DOES-NOT-GENERALIZE;Verdict CRITICAL;Post-correction verdict OVERTURNED。
- **为何独有/新:** 较旧报告未测试 omit 与 merge 的空间共定位。
- **Caveats:** 效应极小(在 ~1,900 µm 基线上 ~147 µm 差距,~7.7%);显著性源于把 49k 个空间相关节点当作 i.i.d. 处理;机制性主张"模型在融合时牺牲相邻细突起"是过度解读。

### 8. (Priority 0.253 · Surprise 0.284) 在 fragments 图上的启发式 A\*(角度 + 半径惩罚)修复了 86% 的 split 边而不引入 merge。
- **来源:** `run-4--ground-truth-error-annotations-revised-version_2026-06-20` (id 39) — 仅此一份报告发现。最新运行。
- **结论:** 对 6,805 个目标 split,该智能体为其中 5,881 个找到了有效(no-merge)路径——每个 split 成功率 86.42%,达到 40% 阈值的两倍。端到端 Edge Accuracy 从 78.71% 提升到 79.13%(净提升 +0.42%)。信念从 Leaning True (0.7083) 移至 Likely True (0.8906)。在所有三个大脑上泛化:794491 成功率 84.72%,EA 增益 +1.18%;794495 成功率 89.71%,EA 增益 +0.53%。
- **沿用判定:** Reproduction REPRODUCED;Generalization GENERALIZES;Verdict MINOR。
- **为何独有/新:** 较旧报告的 merge-cut H39 是其配套发现,但此处通过 A\* 解决的是 SPLIT 而非通过图切断解决 MERGE。相同 id 编号,不同干预,不同错误类型。
- **Caveats:** 成功率没有统计检验或 CI;"不引入 merge"依赖 GT——在没有 GT 的部署中,no-merge 无法保证;+0.42% EA 微弱;>40% 这一阈值非常宽松易被超过。

### 9. (Priority 0.253 · Surprise 0.284) 合并性 segment 表现为"巨型"组件——平均 cable 长度比非合并 segment 大 ~30×。
- **来源:** `run-4--ground-truth-error-annotations-revised-version_2026-06-20` (id 43) — 仅此一份报告发现。最新运行。
- **结论:** 64 个合并 segment 的平均长度 ~15,449 µm(中位数 ~3,099 µm),vs 8,273 个非合并 segment 的平均 ~532 µm(中位数 ~102 µm);对 log-cable-length 的 Welch's t = 16.54,p = 7.32e-25。信念从 Leaning True (0.7083) 移至 Likely True (0.8906)。Generalizes:794491 23×,t=29.03,p≈0;794495 34×,t=24.02,p≈0。
- **沿用判定:** Reproduction REPRODUCED;Generalization GENERALIZES;Verdict OK。
- **为何独有/新:** 较旧报告测量了合并 segment 的分支密度(H53),但未测量其绝对 cable 长度。
- **Caveats:** 部分上同义反复——融合多个神经元的 segment 必然跨越它们——但 ~30× 量级才是信息丰富的主张;样本量不对称 64 vs 8,273 由 Welch's 校正处理。

### 10. (Priority 0.253 · Surprise 0.284) Omit 错误在成像体积的极端 Z 深度比中央深度高 >2×。
- **来源:** `run-4--ground-truth-error-annotations-revised-version_2026-06-20` (id 45) — 仅此一份报告发现。最新运行。
- **结论:** 极端 Z 的 omit 率 4.44%(2,283/51,369)vs 中央 Z 1.94%(11,342/585,909);Cochran-Mantel-Haenszel 合并 OR = 2.3561,p ≈ 0。信念从 Leaning True (0.7083) 移至 Likely True (0.8906)。Generalization PARTIAL:794495 OR=3.1672(p≈0,强力印证);794491 OR=1.0149(p=0.80,无效应)。在 origin 上带 neuron 聚类的修正 GEE 给出 OR = 2.3561 [0.4558, 12.1801],p = 0.31(NS,一旦尊重聚类)。
- **沿用判定:** Reproduction REPRODUCED;Generalization PARTIAL;Verdict MAJOR;Post-correction verdict WEAKENED。
- **为何独有/新:** 较旧报告未测试深度极端的体积效应。
- **Caveats:** "按大脑 ID 分层"在单脑 pkl 上等同于无操作;分箱定义(顶/底 10% vs 中央 20%)是任意的;机制性的"光学衰减"主张被 794491 反驳。

### 11. (Priority 0.253 · Surprise 0.284) 短 omission 缺口通常是单个预测 segment 内部的丢失;长缺口是真正的 fragment 边界。
- **来源:** `run-4--ground-truth-error-annotations-revised-version_2026-06-20` (id 58) — 仅此一份报告发现。最新运行。
- **结论:** 307 条 bridged omit 路径(均值 18.71 µm,中位数 13.55 µm)vs 4,298 条 broken omit 路径(均值 42.54 µm,中位数 20.16 µm);Mann-Whitney U = 458,554,p = 1.95e-19。短 omit 大多为网络内部的丢失,提示是安全的自动填充目标。信念从 Leaning True (0.7083) 移至 Likely True (0.8906)。在所有三个大脑上泛化(中位数 10.99–13.55 vs 15.23–20.16 µm;所有 p ≤ 2.75e-09)。
- **沿用判定:** Reproduction REPRODUCED;Generalization GENERALIZES;Verdict MINOR。
- **为何独有/新:** 较旧报告未按 bridged-vs-broken 预测段连续性对 omit 缺口分组。
- **Caveats:** 定义耦合——"bridged" 要求同一预测 segment 上有足够多的非 omit 邻居,已部分编码了路径长度;中位数比率适中(~1.5×);显著性部分来自 n=4,605 的总样本量。

### 12. (Priority 0.287 · Surprise 0.307) 神经元级 split 率与 omit 率呈强正相关,提示存在共享的上游失效模式。
- **来源:** `ground-truth-error-annotations-revised-version_2026-06-17` (id 30) — 仅此一份报告发现。较旧运行。
- **结论:** 在 12 个超过 50 µm 长度阈值的神经元上,Pearson r = 0.65(p = 0.022),Spearman ρ = 0.881(p = 1.53e-04),OLS R² = 0.422。信念从 Leaning True (0.7083) 移至 Likely True (0.9327)。Generalizes:794491(n=9)Spearman ρ=0.983,p=1.94e-06;794495(n=19)Spearman ρ=0.740,p=2.89e-04。修正后:Spearman ρ 被提升为头条(bootstrap CI [0.5596, 0.9923],permutation p=3.00e-04,Kendall τ=0.7273 p=4.99e-04)。Post-correction verdict UPHELD。
- **沿用判定:** Reproduction REPRODUCED on stats;Generalization GENERALIZES;Verdict MINOR;Post-correction verdict UPHELD。
- **为何独有/新:** 只有较旧报告执行了对 split 率与 omit 率的逐神经元相关。较新报告没有重新审视该问题。
- **Caveats:** n=12 个神经元极小;Pearson 对离群点敏感(codeOutput 标记出 Y≈12.6 附近一个高杠杆点);"提示共享的上游失效模式"是机制性解释,n=12 的相关分析无法将其单独分离。

---

## 共同印证的发现(合并)

### 13. (Priority 0.287 · Surprise 0.307) Merge 错误集中在 fragment-graph 局部密度异常高 / 分支多的"缠结"结构区域。
- **来源:** `ground-truth-error-annotations-revised-version_2026-06-17` (ids 3 [10 µm density]、23 [15 µm density]、49 [distance-to-nearest-fragment-branch]、53 [merge segment 的分支密度]、81 [体积密度]);`run-4--ground-truth-error-annotations-revised-version_2026-06-20` (id 84 [15 µm 局部分支密度,控制组的 ~28×]) — 2 份报告,6 条条目。
- **结论:** Merge 位点周围的 fragment-graph 节点与分支结构远比匹配的控制组更密集。最受充分验证变体的标准数字(较旧的 H3):67 个 merge 位点 10 µm 范围内的局部节点密度均值 = 7.03 vs 控制组 4.39(Mann-Whitney U = 3931,p = 8.99e-15,~60% 提升)。15 µm 范围(H23):11.09 vs 6.33(U = 4100.5,p = 8.95e-17,75% 提升)。到最近分支的距离(H49):merge 位点中位数 4.48 µm vs 随机 179.76 µm(p = 2.21e-17);~40% 恰好与某个分支重合,~86% 在 10 µm 范围内。Merge segment 自身的每 100 µm cable 分支密度高 ~2×(H53:0.108 vs 0.057,p = 2.91e-25)。体积公式(H81):0.001678 vs 0.001083 nodes/µm³(Welch t = 8.93,p = 2.77e-14)。较新报告 H84 的配对设计:在同 segment 控制下 1.13 vs 0.04 branches/15 µm,paired t = 13.35(p = 2.03e-20),AUC = 0.9236。
- **一致:** 两份报告独立地通过多种指标和半径(10 µm、15 µm、体积、同 segment 配对)在所有三个大脑上印证。效应量比率:节点密度提升 41% 到 84%(较旧);分支密度 ~28×(较新配对)。修正后的非参数 Cliff's delta(较旧 H81 修正)= 0.67 到 0.78,跨所有三个大脑。
- **分歧:** 方向上无分歧;较旧报告 H49 的控制构造(按构造为非分支随机节点)被标记为接近循环论证,但 merge 侧的效应如此之大,结论仍然成立。
- **沿用判定:** 所有条目 Reproduction REPRODUCED;6 条条目在所有三个大脑上 Generalization GENERALIZES;Verdicts OK / MINOR;Post-correction verdict UPHELD(若已修正)。
- **Caveats:** Merge 侧 n 较小(origin 67,extras 86、105);较旧报告的 5 种密度变体并非统计上独立(相同的 merge 位点,只是不同的半径/体积公式)——应当算作 1 个有效信号;较新报告 H84 的 AUC 是基于配对控制的样本内 AUC。

### 14. (Priority 0.287 · Surprise 0.307) Split 错误集中在拓扑分支点附近(测地与欧氏距离都是如此)——方向稳健,量级在额外大脑上衰减。
- **来源:** `ground-truth-error-annotations-revised-version_2026-06-17` (ids 13 [测地,去重]、19 [欧氏]、72 [拓扑]);`run-4--ground-truth-error-annotations-revised-version_2026-06-20` (id 64 [测地,均值 516 vs 711 µm]) — 2 份报告,4 条条目。
- **结论:** Split 边比正确边显著更接近 GT 分支点。标准数字(去重 origin H13):Mann-Whitney U = 2.98e9,p = 1.32e-197,n = 1,109,034 correct vs 6,805 split;中位数差 = −184.19 µm。欧氏(H19):split 均值 275.76 µm vs correct 381.98 µm(p = 1.39e-229)。拓扑(H72):均值 518.39 vs 712.57 µm(p = 0)。较新报告 H64(同样总体上的测地距离):均值 516.30 vs 710.59 µm(p = 1.32e-197)。经 cluster-bootstrap 修正后(H13/H72/H64),origin Cliff's delta = −0.22 [−0.28, −0.13],在 extras 上衰减为 −0.08 到 −0.13;在所有大脑上逐神经元配对 Wilcoxon p ≤ 0.027。
- **一致:** 方向(split 更靠近分支)在所有三个大脑上、跨测地/欧氏/拓扑距离公式都得以保持。Cluster-correct 后的效应量"小但稳健"(Cliff's delta 0.08–0.22)。
- **分歧:** 量级。较旧 H13 因记录的 n 是真实 n 的 2 倍(重复 glob 导致双重计数)被标记为 MAJOR;较旧 H72 标记为 MAJOR/WEAKENED,因为 origin 上的 ~184 µm 差距在 794491 上塌缩到 ~39 µm、在 794495 上塌缩到 ~30 µm,而 floor-p 始终为 0。较新 H64 因相同的 n 驱动膨胀被标记为 MAJOR,但修正后以"小效应"量级 UPHELD。较旧 H19 标记为 MINOR。
- **沿用判定:** Reproduction REPRODUCED(并暴露出 H13 双重计数 bug);所有条目在所有三个大脑上 Generalization 方向上 GENERALIZES;Verdicts MINOR / MAJOR;Post-correction verdict 在尊重效应量(小 Cliff's delta)的前提下 UPHELD。
- **Caveats:** 同一骨架内的边不独立——合并后的 p 值高估了证据强度;较新 H64 中"紧密约束在分支点区域"的措辞对 794491 上严重衰减的效应是夸大其词。

### 15. (Priority 0.287 · Surprise 0.307) Split 错误在空间上聚集 / 级联——观察到的 split 间距远短于零假设,且 split 边的 split 邻居数是正确边的 ~10×。
- **来源:** `ground-truth-error-annotations-revised-version_2026-06-17` (id 55 [KS vs uniform null,split 间距短 9×]);`run-4--ground-truth-error-annotations-revised-version_2026-06-20` (id 37 [30 µm 内 split 邻居多 10×]) — 2 份报告。
- **结论:** Split 不是独立事件,而是局部聚集。较旧 H55:观测 split 间距中位数 25.58 µm vs 零假设 236.98 µm(KS D = 0.454,p ≈ 0,~9.3× 短于均匀)。较新 H37:30 µm 内 split 边的其他 split 邻居均值为 0.97,而匹配的正确边为 0.10(Mann-Whitney U = 3.47e7,p ≈ 0;~9.7× 比率)。
- **一致:** 两份报告独立印证 origin 上 split 局部聚集 ~10×;两者在所有三个大脑上泛化(H55 比率 6.0×–14.3×;H37 比率 5.0×–11.5×)。两者的信念变化:Leaning True → Likely True。Cluster-correct(较旧 H55):三个大脑上逐神经元 obs/null 中位数比率 0.07–0.13;matched-pairs r_rb = −1.0(每个神经元都展现 obs<null);每个大脑上的 cluster-permutation p ≤ 0.031。
- **分歧:** 方向或数量级效应上无分歧。
- **沿用判定:** 两者 Reproduction REPRODUCED;两者 Generalization GENERALIZES;Verdicts MINOR;Post-correction verdict(较旧 H55)UPHELD。
- **Caveats:** 较旧 H55 的均匀零假设是粗略选择(未考虑 cable 密度梯度);"级联"这种因果措辞超越了空间聚集所能确立的内容。较新 H37 在设计上部分是自相关("split 与 split 聚集")。

### 16. (Priority 0.287 · Surprise 0.307) Split 缺口端点的方向对齐是一个强且可利用的判别指标(反平行余弦 / 真实延续 ~153° vs 错误 ~90°)。
- **来源:** `ground-truth-error-annotations-revised-version_2026-06-17` (id 67 [余弦相似度,隐式 AUC]);`run-4--ground-truth-error-annotations-revised-version_2026-06-20` (id 33 [角度(度),AUC = 0.9322]) — 2 份报告。
- **结论:** Split 缺口端点表现出强烈的方向对齐,可区分真实延续与错误候选。较旧 H67:525 个 split 对的平均余弦相似度为 −0.685,vs 2,689 个空间相邻但拓扑不相连的控制为 −0.511(Mann-Whitney U = 594,814,p = 1.13e-08)。较新 H33 在 13,582 个 split 节点配置上:真实延续角度均值 = 152.96°,vs 错误的 90.17°(KS = 0.7460,p ≈ 0;ROC AUC = 0.9322)。
- **一致:** 两份报告印证一个可作为 bridging 启发式的尖锐方向特征;两者在所有三个大脑上泛化。较新 H33 各大脑 AUC ≥ 0.9281;较旧 H67 的效应在 extras 上增强(split −0.593 到 −0.791 vs 控制 −0.333 到 −0.527)。
- **分歧:** 较旧 H67 措辞较保守,称之为"判别指标而非清晰的分类器"(与控制组在 ~ −0.51 处重叠),而较新 H33 表述为"强且可靠"(AUC 0.93 毫不含糊)。较新 H33 的更紧操作定义(来自不同神经元的真实延续 vs 错误候选)解释了其更干净的 AUC。
- **沿用判定:** 两者 Reproduction REPRODUCED;两者 Generalization GENERALIZES;两者 Verdicts OK。
- **Caveats:** 较新 H33 的"错误候选"在操作上定义为来自不同神经元的邻近边——可行性主张依赖于该操作定义在推断时无 GT 的情况下仍能获得。

### 17. (Priority 0.287 · Surprise 0.307) Split 错误倾向于发生在更细 / 更远端的突起上(比正确边更靠近 leaf)。
- **来源:** `ground-truth-error-annotations-revised-version_2026-06-17` (id 60 [distance-to-leaf,origin 上短 ~344 µm],id 65 [拓扑距 leaf,origin 上短 38%]) — 仅较旧报告显式测试了 split-vs-leaf;较新报告未直接测量。但是,较旧报告的 H17(distance-to-leaf)和 H65(中位数短 38%)从两个角度描述同一性质并互相印证。
- **结论:** Split 边比正确边更接近 GT 叶子节点。较旧 H60:split 均值 938.60 µm vs correct 均值 1282.69 µm(Mann-Whitney p = 2.16e-134)。较旧 H65:split 中位数 417.22 µm vs correct 中位数 671.61 µm(短 38%,p = 0)。在所有三个大脑上方向上泛化(差距 28–344 µm),但量级在 extras 上塌缩到 13–15%。
- **一致:** 两条较旧条目(H60 与 H65)本质上是不同下采样下的同一测量(distance-to-leaf);都印证方向。
- **分歧:** 跨报告间无分歧,因为只有一份报告测试了它。在较旧报告内部,H60 与 H65 被标记为统计上不独立。
- **沿用判定:** Reproduction REPRODUCED(H60 精确,H65 因无种子下采样有漂移);Generalization GENERALIZES;两者 Verdicts MINOR。
- **Caveats:** Distance-to-leaf 是半径/粗细的代理而非直接测量;H65 在无固定 RNG 种子的情况下下采样使得不同运行间 U 统计量变化;量级是大脑特异的。

### 18. (Priority 0.287 · Surprise 0.307) Omit 错误集中在末端 / 叶子结尾边而非内部边上(~1.3×–1.8× 比率)。
- **来源:** `ground-truth-error-annotations-revised-version_2026-06-17` (id 64 [terminal vs internal,1.82× 比率]);`run-4--ground-truth-error-annotations-revised-version_2026-06-20` (id 85 [terminal vs internal,1.82× 比率]) — 2 份报告。
- **结论:** 末端边 omit 率 ≈ 4.38% vs 内部边 omit 率 ≈ 2.41%(χ² = 4189.94,p < 1e-300;比率 1.82×)。两份报告在 origin 上数字完全一致(相同的边总体和划分)。在所有三个大脑上泛化:794491 比率 1.29×(p = 1.63e-94);794495 比率 1.36×(p = 1.66e-152)。两份报告中信念都从 Leaning True 移至 Likely True。
- **一致:** 两份报告间数字字节级一致。两者都将方向描述为稳健。较旧 H64 经 cluster-bootstrap 修正后:rate ratio origin 1.81 [1.35, 2.44],794491 1.29 [0.96, 1.89](CI 跨过 1!),794495 1.36 [1.14, 1.59]。
- **分歧:** 方向上无分歧;"几乎是两倍"这一头条对 origin 和 794495 是大脑特异的(在 794491 上降到 1.3×);较旧 H64 在 794491 上的 cluster-bootstrap CI 跨过 1.0。
- **沿用判定:** 两者 Reproduction REPRODUCED;两者 Generalization GENERALIZES;两者 Verdicts MINOR;Post-correction verdict(较旧)WEAKENED——方向成立但 794491 量级界跨过 1。
- **Caveats:** 相邻末端边共享叶子且不独立;机制性"远端细突起"框架未被直接检验。

### 19. (Priority 0.287 · Surprise 0.307) Omit 错误在拓扑上集中在 GT 神经元的末端叶子附近(多源 BFS 距离)。
- **来源:** `ground-truth-error-annotations-revised-version_2026-06-17` (id 11) — 主源;由较新报告 id 85 在不同粒度上(terminal vs internal 边)印证。较旧 H11 度量单个 omit 边的连续 distance-to-leaf;较新 H85 划分为 terminal/internal。基础性质相同。
- **结论:** Omit 边到 leaf 的拓扑距离均值为 247.18(中位数 130.5),而正确边为 325.08(中位数 168.5);单侧 Mann-Whitney U = 2.22e10,p ≈ 7.75e-311(origin)。Generalization PARTIAL:在 794491 上合并 MWU 方向翻转(omit 159.76 vs correct 138.76,p=1.000);794495 印证(omit 124.85 vs correct 172.81,p=6.10e-149)。修正后的逐骨架配对 Wilcoxon(H11):origin 12/12 个骨架显示 omit 更近(p=2.44e-04,CI [−90.51, −29.83] µm);794491 8/9 个骨架(p=1.37e-02,CI [−21.45, −4.89] µm);794495 13/19 个骨架(p=8.77e-03,CI [−28.92, −6.16] µm)。Post-correction verdict 反转为 UPHELD/GENERALIZES。
- **一致:** 较旧 H11 与较新 H85(terminal-vs-internal 划分)都指向末端/远端 cable。两份报告中信念变化都是 Leaning True → Likely True。
- **分歧:** 较旧 H11 在 794491 上的合并 MWU 方向翻转最初被标记为 MAJOR(DOES-NOT-GENERALIZE),但检验修正将其反转为 GENERALIZES。一旦尊重骨架级依赖,"远端细突起偏向"这一框架成立。
- **沿用判定:** Reproduction REPRODUCED;Generalization 原本是 MAJOR / "DOES-NOT-GENERALIZE",但 Post-correction verdict UPHELD,在 cluster-aware 配对检验下"GENERALIZES"。
- **Caveats:** 较旧 H11 记录的 p ≈ 0 是样本量驱动的地板效应;extras 上修正后的逐骨架配对效应量较小(~12–17 µm 偏移)。

### 20. (Priority 0.287 · Surprise 0.307) Split 错误在与分支点相邻的边上发生频率是线性边的 ~3.4×。
- **来源:** `ground-truth-error-annotations-revised-version_2026-06-17` (id 10) — 仅此一份报告专门针对 SPLIT 错误进行了发现。较新报告的 id 32 度量了类似的 OMIT 效应(上述发现 #6)。两者都将分支邻近边作为失效位点,但针对不同错误类型——保留为独立发现(split vs omit)。较旧 H10 在此处归入共同印证,因为它是较旧报告其他所有 split-near-branch 发现(H13、H19、H72)和较新 H64 的直接分类对应——即发现 #14 的分类 / 分支边公式。
- **结论:** 分支边 split 率 1.62%(248/15,298)vs 线性边 0.47%(6,557/1,393,747);χ² = 414.5,p = 3.9e-92。在所有三个大脑上泛化:794491 比率 2.07×;794495 比率 2.00×(方向从不翻转)。信念 Leaning True → Likely True。修正后的 cluster-aware 检验:origin rate ratio = 3.4458 [2.7788, 4.2931](cluster-permutation p=1.996e-03);794491 2.0556 [1.7943, 2.4236];794495 2.0087 [1.6411, 2.4111]。
- **一致:** 在所有三个大脑上、在合并和 cluster-aware 检验下,方向与量级都稳健;在分类(边类)层面印证发现 #14。
- **分歧:** 无。
- **沿用判定:** Reproduction REPRODUCED;Generalization GENERALIZES;Verdict MINOR;Post-correction verdict UPHELD。
- **Caveats:** 共享分支节点的边并非严格独立;巨大的 n 使 p 不再有信息——应该以 rate ratio(2.0×–3.4×)作为效应量。

### 21. (Priority 0.287 · Surprise 0.307) 段间 split 缺口具有紧密、特征性的长度尺度(~5.8 µm 95 分位数,~6.5 µm 99 分位数,100% 低于 15 µm)。
- **来源:** `ground-truth-error-annotations-revised-version_2026-06-17` (id 35 [中位数 19.67 µm,缺口比内部边大 3–4×]);`run-4--ground-truth-error-annotations-revised-version_2026-06-20` (id 63 [100% 低于 15 µm;95% 在 ~5.85 µm 之下;99% 在 ~6.50 µm 之下]) — 2 份报告。
- **结论:** Split-bridging 缺口长度短且特征明确。较旧 H35:4,078 个 origin 缺口,中位数 19.67 µm;内部边长度 95 分位数仅 5.77 µm(Mann-Whitney p ≈ 0,缺口中位数与内部 95 分位数的比率 3.4×)。较新 H63:6,805 个 split 缺口的 100.00% 低于 15 µm;95% 由 5.85 µm 覆盖,99% 由 6.50 µm 覆盖。两者在所有三个大脑上以近乎相同的分位数泛化(95 分位数 5.79–5.85 µm;99 分位数 6.41–6.50 µm)。
- **一致:** 两份报告独立印证真实 split 缺口的紧密特征尺度;较旧报告的中位数 19.67 µm 与较新报告的 100%-低于-15-µm + 99 分位数 6.5 µm 互相一致(中位数不同是因为较旧 H35 在中位数中计入了长尾缺口,而较新 H63 报告的是 ECDF 分位数)。
- **分歧:** 头条解释。**较旧 H35 得出"固定半径最近邻启发式将失败"** 的结论,因为缺口分布(中位数 19.67 µm,尾部到 250 µm)远比 intra-fragment 边宽(95 分位数 5.77 µm);任何能覆盖 split 的半径都会扫到假阳性。**较新 H63 得出相反结论:紧密的 ~6.5 µm 半径覆盖 99% 的真实 split。** 这种矛盾可以由控制组的选择来调和——H35 与 intra-fragment 内部边比较(后者更短,因此仅靠距离无法把它们与 split 区分开),而 H63 报告的是缺口的绝对分布。然后较新发现 #1(id 30)通过表明仅靠距离就能以 AUC 0.998 区分真实 split 与 inter-NEURON 缺口,明确解决了这个问题。
- **沿用判定:** 两者 Reproduction REPRODUCED;两者 Generalization GENERALIZES;两者 Verdicts OK。
- **Caveats:** 较旧 H35 的机制性警告("固定半径启发式失败")是以特定控制集为条件的;较新报告重新评估并部分反转了该结论。

### 22. (Priority 0.287 · Surprise 0.307) 在预测的 merge 坐标处进行的几何定向图切断,在将 Edge Accuracy 从 82.3% 提高到 93.7% 的同时,移除了 ~86% 的 merge 边。
- **来源:** `ground-truth-error-annotations-revised-version_2026-06-17` (id 39) — 仅较旧报告得出此具体的 MERGE-CUT 结果。较新报告的 id 39(虽 id 相同,但是 A\* SPLIT-REPAIR 发现,即上文 #8)不印证此 merge-cut 结果。
- **结论:** 通过 KDTree 切断离每个预测 merge 坐标最近的物理 fragments-graph 节点,在 12 个神经元上将全局 mean %-merged-edges 从 13.25% 降到 1.83%(降幅 ~86%),并将 mean edge accuracy 从 82.27% 提到 93.66%;被合并最严重的神经元(N013、N018)从 ~40–43% 合并降至 ~1% 合并。Generalization PARTIAL:794491 从 20.54% 降到 6.15%(~70% 降幅,低于 80% 头条阈值);794495 28.54% → 16.02%(~44% 降幅)。修正后的逐神经元配对 Wilcoxon:每个大脑 p ≤ 2e-3;merged-edges 总分数被移除的 CI = origin [0.7007, 0.9631]、794491 [0.5164, 0.8555]、794495 [0.2264, 0.6461](即使在 origin 上 CI 下界也低于 80%)。
- **一致:** 在本语料中为单一来源,但定性主张(定向切断总是显著减少 merge 并提升准确率,每个神经元都改善)是稳健的。
- **分歧:** 尽管 id 编号相同,较新报告的 "id 39" 是一种不同的干预(A\* 修复 split,而非 merge 切断),并报告了远小得多的端到端 EA 增益(+0.42% vs 较旧的 +11.4%)。它们是不同的可行动原语。
- **沿用判定:** Reproduction REPRODUCED;Generalization PARTIAL;Verdict MAJOR(降级——≥80% 的定量主张无法迁移);Post-correction verdict WEAKENED。
- **Caveats:** 头条 ">80% merge 边降幅"是大脑特异的;应改述为"取决于大脑 44–86% 降幅"。该干预是针对 GROUND-TRUTH merge 坐标评估的,而非实际部署中所需的预测 merge 坐标。

### 23. (Priority 0.287 · Surprise 0.307) Omit 错误沿 GT 骨架形成连续的连串,邻居条件概率为边际率的 ~28×。
- **来源:** `ground-truth-error-annotations-revised-version_2026-06-17` (id 48) — 仅此一份报告直接发现。与较新报告 id 58(bridged vs broken omit 缺口,上文发现 #11)紧密相关,后者也描述了 omit 的连续条带结构。
- **结论:** 边际 P(OMIT) = 0.0317;P(OMIT | neighbor = OMIT) = 0.8908——比率 28.09×,远高于假设的 3× 阈值(χ² = 2.23M,p ≈ 0)。Omit 不是独立事件。Generalizes:794491 比率 18.63×;794495 比率 40.42×(所有 p ≈ 0,所有 >> 3×)。修正后的无序对卡方 = 1,113,217.84(恰为记录的一半,印证了对称双重计数 bug);origin 上比率的 cluster-bootstrap CI [16.85, 47.40];每个大脑上 cluster-permutation p = 3.3e-03。
- **一致:** 这一结构模式也隐含在较新报告 id 58 关于"短 bridged omit 路径"作为内部丢失存在的发现中(连串性的一种表现)。视作独立发现,是因为仅在较旧报告中出现的 H48 直接度量了 Markov 风格的转移概率。
- **分歧:** 无。
- **沿用判定:** Reproduction REPRODUCED;Generalization GENERALIZES;Verdict MINOR;Post-correction verdict UPHELD。
- **Caveats:** 记录代码中的对称无向对计数使卡方加倍(并不使比率加倍);骨架内邻接违反独立性,但 28× 比率对样本量不敏感。

### 24. (Priority 0.287 · Surprise 0.307) Split 边的局部 tortuosity 高于正确重建边(小但稳健的效应)。
- **来源:** `run-4--ground-truth-error-annotations-revised-version_2026-06-20` (id 59 [10-hop window];id 73 [5-hop window]) — 均在较新报告中;较旧报告未印证。在此列出是因为这两条较新报告条目明确互相交叉印证(id 73 被表述为对 id 59 的"独立再确认")。
- **结论:** Split 边比正确边表现出统计显著更高的 tortuosity,但每条边的效应很小。10-hop(H59):中位数 1.1115 vs 1.0764(point-biserial r = 0.0335,p ≈ 0)。5-hop(H73):中位数 1.1132 vs 1.0801,U = 4.71e9,p = 1.66e-276。在所有三个大脑上以同等小量级泛化(r = 0.025–0.061)。修正后的 Cliff's delta 在所有三个大脑上为 +0.21 到 +0.28(按 Vargha-Delaney 为"小");每个大脑上逐神经元配对 Wilcoxon p ≤ 2e-3。
- **一致:** 5-hop 与 10-hop 变体在所有三个大脑上给出数字上一致的结论。
- **分歧:** 跨报告间无分歧,因为只有较新报告测试了 tortuosity。
- **沿用判定:** 两者 Reproduction REPRODUCED;两者 Generalization GENERALIZES;两者 Verdicts MAJOR(大 n 微小效应显著性陷阱);Post-correction verdict WEAKENED (id 59) / UPHELD (id 73)——方向被确认,效应量是"小但真实"。
- **Caveats:** id 73 重用了与 id 59 相同的数据集和近乎相同的指标——并非真正独立;tortuosity 窗口重叠(相邻边共享 4–9 跳),违反独立性;每边 r ≈ 0.03 意味着仅靠 tortuosity 的分类器在实践中无用。

### 25. (Priority 0.287 · Surprise 0.307) Split 缺口连接服从非线性距离-角度权衡,使固定阈值启发式失效——但不泛化。
- **来源:** `ground-truth-error-annotations-revised-version_2026-06-17` (id 47) — 仅较旧报告。在概念上被较新报告 id 30(发现 #1)和 id 33(发现 #16)取代,这两者共同表明仅靠距离(AUC 0.998)和仅靠角度(AUC 0.93)在与 inter-neuron 控制组比较时各自都能作为判别指标——即较旧报告"比固定阈值更丰富的几何"的主张被较新报告"简单阈值终究可行"所取代。
- **结论:** 在 1,072 对真实 split 和 142 个假合并控制上的 logistic 回归给出 distance×angle 交互项系数 = −5.96,p = 0.014(LLR p = 1.63e-17,pseudo R² = 0.093)。决策边界在 ~8 µm 之下容忍 >90° 转折,但到 15 µm 时塌缩至 0° 共线。信念从 Leaning True (0.6667) 移至 Likely True (0.9231)。Generalization DOES-NOT-GENERALIZE:794491 x3 系数 = −0.342,p = 0.800;794495 x3 系数 = −1.454,p = 0.565。修正后的 LR-vs-additive 检验:origin LR=6.6454,p=9.94e-03(在 origin 上 UPHELD);extras p_LR=0.797 和 p_LR=1.000(不能迁移)。
- **一致:** 在本语料中为单一来源。
- **分歧:** 与较新报告隐式分歧:较新的 id 30 和 id 33 表明简单的仅距离和仅角度启发式在与 inter-neuron 控制组比较时都能工作(AUC 0.93–0.998)。较旧报告"比固定阈值配对规则更丰富的几何"这一主张仅在 origin 上、仅针对 False-Merge 控制存活。
- **沿用判定:** Reproduction REPRODUCED;Generalization DOES-NOT-GENERALIZE;Verdict MAJOR;Post-correction verdict UPHELD(在 origin 上)但不能迁移。
- **Caveats:** 类别不平衡 ~7.5:1(1,072 正例 / 142 控制)夸大了交互项 SE;"非线性权衡"是从单一大脑泛化到通用规则。

### 26. (Priority 0.253 · Surprise 0.284) Split 边在空间上比正确边更靠近 merge 位点(~170 µm)——但在 cluster 修正下被 OVERTURNED。
- **来源:** `run-4--ground-truth-error-annotations-revised-version_2026-06-20` (id 61) — 仅较新报告。与独有发现 #7(id 36,omit 靠近 merge)对称,且两者在修正后都达到 OVERTURNED。归入共同印证一节是因为它与发现 #7 配对,共同定义了较新报告中一个统一的(虽然被否定的)"联合失效区"假设。
- **结论:** Split 到 merge 的均值距离 2,084.76 µm(中位数 1,794.64 µm)vs correct 均值 2,265.14 µm(中位数 1,956.93 µm);在 6,805 个 split 边和 1,109,034 个正确边上 Mann-Whitney U p = 3.50e-38,Welch's t p = 7.14e-27。信念从 Leaning True (0.7083) 移至 Likely True (0.8906)。Generalization 在所有三个大脑上方向上 GENERALIZES(在所有大脑上 split 都更近;差距在 794491 上缩小到 ~29 µm)。修正后的按神经元 cluster-bootstrap:origin 上差距 CI [−373.03, +15.55] 跨过零,p = 0.1067(NS);在修正检验下三个大脑中无一达到 cluster-significance。
- **一致:** 与发现 #7(id 36)在"联合失效区"框架下配对;两者都是样本量驱动,两者都未通过 cluster 修正,且都有一个大脑显示方向反转或缺失。
- **分歧:** 跨报告无(单一来源),但 Post-correction verdict OVERTURNED 与原始头条不一致。
- **沿用判定:** Reproduction REPRODUCED;Generalization GENERALIZES(未修正)但 Post-correction verdict OVERTURNED。
- **Caveats:** ~170 µm 差距在 ~2,000 µm 基线上(~8% 相对偏移);显著性源于把 1.1M 条边当作 i.i.d.;一旦尊重聚类,机制性的"联合失效区"主张就不再成立。

---

## 因冗余排除

以下条目被并入上述标准发现(每个集群中的非标准成员)。以 `<run abbreviation> id <id>` → 并入发现 #N 的形式列出。

**较旧报告(`ground-truth-error-annotations-revised-version_2026-06-17`):**
- id 23(merge 位点 15 µm fragment 密度,提升 75%)→ 并入发现 #13(merge-crowding 集群)。
- id 49(~86% merge 位点位于 fragments-graph 分支点 10 µm 内)→ 并入发现 #13。
- id 53(导致 merge 的 fragment 段分支密度 ~2×)→ 并入发现 #13。
- id 81(merge 位点体积 fragment-node 密度高 ~55%)→ 并入发现 #13。
- id 19(split 错误与分支节点空间相关,均值距离短 ~106 µm)→ 并入发现 #14(split-near-branch 集群)。
- id 72(split 错误在拓扑上偏向 GT 分支节点,中位数 ~一半)→ 并入发现 #14。
- id 65(split 边发生在更细的突起上 — 到 leaf 拓扑距中位数短 38%)→ 并入发现 #17(split-distal 集群)。

**较新报告(`run-4--ground-truth-error-annotations-revised-version_2026-06-20`):**
- id 84(merge 位点带有强几何"缠结"特征 — 局部分支密度高 ~28×)→ 并入发现 #13。
- id 64(split 错误聚集在拓扑分支点附近,均值测地 ~516 µm vs ~711 µm)→ 并入发现 #14。
- id 73(独立再确认:split 边的 5-hop 局部 tortuosity 更高)→ 并入发现 #24(仅较新独有 tortuosity 集群)。
- id 85(末端边上 omit 错误几乎是内部边的两倍)→ 并入发现 #18(末端 omit 集群;与较旧 id 64 数字一致)。

未折叠其他条目;每条剩余条目都自成发现,因为 (a) 它检验的是不同的错误类型(split vs merge vs omit),(b) 它检验的是不同的机制(例如距离 vs 角度 vs 分支阶数 vs Z 轴 vs 深度),或 (c) 它达到的判定不被另一条目所涵盖。
