---
title: "Reducing interaction overhead and improving feature discovery"
date: 2026-10-02
bibliography: evolution_discovery_references.bib
---

The recommended change is to make a hypothesis and its executable experiment
batch the unit of agent interaction. Pair this with feature experiments that
measure additional value beyond the current representation. These are design
proposals for this repository, not measured improvements to its current pipeline.

This focused review covers primary work available by October 2, 2026. Asta search
was unavailable because its CLI was not authenticated; papers and metadata were
checked directly against PMLR and arXiv instead. This is not an exhaustive review.

The motivating run is `precision_20260930_215627_9t_i10um`: about 245 minutes,
138 fresh configurations, five promotions, and validation Precision@1000 improving
from 0.115875 to 0.1185. Timestamp accounting attributes approximately 150 minutes
to reviser-session time outside tool-call intervals, 59 minutes to tool execution,
and 31 minutes to outer validation. The first category includes inference,
generation, and SDK waiting; it is not an inference profiler. Only two local
neighborhood inspections were used, and no submitted scorer declared
`LOCAL_CONTEXT`. These are local observations from the saved run, not literature
results.

| Primary paper | Relevant evidence | Scope and limitation |
|---|---|---|
| [CodeAct, ICML 2024](https://proceedings.mlr.press/v235/wang24h.html) [@wang2024] | Executable code composes multiple operations and control flow within one action; the paper reports up to 30% fewer actions on M3ToolEval. | Tool-use benchmarks, not neuron ranking or guaranteed end-to-end speedup. |
| [LLMCompiler, ICML 2024](https://proceedings.mlr.press/v235/kim24y.html) [@kim2024] | Dependency-aware planning, dispatch, and execution reduce sequential function-calling overhead; reported latency gains reach 3.7x over ReAct. | Benchmark maximum, not a forecast for this run; dependencies still constrain execution. |
| [AlphaEvolve, June 2025 white paper](https://arxiv.org/abs/2506.13131) [@novikov2025] | Asynchronous sampling/evaluation, model mixtures, and an evolutionary database support throughput and diverse search. | Algorithm-discovery evidence; concurrency alone does not reduce total LLM work. |
| [GEPA, ICLR 2026](https://arxiv.org/abs/2507.19457) [@agrawal2026] | Execution feedback guides mutations, and per-instance Pareto selection preserves complementary candidates. Its selection ablation favors this over greedy and beam alternatives. | Primarily prompt optimization; fewer rollouts do not necessarily mean less wall time when LLM calls dominate. |
| [FAMOSE, February 2026 preprint](https://arxiv.org/abs/2602.17641) [@burghardt2026] | Feature proposals are evaluated with the existing selected features, followed by algorithmic feature selection. Classification ablations support going beyond selection alone. | Tabular ROC-AUC/RMSE, not Precision@K. It uses iterative ReAct and does not establish a latency advantage. |
| [Crafter, August 2026 preprint](https://arxiv.org/abs/2608.05207) [@wang2026] | Searches for features explaining a frozen forecaster's remaining errors; compositional and LLM proposals share one measured admission rule. | Forecasting rather than classification. Its residual-structure findings motivate an adaptation, not a demonstrated result for skeletons. |

1. **Execute complete experiment batches with fewer model round trips.**

   The repository already has parameter-grid tools in `search_parameters` and
   `train_classifier`. The missing abstraction is a complete experiment packet:
   an automatically assembled TRAIN diagnosis, candidate code and hypothesis,
   a bounded batch of configurations or ablations, and a compact measured result.
   The executor should own snapshots, numeric trials, restoration, and mechanical
   summaries. The agent can retain freedom over its formula, model, and extractor.
   Return to the model for a substantive decision or a bounded repair, rather
   than for each file-management step. This applies CodeAct/LLMCompiler principles
   above the batch tools that already exist.

   A useful proposed target is one hypothesis submission and one result-based
   revision per successful batch, with additional calls only when needed. This
   is a target to evaluate, not a hard cap on research or an expected speedup.
   Independent candidates could later run concurrently with isolated workspaces
   and a serialized archive update. First reduce redundant interactions; measure
   concurrency separately from total token use and cost.

2. **Discover features that address observed failures and prove incremental value.**

   Start with matched TRAIN cases: a missed positive and selected label-0 candidate
   with similar detector score and gap, plus representative cases outside that
   narrow boundary. Batch a small set of local skeleton neighborhoods into the
   diagnosis. The agent proposes a falsifiable distinction and executable feature
   code. An example hypothesis is that endpoint tangent agreement across several
   path lengths distinguishes otherwise similar split candidates. This is only a
   proposed hypothesis; no geometric pattern has been established by this review.

   Compare the same learner and evaluation protocol using existing features,
   existing features plus the proposed group, and a removal ablation. Rank features
   by incremental TRAIN-internal held-out performance, gained/lost positives, and
   extraction cost. Fit preprocessing and models within each training fold. Use
   fragment/spatial grouping where feasible; such checks still cannot establish
   cross-brain generalization. Retain the full candidate-pool ranking for final
   Top-K evaluation rather than reporting precision from a balanced case sample.

   Feature transformations may improve a learner's representation without adding
   observations. Local geometry offers measurements absent from the tabular view.
   Which source helps must be measured. Images and segmentation-volume cutouts
   require additional access infrastructure; the current interface provides
   processed fragment skeleton neighborhoods. Feature exploration and internal
   validation must use TRAIN only. The other brains remain with the outer evaluator.

3. **Remember evidence at the hypothesis level and schedule behavioral diversity.**

   Existing specialist and protected-reference branches already implement part of
   the diversity idea. Extend their records with the targeted failure pattern,
   information source, feature ablation, uncertainty, and measured conditions under
   which a hypothesis failed. Retrieve a short relevant record when proposing the
   next batch. A failed parameter setting does not disprove a whole hypothesis;
   record the scope of the evidence explicitly.

   Let stagnant families yield budget to a different feature source, mechanism,
   or complementary failure slice. Preserve candidates for measured complementary
   behavior, not merely different family names or source hashes. Identical TRAIN
   Top-K sets can still generalize differently, so behavioral similarity should
   inform scheduling rather than automatically prohibit outer evaluation. Keep
   reflection concise and event-triggered; another long conversation after every
   trial would work against the first objective.

Implement the experiment-packet interface first, then a controlled local-feature
batch, and finally hypothesis-level memory and scheduling. Future comparisons
should use the same seed policy, data split, model, and resource budget; report
wall time and tokens per distinct hypothesis, incremental feature gains, distinct
TRAIN rankings, and outer validation improvement over elapsed time. Compare
orchestration and feature-discovery changes separately before combining them.

No evolution, training, data preparation, or tests were run for this review. Only
this note and its bibliography were added. Citation keys were checked statically;
Quarto/Pandoc were unavailable, and no rendering or deployment setup was added.
