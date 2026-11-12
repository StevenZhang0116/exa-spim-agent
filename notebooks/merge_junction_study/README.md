# Merge junction study (2026-11-11)

Offline checks behind `proofreader_evolve/TODO.md` section 7: why the merge scorers separate merge
junctions from true branch points so poorly, and whether the merge labels are part of the reason.
Every script opens the six TRAIN brains only (802449, 747807, 750318, 754610, 754612, 754613), is
read-only for caches, tables, runs and `metrics_out`, and writes to a folder of its own name beside
it (gitignored). Run on a compute node in the `panda` environment from the repository root; logs go
to `notebooks/log/`.

**All stored outputs describe the `length_v1` labels**, the caches as they were until the relabel
of 2026-11-11 (policy `length_v2`, `labeled_cache/policy.py`). Results are recorded in the TODO
section; this file only says what each script is for.

| Script | Question | Result in one line |
|---|---|---|
| `merge_junction_fragment_context.py` | Do fragment size and the size of each arm's subtree separate the classes? | No consistent signal (scorer 0.690 to 0.702 with fragment totals; signs flip between brains). |
| `merge_junction_label_localization.py` | Do the labels agree with the tracing around each candidate? Also draws example sheets with the image (`examples` stage, cloud access). | 91% of positives sit where the tracing shows the fragment leaving or joining a neuron; the disagreeing negatives are explained by the site rule. |
| `merge_junction_hidden_neuron.py` | How many negatives away from any tracing are really merges? Estimated with candidates on one GT neuron as if it were not traced. | 4 of 164 (2.4%, 1.0 to 6.1): the off-tracing negatives are kept as 0. |
| `merge_canonical_site_labels.py` | Which candidates sit at a canonical (segmentation-skeleton-metrics) merge site without a positive label? | 90 of 537 canonical sites had candidates all labeled 0 (92 candidates). |
| `merge_site_walk_replay.py` | Why does the cache record no site there? Replays the metric's walk on the cache graphs. | The port pooled all GT neurons and did not move a site to a branching node; replayed as in the metric, 51 of the 90 get a site. |
| `merge_walk_port_check.py` | What does the repaired port change, computed in memory before any cache is written? | Junction positives 370 to 426, canonical sites with a positive candidate 358 to 406 of 537. |

Shared helpers: the later scripts import from `merge_junction_fragment_context.py` (brain list,
logging, the accepted scorer's holdout scores) and `merge_junction_label_localization.py` (walks and
GT readings), so run them from this folder's path as shown in each docstring.

Decisions that followed (user, 2026-11-11): repair the port and relabel with `length_v2`; keep the
off-tracing negatives; do not use the canonical sites as a label source; report merge statistics
per candidate family without changing the objective.
