# Fragment length of split candidate pairs (2026-10-21)

Script: `notebooks/split_fragment_length.py` (`--measure` per brain on a compute node, `--plot`,
`--canonical` on the companion brains). Pool: the declared split pool `tip_to_any_node|r=80|k=2`,
enumerated through `export_rerank_companion.pool_only_bank` (same rows and labels as the
companions; pool and label digests checked against `notebooks/label_counts/`). Fragment length is
the cable length of a segment in the cached (smoothed, 2 um resampled) fragment graph, summed over
the graph's edges per connected component and then per segment id.

Measured so far: 794495, 802449 (companion brains) and 754613. The other eight brains:

    srun --jobid=<job> --overlap -n1 -c4 bash -lc 'cd exa-spim-agent && \
        /home/zihan.zhang/.conda/envs/panda/bin/python -u notebooks/split_fragment_length.py \
        --measure --brains 789202 794493 794491 747807 750318 751473 754610 754612'
    python notebooks/split_fragment_length.py --plot

## Distributions (median cable length, um)

| brain | class | rows | shorter fragment | longer fragment |
|---|---|---|---|---|
| 794495 | 1 split | 2,497 | 307 | 2,543 |
| 794495 | 0 GT-judged, no split | 3,579 | 308 | 1,997 |
| 794495 | NaN not judgeable | 683,601 | 148 | 304 |
| 794495 | all fragments | 366,904 | 160 | |
| 802449 | 1 | 6,535 | 339 | 2,527 |
| 802449 | 0 | 9,251 | 362 | 2,098 |
| 802449 | NaN | 1,270,731 | 137 | 263 |
| 802449 | all fragments | 737,866 | 145 | |
| 754613 | 1 | 1,048 | 159 | 461 |
| 754613 | 0 | 1,275 | 148 | 384 |
| 754613 | NaN | 104,753 | 119 | 275 |
| 754613 | all fragments | 95,617 | 137 | |

## Label 1 versus 0 (AUC, shorter length first; `auc.csv`)

| brain | shorter fragment, pooled | within gap tertiles | longer fragment, pooled | within gap tertiles | gap itself |
|---|---|---|---|---|---|
| 794495 | 0.503 | 0.487 / 0.508 / 0.462 | 0.463 | 0.452 / 0.494 / 0.528 | 0.883 |
| 802449 | 0.515 | 0.536 / 0.513 / 0.576 | 0.478 | 0.516 / 0.505 / 0.610 | 0.915 |
| 754613 | 0.455 | 0.482 / 0.436 / 0.507 | 0.445 | 0.477 / 0.427 / 0.542 | 0.899 |

## Canonical measurement (`canonical_<brain>.json`, per-fold logistic fits, raw gap baseline)

| brain | variant | AP | delta to raw gap | bootstrap 95% |
|---|---|---|---|---|
| 794495 | shorter fragment length alone | 0.3995 | -0.426 | |
| 794495 | both lengths | 0.4563 | -0.369 | |
| 794495 | refit control: gap alone | 0.8185 | -0.0070 | [-0.0133, -0.0003] |
| 794495 | gap + both lengths | 0.8242 | -0.0013 | [-0.0105, +0.0075] |
| 802449 | shorter fragment length alone | 0.4257 | -0.465 | |
| 802449 | both lengths | 0.4326 | -0.458 | |
| 802449 | refit control: gap alone | 0.8893 | -0.0011 | [-0.0024, +0.0001] |
| 802449 | gap + both lengths | 0.8832 | -0.0073 | [-0.0120, -0.0027] |

The positive share of the labeled rows is 0.41 on both brains, so length alone (AP 0.40 to 0.46)
ranks at chance. Added to gap, length is NULL on 794495 and a LOSS on 802449. `VARIANTS COMPARED 4`
per brain.

## Reading

1. Fragment length does not separate true splits from GT-judged non-splits: the pooled AUCs sit
   within 0.05 of 0.5, the matched-gap AUCs scatter on both sides of 0.5, and the canonical
   increment over the raw gap ranking is null or negative. Not a feature.
2. What length does separate is judgeable from unjudgeable rows: labeled pairs have fragments
   about twice as long (shorter member 307 versus 148 um on 794495, longer member 2,543 versus
   304 um), because a long fragment is more likely to touch a traced neuron (`segment_on_gt`).
   The canonical `label_shift` of the length-only scores is KS 0.16 to 0.50. Any feature that
   correlates with fragment length (node counts, branch counts, path lengths to a soma, total
   intensity sums) inherits this shift: it is measured out of fold on the long-fragment rows and
   deployed on the short-fragment majority. The Firewall rule of the split specification
   (labeled-versus-unlabeled share) already flags this; the length distributions here are the
   mechanism.
3. Side observation: 22.5% of the fragments of 754613 are shorter than 100 um in the cache
   (minimum 70 um) although the caches are `mcl100`. The loader applies the minimum to the raw
   swc path length and then smooths and resamples each irreducible edge
   (`GraphLoader.resample_curve_3d`), which shortens it. Cache lengths are therefore not the
   filter's lengths; thresholds on cable length must be read off the cache.
