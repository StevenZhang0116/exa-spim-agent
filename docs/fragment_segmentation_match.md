# Fragment / segmentation source agreement

Receipts of `python -m labeled_cache.source_check`: sampled interior fragment nodes read back from the
dense segmentation, agreement = exact + within-1-voxel matches over nodes read. Matched sources give
close to 100%; an unrelated segmentation run gives close to 0%.

## 2026-10-29T13:58:45-07:00 on hpc-code.corp.alleninstitute.org

`python -m labeled_cache.source_check --brains 794495 802449 789202 794493 794491 747807 754610 754612 754613 750318 751473 --record`

| brain | segmentation_id | fragments | nodes | exact | within 1 vox | background | other id | failed | agreement | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| 794495 | `raw.unet_449_792_202_splits_and_merges_831600` | 60 | 240 | 230 | 10 | 0 | 0 | 0 | 100.0% | matched |
| 802449 | `raw.unet_r3_ckpt_425250_seedfix_overlapfix` | 60 | 240 | 223 | 17 | 0 | 0 | 0 | 100.0% | matched |
| 789202 | `raw.unet_r3_ckpt_425250_seedfix_overlapfix_rempm` | 30 | 120 | 116 | 4 | 0 | 0 | 0 | 100.0% | matched |
| 794493 | `mean40.stddev105.mask.136168199.no_omitted_20k.ffn.mt_0.1` | 60 | 240 | 237 | 3 | 0 | 0 | 0 | 100.0% | matched |
| 794491 | `mean40.stddev105.mask.136168199.no_omitted_20k.ffn.mt_0.1` | 60 | 240 | 235 | 4 | 0 | 1 | 0 | 99.6% | matched |
| 747807 | `202501_73227862_855_mean80.mask.136168199.no_omitted_20k.ffn.mt_0.1` | 60 | 240 | 215 | 25 | 0 | 0 | 0 | 100.0% | matched |
| 754610 | `202503_73227862_855_mean80.mask.136168199.no_omitted_20k_v3.ffn.mt_0.1` | 60 | 240 | 203 | 37 | 0 | 0 | 0 | 100.0% | matched |
| 754612 | `202505_73227862_855_mean50.mask.136168199.no_omitted_20k.ffn.mt_0.1` | 60 | 240 | 209 | 30 | 1 | 0 | 0 | 99.6% | matched |
| 754613 | `202504_73227862_855_mean50.mask.136168199.no_omitted_20k.ffn.mt_0.1` | 60 | 240 | 193 | 42 | 3 | 2 | 0 | 97.9% | matched |
| 750318 | `202504_73227862_855_mean80_dynamic` | 53 | 212 | 180 | 32 | 0 | 0 | 0 | 100.0% | matched |
| 751473 | `202503_73227862_855_mean80.mask.136168199.no_omitted_20k_v3.ffn.mt_0.1` | 60 | 240 | 233 | 6 | 1 | 0 | 0 | 99.6% | matched |
