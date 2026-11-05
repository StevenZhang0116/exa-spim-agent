# Merge review page

Builds a self-contained `index.html` for manually labeling detected merge sites. Each site is one pre-rendered image (raw MIPs on top, skeletons below) with a sidebar for labels and notes. Labels are saved in the browser and exported as CSV.

One page is built per run (config). Each page holds every evaluation step for that run.

## Files

| File | Role |
| --- | --- |
| `build_review.py` | Reads the CSVs and SWC zips, pulls raw crops from the zarr volume, renders the images, and writes `index.html`. |
| `template.html` | The page (CSS, markup, JS). Contains two placeholders the script fills: `/*__FONTS__*/` and `/*__DATA__*/`. |

## Requirements

- Python packages: `numpy`, `pillow`, `scipy`, `tensorstore`, `fonttools`, `matplotlib` (only for its bundled DejaVu fonts, which get subset and embedded).
- Read access to the raw volume from this machine (Google application default credentials). The script opens `VOLUME` with TensorStore and reads one small crop per site.

## Quick start

```bash
cd /root/capsule/code/review

# 1. Render a few images and look at them (writes PNGs, no HTML)
python build_review.py --cfgs it2 --steps original --limit 3 --preview --out /tmp/preview

# 2. Small end-to-end build to check the page
python build_review.py --cfgs it2 --limit 8 --out /tmp/test

# 3. Full build, all runs, all steps
python build_review.py --out /root/capsule/results/merge_review
```

Output is `<out>/<cfg>/index.html`, one per run. The script prints the site count, unique image count, JPEG quality and file size for each page.

| Flag | Meaning |
| --- | --- |
| `--out` | Output directory. Default `/root/capsule/results/merge_review`. |
| `--cfgs` | Runs to build. Default is all in `CFGS`. |
| `--steps` | Steps to include. Default is all in `STEPS`. |
| `--limit N` | Sites per step, for quick tests. |
| `--preview` | Write the first few PNGs and stop. |
| `--template` | Use a different template file. |

## Expected input layout

```
DATA/eval_<BRAIN>_<cfg>/<one result folder>/<step>/
    <step>_merge_sites.csv                  # any file matching *_merge_sites.csv
    <step>_fragments_with_merges.zip        # merge-N.swc, ground-truth N0xx.0.swc, <fragment>.0.0.swc
```

Merge-site CSV columns, read by position: `Merge_ID, Fragment_Name, Segment_ID, GroundTruth_ID, Label, Voxel, World, Added Cable Length`.

- `Voxel` is `(z, y, x)`.
- `World` is `(x, y, z)` in µm.
- SWC data columns 2 to 4 are world x, y, z in µm, even though the file header says "z y x".

## Reusing it for other results

Edit the constants at the top of `build_review.py`.

| Constant | Change it when |
| --- | --- |
| `DATA`, `BRAIN`, `CFGS`, `STEPS` | You point at a different results folder, brain, set of runs, or set of steps. `STEPS` is `(folder name, label shown in the page)`. |
| `VOLUME` | The raw image lives somewhere else. It must be a zarr array ordered `(t, c, z, y, x)`. |
| `VOX_XYZ` | The voxel size is different (µm per voxel, x y z). |
| `HALF_UM`, `PPU` | You want a wider view or a sharper image. Image size is `PPU * 2 * HALF_UM` pixels per panel. |
| `ALSO_IN_UM` | You want a different distance for "also flagged by" (sites in other runs at the same step). |
| `RAW_PCT` | The raw row looks too dark or too washed out. Each site's raw MIPs are clipped to these percentiles (pooled over the three panels) and stretched to black and white. |
| `RADIUS_SCALE`, `MIN_WIDTH` | Skeleton lines look too thick or thin. Radius in the SWC is scaled by `RADIUS_SCALE` before drawing. |
| `BUDGET_BYTES` | The target page size changes. See the size limit below. |

Colors in the images: red is the fragment at the site, cyan is the ground-truth tracing, the yellow ring marks the site. Other fragments are not drawn. If you change them in `C_*`, update the legend in `template.html` (`renderLegend`, and the `--c-*` CSS variables).

To add a label such as "unsure" or remove one, edit `LABELS` in `template.html`. The summary counts, the filters (`FILTERS`), the CSV values, and the key bindings all derive from it, except the CSS classes for `.lab.<k>` and `.strip div.<k>`, which need a matching color.

To use a different design, give `--template` your own HTML file. It needs `/*__DATA__*/` where the JSON goes (as in `const D = /*__DATA__*/;`) and may use `/*__FONTS__*/` for the embedded fonts.

## What the page does

- Dropdowns: step, ground-truth neuron, filter (all, unreviewed, hide false positives, only false positives, only true merges, only unsure), and sort (added length, recurrence across configs, neuron). The config and step dropdowns are hidden when there is only one.
- Keys: `f` false positive, `t` true merge, `u` unsure, `c` clear, `←`/`a` previous, `→`/`d` next, `Esc` leaves the note box. Labeling advances to the next site.
- A "Raw contrast" slider (1× to 3×) under the image brightens the raw row only. It runs in the browser on a canvas, so the skeleton colors and the baked images are unchanged. The setting is remembered with the view. Letter keys still work while the slider has focus; arrow keys move the slider.
- Labels and notes are stored in `localStorage` under `<dataset>_flags_v1`, keyed by `cfg|step|id|segment`. The dataset name is `merge_review_<BRAIN>_<cfg>`, so each run keeps its own labels. The last view is stored under `<dataset>_view_v1`.
- **False positives as SWC** lists each step with its false-positive count and a download button. Each file has one node per site (`id 2 x y z 10 -1`, world µm, in site order) and is named `<dataset>_<step>_false_positive_merges.swc`. Where downloads are unavailable the button reads "Show SWC" and fills the text box instead.
- **Show CSV** fills a text box. **Download CSV** saves through the artifact runtime and is disabled where that is unavailable. The CSV has one row per site on the page, with a blank label for unreviewed sites. Columns: `config, step, id, segment, gt, label, note, voxel_zyx, world_xyz, also_in`. `step` is dropped when the page has one step. Label values are `false_positive`, `true_merge`, `unsure`.

## Publishing as an artifact

Publish each `index.html` with the Artifact tool, giving each run its own file path so each gets its own URL. The file is an HTML fragment (no doctype or body tags), which is what the tool expects. It also opens directly in a browser.

The download button uses `claude.use("downloads")`, which requires the artifact to be published with the `downloads` capability. If the page is published without it, the button is disabled and the CSV can still be copied from the text box.

## Size limit

Each page must stay under about 16 MB. Images are embedded as base64 JPEGs, and `BUDGET_BYTES` (14 MB) caps the image payload. The script does three things to fit:

1. Identical renders are stored once, so the same spot in several steps shares one image.
2. JPEG quality is lowered from 86 in steps of 4 until the payload fits.
3. If quality 42 is still too large, the images are shrunk and the search restarts.

If a page ends up too soft, build fewer steps (`--steps`) or split the run into several pages.

## Checking a build

Open the page in a headless browser and confirm:

- Every site's image decodes.
- Each filter and sort returns the expected counts and order.
- Labels and notes survive a reload.
- The exported CSV agrees with the stored labels.
- There are no console errors, and the page does not scroll sideways at 390 px wide.

A headless Chromium from a Playwright virtual environment is enough:

```bash
python3 -m venv /tmp/pw && /tmp/pw/bin/pip install playwright
/tmp/pw/bin/playwright install chromium --with-deps
```

## Notes

- The `FFTM NOT subset` lines printed during the build come from font subsetting and are harmless.
- Raw crops are cached by voxel position, so the same spot in several steps is read once.
- A site within `HALF_UM` of the volume edge is padded with zeros rather than skipped.
