# On-demand descriptor contexts (2026-10-29)

Evolution now reads geometry and image crops on demand for any native candidate.
There is no precomputed context prerequisite. The default probe samples the full TRAIN
pool uniformly; `all_candidates` covers the full pool, `pilot` is uniform, and
`boundary` explicitly selects detector-rank neighbours. No-score/raw production still
requires full coverage. Legacy `all_cached` calls and registries normalize to
`all_candidates`; `--context-cache` is ignored. `--no-descriptors` disables the tools.

Probe, batch computation and frozen inference share descriptor_inputs. Legacy image
tier names resolve to physical requests when no offline tier exists. New descriptor
patches are temporary, while numeric descriptor-bank results remain reusable under
source/configuration/code identities. Source failures are explicit. The graph store
serializes cold loads so concurrent input batches share one graph. Validation logs
remain host-only; sampling and input assembly do not use labels.

The preparation script and preflight no longer build or require contexts. The human
error-context plot, fragment/image offset diagnostics and specification checker read live sources too. The old explicit
offline cache builder/reader remain compatibility utilities, outside the evolution
runtime. The derived `proofreader_evolve/context_cache/` data was deleted after
verification (3,707 files, 97.06 GiB); raw volumes, source graph caches, native tables and historical runs stay.

Full candidate pools are much larger than the former detector bands. Use the measured
planning estimates before expensive production calls; cold IO and graph loading count
toward descriptor wall time, and CPU count alone does not bound retrieval throughput.
