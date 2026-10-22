# A coding agent as the discovery engine (AutoDiscovery ablation)

`agentic/run_agent_discovery.py` replaces AutoDiscovery with a coding agent that
receives **exactly the AutoDiscovery materials** and returns an export in the
**same JSON shape**, so `run_discovery_workflow.py`, the detector build and every
audit read it unchanged. It is an ablation, not a new product: no baseline card,
no `canonical_stats` module, no extra brain, no feedback beyond what
AutoDiscovery also lacks.

Two engines share the files and the researcher instructions
(`.claude/agents/discovery-researcher.md`):

| `--engine` | session per batch | default model | cost source |
|---|---|---|---|
| `claude` | Claude Agent SDK, `discovery-researcher` subagent | `claude-opus-5-5` | the SDK's result message |
| `codex` | `codex exec --json` (OpenAI Codex CLI), instructions as the prompt | `gpt-6-astra` | tokens of its `turn.completed` events times `--price-in` / `--price-out` per million; unpriced (0) unless given, then only `--hypotheses` and `--max-hours` stop the run |

`--model` and `--effort` override both (`effort` becomes the SDK effort or
Codex's `model_reasoning_effort`). The Codex engine runs with
`--dangerously-bypass-approvals-and-sandbox` by default (`--codex-sandbox
bypass`): the Slurm job is the sandbox, the scripts need network for the image
volume and write under the work directory; `--codex-sandbox workspace-write`
keeps Codex's own sandbox instead.

## Materials

Linked by their AutoDiscovery names into `autodiscovery/<run>.<engine>/`:

- `task.md` = `markdowns/<kind>_features_for_proofreader_evolve.md`
- `dataset_cache_<brain>_mcl<N>_add.pkl`
- `rerank_companion_<brain>_<kind>.npz` and `.json`

## Loop

Hypotheses are produced in batches (`--batch 5`); every batch is one fresh
session, so context never accumulates. Before each batch the driver writes
`records/ledger.txt` (every hypothesis recorded so far, with prior and
posterior) so the session does not repeat them. For each id the agent writes
`records/<id>.proposal.json` (hypothesis, experimentPlan, prior) before any code,
writes `hypo_<id>.py`, runs it only through the driver-owned `run_hypo.sh`
(timeout, captured stdout/stderr, exit code, runtime), and writes
`records/<id>.result.json` (analysis, review, posterior). The driver stops at
`--hypotheses`, `--max-cost-usd` or `--max-hours`, whichever comes first, or when
a batch completes nothing; the same `--run-name` resumes after the last complete
id. The driver appends one line per batch to `agent_discovery.log.txt` in the
work directory; Codex sessions leave their JSONL events, last message and stderr
under `codex/` there.

## Export

`autodiscovery/<run>.json`: one record per proposed id with the AutoDiscovery
fields. `surprisal = (posterior - prior) x 1.5583` (the constant every record of
the 2026-10-19 export obeys), `isSurprising` when the shift is at least 0.25,
`priorBelief`/`posteriorBelief` as five-vote `boolean_cat` dicts, `status`
`SUCCEEDED` only when the script exited 0 and a result was recorded,
`codeOutput`/`runtimeMs`/`createdAt` from the runner. `manifest.json` in the work
directory records the engine, model, sessions, tokens, cost and agent time.

## Running

On a compute node under `panda` (every script loads the pkl; budget 40 GB). The
Claude engine needs `ANTHROPIC_API_KEY`; the Codex engine a logged-in `codex`
CLI on `PATH`. The run name carries the kind as one `-`/`_`-separated token (the
rule `run_discovery_workflow._is_split_run` applies):

```bash
python agentic/run_agent_discovery.py --engine claude --kind merge --brain 794495 \
    --run-name claude-merge-1_2026-10-22 --hypotheses 100 --batch 5 \
    --max-cost-usd 150 --max-hours 12
python agentic/run_agent_discovery.py --engine codex --model gpt-6-astra \
    --kind merge --brain 794495 --run-name codex-merge-1_2026-10-22 \
    --hypotheses 100 --batch 5 --max-hours 12
python agentic/run_discovery_workflow.py autodiscovery/claude-merge-1_2026-10-22.json \
    --pkl cache/dataset_cache_794495_mcl100_add.pkl --direction predictive \
    --extra-pkl cache/dataset_cache_802449_mcl100_add.pkl
```

`--assemble-only` rebuilds the export from the records without a session.
Compare the engines with each other and with the AutoDiscovery run of the same
specification through the same downstream workflow: SOUND/GAIN features,
cross-brain survivors, fabricated or substituted records, cost and wall time.
