"""What a tool reply to the reviser session carries inline, and what stays in a file.

The CLI hands a session a tool result above about 50 KB of text as a 2 KB preview plus the path
of a file holding the rest. In the three runs of 2026-11-12 (``notebooks/feedback_use_audit.py``)
105 of 111 replies of the training tools arrived that way, the sessions read the file for 38 of
them, and the per-family statistics, the other rank statistics and the ranking changes reached
the session in about a third of the replies. Raising ``MAX_MCP_OUTPUT_TOKENS`` (2026-10) does
not move that threshold.

Since 2026-11-13 every reply therefore fits ``REPLY_BUDGET_CHARS``: it carries the conclusions
and names the file that holds the detail. Nothing is withheld; the files are the ones the
harness already wrote (an attempt's ``feedback.json``, a descriptor run's or probe's
``result.json``), inside the directories a session may read.

This module only shapes replies. The records the harness keeps (ledger, memory, result files)
are written from the full values before these functions are applied.
"""
from copy import deepcopy
import json

VERSION = 'compact-tool-replies-v1'
# The observed threshold is between 52 and 55 KB of reply text; the budget leaves room for the
# research receipt appended to every reply and for growth of a record.
REPLY_BUDGET_CHARS = 30_000
RECEIPT_CHARS = 500         # the research receipt the tool layer appends to every reply
OUTPUT_TAIL_CHARS = 2_000   # of a successful fit; a failed fit keeps the whole recorded tail
# search_memory: tried in order until the reply fits (characters of each diff, evidence records
# kept, hypothesis and strategy cut). Few experiments keep everything; five keep a shorter diff.
MEMORY_STEPS = ((3_000, True, False), (3_000, False, False), (2_000, False, False), (800, False, True))
MEMORY_TEXT_CHARS_TIGHT = 600

# The conclusions of an attempt come first, so that even a preview would show them.
ATTEMPT_FIRST = ('experiment', 'status', 'error', 'failure_stage', 'cached', 'cached_from', 'target_objective',
                 'target_precision', 'in_sample_precision', 'train_gate', 'selection_gate', 'selection_comparison')
FEEDBACK_INLINE = ('format', 'target_kind', 'candidate_train', 'candidate_selection', 'ranking_delta',
                   'columns_pruned', 'examples_omitted_cells')
COMPARISON_SUMMARY = ('diagnostic_only', 'reference', 'objective', 'aggregation', 'overall', 'holdout',
                      'overall_precision', 'holdout_precision', 'warnings')
TRAINING_FILE_ONLY = ('implementation_sha256', 'artifact_files')
MEMORY_KEYS = ('experiment', 'status', 'error', 'hypothesis', 'strategy', 'family', 'search_mode', 'search_parent',
               'cached_from', 'classifier', 'parameters', 'target_objective', 'target_precision',
               'in_sample_precision', 'train_gate', 'selection_gate', 'selection_comparison', 'feature_evidence',
               'snapshot', 'diff_file', 'code_diff', 'diff_truncated', 'hypothesis_evidence')


def size(value):
    return len(json.dumps(value, allow_nan=False))


def over_budget(value):
    """Whether a reply would exceed the budget once the receipt is appended."""
    return size(value) > REPLY_BUDGET_CHARS - RECEIPT_CHARS


def without_cells(report):
    """A report's per-kind figures; the per-brain cells stay in the file."""
    return {key: value for key, value in report.items() if key != 'cells'} if isinstance(report, dict) else report


def comparison_summary(comparison):
    """A selection comparison without its per-brain rows."""
    if not isinstance(comparison, dict):
        return comparison
    return {key: comparison[key] for key in COMPARISON_SUMMARY if key in comparison}


def feedback_digest(feedback, path):
    """The part of one attempt's feedback that a reply carries; ``path`` holds all of it."""
    digest = {key: (without_cells(feedback[key]) if key in ('candidate_train', 'candidate_selection') else feedback[key])
              for key in FEEDBACK_INLINE if key in feedback}
    held = sorted(set(feedback) - set(digest)) + [f'{key}.cells' for key in ('candidate_train', 'candidate_selection')
                                                  if isinstance(feedback.get(key), dict) and 'cells' in feedback[key]]
    digest.update(file=str(path), file_holds=held,
                  note='This reply carries the conclusions. The file holds the rest of this attempt\'s feedback: the '
                       'example rows with their candidate_refs, the per-brain cells, reading_guide, k_rule and '
                       'selection_protocol. Read it when you need them.')
    return digest


def training_digest(training):
    """A fit's summary without the hash tables; a successful fit's output is cut to its tail."""
    if not isinstance(training, dict):
        return training
    digest = {key: value for key, value in training.items() if key not in TRAINING_FILE_ONLY}
    tail = digest.get('training_output_tail')
    if isinstance(tail, str) and len(tail) > OUTPUT_TAIL_CHARS:
        digest['training_output_tail'] = tail[-OUTPUT_TAIL_CHARS:]
        digest['training_output_tail_cut'] = f'last {OUTPUT_TAIL_CHARS} characters; worker.log in fit_snapshot holds the rest'
    return digest


def attempt_reply(entry, feedback, path):
    """The reply for one measured attempt: its record, conclusions first, with the feedback digest."""
    public = {key: entry[key] for key in ATTEMPT_FIRST if key in entry}
    public.update(entry)
    for key in ('train', 'selection'):
        if key in public:
            public[key] = without_cells(public[key])
    if 'training' in public:
        public['training'] = training_digest(public['training'])
    public['feedback'] = feedback_digest(feedback, path)
    return public


def grid_reply(response):
    """The reply of a grid (train_classifier, search_parameters): one short row per configuration,
    the restored best in full. Each row's per-brain comparison is in its attempt's feedback.json."""
    response = dict(response)
    response['candidates'] = [{**row, 'selection_comparison': comparison_summary(row.get('selection_comparison'))}
                              for row in response['candidates']]
    response['candidates_note'] = ('Each row carries the overall and holdout comparison. The per-brain comparison of the '
                                   'restored candidate is in best; that of any other row is in its attempt\'s '
                                   'experiments/<attempt>/feedback.json (selection_comparison.per_brain).')
    # A long grid: shorten the rows further rather than lose the reply to the size limit.
    for dropped in ('timings', 'selection_comparison', 'train_gate', 'program_sha256', 'selection_gate'):
        if not over_budget(response):
            break
        response['candidates'] = [{key: value for key, value in row.items() if key != dropped}
                                  for row in response['candidates']]
        response.setdefault('candidates_trimmed', []).append(dropped)
    return response


def _rounded(value):
    return None if value is None else float(f'{float(value):.4g}')


def column_digest(summaries):
    """Per descriptor column, across brains: support, range and medians (by label when summarised).

    ``summaries`` is {brain: {column: diagnostics}} as ``descriptor_probe.diagnostics`` returns it."""
    digest = {}
    for brain, columns in summaries.items():
        for column, record in columns.items():
            item = digest.setdefault(column, {'finite_fraction_min': 1.0, 'min': None, 'max': None, 'constant_in': [],
                                              'median': {}})
            item['finite_fraction_min'] = min(item['finite_fraction_min'], float(record.get('finite_fraction') or 0.))
            for name, pick in (('min', min), ('max', max)):
                if record.get(name) is not None:
                    item[name] = _rounded(record[name] if item[name] is None else pick(item[name], record[name]))
            if record.get('constant'):
                item['constant_in'].append(brain)
            quartiles = record.get('q25_q50_q75') or [None] * 3
            item['median'][brain] = _rounded(quartiles[1])
            for label in ('0', '1'):
                if isinstance(record.get(label), dict):
                    quartiles = record[label].get('q25_q50_q75') or [None] * 3
                    item.setdefault(f'median_label{label}', {})[brain] = _rounded(quartiles[1])
    for item in digest.values():
        item['finite_fraction_min'] = _rounded(item['finite_fraction_min'])
    return digest


def descriptor_reply(public, result_file):
    """The reply of compute_descriptors: one line per column instead of one record per column and brain."""
    public = dict(public)
    if isinstance(public.get('summaries'), dict):
        public['column_summary'] = column_digest(public.pop('summaries'))
        public['summaries_file'] = str(result_file)
        public['summaries_note'] = ('column_summary gives each column\'s lowest finite fraction, range, the brains where it '
                                    'is constant and its median per brain, overall and by label. The file holds the full '
                                    'diagnostics per brain (counts, quartiles, example candidate_refs).')
    return public


def probe_reply(response, result_file):
    """The reply of probe_descriptor: the column digest and the cases, values rounded."""
    public = deepcopy(response)
    brains = public.get('brains') or {}
    summaries = {brain: record.pop('summaries') for brain, record in brains.items() if 'summaries' in record}
    if summaries:
        public['column_summary'] = column_digest(summaries)
    for record in brains.values():
        for case in record.get('cases', []):
            case['values'] = {column: _rounded(value) for column, value in case.get('values', {}).items()}
    public['result_file'] = str(result_file)
    public['result_note'] = ('column_summary digests each column across brains; the file holds the full diagnostics per '
                             'brain and the *_values.npz files beside it every sampled value.')
    # Many columns: keep one case per brain, then none, rather than lose the reply to the size limit.
    for keep in (1, 0):
        if not over_budget(public):
            break
        for record in brains.values():
            record['cases'] = record.get('cases', [])[:keep]
        public['cases_trimmed_to_per_brain'] = keep
    return public


def memory_reply(entries):
    """The reply of search_memory: what an earlier experiment was, how it did and its diff."""
    def digest(entry, diff_chars, evidence_kept, tight):
        row = {key: entry[key] for key in MEMORY_KEYS if key in entry}
        row['selection_comparison'] = comparison_summary(row.get('selection_comparison'))
        text = row.get('code_diff') or ''
        if len(text) > diff_chars:
            row.update(code_diff=text[:diff_chars], diff_truncated=True)
        evidence = row.get('hypothesis_evidence')
        if isinstance(evidence, dict):
            row['hypothesis_evidence'] = {key: evidence[key] for key in ('handoff', 'records', 'records_omitted')
                                          if key in evidence}
        if not evidence_kept and row.pop('hypothesis_evidence', None) is not None:
            row['hypothesis_evidence_note'] = 'Left out to fit the reply; hypothesis_memory.json holds the records.'
        if tight:
            for key in ('hypothesis', 'strategy'):
                if isinstance(row.get(key), str) and len(row[key]) > MEMORY_TEXT_CHARS_TIGHT:
                    row[key] = row[key][:MEMORY_TEXT_CHARS_TIGHT] + ' [cut; see result.json]'
        row['note'] = 'result.json and feedback.json under snapshot hold the full record of this experiment.'
        return row

    for step in MEMORY_STEPS:
        rows = [digest(entry, *step) for entry in entries]
        if not over_budget(rows):
            break
    return rows


def reply_chars(response):
    """Characters of text a reply sends, whether it is a plain record or already SDK content."""
    if isinstance(response, dict) and isinstance(response.get('content'), list):
        return sum(len(block.get('text', '')) for block in response['content'] if block.get('type') == 'text')
    return size(response)
