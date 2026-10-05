"""Separate recorded image access, scoring coverage and paired TRAIN evidence."""
import json


def generation_image_evidence(directory, record):
    analyses, unreadable = [], 0
    for path in sorted(directory.glob('volume_analyses/analysis*/result.json')):
        try:
            analyses.append(json.loads(path.read_text()))
        except (OSError, ValueError):
            unreadable += 1
    attempts = []
    for entry in record.get('experiments', []):
        cells = (entry.get('train') or {}).get('cells', {})
        coverage = {name: cell['image_context'] for name, cell in cells.items()
                    if name.endswith('/' + entry.get('target_kind', '')) and cell.get('image_context')}
        if coverage and entry.get('status') == 'evaluated':
            attempts.append({'experiment': entry['experiment'], 'cached': entry.get('cached', False),
                             'cells': coverage})
    diagnostics = []
    for report in record.get('feature_diagnostics', []):
        image = report.get('image_evidence', {})
        if image.get('image_inputs_present'):
            diagnostics.append({key: report.get(key) for key in (
                'experiment', 'status', 'protocol', 'delta_precision', 'program_sha256',
                'parameters_sha256', 'feature_columns', 'signature', 'image_evidence')})
    return {'version': 'image-evidence-v1',
            'volume_analyses_attempted': len(analyses) + unreadable,
            'volume_analyses_succeeded': sum(a.get('status') == 'analyzed' for a in analyses),
            'unreadable_analysis_records': unreadable,
            'preview_records': len(list(directory.glob('image_inspections/inspection*.json'))),
            'image_scored_attempts': attempts, 'paired_diagnostics': diagnostics,
            'scope': 'Successful access and image-scored attempts do not establish image benefit. '
                     'Only image-only paired diagnostics isolate a TRAIN increment at their exact coverage; '
                     'promotion alone does not attribute validation gain to images.'}
