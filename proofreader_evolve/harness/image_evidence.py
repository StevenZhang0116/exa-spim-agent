"""Separate recorded image access, scoring coverage and paired TRAIN evidence."""
import json


def descriptor_ablation_evidence(report, registry):
    """Record image-consuming descriptor provenance without inferring pixel causality."""
    evidence = dict(report.get('image_evidence') or {})
    columns = {column: registry[column] for column in report['feature_columns']
               if column in registry and registry[column]['inputs'] in ('image', 'both')}
    if columns:
        evidence.update(image_inputs_present=True, descriptor_columns=columns,
                        image_only_control=False, all_image_inputs_removed_in_control=False,
                        conclusion='descriptor_group_increment_not_isolated_to_pixels',
                        scope='Paired removal of declared descriptor columns. Image-consuming descriptors '
                              'also receive geometry/features; their joint effect does not isolate pixel contribution.')
    return evidence


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
    probes, plans = [], []
    for pattern, records in (('descriptor_probes/probe*/result.json', probes), ('descriptor_plans/plan*.json', plans)):
        for path in sorted(directory.glob(pattern)):
            try:
                records.append(json.loads(path.read_text()))
            except (OSError, ValueError):
                records.append({'status': 'unreadable'})
    descriptor_runs = [{key: run.get(key) for key in ('run', 'status', 'columns', 'scope', 'inputs', 'image_tier', 'image',
                                                     'cached', 'wall_seconds')}
                       for run in record.get('descriptor_runs') or [] if run.get('inputs') in ('image', 'both')]
    probe_previews = sum(len(p.get('previews', [])) for p in probes)
    return {'version': 'image-evidence-v3',
            'volume_analyses_attempted': len(analyses) + unreadable,
            'volume_analyses_succeeded': sum(a.get('status') == 'analyzed' for a in analyses),
            'unreadable_analysis_records': unreadable,
            'preview_records': len(list(directory.glob('image_inspections/inspection*.json'))) + probe_previews,
            'descriptor_probe_previews': probe_previews,
            'descriptor_probes_attempted': len(probes),
            'descriptor_probes_succeeded': sum(p.get('status') == 'probed' for p in probes),
            'descriptor_image_probes_succeeded': sum(p.get('status') == 'probed' and p.get('inputs') in ('image', 'both') for p in probes),
            'descriptor_plans_attempted': len(plans),
            'image_scored_attempts': attempts, 'paired_diagnostics': diagnostics,
            'descriptor_image_computations': descriptor_runs,
            'scope': 'Successful access and image-scored attempts do not establish image benefit. '
                     'Only image-only paired diagnostics isolate a TRAIN increment at their exact coverage; '
                     'promotion alone does not attribute validation gain to images. Pool-scale image descriptor '
                     'computations register predictors; their increment still needs a paired measurement.'}
