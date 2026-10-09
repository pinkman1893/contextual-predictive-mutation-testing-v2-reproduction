"""Independent saved-result verification; --results-only requires only Python."""
import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def read_rows(out, name):
    with (out / name).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

def check_metrics(rows, field, threshold, positive, expected):
    cm = [[0, 0], [0, 0]]
    for row in rows:
        label = int(row['label_killed'])
        score = float(row[field])
        if label not in (0, 1) or not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError('Invalid label or probability')
        cm[label][int(score > threshold)] += 1
    tp, fp, fn = cm[positive][positive], cm[1-positive][positive], cm[positive][1-positive]
    p = tp/(tp+fp) if tp+fp else 0
    r = tp/(tp+fn) if tp+fn else 0
    values = dict(precision=p, recall=r, f1=2*p*r/(p+r) if p+r else 0,
                  accuracy=(cm[0][0]+cm[1][1])/len(rows))
    if expected['positive_label'] != positive or expected['n'] != len(rows):
        raise ValueError('Metric scope mismatch')
    if cm != expected['confusion_matrix_labels_0_1']:
        raise ValueError('Confusion matrix mismatch')
    for key, value in values.items():
        if not math.isclose(value, expected[key], rel_tol=0, abs_tol=1e-12):
            raise ValueError('Metric mismatch: '+key)
    return cm

def check_hash(path, expected):
    with path.open('rb') as f:
        if hashlib.file_digest(f, 'sha256').hexdigest() != expected:
            raise ValueError('File hash mismatch: '+str(path))

def verify(out=None, results_only=False):
    out = Path(out) if out else ROOT / 'results/v2'
    summary = json.loads((out / 'summary.json').read_text(encoding='utf-8'))
    suites = read_rows(out, 'test_suite_predictions.csv')
    pairs = read_rows(out, 'test_pair_predictions.csv')
    if len(suites) != 1040 or len(pairs) != 42687 or not summary['fresh_inference']:
        raise ValueError('Incorrect reproduction scope')
    check_metrics(pairs, 'p_killed', .5, 1, summary['pair_level_suite_checkpoint'])
    for t, key in [(.25, 'suite_level_threshold_0_25'), (.9, 'suite_level_threshold_0_90')]:
        check_metrics(suites, 'max_p_killed', t, 0, summary[key])
    groups = defaultdict(list)
    for row in pairs:
        groups[int(row['suite_index'])].append(row)
        if int(row['pred_killed_0_5']) != int(float(row['p_killed']) > .5):
            raise ValueError('Saved pair classification mismatch')
    if set(groups) != set(range(1040)):
        raise ValueError('Missing or duplicate suite groups')
    alignment = json.loads((out / 'author_alignment.json').read_text())
    mapping = alignment['author_index_for_local_suite']
    if len(mapping) != 1040 or set(mapping) != set(range(1040)):
        raise ValueError('Invalid author-index bijection')
    mismatches = 0
    if [int(row['sample_index']) for row in suites] != list(range(1040)):
        raise ValueError('Suite index order mismatch')
    for row in suites:
        i = int(row['sample_index'])
        group = groups[i]
        if [int(p['test_index']) for p in group] != list(range(len(group))):
            raise ValueError('Test index order mismatch')
        if len(group) != int(row['n_covering_tests']):
            raise ValueError('Covering-test count mismatch')
        if max(float(p['p_killed']) for p in group) != float(row['max_p_killed']):
            raise ValueError('Suite aggregation mismatch')
        retained = max(int(p['label_killed']) for p in group)
        if retained != int(row['retained_tests_or_label']):
            raise ValueError('Retained label mismatch')
        mismatches += retained != int(row['label_killed'])
        if int(row['author_suite_index']) != mapping[i]:
            raise ValueError('Suite author index mismatch')
        for pair in group:
            for key in ['source_shard', 'source_shard_suite_index', 'author_suite_index']:
                if pair[key] != row[key]:
                    raise ValueError('Pair source alignment mismatch')
    if mismatches != summary['suite_label_vs_pair_or_mismatches']:
        raise ValueError('Suite label audit mismatch')
    references = read_rows(out, 'author_reference_pair_predictions.csv')
    by_reference = {(v['suite_index'], v['test_index']): v for v in references}
    if len(references) != 42687 or len(by_reference) != 42687:
        raise ValueError('Reference count or key uniqueness mismatch')
    differences = []
    decisions = {str(t): 0 for t in [.25, .5, .9]}
    for actual in pairs:
        ref = by_reference[(actual['author_suite_index'], actual['test_index'])]
        if actual['label_killed'] != ref['label_killed']:
            raise ValueError('Author pair labels differ')
        x, y = float(actual['p_killed']), float(ref['p_killed'])
        differences.append(abs(x-y))
        for t in [.25, .5, .9]:
            decisions[str(t)] += int((x > t) != (y > t))
    comparison = summary['author_stored_prediction_comparison']
    for key, value in [('max_pair_probability_abs_difference', max(differences)),
                       ('mean_pair_probability_abs_difference', sum(differences)/len(differences))]:
        if not math.isclose(value, comparison[key], rel_tol=0, abs_tol=1e-12):
            raise ValueError('Reference comparison mismatch: '+key)
    if decisions != comparison['pair_threshold_decision_disagreements']:
        raise ValueError('Author pair decisions mismatch')
    author_suites = read_rows(out, 'author_reference_suite_predictions.csv')
    if len(author_suites) != 1040:
        raise ValueError('Author suite count mismatch')
    suite_diff = []
    suite_disagreements = 0
    for row in suites:
        ref = author_suites[int(row['author_suite_index'])]
        if row['label_killed'] != ref['label_killed'] or row['n_covering_tests'] != ref['n_covering_tests']:
            raise ValueError('Author suite labels or counts mismatch')
        x, y = float(row['max_p_killed']), float(ref['max_p_killed'])
        suite_diff.append(abs(x-y))
        suite_disagreements += (x > .25) != (y > .25)
    if suite_disagreements != comparison['decision_disagreements_threshold_0_25']:
        raise ValueError('Author suite decisions mismatch')
    if not math.isclose(max(suite_diff), comparison['max_suite_probability_abs_difference'], rel_tol=0, abs_tol=1e-12):
        raise ValueError('Author suite maximum difference mismatch')
    parity = json.loads((out / 'forward_parity.json').read_text())
    if parity['n'] != 64 or parity['max_abs_difference'] > 2e-5 or any(parity['threshold_decision_disagreements'].values()):
        raise ValueError('Historical forward parity mismatch')
    checked = 0
    if not results_only:
        for name, sha in {**summary['input_file_sha256'], **summary['raw_source_file_sha256']}.items():
            check_hash(ROOT / name, sha)
            checked += 1
        manifest = json.loads((ROOT / 'models/v2_suite_checkpoint_manifest.json').read_text())
        check_hash(ROOT / manifest['inference_checkpoint'], manifest['sha256'])
        if manifest['sha256'] != summary['checkpoint']['sha256']:
            raise ValueError('Run/checkpoint manifest mismatch')
    result = dict(passed=True, mode='results-only' if results_only else 'full',
                  fresh_local_inference=summary['fresh_inference'], suites=len(suites), pairs=len(pairs),
                  input_and_raw_hashes_checked=checked, checkpoint_hash_checked=not results_only,
                  independent_count_metrics_checked=True, author_pair_probabilities_compared=len(differences),
                  max_author_pair_probability_abs_difference=max(differences),
                  author_pair_decision_disagreements=decisions)
    if not results_only:
        (out / 'verification.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    return result

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-only', action='store_true', help='Verify committed CSV/JSON without external assets or GPU')
    args = parser.parse_args()
    print(json.dumps(verify(results_only=args.results_only), indent=2))
