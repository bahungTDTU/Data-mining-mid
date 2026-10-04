"""Audit a saved real run independently; verify splits, train-only counts and metrics."""
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import sys
import numpy as np
from ml.models import Baselines
from scripts.survey_data import ROOT, digest, write_json


def verify(run_id=None):
    if run_id is None:
        run_id = json.loads((ROOT / 'experiments/m1/latest.json').read_text())['run_id']
    output = ROOT / 'experiments/m1' / run_id
    read = lambda path: json.loads(path.read_text(encoding='utf-8'))
    manifest = read(output / 'manifest.json')
    artifact = ROOT / manifest['artifact_path']
    for name, expected in manifest['artifact_hashes'].items():
        assert digest(artifact / name) == expected, name
    for name, expected in manifest.get('report_hashes', {}).items():
        assert digest(output / name) == expected, name
    source = ROOT / manifest['data_config']['processed']
    for name, expected in manifest['input_hashes'].items():
        assert digest(source / name) == expected, name
    split = read(artifact / 'splits.json')
    sets = {part: set(users) for part, users in split.items()}
    assert all(len(split[part]) == len(ids) for part, ids in sets.items())
    assert sets['train'].isdisjoint(sets['validation'])
    assert sets['train'].isdisjoint(sets['test'])
    assert sets['test'].isdisjoint(sets['validation'])
    assert set.union(*sets.values()) == set(read(source / 'users.json'))
    history, train_counts = defaultdict(set), Counter()
    for line in (source / 'interactions.jsonl').read_text().splitlines():
        row = json.loads(line)
        history[row['user_id']].add(row['source_id'])
    for user in sets['train']:
        train_counts.update(history[user])
    model = Baselines.load(artifact / 'model')
    assert set(model.catalog) == set(train_counts)
    np.testing.assert_array_equal(model.popularity, [train_counts[item] for item in model.catalog])
    # The persisted interaction matrix must contain train users ONLY, in defined order.
    assert model.interactions.shape[0] == len(split['train'])
    for row, user in enumerate(sorted(split['train'])):
        assert {model.catalog[i] for i in model.interactions[row].indices} == history[user]
    cases = read(artifact / 'cases.json')
    lookup = {}
    for part, cohorts in cases.items():
        for cohort, levels in cohorts.items():
            for k, records in levels.items():
                for case in records:
                    user = case['user_id']
                    assert user in sets[part]
                    assert len(case['support']) == int(k)
                    assert set(case['support']).isdisjoint(case['query'])
                    assert set(case['support'] + case['query']) <= history[user]
                    assert set(case['support'] + case['query']) <= set(model.catalog)
                    assert case['query']
                    key = (part, cohort, int(k), user)
                    assert key not in lookup
                    lookup[key] = case
        common = cohorts['common']
        assert [c['user_id'] for c in common['1']] == [c['user_id'] for c in common['3']] == [c['user_id'] for c in common['5']]
        for a, b, c in zip(common['1'], common['3'], common['5']):
            assert a['support'] == c['support'][:1] and b['support'] == c['support'][:3]
            assert a['query'] == b['query'] == c['query']
    aggregates = defaultdict(list)
    for line in (artifact / 'predictions.jsonl').read_text().splitlines():
        row = json.loads(line)
        group = row['split'], row['cohort'], len(row['support']), row['model']
        case = lookup[(*group[:3], row['user_id'])]
        assert row['query'] == case['query'] and row['support'] == case['support']
        ids = row['ranking']
        assert len(ids) == len(set(ids)) <= 10
        assert set(ids) <= set(model.catalog)
        assert set(ids).isdisjoint(case['support'])
        relevance = [int(item in case['query']) for item in ids]
        recall = sum(relevance) / len(case['query'])
        ideal = sum(1/math.log2(i+2) for i in range(min(10, len(case['query']))))
        ndcg = sum(hit/math.log2(i+2) for i, hit in enumerate(relevance))/ideal
        assert math.isclose(row['recall_at_10'], recall, abs_tol=1e-12)
        assert math.isclose(row['ndcg_at_10'], ndcg, abs_tol=1e-12)
        # Spot-check persisted model inference for each group; all predictions' metrics checked.
        if not aggregates[group]:
            ranking, reason = model.recommend(group[-1], case['support'])
            assert [item for item, _ in ranking] == ids and reason == row['fallback']
        aggregates[group].append(row)
    summaries = read(output / 'metrics.json')
    for summary in summaries:
        group = summary['split'], summary['cohort'], summary['support'], summary['model']
        predictions = aggregates[group]
        assert len(predictions) == summary['users']
        assert len({row['user_id'] for row in predictions}) == len(predictions)
        expected_users = {case['user_id'] for case in cases[group[0]][group[1]][str(group[2])]}
        assert {row['user_id'] for row in predictions} == expected_users
        if not predictions:
            assert summary['recall_at_10'] is None and summary['ndcg_at_10'] is None
            continue
        for metric in ('recall_at_10', 'ndcg_at_10'):
            assert math.isclose(sum(row[metric] for row in predictions)/len(predictions), summary[metric], abs_tol=1e-12)
        coverage = len({item for row in predictions for item in row['ranking']}) / len(model.catalog)
        assert math.isclose(coverage, summary['coverage_at_10'], abs_tol=1e-12)
    trials = read(output / 'validation_trials.json')
    best = sorted(trials, key=lambda trial: (-trial['mean_ndcg_at_10'], trial['neighbors']))[0]
    assert model.neighbors == best['neighbors'] == read(output / 'selection.json')['selected_neighbors']
    write_json(output / 'verification.json', {
        'status': 'passed', 'run_id': run_id, 'prediction_rows_checked': sum(map(len, aggregates.values())),
        'summary_rows_checked': len(summaries), 'cases_checked': len(lookup),
        'checks': ['input/report/artifact hashes', 'user-disjoint exhaustive split', 'train-only catalog and interaction matrix',
                   'support/query disjoint and nested common cohort', 'independent per-user Recall/NDCG recomputation',
                   'macro metrics/coverage recomputation', 'same users/candidates across models', 'persisted model inference spot checks',
                   'validation-only selection consistency']})
    print(f'PASS: {run_id}; {len(lookup)} cases, {sum(map(len, aggregates.values()))} predictions, {len(summaries)} aggregate rows')
    return output


if __name__ == '__main__':
    verify(sys.argv[1] if len(sys.argv) > 1 else None)
