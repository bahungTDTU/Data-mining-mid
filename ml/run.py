"""One command: verify raw -> prepare data -> split -> fit/tune -> evaluate -> save."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import time
import numpy as np
import pandas as pd
import psutil
from ml.data import split_users, make_cases
from ml.models import Baselines
from ml.evaluation import evaluate
from ml.plots import plot_eda
from scripts.survey_data import ROOT, digest, survey, write_json


def run(config_path):
    started = time.perf_counter()
    config = json.loads(config_path.read_text(encoding='utf-8'))
    if config['k'] != 10 or config['protocol_version'] != 'v1':
        raise ValueError('This implementation supports protocol v1 and K=10 only')
    data_config = json.loads((ROOT / config['source_config']).read_text(encoding='utf-8'))
    survey(data_config)  # verifies pinned raw checksums; reconstructs complete sample
    source = ROOT / data_config['processed']
    users = json.loads((source / 'users.json').read_text(encoding='utf-8'))
    games = {game['source_id']: game for game in json.loads((source / 'games.json').read_text(encoding='utf-8'))}
    histories = {user: [] for user in users}
    seen = set()
    for line in (source / 'interactions.jsonl').read_text(encoding='utf-8').splitlines():
        row = json.loads(line)
        pair = row['user_id'], row['source_id']
        if pair in seen or row['positive'] is not True or pair[0] not in histories or pair[1] not in games:
            raise ValueError('Invalid processed interaction')
        seen.add(pair)
        histories[pair[0]].append(pair[1])
    splits = split_users(users, config['seed'])
    train = {user: histories[user] for user in splits['train']}
    fit_started = time.perf_counter()
    model = Baselines().fit(train, games, neighbors=config['neighbors_grid'][0])
    initial_fit_seconds = time.perf_counter() - fit_started
    # Make validation/test manifests identically, before model-dependent evaluation.
    cases, audits = {}, {}
    for part in ('validation', 'test'):
        cases[part], audits[part] = make_cases(splits[part], histories, model.catalog, config['seed'])
    if not cases['validation']['common']['5'] or not cases['test']['common']['5']:
        raise ValueError('Empty common cohort; do not report NaN as evaluation')
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '-seed' + str(config['seed'])
    output = ROOT / 'experiments/m1' / run_id
    artifacts = ROOT / 'artifacts/m1' / run_id
    output.mkdir(parents=True)
    artifacts.mkdir(parents=True)
    write_json(artifacts / 'splits.json', splits)
    write_json(artifacts / 'cases.json', cases)
    write_json(output / 'cohorts.json', audits)
    write_json(output / 'config.json', config)
    write_json(output / 'data_manifest.json', json.loads((ROOT / data_config['output'] / 'manifest.json').read_text(encoding='utf-8')))
    trials = []
    for neighbors in config['neighbors_grid']:
        t = time.perf_counter()
        model.cf = model.build_cf(neighbors)
        build_seconds = time.perf_counter() - t
        results = [evaluate(model, 'itemcf', cases['validation']['common'][str(k)])[0] for k in (1, 3, 5)]
        trials.append({'neighbors': neighbors, 'mean_ndcg_at_10': float(np.mean([r['ndcg_at_10'] for r in results])),
                       'build_seconds': build_seconds, 'per_support': results})
    selected = sorted(trials, key=lambda trial: (-trial['mean_ndcg_at_10'], trial['neighbors']))[0]
    model.neighbors = selected['neighbors']
    t = time.perf_counter()
    model.cf = model.build_cf(model.neighbors)
    model.fit_seconds['itemcf_selected'] = time.perf_counter() - t
    write_json(output / 'validation_trials.json', trials)
    # This file is written BEFORE any test-development scores are computed.
    write_json(output / 'selection.json', {'selected_neighbors': model.neighbors, 'criterion': 'validation common mean NDCG@10 over 1/3/5; tie smaller N',
        'locked_at_utc': datetime.now(timezone.utc).isoformat(), 'config_sha256': digest(output / 'config.json'),
        'purpose': config['purpose']})
    model.save(artifacts / 'model')
    rows, detailed = [], []
    for part in ('validation', 'test'):
        for cohort, levels in cases[part].items():
            for support, user_cases in levels.items():
                for name in ('popular', 'content', 'itemcf'):
                    summary, per_user = evaluate(model, name, user_cases)
                    metadata = {'split': part, 'cohort': cohort, 'support': int(support), 'model': name}
                    rows.append({**metadata, **summary})
                    detailed.extend({**metadata, **row, 'support_size': int(support)} for row in per_user)
    write_json(output / 'metrics.json', rows)
    pd.DataFrame(rows).to_csv(output / 'metrics.csv', index=False, encoding='utf-8')
    with (artifacts / 'predictions.jsonl').open('w', encoding='utf-8') as handle:
        for row in detailed:
            handle.write(json.dumps(row, sort_keys=True) + '\n')
    plot_eda(histories, model, games, audits, cases['test']['common']['5'], output / 'eda.png')
    counts = Counter(map(len, histories.values()))
    write_json(output / 'data_summary.json', {
        'users': len(users), 'sample_items': len(games), 'interactions': len(seen),
        'split_users': {key: len(value) for key, value in splits.items()},
        'catalog_size': len(model.catalog), 'train_interactions': int(model.interactions.nnz),
        'train_sparsity': 1-model.interactions.nnz / np.prod(model.interactions.shape),
        'history_counts': dict(counts), 'train_items_without_features': int(sum(model.features.getnnz(axis=1) == 0))})
    memory = psutil.Process().memory_info()
    manifest = {
        'run_id': run_id, 'schema_version': 'm1.1', 'protocol_version': config['protocol_version'],
        'purpose': config['purpose'], 'python': platform.python_version(), 'platform': platform.platform(),
        'dependencies': {name: importlib.metadata.version(name) for name in ('numpy','scipy','scikit-learn','pandas','matplotlib','psutil')},
        'config': config, 'data_config': data_config,
        'input_hashes': {name: digest(source / name) for name in ('users.json', 'games.json', 'interactions.jsonl')},
        'raw_hashes': {name: digest(ROOT / data_config[name]) for name in ('reviews','metadata')},
        'code_hashes': {str(p.relative_to(ROOT)): digest(p) for p in sorted((ROOT / 'ml').glob('*.py'))},
        'survey_code_sha256': digest(ROOT / 'scripts/survey_data.py'),
        'protocol_sha256': digest(ROOT / 'docs/research/protocol.md'), 'lock_sha256': digest(ROOT / 'requirements.lock'),
        'artifact_path': str(artifacts.relative_to(ROOT)),
        'artifact_hashes': {str(p.relative_to(artifacts)): digest(p) for p in sorted(artifacts.rglob('*')) if p.is_file()},
        'initial_fit_seconds': initial_fit_seconds, 'elapsed_seconds': time.perf_counter() - started,
        'fit_seconds': model.fit_seconds,
        'process_peak_working_set_bytes': getattr(memory, 'peak_wset', None),
        'memory_scope': 'whole process OS high-water mark, includes libraries/data/fit/evaluation/plots; not per-model peak',
        'api_latency': 'not measured',
    }
    manifest['report_hashes'] = {p.name: digest(p) for p in sorted(output.iterdir()) if p.is_file()}
    write_json(output / 'manifest.json', manifest)
    write_json(ROOT / 'experiments/m1/latest.json', {'run_id': run_id})
    print(f'M1 completed: {output}')
    for row in rows:
        if row['split'] == 'test' and row['cohort'] == 'common':
            print({key: row[key] for key in ('model','support','users','recall_at_10','ndcg_at_10')})
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/m1.json')
    args = parser.parse_args()
    run(ROOT / args.config)
