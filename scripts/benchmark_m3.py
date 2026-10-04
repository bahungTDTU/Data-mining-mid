"""Repeatable local inference benchmark; frozen validation supports, no tuning."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
RUN = '20260915T050327853533Z-seed42'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def worker(run, name, repeats):
    import numpy as np
    import psutil
    from ml.models import Baselines
    artifact = ROOT / 'artifacts/m1' / run
    levels = json.loads((artifact / 'cases.json').read_text())['validation']['common']
    process = psutil.Process()
    rss_before = process.memory_info().rss
    started = time.perf_counter()
    model = Baselines.load(artifact / 'model')
    load_seconds = time.perf_counter() - started
    rss_loaded = process.memory_info().rss
    arrays = {'popular': [model.popularity],
              'content': [model.features.data, model.features.indices, model.features.indptr, model.idf],
              'itemcf': [model.cf.data, model.cf.indices, model.cf.indptr]}
    rows = []
    for k in ('1', '3', '5'):
        supports = [case['support'] for case in levels[k]]
        # Entire cohort warmup once, then repeat same order; query never passed.
        expected = [model.recommend(name, support, k=10) for support in supports]
        samples = []
        for repeat in range(repeats):
            for support, reference in zip(supports, expected):
                start = time.perf_counter_ns()
                result = model.recommend(name, support, k=10)
                samples.append((time.perf_counter_ns() - start) / 1e6)
                assert result == reference
                assert not set(support).intersection(item for item, score in result[0])
        rows.append({'support': int(k), 'users': len(supports), 'calls': len(samples),
                     'warmup_calls': len(supports), 'latency_ms': samples,
                     'p50_ms': float(np.percentile(samples, 50)),
                     'p95_ms': float(np.percentile(samples, 95)),
                     'mean_ms': float(np.mean(samples)),
                     'ranking_sha256': hashlib.sha256(json.dumps(expected, sort_keys=True).encode()).hexdigest()})
    mem = process.memory_info()
    return {'model': name, 'load_bundle_seconds': load_seconds,
            'rss_before_load_bytes': rss_before, 'rss_loaded_bundle_bytes': rss_loaded,
            'peak_process_working_set_bytes': getattr(mem, 'peak_wset', None),
            'model_numeric_arrays_bytes': sum(a.nbytes for a in arrays[name]),
            'memory_scope': 'Process includes Python, libraries, workload and ALL baseline artifacts. Numeric array bytes exclude Python objects, catalog and shared popularity fallback. Not standalone per-model peak RAM.',
            'rows': rows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', default=RUN)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--worker', choices=['popular', 'content', 'itemcf'])
    args = parser.parse_args()
    if args.repeats < 2 or Path(args.run).name != args.run:
        parser.error('Use repeats >= 2 and a run directory name')
    if args.worker:
        print(json.dumps(worker(args.run, args.worker, args.repeats)))
        return
    import psutil
    from scripts.verify_m1 import verify
    verify(args.run)
    folder = ROOT / 'experiments/m3' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    folder.mkdir(parents=True)
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    results = []
    for name in ('popular', 'content', 'itemcf'):
        completed = subprocess.run([sys.executable, '-m', 'scripts.benchmark_m3', '--worker', name,
                                   '--run', args.run, '--repeats', str(args.repeats)],
                                  cwd=ROOT, env=env, capture_output=True, text=True, check=True)
        results.append(json.loads(completed.stdout))
    cpu = platform.processor()
    if platform.system() == 'Windows':
        cpu = subprocess.check_output(['powershell', '-NoProfile', '-Command',
                                      '(Get-CimInstance Win32_Processor).Name'], text=True).strip()
    report = {'created_at_utc': datetime.now(timezone.utc).isoformat(), 'run_id': args.run,
              'environment': {'os': platform.platform(), 'python': platform.python_version(), 'cpu': cpu,
                              'physical_cores': psutil.cpu_count(logical=False), 'logical_cores': psutil.cpu_count(),
                              'ram_bytes': psutil.virtual_memory().total, 'threads': 1},
              'protocol': {'split': 'validation', 'cohort': 'common', 'top_k': 10, 'repeats': args.repeats,
                           'candidate': 'full frozen train catalog minus supplied support',
                           'scope': 'Sequential in-process recommend including scoring/filter/sort; no HTTP/SQL/serialization; no quality tuning'},
              'hashes': {p: sha(ROOT/p) for p in ['ml/models.py', 'scripts/benchmark_m3.py', 'requirements.lock',
                       f'artifacts/m1/{args.run}/cases.json', f'experiments/m1/{args.run}/manifest.json']},
              'results': results}
    (folder/'benchmark.json').write_text(json.dumps(report, indent=2)+'\n')
    (ROOT/'experiments/m3/latest.json').write_text(json.dumps({'path': str(folder.relative_to(ROOT)).replace('\\','/')})+'\n')
    print(folder)


if __name__ == '__main__':
    main()
