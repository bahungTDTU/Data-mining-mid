"""M0: safely audit complete source files and materialize a deterministic user sample.

No model fitting or train/test selection happens here. Only Python standard library.
"""
import argparse
import ast
from collections import Counter, defaultdict
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import platform
import random
import time

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "reviews": "https://mcauleylab.ucsd.edu/public_datasets/data/steam/australian_user_reviews.json.gz",
    "metadata": "https://cseweb.ucsd.edu/~wckang/steam_games.json.gz",
}


def digest(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def read_records(path):
    with gzip.open(path, 'rt', encoding='utf-8') as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                record = ast.literal_eval(line)
                if not isinstance(record, dict):
                    raise ValueError('expected dictionary')
            except (SyntaxError, ValueError) as exc:
                raise ValueError(f'{path.name}:{line_number}: invalid literal record') from exc
            yield record


def source_id(value):
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        return None
    value = str(value).strip()
    return value if value.isascii() and value.isdigit() else None


def posted_date(value):
    if not isinstance(value, str):
        return None
    try:
        return datetime.strptime(value.removeprefix('Posted ').rstrip('.'), '%B %d, %Y').date().isoformat()
    except ValueError:
        return None  # Missing year must not be invented from collection time.


def distribution(values):
    values = sorted(values)
    return {
        'count': len(values), 'min': min(values, default=0), 'max': max(values, default=0),
        'mean': sum(values) / len(values) if values else 0,
        'histogram': dict(sorted(Counter(values).items())),
    }


def eligibility(histories):
    return {str(k): sum(len(items) >= k + 1 for items in histories.values()) for k in (1, 3, 5)}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def survey(config):
    started = time.perf_counter()
    for key, expected in config.get('expected_sha256', {}).items():
        if digest(ROOT / config[key]) != expected:
            raise ValueError(f'Checksum mismatch for {key}; raw input changed')
    counters = Counter()
    games = {}
    fields = {'metadata': Counter(), 'user': Counter(), 'review': Counter()}
    for record in read_records(ROOT / config['metadata']):
        counters['metadata_rows'] += 1
        fields['metadata'].update(record.keys())
        item = source_id(record.get('id'))
        if item is None:
            counters['metadata_missing_id'] += 1
            continue
        if item in games:
            counters['metadata_duplicate_id'] += 1
            continue  # Stable first source row wins; counted explicitly.
        games[item] = {
            'source_id': item, 'title': record.get('title') or record.get('app_name'),
            'tags': record.get('tags') or [], 'genres': record.get('genres') or [],
        }
    users = set()
    pairs = defaultdict(list)
    all_items = set()
    for record in read_records(ROOT / config['reviews']):
        counters['user_rows'] += 1
        fields['user'].update(record.keys())
        user = record.get('user_id')
        if not isinstance(user, str) or not user.strip():
            counters['user_rows_missing_id'] += 1
            continue
        if user in users:
            counters['duplicate_user_rows'] += 1
        users.add(user)
        reviews = record.get('reviews')
        if not isinstance(reviews, list):
            raise ValueError('reviews must be a list')
        if not reviews:
            counters['empty_user_rows'] += 1
        for review in reviews:
            counters['review_rows'] += 1
            fields['review'].update(review.keys())
            item = source_id(review.get('item_id'))
            if item is None:
                counters['reviews_missing_item_id'] += 1
                continue
            all_items.add(item)
            if item not in games:
                counters['review_rows_without_metadata'] += 1
            recommend = review.get('recommend')
            if type(recommend) is not bool:
                counters['reviews_invalid_recommend'] += 1
                continue
            counters['positive_review_rows' if recommend else 'negative_review_rows'] += 1
            date = posted_date(review.get('posted'))
            counters['review_dates_complete' if date else 'review_dates_incomplete'] += 1
            pairs[user, item].append((recommend, date))
    histories = {user: {} for user in users}
    for (user, item), observations in pairs.items():
        counters['duplicate_pair_rows'] += len(observations) - 1
        labels = {label for label, _ in observations}
        if len(labels) > 1:
            counters['conflicting_pairs_removed'] += 1
        elif True in labels:
            counters['positive_unique_pairs_before_join'] += 1
            if item in games:
                dates = sorted(date for _, date in observations if date)
                histories[user][item] = dates[0] if dates else None
            else:
                counters['positive_unique_pairs_without_metadata'] += 1
    item_counts = Counter(item for items in histories.values() for item in items)
    positive_count = sum(item_counts.values())
    selected = sorted(random.Random(config['seed']).sample(sorted(users), min(config['sample_users'], len(users))))
    selected_histories = {user: histories[user] for user in selected}
    processed = ROOT / config['processed']
    processed.mkdir(parents=True, exist_ok=True)
    selected_items = sorted({item for items in selected_histories.values() for item in items})
    # Pseudonyms avoid publishing Steam handles; do not claim irreversible anonymization.
    pseudonym = lambda user: hashlib.sha256(('ucsd-steam-v1:' + user).encode()).hexdigest()
    with (processed / 'interactions.jsonl').open('w', encoding='utf-8') as handle:
        for user, items in selected_histories.items():
            for item, date in sorted(items.items()):
                handle.write(json.dumps({'user_id': pseudonym(user), 'source_id': item, 'positive': True, 'posted_date': date}) + '\n')
    write_json(processed / 'games.json', [games[item] for item in selected_items])
    write_json(processed / 'users.json', [pseudonym(user) for user in selected])
    report = {
        'schema_version': config['schema_version'], 'scope': 'full downloaded source audit; sample is for M1 development, not ranking results',
        'counts': dict(counters), 'unique_source_users': len(users), 'unique_review_items': len(all_items),
        'unique_metadata_items': len(games), 'joined_positive_interactions': positive_count,
        'joined_positive_items': len(item_counts), 'user_history': distribution(map(len, histories.values())),
        'item_popularity': distribution(item_counts.values()),
        'matrix_sparsity': 1 - positive_count / (len(users) * len(item_counts)) if users and item_counts else None,
        'eligible_before_split_k_plus_one': eligibility(histories),
        'common_cohort_before_split': eligibility(histories)['5'],
        'metadata_missing_title': sum(not game['title'] for game in games.values()),
        'joined_items_missing_title': sum(not games[item]['title'] for item in item_counts),
        'metadata_missing_tags': sum(not game['tags'] for game in games.values()),
        'joined_items_missing_tags': sum(not games[item]['tags'] for item in item_counts),
        'joined_items_missing_genres': sum(not games[item]['genres'] for item in item_counts),
        'review_metadata_unmatched_rate': counters['review_rows_without_metadata'] / counters['review_rows'],
        'duplicate_pair_row_rate': counters['duplicate_pair_rows'] / counters['review_rows'],
        'field_presence': {key: dict(value) for key, value in fields.items()},
        'tag_counts_joined_items': dict(Counter(tag for item in item_counts for tag in games[item]['tags']).most_common()),
        'sample': {'users': len(selected), 'items': len(selected_items), 'interactions': sum(map(len, selected_histories.values())),
                   'eligible_before_split_k_plus_one': eligibility(selected_histories), 'seed': config['seed'],
                   'method': 'uniform sample of sorted unique raw users, retaining every cleaned positive joined pair; no k-core'},
    }
    output = ROOT / config['output']
    write_json(output / 'survey.json', report)
    manifest = {
        'schema_version': config['schema_version'], 'generated_at_utc': datetime.now(timezone.utc).isoformat(),
        'python': platform.python_version(), 'platform': platform.platform(), 'config': config,
        'inputs': {key: {'url': SOURCES[key], 'bytes': (ROOT / config[key]).stat().st_size, 'sha256': digest(ROOT / config[key])} for key in SOURCES},
        'outputs': {p.name: digest(p) for p in sorted(processed.iterdir()) if p.is_file()},
        'script_sha256': digest(Path(__file__)), 'config_sha256': hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
        'elapsed_seconds': round(time.perf_counter() - started, 3), 'peak_ram': 'not measured',
    }
    write_json(output / 'manifest.json', manifest)
    print(json.dumps({key: report[key] for key in ('counts', 'sample', 'eligible_before_split_k_plus_one')}, indent=2))
    print(f"Survey completed in {manifest['elapsed_seconds']} seconds")
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/m0.json')
    args = parser.parse_args()
    survey(json.loads((ROOT / args.config).read_text(encoding='utf-8')))
