"""Deterministic user-disjoint split; no fitting or scoring here."""
from collections import Counter
import hashlib
import random


def split_users(users, seed):
    ordered = sorted(users)
    if len(set(ordered)) != len(ordered):
        raise ValueError('Duplicate user IDs')
    random.Random(seed).shuffle(ordered)
    a, b = int(len(ordered) * .7), int(len(ordered) * .15)
    return {'train': ordered[:a], 'validation': ordered[a:a+b], 'test': ordered[a+b:]}


def make_cases(users, histories, catalog, seed):
    catalog = set(catalog)
    cases = {'common': {str(k): [] for k in (1, 3, 5)},
             'short_observed': {str(k): [] for k in (1, 3, 5)}}
    audits = {}
    for cohort, levels in cases.items():
        for key, rows in levels.items():
            k = int(key)
            count = Counter(total_split_users=len(users), eligible_history=0, support_outside_users=0,
                            support_items_total=0, support_items_outside=0, query_items_total=0,
                            query_items_outside=0, empty_query_users=0, included_users=0)
            for user in sorted(users):
                items = sorted(histories[user])
                n = len(items)
                if not (n >= 6 if cohort == 'common' else k + 1 <= n <= 5):
                    continue
                count['eligible_history'] += 1
                local_seed = int.from_bytes(hashlib.sha256(f'{seed}:{user}'.encode()).digest(), 'big')
                random.Random(local_seed).shuffle(items)
                pool_size = 5 if cohort == 'common' else k
                pool, query_all = items[:pool_size], items[pool_size:]
                # Common-cohort admission always checks the SAME five items.
                outside_support = set(pool) - catalog
                query = sorted(set(query_all) & catalog)
                count['support_items_total'] += len(pool)
                count['support_items_outside'] += len(outside_support)
                count['query_items_total'] += len(query_all)
                count['query_items_outside'] += len(set(query_all) - catalog)
                if outside_support:
                    count['support_outside_users'] += 1
                elif not query:
                    count['empty_query_users'] += 1
                else:
                    rows.append({'user_id': user, 'support': pool[:k], 'query': query,
                                 'history_size': n, 'query_original_size': len(query_all)})
                    count['included_users'] += 1
            audits[f'{cohort}/{k}'] = dict(count)
    return cases, audits
