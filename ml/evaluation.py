from collections import Counter
import numpy as np
from ml.metrics import ranking_metrics


def evaluate(model, name, cases, k=10):
    per_user, recommended, diversity = [], set(), []
    popularities, fallbacks = [], Counter()
    for case in cases:
        ranking, reason = model.recommend(name, case['support'], k)
        ids = [item for item, _ in ranking]
        if set(ids) & set(case['support']) or not set(case['query']).isdisjoint(case['support']):
            raise AssertionError('Support leaked into ranking or query')
        metrics = ranking_metrics(ids, case['query'], k)
        per_user.append({'user_id': case['user_id'], **metrics, 'ranking': ids,
                         'query': case['query'], 'support': case['support'], 'fallback': reason})
        recommended.update(ids)
        indices = [model.index[item] for item in ids]
        popularities.append(float(model.popularity[indices].mean()) if indices else 0.0)
        features = model.features[indices]
        features = features[np.asarray(features.getnnz(axis=1)) > 0]
        if features.shape[0] >= 2:
            similarities = (features @ features.T).toarray()  # at most K x K
            diversity.append(float(np.mean(1 - similarities[np.triu_indices(len(similarities), 1)])))
        if reason:
            fallbacks[reason] += 1
    summary = {
        'users': len(cases), 'catalog_size': len(model.catalog),
        'candidates_per_user': len(model.catalog) - len(cases[0]['support']) if cases else None,
        'recall_at_10': float(np.mean([row['recall_at_10'] for row in per_user])) if per_user else None,
        'ndcg_at_10': float(np.mean([row['ndcg_at_10'] for row in per_user])) if per_user else None,
        'coverage_at_10': len(recommended) / len(model.catalog) if cases else None,
        'ild': float(np.mean(diversity)) if diversity else None,
        'ild_excluded_users': len(cases) - len(diversity),
        'average_recommendation_popularity': float(np.mean(popularities)) if cases else None,
        'fallback_counts': dict(fallbacks),
    }
    return summary, per_user
