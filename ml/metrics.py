"""Binary relevance, macro averaging done by caller."""
import math


def ranking_metrics(ranking, query, k=10):
    query = set(query)
    if not query:
        raise ValueError('Empty query must be counted and excluded before evaluation')
    if len(ranking) != len(set(ranking)):
        raise ValueError('Duplicate recommendations')
    hits = [int(item in query) for item in ranking[:k]]
    recall = sum(hits) / len(query)
    dcg = sum(hit / math.log2(rank + 2) for rank, hit in enumerate(hits))
    ideal = sum(1 / math.log2(rank + 2) for rank in range(min(k, len(query))))
    return {'recall_at_10': recall, 'ndcg_at_10': dcg / ideal}
