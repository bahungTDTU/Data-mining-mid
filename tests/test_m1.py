"""Synthetic fixtures with hand-computed expectations; never research data."""
import math
import numpy as np
import pytest
from ml.data import split_users, make_cases
from ml.metrics import ranking_metrics
from ml.models import Baselines
from ml.evaluation import evaluate


def games():
    return {'a': {'tags': ['Action'], 'genres': []},
            'b': {'tags': ['Action'], 'genres': []},
            'c': {'tags': ['Puzzle'], 'genres': []}}


def model():
    return Baselines().fit({'train1': ['a', 'b'], 'train2': ['a', 'c']}, games(), neighbors=2)


def test_metric_hand_calculation():
    measured = ranking_metrics(['x', 'b', 'c'], ['b', 'z'])
    assert measured['recall_at_10'] == .5
    assert measured['ndcg_at_10'] == pytest.approx((1 / math.log2(3)) / (1 + 1 / math.log2(3)))
    assert ranking_metrics(['b', 'z'], ['b', 'z'])['ndcg_at_10'] == 1
    assert ranking_metrics(['x'], ['z'])['recall_at_10'] == 0
    with pytest.raises(ValueError):
        ranking_metrics(['b', 'b'], ['b'])
    with pytest.raises(ValueError):
        ranking_metrics(['b'], [])


def test_split_disjoint_reproducible_and_exact_sizes():
    users = [f'u{i}' for i in range(100)]
    split = split_users(users, 42)
    assert split == split_users(reversed(users), 42)
    assert [len(split[p]) for p in ('train', 'validation', 'test')] == [70, 15, 15]
    assert set(split['train']).isdisjoint(split['test'])
    assert set(split['train']).isdisjoint(split['validation'])
    assert set(split['test']).isdisjoint(split['validation'])
    assert set(sum(split.values(), [])) == set(users)


def test_nested_support_same_query_and_common_users():
    history = {'u': list('abcdefgh'), 'v': ['a'], 'w': ['a', 'b']}
    cases, audit = make_cases(history, history, list('abcdefgh'), 42)
    rows = [cases['common'][str(k)][0] for k in (1, 3, 5)]
    assert rows[0]['support'] == rows[2]['support'][:1]
    assert rows[1]['support'] == rows[2]['support'][:3]
    assert rows[0]['query'] == rows[1]['query'] == rows[2]['query']
    assert set(rows[2]['support']).isdisjoint(rows[2]['query'])
    assert len(rows[0]['query']) == 3
    assert audit['common/1']['included_users'] == 1
    assert len(cases['short_observed']['1']) == 1
    assert cases['short_observed']['5'] == []


def test_catalog_exclusions_counted_without_support_replacement():
    history = {'u': list('abcdef')}
    full, _ = make_cases(history, history, list('abcdef'), 42)
    support = full['common']['5'][0]['support']
    query = full['common']['5'][0]['query']
    cases, audit = make_cases(history, history, support, 42)
    assert not cases['common']['1']
    assert audit['common/1']['query_items_outside'] == 1
    assert audit['common/1']['empty_query_users'] == 1
    cases, audit = make_cases(history, history, list(set('abcdef') - {support[-1]}), 42)
    assert not cases['common']['1']  # admission must also check unavailable 5th support
    assert audit['common/1']['support_outside_users'] == 1


def test_popularity_and_cf_manual_values():
    m = model()
    assert m.popularity.tolist() == [2, 1, 1]
    assert m.cf[0, 1] == pytest.approx(1 / math.sqrt(2))
    assert m.cf[1, 2] == 0
    assert np.all(m.cf.diagonal() == 0)
    scores, _ = m.scores('itemcf', ['b'])
    assert scores[0] == pytest.approx(1 / math.sqrt(2))
    assert m.build_cf(1).getnnz(axis=1).max() <= 1


def test_content_cosine_and_train_only_vocabulary():
    train = {'t': ['a', 'b']}
    g = games()
    g['query_only'] = {'tags': ['DoNotLeak'], 'genres': []}
    m = Baselines().fit(train, g)
    assert 'tags:donotleak' not in m.vocabulary
    assert 'tags:puzzle' not in m.vocabulary
    assert m.catalog == ['a', 'b']
    scores, _ = m.scores('content', ['a'])
    assert scores.tolist() == pytest.approx([1, 1])
    g['query_only']['tags'] = ['DifferentQuery']
    other = Baselines().fit(train, g)
    np.testing.assert_array_equal(m.idf, other.idf)
    np.testing.assert_array_equal(m.popularity, other.popularity)
    assert (m.features != other.features).nnz == 0


@pytest.mark.parametrize('name', ['popular', 'content', 'itemcf'])
def test_shared_candidate_filter_and_zero_history(name):
    m = model()
    ranking, _ = m.recommend(name, ['a'], hidden=['b'])
    assert [item for item, _ in ranking] == ['c']
    ranking, reason = m.recommend(name, [])
    assert [item for item, _ in ranking] == ['a', 'b', 'c']
    assert reason == (None if name == 'popular' else 'no_known_support')
    assert m.recommend(name, [], hidden=['a', 'b', 'c'])[0] == []


def test_missing_content_fallback():
    m = Baselines().fit({'t': ['a', 'b']}, {'a': {}, 'b': {}})
    ranking, reason = m.recommend('content', ['a'])
    assert ranking == [('b', 1.0)]
    assert reason == 'no_content_signal'


def test_saved_artifacts_roundtrip_and_unknown_schema(tmp_path):
    m = model()
    m.save(tmp_path)
    loaded = Baselines.load(tmp_path)
    for name in ('popular', 'content', 'itemcf'):
        assert m.recommend(name, ['a']) == loaded.recommend(name, ['a'])
    (tmp_path / 'model.json').write_text('{"schema_version":"bad"}')
    with pytest.raises(ValueError):
        Baselines.load(tmp_path)


def test_evaluation_query_does_not_change_ranking():
    m = model()
    _, first = evaluate(m, 'content', [{'user_id':'u', 'support':['a'], 'query':['b']}])
    _, second = evaluate(m, 'content', [{'user_id':'u', 'support':['a'], 'query':['c']}])
    assert first[0]['ranking'] == second[0]['ranking']
    empty, _ = evaluate(m, 'popular', [])
    assert empty['users'] == 0 and empty['ndcg_at_10'] is None
