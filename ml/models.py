"""Train-only sparse baselines, persisted as NPZ + JSON (no pickle)."""
import json
import logging
import time
from pathlib import Path
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

LOG = logging.getLogger(__name__)


def metadata_tokens(game):
    # Preserve multiword tags as atomic features, with type namespace.
    return sorted({f'{field}:{str(value).strip().casefold()}'
                   for field in ('tags', 'genres') for value in game.get(field, []) if str(value).strip()})


class Baselines:
    def fit(self, train_histories, games, neighbors=50):
        started = time.perf_counter()
        self.catalog = sorted({item for items in train_histories.values() for item in items if item in games})
        if not self.catalog:
            raise ValueError('Empty training catalog')
        self.index = {item: i for i, item in enumerate(self.catalog)}
        rows, cols = [], []
        for row, user in enumerate(sorted(train_histories)):
            for item in sorted(set(train_histories[user])):
                if item in self.index:
                    rows.append(row)
                    cols.append(self.index[item])
        self.interactions = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)),
                                              shape=(len(train_histories), len(self.catalog)))
        self.popularity = np.asarray(self.interactions.sum(axis=0)).ravel()
        self.fit_seconds = {'popular_including_matrix': time.perf_counter() - started}
        started = time.perf_counter()
        docs = ['\n'.join(metadata_tokens(games[item])) for item in self.catalog]
        vectorizer = TfidfVectorizer(tokenizer=str.splitlines, token_pattern=None, lowercase=False, norm='l2')
        if any(docs):
            self.features = vectorizer.fit_transform(docs).tocsr()
            self.vocabulary = vectorizer.vocabulary_
            self.idf = vectorizer.idf_
        else:
            self.features = sparse.csr_matrix((len(self.catalog), 0))
            self.vocabulary, self.idf = {}, np.array([])
        self.fit_seconds['content'] = time.perf_counter() - started
        started = time.perf_counter()
        self.cf = self.build_cf(neighbors)
        self.fit_seconds['itemcf'] = time.perf_counter() - started
        self.neighbors = neighbors
        return self

    def build_cf(self, neighbors):
        if neighbors < 1:
            raise ValueError('neighbors must be positive')
        item_users = normalize(self.interactions.T.tocsr(), norm='l2')
        data, rows, cols = [], [], []
        # Blocks bound intermediate memory; never a dense user x item matrix.
        for start in range(0, len(self.catalog), 128):
            block = (item_users[start:start+128] @ item_users.T).tocsr()
            for local in range(block.shape[0]):
                row = start + local
                lo, hi = block.indptr[local:local+2]
                ids, values = block.indices[lo:hi], block.data[lo:hi]
                mask = (ids != row) & (values > 0)
                ids, values = ids[mask], values[mask]
                order = np.lexsort((ids, -values))[:neighbors]
                rows.extend([row] * len(order))
                cols.extend(ids[order].tolist())
                data.extend(values[order].tolist())
        return sparse.csr_matrix((data, (rows, cols)), shape=(len(self.catalog), len(self.catalog)))

    def scores(self, model, support):
        if model not in ('popular', 'content', 'itemcf'):
            raise ValueError('Unknown model')
        indices = sorted({self.index[item] for item in support if item in self.index})
        reason = None
        if model == 'popular':
            scores = self.popularity.copy()
        elif not indices:
            scores, reason = self.popularity.copy(), 'no_known_support'
        elif model == 'content':
            if self.features.shape[1] == 0:
                return self.popularity.copy(), 'no_content_signal'
            profile = sparse.csr_matrix(self.features[indices].sum(axis=0))
            profile = normalize(profile)
            scores = np.asarray((self.features @ profile.T).toarray()).ravel()
            if not np.any(scores > 0):
                scores, reason = self.popularity.copy(), 'no_content_signal'
        else:
            scores = np.asarray(self.cf[indices].sum(axis=0)).ravel()
            if not np.any(scores > 0):
                scores, reason = self.popularity.copy(), 'no_cf_signal'
        if not np.all(np.isfinite(scores)):
            raise ValueError('Non-finite model scores')
        return scores, reason

    def recommend(self, model, support, k=10, hidden=()):
        if not 1 <= k <= 100:
            raise ValueError('k must be between 1 and 100')
        support, hidden = set(support), set(hidden)
        scores, reason = self.scores(model, support)
        blocked = support | hidden
        allowed = np.array([i for i, item in enumerate(self.catalog) if item not in blocked], dtype=int)
        if reason:
            LOG.debug('Recommendation fallback model=%s reason=%s', model, reason)
        order = allowed[np.lexsort((allowed, -scores[allowed]))[:k]]
        return [(self.catalog[i], float(scores[i])) for i in order], reason

    def save(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        for name in ('features', 'cf', 'interactions'):
            sparse.save_npz(path / f'{name}.npz', getattr(self, name))
        np.savez(path / 'arrays.npz', popularity=self.popularity, idf=self.idf)
        (path / 'model.json').write_text(json.dumps({'schema_version': 'm1.1', 'catalog': self.catalog,
            'vocabulary': self.vocabulary, 'neighbors': self.neighbors}, sort_keys=True), encoding='utf-8')

    @classmethod
    def load(cls, path):
        path = Path(path)
        meta = json.loads((path / 'model.json').read_text(encoding='utf-8'))
        if meta['schema_version'] != 'm1.1':
            raise ValueError('Unsupported artifact schema')
        model = cls()
        model.catalog, model.vocabulary, model.neighbors = meta['catalog'], meta['vocabulary'], meta['neighbors']
        model.index = {item: i for i, item in enumerate(model.catalog)}
        for name in ('features', 'cf', 'interactions'):
            setattr(model, name, sparse.load_npz(path / f'{name}.npz'))
        with np.load(path / 'arrays.npz', allow_pickle=False) as arrays:
            model.popularity, model.idf = arrays['popularity'], arrays['idf']
        n = len(model.catalog)
        if model.cf.shape != (n, n) or model.features.shape[0] != n or model.popularity.shape != (n,):
            raise ValueError('Artifact dimensions do not match catalog')
        for values in (model.features.data, model.cf.data, model.popularity):
            if not np.isfinite(values).all():
                raise ValueError('Artifact contains non-finite values')
        return model
