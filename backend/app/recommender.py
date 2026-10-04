"""Verified offline model adapter; query/test never enter online profiles."""
import hashlib
import json
import logging
import os
from pathlib import Path
import numpy as np
from scipy import sparse
from sklearn.preprocessing import normalize
from ml.models import Baselines
from backend.app.config import ROOT

LOG = logging.getLogger(__name__)


def sha(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def checked_path(base, relative):
    path = (base / relative.replace('\\', '/')).resolve()
    if not path.is_relative_to(base.resolve()):
        raise ValueError('Artifact path escapes root')
    return path


class RecommendationService:
    def __init__(self, run_id=None):
        self.model = None
        self.run_id = run_id or os.getenv('MODEL_RUN_ID', '20260915T050327853533Z-seed42')
        if '/' in self.run_id or '\\' in self.run_id or self.run_id.startswith('.'):
            raise ValueError('Invalid model run ID')
        self.folder = ROOT / 'experiments/m1' / self.run_id
        self.manifest = json.loads((self.folder / 'manifest.json').read_text())
        artifact = checked_path(ROOT, self.manifest['artifact_path'])
        for name, expected in self.manifest['artifact_hashes'].items():
            if name.replace('\\','/').startswith('model/') and sha(checked_path(artifact, name)) != expected:
                raise ValueError('Model checksum mismatch')
        for name in ('metrics.json', 'cohorts.json'):
            if sha(self.folder/name) != self.manifest['report_hashes'][name]:
                raise ValueError('Report checksum mismatch')
        self.model = Baselines.load(artifact / 'model')
        self.feature_names = {index: name for name, index in self.model.vocabulary.items()}

    def recommend(self, support, hidden, tags, k):
        m = self.model
        profile = None
        method = 'content' if support else 'popular'
        ranking, fallback = m.recommend(method, support, k, hidden)
        ids = [m.index[item] for item in sorted(set(support)) if item in m.index]
        if method == 'content' and not fallback:
            profile = normalize(sparse.csr_matrix(m.features[ids].sum(axis=0)))
        elif not support and tags:
            # Only explicitly supplied tags; transform using train-fitted vocabulary/IDF.
            columns = sorted({m.vocabulary[f'tags:{t.casefold()}'] for t in tags if f'tags:{t.casefold()}' in m.vocabulary})
            if columns:
                profile = normalize(sparse.csr_matrix(([m.idf[c] for c in columns], ([0]*len(columns), columns)), shape=(1, m.features.shape[1])))
                scores = (m.features @ profile.T).toarray().ravel()
                allowed = np.array([i for i, item in enumerate(m.catalog) if item not in set(hidden)])
                order = allowed[np.lexsort((allowed, -scores[allowed]))[:k]]
                ranking = [(m.catalog[i], float(scores[i])) for i in order]
                method, fallback = 'tags', None
            else:
                fallback = 'unknown_tags'
        items = []
        for item, score in ranking:
            index = m.index[item]
            explanation = {'type': 'popular', 'text': 'Được nhiều người đánh giá tích cực trong dữ liệu tham chiếu.', 'shared_tags': [], 'reference_source_ids': []}
            if profile is not None and score > 0:
                contributions = m.features[index].multiply(profile).tocsr()
                ordered = sorted(zip(contributions.indices, contributions.data), key=lambda pair: -pair[1])[:4]
                features = [{'feature': self.feature_names[int(c)], 'contribution': float(v)} for c,v in ordered]
                labels = [entry['feature'].split(':', 1)[1] for entry in features]
                refs = []
                if ids:
                    sims = (m.features[ids] @ m.features[index].T).toarray().ravel()
                    refs = [m.catalog[ids[j]] for j in np.argsort(-sims)[:2] if sims[j] > 0]
                explanation = {'type': method, 'text': 'Có đặc điểm chung: ' + ', '.join(labels),
                               'shared_tags': labels, 'reference_source_ids': refs, 'components': features}
            elif profile is not None:
                explanation = {'type': 'no_signal', 'text': 'Chưa có tín hiệu tương đồng rõ với sở thích của bạn.', 'shared_tags': [], 'reference_source_ids': []}
            items.append({'source_id': item, 'score': score, 'explanation': explanation})
        if fallback:
            LOG.info('recommendation_fallback model=%s reason=%s', self.run_id, fallback)
        return items, method, fallback
