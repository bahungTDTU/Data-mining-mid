"""Idempotent catalog/metrics import from a verified M1 run; no demo users."""
import json
from sqlalchemy import select
from backend.app.config import ROOT
from backend.app.db import connect
from backend.app.models import Game, Tag, ModelVersion, ExperimentRun
from backend.app.recommender import RecommendationService, sha, checked_path


def seed(db):
    service = RecommendationService()
    manifest = service.manifest
    source = checked_path(ROOT, manifest['data_config']['processed']) / 'games.json'
    if sha(source) != manifest['input_hashes']['games.json']:
        raise ValueError('Catalog metadata checksum mismatch')
    metadata = {g['source_id']: g for g in json.loads(source.read_text(encoding='utf-8'))}
    tags = {tag.name:tag for tag in db.scalars(select(Tag))}
    current = {game.source_id:game for game in db.scalars(select(Game))}
    for source_id in service.model.catalog:
        record = metadata[source_id]
        game = current.get(source_id)
        if game is None:
            game = Game(source_id=source_id, title=record['title'], genres=record['genres'])
            db.add(game)
        game.train_popularity = int(service.model.popularity[service.model.index[source_id]])
        attached = []
        for name in sorted(set(record['tags'])):
            if name not in tags:
                tags[name] = Tag(name=name)
                db.add(tags[name])
            attached.append(tags[name])
        game.tags = attached
    db.merge(ModelVersion(id=service.run_id, artifact_path=manifest['artifact_path'], manifest_hash=sha(service.folder/'manifest.json'), status='ready'))
    db.flush()
    db.merge(ExperimentRun(id=service.run_id, model_version=service.run_id, seed=manifest['config']['seed'], protocol=manifest['protocol_version'],
        metrics=json.loads((service.folder/'metrics.json').read_text()), cohorts=json.loads((service.folder/'cohorts.json').read_text()), config=manifest['config']))
    db.commit()
    return len(service.model.catalog)


if __name__ == '__main__':
    engine, factory = connect()
    with factory() as db:
        print(f'Imported {seed(db)} verified games and measured experiment metrics')
    engine.dispose()
