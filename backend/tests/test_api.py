"""Integration tests in a new isolated PostgreSQL schema, migrated via Alembic."""
import uuid
from datetime import timedelta
import pytest
from alembic import command
from alembic.config import Config
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select, text, func
from sqlalchemy.engine import make_url
from backend.app.config import database_url
from backend.app.models import User, LoginSession, Event, now
from backend.app.main import create_app, PASSWORDS
from backend.seed import seed

ORIGIN = 'http://localhost:5173'


@pytest.fixture(scope='module')
def app():
    # Never drop public schema or alter the application's existing accounts.
    schema = 'test_m2_' + uuid.uuid4().hex
    engine = create_engine(database_url())
    with engine.begin() as connection:
        connection.execute(text(f'CREATE SCHEMA "{schema}"'))
    url = make_url(database_url()).update_query_dict({'options': f'-csearch_path={schema}'})
    isolated = create_engine(url)
    try:
        config = Config('backend/alembic.ini')
        with isolated.begin() as connection:
            config.attributes['connection'] = connection
            command.upgrade(config, 'head')
        from sqlalchemy.orm import Session
        with Session(isolated) as db:
            assert seed(db) == 2235
            assert seed(db) == 2235  # seed remains idempotent
        yield create_app(url)
        with isolated.begin() as connection:
            config.attributes['connection'] = connection
            command.downgrade(config, 'base')
            command.upgrade(config, 'head')
    finally:
        isolated.dispose()
        with engine.begin() as connection:
            connection.execute(text(f'DROP SCHEMA "{schema}" CASCADE'))
        engine.dispose()


@pytest.fixture
def client(app):
    with TestClient(app, headers={'Origin':ORIGIN}) as client:
        yield client


def login(client):
    email=f'{uuid.uuid4().hex}@example.com'
    data={'email':email,'password':'test-password-42'}
    assert client.post('/api/auth/register',json=data).status_code==201
    response=client.post('/api/auth/login',json=data)
    assert response.status_code==200
    client.headers['X-CSRF-Token']=response.json()['csrf_token']
    return data, response


def test_auth_security_and_logout(client):
    assert client.get('/api/me/preferences').status_code==401
    data, response=login(client)
    assert 'httponly' in response.headers['set-cookie'].lower()
    assert 'samesite=lax' in response.headers['set-cookie'].lower()
    assert client.post('/api/auth/register',json=data).status_code==409
    assert client.post('/api/auth/register',json={**data,'role':'admin'}).status_code==422
    assert client.post('/api/auth/login',json={**data,'password':'wrong-pass'}).status_code==401
    assert client.get('/api/admin/experiments').status_code==403
    assert client.put('/api/me/preferences',json={'game_ids':[]},headers={'X-CSRF-Token':'bad'}).status_code==403
    assert client.put('/api/me/preferences',json={'game_ids':[]},headers={'Origin':'https://evil.invalid'}).status_code==403
    with client.app.state.sessions() as db:
        user=db.scalar(select(User).where(User.email==data['email']))
        assert user.password_hash!=data['password']
        assert PASSWORDS.verify(user.password_hash,data['password'])
    token=client.cookies.get('playnext_midterm_session')
    assert token
    client.cookies.set('game_session','unrelated-final-term-session')
    assert client.get('/api/auth/me').status_code==200
    assert client.post('/api/auth/logout').status_code==204
    client.cookies.set('playnext_midterm_session',token)
    assert client.get('/api/auth/me').status_code==401


def test_profile_lock_refreshes_previously_loaded_identity(client):
    from backend.app.main import locked
    credentials, _ = login(client)
    with client.app.state.sessions() as first, client.app.state.sessions() as second:
        stale = first.scalar(select(User).where(User.email == credentials['email']))
        current = second.get(User, stale.id)
        current.profile_version = 7
        second.commit()
        assert stale.profile_version == 0
        assert locked(first, stale).profile_version == 7


def test_full_profile_zero_one_three_five_and_persistence(client):
    credentials,_=login(client)
    games=client.get('/api/games?page_size=10').json()['items']
    ids=[g['id'] for g in games[:5]]
    r=client.get('/api/me/recommendations').json()
    assert r['method']=='popular' and r['support_count']==0
    assert len(r['items'])==10
    for count in (1,3,5):
        p=client.put('/api/me/preferences',json={'game_ids':ids[:count]}).json()
        assert len(p['game_ids'])==count
        r=client.get('/api/me/recommendations').json()
        assert r['support_count']==count and r['method']=='content'
        assert {e['game']['id'] for e in r['items']}.isdisjoint(ids[:count])
        assert r['model_version']==client.app.state.recommender.run_id
        for e in r['items']:
            assert e['score']>=0
            if e['score']>0:
                assert e['explanation']['components']
                assert sum(c['contribution'] for c in e['explanation']['components']) <= e['score']+1e-10
    saved=r['items'][0]['game']['id']
    assert client.post('/api/me/wishlist',json={'game_id':saved}).status_code==201
    assert client.post('/api/me/wishlist',json={'game_id':saved}).status_code==200
    assert client.post('/api/auth/logout').status_code==204
    client.headers['X-CSRF-Token']=client.post('/api/auth/login',json=credentials).json()['csrf_token']
    assert client.get('/api/me/preferences').json()['game_ids']==sorted(ids)
    assert [g['id'] for g in client.get('/api/me/wishlist').json()['items']]==[saved]
    # Complete removal of preferences, including likes, must stay removed.
    client.put('/api/me/preferences',json={'game_ids':[]})
    assert client.get('/api/me/recommendations').json()['support_count']==0


def test_feedback_idempotency_hidden_and_like(client):
    login(client)
    g=client.get('/api/me/recommendations').json()['items'][0]['game']['id']
    payload={'game_id':g,'type':'dislike','idempotency_key':str(uuid.uuid4())}
    first=client.post('/api/me/feedback',json=payload)
    second=client.post('/api/me/feedback',json=payload)
    assert first.status_code==201 and first.json()==second.json()
    assert client.post('/api/me/feedback',json={**payload,'type':'like'}).status_code==409
    assert g not in [e['game']['id'] for e in client.get('/api/me/recommendations').json()['items']]
    client.post('/api/me/feedback',json={**payload,'type':'like','idempotency_key':str(uuid.uuid4())})
    p=client.get('/api/me/preferences').json()
    assert p['game_ids']==[g] and g not in p['hidden_ids']
    version=p['profile_version']
    client.post('/api/me/feedback',json={**payload,'type':'click','idempotency_key':str(uuid.uuid4())})
    assert client.get('/api/me/preferences').json()['profile_version']==version
    client.put('/api/me/preferences',json={'game_ids':[]})
    assert client.get('/api/me/preferences').json()['game_ids']==[]


def test_cross_user_data_isolation(client):
    a,_=login(client)
    game=client.get('/api/games').json()['items'][0]['id']
    client.post('/api/me/wishlist',json={'game_id':game})
    client.put('/api/me/preferences',json={'game_ids':[game]})
    client.post('/api/auth/logout')
    login(client)
    assert client.get('/api/me/wishlist').json()['total']==0
    assert client.get('/api/me/preferences').json()['game_ids']==[]
    client.delete('/api/me/wishlist/'+str(game))
    client.post('/api/auth/logout')
    client.headers['X-CSRF-Token']=client.post('/api/auth/login',json=a).json()['csrf_token']
    assert client.get('/api/me/wishlist').json()['total']==1


def test_validation_catalog_tags_and_admin_real_metrics(client):
    credentials,_=login(client)
    assert client.get('/api/games?page_size=101').status_code==422
    assert client.get('/api/me/recommendations?k=51').status_code==422
    assert client.put('/api/me/preferences',json={'game_ids':[1,1]}).status_code==422
    assert client.put('/api/me/preferences',json={'game_ids':[999999]}).status_code==422
    assert client.get('/api/games/999999').status_code==404
    assert client.get('/api/games?q=unlikely_missing_title_39283').json()['total']==0
    tags=client.get('/api/tags').json()
    action=next(t for t in tags if t['name']=='Action')
    client.put('/api/me/preferences',json={'game_ids':[],'tag_ids':[action['id']]})
    rec=client.get('/api/me/recommendations').json()
    assert rec['method']=='tags' and rec['support_count']==0
    assert all(e['explanation']['type']=='tags' for e in rec['items'])
    with client.app.state.sessions() as db:
        user=db.scalar(select(User).where(User.email==credentials['email']))
        user.role='admin'; db.commit()
    runs=client.get('/api/admin/experiments').json()['items']
    assert len(runs)==1 and len(runs[0]['metrics'])==36
    assert runs[0]['metrics'][0]['recall_at_10'] is not None


def test_expiry_and_degraded_model(client):
    credentials,_=login(client)
    service=client.app.state.recommender
    try:
        client.app.state.recommender=None
        assert client.get('/api/health').json()['status']=='degraded'
        r=client.get('/api/me/recommendations').json()
        assert r['fallback_reason']=='artifact_unavailable' and r['model_version'] is None
        assert len(r['items'])==10
    finally:
        client.app.state.recommender=service
    with client.app.state.sessions() as db:
        user=db.scalar(select(User).where(User.email==credentials['email']))
        session=db.scalar(select(LoginSession).where(LoginSession.user_id==user.id))
        session.expires_at=now()-timedelta(seconds=1); db.commit()
    assert client.get('/api/auth/me').status_code==401


def test_corrupted_model_is_rejected_before_loading(monkeypatch):
    from backend.app import recommender
    monkeypatch.setattr(recommender, 'sha', lambda _: 'invalid-checksum')
    with pytest.raises(ValueError, match='checksum'):
        recommender.RecommendationService()


def test_artifact_path_cannot_escape_root():
    from backend.app.recommender import checked_path, ROOT
    with pytest.raises(ValueError):
        checked_path(ROOT, '../outside/model.npz')
