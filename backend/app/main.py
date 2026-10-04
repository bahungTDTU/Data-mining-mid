from contextlib import asynccontextmanager
from datetime import timedelta
import hashlib
import logging
import os
import secrets
import uuid
from argon2 import PasswordHasher
from argon2.exceptions import VerificationError
from fastapi import FastAPI, Request, Response, Depends, HTTPException, Query
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy import select, delete, func, text
from sqlalchemy.exc import IntegrityError
from backend.app.config import ROOT, origins
from backend.app.db import connect, get_db
from backend.app.models import User, LoginSession, Game, Tag, Preference, TagPreference, HiddenGame, Wishlist, Event, ExperimentRun, ModelVersion, now
from backend.app.schemas import Credentials, Preferences, GameInput, Feedback
from backend.app.recommender import RecommendationService, sha

LOG = logging.getLogger(__name__)
PASSWORDS = PasswordHasher()
DUMMY_HASH = PASSWORDS.hash(secrets.token_hex(24))


def token_hash(value):
    return hashlib.sha256(value.encode()).hexdigest()


def identity(request: Request, db=Depends(get_db)):
    session = db.get(LoginSession, token_hash(request.cookies.get('playnext_midterm_session', '')))
    if not session or session.expires_at <= now():
        raise HTTPException(401, 'Vui lòng đăng nhập lại.')
    user = db.get(User, session.user_id)
    if request.method not in ('GET', 'HEAD', 'OPTIONS'):
        if not secrets.compare_digest(request.headers.get('x-csrf-token', ''), session.csrf_token):
            raise HTTPException(403, 'Phiên bảo vệ không hợp lệ. Hãy tải lại trang.')
    return user, session


def game_json(game):
    return {'id': game.id, 'source_id': game.source_id, 'title': game.title, 'genres': game.genres,
            'tags': [tag.name for tag in sorted(game.tags, key=lambda tag: tag.name)], 'source': 'UCSD Steam'}


def locked(db, user):
    return db.scalar(select(User).where(User.id == user.id).with_for_update().execution_options(populate_existing=True))


def exists_game(db, game_id):
    game = db.get(Game, game_id)
    if not game:
        raise HTTPException(404, 'Không tìm thấy game.')
    return game


def preferences_json(db, user):
    game_ids = list(db.scalars(select(Preference.game_id).where(Preference.user_id == user.id).order_by(Preference.game_id)))
    tag_ids = list(db.scalars(select(TagPreference.tag_id).where(TagPreference.user_id == user.id).order_by(TagPreference.tag_id)))
    hidden_ids = list(db.scalars(select(HiddenGame.game_id).where(HiddenGame.user_id == user.id)))
    games = list(db.scalars(select(Game).where(Game.id.in_(game_ids)).order_by(Game.id))) if game_ids else []
    return {'game_ids': game_ids, 'tag_ids': tag_ids, 'hidden_ids': hidden_ids, 'games': [game_json(g) for g in games], 'profile_version': user.profile_version}


def create_app(db_url=None):
    @asynccontextmanager
    async def lifespan(app):
        app.state.engine, app.state.sessions = connect(db_url)
        try:
            app.state.recommender = RecommendationService()
            service = app.state.recommender
            with app.state.sessions() as db:
                registered = db.get(ModelVersion, service.run_id)
                if not registered or registered.status != 'ready' or registered.manifest_hash != sha(service.folder/'manifest.json'):
                    raise ValueError('Model not registered with matching import manifest')
        except (OSError, ValueError, KeyError) as exc:
            LOG.error('model_unavailable error_type=%s', type(exc).__name__)
            app.state.recommender = None
        yield
        app.state.engine.dispose()

    app = FastAPI(title='Playnext — Game recommendations', version='0.2.0', lifespan=lifespan)
    app.add_middleware(CORSMiddleware, allow_origins=origins(), allow_credentials=True,
                       allow_methods=['GET','POST','PUT','DELETE'], allow_headers=['Content-Type','X-CSRF-Token'])

    @app.middleware('http')
    async def guard(request, call_next):
        request.state.request_id = str(uuid.uuid4())
        if request.method in ('POST','PUT','PATCH','DELETE') and request.headers.get('origin') not in origins():
            return JSONResponse({'error': {'code': 'origin_forbidden', 'message': 'Nguồn yêu cầu không hợp lệ.'}, 'request_id': request.state.request_id}, status_code=403)
        response = await call_next(request)
        response.headers['X-Request-ID'] = request.state.request_id
        response.headers['X-Content-Type-Options'] = 'nosniff'
        if request.url.path.startswith('/api/me') or request.url.path.startswith('/api/auth'):
            response.headers['Cache-Control'] = 'no-store'
        return response

    @app.exception_handler(HTTPException)
    async def http_error(request, exc):
        return JSONResponse({'error': {'code': str(exc.status_code), 'message': exc.detail}, 'request_id': request.state.request_id}, status_code=exc.status_code)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request, exc):
        # Do not echo submitted password/token values in error details.
        return JSONResponse({'error': {'code': 'validation', 'message': 'Dữ liệu không hợp lệ. Kiểm tra email, mật khẩu và các lựa chọn.'}, 'request_id': request.state.request_id}, status_code=422)

    @app.get('/api/health')
    def health(request: Request, db=Depends(get_db)):
        db.execute(text('SELECT 1'))
        return {'status': 'ok' if request.app.state.recommender else 'degraded', 'model_loaded': bool(request.app.state.recommender)}

    @app.post('/api/auth/register', status_code=201)
    def register(data: Credentials, db=Depends(get_db)):
        user = User(email=str(data.email).casefold(), password_hash=PASSWORDS.hash(data.password))
        db.add(user)
        try:
            db.commit()
        except IntegrityError:
            db.rollback()
            raise HTTPException(409, 'Email đã được sử dụng.')
        return {'id': user.id, 'email': user.email}

    @app.post('/api/auth/login')
    def login(data: Credentials, request: Request, response: Response, db=Depends(get_db)):
        user = db.scalar(select(User).where(User.email == str(data.email).casefold()))
        try:
            PASSWORDS.verify(user.password_hash if user else DUMMY_HASH, data.password)
        except VerificationError:
            raise HTTPException(401, 'Email hoặc mật khẩu không đúng.')
        if not user:
            raise HTTPException(401, 'Email hoặc mật khẩu không đúng.')
        old = db.get(LoginSession, token_hash(request.cookies.get('playnext_midterm_session','')))
        if old:
            db.delete(old)
        token, csrf = secrets.token_urlsafe(32), secrets.token_hex(32)
        db.add(LoginSession(token_hash=token_hash(token), user_id=user.id, csrf_token=csrf, expires_at=now()+timedelta(days=7)))
        db.commit()
        response.set_cookie('playnext_midterm_session', token, httponly=True, secure=os.getenv('COOKIE_SECURE','false').lower()=='true', samesite='lax', max_age=604800)
        return {'id': user.id, 'email': user.email, 'role': user.role, 'csrf_token': csrf}

    @app.get('/api/auth/me')
    def me(auth=Depends(identity)):
        user, session = auth
        return {'id': user.id, 'email': user.email, 'role': user.role, 'csrf_token': session.csrf_token}

    @app.post('/api/auth/logout', status_code=204)
    def logout(response: Response, auth=Depends(identity), db=Depends(get_db)):
        db.delete(auth[1]); db.commit()
        response.delete_cookie('playnext_midterm_session')

    @app.get('/api/games')
    def games(q: str = Query('', max_length=100), tag: int | None = None, page: int = Query(1, ge=1), page_size: int = Query(12, ge=1, le=100), db=Depends(get_db)):
        query = select(Game)
        if q.strip():
            query = query.where(Game.title.icontains(q.strip(), autoescape=True))
        if tag is not None:
            query = query.where(Game.tags.any(Tag.id == tag))
        total = db.scalar(select(func.count()).select_from(query.subquery()))
        items = db.scalars(query.order_by(Game.train_popularity.desc(), Game.id).offset((page-1)*page_size).limit(page_size)).all()
        return {'items': [game_json(g) for g in items], 'page': page, 'page_size': page_size, 'total': total}

    @app.get('/api/tags')
    def tags(db=Depends(get_db)):
        return [{'id': t.id, 'name': t.name} for t in db.scalars(select(Tag).order_by(Tag.name))]

    @app.get('/api/games/{game_id}')
    def game_detail(game_id: int, request: Request, db=Depends(get_db)):
        game = exists_game(db, game_id)
        result = {**game_json(game), 'description': None, 'similar_games': []}
        service = request.app.state.recommender
        if service:
            ranking, _, _ = service.recommend([game.source_id], [], [], 6)
            indexed = {g.source_id: g for g in db.scalars(select(Game).where(Game.source_id.in_([r['source_id'] for r in ranking])))}
            result['similar_games'] = [game_json(indexed[r['source_id']]) for r in ranking if r['score']>0 and r['source_id'] in indexed]
        return result

    @app.get('/api/me/preferences')
    def preferences(auth=Depends(identity), db=Depends(get_db)):
        return preferences_json(db, auth[0])

    @app.put('/api/me/preferences')
    def put_preferences(data: Preferences, auth=Depends(identity), db=Depends(get_db)):
        user = locked(db, auth[0])
        if len(db.scalars(select(Game.id).where(Game.id.in_(data.game_ids))).all()) != len(data.game_ids) or len(db.scalars(select(Tag.id).where(Tag.id.in_(data.tag_ids))).all()) != len(data.tag_ids):
            raise HTTPException(422, 'Game hoặc tag không tồn tại.')
        db.execute(delete(Preference).where(Preference.user_id==user.id))
        db.execute(delete(TagPreference).where(TagPreference.user_id==user.id))
        db.execute(delete(HiddenGame).where(HiddenGame.user_id==user.id, HiddenGame.game_id.in_(data.game_ids)))
        db.add_all([Preference(user_id=user.id, game_id=i) for i in data.game_ids])
        db.add_all([TagPreference(user_id=user.id, tag_id=i) for i in data.tag_ids])
        user.profile_version += 1
        db.commit()
        return preferences_json(db, user)

    @app.get('/api/me/recommendations')
    def recommend(request: Request, k: int = Query(10, ge=1, le=50), auth=Depends(identity), db=Depends(get_db)):
        user = auth[0]
        prefs = preferences_json(db, user)
        service = request.app.state.recommender
        support = [g['source_id'] for g in prefs['games']]
        hidden = list(db.scalars(select(Game.source_id).where(Game.id.in_(prefs['hidden_ids']))))
        tags = list(db.scalars(select(Tag.name).where(Tag.id.in_(prefs['tag_ids']))))
        if service:
            entries, method, fallback = service.recommend(support, hidden, tags, k)
            indexed = {g.source_id: g for g in db.scalars(select(Game).where(Game.source_id.in_([e['source_id'] for e in entries])))}
            items = [{**e, 'game': game_json(indexed[e['source_id']])} for e in entries if e['source_id'] in indexed]
            version = service.run_id
        else:
            games = db.scalars(select(Game).where(Game.id.not_in(prefs['game_ids']+prefs['hidden_ids'])).order_by(Game.train_popularity.desc(),Game.id).limit(k)).all()
            items = [{'game': game_json(g), 'score': float(g.train_popularity), 'explanation': {'type':'popular', 'text':'Danh sách phổ biến từ dữ liệu đã nhập; mô hình cá nhân hóa tạm thời chưa sẵn sàng.', 'shared_tags':[], 'reference_source_ids':[]}} for g in games]
            method, fallback, version = 'popular', 'artifact_unavailable', None
            LOG.warning('recommendation_fallback reason=artifact_unavailable')
        return {'items': items, 'model_version': version, 'method': method, 'profile_version': user.profile_version, 'support_count': len(support), 'fallback_reason': fallback}

    @app.post('/api/me/feedback', status_code=201)
    def feedback(data: Feedback, auth=Depends(identity), db=Depends(get_db)):
        user = locked(db, auth[0]); exists_game(db, data.game_id)
        key = str(data.idempotency_key)
        prior = db.scalar(select(Event).where(Event.user_id==user.id, Event.idempotency_key==key))
        if prior:
            if prior.game_id != data.game_id or prior.type != data.type:
                raise HTTPException(409, 'Khóa yêu cầu đã được dùng cho phản hồi khác.')
            return {'event_id': prior.id, 'profile_version': user.profile_version}
        event = Event(user_id=user.id, game_id=data.game_id, type=data.type, idempotency_key=key)
        db.add(event)
        if data.type == 'like':
            if not db.get(Preference, (user.id,data.game_id)):
                if db.scalar(select(func.count()).select_from(Preference).where(Preference.user_id==user.id)) >= 100:
                    raise HTTPException(422, 'Tối đa 100 game yêu thích; hãy chỉnh sở thích trước.')
                db.add(Preference(user_id=user.id, game_id=data.game_id))
            db.execute(delete(HiddenGame).where(HiddenGame.user_id==user.id, HiddenGame.game_id==data.game_id))
        elif data.type == 'dislike':
            db.execute(delete(Preference).where(Preference.user_id==user.id, Preference.game_id==data.game_id))
            if not db.get(HiddenGame, (user.id,data.game_id)):
                db.add(HiddenGame(user_id=user.id, game_id=data.game_id))
        if data.type != 'click':
            user.profile_version += 1
        db.commit()
        return {'event_id': event.id, 'profile_version': user.profile_version}

    @app.get('/api/me/wishlist')
    def wishlist(page: int=Query(1,ge=1), page_size: int=Query(24,ge=1,le=100), auth=Depends(identity), db=Depends(get_db)):
        query = select(Game).join(Wishlist, Game.id==Wishlist.game_id).where(Wishlist.user_id==auth[0].id)
        total = db.scalar(select(func.count()).select_from(query.subquery()))
        games = db.scalars(query.order_by(Wishlist.created_at.desc(), Game.id).offset((page-1)*page_size).limit(page_size)).all()
        return {'items':[game_json(g) for g in games], 'total':total,'page':page,'page_size':page_size}

    @app.post('/api/me/wishlist', status_code=201)
    def save_wishlist(data: GameInput, response: Response, auth=Depends(identity), db=Depends(get_db)):
        user = locked(db, auth[0]); exists_game(db, data.game_id)
        if db.get(Wishlist, (user.id,data.game_id)):
            response.status_code = 200
        else:
            db.add(Wishlist(user_id=user.id,game_id=data.game_id)); db.commit()
        return {'game_id': data.game_id}

    @app.delete('/api/me/wishlist/{game_id}', status_code=204)
    def remove_wishlist(game_id: int, auth=Depends(identity), db=Depends(get_db)):
        db.execute(delete(Wishlist).where(Wishlist.user_id==auth[0].id, Wishlist.game_id==game_id)); db.commit()

    @app.get('/api/admin/experiments')
    def experiments(page: int=Query(1,ge=1), page_size: int=Query(20,ge=1,le=100), auth=Depends(identity), db=Depends(get_db)):
        if auth[0].role != 'admin':
            raise HTTPException(403, 'Chỉ quản trị viên được xem nghiên cứu.')
        runs = db.scalars(select(ExperimentRun).order_by(ExperimentRun.id.desc()).offset((page-1)*page_size).limit(page_size)).all()
        return {'items':[{'id':r.id,'seed':r.seed,'protocol':r.protocol,'metrics':r.metrics,'cohorts':r.cohorts,'config':r.config} for r in runs], 'total':db.scalar(select(func.count()).select_from(ExperimentRun))}

    dist = ROOT/'frontend/dist'
    if dist.exists():
        app.mount('/', StaticFiles(directory=dist, html=True), name='frontend')
    return app


app = create_app()
