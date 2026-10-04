from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from fastapi import Request
from backend.app.config import database_url


def connect(url=None):
    engine = create_engine(url or database_url(), pool_pre_ping=True, connect_args={'connect_timeout': 5})
    return engine, sessionmaker(engine, expire_on_commit=False)


def get_db(request: Request):
    with request.app.state.sessions() as session:
        yield session
