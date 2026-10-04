"""Application users are separate from historical research users."""
from datetime import datetime, timezone
import uuid
from sqlalchemy import String, Integer, DateTime, ForeignKey, CheckConstraint, UniqueConstraint, JSON
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def now():
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = 'users'
    __table_args__ = (CheckConstraint("role IN ('user','admin')"),)
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    email: Mapped[str] = mapped_column(String(254), unique=True)
    password_hash: Mapped[str] = mapped_column(String(255))
    role: Mapped[str] = mapped_column(String(10), default='user')
    profile_version: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class LoginSession(Base):
    __tablename__ = 'sessions'
    token_hash: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[str] = mapped_column(ForeignKey('users.id', ondelete='CASCADE'), index=True)
    csrf_token: Mapped[str] = mapped_column(String(64))
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class GameTag(Base):
    __tablename__ = 'game_tags'
    game_id: Mapped[int] = mapped_column(ForeignKey('games.id', ondelete='CASCADE'), primary_key=True)
    tag_id: Mapped[int] = mapped_column(ForeignKey('tags.id', ondelete='CASCADE'), primary_key=True, index=True)


class Game(Base):
    __tablename__ = 'games'
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    source_id: Mapped[str] = mapped_column(String(30), unique=True)
    title: Mapped[str] = mapped_column(String(500))
    genres: Mapped[list] = mapped_column(JSON, default=list)
    train_popularity: Mapped[int] = mapped_column(Integer, default=0)
    tags: Mapped[list['Tag']] = relationship(secondary='game_tags', lazy='selectin')


class Tag(Base):
    __tablename__ = 'tags'
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(100), unique=True)


class Preference(Base):
    __tablename__ = 'user_preferences'
    user_id: Mapped[str] = mapped_column(ForeignKey('users.id', ondelete='CASCADE'), primary_key=True)
    game_id: Mapped[int] = mapped_column(ForeignKey('games.id', ondelete='CASCADE'), primary_key=True)


class TagPreference(Base):
    __tablename__ = 'user_tag_preferences'
    user_id: Mapped[str] = mapped_column(ForeignKey('users.id', ondelete='CASCADE'), primary_key=True)
    tag_id: Mapped[int] = mapped_column(ForeignKey('tags.id', ondelete='CASCADE'), primary_key=True)


class HiddenGame(Base):
    __tablename__ = 'hidden_games'
    user_id: Mapped[str] = mapped_column(ForeignKey('users.id', ondelete='CASCADE'), primary_key=True)
    game_id: Mapped[int] = mapped_column(ForeignKey('games.id', ondelete='CASCADE'), primary_key=True)


class Event(Base):
    __tablename__ = 'interaction_events'
    __table_args__ = (UniqueConstraint('user_id', 'idempotency_key'), CheckConstraint("type IN ('like','dislike','click')"))
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id: Mapped[str] = mapped_column(ForeignKey('users.id', ondelete='CASCADE'), index=True)
    game_id: Mapped[int] = mapped_column(ForeignKey('games.id', ondelete='CASCADE'))
    type: Mapped[str] = mapped_column(String(10))
    idempotency_key: Mapped[str] = mapped_column(String(36))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class Wishlist(Base):
    __tablename__ = 'wishlist'
    user_id: Mapped[str] = mapped_column(ForeignKey('users.id', ondelete='CASCADE'), primary_key=True)
    game_id: Mapped[int] = mapped_column(ForeignKey('games.id', ondelete='CASCADE'), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class ModelVersion(Base):
    __tablename__ = 'model_versions'
    id: Mapped[str] = mapped_column(String(100), primary_key=True)
    artifact_path: Mapped[str] = mapped_column(String(500))
    manifest_hash: Mapped[str] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(20), default='ready')


class ExperimentRun(Base):
    __tablename__ = 'experiment_runs'
    id: Mapped[str] = mapped_column(String(100), primary_key=True)
    model_version: Mapped[str] = mapped_column(ForeignKey('model_versions.id'))
    seed: Mapped[int] = mapped_column(Integer)
    protocol: Mapped[str] = mapped_column(String(30))
    metrics: Mapped[list] = mapped_column(JSON)
    cohorts: Mapped[dict] = mapped_column(JSON)
    config: Mapped[dict] = mapped_column(JSON)
