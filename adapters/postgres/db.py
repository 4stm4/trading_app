from __future__ import annotations

from contextlib import contextmanager
from collections.abc import Iterator

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from .base import Base


def create_postgres_engine(database_url: str, *, echo: bool = False) -> Engine:
    return create_engine(database_url, future=True, echo=echo, pool_pre_ping=True)


def create_session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(
        bind=engine,
        class_=Session,
        autoflush=False,
        autocommit=False,
        expire_on_commit=False,
    )


@contextmanager
def session_scope(session_factory: sessionmaker[Session]) -> Iterator[Session]:
    session = session_factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def create_schema(engine: Engine) -> None:
    create_instrument_schema(engine)
    create_user_schema(engine)


def create_instrument_schema(engine: Engine) -> None:
    from .instrument.tables import InstrumentTable

    Base.metadata.create_all(bind=engine, tables=[InstrumentTable.__table__])


def create_user_schema(engine: Engine) -> None:
    from .user.tables import UserTable

    Base.metadata.create_all(bind=engine, tables=[UserTable.__table__])
