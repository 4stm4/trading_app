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
    create_user_schema(engine)
    create_instrument_schema(engine)
    create_market_candle_schema(engine)
    create_recommended_sale_schema(engine)
    create_trading_model_schema(engine)
    create_trading_system_schema(engine)
    create_trading_system_version_schema(engine)
    create_trading_system_run_schema(engine)
    create_trading_system_signal_schema(engine)


def create_instrument_schema(engine: Engine) -> None:
    from .instrument.tables import InstrumentTable

    Base.metadata.create_all(bind=engine, tables=[InstrumentTable.__table__])


def create_user_schema(engine: Engine) -> None:
    from .user.tables import UserTable

    Base.metadata.create_all(bind=engine, tables=[UserTable.__table__])


def create_recommended_sale_schema(engine: Engine) -> None:
    from .recommended_sale.tables import RecommendedSaleTable

    Base.metadata.create_all(bind=engine, tables=[RecommendedSaleTable.__table__])


def create_market_candle_schema(engine: Engine) -> None:
    from .market_candle.tables import MarketCandleTable

    Base.metadata.create_all(bind=engine, tables=[MarketCandleTable.__table__])


def create_trading_model_schema(engine: Engine) -> None:
    from .trading_model.tables import TradingModelTable

    Base.metadata.create_all(bind=engine, tables=[TradingModelTable.__table__])


def create_trading_system_schema(engine: Engine) -> None:
    from .trading_system.tables import TradingSystemTable

    Base.metadata.create_all(bind=engine, tables=[TradingSystemTable.__table__])


def create_trading_system_version_schema(engine: Engine) -> None:
    from .trading_system_version.tables import TradingSystemVersionTable

    Base.metadata.create_all(bind=engine, tables=[TradingSystemVersionTable.__table__])


def create_trading_system_run_schema(engine: Engine) -> None:
    from .trading_system_run.tables import TradingSystemRunTable

    Base.metadata.create_all(bind=engine, tables=[TradingSystemRunTable.__table__])


def create_trading_system_signal_schema(engine: Engine) -> None:
    from .trading_system_signal.tables import TradingSystemSignalTable

    Base.metadata.create_all(bind=engine, tables=[TradingSystemSignalTable.__table__])
