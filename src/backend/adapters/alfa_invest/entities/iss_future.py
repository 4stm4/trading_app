from datetime import date
from typing import Optional

from pydantic import BaseModel, Field


class FutureSecurity(BaseModel):
    secid: str = Field(alias="SECID")
    shortname: str = Field(alias="SHORTNAME")
    secname: str = Field(alias="SECNAME")
    assetcode: str = Field(alias="ASSETCODE")
    boardid: str = Field(alias="BOARDID")
    decimals: Optional[int] = Field(alias="DECIMALS", default=None)
    minstep: Optional[float] = Field(alias="MINSTEP", default=None)
    lastsettleprice: Optional[float] = Field(alias="LASTSETTLEPRICE", default=None)
    prevsettleprice: Optional[float] = Field(alias="PREVSETTLEPRICE", default=None)
    prevprice: Optional[float] = Field(alias="PREVPRICE", default=None)
    lasttradedate: Optional[date] = Field(alias="LASTTRADEDATE", default=None)
    lastdeldate: Optional[date] = Field(alias="LASTDELDATE", default=None)
    lotvolume: Optional[int] = Field(alias="LOTVOLUME", default=None)
    steprice: Optional[float] = Field(alias="STEPPRICE", default=None)
    scalperfee: Optional[float] = Field(alias="SCALPERFEE", default=None)
    buysellfee: Optional[float] = Field(alias="BUYSELLFEE", default=None)
    negotiatedfee: Optional[float] = Field(alias="NEGOTIATEDFEE", default=None)
    exercisefee: Optional[float] = Field(alias="EXERCISEFEE", default=None)
    initialmargin: Optional[float] = Field(alias="INITIALMARGIN", default=None)
    prevopenposition: Optional[int] = Field(alias="PREVOPENPOSITION", default=None)
    highlimit: Optional[float] = Field(alias="HIGHLIMIT", default=None)
    lowlimit: Optional[float] = Field(alias="LOWLIMIT", default=None)

    model_config = {"populate_by_name": True}


class FutureMarketData(BaseModel):
    secid: str = Field(alias="SECID")
    boardid: str = Field(alias="BOARDID")
    bid: Optional[float] = Field(alias="BID", default=None)
    offer: Optional[float] = Field(alias="OFFER", default=None)
    spread: Optional[float] = Field(alias="SPREAD", default=None)
    open: Optional[float] = Field(alias="OPEN", default=None)
    high: Optional[float] = Field(alias="HIGH", default=None)
    low: Optional[float] = Field(alias="LOW", default=None)
    last: Optional[float] = Field(alias="LAST", default=None)
    quantity: Optional[int] = Field(alias="QUANTITY", default=None)
    lastchange: Optional[float] = Field(alias="LASTCHANGE", default=None)
    settleprice: Optional[float] = Field(alias="SETTLEPRICE", default=None)
    settletoprevsettle: Optional[float] = Field(alias="SETTLETOPREVSETTLE", default=None)
    settletoprevsettleprc: Optional[float] = Field(alias="SETTLETOPREVSETTLEPRC", default=None)
    openposition: Optional[int] = Field(alias="OPENPOSITION", default=None)
    numtrades: Optional[int] = Field(alias="NUMTRADES", default=None)
    voltoday: Optional[int] = Field(alias="VOLTODAY", default=None)
    valtoday: Optional[int] = Field(alias="VALTODAY", default=None)
    valtoday_usd: Optional[int] = Field(alias="VALTODAY_USD", default=None)
    time: Optional[str] = Field(alias="TIME", default=None)
    tradedate: Optional[date] = Field(alias="TRADEDATE", default=None)
    trade_session_date: Optional[date] = Field(alias="TRADE_SESSION_DATE", default=None)

    model_config = {"populate_by_name": True}
