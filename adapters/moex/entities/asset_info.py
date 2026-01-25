from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import date


class Group(BaseModel):
    id: int = Field(alias="IdObjectGroup")
    name: str = Field(alias="NameObjectGroup")

    model_config = {"populate_by_name": True}


class Type(BaseModel):
    id: int = Field(alias="IdObjectType")
    group_id: int = Field(alias="IdObjectGroup")
    code: str = Field(alias="CodeObjectType")
    name: str = Field(alias="NameObjectType")
    short_name: str = Field(alias="ShortNameObjectType")

    model_config = {"populate_by_name": True}


class Instrument(BaseModel):
    id: int = Field(alias="IdFi")
    r_code: str = Field(alias="RCode")
    is_liquid: bool = Field(alias="IsLiquid")
    market_board_id: int = Field(alias="IdMarketBoard")

    model_config = {"populate_by_name": True}


class MarketBoard(BaseModel):
    id: int = Field(alias="IdMarketBoard")
    name: str = Field(alias="NameMarketBoard")
    desc: str = Field(alias="DescMarketBoard")
    r_code: str = Field(alias="RCode")
    currency_id: Optional[int] = Field(alias="IdObjectCurrency", default=None)

    model_config = {"populate_by_name": True}


class AssetInfo(BaseModel):
    id: int = Field(alias="IdObject")
    ticker: str = Field(alias="Ticker")
    isin: str = Field(alias="ISIN")
    name: str = Field(alias="Name")
    description: str = Field(alias="Description")
    nominal: float = Field(alias="Nominal")
    type_id: int = Field(alias="IdObjectType")
    group_id: int = Field(alias="IdObjectGroup")
    base_id: Optional[int] = Field(alias="IdObjectBase", default=None)
    face_unit_id: int = Field(alias="IdObjectFaceUnit")
    mat_date: Optional[date] = Field(alias="MatDateObject", default=None)
    instruments: List[Instrument] = Field(alias="Instruments")

    model_config = {"populate_by_name": True}
