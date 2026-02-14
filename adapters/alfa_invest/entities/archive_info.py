from pydantic import BaseModel, Field, model_validator, field_validator
from typing import Optional, List, Union
from datetime import datetime
import uuid


class ArchiveQueryPayload(BaseModel):
    id: int = Field(alias="IdFi")
    candle_type: int = Field(alias="CandleType", default=0)
    interval: Optional[str] = Field(alias="Interval", default=None)
    period: Optional[int] = Field(alias="Period", default=None)
    time_frame: Optional[str] = Field(alias="TimeFrame", default=None)
    first_day: datetime = Field(alias="FirstDay")
    days_count: Optional[int] = Field(alias="DaysCount", default=None)
    last_day: Optional[datetime] = Field(alias="LastDay", default=None)
    at_least_one_candle: bool = Field(alias="AtLeastOneCandle", default=False)
    take_last_n_candles: Optional[int] = Field(alias="TakeLastNCandles", default=None)
    scope: Optional[str] = Field(alias="Scope", default=None)

    @model_validator(mode="after")
    def check_time_params(self):
        if not self.time_frame and (self.interval is None or self.period is None):
            raise ValueError(
                "Either 'TimeFrame' must be provided, or both 'Interval' and 'Period'."
            )
        if self.days_count is None and self.last_day is None:
            raise ValueError("Either 'DaysCount' or 'LastDay' must be specified.")
        return self

    model_config = {"populate_by_name": True}


class ArchiveQueryRequest(BaseModel):
    command: str = Field(alias="Command")
    channel: str = Field(alias="Channel")
    id: str = Field(alias="Id")
    payload: ArchiveQueryPayload = Field(alias="Payload")

    @field_validator("id")
    @classmethod
    def validate_id_is_uuid(cls, v: str) -> str:
        uuid.UUID(v)
        return v

    @field_validator("channel")
    @classmethod
    def validate_channel(cls, v: str) -> str:
        if v != "#Archive.Query":
            raise ValueError("Channel must be '#Archive.Query'")
        return v

    model_config = {"populate_by_name": True}


class ArchiveListenRequest(BaseModel):
    command: str = Field(alias="Command")
    channel: str = Field(alias="Channel")

    model_config = {"populate_by_name": True}


class OHLCVCandle(BaseModel):
    open: float = Field(alias="Open")
    close: float = Field(alias="Close")
    low: float = Field(alias="Low")
    high: float = Field(alias="High")
    volume: int = Field(alias="Volume")
    volume_ask: int = Field(alias="VolumeAsk")
    open_int: int = Field(alias="OpenInt")
    time: datetime = Field(alias="Time")

    model_config = {"populate_by_name": True}


class MPVLevel(BaseModel):
    price: float = Field(alias="Price")
    volume: int = Field(alias="Volume")
    volume_ask: int = Field(alias="VolumeAsk")

    model_config = {"populate_by_name": True}


class MPVCandle(BaseModel):
    open: float = Field(alias="Open")
    close: float = Field(alias="Close")
    time: datetime = Field(alias="Time")
    levels: List[MPVLevel] = Field(alias="Levels")

    model_config = {"populate_by_name": True}


ArchiveCandle = Union[OHLCVCandle, MPVCandle]
