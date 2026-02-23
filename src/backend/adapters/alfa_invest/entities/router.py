from typing import Optional

from pydantic import BaseModel, Field


class RoutingRequest(BaseModel):
    """Base router message as per Alfa Investments PRO WebSocket API."""

    command: str = Field(alias="Command")
    channel: str = Field(alias="Channel")
    payload: Optional[str] = Field(default=None, alias="Payload")
    id: Optional[str] = Field(default=None, alias="Id")

    model_config = {"populate_by_name": True}


class RoutingError(BaseModel):
    command: str = Field(alias="Command")
    channel: str = Field(alias="Channel")
    message: str = Field(alias="Message")
    id: Optional[str] = Field(default=None, alias="Id")

    model_config = {"populate_by_name": True}
