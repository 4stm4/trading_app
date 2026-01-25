from typing import List
from datetime import datetime

from pydantic import BaseModel


class KLine(BaseModel):
    open_time: datetime
    open_price: str
    max_price: str
    min_price: str
    close_price: str
    value: str
    close_time: datetime
    volume_quota_currency: str
    amount_deals: int
    base_asset_volume: str
    quote_asset_volume: str
    ignore_field: str


    class Config:
        json_encoders = {
            datetime: lambda v: v.timestamp(),
        }


class KLines(BaseModel):
    lines: List[KLine]


    def to_list(self):
        klines_list = []
        for line in self.lines:
            klines_list.append(
                {
                    'Date': line.close_time,
                    'Open': line.open_price,
                    'High': line.max_price,
                    'Low': line.min_price,
                    'Close': line.close_price,
                }
            )
        return klines_list
