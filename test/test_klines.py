from typing import List
from datetime import datetime
from klines_data import klines_data
from pydantic import BaseModel


lines = []
for line in klines_data:
    lines.append(
        KLine(
            **{
                'open_time': line[0],
                'open_price': line[1],
                'max_price': line[2],
                'min_price': line[3],
                'close_price': line[4],
                'value': line[5],
                'close_time': line[6],
                'volume_quota_currency': line[7],
                'amount_deals': line[8],
                'base_asset_volume': line[9],
                'quote_asset_volume': line[10],
                'ignore_field': line[11]
            }
        )
    )
for line in lines:
    print(line)

klines = KLines(lines=lines)
print(klines)
