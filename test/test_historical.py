from datetime import datetime
from historical_data import historical_data
from pydantic import BaseModel


class HistoricalTrades(BaseModel):
    id: int
    price: str
    qty: str
    quoteQty: str
    time: datetime
    isBuyerMaker: bool
    isBestMatch: bool

    class Config:
        json_encoders = {
            datetime: lambda v: v.timestamp(),
        }

types_dict = {
    "<class 'str'>": 'str',
    "<class 'int'>": 'int',
    "<class 'bool'>": 'bool',
}

print(type(historical_data[0]))

dict_keys = historical_data[0].keys()

for key in dict_keys:
    print('{0}: {1}'.format(key, types_dict[str(type(historical_data[0][key]))]))


for data in historical_data:
    print(HistoricalTrades(**data))