from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import date, datetime


class FinInfoParams(BaseModel):
    id: int = Field(alias="IdFi")
    session_id: int = Field(alias="IdSession")
    trade_period_status_id: int = Field(alias="IdTradePeriodStatus")
    session_date: date = Field(alias="SessionDate")
    lot: int = Field(alias="Lot")
    price_step: float = Field(alias="PriceStep")
    price_step_cost: float = Field(alias="PriceStepCost")
    object_currency_id: int = Field(alias="IdObjectCurrency")  # выпуск валюты цены
    pstnkd: float = Field(alias="PSTNKD")  # НКД
    up_price: float = Field(alias="UpPrice")
    down_price: float = Field(alias="DownPrice")
    gt_buy: float = Field(alias="GtBuy")
    gt_sell: float = Field(alias="GtSell")
    face_value: float = Field(alias="FaceValue")

    model_config = {"populate_by_name": True}


class FinInfoLast(BaseModel):
    id: int = Field(alias="IdFi")
    session_id: int = Field(alias="IdSession")
    trade_period_status_id: int = Field(alias="IdTradePeriodStatus")
    last: float = Field(alias="Last")
    prev_last: float = Field(alias="PrevLast")
    last_qty: int = Field(alias="LastQty")
    last_time: datetime = Field(alias="LastTime")
    yield_: float = Field(alias="Yield")
    open: float = Field(alias="Open")
    high: float = Field(alias="High")
    low: float = Field(alias="Low")
    wa_price: float = Field(alias="WaPrice")
    yield_wa_price: float = Field(alias="YieldWaPrice")
    num_trades: float = Field(alias="NumTrades")
    val_today: float = Field(alias="ValToday")

    model_config = {"populate_by_name": True}

class FinInfoOrderBook(BaseModel):
    id: int = Field(alias="IdFi")
    session_id: int = Field(alias="IdSession")
    trade_period_status_id: int = Field(alias="IdTradePeriodStatus")
    bid: float = Field(alias="Bid")
    ask: float = Field(alias="Ask")
    bid_qty: float = Field(alias="BidQty")
    ask_qty: float = Field(alias="AskQty")
    sum_bid: float = Field(alias="SumBid")
    sum_ask: float = Field(alias="SumAsk")
    num_bids: int = Field(alias="NumBids")
    num_asks: int = Field(alias="NumAsks")
    high_bid: float = Field(alias="HighBid")
    low_ask: float = Field(alias="LowAsk")

    model_config = {"populate_by_name": True}

class OrderBookLine(BaseModel):
    price: float = Field(alias="Price")
    buy_qty: int = Field(alias="BuyQty")
    sell_qty: int = Field(alias="SellQty")

    model_config = {"populate_by_name": True}


class OrderBook(BaseModel):
    id: int = Field(alias="IdFi")
    lines: List[OrderBookLine] = Field(alias="Lines")

    model_config = {"populate_by_name": True}


class Trade(BaseModel):
    id: int = Field(alias="IdFI")
    trade_no: int = Field(alias="TradeNo")
    trade_type_id: int = Field(alias="IdTradeType")
    trade_time: datetime = Field(alias="TradeTime")
    price: float = Field(alias="Price")
    buy_sell: int = Field(alias="BuySell")
    yield_: float = Field(alias="Yield")
    qty: float = Field(alias="Qty")
    value: float = Field(alias="Value")
    pos: int = Field(alias="Pos")
    ba_time: datetime = Field(alias="BaTime")

    model_config = {"populate_by_name": True}
