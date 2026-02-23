from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class ClientAccount(BaseModel):
    id: int = Field(alias="IdAccount")
    model_config = {"populate_by_name": True}


class ClientSubAccount(BaseModel):
    id: int = Field(alias="IdSubAccount")
    account_id: int = Field(alias="IdAccount")
    model_config = {"populate_by_name": True}


class SubAccountRazdel(BaseModel):
    id: int = Field(alias="IdRazdel")
    account_id: int = Field(alias="IdAccount")
    sub_account_id: int = Field(alias="IdSubAccount")
    razdel_group_id: int = Field(alias="IdRazdelGroup")
    r_code: str = Field(alias="RCode")
    model_config = {"populate_by_name": True}


class ClientPosition(BaseModel):
    id: int = Field(alias="IdPosition")
    account_id: int = Field(alias="IdAccount")
    sub_account_id: int = Field(alias="IdSubAccount")
    razdel_id: int = Field(alias="IdRazdel")
    object_id: int = Field(alias="IdObject")
    fi_balance_id: int = Field(alias="IdFiBalance")
    balance_group_id: int = Field(alias="IdBalanceGroup")
    assets_percent: float = Field(alias="AssetsPercent")
    pstnkd: float = Field(alias="PSTNKD")
    is_money: bool = Field(alias="IsMoney")
    is_rur: bool = Field(alias="IsRur")
    uch_price: float = Field(alias="UchPrice")
    torg_pos: float = Field(alias="TorgPos")
    price: float = Field(alias="Price")
    daily_pl: float = Field(alias="DailyPL")
    daily_pl_percent_to_market_cur_price: float = Field(alias="DailyPLPercentToMarketCurPrice")
    back_pos: float = Field(alias="BackPos")
    prev_quote: float = Field(alias="PrevQuote")
    trn_in: float = Field(alias="TrnIn")
    trn_out: float = Field(alias="TrnOut")
    daily_buy_volume: float = Field(alias="DailyBuyVolume")
    daily_sell_volume: float = Field(alias="DailySellVolume")
    daily_buy_quantity: float = Field(alias="DailyBuyQuantity")
    daily_sell_quantity: float = Field(alias="DailySellQuantity")
    nkd: float = Field(alias="NKD")
    price_step: float = Field(alias="PriceStep")
    lot: int = Field(alias="Lot")
    npl_to_market_cur_price: float = Field(alias="NPLtoMarketCurPrice")
    npl_percent: float = Field(alias="NPLPercent")
    plan_long: float = Field(alias="PlanLong")
    plan_short: float = Field(alias="PlanShort")
    model_config = {"populate_by_name": True}


class ClientBalance(BaseModel):
    id: int = Field(alias="DataId")
    account_id: int = Field(alias="IdAccount")
    sub_account_id: int = Field(alias="IdSubAccount")
    razdel_group_id: int = Field(alias="IdRazdelGroup")
    margin_initial: float = Field(alias="MarginInitial")
    margin_minimum: float = Field(alias="MarginMinimum")
    margin_requirement: float = Field(alias="MarginRequirement")
    money: float = Field(alias="Money")
    money_initial: float = Field(alias="MoneyInitial")
    balance: float = Field(alias="Balance")
    prev_balance: float = Field(alias="PrevBalance")
    portfolio_cost: float = Field(alias="PortfolioCost")
    liquid_balance: float = Field(alias="LiquidBalance")
    requirements: float = Field(alias="Requirements")
    immediate_requirements: float = Field(alias="ImmediateRequirements")
    npl: float = Field(alias="NPL")
    daily_pl: float = Field(alias="DailyPL")
    npl_percent: float = Field(alias="NPLPercent")
    daily_pl_percent: float = Field(alias="DailyPLPercent")
    nkd: float = Field(alias="NKD")
    model_config = {"populate_by_name": True}


class Order(BaseModel):
    num_e_document: int = Field(alias="NumEDocument")
    client_order_num: int = Field(alias="ClientOrderNum")
    account_id: int = Field(alias="IdAccount")
    sub_account_id: int = Field(alias="IdSubAccount")
    razdel_id: int = Field(alias="IdRazdel")
    allowed_order_params_id: int = Field(alias="IdAllowedOrderParams")
    accept_time: datetime = Field(alias="AcceptTime")
    order_type_id: int = Field(alias="IdOrderType")
    object_id: int = Field(alias="IdObject")
    market_board_id: int = Field(alias="IdMarketBoard")
    limit_price: Optional[float] = Field(alias="LimitPrice", default=None)
    buy_sell: int = Field(alias="BuySell")
    quantity: int = Field(alias="Quantity")
    comment: Optional[str] = Field(alias="Comment", default=None)
    login: str = Field(alias="Login")
    order_status_id: int = Field(alias="IdOrderStatus")
    rest: int = Field(alias="Rest")
    price: float = Field(alias="Price")
    broker_comment: Optional[str] = Field(alias="BrokerComment", default=None)
    model_config = {"populate_by_name": True}


class OrderRevision(BaseModel):
    quantity: int = Field(alias="Quantity")
    order_type_id: int = Field(alias="IdOrderType")
    price: float = Field(alias="Price")
    rest: int = Field(alias="Rest")
    filled_price: float = Field(alias="FilledPrice")
    limit_price: Optional[float] = Field(alias="LimitPrice", default=None)
    stop_price: Optional[float] = Field(alias="StopPrice", default=None)
    order_status_id: int = Field(alias="IdOrderStatus")
    price_type_id: int = Field(alias="IdPriceType")
    quantity_type_id: int = Field(alias="IdQuantityType")
    with_draw_time: Optional[datetime] = Field(alias="WithDrawTime", default=None)
    change_time: datetime = Field(alias="ChangeTime")
    life_time_id: int = Field(alias="IdLifeTime")
    open_quantity: int = Field(alias="OpenQuantity")
    broker_comment: Optional[str] = Field(alias="BrokerComment", default=None)
    broker_error_code: Optional[str] = Field(alias="BrokerErrorCode", default=None)
    order_status_revision: int = Field(alias="OrderStatusRevision")
    model_config = {"populate_by_name": True}


class ClientOperation(BaseModel):
    id: int = Field(alias="IdOperation")
    time_operation: datetime = Field(alias="TimeOperation")
    market_board_id: int = Field(alias="IdMarketBoard")
    operation_type_id: str = Field(alias="IdOperationType")
    object_id: int = Field(alias="IdObject")
    buy_sell: int = Field(alias="BuySell")
    quantity: int = Field(alias="Quantity")
    price: float = Field(alias="Price")
    account_id: int = Field(alias="IdAccount")
    sub_account_id: int = Field(alias="IdSubAccount")
    razdel_id: int = Field(alias="IdRazdel")
    num_e_document: int = Field(alias="NumEDocument")
    model_config = {"populate_by_name": True}


class AllowedOrderParam(BaseModel):
    id: int = Field(alias="IdAllowedOrderParams")
    object_group_id: int = Field(alias="IdObjectGroup")
    market_board_id: int = Field(alias="IdMarketBoard")
    order_type_id: int = Field(alias="IdOrderType")
    document_type_id: int = Field(alias="IdDocumentType")
    quantity_type_id: int = Field(alias="IdQuantityType")
    price_type_id: int = Field(alias="IdPriceType")
    life_time_id: int = Field(alias="IdLifeTime")
    execution_type: int = Field(alias="ExecutionType")
    model_config = {"populate_by_name": True}
