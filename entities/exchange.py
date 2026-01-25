from typing import Optional, Any, List
from enum import Enum
from pydantic import BaseModel


class FilterType(Enum):
    price_filter = 'PRICE_FILTER'
    lot_size = 'LOT_SIZE'
    iceberg_parts = 'ICEBERG_PARTS'
    market_lot_size = 'MARKET_LOT_SIZE'
    trailing_delta = 'TRAILING_DELTA'
    percent_price_by_side = 'PERCENT_PRICE_BY_SIDE'
    notional = 'NOTIONAL'
    max_num_orders = 'MAX_NUM_ORDERS'
    max_num_algo_orders = 'MAX_NUM_ALGO_ORDERS'


class Filter(BaseModel):
    filterType: FilterType
    minPrice: Optional[str] = None
    maxPrice: Optional[str] = None
    tickSize: Optional[str] = None
    filterType: Optional[str] = None
    minQty: Optional[str] = None
    maxQty: Optional[str] = None
    stepSize: Optional[str] = None
    filterType: Optional[str] = None
    limit: Optional[int] = None
    filterType: Optional[str] = None
    minQty: Optional[str] = None
    maxQty: Optional[str] = None
    stepSize: Optional[str] = None
    filterType: Optional[str] = None
    minTrailingAboveDelta: Optional[int] = None
    maxTrailingAboveDelta: Optional[int] = None
    minTrailingBelowDelta: Optional[int] = None
    maxTrailingBelowDelta: Optional[int] = None
    filterType: Optional[str] = None
    bidMultiplierUp: Optional[str] = None
    bidMultiplierDown: Optional[str] = None
    askMultiplierUp: Optional[str] = None
    askMultiplierDown: Optional[str] = None
    avgPriceMins: Optional[int] = None
    filterType: Optional[str] = None
    minNotional: Optional[str] = None
    applyMinToMarket: Optional[bool] = None
    maxNotional: Optional[str] = None
    applyMaxToMarket: Optional[bool] = None
    avgPriceMins: Optional[int] = None
    filterType: Optional[str] = None
    maxNumOrders: Optional[int] = None
    filterType: Optional[str] = None
    maxNumAlgoOrders: Optional[int] = None


class OrderType(Enum):
    limit = 'LIMIT'
    limit_maker = 'LIMIT_MAKER'
    market = 'MARKET'
    stop_loss_limit = 'STOP_LOSS_LIMIT'
    take_profit_limit = 'TAKE_PROFIT_LIMIT'


class AllowedSelfTradePreventionMode(Enum):
    expire_taker = 'EXPIRE_TAKER'
    expire_maker = 'EXPIRE_MAKER'
    expire_both = 'EXPIRE_BOTH'


class Permission(Enum):
    spot = 'SPOT'
    margin = 'MARGIN'
    leveraged = 'LEVERAGED'
    trd_grp_004 = 'TRD_GRP_004'
    trd_grp_005 = 'TRD_GRP_005'
    trd_grp_006 = 'TRD_GRP_006'
    trd_grp_008 = 'TRD_GRP_008'
    trd_grp_009 = 'TRD_GRP_009'
    trd_grp_010 = 'TRD_GRP_010'
    trd_grp_011 = 'TRD_GRP_011'
    trd_grp_012 = 'TRD_GRP_012'
    trd_grp_013 = 'TRD_GRP_013'
    trd_grp_014 = 'TRD_GRP_014'
    trd_grp_015 = 'TRD_GRP_015'
    trd_grp_016 = 'TRD_GRP_016'
    trd_grp_017 = 'TRD_GRP_017'
    trd_grp_018 = 'TRD_GRP_018'
    trd_grp_019 = 'TRD_GRP_019'
    trd_grp_020 = 'TRD_GRP_020'
    trd_grp_021 = 'TRD_GRP_021'
    trd_grp_022 = 'TRD_GRP_022'
    trd_grp_023 = 'TRD_GRP_023'
    trd_grp_024 = 'TRD_GRP_024'
    trd_grp_025 = 'TRD_GRP_025'


class Symbol(BaseModel):
    symbol: str
    status: str
    baseAsset: str
    baseAssetPrecision: int
    quoteAsset: str
    quotePrecision: int
    quoteAssetPrecision: int
    baseCommissionPrecision: int
    quoteCommissionPrecision: int
    orderTypes: List[OrderType]
    icebergAllowed: bool
    ocoAllowed: bool
    quoteOrderQtyMarketAllowed: bool
    allowTrailingStop: bool
    cancelReplaceAllowed: bool
    isSpotTradingAllowed: bool
    isMarginTradingAllowed: bool
    filters: List[Filter]
    permissions: List[Permission]
    defaultSelfTradePreventionMode: str
    allowedSelfTradePreventionModes: List[AllowedSelfTradePreventionMode]


class RateLimit(BaseModel):
    rateLimitType: str
    interval: str
    intervalNum: int
    limit: int


class ExchangeInfo(BaseModel):
    timezone: str
    serverTime: int    
    rateLimits: List[RateLimit]
    exchangeFilters: List[Any]
    symbols: List[Symbol]
