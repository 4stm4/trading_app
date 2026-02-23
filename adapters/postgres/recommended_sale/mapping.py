from __future__ import annotations

try:
    from entities.recommended_sale import RecommendedSale
except ModuleNotFoundError:  # pragma: no cover - package import fallback
    from trading_app.entities.recommended_sale import RecommendedSale

from .tables import RecommendedSaleTable


def to_entity(table_row: RecommendedSaleTable) -> RecommendedSale:
    return RecommendedSale(
        id=table_row.id,
        exchange=table_row.exchange,
        symbol=table_row.symbol,
        timeframe=table_row.timeframe,
        model_name=table_row.model_name,
        entry=table_row.entry,
        stop=table_row.stop,
        target=table_row.target,
        rr=table_row.rr,
        confidence=table_row.confidence,
        market_regime=table_row.market_regime,
        status=table_row.status,
        note=table_row.note,
        recommended_at=table_row.recommended_at,
        created_at=table_row.created_at,
        updated_at=table_row.updated_at,
    )


def to_table(
    recommended_sale: RecommendedSale,
    target: RecommendedSaleTable | None = None,
) -> RecommendedSaleTable:
    table_row = target or RecommendedSaleTable()
    table_row.exchange = str(recommended_sale.exchange or "").strip().lower()
    table_row.symbol = str(recommended_sale.symbol or "").strip().upper()
    table_row.timeframe = str(recommended_sale.timeframe or "1h").strip().lower() or "1h"
    table_row.model_name = str(recommended_sale.model_name or "balanced").strip().lower() or "balanced"
    table_row.entry = float(recommended_sale.entry) if recommended_sale.entry is not None else None
    table_row.stop = float(recommended_sale.stop) if recommended_sale.stop is not None else None
    table_row.target = float(recommended_sale.target) if recommended_sale.target is not None else None
    table_row.rr = float(recommended_sale.rr) if recommended_sale.rr is not None else None
    table_row.confidence = str(recommended_sale.confidence or "none").strip().lower() or "none"
    market_regime = str(recommended_sale.market_regime or "").strip().lower()
    table_row.market_regime = market_regime or None
    table_row.status = str(recommended_sale.status or "new").strip().lower() or "new"
    note = str(recommended_sale.note or "").strip()
    table_row.note = note or None
    if recommended_sale.recommended_at is not None:
        table_row.recommended_at = recommended_sale.recommended_at
    return table_row
