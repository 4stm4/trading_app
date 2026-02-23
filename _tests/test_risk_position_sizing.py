from services.strategy_engine.risk import calculate_position_risk


def test_position_risk_respects_contract_multiplier_for_futures():
    # CCH6-подобный кейс: 1.0 цены = 10 ₽
    params = calculate_position_risk(
        entry=291.0,
        stop=293.04,
        target=286.93,
        deposit=3100.0,
        max_risk_percent=0.7,
        min_rr=2.0,
        max_position_notional_pct=80.0,
        position_step=1.0,
        contract_margin_rub=438.79,
        contract_multiplier=10.0,
        futures_margin_safety_factor=1.4,
    )

    assert params.valid is True
    # По риску должно быть не больше 1 контракта:
    # risk_per_contract = 2.04 * 10 = 20.4 ₽, max_risk = 21.7 ₽.
    assert params.position_size == 1.0
    assert round(params.risk_rub, 2) == 20.40
    assert round(params.risk_percent, 2) == 0.66
    assert round(params.potential_profit, 2) == 40.70
    assert round(params.rr, 2) == 2.00

