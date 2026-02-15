"""
Flask API для торговой системы
"""

from flask import Flask, jsonify, request
from flask.views import MethodView
import logging
from typing import Any

from services.strategy_engine import (
    MODELS,
    get_model,
    generate_signal,
    run_backtest,
    compare_models_results
)
from adapters.moex import load_data_with_indicators
from adapters.moex.iss_client import MOEXAdapter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _resolve_contract_risk_params(adapter: Any, ticker: str, board: str) -> dict[str, float]:
    """Возвращает параметры фьючерсного контракта для риск-расчёта."""
    getter = getattr(adapter, "get_security_spec", None)
    if not callable(getter):
        return {}
    try:
        spec = getter(ticker, board=board) or {}
    except Exception:
        return {}

    result: dict[str, float] = {}
    try:
        margin = float(spec.get("initial_margin"))
        if margin > 0:
            result["contract_margin_rub"] = margin
    except (TypeError, ValueError):
        pass
    try:
        mult = float(spec.get("contract_multiplier"))
        if mult > 0:
            result["contract_multiplier"] = mult
    except (TypeError, ValueError):
        pass
    return result


def create_app():
    """Создание Flask приложения"""
    app = Flask(__name__)
    app.config['JSON_AS_ASCII'] = False  # Поддержка кириллицы

    # Регистрация endpoints
    app.add_url_rule('/api/health', view_func=health_check)
    app.add_url_rule('/api/models', view_func=ModelsView.as_view('models'))
    app.add_url_rule('/api/signal', view_func=SignalView.as_view('signal'))
    app.add_url_rule('/api/backtest', view_func=BacktestView.as_view('backtest'))
    app.add_url_rule('/api/optimize', view_func=OptimizeView.as_view('optimize'))

    # Error handlers
    @app.errorhandler(400)
    def bad_request(e):
        return jsonify({'error': 'Bad Request', 'message': str(e)}), 400

    @app.errorhandler(404)
    def not_found(e):
        return jsonify({'error': 'Not Found', 'message': str(e)}), 404

    @app.errorhandler(500)
    def internal_error(e):
        return jsonify({'error': 'Internal Server Error', 'message': str(e)}), 500

    return app


def health_check():
    """Проверка работоспособности API"""
    return jsonify({
        'status': 'ok',
        'service': 'Trading System API',
        'version': '1.0.0',
        'models_count': len(MODELS)
    })


class ModelsView(MethodView):
    """Endpoints для работы с моделями"""

    def get(self):
        """GET /api/models - Список доступных моделей"""
        models_info = {}
        for name, model in MODELS.items():
            models_info[name] = {
                'name': model.name,
                'description': model.description,
                'min_rr': model.min_rr,
                'max_risk_percent': model.max_risk_percent,
                'min_volume_ratio': model.min_volume_ratio,
                'atr_multiplier_stop': model.atr_multiplier_stop,
                'trend_required': model.trend_required,
                'allow_range': model.allow_range,
            }

        return jsonify({
            'models': models_info,
            'count': len(models_info)
        })


class SignalView(MethodView):
    """Endpoints для генерации сигналов"""

    def post(self):
        """
        POST /api/signal - Генерация торгового сигнала

        Body:
        {
            "ticker": "SBER",
            "timeframe": "1h",
            "deposit": 100000,
            "model": "balanced",
            "engine": "stock",
            "market": "shares",
            "board": "TQBR"
        }
        """
        data = request.get_json()

        # Валидация
        required_fields = ['ticker', 'deposit']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400

        ticker = data['ticker']
        deposit = data['deposit']
        timeframe = data.get('timeframe', '1h')
        model_name = data.get('model', 'balanced')
        engine = data.get('engine', 'stock')
        market = data.get('market', 'shares')
        board = data.get('board')

        if board is None:
            board = 'RFUD' if engine == 'futures' else 'TQBR'

        try:
            # Получаем модель
            model = get_model(model_name)

            # Загружаем данные
            adapter = MOEXAdapter(engine=engine, market=market)
            df, _ = load_data_with_indicators(
                ticker=ticker,
                timeframe=timeframe,
                board=board,
                adapter=adapter
            )

            if df.empty:
                return jsonify({'error': f'No data for {ticker}'}), 404

            # Генерируем сигнал
            risk_params = _resolve_contract_risk_params(adapter, ticker, board)
            signal = generate_signal(df, deposit, model, **risk_params)

            return jsonify({
                'ticker': ticker,
                'timeframe': timeframe,
                'model': model_name,
                'data_points': len(df),
                'period': {
                    'start': str(df.index[0]),
                    'end': str(df.index[-1])
                },
                'signal': signal.to_dict()
            })

        except ValueError as e:
            return jsonify({'error': str(e)}), 400
        except Exception as e:
            logger.error(f"Error generating signal: {e}")
            return jsonify({'error': 'Internal error', 'message': str(e)}), 500


class BacktestView(MethodView):
    """Endpoints для бэктестинга"""

    def post(self):
        """
        POST /api/backtest - Запуск бэктеста стратегии

        Body:
        {
            "ticker": "SBER",
            "timeframe": "1h",
            "deposit": 100000,
            "model": "conservative",
            "engine": "stock",
            "market": "shares"
        }
        """
        data = request.get_json()

        required_fields = ['ticker', 'deposit']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400

        ticker = data['ticker']
        deposit = data['deposit']
        timeframe = data.get('timeframe', '1h')
        model_name = data.get('model', 'balanced')
        engine = data.get('engine', 'stock')
        market = data.get('market', 'shares')
        board = data.get('board')

        if board is None:
            board = 'RFUD' if engine == 'futures' else 'TQBR'

        try:
            # Получаем модель
            model = get_model(model_name)

            # Загружаем данные
            adapter = MOEXAdapter(engine=engine, market=market)
            df, _ = load_data_with_indicators(
                ticker=ticker,
                timeframe=timeframe,
                board=board,
                adapter=adapter
            )

            if df.empty:
                return jsonify({'error': f'No data for {ticker}'}), 404

            risk_params = _resolve_contract_risk_params(adapter, ticker, board)
            risk_params["sl_tp_only"] = True

            # Запускаем бэктест
            results = run_backtest(
                df=df,
                signal_generator=generate_signal,
                deposit=deposit,
                model=model,
                signal_kwargs=risk_params or None,
            )

            return jsonify({
                'ticker': ticker,
                'timeframe': timeframe,
                'model': model_name,
                'data_points': len(df),
                'period': {
                    'start': str(df.index[0]),
                    'end': str(df.index[-1])
                },
                'results': results.to_dict()
            })

        except ValueError as e:
            return jsonify({'error': str(e)}), 400
        except Exception as e:
            logger.error(f"Error running backtest: {e}")
            return jsonify({'error': 'Internal error', 'message': str(e)}), 500


class OptimizeView(MethodView):
    """Endpoints для оптимизации моделей"""

    def post(self):
        """
        POST /api/optimize - Сравнение всех моделей

        Body:
        {
            "ticker": "SBER",
            "timeframe": "1h",
            "deposit": 100000,
            "engine": "stock",
            "market": "shares"
        }
        """
        data = request.get_json()

        required_fields = ['ticker', 'deposit']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400

        ticker = data['ticker']
        deposit = data['deposit']
        timeframe = data.get('timeframe', '1h')
        engine = data.get('engine', 'stock')
        market = data.get('market', 'shares')
        board = data.get('board')

        if board is None:
            board = 'RFUD' if engine == 'futures' else 'TQBR'

        try:
            # Загружаем данные
            adapter = MOEXAdapter(engine=engine, market=market)
            df, _ = load_data_with_indicators(
                ticker=ticker,
                timeframe=timeframe,
                board=board,
                adapter=adapter
            )

            if df.empty:
                return jsonify({'error': f'No data for {ticker}'}), 404

            risk_params = _resolve_contract_risk_params(adapter, ticker, board)
            risk_params["sl_tp_only"] = True

            # Запускаем бэктест для всех моделей
            results = []
            for model_name in MODELS.keys():
                model = get_model(model_name)
                backtest_result = run_backtest(
                    df=df,
                    signal_generator=generate_signal,
                    deposit=deposit,
                    model=model,
                    signal_kwargs=risk_params or None,
                )
                results.append(backtest_result)

            # Находим лучшую модель
            best_model = max(results, key=lambda x: x.expectancy)

            return jsonify({
                'ticker': ticker,
                'timeframe': timeframe,
                'data_points': len(df),
                'period': {
                    'start': str(df.index[0]),
                    'end': str(df.index[-1])
                },
                'models_tested': len(results),
                'results': [r.to_dict() for r in results],
                'best_model': {
                    'name': best_model.model_name,
                    'expectancy': best_model.expectancy,
                    'winrate': best_model.winrate,
                    'profit_factor': best_model.profit_factor
                }
            })

        except ValueError as e:
            return jsonify({'error': str(e)}), 400
        except Exception as e:
            logger.error(f"Error optimizing models: {e}")
            return jsonify({'error': 'Internal error', 'message': str(e)}), 500


if __name__ == '__main__':
    app = create_app()
    app.run(host='0.0.0.0', port=5000, debug=True)
