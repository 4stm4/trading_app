"""
Dash веб-интерфейс для торговой системы
"""

import dash
from dash import dcc, html, Input, Output, State, dash_table
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import requests
from datetime import datetime

from adapters import (
    build_exchange_adapter,
    load_data_with_indicators_for_exchange,
    resolve_default_board,
)
from services.strategy_engine import get_model, generate_signal, MODELS

# Инициализация приложения
app = dash.Dash(__name__, title="Trading System")
app.config.suppress_callback_exceptions = True

# CSS для темной темы
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            /* Старые стили для Select (Dash <= 2.0) */
            .Select-control {
                background-color: #2d2d2d !important;
                border-color: #555 !important;
            }
            .Select-menu-outer {
                background-color: #2d2d2d !important;
                border-color: #555 !important;
            }
            .Select-option {
                background-color: #2d2d2d !important;
                color: #fafafa !important;
            }
            .Select-option.is-focused {
                background-color: #3d3d3d !important;
            }
            .Select-value-label {
                color: #fafafa !important;
            }
            .Select-placeholder {
                color: #999 !important;
            }
            .Select-input > input {
                color: #fafafa !important;
            }

            /* Новые стили для Dropdown (Dash >= 2.1) */
            .dash-dropdown {
                background-color: #2d2d2d !important;
            }
            .dash-dropdown .Select-control {
                background-color: #2d2d2d !important;
                border-color: #555 !important;
            }
            .dash-dropdown .Select-value-label,
            .dash-dropdown .Select-placeholder {
                color: #fafafa !important;
            }
            .dash-dropdown .Select-menu-outer {
                background-color: #2d2d2d !important;
                border-color: #555 !important;
            }
            .dash-dropdown .VirtualizedSelectOption {
                background-color: #2d2d2d !important;
                color: #fafafa !important;
            }
            .dash-dropdown .VirtualizedSelectFocusedOption {
                background-color: #3d3d3d !important;
                color: #fafafa !important;
            }

            /* Input fields */
            input[type="text"],
            input[type="number"] {
                background-color: #2d2d2d !important;
                color: #fafafa !important;
                border: 1px solid #555 !important;
            }

            /* Buttons hover effects */
            button:hover {
                opacity: 0.9;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''


# Стили
COLORS = {
    'background': '#0e1117',
    'text': '#fafafa',
    'card': '#1e2130',
    'primary': '#00d4ff',
    'success': '#00ff9f',
    'danger': '#ff4444',
    'warning': '#ffa500',
}

CARD_STYLE = {
    'backgroundColor': COLORS['card'],
    'padding': '20px',
    'borderRadius': '10px',
    'marginBottom': '20px',
    'color': COLORS['text']
}

INPUT_STYLE = {
    'width': '100%',
    'padding': '10px',
    'borderRadius': '5px',
    'border': '1px solid #555',
    'backgroundColor': '#2d2d2d',
    'color': '#fafafa',
    'fontSize': '14px'
}

DROPDOWN_STYLE = {
    'backgroundColor': '#2d2d2d',
    'color': '#fafafa',
    'borderColor': '#555'
}

# Layout приложения
app.layout = html.Div(style={'backgroundColor': COLORS['background'], 'minHeight': '100vh', 'padding': '20px'}, children=[
    # Заголовок
    html.Div([
        html.H1('📈 Trading System Dashboard',
                style={'color': COLORS['primary'], 'textAlign': 'center', 'marginBottom': '10px'}),
        html.P('Профессиональная торговая система с техническим анализом',
               style={'color': COLORS['text'], 'textAlign': 'center', 'opacity': '0.7'})
    ]),

    # Контрольная панель
    html.Div(style=CARD_STYLE, children=[
        html.H3('⚙️ Настройки', style={'color': COLORS['primary'], 'marginBottom': '15px'}),

        html.Div([
            # Тикер
            html.Div([
                html.Label('Инструмент:', style={'color': COLORS['text'], 'fontWeight': 'bold'}),
                dcc.Input(
                    id='ticker-input',
                    type='text',
                    value='SBER',
                    placeholder='Введите тикер (SBER, GAZP, CCH6...)',
                    style=INPUT_STYLE
                ),
            ], style={'marginBottom': '15px'}),

            # Таймфрейм и депозит в одной строке
            html.Div([
                html.Div([
                    html.Label('Таймфрейм:', style={'color': COLORS['text'], 'fontWeight': 'bold'}),
                    dcc.Dropdown(
                        id='timeframe-dropdown',
                        options=[
                            {'label': '5 минут', 'value': '5m'},
                            {'label': '10 минут', 'value': '10m'},
                            {'label': '15 минут', 'value': '15m'},
                            {'label': '30 минут', 'value': '30m'},
                            {'label': '1 час', 'value': '1h'},
                            {'label': '4 часа', 'value': '4h'},
                            {'label': '1 день', 'value': '1d'},
                        ],
                        value='1h',
                        style=DROPDOWN_STYLE
                    ),
                ], style={'width': '48%', 'display': 'inline-block'}),

                html.Div([
                    html.Label('Депозит (₽):', style={'color': COLORS['text'], 'fontWeight': 'bold'}),
                    dcc.Input(
                        id='deposit-input',
                        type='number',
                        value=100000,
                        min=10000,
                        step=10000,
                        style=INPUT_STYLE
                    ),
                ], style={'width': '48%', 'display': 'inline-block', 'marginLeft': '4%'}),
            ], style={'marginBottom': '15px'}),

            # Модель и тип рынка
            html.Div([
                html.Div([
                    html.Label('Торговая модель:', style={'color': COLORS['text'], 'fontWeight': 'bold'}),
                    dcc.Dropdown(
                        id='model-dropdown',
                        options=[
                            {'label': f'{name} (RR {MODELS[name].min_rr})', 'value': name}
                            for name in MODELS.keys()
                        ],
                        value='balanced',
                        style=DROPDOWN_STYLE
                    ),
                ], style={'width': '48%', 'display': 'inline-block'}),

                html.Div([
                    html.Label('Тип рынка:', style={'color': COLORS['text'], 'fontWeight': 'bold'}),
                    dcc.Dropdown(
                        id='market-type-dropdown',
                        options=[
                            {'label': 'Акции (Stock)', 'value': 'stock'},
                            {'label': 'Фьючерсы (Futures)', 'value': 'futures'},
                        ],
                        value='stock',
                        style=DROPDOWN_STYLE
                    ),
                ], style={'width': '48%', 'display': 'inline-block', 'marginLeft': '4%'}),
            ], style={'marginBottom': '15px'}),

            # Кнопки
            html.Div([
                html.Button('🎯 Получить сигнал', id='signal-button', n_clicks=0,
                           style={'padding': '12px 24px', 'backgroundColor': COLORS['primary'],
                                  'color': 'white', 'border': 'none', 'borderRadius': '5px',
                                  'cursor': 'pointer', 'fontWeight': 'bold', 'marginRight': '10px'}),
                html.Button('📊 Запустить бэктест', id='backtest-button', n_clicks=0,
                           style={'padding': '12px 24px', 'backgroundColor': COLORS['success'],
                                  'color': 'white', 'border': 'none', 'borderRadius': '5px',
                                  'cursor': 'pointer', 'fontWeight': 'bold'}),
            ], style={'textAlign': 'center', 'marginTop': '20px'}),
        ]),
    ]),

    # Индикатор загрузки
    dcc.Loading(
        id="loading",
        type="circle",
        children=[
            # График
            html.Div(id='chart-container', style=CARD_STYLE),

            # Сигнал
            html.Div(id='signal-container', style=CARD_STYLE),

            # Бэктест результаты
            html.Div(id='backtest-container', style=CARD_STYLE),
        ]
    ),

    # Хранилище данных
    dcc.Store(id='data-store'),
])


def create_candlestick_chart(df, signal=None):
    """Создание свечного графика с индикаторами"""

    # Создаем субплоты (график + RSI)
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.7, 0.3],
        subplot_titles=('Цена и индикаторы', 'RSI')
    )

    # Свечи
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='Цена',
            increasing_line_color=COLORS['success'],
            decreasing_line_color=COLORS['danger']
        ),
        row=1, col=1
    )

    # MA50
    if 'ma50' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['ma50'],
                name='MA50',
                line=dict(color='orange', width=1.5)
            ),
            row=1, col=1
        )

    # MA200
    if 'ma200' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['ma200'],
                name='MA200',
                line=dict(color='blue', width=1.5)
            ),
            row=1, col=1
        )

    # Сигнал (точка входа)
    if signal and signal.signal != 'none':
        color = COLORS['success'] if signal.signal == 'long' else COLORS['danger']
        fig.add_trace(
            go.Scatter(
                x=[df.index[-1]],
                y=[signal.entry],
                mode='markers',
                name=f'Вход ({signal.signal.upper()})',
                marker=dict(color=color, size=15, symbol='star')
            ),
            row=1, col=1
        )

        # Линии стопа и тейка
        fig.add_hline(y=signal.stop, line_dash="dash", line_color=COLORS['danger'],
                     annotation_text=f"Stop: {signal.stop:.2f}", row=1, col=1)
        fig.add_hline(y=signal.target, line_dash="dash", line_color=COLORS['success'],
                     annotation_text=f"Target: {signal.target:.2f}", row=1, col=1)

    # RSI
    if 'rsi' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['rsi'],
                name='RSI',
                line=dict(color=COLORS['primary'], width=2)
            ),
            row=2, col=1
        )

        # Зоны перекупленности/перепроданности
        fig.add_hline(y=70, line_dash="dot", line_color='red', annotation_text="Overbought",
                     row=2, col=1)
        fig.add_hline(y=30, line_dash="dot", line_color='green', annotation_text="Oversold",
                     row=2, col=1)

    # Настройка layout
    fig.update_layout(
        template='plotly_dark',
        height=700,
        showlegend=True,
        xaxis_rangeslider_visible=False,
        hovermode='x unified',
        dragmode='zoom',  # Зум по умолчанию
        paper_bgcolor=COLORS['card'],
        plot_bgcolor=COLORS['background'],
        font=dict(color=COLORS['text']),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#2d2d2d')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#2d2d2d')

    return fig


@app.callback(
    [Output('chart-container', 'children'),
     Output('signal-container', 'children'),
     Output('data-store', 'data')],
    [Input('signal-button', 'n_clicks')],
    [State('ticker-input', 'value'),
     State('timeframe-dropdown', 'value'),
     State('deposit-input', 'value'),
     State('model-dropdown', 'value'),
     State('market-type-dropdown', 'value')]
)
def update_signal(n_clicks, ticker, timeframe, deposit, model_name, market_type):
    """Обновление графика и сигнала"""
    if n_clicks == 0:
        return html.Div(), html.Div(), None

    try:
        from datetime import datetime, timedelta

        # Загружаем данные
        exchange = 'moex'
        engine = 'futures' if market_type == 'futures' else 'stock'
        market = 'forts' if market_type == 'futures' else 'shares'
        board = resolve_default_board(exchange, engine)

        # Определяем период данных
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')

        adapter = build_exchange_adapter(exchange, engine, market)
        df, _ = load_data_with_indicators_for_exchange(
            exchange=exchange,
            ticker=ticker,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            board=board,
            adapter=adapter,
        )

        if df.empty:
            return (
                html.Div([html.H4('❌ Нет данных', style={'color': COLORS['danger']})]),
                html.Div(),
                None
            )

        # Генерируем сигнал
        model = get_model(model_name)
        signal = generate_signal(df, deposit, model)

        # Создаем график
        fig = create_candlestick_chart(df, signal)
        chart = dcc.Graph(
            figure=fig,
            config={
                'displayModeBar': True,
                'displaylogo': False,
                'scrollZoom': True,  # Зум колесом мыши
                'modeBarButtonsToRemove': ['select2d', 'lasso2d'],
                'doubleClick': 'reset'  # Двойной клик сбросит зум
            }
        )

        # Формируем карточку сигнала
        signal_dict = signal.to_dict()

        signal_emoji = {'long': '🟢 LONG', 'short': '🔴 SHORT', 'none': '⚪ НЕТ СИГНАЛА'}
        signal_color = {'long': COLORS['success'], 'short': COLORS['danger'], 'none': COLORS['text']}

        signal_card = html.Div([
            html.H3('🎯 Торговый сигнал', style={'color': COLORS['primary'], 'marginBottom': '15px'}),

            html.Div([
                html.H2(signal_emoji[signal_dict['signal']],
                       style={'color': signal_color[signal_dict['signal']], 'textAlign': 'center'}),
                html.P(f"Уверенность: {signal_dict['confidence'].upper()}",
                      style={'textAlign': 'center', 'opacity': '0.8'})
            ]),

            html.Hr(style={'borderColor': '#444'}),

            # Параметры сделки
            html.Div([
                html.Div([
                    html.H4('💰 Параметры сделки', style={'color': COLORS['primary']}),
                    html.Table([
                        html.Tr([html.Td('Вход:', style={'fontWeight': 'bold'}),
                                html.Td(f"{signal_dict['entry']:.2f}")]),
                        html.Tr([html.Td('Стоп:', style={'fontWeight': 'bold'}),
                                html.Td(f"{signal_dict['stop']:.2f}")]),
                        html.Tr([html.Td('Цель:', style={'fontWeight': 'bold'}),
                                html.Td(f"{signal_dict['target']:.2f}")]),
                        html.Tr([html.Td('RR:', style={'fontWeight': 'bold'}),
                                html.Td(f"{signal_dict['rr']:.2f}")]),
                    ], style={'width': '100%', 'color': COLORS['text']})
                ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'}),

                html.Div([
                    html.H4('📊 Риск-менеджмент', style={'color': COLORS['primary']}),
                    html.Table([
                        html.Tr([html.Td('Позиция:', style={'fontWeight': 'bold'}),
                                html.Td(f"{signal_dict['position_size']:.0f} контр.")]),
                        html.Tr([html.Td('Риск ₽:', style={'fontWeight': 'bold'}),
                                html.Td(f"{signal_dict['risk_rub']:.2f}")]),
                        html.Tr([html.Td('Риск %:', style={'fontWeight': 'bold'}),
                                html.Td(f"{signal_dict['risk_percent']:.2f}%")]),
                        html.Tr([html.Td('Потенциал:', style={'fontWeight': 'bold'}),
                                html.Td(f"{abs(signal_dict['target'] - signal_dict['entry']) * signal_dict['position_size']:.2f} ₽")]),
                    ], style={'width': '100%', 'color': COLORS['text']})
                ], style={'width': '48%', 'display': 'inline-block', 'marginLeft': '4%', 'verticalAlign': 'top'}),
            ]),

            html.Hr(style={'borderColor': '#444'}),

            # Индикаторы
            html.H4('📈 Индикаторы', style={'color': COLORS['primary']}),
            html.Table([
                html.Tr([html.Td('Структура:', style={'fontWeight': 'bold'}),
                        html.Td(signal_dict['structure'])]),
                html.Tr([html.Td('Фаза:', style={'fontWeight': 'bold'}),
                        html.Td(signal_dict['phase'])]),
                html.Tr([html.Td('RSI:', style={'fontWeight': 'bold'}),
                        html.Td(f"{signal_dict['rsi']:.1f}")]),
                html.Tr([html.Td('Volume ratio:', style={'fontWeight': 'bold'}),
                        html.Td(f"{signal_dict['volume_ratio']:.2f}x")]),
                html.Tr([html.Td('ATR:', style={'fontWeight': 'bold'}),
                        html.Td(f"{signal_dict['atr']:.2f}")]),
            ], style={'width': '100%', 'color': COLORS['text']}),

            # Предупреждения
            html.Div([
                html.H4('⚠️ Предупреждения', style={'color': COLORS['warning']}),
                html.Ul([html.Li(w) for w in signal_dict['warnings']])
            ]) if signal_dict['warnings'] else html.Div(),
        ])

        return chart, signal_card, df.to_json()

    except Exception as e:
        error_msg = html.Div([
            html.H4('❌ Ошибка', style={'color': COLORS['danger']}),
            html.P(str(e))
        ])
        return error_msg, html.Div(), None


@app.callback(
    Output('backtest-container', 'children'),
    [Input('backtest-button', 'n_clicks')],
    [State('ticker-input', 'value'),
     State('timeframe-dropdown', 'value'),
     State('deposit-input', 'value'),
     State('model-dropdown', 'value'),
     State('market-type-dropdown', 'value')]
)
def update_backtest(n_clicks, ticker, timeframe, deposit, model_name, market_type):
    """Обновление результатов бэктеста"""
    if n_clicks == 0:
        return html.Div()

    try:
        from datetime import datetime, timedelta
        from services.strategy_engine import run_backtest

        # Загружаем данные
        exchange = 'moex'
        engine = 'futures' if market_type == 'futures' else 'stock'
        market = 'forts' if market_type == 'futures' else 'shares'
        board = resolve_default_board(exchange, engine)

        # Определяем период данных
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')

        adapter = build_exchange_adapter(exchange, engine, market)
        df, _ = load_data_with_indicators_for_exchange(
            exchange=exchange,
            ticker=ticker,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            board=board,
            adapter=adapter,
        )

        if df.empty:
            return html.H4('❌ Нет данных', style={'color': COLORS['danger']})

        # Запускаем бэктест
        model = get_model(model_name)
        results = run_backtest(df, generate_signal, deposit, model)
        res_dict = results.to_dict()

        # Проверка на отсутствие сделок
        if res_dict['total_trades'] == 0:
            return html.Div([
                html.H4('⚠️ Нет сделок', style={'color': COLORS['warning']}),
                html.P('На выбранном периоде система не сгенерировала ни одной сделки. Попробуйте:'),
                html.Ul([
                    html.Li('Изменить таймфрейм'),
                    html.Li('Выбрать другую модель'),
                    html.Li('Проверить доступность данных')
                ])
            ])

        # Оценка системы
        score = 0
        checks = []

        winrate = res_dict.get('winrate', 0) or 0
        if winrate >= 40:
            score += 1
            checks.append(('✅', 'Winrate >= 40%'))
        else:
            checks.append(('❌', 'Winrate < 40%'))

        profit_factor = res_dict.get('profit_factor', 0) or 0
        if profit_factor >= 1.5:
            score += 1
            checks.append(('✅', 'Profit Factor >= 1.5'))
        else:
            checks.append(('❌', 'Profit Factor < 1.5'))

        expectancy = res_dict.get('expectancy', 0) or 0
        if expectancy > 0:
            score += 1
            checks.append(('✅', 'Expectancy > 0'))
        else:
            checks.append(('❌', 'Expectancy <= 0'))

        max_dd = res_dict.get('max_drawdown_percent', 0) or 0
        if max_dd < 20:
            score += 1
            checks.append(('✅', 'Drawdown < 20%'))
        else:
            checks.append(('⚠️', 'Drawdown >= 20%'))

        rating_color = COLORS['success'] if score >= 3 else COLORS['warning'] if score >= 2 else COLORS['danger']
        rating_text = '🌟 ПЕРСПЕКТИВНА' if score >= 3 else '⚠️ ТРЕБУЕТ ДОРАБОТКИ' if score >= 2 else '❌ НЕ РЕКОМЕНДУЕТСЯ'

        # Безопасное получение значений с дефолтами
        total_trades = res_dict.get('total_trades', 0) or 0
        winrate = res_dict.get('winrate', 0) or 0
        profit_factor = res_dict.get('profit_factor', 0) or 0
        return_pct = res_dict.get('return_pct', 0) or 0
        final_balance = res_dict.get('final_balance', deposit) or deposit
        total_profit = res_dict.get('total_profit', 0) or 0
        max_dd_pct = res_dict.get('max_drawdown_percent', 0) or 0
        avg_win = res_dict.get('avg_win', 0) or 0
        avg_loss = res_dict.get('avg_loss', 0) or 0
        expectancy_val = res_dict.get('expectancy', 0) or 0
        sharpe = res_dict.get('sharpe_ratio', 0) or 0

        return html.Div([
            html.H3('📊 Результаты бэктеста', style={'color': COLORS['primary'], 'marginBottom': '15px'}),

            # Основные метрики
            html.Div([
                html.Div([
                    html.H2(f"{total_trades}", style={'color': COLORS['primary'], 'margin': '0'}),
                    html.P('Всего сделок', style={'opacity': '0.7', 'margin': '5px 0'})
                ], style={'textAlign': 'center', 'width': '24%', 'display': 'inline-block'}),

                html.Div([
                    html.H2(f"{winrate:.1f}%", style={'color': COLORS['success'], 'margin': '0'}),
                    html.P('Winrate', style={'opacity': '0.7', 'margin': '5px 0'})
                ], style={'textAlign': 'center', 'width': '24%', 'display': 'inline-block'}),

                html.Div([
                    html.H2(f"{profit_factor:.2f}", style={'color': COLORS['primary'], 'margin': '0'}),
                    html.P('Profit Factor', style={'opacity': '0.7', 'margin': '5px 0'})
                ], style={'textAlign': 'center', 'width': '24%', 'display': 'inline-block'}),

                html.Div([
                    html.H2(f"{return_pct:+.1f}%",
                           style={'color': COLORS['success'] if return_pct > 0 else COLORS['danger'], 'margin': '0'}),
                    html.P('Доходность', style={'opacity': '0.7', 'margin': '5px 0'})
                ], style={'textAlign': 'center', 'width': '24%', 'display': 'inline-block'}),
            ], style={'marginBottom': '20px'}),

            html.Hr(style={'borderColor': '#444'}),

            # Детальная статистика
            html.Div([
                html.Div([
                    html.H4('💰 Финансы', style={'color': COLORS['primary']}),
                    html.Table([
                        html.Tr([html.Td('Начальный депо:', style={'fontWeight': 'bold'}),
                                html.Td(f"{deposit:,.0f} ₽")]),
                        html.Tr([html.Td('Конечный депо:', style={'fontWeight': 'bold'}),
                                html.Td(f"{final_balance:,.0f} ₽")]),
                        html.Tr([html.Td('Прибыль:', style={'fontWeight': 'bold'}),
                                html.Td(f"{total_profit:+,.0f} ₽")]),
                        html.Tr([html.Td('Max DD:', style={'fontWeight': 'bold'}),
                                html.Td(f"{max_dd_pct:.2f}%")]),
                    ], style={'width': '100%', 'color': COLORS['text']})
                ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'}),

                html.Div([
                    html.H4('📈 Метрики', style={'color': COLORS['primary']}),
                    html.Table([
                        html.Tr([html.Td('Средний выигрыш:', style={'fontWeight': 'bold'}),
                                html.Td(f"{avg_win:.2f} ₽")]),
                        html.Tr([html.Td('Средний проигрыш:', style={'fontWeight': 'bold'}),
                                html.Td(f"{abs(avg_loss):.2f} ₽")]),
                        html.Tr([html.Td('Expectancy:', style={'fontWeight': 'bold'}),
                                html.Td(f"{expectancy_val:.2f} ₽")]),
                        html.Tr([html.Td('Sharpe Ratio:', style={'fontWeight': 'bold'}),
                                html.Td(f"{sharpe:.2f}")]),
                    ], style={'width': '100%', 'color': COLORS['text']})
                ], style={'width': '48%', 'display': 'inline-block', 'marginLeft': '4%', 'verticalAlign': 'top'}),
            ]),

            html.Hr(style={'borderColor': '#444'}),

            # Оценка системы
            html.Div([
                html.H4('🎯 Оценка системы', style={'color': COLORS['primary']}),
                html.Ul([html.Li(f"{emoji} {text}") for emoji, text in checks]),
                html.H3(f"Оценка: {score}/4 - {rating_text}", style={'color': rating_color, 'textAlign': 'center'})
            ])
        ])

    except Exception as e:
        return html.Div([
            html.H4('❌ Ошибка бэктеста', style={'color': COLORS['danger']}),
            html.P(str(e))
        ])


if __name__ == '__main__':
    app.run_server(debug=True, port=8050)
