import dash_bootstrap_components as dbc
from dash import html, register_page
from dash_bootstrap_components._components.Container import Container

from trading_app.frontend.labels import signup
from trading_app.settings import language
from trading_app.frontend.components import navi_bar


register_page(
    __name__,
    path='/',
    title='CryptoManager',
    name='Our Analytics Dashboard'
)

layout = html.Div(
    children=[
        navi_bar,
        dbc.Container(
            class_name='content-layout',
            children=[
                dbc.Alert(
                    [
                        html.I(className="bi bi-x-octagon-fill me-2"),
                        "В процессе разработки.",
                    ],
                    color="danger",
                    className="d-flex align-items-center",
                ),
            ]
        )
    ]
)