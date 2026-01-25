import dash_bootstrap_components as dbc
from dash import html, register_page

from trading_app.frontend.components import navi_bar


register_page(__name__, path="/", title="CryptoManager", name="Our Analytics Dashboard")

layout = html.Div(
    children=[
        navi_bar,
        dbc.Container(
            class_name="content-layout",
            children=[
                dbc.Alert(
                    [
                        html.I(className="bi bi-x-octagon-fill me-2"),
                        "В процессе разработки.",
                    ],
                    color="danger",
                    className="d-flex align-items-center",
                ),
            ],
        ),
    ]
)
