import dash_bootstrap_components as dbc
from dash import html

from trading_app.frontend import labels
from trading_app.settings import language

LOGO = "assets/images/crypto_manager_middle.png"
HOME_PAGE = "http://crypto-manager.online"

navi_bar = dbc.Navbar(
    class_name="header-container",
    children=[
        html.Div(
            className="header-leftside",
            children=[
                html.A(
                    dbc.Row(
                        [
                            dbc.Col(html.Img(src=LOGO, height="40px")),
                        ],
                        align="center",
                        className="g-0",
                    ),
                    href=HOME_PAGE,
                    style={"textDecoration": "none"},
                ),
            ],
        ),
        html.Div(
            className="header-censterside",
            children=[],
        ),
        html.Div(
            className="header-rightside",
            children=[
                html.Div(
                    className="bar-container",
                    children=[
                        html.A(
                            "Войти",
                            className="button-login",
                            href="signup",
                        ),
                    ],
                ),
                html.Div(
                    className="bar-container",
                    children=[
                        dbc.Button(
                            children=[labels.navi_bar[language]["Registration"]],
                            className="button-registration",
                            href="register",
                        ),
                    ],
                ),
                html.Div(
                    className="bar-container",
                    children=[html.I(className="fa-solid fa-globe icon-language")],
                ),
                html.Div(
                    className="bar-container",
                    children=[
                        dbc.Button(
                            class_name="themes-switch",
                            title="Switch to light theme",
                            children=[
                                html.Span(
                                    className="check",
                                    children=[
                                        html.Span(
                                            className="icon",
                                            children=[
                                                html.I(className="fa-regular fa-sun sun"),
                                                html.I(className="fa-regular fa-moon moon"),
                                            ],
                                        ),
                                    ],
                                )
                            ],
                        )
                    ],
                ),
            ],
        ),
    ],
    color="dark",
    dark=True,
)
