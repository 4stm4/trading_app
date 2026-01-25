import dash_bootstrap_components as dbc
import requests
from dash import (
    Input,
    Output,
    State,
    callback,
    dcc,
    html,
    no_update,
    register_page,
)
from flask import request as flask_request

from trading_app.frontend.labels import signup
from trading_app.settings import language


IMG_LOGO_GOOGLE = "/assets/images/logogoogle.png"
IMG_LOGO_CRYPTOMANAGER = "/assets/images/crypto_manager_long.png"

register_page(
    __name__,
    path="/signup",
    title="{0}| CryptoManager".format(signup[language]["Registration"]),
    name="Our Analytics Dashboard",
)

layout = dbc.Container(
    class_name="content-layout",
    children=[
        dcc.Store(id="auth-user"),
        html.Div(
            className="content-card",
            children=[
                html.Div(
                    children=[
                        # LOGO
                        html.Div(
                            className="icon-wrap",
                            children=[
                                html.Img(src=IMG_LOGO_CRYPTOMANAGER, className="default-icon block")
                            ],
                        ),
                        # надпись
                        html.Div(
                            className="flex justify-between items-center mb-8",
                            children=[
                                html.Div("Войти", className="card-page-title"),
                            ],
                        ),
                        # форма для входа через почту или сотовый
                        html.Form(
                            children=[
                                html.Div(
                                    children=[
                                        html.Div(
                                            "Эл. почта/номер телефона",
                                            className="bn-formItem-label",
                                        ),
                                        html.Div(
                                            children=[
                                                html.Div(
                                                    children=[
                                                        dbc.Input(
                                                            class_name="bn-textField-input",
                                                            id="auth-email",
                                                            name="username",
                                                            type="text",
                                                            autoComplete="username",
                                                        )
                                                    ],
                                                    className="username-input-field",
                                                )
                                            ],
                                            className="css-4cffwv",
                                        ),
                                    ]
                                ),
                                html.Div(
                                    children=[
                                        html.Div("Пароль", className="bn-formItem-label"),
                                        html.Div(
                                            children=[
                                                html.Div(
                                                    children=[
                                                        dbc.Input(
                                                            class_name="bn-textField-input",
                                                            id="auth-password",
                                                            name="password",
                                                            type="password",
                                                            autoComplete="current-password",
                                                        )
                                                    ],
                                                    className="username-input-field",
                                                )
                                            ],
                                            className="css-4cffwv",
                                        ),
                                    ]
                                ),
                                html.Button(
                                    "Войти", className="next-button", id="auth-submit", n_clicks=0
                                ),
                                html.Div(id="auth-feedback", className="auth-feedback"),
                            ]
                        ),
                        html.Div(
                            className="oauth-container",
                            children=[
                                html.Div(className="oauth-lines"),
                                html.Div("или", className="oauth-label"),
                                html.Div(className="oauth-lines"),
                            ],
                        ),
                        html.Div(
                            className="company-container",
                            children=[
                                html.Div(
                                    className="company-button",
                                    children=[
                                        html.Img(
                                            className="company-image",
                                            src=IMG_LOGO_GOOGLE,
                                        ),
                                        html.Div("Продолжить с Google", className="company-label"),
                                    ],
                                )
                            ],
                        ),
                        html.Div(
                            className="oauth-enter-label",
                            children=[
                                "Уже есть аккаунт?",
                                html.Div("Войти", className="oauth-enter-label-select"),
                            ],
                        ),
                    ],
                )
            ],
        ),
    ],
)


@callback(
    Output("auth-feedback", "children"),
    Output("auth-user", "data"),
    Input("auth-submit", "n_clicks"),
    State("auth-email", "value"),
    State("auth-password", "value"),
    prevent_initial_call=True,
)
def handle_login(n_clicks: int, email: str, password: str):
    if not n_clicks:
        return no_update, no_update
    if not email or not password:
        return "Введите email и пароль", no_update

    base_url = flask_request.host_url.rstrip("/")
    try:
        resp = requests.post(
            f"{base_url}/api/auth/login",
            json={"email": email.strip(), "password": password},
            timeout=5,
        )
    except Exception as exc:  # network/timeout etc.
        return f"Ошибка при обращении к серверу: {exc}", no_update

    if resp.status_code == 200:
        data = resp.json()
        return "Вход выполнен", data

    if resp.headers.get("Content-Type", "").startswith("application/json"):
        err = resp.json().get("error") or "Ошибка авторизации"
    else:
        err = f"Ошибка авторизации (код {resp.status_code})"
    return err, no_update
