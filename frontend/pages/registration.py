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
    path="/register",
    title="{0}| CryptoManager".format(signup[language]["Registration"]),
    name="Our Analytics Dashboard",
)

layout = dbc.Container(
    class_name="content-layout",
    children=[
        dcc.Store(id="register-user"),
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
                                html.Div(signup[language]["welcome"], className="card-page-title"),
                            ],
                        ),
                        # форма для входа через почту или сотовый
                        html.Form(
                            children=[
                                html.Div(
                                    children=[
                                        html.Div(
                                            signup[language]["email_phone"],
                                            className="bn-formItem-label",
                                        ),
                                        html.Div(
                                            children=[
                                                html.Div(
                                                    children=[
                                                        dbc.Input(
                                                            class_name="bn-textField-input",
                                                            id="register-email",
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
                                        html.Div(
                                            signup[language]["Password"]
                                            if "Password" in signup[language]
                                            else "Пароль",
                                            className="bn-formItem-label",
                                        ),
                                        html.Div(
                                            children=[
                                                html.Div(
                                                    children=[
                                                        dbc.Input(
                                                            class_name="bn-textField-input",
                                                            id="register-password",
                                                            name="password",
                                                            type="password",
                                                            autoComplete="new-password",
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
                                    signup[language]["further"],
                                    className="next-button",
                                    id="register-submit",
                                    n_clicks=0,
                                ),
                                html.Div(id="register-feedback", className="auth-feedback"),
                            ]
                        ),
                        html.Div(
                            className="oauth-container",
                            children=[
                                html.Div(className="oauth-lines"),
                                html.Div(signup[language]["or"], className="oauth-label"),
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
                                        html.Div(
                                            signup[language]["Continue with Google"],
                                            className="company-label",
                                        ),
                                    ],
                                )
                            ],
                        ),
                        html.Div(
                            className="oauth-enter-label",
                            children=[
                                signup[language]["Already have an account?"],
                                html.Div(
                                    signup[language]["Login"], className="oauth-enter-label-select"
                                ),
                            ],
                        ),
                    ],
                )
            ],
        ),
    ],
)


@callback(
    Output("register-feedback", "children"),
    Output("register-user", "data"),
    Input("register-submit", "n_clicks"),
    State("register-email", "value"),
    State("register-password", "value"),
    prevent_initial_call=True,
)
def handle_register(n_clicks: int, email: str, password: str):
    if not n_clicks:
        return no_update, no_update
    if not email or not password:
        return signup[language].get("email_phone", "Введите email") + " и пароль", no_update

    base_url = flask_request.host_url.rstrip("/")
    try:
        resp = requests.post(
            f"{base_url}/api/auth/register",
            json={"email": email.strip(), "password": password},
            timeout=5,
        )
    except Exception as exc:
        return f"Ошибка при обращении к серверу: {exc}", no_update

    if resp.status_code in (200, 201):
        data = resp.json()
        return "Регистрация выполнена", data

    if resp.headers.get("Content-Type", "").startswith("application/json"):
        err = resp.json().get("error") or "Ошибка регистрации"
    else:
        err = f"Ошибка регистрации (код {resp.status_code})"
    return err, no_update
