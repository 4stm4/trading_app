    # dcc.Checklist(
    #     id='toggle-rangeslider',
    #     options=[{'label': 'Include Rangeslider', 
    #               'value': 'slider'}],
    #     value=['slider']
    # ),
    # dcc.Graph(id="graph", style={'width': '90vh', 'height': '60vh'}),
])


# @app.callback(
#     Output("graph", "figure"), 
#     Input("toggle-rangeslider", "value"))
# def display_candlestick(value):
#     df = pd.DataFrame.from_dict(klines.to_list())
#     fig = go.Figure(go.Candlestick(
#         x=df['Date'],
#         open=df['Open'],
#         high=df['High'],
#         low=df['Low'],
#         close=df['Close']
#     ))

#     fig.update_layout(
#         xaxis_rangeslider_visible='slider' in value
#     )

#     return fig
