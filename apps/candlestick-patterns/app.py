# -*- coding: utf-8 -*-

# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

import dash
import dash_table
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
from dash.dependencies import State, Input, Output
from flask_caching import Cache
import plotly.graph_objects as go

import os
import numpy as np
import pandas as pd
import json
import random
import yfinance as yf
import talib
from talib import abstract
from talib._ta_lib import (
    CandleSettingType,
    RangeType,
    _ta_set_candle_settings
)
import vectorbt as vbt
from vectorbt.utils.config import merge_kwargs
from vectorbt.portfolio.enums import InitCashMode, AccumulateExitMode

USE_CACHING = os.environ.get(
    "USE_CACHING",
    "True",
) == "True"
HOST = os.environ.get(
    "HOST",
    "127.0.0.1",
)
PORT = int(os.environ.get(
    "PORT",
    8050,
))
DEBUG = os.environ.get(
    "DEBUG",
    "True",
) == "True"
GITHUB_LINK = os.environ.get(
    "GITHUB_LINK",
    "https://github.com/polakowo/vectorbt/tree/master/apps/candlestick-patterns",
)

app = dash.Dash(
    __name__,
    meta_tags=[
        {
            "name": "viewport",
            "content": "width=device-width, initial-scale=1, maximum-scale=1.0, user-scalable=no",
        }
    ],
    external_stylesheets=[dbc.themes.GRID]
)
CACHE_CONFIG = {
    'CACHE_TYPE': 'filesystem' if USE_CACHING else 'null',
    'CACHE_DIR': 'data',
    'CACHE_DEFAULT_TIMEOUT': 0,
    'CACHE_THRESHOLD': 50
}
cache = Cache()
cache.init_app(app.server, config=CACHE_CONFIG)

# Settings
periods = ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max']
intervals = ['1m', '2m', '5m', '15m', '30m', '60m', '90m', '1d', '5d', '1wk', '1mo', '3mo']
patterns = talib.get_function_groups()['Pattern Recognition']
stats_table_columns = ["Metric", "Holding", "Random (Median)", "Strategy", "Z-Score"]

data_path = 'data/data.h5'
default_metric = 'Total Return [%]'
default_symbol = 'BTC-USD'
default_period = '1y'
default_interval = '1d'
default_date_range = [0, 1]
default_fees = 0.1
default_fixed_fees = 0.
default_slippage = 5.
default_yf_options = ['auto_adjust']
default_exit_n_random = default_entry_n_random = 5
default_entry_patterns = [
    'CDLHAMMER',
    'CDLINVERTEDHAMMER',
    'CDLPIERCING',
    'CDLMORNINGSTAR',
    'CDL3WHITESOLDIERS'
]
default_exit_options = []
default_exit_patterns = [
    'CDLHANGINGMAN',
    'CDLSHOOTINGSTAR',
    'CDLEVENINGSTAR',
    'CDL3BLACKCROWS',
    'CDLDARKCLOUDCOVER'
]
default_candle_settings = pd.DataFrame({
    'SettingType': [
        'BodyLong',
        'BodyVeryLong',
        'BodyShort',
        'BodyDoji',
        'ShadowLong',
        'ShadowVeryLong',
        'ShadowShort',
        'ShadowVeryShort',
        'Near',
        'Far',
        'Equal'
    ],
    'RangeType': [
        'RealBody',
        'RealBody',
        'RealBody',
        'HighLow',
        'RealBody',
        'RealBody',
        'Shadows',
        'HighLow',
        'HighLow',
        'HighLow',
        'HighLow'
    ],
    'AvgPeriod': [
        10,
        10,
        10,
        10,
        0,
        0,
        10,
        10,
        5,
        5,
        5
    ],
    'Factor': [
        1.0,
        3.0,
        1.0,
        0.1,
        1.0,
        2.0,
        1.0,
        0.1,
        0.2,
        0.6,
        0.05
    ]
})
default_sim_options = ['allow_inc_position', 'allow_dec_position']
default_n_random_strat = 50
default_stats_options = ['incl_unrealized']
default_layout = dict(
    height=200,
    autosize=True,
    automargin=True,
    margin=dict(b=40, t=20),
    font=dict(
        color="#9fa6b7"
    ),
    plot_bgcolor="#1f2536",
    paper_bgcolor="#1f2536",
    legend=dict(
        font=dict(size=10),
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ),
)

app.layout = html.Div(
    children=[
        html.Div(
            className="banner",
            children=[
                html.H6("Candlestick patterns"),
                html.Div(
                    html.A(
                        "View on GitHub",
                        href=GITHUB_LINK,
                        target="_blank",
                        className="button",
                    )
                ),
            ],
        ),
        dbc.Row(
            children=[
                dbc.Col(
                    lg=8, sm=12,
                    children=[
                        html.Div(
                            className="pretty-container",
                            children=[
                                html.Div(
                                    className="banner",
                                    children=[
                                        html.H6("OHLCV and indicators")
                                    ],
                                ),
                                dcc.Loading(
                                    id="ohlcv-loading",
                                    type="default",
                                    color="#387c9e",
                                    children=[
                                        dcc.Graph(
                                            id="ohlcv-graph",
                                            figure={
                                                "layout": default_layout
                                            }
                                        )
                                    ],
                                ),
                            ],
                        ),
                        dbc.Row(
                            children=[
                                dbc.Col(
                                    children=[
                                        html.Div(
                                            className="pretty-container",
                                            children=[
                                                html.Div(
                                                    className="banner",
                                                    children=[
                                                        html.H6("Orders, trades and value")
                                                    ],
                                                ),
                                                dcc.Loading(
                                                    id="value-loading",
                                                    type="default",
                                                    color="#387c9e",
                                                    children=[
                                                        dcc.Graph(
                                                            id="value-graph",
                                                            figure={
                                                                "layout": default_layout
                                                            }
                                                        )
                                                    ],
                                                ),
                                            ],
                                        ),

                                    ]
                                )
                            ]
                        ),
                        dbc.Row(
                            children=[
                                dbc.Col(
                                    children=[
                                        html.Div(
                                            className="pretty-container",
                                            children=[
                                                html.Div(
                                                    className="banner",
                                                    children=[
                                                        html.H6("Stats")
                                                    ],
                                                ),
                                                dcc.Loading(
                                                    id="stats-loading",
                                                    type="default",
                                                    color="#387c9e",
                                                    children=[
                                                        dash_table.DataTable(
                                                            id="stats-table",
                                                            columns=[
                                                                {
                                                                    "name": c,
                                                                    "id": c,
                                                                }
                                                                for c in stats_table_columns
                                                            ],
                                                            style_data_conditional=[{
                                                                "if": {"column_id": stats_table_columns[1]},
                                                                "fontWeight": "bold",
                                                                "borderLeft": "1px solid dimgrey"
                                                            }, {
                                                                "if": {"column_id": stats_table_columns[2]},
                                                                "fontWeight": "bold",
                                                            }, {
                                                                "if": {"column_id": stats_table_columns[3]},
                                                                "fontWeight": "bold",
                                                            }, {
                                                                "if": {"column_id": stats_table_columns[4]},
                                                                "fontWeight": "bold",
                                                            }, {
                                                                "if": {"state": "selected"},
                                                                "backgroundColor": "#171b26",
                                                                "color": "#88ccee",
                                                                "border": "1px solid #88ccee",
                                                            }, {
                                                                "if": {"state": "active"},
                                                                "backgroundColor": "#171b26",
                                                                "color": "#88ccee",
                                                                "border": "1px solid #88ccee",
                                                            }],
                                                            style_header={
                                                                "border": "none",
                                                                "backgroundColor": "#1f2536",
                                                                "fontWeight": "bold",
                                                                "padding": "0px 5px"
                                                            },
                                                            style_data={
                                                                "border": "none",
                                                                "backgroundColor": "#1f2536",
                                                                "color": "#7b7d8d",
                                                                "paddingRight": "10px"
                                                            },
                                                            style_table={
                                                                'overflowX': 'scroll',
                                                            },
                                                            style_as_list_view=False,
                                                            editable=False,
                                                        ),
                                                    ],
                                                ),
                                            ],
                                        ),
                                    ]
                                )
                            ]
                        ),
                        dbc.Row(
                            children=[
                                dbc.Col(
                                    children=[
                                        html.Div(
                                            className="pretty-container",
                                            children=[
                                                html.Div(
                                                    className="banner",
                                                    children=[
                                                        html.H6("Metric stats")
                                                    ],
                                                ),
                                                dcc.Loading(
                                                    id="metric-stats-loading",
                                                    type="default",
                                                    color="#387c9e",
                                                    children=[
                                                        html.Label("Metric:"),
                                                        dbc.Row(
                                                            children=[
                                                                dbc.Col(
                                                                    lg=4, sm=12,
                                                                    children=[
                                                                        dcc.Dropdown(
                                                                            id="metric-dropdown"
                                                                        ),
                                                                    ]
                                                                ),
                                                                dbc.Col(
                                                                    lg=8, sm=12,
                                                                    children=[
                                                                        dcc.Graph(
                                                                            id="metric-graph",
                                                                            figure={
                                                                                "layout": default_layout
                                                                            }
                                                                        )
                                                                    ]
                                                                )
                                                            ],
                                                        ),
                                                    ],
                                                ),
                                            ],
                                        ),
                                    ]
                                )
                            ]
                        ),

                    ]
                ),
                dbc.Col(
                    lg=4, sm=12,
                    children=[
                        html.Div(
                            className="pretty-container",
                            children=[
                                html.Div(
                                    className="banner",
                                    children=[
                                        html.H6("Settings")
                                    ],
                                ),
                                html.Button(
                                    "Reset",
                                    id="reset-button"
                                ),
                                html.Details(
                                    open=True,
                                    children=[
                                        html.Summary(
                                            className="section-title",
                                            children="Data",
                                        ),
                                        html.Label("Yahoo! Finance symbol:"),
                                        dcc.Input(
                                            id="symbol-input",
                                            className="input-control",
                                            type="text",
                                            value=default_symbol,
                                            placeholder="Enter symbol...",
                                            debounce=True
                                        ),
                                        dbc.Row(
                                            children=[
                                                dbc.Col(
                                                    children=[
                                                        html.Label("Period:"),
                                                        dcc.Dropdown(
                                                            id="period-dropdown",
                                                            options=[{"label": i, "value": i} for i in periods],
                                                            value=default_period,
                                                        ),
                                                    ]
                                                ),
                                                dbc.Col(
                                                    children=[
                                                        html.Label("Interval:"),
                                                        dcc.Dropdown(
                                                            id="interval-dropdown",
                                                            options=[{"label": i, "value": i} for i in intervals],
                                                            value=default_interval,
                                                        ),
                                                    ]
                                                ),
                                                dbc.Col()
                                            ],
                                        ),
                                        html.Label("Filter by date (or select range in chart):"),
                                        dcc.RangeSlider(
                                            id="date-slider",
                                            min=0.,
                                            max=1.,
                                            value=default_date_range,
                                            allowCross=False
                                        ),
                                        dcc.Checklist(
                                            id="yf-checklist",
                                            options=[{
                                                "label": "Adjust all OHLC automatically",
                                                "value": "auto_adjust"
                                            }, {
                                                "label": "Use back-adjusted data to mimic true historical prices",
                                                "value": "back_adjust"
                                            }],
                                            value=default_yf_options,
                                            style={
                                                "color": "#7b7d8d"
                                            }
                                        ),
                                    ],
                                ),
                                html.Details(
                                    open=True,
                                    children=[
                                        html.Summary(
                                            className="section-title",
                                            children="Entry patterns",
                                        ),
                                        html.Div(
                                            id='entry-settings',
                                            children=[
                                                html.Button(
                                                    "All",
                                                    id="entry-all-button"
                                                ),
                                                html.Button(
                                                    "Random",
                                                    id="entry-random-button"
                                                ),
                                                html.Button(
                                                    "Clear",
                                                    id="entry-clear-button"
                                                ),
                                                html.Label("Number of random patterns:"),
                                                dbc.Row(
                                                    children=[
                                                        dbc.Col(
                                                            children=[
                                                                dcc.Input(
                                                                    id="entry-n-random-input",
                                                                    className="input-control",
                                                                    value=default_entry_n_random,
                                                                    placeholder="Enter number...",
                                                                    debounce=True,
                                                                    type="number",
                                                                    min=1, max=len(patterns), step=1
                                                                ),
                                                            ]
                                                        ),
                                                        dbc.Col(),
                                                        dbc.Col()
                                                    ],
                                                ),
                                                html.Label("Select patterns:"),
                                                dcc.Dropdown(
                                                    id="entry-pattern-dropdown",
                                                    options=[{"label": i, "value": i} for i in patterns],
                                                    multi=True,
                                                    value=default_entry_patterns,
                                                ),

                                            ],
                                        ),
                                    ],
                                ),
                                html.Details(
                                    open=True,
                                    children=[
                                        html.Summary(
                                            className="section-title",
                                            children="Exit patterns",
                                        ),
                                        dcc.Checklist(
                                            id="exit-checklist",
                                            options=[{
                                                "label": "Same as entry patterns",
                                                "value": "same_as_entry"
                                            }],
                                            value=default_exit_options,
                                            style={
                                                "color": "#7b7d8d"
                                            }
                                        ),
                                        html.Div(
                                            id='exit-settings',
                                            hidden="same_as_entry" in default_exit_options,
                                            children=[
                                                html.Button(
                                                    "All",
                                                    id="exit-all-button"
                                                ),
                                                html.Button(
                                                    "Random",
                                                    id="exit-random-button"
                                                ),
                                                html.Button(
                                                    "Clear",
                                                    id="exit-clear-button"
                                                ),
                                                html.Label("Number of random patterns:"),
                                                dbc.Row(
                                                    children=[
                                                        dbc.Col(
                                                            children=[
                                                                dcc.Input(
                                                                    id="exit-n-random-input",
                                                                    className="input-control",
                                                                    value=default_exit_n_random,
                                                                    placeholder="Enter number...",
                                                                    debounce=True,
                                                                    type="number",
                                                                    min=1, max=len(patterns), step=1
                                                                ),
                                                            ]
                                                        ),
                                                        dbc.Col(),
                                                        dbc.Col()
                                                    ],
                                                ),
                                                html.Label("Select patterns:"),
                                                dcc.Dropdown(
                                                    id="exit-pattern-dropdown",
                                                    options=[{"label": i, "value": i} for i in patterns],
                                                    multi=True,
                                                    value=default_exit_patterns,
                                                ),

                                            ],
                                        ),
                                    ],
                                ),
                                html.Details(
                                    children=[
                                        html.Summary(
                                            className="section-title",
                                            children="Candle settings",
                                        ),
                                        dash_table.DataTable(
                                            id="candle-settings-table",
                                            columns=[
                                                {
                                                    "name": c,
                                                    "id": c,
                                                    "editable": i in (2, 3),
                                                    "type": "numeric" if i in (2, 3) else "any"
                                                }
                                                for i, c in enumerate(default_candle_settings.columns)
                                            ],
                                            data=default_candle_settings.to_dict("records"),
                                            style_data_conditional=[{
                                                "if": {"column_editable": True},
                                                "backgroundColor": "#171b26",
                                                "border": "1px solid dimgrey"
                                            }, {
                                                "if": {"state": "selected"},
                                                "backgroundColor": "#171b26",
                                                "color": "#88ccee",
                                                "border": "1px solid #88ccee",
                                            }, {
                                                "if": {"state": "active"},
                                                "backgroundColor": "#171b26",
                                                "color": "#88ccee",
                                                "border": "1px solid #88ccee",
                                            }],
                                            style_header={
                                                "border": "none",
                                                "backgroundColor": "#1f2536",
                                                "fontWeight": "bold",
                                                "padding": "0px 5px"
                                            },
                                            style_data={
                                                "border": "none",
                                                "backgroundColor": "#1f2536",
                                                "color": "#7b7d8d"
                                            },
                                            style_table={
                                                'overflowX': 'scroll',
                                            },
                                            style_as_list_view=False,
                                            editable=True,
                                        ),
                                    ],
                                ),
                                html.Details(
                                    open=True,
                                    children=[
                                        html.Summary(
                                            className="section-title",
                                            children="Simulation settings",
                                        ),
                                        dbc.Row(
                                            children=[
                                                dbc.Col(
                                                    children=[
                                                        html.Label("Fees (in %):"),
                                                    ]
                                                ),
                                                dbc.Col(
                                                    children=[
                                                        html.Label("Fixed fees:"),
                                                    ]
                                                ),
                                                dbc.Col(
                                                    children=[
                                                        html.Label("Slippage (in % of H-O):"),
                                                    ]

                                                )
                                            ],
                                        ),
                                        dbc.Row(
                                            children=[
                                                dbc.Col(
                                                    children=[
                                                        dcc.Input(
                                                            id="fees-input",
                                                            className="input-control",
                                                            type="number",
                                                            value=default_fees,
                                                            placeholder="Enter fees...",
                                                            debounce=True,
                                                            min=0, max=100
                                                        ),
                                                    ]
                                                ),
                                                dbc.Col(
                                                    children=[
                                                        dcc.Input(
                                                            id="fixed-fees-input",
                                                            className="input-control",
                                                            type="number",
                                                            value=default_fixed_fees,
                                                            placeholder="Enter fixed fees...",
                                                            debounce=True,
                                                            min=0
                                                        ),
                                                    ]
                                                ),
                                                dbc.Col(
                                                    children=[
                                                        dcc.Input(
                                                            id="slippage-input",
                                                            className="input-control",
                                                            type="number",
                                                            value=default_slippage,
                                                            placeholder="Enter slippage...",
                                                            debounce=True,
                                                            min=0, max=100
                                                        ),
                                                    ]

                                                )
                                            ],
                                        ),
                                        dcc.Checklist(
                                            id="sim-checklist",
                                            options=[{
                                                "label": "Allow increasing position",
                                                "value": "allow_inc_position"
                                            }, {
                                                "label": "Allow decreasing position",
                                                "value": "allow_dec_position"
                                            }],
                                            value=default_sim_options,
                                            style={
                                                "color": "#7b7d8d"
                                            }
                                        ),
                                        html.Label("Number of random strategies to test against:"),
                                        dbc.Row(
                                            children=[
                                                dbc.Col(
                                                    children=[
                                                        dcc.Input(
                                                            id="n-random-strat-input",
                                                            className="input-control",
                                                            value=default_n_random_strat,
                                                            placeholder="Enter number...",
                                                            debounce=True,
                                                            type="number",
                                                            min=10, max=1000, step=1
                                                        ),
                                                    ]
                                                ),
                                                dbc.Col(),
                                                dbc.Col()
                                            ],
                                        ),
                                        dcc.Checklist(
                                            id="stats-checklist",
                                            options=[{
                                                "label": "Include unrealized P&L in stats",
                                                "value": "incl_unrealized"
                                            }],
                                            value=default_stats_options,
                                            style={
                                                "color": "#7b7d8d"
                                            }
                                        ),
                                    ],
                                ),
                            ],
                        ),
                    ]
                )
            ]
        ),
        html.Div(id='data-signal', style={'display': 'none'}),
        html.Div(id='candle-settings-signal', style={'display': 'none'}),
        html.Div(id='stats-signal', style={'display': 'none'})
    ],
)


@cache.memoize()
def global_data_store(symbol, period, interval, auto_adjust, back_adjust):
    # perform expensive computations in this "global store"
    # these computations are cached in a globally available
    # redis memory store which is available across processes
    # and for all time.

    return yf.Ticker(symbol).history(
        period=period,
        interval=interval,
        actions=False,
        auto_adjust=auto_adjust,
        back_adjust=back_adjust
    )


@app.callback(
    Output(component_id='data-signal', component_property='children'),
    [Input(component_id='symbol-input', component_property='value'),
     Input(component_id='period-dropdown', component_property='value'),
     Input(component_id='interval-dropdown', component_property='value'),
     Input(component_id="yf-checklist", component_property="value")]
)
def update_data(symbol, period, interval, yf_options):
    # compute value and send a signal when done
    auto_adjust = 'auto_adjust' in yf_options
    back_adjust = 'back_adjust' in yf_options
    global_data_store(symbol, period, interval, auto_adjust, back_adjust)
    return symbol, period, interval, auto_adjust, back_adjust


@app.callback(
    Output(component_id='candle-settings-signal', component_property='children'),
    [Input(component_id='candle-settings-table', component_property='data')]
)
def set_candle_settings(data):
    for d in data:
        AvgPeriod = d["AvgPeriod"]
        if isinstance(AvgPeriod, float) and float.is_integer(AvgPeriod):
            AvgPeriod = int(AvgPeriod)
        Factor = float(d["Factor"])
        _ta_set_candle_settings(
            getattr(CandleSettingType, d["SettingType"]),
            getattr(RangeType, d["RangeType"]),
            AvgPeriod,
            Factor
        )


@app.callback(
    [Output(component_id='date-slider', component_property='min'),
     Output(component_id='date-slider', component_property='max'),
     Output(component_id='date-slider', component_property='value')],
    [Input(component_id='data-signal', component_property='children')]
)
def update_date_range(inputs):
    df = global_data_store(*inputs)
    return 0, len(df.index), [0, len(df.index)]


@app.callback(
    Output(component_id='entry-pattern-dropdown', component_property='value'),
    [Input(component_id="entry-all-button", component_property="n_clicks"),
     Input(component_id="entry-random-button", component_property="n_clicks"),
     Input(component_id="entry-clear-button", component_property="n_clicks"),
     Input(component_id="reset-button", component_property="n_clicks")],
    [State(component_id="entry-n-random-input", component_property="value")]
)
def select_all_entry_patterns(_1, _2, _3, _4, n_random):
    ctx = dash.callback_context
    if ctx.triggered:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        if button_id == 'entry-all-button':
            return patterns
        elif button_id == 'entry-random-button':
            return random.sample(patterns, n_random)
        elif button_id == 'entry-clear-button':
            return []
        elif button_id == 'reset-button':
            return default_entry_patterns
    return dash.no_update


@app.callback(
    [Output(component_id="exit-settings", component_property="hidden"),
     Output(component_id="exit-n-random-input", component_property="value"),
     Output(component_id='exit-pattern-dropdown', component_property='value')],
    [Input(component_id="exit-checklist", component_property="value"),
     Input(component_id="exit-all-button", component_property="n_clicks"),
     Input(component_id="exit-random-button", component_property="n_clicks"),
     Input(component_id="exit-clear-button", component_property="n_clicks"),
     Input(component_id="reset-button", component_property="n_clicks"),
     Input(component_id="entry-n-random-input", component_property="value"),
     Input(component_id='entry-pattern-dropdown', component_property='value')],
    [State(component_id="exit-n-random-input", component_property="value")]
)
def select_all_exit_patterns(exit_options, _1, _2, _3, _4, entry_n_random, entry_patterns, exit_n_random):
    ctx = dash.callback_context
    same_as_entry = 'same_as_entry' in exit_options
    if ctx.triggered:
        control_id = ctx.triggered[0]['prop_id'].split('.')[0]
        if control_id == 'exit-checklist':
            return same_as_entry, entry_n_random, entry_patterns
        elif control_id == 'exit-all-button':
            return same_as_entry, exit_n_random, patterns
        elif control_id == 'exit-random-button':
            return same_as_entry, exit_n_random, random.sample(patterns, exit_n_random)
        elif control_id == 'exit-clear-button':
            return same_as_entry, exit_n_random, []
        elif control_id == 'reset-button':
            default_same_as_entry = 'same_as_entry' in default_exit_options
            return default_same_as_entry, default_exit_n_random, default_exit_patterns
        elif control_id in ('entry-n-random-input', 'entry-pattern-dropdown'):
            if same_as_entry:
                return same_as_entry, entry_n_random, entry_patterns
    return dash.no_update


@app.callback(
    Output(component_id='ohlcv-graph', component_property='figure'),
    [Input(component_id='data-signal', component_property='children'),
     Input(component_id='date-slider', component_property='value'),
     Input(component_id='entry-pattern-dropdown', component_property='value'),
     Input(component_id='exit-pattern-dropdown', component_property='value'),
     Input(component_id='candle-settings-signal', component_property='children')]
)
def update_ohlcv(inputs, date_range, entry_patterns, exit_patterns, _):
    # Get data
    df = global_data_store(*inputs)

    # Filter by date
    df = df.iloc[date_range[0]:date_range[1]]

    # Run pattern recognition indicators and combine results
    talib_inputs = {
        'open': df['Open'].values,
        'high': df['High'].values,
        'low': df['Low'].values,
        'close': df['Close'].values,
        'volume': df['Volume'].values
    }
    all_patterns = set(entry_patterns + exit_patterns)
    signal_df = pd.DataFrame.vbt.empty(
        (len(df.index), len(all_patterns)),
        fill_value=0.,
        index=df.index,
        columns=all_patterns
    )
    for pattern in all_patterns:
        signal_df[pattern] = abstract.Function(pattern)(talib_inputs)
    entry_signal_df = signal_df[entry_patterns]
    exit_signal_df = signal_df[exit_patterns]

    # Entry patterns
    entry_df = entry_signal_df[(entry_signal_df > 0).any(axis=1)]
    entry_patterns = []
    for row_i, row in entry_df.iterrows():
        entry_patterns.append('<br>'.join(row.index[row != 0]))
    entry_patterns = np.asarray(entry_patterns)

    # Exit patterns
    exit_df = exit_signal_df[(exit_signal_df < 0).any(axis=1)]
    exit_patterns = []
    for row_i, row in exit_df.iterrows():
        exit_patterns.append('<br>'.join(row.index[row != 0]))
    exit_patterns = np.asarray(exit_patterns)

    # Prepare scatter data
    highest_high = df['High'].max()
    lowest_low = df['Low'].min()
    distance = (highest_high - lowest_low) / 5
    entry_y = df.loc[entry_df.index, 'Low'] - distance
    exit_y = df.loc[exit_df.index, 'High'] + distance

    # Color volume
    close_open_diff = df['Close'].values - df['Open'].values
    volume_color = np.empty(df['Volume'].shape, dtype=np.object)
    volume_color[close_open_diff > 0] = '#0b623e'
    volume_color[close_open_diff == 0] = 'gray'
    volume_color[close_open_diff < 0] = '#bb6704'

    # Build graph
    return dict(
        data=[
            go.Ohlc(
                x=pd.to_datetime(df.index),
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                name='OHLC',
                yaxis="y2",
                xaxis="x",
                increasing_line_color='#0f8554',
                decreasing_line_color='#e17c05'
            ),
            go.Scatter(
                x=pd.to_datetime(entry_y.index),
                y=entry_y,
                customdata=entry_patterns[:, None],
                hovertemplate='%{x}<br>%{customdata[0]}',
                mode='markers',
                marker_size=8,
                marker_symbol='triangle-up',
                marker_color='#38a6a5',
                name='Bullish signal',
                yaxis="y2",
                xaxis="x",
            ),
            go.Scatter(
                x=pd.to_datetime(exit_y.index),
                y=exit_y,
                customdata=exit_patterns[:, None],
                hovertemplate='%{x}<br>%{customdata[0]}',
                mode='markers',
                marker_size=8,
                marker_symbol='triangle-down',
                marker_color='#cc503e',
                name='Bearish signal',
                yaxis="y2",
                xaxis="x",
            ),
            go.Bar(
                x=pd.to_datetime(df.index),
                y=df['Volume'],
                marker_color=volume_color,
                marker_line_width=0,
                name='Volume',
                yaxis="y",
                xaxis="x"
            )
        ],
        layout=merge_kwargs(
            default_layout,
            dict(
                height=400,
                margin=dict(r=40),
                hovermode="closest",
                xaxis=dict(
                    gridcolor='#323b56',
                    rangeslider=dict(
                        visible=False
                    ),
                    spikemode='across+marker',
                    title='Date'
                ),
                yaxis=dict(
                    gridcolor='#323b56',
                    domain=[0, 0.3],
                    spikemode='across+marker',
                    title='Volume'
                ),
                yaxis2=dict(
                    gridcolor='#323b56',
                    domain=[0.4, 1],
                    spikemode='across+marker',
                    title='Price',
                ),
                bargap=0
            )
        )
    )


def simulate_portfolio(inputs, date_range, entry_patterns, exit_patterns, fees,
                       fixed_fees, slippage, sim_options, n_random_strat):
    # Get data
    df = global_data_store(*inputs)

    # Filter by date
    df = df.iloc[date_range[0]:date_range[1]]

    # Run pattern recognition indicators and combine results
    talib_inputs = {
        'open': df['Open'].values,
        'high': df['High'].values,
        'low': df['Low'].values,
        'close': df['Close'].values,
        'volume': df['Volume'].values
    }
    all_patterns = list(set(entry_patterns + exit_patterns))
    entry_i = [all_patterns.index(p) for p in entry_patterns]
    exit_i = [all_patterns.index(p) for p in exit_patterns]
    signals = np.empty((len(df.index), len(all_patterns)), dtype=np.float_)
    for i, pattern in enumerate(all_patterns):
        # TA-Lib functions have output in increments of 100
        signals[:, i] = abstract.Function(pattern)(talib_inputs) / 100

    # Generate size for main
    def _generate_size(signals):
        entry_signals = signals[:, entry_i]
        exit_signals = signals[:, exit_i]
        return np.where(entry_signals > 0, entry_signals, 0).sum(axis=1) + \
               np.where(exit_signals < 0, exit_signals, 0).sum(axis=1)

    main_size = np.empty((len(df.index),), dtype=np.float_)
    main_size[0] = 0  # avoid looking into future
    main_size[1:] = _generate_size(signals)[:-1]

    # Generate size for holding
    hold_size = np.full_like(main_size, 0.)
    hold_size[0] = np.inf

    # Generate size for random
    def _shuffle_along_axis(a, axis):
        idx = np.random.rand(*a.shape).argsort(axis=axis)
        return np.take_along_axis(a, idx, axis=axis)

    rand_size = np.empty((len(df.index), n_random_strat), dtype=np.float_)
    rand_size[0] = 0  # avoid looking into future
    for i in range(n_random_strat):
        rand_signals = _shuffle_along_axis(signals, 0)
        rand_size[1:, i] = _generate_size(rand_signals)[:-1]

    # Simulate portfolio
    def _simulate_portfolio(size, init_cash):
        accumulate = 'allow_inc_position' in sim_options
        accumulate_exit_mode = AccumulateExitMode.Reduce \
            if 'allow_dec_position' in sim_options else AccumulateExitMode.Close
        return vbt.Portfolio.from_signals(
            close=df['Close'],
            entries=size > 0,
            exits=size < 0,
            entry_price=df['Open'],
            exit_price=df['Open'],
            size=np.abs(size),
            accumulate=accumulate,
            accumulate_exit_mode=accumulate_exit_mode,
            init_cash=init_cash,
            fees=float(fees) / 100,
            fixed_fees=float(fixed_fees),
            slippage=(float(slippage) / 100) * (df['High'] / df['Open'] - 1),
            freq=inputs[2]
        )

    # Align initial cash across main and random strategies
    aligned_portfolio = _simulate_portfolio(np.hstack((main_size[:, None], rand_size)), InitCashMode.AutoAlign)
    # Fixate initial cash for indexing
    aligned_portfolio = aligned_portfolio.copy(
        init_cash=aligned_portfolio.init_cash,
        init_cash_mode=None
    )
    # Separate portfolios
    main_portfolio = aligned_portfolio.iloc[0]
    rand_portfolio = aligned_portfolio.iloc[1:]

    # Simulate holding portfolio
    hold_portfolio = _simulate_portfolio(hold_size, main_portfolio.init_cash)

    return main_portfolio, hold_portfolio, rand_portfolio


@app.callback(
    [Output(component_id='value-graph', component_property='figure'),
     Output(component_id='stats-table', component_property='data'),
     Output(component_id='stats-signal', component_property='children'),
     Output(component_id='metric-dropdown', component_property='options'),
     Output(component_id='metric-dropdown', component_property='value')],
    [Input(component_id='data-signal', component_property='children'),
     Input(component_id='date-slider', component_property='value'),
     Input(component_id='entry-pattern-dropdown', component_property='value'),
     Input(component_id='exit-pattern-dropdown', component_property='value'),
     Input(component_id='candle-settings-signal', component_property='children'),
     Input(component_id='fees-input', component_property='value'),
     Input(component_id='fixed-fees-input', component_property='value'),
     Input(component_id='slippage-input', component_property='value'),
     Input(component_id='sim-checklist', component_property='value'),
     Input(component_id='n-random-strat-input', component_property='value'),
     Input(component_id='stats-checklist', component_property='value'),
     Input(component_id="reset-button", component_property="n_clicks")],
    [State(component_id='metric-dropdown', component_property='value')]
)
def update_stats(inputs, date_range, entry_patterns, exit_patterns, _1, fees, fixed_fees,
                 slippage, sim_options, n_random_strat, stats_options, _2, curr_metric):
    # Simulate portfolio
    main_portfolio, hold_portfolio, rand_portfolio = simulate_portfolio(
        inputs, date_range, entry_patterns, exit_patterns,
        fees, fixed_fees, slippage, sim_options, n_random_strat)

    # Get orders
    buy_trace, sell_trace = main_portfolio.orders.plot().data[1:]
    buy_trace.update(dict(
        x=pd.to_datetime(buy_trace.x),
        marker_line=None,
        marker_size=8,
        marker_symbol='triangle-up',
        marker_color='#38a6a5',
        yaxis='y4'
    ))
    sell_trace.update(dict(
        x=pd.to_datetime(sell_trace.x),
        marker_line=None,
        marker_size=8,
        marker_symbol='triangle-down',
        marker_color='#cc503e',
        yaxis='y4'
    ))

    # Get returns
    incl_unrealized = 'incl_unrealized' in stats_options
    returns = main_portfolio.trades(incl_unrealized=incl_unrealized).returns
    profit_mask = returns.mapped_arr > 0
    loss_mask = returns.mapped_arr < 0

    figure = dict(
        data=[
            go.Scatter(
                x=pd.to_datetime(main_portfolio.wrapper.index),
                y=main_portfolio.shares(),
                name="Holdings",
                yaxis="y2",
                line_color='#1f77b4'
            ),
            go.Scatter(
                x=pd.to_datetime(main_portfolio.wrapper.index),
                y=main_portfolio.value(),
                name="Value",
                line_color='#2ca02c'
            ),
            go.Scatter(
                x=pd.to_datetime(hold_portfolio.wrapper.index),
                y=hold_portfolio.value(),
                name=f"Value (Holding)",
                line_color='#ff7f0e'
            ),
            go.Scatter(
                x=pd.to_datetime(main_portfolio.wrapper.index[returns.idx_arr[profit_mask]]),
                y=returns.mapped_arr[profit_mask],
                marker_color='#2ca02c',
                marker_size=8,
                mode='markers',
                name="Profit",
                yaxis="y3",
            ),
            go.Scatter(
                x=pd.to_datetime(main_portfolio.wrapper.index[returns.idx_arr[loss_mask]]),
                y=returns.mapped_arr[loss_mask],
                marker_color='#d62728',
                marker_size=8,
                mode='markers',
                name="Loss",
                yaxis="y3"
            ),
            buy_trace,
            sell_trace
        ],
        layout=merge_kwargs(
            default_layout,
            dict(
                height=500,
                hovermode="closest",
                xaxis=dict(
                    gridcolor='#323b56',
                    title='Date',
                ),
                yaxis=dict(
                    gridcolor='#323b56',
                    title='Value',
                    domain=[0, 0.4]
                ),
                yaxis2=dict(
                    showgrid=False,
                    overlaying="y",
                    side="right",
                    title='Holdings',
                ),
                yaxis3=dict(
                    gridcolor='#323b56',
                    title='Trade return',
                    domain=[0.45, 0.7],
                    tickformat='%'
                ),
                yaxis4=dict(
                    gridcolor='#323b56',
                    title='Order price',
                    domain=[0.75, 1],
                )
            )
        )
    )

    def _chop_microseconds(delta):
        return delta - pd.Timedelta(microseconds=delta.microseconds, nanoseconds=delta.nanoseconds)

    def _metric_to_str(x):
        if isinstance(x, float):
            return '%.2f' % x
        if isinstance(x, pd.Timedelta):
            return str(_chop_microseconds(x))
        return str(x)

    main_stats = main_portfolio.stats(incl_unrealized=incl_unrealized)
    hold_stats = hold_portfolio.stats(incl_unrealized=True)
    rand_stats = rand_portfolio.stats(incl_unrealized=incl_unrealized, agg_func=None)
    rand_stats_median = rand_stats.iloc[:, 3:].median(axis=0)
    rand_stats_mean = rand_stats.iloc[:, 3:].mean(axis=0)
    rand_stats_std = rand_stats.iloc[:, 3:].std(axis=0, ddof=0)
    stats_mean_diff = main_stats.iloc[3:] - rand_stats_mean

    def _to_float(x):
        if pd.isnull(x):
            return np.nan
        if isinstance(x, float):
            if np.allclose(x, 0):
                return 0.
        if isinstance(x, pd.Timedelta):
            return float(x.total_seconds())
        return float(x)

    z = stats_mean_diff.apply(_to_float) / rand_stats_std.apply(_to_float)

    table_data = pd.DataFrame(columns=stats_table_columns)
    table_data.iloc[:, 0] = main_stats.index
    table_data.iloc[:, 1] = hold_stats.apply(_metric_to_str).values
    table_data.iloc[:3, 2] = table_data.iloc[:3, 1]
    table_data.iloc[3:, 2] = rand_stats_median.apply(_metric_to_str).values
    table_data.iloc[:, 3] = main_stats.apply(_metric_to_str).values
    table_data.iloc[3:, 4] = z.apply(_metric_to_str).values

    metric = curr_metric
    ctx = dash.callback_context
    if ctx.triggered:
        control_id = ctx.triggered[0]['prop_id'].split('.')[0]
        if control_id == 'reset-button':
            metric = default_metric
    if metric is None:
        metric = default_metric
    return figure, \
           table_data.to_dict("records"), \
           json.dumps({
               'main': {m: [_to_float(main_stats[m])] for m in main_stats.index[3:]},
               'hold': {m: [_to_float(hold_stats[m])] for m in main_stats.index[3:]},
               'rand': {m: rand_stats[m].apply(_to_float).values.tolist() for m in main_stats.index[3:]}
           }), \
           [{"label": i, "value": i} for i in main_stats.index[3:]], \
           metric


@app.callback(
    Output(component_id='metric-graph', component_property='figure'),
    [Input(component_id='stats-signal', component_property='children'),
     Input(component_id='metric-dropdown', component_property='value')]
)
def update_metric_stats(stats_json, metric):
    stats_dict = json.loads(stats_json)
    return dict(
        data=[
            go.Box(
                x=stats_dict['rand'][metric],
                quartilemethod="linear",
                jitter=0.3,
                pointpos=1.8,
                boxpoints='all',
                boxmean='sd',
                hoveron="points",
                hovertemplate='%{x}<br>Random',
                name='',
                marker=dict(
                    color="#1f77b4",
                    opacity=0.5,
                    size=8,
                ),
            ),
            go.Box(
                x=stats_dict['hold'][metric],
                quartilemethod="linear",
                boxpoints="all",
                jitter=0,
                pointpos=1.8,
                hoveron="points",
                hovertemplate='%{x}<br>Holding',
                fillcolor="rgba(0,0,0,0)",
                line=dict(color="rgba(0,0,0,0)"),
                name='',
                marker=dict(
                    color="#ff7f0e",
                    size=8,
                ),
            ),
            go.Box(
                x=stats_dict['main'][metric],
                quartilemethod="linear",
                boxpoints="all",
                jitter=0,
                pointpos=1.8,
                hoveron="points",
                hovertemplate='%{x}<br>Strategy',
                fillcolor="rgba(0,0,0,0)",
                line=dict(color="rgba(0,0,0,0)"),
                name='',
                marker=dict(
                    color="#2ca02c",
                    size=8,
                ),
            ),
        ],
        layout=merge_kwargs(
            default_layout,
            dict(
                showlegend=False,
                height=300,
                margin=dict(l=60, r=20, t=40, b=20),
                hovermode="closest",
                xaxis=dict(
                    gridcolor='#323b56',
                    title=metric,
                    side='top'
                ),
                yaxis=dict(
                    gridcolor='#323b56'
                ),
            )
        )
    )


@app.callback(
    [Output(component_id='symbol-input', component_property='value'),
     Output(component_id='period-dropdown', component_property='value'),
     Output(component_id='interval-dropdown', component_property='value'),
     Output(component_id="yf-checklist", component_property="value"),
     Output(component_id="entry-n-random-input", component_property="value"),
     Output(component_id="exit-checklist", component_property="value"),
     Output(component_id='candle-settings-table', component_property='data'),
     Output(component_id='fees-input', component_property='value'),
     Output(component_id='fixed-fees-input', component_property='value'),
     Output(component_id='slippage-input', component_property='value'),
     Output(component_id='sim-checklist', component_property='value'),
     Output(component_id='n-random-strat-input', component_property='value'),
     Output(component_id='stats-checklist', component_property='value')],
    [Input(component_id="reset-button", component_property="n_clicks")],
    prevent_initial_call=True
)
def reset_settings(_):
    return default_symbol, \
           default_period, \
           default_interval, \
           default_yf_options, \
           default_entry_n_random, \
           default_exit_options, \
           default_candle_settings.to_dict("records"), \
           default_fees, \
           default_fixed_fees, \
           default_slippage, \
           default_sim_options, \
           default_n_random_strat, \
           default_stats_options


if __name__ == '__main__':
    app.run_server(host=HOST, port=PORT, debug=DEBUG)
