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
from vectorbt.enums import InitCashMode, AccumulationMode
from vectorbt.utils.config import merge_kwargs

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
stats_table_columns = ["Metric", "Buy & Hold", "Random (Median)", "Strategy", "Z-Score"]

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
default_prob_options = ['mimic_strategy']
default_entry_prob = 0.1
default_exit_prob = 0.1
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
default_entry_dates = []
default_exit_dates = []
default_sim_options = ['allow_inc_position', 'allow_dec_position']
default_n_random_strat = 50
default_stats_options = ['incl_unrealized']
default_layout = dict(
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
                html.H6("Candlestick patterns @ VBT"),
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
                                        html.H6("OHLCV and signals")
                                    ],
                                ),
                                dcc.Loading(
                                    id="ohlcv_loading",
                                    type="default",
                                    color="#387c9e",
                                    children=[
                                        dcc.Graph(
                                            id="ohlcv_graph",
                                            figure={
                                                "layout": default_layout
                                            }
                                        )
                                    ],
                                ),
                                html.Small("Hint: Use Box and Lasso Select to filter signals"),
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
                                                    id="value_loading",
                                                    type="default",
                                                    color="#387c9e",
                                                    children=[
                                                        dcc.Graph(
                                                            id="value_graph",
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
                                                    id="stats_loading",
                                                    type="default",
                                                    color="#387c9e",
                                                    children=[
                                                        dash_table.DataTable(
                                                            id="stats_table",
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
                                                    id="metric_stats_loading",
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
                                                                            id="metric_dropdown"
                                                                        ),
                                                                    ]
                                                                ),
                                                                dbc.Col(
                                                                    lg=8, sm=12,
                                                                    children=[
                                                                        dcc.Graph(
                                                                            id="metric_graph",
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
                                    id="reset_button"
                                ),
                                html.Details(
                                    open=True,
                                    children=[
                                        html.Summary(
                                            className="section-title",
                                            children="Data",
                                        ),
                                        dbc.Row(
                                            children=[
                                                dbc.Col(
                                                    children=[
                                                        html.Label("Yahoo! Finance symbol:"),
                                                        dcc.Input(
                                                            id="symbol_input",
                                                            className="input-control",
                                                            type="text",
                                                            value=default_symbol,
                                                            placeholder="Enter symbol...",
                                                            debounce=True
                                                        ),
                                                    ]
                                                ),
                                                dbc.Col()
                                            ],
                                        ),
                                        dbc.Row(
                                            children=[
                                                dbc.Col(
                                                    children=[
                                                        html.Label("Period:"),
                                                        dcc.Dropdown(
                                                            id="period_dropdown",
                                                            options=[{"label": i, "value": i} for i in periods],
                                                            value=default_period,
                                                        ),
                                                    ]
                                                ),
                                                dbc.Col(
                                                    children=[
                                                        html.Label("Interval:"),
                                                        dcc.Dropdown(
                                                            id="interval_dropdown",
                                                            options=[{"label": i, "value": i} for i in intervals],
                                                            value=default_interval,
                                                        ),
                                                    ]
                                                )
                                            ],
                                        ),
                                        html.Label("Filter period:"),
                                        dcc.RangeSlider(
                                            id="date_slider",
                                            min=0.,
                                            max=1.,
                                            value=default_date_range,
                                            allowCross=False,
                                            tooltip={
                                                'placement': 'bottom'
                                            }
                                        ),
                                        dcc.Checklist(
                                            id="yf_checklist",
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
                                            id='entry_settings',
                                            children=[
                                                html.Button(
                                                    "All",
                                                    id="entry_all_button"
                                                ),
                                                html.Button(
                                                    "Random",
                                                    id="entry_random_button"
                                                ),
                                                html.Button(
                                                    "Clear",
                                                    id="entry_clear_button"
                                                ),
                                                html.Label("Number of random patterns:"),
                                                dbc.Row(
                                                    children=[
                                                        dbc.Col(
                                                            children=[
                                                                dcc.Input(
                                                                    id="entry_n_random_input",
                                                                    className="input-control",
                                                                    value=default_entry_n_random,
                                                                    placeholder="Enter number...",
                                                                    debounce=True,
                                                                    type="number",
                                                                    min=1, max=len(patterns), step=1
                                                                ),
                                                            ]
                                                        ),
                                                        dbc.Col()
                                                    ],
                                                ),
                                                html.Label("Select patterns:"),
                                                dcc.Dropdown(
                                                    id="entry_pattern_dropdown",
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
                                            id="exit_checklist",
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
                                            id='exit_settings',
                                            hidden="same_as_entry" in default_exit_options,
                                            children=[
                                                html.Button(
                                                    "All",
                                                    id="exit_all_button"
                                                ),
                                                html.Button(
                                                    "Random",
                                                    id="exit_random_button"
                                                ),
                                                html.Button(
                                                    "Clear",
                                                    id="exit_clear_button"
                                                ),
                                                html.Label("Number of random patterns:"),
                                                dbc.Row(
                                                    children=[
                                                        dbc.Col(
                                                            children=[
                                                                dcc.Input(
                                                                    id="exit_n_random_input",
                                                                    className="input-control",
                                                                    value=default_exit_n_random,
                                                                    placeholder="Enter number...",
                                                                    debounce=True,
                                                                    type="number",
                                                                    min=1, max=len(patterns), step=1
                                                                ),
                                                            ]
                                                        ),
                                                        dbc.Col()
                                                    ],
                                                ),
                                                html.Label("Select patterns:"),
                                                dcc.Dropdown(
                                                    id="exit_pattern_dropdown",
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
                                            id="candle_settings_table",
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
                                    open=False,
                                    children=[
                                        html.Summary(
                                            className="section-title",
                                            children="Custom pattern",
                                        ),
                                        html.Div(
                                            id='custom_settings',
                                            children=[
                                                html.Label("Select entry dates:"),
                                                dcc.Dropdown(
                                                    id="custom_entry_dropdown",
                                                    options=[],
                                                    multi=True,
                                                    value=default_entry_dates,
                                                ),
                                                html.Label("Select exit dates:"),
                                                dcc.Dropdown(
                                                    id="custom_exit_dropdown",
                                                    options=[],
                                                    multi=True,
                                                    value=default_exit_dates,
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
                                            children="Simulation settings",
                                        ),
                                        dbc.Row(
                                            children=[
                                                dbc.Col(
                                                    children=[
                                                        html.Label("Fees (in %):"),
                                                        dcc.Input(
                                                            id="fees_input",
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
                                                        html.Label("Fixed fees:"),
                                                        dcc.Input(
                                                            id="fixed_fees_input",
                                                            className="input-control",
                                                            type="number",
                                                            value=default_fixed_fees,
                                                            placeholder="Enter fixed fees...",
                                                            debounce=True,
                                                            min=0
                                                        ),
                                                    ]
                                                ),
                                            ]
                                        ),
                                        dbc.Row(
                                            children=[
                                                dbc.Col(
                                                    children=[
                                                        html.Label("Slippage (in % of H-O):"),
                                                        dcc.Input(
                                                            id="slippage_input",
                                                            className="input-control",
                                                            type="number",
                                                            value=default_slippage,
                                                            placeholder="Enter slippage...",
                                                            debounce=True,
                                                            min=0, max=100
                                                        ),
                                                    ]
                                                ),
                                                dbc.Col()
                                            ],
                                        ),
                                        dcc.Checklist(
                                            id="sim_checklist",
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
                                                            id="n_random_strat_input",
                                                            className="input-control",
                                                            value=default_n_random_strat,
                                                            placeholder="Enter number...",
                                                            debounce=True,
                                                            type="number",
                                                            min=10, max=1000, step=1
                                                        ),
                                                    ]
                                                ),
                                                dbc.Col()
                                            ],
                                        ),
                                        dcc.Checklist(
                                            id="prob_checklist",
                                            options=[{
                                                "label": "Mimic strategy by shuffling",
                                                "value": "mimic_strategy"
                                            }],
                                            value=default_prob_options,
                                            style={
                                                "color": "#7b7d8d"
                                            }
                                        ),
                                        html.Div(
                                            id='prob_settings',
                                            hidden="mimic_strategy" in default_prob_options,
                                            children=[
                                                dbc.Row(
                                                    children=[
                                                        dbc.Col(
                                                            children=[
                                                                html.Label("Entry probability (in %):"),
                                                                dcc.Input(
                                                                    id="entry_prob_input",
                                                                    className="input-control",
                                                                    value=default_entry_prob,
                                                                    placeholder="Enter number...",
                                                                    debounce=True,
                                                                    type="number",
                                                                    min=0, max=100
                                                                ),
                                                            ]
                                                        ),
                                                        dbc.Col(
                                                            children=[
                                                                html.Label("Exit probability (in %):"),
                                                                dcc.Input(
                                                                    id="exit_prob_input",
                                                                    className="input-control",
                                                                    value=default_exit_prob,
                                                                    placeholder="Enter number...",
                                                                    debounce=True,
                                                                    type="number",
                                                                    min=0, max=100
                                                                ),
                                                            ]
                                                        ),
                                                    ],
                                                ),
                                            ]
                                        ),
                                        dcc.Checklist(
                                            id="stats_checklist",
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
        html.Div(id='data_signal', style={'display': 'none'}),
        html.Div(id='index_signal', style={'display': 'none'}),
        html.Div(id='candle_settings_signal', style={'display': 'none'}),
        html.Div(id='stats_signal', style={'display': 'none'})
    ],
)


@cache.memoize()
def fetch_data(symbol, period, interval, auto_adjust, back_adjust):
    """Fetch OHLCV data from Yahoo! Finance."""
    return yf.Ticker(symbol).history(
        period=period,
        interval=interval,
        actions=False,
        auto_adjust=auto_adjust,
        back_adjust=back_adjust
    )


@app.callback(
    [Output('data_signal', 'children'),
     Output('index_signal', 'children')],
    [Input('symbol_input', 'value'),
     Input('period_dropdown', 'value'),
     Input('interval_dropdown', 'value'),
     Input("yf_checklist", "value")]
)
def update_data(symbol, period, interval, yf_options):
    """Store data into a hidden DIV to avoid repeatedly calling Yahoo's API."""
    auto_adjust = 'auto_adjust' in yf_options
    back_adjust = 'back_adjust' in yf_options
    df = fetch_data(symbol, period, interval, auto_adjust, back_adjust)
    return df.to_json(date_format='iso', orient='split'), df.index.tolist()


@app.callback(
    [Output('date_slider', 'min'),
     Output('date_slider', 'max'),
     Output('date_slider', 'value')],
    [Input('index_signal', 'children')]
)
def update_date_slider(date_list):
    """Once index (dates) has changed, reset the date slider."""
    return 0, len(date_list) - 1, [0, len(date_list) - 1]


@app.callback(
    [Output('custom_entry_dropdown', 'options'),
     Output('custom_exit_dropdown', 'options')],
    [Input('index_signal', 'children'),
     Input('date_slider', 'value')]
)
def update_custom_options(date_list, date_range):
    """Once dates have changed, update entry/exit dates in custom pattern section.

    If selected dates cannot be found in new dates, they will be automatically removed."""
    filtered_dates = np.asarray(date_list)[date_range[0]:date_range[1] + 1].tolist()
    custom_options = [{"label": i, "value": i} for i in filtered_dates]
    return custom_options, custom_options


@app.callback(
    Output('entry_pattern_dropdown', 'value'),
    [Input("entry_all_button", "n_clicks"),
     Input("entry_random_button", "n_clicks"),
     Input("entry_clear_button", "n_clicks"),
     Input("reset_button", "n_clicks")],
    [State("entry_n_random_input", "value")]
)
def select_entry_patterns(_1, _2, _3, _4, n_random):
    """Select all/random entry patterns or clear."""
    ctx = dash.callback_context
    if ctx.triggered:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        if button_id == 'entry_all_button':
            return patterns
        elif button_id == 'entry_random_button':
            return random.sample(patterns, n_random)
        elif button_id == 'entry_clear_button':
            return []
        elif button_id == 'reset_button':
            return default_entry_patterns
    return dash.no_update


@app.callback(
    [Output("exit_settings", "hidden"),
     Output("exit_n_random_input", "value"),
     Output('exit_pattern_dropdown', 'value')],
    [Input("exit_checklist", "value"),
     Input("exit_all_button", "n_clicks"),
     Input("exit_random_button", "n_clicks"),
     Input("exit_clear_button", "n_clicks"),
     Input("reset_button", "n_clicks"),
     Input("entry_n_random_input", "value"),
     Input('entry_pattern_dropdown', 'value')],
    [State("exit_n_random_input", "value")]
)
def select_exit_patterns(exit_options, _1, _2, _3, _4, entry_n_random, entry_patterns, exit_n_random):
    """Select all/random exit patterns, clear, or configure the same way as entry patterns."""
    ctx = dash.callback_context
    same_as_entry = 'same_as_entry' in exit_options
    if ctx.triggered:
        control_id = ctx.triggered[0]['prop_id'].split('.')[0]
        if control_id == 'exit_checklist':
            return same_as_entry, entry_n_random, entry_patterns
        elif control_id == 'exit_all_button':
            return same_as_entry, exit_n_random, patterns
        elif control_id == 'exit_random_button':
            return same_as_entry, exit_n_random, random.sample(patterns, exit_n_random)
        elif control_id == 'exit_clear_button':
            return same_as_entry, exit_n_random, []
        elif control_id == 'reset_button':
            default_same_as_entry = 'same_as_entry' in default_exit_options
            return default_same_as_entry, default_exit_n_random, default_exit_patterns
        elif control_id in ('entry_n_random_input', 'entry_pattern_dropdown'):
            if same_as_entry:
                return same_as_entry, entry_n_random, entry_patterns
    return dash.no_update


@app.callback(
    Output('candle_settings_signal', 'children'),
    [Input('candle_settings_table', 'data')]
)
def set_candle_settings(data):
    """Update candle settings in TA-Lib."""
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
    [Output('ohlcv_graph', 'figure'),
     Output("prob_settings", "hidden"),
     Output("entry_prob_input", "value"),
     Output("exit_prob_input", "value")],
    [Input('data_signal', 'children'),
     Input('date_slider', 'value'),
     Input('entry_pattern_dropdown', 'value'),
     Input('exit_pattern_dropdown', 'value'),
     Input('candle_settings_signal', 'children'),
     Input('custom_entry_dropdown', 'value'),
     Input('custom_exit_dropdown', 'value'),
     Input('prob_checklist', 'value'),
     Input("reset_button", "n_clicks")],
    [State("entry_prob_input", "value"),
     State("exit_prob_input", "value")]
)
def update_ohlcv(df_json, date_range, entry_patterns, exit_patterns, _1,
                 entry_dates, exit_dates, prob_options, _2, entry_prob, exit_prob):
    """Update OHLCV graph.

    Also update probability settings, as they also depend upon conversion of patterns into signals."""
    df = pd.read_json(df_json, orient='split')

    # Filter by date
    df = df.iloc[date_range[0]:date_range[1] + 1]

    # Run pattern recognition indicators and combine results
    talib_inputs = {
        'open': df['Open'].values,
        'high': df['High'].values,
        'low': df['Low'].values,
        'close': df['Close'].values,
        'volume': df['Volume'].values
    }
    entry_patterns += ['CUSTOM']
    exit_patterns += ['CUSTOM']
    all_patterns = list(set(entry_patterns + exit_patterns))
    signal_df = pd.DataFrame.vbt.empty(
        (len(df.index), len(all_patterns)),
        fill_value=0.,
        index=df.index,
        columns=all_patterns
    )
    for pattern in all_patterns:
        if pattern != 'CUSTOM':
            signal_df[pattern] = abstract.Function(pattern)(talib_inputs)
    signal_df['CUSTOM'].loc[entry_dates] += 100.
    signal_df['CUSTOM'].loc[exit_dates] += -100.
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
    figure = dict(
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
    mimic_strategy = 'mimic_strategy' in prob_options
    ctx = dash.callback_context
    if ctx.triggered:
        control_id = ctx.triggered[0]['prop_id'].split('.')[0]
        if control_id == 'reset_button':
            mimic_strategy = 'mimic_strategy' in default_prob_options
            entry_prob = default_entry_prob
            exit_prob = default_exit_prob
    if mimic_strategy:
        entry_prob = np.round(len(entry_df.index) / len(df.index) * 100, 4)
        exit_prob = np.round(len(exit_df.index) / len(df.index) * 100, 4)
    return figure, mimic_strategy, entry_prob, exit_prob


def simulate_portfolio(df, interval, date_range, selected_data, entry_patterns, exit_patterns,
                       entry_dates, exit_dates, fees, fixed_fees, slippage, sim_options,
                       n_random_strat, prob_options, entry_prob, exit_prob):
    """Simulate portfolio of the main strategy, buy & hold strategy, and a bunch of random strategies."""
    # Filter by date
    df = df.iloc[date_range[0]:date_range[1] + 1]

    # Run pattern recognition indicators and combine results
    talib_inputs = {
        'open': df['Open'].values,
        'high': df['High'].values,
        'low': df['Low'].values,
        'close': df['Close'].values,
        'volume': df['Volume'].values
    }
    entry_patterns += ['CUSTOM']
    exit_patterns += ['CUSTOM']
    all_patterns = list(set(entry_patterns + exit_patterns))
    entry_i = [all_patterns.index(p) for p in entry_patterns]
    exit_i = [all_patterns.index(p) for p in exit_patterns]
    signals = np.full((len(df.index), len(all_patterns)), 0., dtype=np.float_)
    for i, pattern in enumerate(all_patterns):
        if pattern != 'CUSTOM':
            signals[:, i] = abstract.Function(pattern)(talib_inputs)
    signals[np.flatnonzero(df.index.isin(entry_dates)), all_patterns.index('CUSTOM')] += 100.
    signals[np.flatnonzero(df.index.isin(exit_dates)), all_patterns.index('CUSTOM')] += -100.
    signals /= 100.  # TA-Lib functions have output in increments of 100

    # Filter signals
    if selected_data is not None:
        new_signals = np.full_like(signals, 0.)
        for point in selected_data['points']:
            if 'customdata' in point:
                point_patterns = point['customdata'][0].split('<br>')
                pi = df.index.get_loc(point['x'])
                for p in point_patterns:
                    pc = all_patterns.index(p)
                    new_signals[pi, pc] = signals[pi, pc]
        signals = new_signals

    # Generate size for main
    def _generate_size(signals):
        entry_signals = signals[:, entry_i]
        exit_signals = signals[:, exit_i]
        return np.where(entry_signals > 0, entry_signals, 0).sum(axis=1) + \
               np.where(exit_signals < 0, exit_signals, 0).sum(axis=1)

    main_size = np.empty((len(df.index),), dtype=np.float_)
    main_size[0] = 0  # avoid looking into future
    main_size[1:] = _generate_size(signals)[:-1]

    # Generate size for buy & hold
    hold_size = np.full_like(main_size, 0.)
    hold_size[0] = np.inf

    # Generate size for random
    def _shuffle_along_axis(a, axis):
        idx = np.random.rand(*a.shape).argsort(axis=axis)
        return np.take_along_axis(a, idx, axis=axis)

    rand_size = np.empty((len(df.index), n_random_strat), dtype=np.float_)
    rand_size[0] = 0  # avoid looking into future
    if 'mimic_strategy' in prob_options:
        for i in range(n_random_strat):
            rand_signals = _shuffle_along_axis(signals, 0)
            rand_size[1:, i] = _generate_size(rand_signals)[:-1]
    else:
        entry_signals = pd.DataFrame.vbt.signals.generate_random(
            (rand_size.shape[0] - 1, rand_size.shape[1]), prob=entry_prob / 100).values
        exit_signals = pd.DataFrame.vbt.signals.generate_random(
            (rand_size.shape[0] - 1, rand_size.shape[1]), prob=exit_prob / 100).values
        rand_size[1:, :] = np.where(entry_signals, 1., 0.) - np.where(exit_signals, 1., 0.)

    # Simulate portfolio
    def _simulate_portfolio(size, init_cash):
        accumulate = 'allow_inc_position' in sim_options
        accumulate_exit_mode = AccumulationMode.Reduce \
            if 'allow_dec_position' in sim_options else AccumulationMode.Close
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
            freq=interval
        )

    # Align initial cash across main and random strategies
    aligned_portfolio = _simulate_portfolio(np.hstack((main_size[:, None], rand_size)), InitCashMode.AutoAlign)
    # Fixate initial cash for indexing
    aligned_portfolio = aligned_portfolio.copy(
        init_cash=aligned_portfolio.init_cash()
    )
    # Separate portfolios
    main_portfolio = aligned_portfolio.iloc[0]
    rand_portfolio = aligned_portfolio.iloc[1:]

    # Simulate buy & hold portfolio
    hold_portfolio = _simulate_portfolio(hold_size, main_portfolio.init_cash())

    return main_portfolio, hold_portfolio, rand_portfolio


@app.callback(
    [Output('value_graph', 'figure'),
     Output('stats_table', 'data'),
     Output('stats_signal', 'children'),
     Output('metric_dropdown', 'options'),
     Output('metric_dropdown', 'value')],
    [Input('data_signal', 'children'),
     Input('interval_dropdown', 'value'),
     Input('date_slider', 'value'),
     Input('ohlcv_graph', 'selectedData'),
     Input('entry_pattern_dropdown', 'value'),
     Input('exit_pattern_dropdown', 'value'),
     Input('candle_settings_signal', 'children'),
     Input('custom_entry_dropdown', 'value'),
     Input('custom_exit_dropdown', 'value'),
     Input('fees_input', 'value'),
     Input('fixed_fees_input', 'value'),
     Input('slippage_input', 'value'),
     Input('sim_checklist', 'value'),
     Input('n_random_strat_input', 'value'),
     Input('prob_checklist', 'value'),
     Input("entry_prob_input", "value"),
     Input("exit_prob_input", "value"),
     Input('stats_checklist', 'value'),
     Input("reset_button", "n_clicks")],
    [State('metric_dropdown', 'value')]
)
def update_stats(df_json, interval, date_range, selected_data, entry_patterns, exit_patterns,
                 _1, entry_dates, exit_dates, fees, fixed_fees, slippage, sim_options, n_random_strat,
                 prob_options, entry_prob, exit_prob, stats_options, _2, curr_metric):
    """Final stage where we calculate key performance metrics and compare strategies."""
    df = pd.read_json(df_json, orient='split')

    # Simulate portfolio
    main_portfolio, hold_portfolio, rand_portfolio = simulate_portfolio(
        df, interval, date_range, selected_data, entry_patterns, exit_patterns,
        entry_dates, exit_dates, fees, fixed_fees, slippage, sim_options,
        n_random_strat, prob_options, entry_prob, exit_prob)

    # Get orders
    buy_trace, sell_trace = main_portfolio.orders().plot().data[1:]
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
                name=f"Value (Buy & Hold)",
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
        if control_id == 'reset_button':
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
    Output('metric_graph', 'figure'),
    [Input('stats_signal', 'children'),
     Input('metric_dropdown', 'value')]
)
def update_metric_stats(stats_json, metric):
    """Once a new metric has been selected, plot its distribution."""
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
                hovertemplate='%{x}<br>Buy & Hold',
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
    [Output('symbol_input', 'value'),
     Output('period_dropdown', 'value'),
     Output('interval_dropdown', 'value'),
     Output("yf_checklist", "value"),
     Output("entry_n_random_input", "value"),
     Output("exit_checklist", "value"),
     Output('candle_settings_table', 'data'),
     Output('custom_entry_dropdown', 'value'),
     Output('custom_exit_dropdown', 'value'),
     Output('fees_input', 'value'),
     Output('fixed_fees_input', 'value'),
     Output('slippage_input', 'value'),
     Output('sim_checklist', 'value'),
     Output('n_random_strat_input', 'value'),
     Output("prob_checklist", "value"),
     Output('stats_checklist', 'value')],
    [Input("reset_button", "n_clicks")],
    prevent_initial_call=True
)
def reset_settings(_):
    """Reset most settings. Other settings are reset in their callbacks."""
    return default_symbol, \
           default_period, \
           default_interval, \
           default_yf_options, \
           default_entry_n_random, \
           default_exit_options, \
           default_candle_settings.to_dict("records"), \
           default_entry_dates, \
           default_exit_dates, \
           default_fees, \
           default_fixed_fees, \
           default_slippage, \
           default_sim_options, \
           default_n_random_strat, \
           default_prob_options, \
           default_stats_options


if __name__ == '__main__':
    app.run_server(host=HOST, port=PORT, debug=DEBUG)
