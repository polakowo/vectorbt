# -*- coding: utf-8 -*-

# Copyright (c) 2021 Oleg Polakow. All rights reserved.
# This code is licensed under Apache 2.0 with Commons Clause license (see LICENSE.md for details)

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
from vectorbt import settings
from vectorbt.utils.config import merge_dicts
from vectorbt.utils.colors import adjust_opacity
from vectorbt.portfolio.enums import Direction, DirectionConflictMode
from vectorbt.portfolio.base import Portfolio

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
directions = Direction._fields
conflict_modes = DirectionConflictMode._fields
plot_types = ['OHLC', 'Candlestick']

# Colors
color_schema = settings['plotting']['color_schema']
bgcolor = "#272a32"
dark_bgcolor = "#1d2026"
fontcolor = "#9fa6b7"
dark_fontcolor = "#7b7d8d"
gridcolor = "#323b56"
loadcolor = "#387c9e"
active_color = "#88ccee"

# Defaults
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
default_direction = directions[0]
default_conflict_mode = conflict_modes[0]
default_sim_options = ['allow_accumulate']
default_n_random_strat = 50
default_stats_options = ['incl_open']
default_layout = dict(
    autosize=True,
    margin=dict(b=40, t=20),
    font=dict(
        color=fontcolor
    ),
    plot_bgcolor=bgcolor,
    paper_bgcolor=bgcolor,
    legend=dict(
        font=dict(size=10),
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ),
)
default_subplots = ['orders', 'trade_pnl', 'cum_returns']
default_plot_type = 'OHLC'

app.layout = html.Div(
    children=[
        html.Div(
            className="banner",
            children=[
                html.H6("vectorbt: candlestick patterns"),
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
                                dbc.Row(
                                    children=[
                                        dbc.Col(
                                            lg=4, sm=12,
                                            children=[
                                                html.Label("Select plot type:"),
                                                dcc.Dropdown(
                                                    id="plot_type_dropdown",
                                                    options=[{"value": i, "label": i} for i in plot_types],
                                                    value=default_plot_type,
                                                ),
                                            ]
                                        )
                                    ],
                                ),
                                dcc.Loading(
                                    id="ohlcv_loading",
                                    type="default",
                                    color=loadcolor,
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
                                                        html.H6("Portfolio")
                                                    ],
                                                ),
                                                dbc.Row(
                                                    children=[
                                                        dbc.Col(
                                                            lg=6, sm=12,
                                                            children=[
                                                                html.Label("Select subplots:"),
                                                                dcc.Dropdown(
                                                                    id="subplot_dropdown",
                                                                    options=[
                                                                        {"value": k, "label": v['title']}
                                                                        for k, v in Portfolio.subplots.items()
                                                                    ],
                                                                    multi=True,
                                                                    value=default_subplots,
                                                                ),
                                                            ]
                                                        )
                                                    ],
                                                ),
                                                dcc.Loading(
                                                    id="portfolio_loading",
                                                    type="default",
                                                    color=loadcolor,
                                                    children=[
                                                        dcc.Graph(
                                                            id="portfolio_graph",
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
                                                    color=loadcolor,
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
                                                                "backgroundColor": dark_bgcolor,
                                                                "color": active_color,
                                                                "border": "1px solid " + active_color,
                                                            }, {
                                                                "if": {"state": "active"},
                                                                "backgroundColor": dark_bgcolor,
                                                                "color": active_color,
                                                                "border": "1px solid " + active_color,
                                                            }],
                                                            style_header={
                                                                "border": "none",
                                                                "backgroundColor": bgcolor,
                                                                "fontWeight": "bold",
                                                                "padding": "0px 5px"
                                                            },
                                                            style_data={
                                                                "border": "none",
                                                                "backgroundColor": bgcolor,
                                                                "color": dark_fontcolor,
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
                                                    color=loadcolor,
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
                                                            options=[{"value": i, "label": i} for i in periods],
                                                            value=default_period,
                                                        ),
                                                    ]
                                                ),
                                                dbc.Col(
                                                    children=[
                                                        html.Label("Interval:"),
                                                        dcc.Dropdown(
                                                            id="interval_dropdown",
                                                            options=[{"value": i, "label": i} for i in intervals],
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
                                                "color": dark_fontcolor
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
                                                    options=[{"value": i, "label": i} for i in patterns],
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
                                                "color": dark_fontcolor
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
                                                    options=[{"value": i, "label": i} for i in patterns],
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
                                                "backgroundColor": dark_bgcolor,
                                                "border": "1px solid dimgrey"
                                            }, {
                                                "if": {"state": "selected"},
                                                "backgroundColor": dark_bgcolor,
                                                "color": active_color,
                                                "border": "1px solid " + active_color,
                                            }, {
                                                "if": {"state": "active"},
                                                "backgroundColor": dark_bgcolor,
                                                "color": active_color,
                                                "border": "1px solid " + active_color,
                                            }],
                                            style_header={
                                                "border": "none",
                                                "backgroundColor": bgcolor,
                                                "fontWeight": "bold",
                                                "padding": "0px 5px"
                                            },
                                            style_data={
                                                "border": "none",
                                                "backgroundColor": bgcolor,
                                                "color": dark_fontcolor
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
                                        dbc.Row(
                                            children=[
                                                dbc.Col(
                                                    children=[
                                                        html.Label("Direction:"),
                                                        dcc.Dropdown(
                                                            id="direction_dropdown",
                                                            options=[{"value": i, "label": i} for i in directions],
                                                            value=default_direction,
                                                        ),
                                                    ]
                                                ),
                                                dbc.Col(
                                                    children=[
                                                        html.Label("Conflict Mode:"),
                                                        dcc.Dropdown(
                                                            id="conflict_mode_dropdown",
                                                            options=[{"value": i, "label": i} for i in conflict_modes],
                                                            value=default_conflict_mode
                                                        ),
                                                    ]
                                                ),
                                            ],
                                        ),
                                        dcc.Checklist(
                                            id="sim_checklist",
                                            options=[{
                                                "label": "Allow signal accumulation",
                                                "value": "allow_accumulate"
                                            }],
                                            value=default_sim_options,
                                            style={
                                                "color": dark_fontcolor
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
                                                "color": dark_fontcolor
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
                                                "label": "Include open trades in stats",
                                                "value": "incl_open"
                                            }, {
                                                "label": "Use positions instead of trades in stats",
                                                "value": "use_positions"
                                            }],
                                            value=default_stats_options,
                                            style={
                                                "color": dark_fontcolor
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
        html.Div(id='stats_signal', style={'display': 'none'}),
        html.Div(id="window_width", style={'display': 'none'}),
        dcc.Location(id="url")
    ],
)

app.clientside_callback(
    """
    function(href) {
        return window.innerWidth;
    }
    """,
    Output("window_width", "children"),
    [Input("url", "href")],
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
    custom_options = [{"value": i, "label": i} for i in filtered_dates]
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
    [Input('window_width', 'children'),
     Input('plot_type_dropdown', 'value'),
     Input('data_signal', 'children'),
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
def update_ohlcv(window_width, plot_type, df_json, date_range, entry_patterns, exit_patterns, _1,
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
    entry_y.index = pd.to_datetime(entry_y.index)
    exit_y = df.loc[exit_df.index, 'High'] + distance
    exit_y.index = pd.to_datetime(exit_y.index)

    # Prepare signals
    entry_signals = pd.Series.vbt.empty_like(entry_y, True)
    exit_signals = pd.Series.vbt.empty_like(exit_y, True)

    # Build graph
    height = int(9 / 21 * 2 / 3 * window_width)
    fig = df.vbt.ohlcv.plot(
        plot_type=plot_type,
        **merge_dicts(
            default_layout,
            dict(
                width=None,
                height=max(500, height),
                margin=dict(r=40),
                hovermode="closest",
                xaxis2=dict(
                    title='Date'
                ),
                yaxis2=dict(
                    title='Volume'
                ),
                yaxis=dict(
                    title='Price',
                )
            )
        )
    )
    entry_signals.vbt.signals.plot_as_entry_markers(
        y=entry_y,
        trace_kwargs=dict(
            customdata=entry_patterns[:, None],
            hovertemplate='%{x}<br>%{customdata[0]}',
            name='Bullish signal'
        ),
        add_trace_kwargs=dict(row=1, col=1),
        fig=fig
    )
    exit_signals.vbt.signals.plot_as_exit_markers(
        y=exit_y,
        trace_kwargs=dict(
            customdata=exit_patterns[:, None],
            hovertemplate='%{x}<br>%{customdata[0]}',
            name='Bearish signal'
        ),
        add_trace_kwargs=dict(row=1, col=1),
        fig=fig
    )
    fig.update_xaxes(gridcolor=gridcolor)
    fig.update_yaxes(gridcolor=gridcolor, zerolinecolor=gridcolor)
    figure = dict(data=fig.data, layout=fig.layout)

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
                       entry_dates, exit_dates, fees, fixed_fees, slippage, direction, conflict_mode,
                       sim_options, n_random_strat, prob_options, entry_prob, exit_prob):
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
    signals = np.full((len(df.index), len(all_patterns)), 0., dtype=np.float64)
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

    main_size = np.empty((len(df.index),), dtype=np.float64)
    main_size[0] = 0  # avoid looking into future
    main_size[1:] = _generate_size(signals)[:-1]

    # Generate size for buy & hold
    hold_size = np.full_like(main_size, 0.)
    hold_size[0] = np.inf

    # Generate size for random
    def _shuffle_along_axis(a, axis):
        idx = np.random.rand(*a.shape).argsort(axis=axis)
        return np.take_along_axis(a, idx, axis=axis)

    rand_size = np.empty((len(df.index), n_random_strat), dtype=np.float64)
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
    def _simulate_portfolio(size, init_cash='autoalign'):
        return Portfolio.from_signals(
            close=df['Close'],
            entries=size > 0,
            exits=size < 0,
            price=df['Open'],
            size=np.abs(size),
            direction=direction,
            upon_dir_conflict=conflict_mode,
            accumulate='allow_accumulate' in sim_options,
            init_cash=init_cash,
            fees=float(fees) / 100,
            fixed_fees=float(fixed_fees),
            slippage=(float(slippage) / 100) * (df['High'] / df['Open'] - 1),
            freq=interval
        )

    # Align initial cash across main and random strategies
    aligned_portfolio = _simulate_portfolio(np.hstack((main_size[:, None], rand_size)))
    # Fixate initial cash for indexing
    aligned_portfolio = aligned_portfolio.replace(
        init_cash=aligned_portfolio.init_cash
    )
    # Separate portfolios
    main_portfolio = aligned_portfolio.iloc[0]
    rand_portfolio = aligned_portfolio.iloc[1:]

    # Simulate buy & hold portfolio
    hold_portfolio = _simulate_portfolio(hold_size, init_cash=main_portfolio.init_cash)

    return main_portfolio, hold_portfolio, rand_portfolio


@app.callback(
    [Output('portfolio_graph', 'figure'),
     Output('stats_table', 'data'),
     Output('stats_signal', 'children'),
     Output('metric_dropdown', 'options'),
     Output('metric_dropdown', 'value')],
    [Input('window_width', 'children'),
     Input('subplot_dropdown', 'value'),
     Input('data_signal', 'children'),
     Input('symbol_input', 'value'),
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
     Input('direction_dropdown', 'value'),
     Input('conflict_mode_dropdown', 'value'),
     Input('sim_checklist', 'value'),
     Input('n_random_strat_input', 'value'),
     Input('prob_checklist', 'value'),
     Input("entry_prob_input", "value"),
     Input("exit_prob_input", "value"),
     Input('stats_checklist', 'value'),
     Input("reset_button", "n_clicks")],
    [State('metric_dropdown', 'value')]
)
def update_stats(window_width, subplots, df_json, symbol, interval, date_range, selected_data,
                 entry_patterns, exit_patterns, _1, entry_dates, exit_dates, fees, fixed_fees,
                 slippage, direction, conflict_mode, sim_options, n_random_strat, prob_options,
                 entry_prob, exit_prob, stats_options, _2, curr_metric):
    """Final stage where we calculate key performance metrics and compare strategies."""
    df = pd.read_json(df_json, orient='split')

    # Simulate portfolio
    main_portfolio, hold_portfolio, rand_portfolio = simulate_portfolio(
        df, interval, date_range, selected_data, entry_patterns, exit_patterns,
        entry_dates, exit_dates, fees, fixed_fees, slippage, direction, conflict_mode,
        sim_options, n_random_strat, prob_options, entry_prob, exit_prob)

    subplot_settings = dict()
    if 'cum_returns' in subplots:
        subplot_settings['cum_returns'] = dict(
            benchmark_kwargs=dict(
                trace_kwargs=dict(
                    line=dict(
                        color=adjust_opacity(color_schema['yellow'], 0.5)
                    ),
                    name=symbol
                )
            )
        )
    height = int(6 / 21 * 2 / 3 * window_width)
    fig = main_portfolio.plot(
        subplots=subplots,
        subplot_settings=subplot_settings,
        **merge_dicts(
            default_layout,
            dict(
                width=None,
                height=len(subplots) * max(300, height) if len(subplots) > 1 else max(350, height)
            )
        )
    )
    fig.update_traces(xaxis="x" if len(subplots) == 1 else "x" + str(len(subplots)))
    fig.update_xaxes(
        gridcolor=gridcolor
    )
    fig.update_yaxes(
        gridcolor=gridcolor,
        zerolinecolor=gridcolor
    )

    def _chop_microseconds(delta):
        return delta - pd.Timedelta(microseconds=delta.microseconds, nanoseconds=delta.nanoseconds)

    def _metric_to_str(x):
        if isinstance(x, float):
            return '%.2f' % x
        if isinstance(x, pd.Timedelta):
            return str(_chop_microseconds(x))
        return str(x)

    incl_open = 'incl_open' in stats_options
    use_positions = 'use_positions' in stats_options
    main_stats = main_portfolio.stats(settings=dict(incl_open=incl_open, use_positions=use_positions))
    hold_stats = hold_portfolio.stats(settings=dict(incl_open=True, use_positions=use_positions))
    rand_stats = rand_portfolio.stats(settings=dict(incl_open=incl_open, use_positions=use_positions), agg_func=None)
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
    return dict(data=fig.data, layout=fig.layout), \
           table_data.to_dict("records"), \
           json.dumps({
               'main': {m: [_to_float(main_stats[m])] for m in main_stats.index[3:]},
               'hold': {m: [_to_float(hold_stats[m])] for m in main_stats.index[3:]},
               'rand': {m: rand_stats[m].apply(_to_float).values.tolist() for m in main_stats.index[3:]}
           }), \
           [{"value": i, "label": i} for i in main_stats.index[3:]], \
           metric


@app.callback(
    Output('metric_graph', 'figure'),
    [Input('window_width', 'children'),
     Input('stats_signal', 'children'),
     Input('metric_dropdown', 'value')]
)
def update_metric_stats(window_width, stats_json, metric):
    """Once a new metric has been selected, plot its distribution."""
    stats_dict = json.loads(stats_json)
    height = int(9 / 21 * 2 / 3 * 2 / 3 * window_width)
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
                    color=color_schema['blue'],
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
                    color=color_schema['orange'],
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
                    color=color_schema['green'],
                    size=8,
                ),
            ),
        ],
        layout=merge_dicts(
            default_layout,
            dict(
                height=max(350, height),
                showlegend=False,
                margin=dict(l=60, r=20, t=40, b=20),
                hovermode="closest",
                xaxis=dict(
                    gridcolor=gridcolor,
                    title=metric,
                    side='top'
                ),
                yaxis=dict(
                    gridcolor=gridcolor
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
     Output('conflict_mode_dropdown', 'value'),
     Output('direction_dropdown', 'value'),
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
           default_conflict_mode, \
           default_direction, \
           default_sim_options, \
           default_n_random_strat, \
           default_prob_options, \
           default_stats_options


if __name__ == '__main__':
    app.run_server(host=HOST, port=PORT, debug=DEBUG)
