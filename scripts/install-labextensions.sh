#!/bin/bash

jupyter labextension install jupyterlab-plotly --no-build
jupyter labextension install @jupyter-widgets/jupyterlab-manager plotlywidget --no-build
jupyter lab build