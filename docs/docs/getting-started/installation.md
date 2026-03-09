---
title: Installation
---

# Installation

You can install VectorBT with pip, the Python package manager, or with Docker.

## With pip

```sh
pip install -U vectorbt
```

To also install optional dependencies:

```sh
pip install -U "vectorbt[full]"
```

## With Docker

You can pull the most recent Docker image if you [have Docker installed](https://docs.docker.com/install/).

```sh
docker run --rm -p 8888:8888 -v "$PWD":/home/jovyan/work polakowo/vectorbt:latest-full
```

This command pulls the latest `polakowo/vectorbt:latest-full` image from Docker Hub.
It then starts a container running a Jupyter server and exposes it on host port 8888.
Visiting `http://127.0.0.1:8888/?token=<token>` in a browser opens JupyterLab, where
`<token>` is the secret token printed in the console.

Docker removes the container when the notebook server exits, but any files written to the
working directory inside the container remain available in the working directory on the host
(because the current directory is mounted into the container). See [Jupyter Docker Stacks -
Quick Start](https://github.com/jupyter/docker-stacks#quick-start).

There are two types of images in [polakowo/vectorbt](https://hub.docker.com/r/polakowo/vectorbt):
the vanilla version and the full version (with optional dependencies). To use the vanilla version,
use `polakowo/vectorbt:latest` instead of `polakowo/vectorbt:latest-full`.

Each Docker image is based on [jupyter/scipy-notebook](https://hub.docker.com/r/jupyter/scipy-notebook)
and comes with a Jupyter environment, vectorbt, and other scientific packages installed.

!!! note
    If you already have a Jupyter server running locally on port 8888, use another port, such as 10000:

    ```sh
    docker run --rm -p 10000:8888 -v "$PWD":/home/jovyan/work polakowo/vectorbt:latest-full
    ```

    Then open `http://127.0.0.1:10000/?token=<token>` in your browser (note the port change from `8888` to `10000`).

## With git

Of course, you can pull vectorbt directly from `git`:

```sh
git clone git@github.com:polakowo/vectorbt.git vectorbt
```

Install the package:

```sh
pip install -e vectorbt
```

## Troubleshooting

* [TA-Lib](https://github.com/mrjbq7/ta-lib#dependencies)
* [Jupyter Notebook and JupyterLab](https://plotly.com/python/getting-started/#jupyter-notebook-support)
* [Apple M1](https://github.com/polakowo/vectorbt/issues/320)