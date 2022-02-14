---
title: Installation
---

# Installation

You can install vectorbt with pip, the Python package manager, or with Docker.

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
docker run --rm -p 8888:8888 -v "$PWD":/home/jovyan/work polakowo/vectorbt
```

This command pulls the latest `polakowo/vectorbt` image from Docker Hub. It then starts a container running 
a Jupyter Notebook server and exposes the server on host port 8888. Visiting `http://127.0.0.1:8888/?token=<token>` 
in a browser loads JupyterLab, where token is the secret token printed in the console. Docker destroys 
the container after notebook server exit, but any files written to the working directory in the container 
remain intact in the working directory on the host. See [Jupyter Docker Stacks - Quick Start](https://github.com/jupyter/docker-stacks#quick-start).

There are two types of images: 

* **[polakowo/vectorbt](https://hub.docker.com/r/polakowo/vectorbt)**: vanilla version (default)
* **[polakowo/vectorbt-full](https://hub.docker.com/r/polakowo/vectorbt-full)**: full version (with optional dependencies)

Each Docker image is based on [jupyter/scipy-notebook](https://hub.docker.com/r/jupyter/scipy-notebook) 
and comes with Jupyter environment, vectorbt, and other scientific packages installed.

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