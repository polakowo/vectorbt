ARG BASE_IMAGE=quay.io/jupyter/scipy-notebook:python-3.11
FROM ${BASE_IMAGE} AS vectorbt

LABEL org.opencontainers.image.title="vectorbt"
LABEL org.opencontainers.image.description="VectorBT in Jupyter"
LABEL org.opencontainers.image.source="https://github.com/polakowo/vectorbt"

USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
      git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

USER ${NB_UID}
WORKDIR /home/${NB_USER}/work

ARG VBT_EXTRAS=""

COPY --chown=${NB_UID}:${NB_GID} . /tmp/vectorbt

RUN python -m pip install --no-cache-dir -U pip setuptools wheel \
    && python -m pip install --no-cache-dir "/tmp/vectorbt${VBT_EXTRAS:+[${VBT_EXTRAS}]}" \
    && rm -rf /tmp/vectorbt

EXPOSE 8888

CMD ["start-notebook.py", "--ServerApp.ip=0.0.0.0", "--ServerApp.port=8888"]