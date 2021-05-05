FROM jupyter/scipy-notebook:bbf0ada0a935

USER root
WORKDIR /tmp

RUN apt-get update && \
 apt-get install -yq --no-install-recommends curl && \
 apt-get clean && \
 rm -rf /var/lib/apt/lists/*

COPY . vectorbt
WORKDIR vectorbt
RUN chmod -R +x scripts

ARG FULL="yes"

RUN if [[ -n "${FULL}" ]] ; then \
    scripts/install-talib.sh && pip install --no-cache-dir .[full] ; else \
    pip install --no-cache-dir . ; fi

RUN scripts/install-labextensions.sh && \
    jupyter lab clean && \
    npm cache clean --force && \
    rm -rf /home/$NB_USER/.cache/yarn && \
    rm -rf $CONDA_DIR/share/jupyter/lab/staging

USER $NB_UID

ARG TEST

RUN if [[ -n "${TEST}" ]] ; then \
    pip install --no-cache-dir pytest && \
    export NUMBA_BOUNDSCHECK=1 && \
    export NUMBA_DISABLE_JIT=1 && \
    python -m pytest tests ; fi

WORKDIR "$HOME/work"

ENV JUPYTER_ENABLE_LAB "yes"