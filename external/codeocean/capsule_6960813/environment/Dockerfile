# hash:sha256:4108b8d9b14908b2e883489808957dd68711ac8dcd109494343023906eca20e6
FROM registry.codeocean.allenneuraldynamics.org/codeocean/mambaforge3:24.5.0-0-python3.12.4-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG GIT_ASKPASS
COPY git-ask-pass /

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        unzip=6.0-26ubuntu3.2 \
        zip=3.0-12build2 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install -U --no-cache-dir \
    black==26.5.1 \
    flake8==7.3.0 \
    jupyterlab==4.6.3

COPY postInstall /
RUN --mount=type=secret,id=secrets . /run/secrets/secrets \
    && /postInstall
