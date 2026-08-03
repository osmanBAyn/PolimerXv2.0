# Polsen — container image (Railway / any Docker host)
#
# Derived from the previously-working image: SAME system libraries and SAME port (8501).
# Only two deliberate changes: the explicit PyTorch install is gone (USE_T5_RETRO is off, so
# torch is unused and cost ~2.8 GB), and the official xtb binary is added for the quantum
# verification panel.

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# System libraries.
#  - libxrender1 / libxext6 / libsm6 / libx11-dev / libgl1 are REQUIRED by RDKit. Without
#    them RDKit fails to import ("libXrender.so.1: cannot open shared object file") and the
#    container dies at startup — which is a 502 at the edge.
#  - build-essential / git are kept from the previous working image so that any package
#    without a prebuilt wheel can still compile.
#  - curl / xz-utils / libgomp1 are needed to fetch and run xtb.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        libxrender1 \
        libxext6 \
        libsm6 \
        libx11-dev \
        libgl1 \
        libgomp1 \
        curl \
        xz-utils \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# ---- xtb (GFN2 quantum cross-check) --------------------------------------------------
# Official Linux binary, version-pinned and checksum-verified. Deliberately NON-FATAL: the
# app never requires xtb (the verification panel auto-hides when it is missing), so a network
# hiccup here must not take the whole deployment down.
ARG XTB_VERSION=6.7.1
ARG XTB_SHA256=62a8d18778286e815292ee53d76ce447daf460a4dea3782c0f25cbac7019b5df
RUN set -eu; \
    ( curl -fsSL -o /tmp/xtb.tar.xz \
        "https://github.com/grimme-lab/xtb/releases/download/v${XTB_VERSION}/xtb-${XTB_VERSION}-linux-x86_64.tar.xz" \
      && echo "${XTB_SHA256}  /tmp/xtb.tar.xz" | sha256sum -c - \
      && mkdir -p /opt && tar -xJf /tmp/xtb.tar.xz -C /opt \
      && /opt/xtb-dist/bin/xtb --version \
      && echo "xtb installed OK" ) \
    || echo "WARNING: xtb install skipped — the app will run with the verification panel hidden"; \
    rm -f /tmp/xtb.tar.xz

ENV PATH="/opt/xtb-dist/bin:${PATH}" \
    XTBPATH="/opt/xtb-dist/share/xtb" \
    XTB_EXE="/opt/xtb-dist/bin/xtb" \
    OMP_NUM_THREADS=2 \
    MKL_NUM_THREADS=2

# ---- python deps ---------------------------------------------------------------------
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---- app ------------------------------------------------------------------------------
COPY . .

# Port: FIXED at 8501, matching the previously-working image and the platform's routing.
#
# Do NOT switch this to ${PORT}. Railway injects PORT=8080, but its HTTP router for this
# service targets 8501 (inherited from the original image, whose CMD passed no --server.port
# and therefore used Streamlit's 8501 default). Honouring $PORT moved Streamlit to 8080 while
# the router kept probing 8501 -> nothing listening -> 502. The deploy log makes this visible:
# it printed "URL: http://0.0.0.0:8080" while the site was down.
#
# If the service is ever reconfigured to target a different port, change BOTH lines below.
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true", "--browser.gatherUsageStats=false"]
