# Polsen — container image (Railway, Fly, Render, any Docker host)
#
# Railway uses this Dockerfile automatically when it is present in the repo root.
# It installs the OFFICIAL xtb Linux binary (27 MB, LGPL-3.0-or-later) so the quantum
# verification panel works in production. The app does not require xtb — if this step is
# removed the panel simply auto-hides — so it is safe to drop for a slimmer image.

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# ---- system deps -------------------------------------------------------------------
# curl+xz to fetch/unpack xtb; libgomp1 is xtb's OpenMP runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl xz-utils libgomp1 ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# ---- xtb (GFN2 quantum cross-check) --------------------------------------------------
# Pinned version + published SHA256 so the build is reproducible and tamper-evident.
ARG XTB_VERSION=6.7.1
ARG XTB_SHA256=62a8d18778286e815292ee53d76ce447daf460a4dea3782c0f25cbac7019b5df
RUN set -eux; \
    curl -fsSL -o /tmp/xtb.tar.xz \
      "https://github.com/grimme-lab/xtb/releases/download/v${XTB_VERSION}/xtb-${XTB_VERSION}-linux-x86_64.tar.xz"; \
    echo "${XTB_SHA256}  /tmp/xtb.tar.xz" | sha256sum -c -; \
    mkdir -p /opt && tar -xJf /tmp/xtb.tar.xz -C /opt; \
    rm /tmp/xtb.tar.xz; \
    /opt/xtb-dist/bin/xtb --version

# xtb finds its parameter files through XTBPATH; appv3 finds the binary through XTB_EXE.
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

# Railway injects $PORT; default to 8080 for local `docker run`.
ENV PORT=8080
EXPOSE 8080
CMD ["sh", "-c", "streamlit run app.py --server.port=${PORT:-8080} --server.address=0.0.0.0 --server.headless=true --browser.gatherUsageStats=false"]
