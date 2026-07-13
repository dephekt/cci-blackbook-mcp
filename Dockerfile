FROM ghcr.io/astral-sh/uv:0.9.21 AS uv

# The dual-index backend embeds via Voyage's HTTP API (api.voyageai.com); the
# container needs outbound HTTPS at ingest and query time. PyMuPDF/Pillow ship
# self-contained manylinux wheels, so no poppler/system PDF deps are needed.
# The Intel-GPU/OpenVINO layers below are legacy (FastEmbed fallback) and are
# removed atomically in the deferred GPU-stack follow-up.
FROM ubuntu:24.04

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH="/opt/venv/bin:${PATH}"

COPY --from=uv /uv /uvx /usr/local/bin/

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        software-properties-common \
        python3.12 \
        python3.12-venv \
        libgomp1 \
        ocl-icd-libopencl1 \
    && add-apt-repository -y ppa:kobuk-team/intel-graphics \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        clinfo \
        intel-opencl-icd \
        libigc2 \
        libigdgmm12 \
        libze1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-install-project

COPY app/cci_blackbook /app/app/cci_blackbook
RUN uv sync --locked --no-dev \
    && openvino_version="$(python -c 'import tomllib; packages = tomllib.load(open("uv.lock", "rb"))["package"]; print(next(p["version"] for p in packages if p["name"] == "onnxruntime-openvino"))')" \
    && uv pip install --no-deps --force-reinstall "onnxruntime-openvino==${openvino_version}"

EXPOSE 8000

CMD ["cci-blackbook-mcp"]
