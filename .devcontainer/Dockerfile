# add platform for enforcing compiling for linux 64 bit
FROM --platform=linux/amd64 python:3.11-slim        

ENV PORT 8000

# OpenCV dependencies
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

COPY pyproject.toml pyproject.toml
COPY uv.lock uv.lock

ENV PIP_INDEX_URL=https://download.pytorch.org/whl/cpu

# Install dependencies
RUN uv sync --frozen --no-dev

ADD ./src /src

ENV PATH="/.venv/bin:$PATH"