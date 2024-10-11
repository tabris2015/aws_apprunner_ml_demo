FROM python:3.11-slim

ENV PORT 8000

# dependencias para OpenCV
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

ADD ./src /src
COPY pyproject.toml pyproject.toml
COPY uv.lock uv.lock

ENV PIP_INDEX_URL=https://download.pytorch.org/whl/cpu
RUN uv sync --frozen --no-dev

CMD uv run fastapi run src/main.py --host 0.0.0.0 --port ${PORT}