# aws_apprunner_ml_demo

## Running locally

First create an environment with uv:

```
uv venv
```

then, install dependencies:

```
uv pip install -r requirements.txt
```

finally, run the dev server:

```
uv run fastapi dev src/main.py
```

## Running with docker
```
docker compose up
```

## Local development with Devcontainer
You can also use a devcontainer for local development, using VSCode you can build the devcontainer and use it for local development and debugging.