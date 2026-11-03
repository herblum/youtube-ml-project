FROM python:3.11-slim

RUN pip install uv

WORKDIR /app

COPY ".python-version" "pyproject.toml" "uv.lock" ./
RUN uv sync --locked

COPY "predict.py" "model.bin" ./

EXPOSE 8000

ENTRYPOINT ["uv", "run", "uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "8000"]