# CPU-only, reproducible educational lab. This Docker image does not download Qwen.
FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /workspace
# PyTorch official CPU index avoids CUDA-wheel dependencies.
RUN python -m pip install --upgrade pip && \
    python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
COPY pyproject.toml README.md ./
COPY src/ ./src/
COPY tests/ ./tests/
RUN python -m pip install -e '.[test]'
RUN mkdir -p /workspace/runs
ENTRYPOINT ["wslab"]
CMD ["pipeline", "--out", "/workspace/runs/demo"]
