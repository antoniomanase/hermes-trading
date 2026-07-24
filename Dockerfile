FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates && rm -rf /var/lib/apt/lists/*
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"
COPY pyproject.toml uv.lock ./
COPY hermes_trading ./hermes_trading
# Bake initial state as *defaults*; the volume mounts empty over /app/state,
# so entrypoint.sh seeds these onto the volume on first boot only.
COPY state ./state_default
COPY entrypoint.sh ./entrypoint.sh
RUN uv sync
ENV HERMES_TRADING_MODE=paper
ENTRYPOINT ["/bin/bash", "/app/entrypoint.sh"]
CMD ["uv", "run", "python", "-m", "hermes_trading.run"]
