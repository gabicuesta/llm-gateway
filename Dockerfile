FROM python:3.13-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir . && useradd --create-home gateway
USER gateway
EXPOSE 8080
VOLUME ["/data"]
ENV LLM_GATEWAY_CONFIG=/config/config.yaml
CMD ["uvicorn", "--factory", "llm_gateway.cli:app_from_env", "--host", "0.0.0.0", "--port", "8080"]
