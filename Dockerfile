# Base stage for building the environment
FROM python:3.10-slim AS builder
COPY requirements-lock.txt .
RUN pip install --user --no-cache-dir -r requirements-lock.txt

FROM python:3.10-slim
WORKDIR /app
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH
