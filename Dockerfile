FROM pytorch/pytorch:2.3.0-cuda11.8-cudnn8-runtime

WORKDIR /app

# iproute2 provides tc for link bandwidth simulation
RUN apt-get update && apt-get install -y --no-install-recommends iproute2 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
RUN pip install --no-cache-dir \
    fastapi \
    "uvicorn[standard]" \
    httpx \
    pyyaml \
    pydantic \
    torchvision \
    transformers \
    numpy \
    pandas

COPY framework/ ./framework/
COPY models/ ./models/
COPY entrypoint.sh ./entrypoint.sh
RUN chmod +x entrypoint.sh

ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["python", "-m", "framework.node.server"]
