FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

WORKDIR /app

COPY pyproject.toml .
COPY framework/ framework/
COPY models/llama/ models/llama/

RUN pip install --no-cache-dir -e .

# HF cache location — weights volume is mounted here by the orchestrator
ENV HF_HOME=/hf_cache

# Partition .pt files are mounted here at runtime
ENV PARTITIONS_DIR=/partitions

EXPOSE 8000

CMD ["python", "-m", "framework.node.llama_server"]
