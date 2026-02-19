FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

WORKDIR /app
COPY partition_resnet56.py /app/partition_resnet56.py

# Optional: reduce Python buffering
ENV PYTHONUNBUFFERED=1

CMD ["bash", "-lc", "python -V && nvidia-smi && echo Ready"]