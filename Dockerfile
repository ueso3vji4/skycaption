FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn8-runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl wget \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY app.py /app/app.py
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

WORKDIR /app

RUN mkdir -p /workspace/datasets /workspace/captioned /workspace/hf_cache

ENV HF_HOME=/workspace/hf_cache
ENV TRANSFORMERS_CACHE=/workspace/hf_cache
ENV PYTHONUNBUFFERED=1

EXPOSE 5000

ENTRYPOINT ["/app/entrypoint.sh"]
