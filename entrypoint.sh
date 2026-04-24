#!/bin/bash
set -e

export HF_HOME=/workspace/hf_cache
export TRANSFORMERS_CACHE=/workspace/hf_cache

echo ""
echo "╔══════════════════════════════════════════╗"
echo "║              SkyCaption                  ║"
echo "╚══════════════════════════════════════════╝"
echo ""

mkdir -p /workspace/datasets /workspace/captioned /workspace/hf_cache

MODEL_CACHE="/workspace/hf_cache/models--fancyfeast--llama-joycaption-beta-one-hf-llava"

echo "[1/2] Checking model cache..."
if [ -d "$MODEL_CACHE" ]; then
    echo "      ✅ Model found — skipping download"
else
    echo "      ⬇  Downloading model (~7 GB, one time only)..."
    python3 -c "
from transformers import AutoProcessor, LlavaForConditionalGeneration
import torch, os
MODEL_ID = 'fancyfeast/llama-joycaption-beta-one-hf-llava'
CACHE    = '/workspace/hf_cache'
os.makedirs(CACHE, exist_ok=True)
print('      Downloading processor...')
AutoProcessor.from_pretrained(MODEL_ID, cache_dir=CACHE)
print('      Downloading model weights...')
LlavaForConditionalGeneration.from_pretrained(
    MODEL_ID, device_map='auto', torch_dtype=torch.bfloat16, cache_dir=CACHE)
print('      ✅ Done')
"
fi

echo ""
echo "[2/2] Starting SkyCaption on port 5000..."
echo ""
echo "╔══════════════════════════════════════════╗"
echo "║  ✅ Open port 5000 to access the UI      ║"
echo "╚══════════════════════════════════════════╝"
echo ""

exec python3 /app/app.py
