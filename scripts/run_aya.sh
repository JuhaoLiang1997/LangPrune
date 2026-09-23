#!/bin/bash
# Lang-Prune: Aya-Expanse-8B, 70% sparsity, Max aggregation
# Reproduces Table 2 (main text), row "Lang-Prune-Max 70%"
#
# Paper numbers (Table 2, mC4 validation PPL, 100 sequences per language):
#   ar=44.03, cs=69.71, de=64.85, en=158.33, es=72.53,
#   id=70.08, iw=51.62, ru=60.29, zh=46.18, avg=70.85
# Individual runs can vary by roughly ±10-15% with the random seed.
#
# Requirements: ~80GB GPU memory (A100/H100 recommended)
# Data: mC4 validation splits under ./data/c4/
#
# IMPORTANT: Set BASE_MODEL to your local Aya-Expanse-8B path:
#   export BASE_MODEL=/path/to/aya-expanse-8b
#   bash scripts/run_aya.sh

cd "$(dirname "$0")/.."

BASE_MODEL=${BASE_MODEL:-aya-expanse-8b}

echo "Using base model: $BASE_MODEL"
if [ ! -d "$BASE_MODEL" ] && [ "$BASE_MODEL" != "aya-expanse-8b" ]; then
    echo "WARNING: Model directory not found: $BASE_MODEL"
    echo "Attempting to load from HuggingFace Hub..."
fi

python main.py \
    --base_model "$BASE_MODEL" \
    --save_ckpt_log_name langprune_aya_sp70_max \
    --pruning_ratio 0.7 \
    --block_wise --global_pruning \
    --block_attention_layer_start 0 --block_attention_layer_end 32 \
    --block_mlp_layer_start 0 --block_mlp_layer_end 32 \
    --multi_lang_important True --merge_methods max \
    --calibration_languages ar iw cs ru de en es id zh \
    --eval_languages ar iw cs ru de en es id zh \
    --multilingual_eval \
    --num_examples 100 \
    --test_after_train --save_model
