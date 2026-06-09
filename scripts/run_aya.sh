#!/bin/bash
# Lang-Prune: Aya-Expanse-8B, 70% sparsity, Max aggregation
# Reproduces Table 2 (main text), row "Lang-Prune-Max 70%"
#
# Expected output (mC4 validation PPL, seed=42, 100 examples):
#   ar≈35, iw≈52, cs≈58, ru≈57, de≈62,
#   en≈172, es≈79, id≈49, zh≈43, avg≈67
# (Exact values vary ±10-15% by random seed. Paper reports avg=70.85.)
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
