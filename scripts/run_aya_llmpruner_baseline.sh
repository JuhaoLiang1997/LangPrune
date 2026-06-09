#!/bin/bash
# LLM-Pruner baseline (language-agnostic): Aya-Expanse-8B, 70% sparsity
# Uses mixed-language calibration WITHOUT per-language importance aggregation.
# This is the baseline compared against Lang-Prune in Table 2.
#
# IMPORTANT: Set BASE_MODEL to your local Aya-Expanse-8B path:
#   export BASE_MODEL=/path/to/aya-expanse-8b
#   bash scripts/run_aya_llmpruner_baseline.sh

cd "$(dirname "$0")/.."

BASE_MODEL=${BASE_MODEL:-aya-expanse-8b}

echo "Using base model: $BASE_MODEL"

python main.py \
    --base_model "$BASE_MODEL" \
    --save_ckpt_log_name llmpruner_aya_sp70_mixed \
    --pruning_ratio 0.7 \
    --block_wise --global_pruning \
    --block_attention_layer_start 0 --block_attention_layer_end 32 \
    --block_mlp_layer_start 0 --block_mlp_layer_end 32 \
    --multi_lang_important False \
    --calibration_languages ar iw cs ru de en es id zh \
    --eval_languages ar iw cs ru de en es id zh \
    --multilingual_eval \
    --num_examples 900 \
    --test_after_train --save_model
