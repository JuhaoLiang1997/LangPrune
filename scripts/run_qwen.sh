#!/bin/bash
# Lang-Prune: Qwen3-8B, 50% sparsity, Max aggregation
# Reproduces Table 22 (Appendix A.6.2), row "9-calib, all-25 avg"
#
# Expected output (mC4 validation PPL, 25 languages):
#   Core 9 avg ≈ 17.6, OOD 16 avg ≈ 13.5, All 25 avg ≈ 15.0
#
# Requirements: ~80GB GPU memory (A100/H100 recommended)
# Data: mC4 validation splits under ./data/c4/
#
# IMPORTANT: Set BASE_MODEL to your local Qwen3-8B path before running:
#   export BASE_MODEL=/path/to/Qwen3-8B
#   bash scripts/run_qwen.sh
#
# Or edit the default below.

cd "$(dirname "$0")/.."

# Set your Qwen3-8B path here, or use the environment variable
BASE_MODEL=${BASE_MODEL:-/path/to/Qwen3-8B}

echo "Using base model: $BASE_MODEL"
if [ ! -d "$BASE_MODEL" ]; then
    echo "ERROR: Model directory not found: $BASE_MODEL"
    echo "Please set BASE_MODEL environment variable or edit this script."
    echo "  export BASE_MODEL=/path/to/Qwen3-8B"
    exit 1
fi

python main.py \
    --base_model "$BASE_MODEL" \
    --save_ckpt_log_name langprune_qwen_sp50_max \
    --pruning_ratio 0.5 \
    --block_wise --global_pruning \
    --block_attention_layer_start 0 --block_attention_layer_end 36 \
    --block_mlp_layer_start 0 --block_mlp_layer_end 36 \
    --multi_lang_important True --merge_methods max \
    --calibration_languages ar iw cs ru de en es id zh \
    --eval_languages ar iw cs ru de en es id zh fa uk pl nl fr ko sv da it pt my ja vi bg ur am \
    --multilingual_eval \
    --num_examples 100 \
    --test_after_train --save_model
