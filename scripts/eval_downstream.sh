#!/bin/bash
# Downstream zero-shot evaluation using lm-evaluation-harness.
# Supports Belebele, Global-MMLU, and standard MMLU tasks.
#
# Prerequisites:
#   pip install lm-eval>=0.4.0
#
# Usage:
#   # Evaluate an unpruned model
#   bash scripts/eval_downstream.sh /path/to/model unpruned
#
#   # Evaluate a pruned model (uses the hf_model/ dir saved by main.py)
#   bash scripts/eval_downstream.sh prune_log/langprune_aya_sp70_max/hf_model langprune_sp70

set -e

MODEL_PATH=${1:?Please provide model path as first argument}
OUTPUT_NAME=${2:?Please provide output name as second argument}
TASK_SET=${3:-belebele}  # belebele, global_mmlu, or mmlu

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
mkdir -p "${REPO_ROOT}/results"

echo "=== Downstream Evaluation ==="
echo "Model:     $MODEL_PATH"
echo "Output:    $OUTPUT_NAME"
echo "Task set:  $TASK_SET"

# ---- Detect pruned vs unpruned model ----
PARENT_DIR="$(dirname "$(realpath "$MODEL_PATH")")"
TORCH_BIN="${PARENT_DIR}/pytorch_model.bin"

if [ -f "$TORCH_BIN" ]; then
    echo "Detected pruned model checkpoint: $TORCH_BIN"
    MODEL_ARGS="checkpoint=${TORCH_BIN},config_pretrained=${MODEL_PATH},trust_remote_code=True"
else
    echo "Using standard HuggingFace model"
    MODEL_ARGS="pretrained=${MODEL_PATH},dtype=float16,trust_remote_code=True"
fi

# ---- Task definitions ----
case "$TASK_SET" in
    belebele)
        TASKS="belebele_arb_Arab,belebele_heb_Hebr,belebele_ces_Latn,belebele_rus_Cyrl,belebele_deu_Latn,belebele_spa_Latn,belebele_ind_Latn,belebele_zho_Hans"
        NUM_FEWSHOT=3
        OUTPUT_FILE="${REPO_ROOT}/results/belebele_${OUTPUT_NAME}.json"
        ;;
    global_mmlu)
        TASKS="global_mmlu_ar,global_mmlu_cs,global_mmlu_ru,global_mmlu_de,global_mmlu_es,global_mmlu_id,global_mmlu_zh"
        NUM_FEWSHOT=3
        OUTPUT_FILE="${REPO_ROOT}/results/global_mmlu_${OUTPUT_NAME}.json"
        ;;
    mmlu)
        TASKS="mmlu"
        NUM_FEWSHOT=5
        OUTPUT_FILE="${REPO_ROOT}/results/mmlu_${OUTPUT_NAME}.json"
        ;;
    *)
        echo "Unknown task set: $TASK_SET (choose: belebele, global_mmlu, mmlu)"
        exit 1
        ;;
esac

echo "Tasks: $TASKS"

lm_eval \
    --model hf \
    --model_args "${MODEL_ARGS}" \
    --tasks "$TASKS" \
    --num_fewshot "$NUM_FEWSHOT" \
    --batch_size 8 \
    --output_path "$OUTPUT_FILE" \
    --no_cache

echo "=== Evaluation complete ==="
echo "Results saved to: $OUTPUT_FILE"
