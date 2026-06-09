#!/bin/bash
# Full reproduction script for Lang-Prune paper results.
#
# This script runs the main experiments from the paper:
#   Table 2:  Aya-Expanse-8B, 70% sparsity (Lang-Prune-Max vs LLM-Pruner baselines)
#   Table 22: Qwen3-8B, 50% sparsity (Calibration-language scaling)
#
# Prerequisites:
#   1. Install dependencies: pip install -r requirements.txt
#   2. Prepare mC4 data: see INSTALL.md
#   3. Download models: Aya-Expanse-8B, Qwen3-8B
#
# Usage:
#   # Run all experiments
#   bash scripts/run_reproduction.sh all
#
#   # Run only Aya experiments
#   bash scripts/run_reproduction.sh aya
#
#   # Run only Qwen experiments
#   bash scripts/run_reproduction.sh qwen
#
#   # Dry-run (print commands without executing)
#   bash scripts/run_reproduction.sh --dry-run aya
#
# Model paths (set before running):
#   export AYA_MODEL=/path/to/aya-expanse-8b
#   export QWEN_MODEL=/path/to/Qwen3-8B

set -e

DRY_RUN=false
RUN_AYA=false
RUN_QWEN=false

for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=true ;;
        all) RUN_AYA=true; RUN_QWEN=true ;;
        aya) RUN_AYA=true ;;
        qwen) RUN_QWEN=true ;;
    esac
done

# Default: run all if nothing specified
if [ "$RUN_AYA" = false ] && [ "$RUN_QWEN" = false ]; then
    RUN_AYA=true
    RUN_QWEN=true
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

echo "============================================"
echo "  Lang-Prune Paper Reproduction Suite"
echo "  Repo: $REPO_ROOT"
echo "============================================"

run_cmd() {
    local desc="$1"
    local cmd="$2"
    echo ""
    echo "=== $desc ==="
    echo "  \$ $cmd"
    if [ "$DRY_RUN" = false ]; then
        eval "$cmd" 2>&1 | tee -a "reproduction_$(date +%Y%m%d_%H%M%S).log"
        echo "  [OK] $desc completed."
    else
        echo "  [DRY-RUN] skipped."
    fi
}

collect_results() {
    echo ""
    echo "============================================"
    echo "  Results Summary"
    echo "============================================"
    python - <<'PYEOF'
import json, os, glob

LANGS_AYA = ["ar", "iw", "cs", "ru", "de", "en", "es", "id", "zh"]
LANGS_QWEN = ["ar", "iw", "cs", "ru", "de", "en", "es", "id", "zh",
              "fa", "uk", "pl", "nl", "fr", "ko", "sv", "da", "it",
              "pt", "my", "ja", "vi", "bg", "ur", "am"]

for run_dir in sorted(glob.glob("prune_log/*/")):
    ppl_file = os.path.join(run_dir, "ppl_results.json")
    if not os.path.exists(ppl_file):
        continue
    with open(ppl_file) as f:
        ppl = json.load(f)
    langs = [l for l in ppl if l in LANGS_QWEN]
    if not langs:
        continue
    avg = sum(ppl[l] for l in langs) / len(langs)
    print(f"\n{os.path.basename(run_dir.rstrip('/'))}:")
    print(f"  Avg PPL ({len(langs)} langs): {avg:.2f}")
    for l in sorted(langs):
        print(f"    {l}: {ppl[l]:.2f}")
PYEOF
}

# ============================================================================
# Aya-Expanse-8B experiments (Table 2)
# ============================================================================
if [ "$RUN_AYA" = true ]; then
    AYA_MODEL=${AYA_MODEL:-aya-expanse-8b}

    echo ""
    echo "--- Aya-Expanse-8B Experiments ---"
    echo "Model: $AYA_MODEL"

    # Check if model exists (local or HF hub)
    if [ ! -d "$AYA_MODEL" ] && [ "$AYA_MODEL" != "aya-expanse-8b" ]; then
        echo "WARNING: $AYA_MODEL not found. Attempting HuggingFace Hub..."
    fi

    AYA_COMMON="--block_wise --global_pruning \
        --block_attention_layer_start 0 --block_attention_layer_end 32 \
        --block_mlp_layer_start 0 --block_mlp_layer_end 32 \
        --calibration_languages ar iw cs ru de en es id zh \
        --eval_languages ar iw cs ru de en es id zh \
        --multilingual_eval --num_examples 100 \
        --test_after_train --save_model"

    # 1a. Lang-Prune-Max 70% (our method)
    run_cmd "Aya: Lang-Prune-Max 70%" \
        "python main.py --base_model '$AYA_MODEL' --save_ckpt_log_name langprune_aya_sp70_max \
            --pruning_ratio 0.7 --multi_lang_important True --merge_methods max \
            $AYA_COMMON"

    # 1b. LLM-Pruner baseline (mixed-language, 70%)
    run_cmd "Aya: LLM-Pruner mixed 70%" \
        "python main.py --base_model '$AYA_MODEL' --save_ckpt_log_name llmpruner_aya_sp70_mixed \
            --pruning_ratio 0.7 --multi_lang_important False \
            $AYA_COMMON --num_examples 900"

    # 1c. LLM-Pruner baseline (mixed-language, 50%)
    run_cmd "Aya: LLM-Pruner mixed 50%" \
        "python main.py --base_model '$AYA_MODEL' --save_ckpt_log_name llmpruner_aya_sp50_mixed \
            --pruning_ratio 0.5 --multi_lang_important False \
            $AYA_COMMON --num_examples 900"

    # 1d. LLM-Pruner baseline (mixed-language, 30%)
    run_cmd "Aya: LLM-Pruner mixed 30%" \
        "python main.py --base_model '$AYA_MODEL' --save_ckpt_log_name llmpruner_aya_sp30_mixed \
            --pruning_ratio 0.3 --multi_lang_important False \
            $AYA_COMMON --num_examples 900"

    # 1e. Lang-Prune-Max 50%
    run_cmd "Aya: Lang-Prune-Max 50%" \
        "python main.py --base_model '$AYA_MODEL' --save_ckpt_log_name langprune_aya_sp50_max \
            --pruning_ratio 0.5 --multi_lang_important True --merge_methods max \
            $AYA_COMMON"

    # 1f. Lang-Prune-Max 30%
    run_cmd "Aya: Lang-Prune-Max 30%" \
        "python main.py --base_model '$AYA_MODEL' --save_ckpt_log_name langprune_aya_sp30_max \
            --pruning_ratio 0.3 --multi_lang_important True --merge_methods max \
            $AYA_COMMON"

    # 1g. Unpruned baseline (PP only)
    run_cmd "Aya: Unpruned baseline (PPL evaluation only)" \
        "python main.py --base_model '$AYA_MODEL' --save_ckpt_log_name unpruned_aya_baseline \
            --pruning_ratio 0.0 \
            --eval_languages ar iw cs ru de en es id zh \
            --multilingual_eval --num_examples 100 \
            --test_before_train --test_before_train_then_end_exp"
fi

# ============================================================================
# Qwen3-8B experiments (Table 22)
# ============================================================================
if [ "$RUN_QWEN" = true ]; then
    QWEN_MODEL=${QWEN_MODEL:-/path/to/Qwen3-8B}

    echo ""
    echo "--- Qwen3-8B Experiments ---"
    echo "Model: $QWEN_MODEL"

    if [ ! -d "$QWEN_MODEL" ]; then
        if [ "$DRY_RUN" = false ]; then
            echo "ERROR: Qwen3-8B not found at $QWEN_MODEL"
            echo "Set QWEN_MODEL environment variable or edit this script."
            exit 1
        fi
    fi

    QWEN_COMMON="--block_wise --global_pruning \
        --block_attention_layer_start 0 --block_attention_layer_end 36 \
        --block_mlp_layer_start 0 --block_mlp_layer_end 36 \
        --calibration_languages ar iw cs ru de en es id zh \
        --multilingual_eval --num_examples 100 \
        --test_after_train --save_model"

    # 2a. Lang-Prune-Max 50% (25-language eval)
    run_cmd "Qwen: Lang-Prune-Max 50% (25 langs)" \
        "python main.py --base_model '$QWEN_MODEL' --save_ckpt_log_name langprune_qwen_sp50_max_25langs \
            --pruning_ratio 0.5 --multi_lang_important True --merge_methods max \
            --eval_languages ar iw cs ru de en es id zh fa uk pl nl fr ko sv da it pt my ja vi bg ur am \
            $QWEN_COMMON"

    # 2b. LLM-Pruner baseline (mixed, 50%)
    run_cmd "Qwen: LLM-Pruner mixed 50%" \
        "python main.py --base_model '$QWEN_MODEL' --save_ckpt_log_name llmpruner_qwen_sp50_mixed \
            --pruning_ratio 0.5 --multi_lang_important False \
            --eval_languages ar iw cs ru de en es id zh fa uk pl nl fr ko sv da it pt my ja vi bg ur am \
            --num_examples 900 \
            $QWEN_COMMON"
fi

# ============================================================================
# Results summary
# ============================================================================
if [ "$DRY_RUN" = false ]; then
    collect_results
else
    echo ""
    echo "[DRY-RUN] Reproduction commands printed above. Remove --dry-run to execute."
fi

echo ""
echo "============================================"
echo "  Reproduction complete!"
echo "  Results saved in prune_log/*/ppl_results.json"
echo "============================================"
