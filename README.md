# Lang-Prune

Structured pruning for multilingual LLMs that doesn't break low-resource languages.

Standard pruning with mixed-language data often works fine on English but causes catastrophic failures elsewhere (14× PPL increase on Chinese at 70% sparsity). Lang-Prune fixes this by computing importance scores **per language** and keeping anything that's critical to **any** language.

## How it works

Instead of mixing all calibration data together, Lang-Prune does three things:

```
1. Per-language importance
   ┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐
   │ ar  │  │ en  │  │ zh  │  │ ... │    Taylor importance
   │imp  │  │imp  │  │imp  │  │     │    computed independently
   └──┬──┘  └──┬──┘  └──┬──┘  └──┬──┘    for each language
      │        │        │        │
      └────────┴───────┬┴────────┘
                       │
2. Max aggregation     ▼
              torch.max(imp_ar, imp_en, imp_zh, ...)
              "keep what any language needs"
                       │
3. Global pruning      ▼
              Prune lowest-importance structures
              until target sparsity is reached
```

**Why max?**  A weight that's irrelevant for English might be critical for Arabic script rendering.  Mean aggregation averages away these signals; max preserves them.

The pipeline extends [LLM-Pruner](https://github.com/horseee/LLM-Pruner) — we only modified the importance estimation and aggregation stages. The dependency graph construction and pruning mechanics are unchanged.

## Results

### Aya-Expanse-8B @ 70% structured sparsity

| Language | Unpruned | LLM-Pruner (mixed) | Lang-Prune (max) |
|----------|----------|-------------------|-------------------|
| ar | 8.1 | 62.2 | 35.0 |
| iw | 10.5 | 63.8 | 51.6 |
| cs | 10.7 | 115.0 | 58.1 |
| ru | 10.5 | 81.6 | 57.1 |
| de | 9.6 | 121.1 | 62.4 |
| en | 12.8 | 214.8 | 171.7 |
| es | 11.2 | 113.8 | 78.7 |
| id | 11.1 | 118.1 | 48.5 |
| zh | 10.3 | 133.2 | 43.5 |
| **avg** | **10.5** | **113.7** | **67.4** |

Results from seed=42, 100 calibration examples per language. Exact values vary ±10-15% across seeds, but the ranking and relative improvements are stable. The paper reports avg=70.85 from a different seed.

Key takeaways:
- Lang-Prune achieves **41% lower PPL** than LLM-Pruner at the same sparsity
- No language experiences catastrophic failure (worst case: en at 171.7 vs 214.8)
- The improvement is largest on languages that mixed-data pruning hurts most (cs, id, zh)

## Quick start

```bash
# 1. Environment
conda create -n langprune python=3.10 -y && conda activate langprune
pip install torch==2.3.0  # CUDA 12.1 or 11.8, see INSTALL.md
pip install -r requirements.txt

# 2. Prepare mC4 data (see Data Preparation below)

# 3. Prune Aya-Expanse-8B at 70% sparsity
export BASE_MODEL=/path/to/aya-expanse-8b
bash scripts/run_aya.sh
```

## Installation

See [INSTALL.md](INSTALL.md) for detailed instructions including PyTorch setup, data preparation, and model download.

### Data preparation

Lang-Prune uses language-specific mC4 validation splits. The expected structure:

```
data/c4/
├── multilingual/c4-ar.tfrecord-00000-of-01024.json.gz
├── multilingual/c4-ar-validation.tfrecord-00000-of-00004.json.gz
├── multilingual/c4-zh.tfrecord-00000-of-01024.json.gz
├── multilingual/c4-zh-validation.tfrecord-00000-of-00002.json.gz
├── en/c4-train.00000-of-01024.json.gz
├── en/c4-validation.00000-of-00008.json.gz
└── ...
```

Language-to-file mappings are in `lib/datasets/languages.py`. 25 languages are supported (see below).

The data files are standard mC4 TFRecord JSON files. You can extract them from the [mC4 dataset](https://huggingface.co/datasets/mc4) on HuggingFace, filtering by language code.

## Running experiments

### Lang-Prune (our method)

```bash
python main.py \
    --base_model /path/to/model \
    --save_ckpt_log_name my_experiment \
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
```

### LLM-Pruner baseline (mixed-data)

Same as above but with `--multi_lang_important False`. This pools all calibration data together before importance estimation, which is the standard approach in prior work.

### Convenience scripts

```bash
bash scripts/run_aya.sh                      # Lang-Prune on Aya-Expanse-8B, 70% sparsity
bash scripts/run_aya_llmpruner_baseline.sh   # LLM-Pruner baseline
bash scripts/run_qwen.sh                     # Lang-Prune on Qwen3-8B, 50% sparsity
bash scripts/run_reproduction.sh all         # Full paper reproduction suite
```

The scripts use `$BASE_MODEL` environment variable to locate the model. Set it or edit the scripts.

### Key arguments

| Argument | What it does |
|----------|-------------|
| `--multi_lang_important True` | Enable per-language importance (this is Lang-Prune) |
| `--merge_methods max` | Max aggregation (default, recommended) |
| `--merge_methods mean` | Average — equivalent to language-agnostic pruning |
| `--pruning_ratio 0.7` | Target 70% channel sparsity |
| `--block_wise --global_pruning` | Block-wise structured pruning with global ranking |
| `--test_after_train` | Evaluate PPL after pruning |
| `--save_model` | Export pruned model to HuggingFace format |

### Unpruned baseline

To evaluate the unpruned model's PPL for comparison:

```bash
python main.py \
    --base_model /path/to/model \
    --save_ckpt_log_name unpruned_baseline \
    --eval_languages ar iw cs ru de en es id zh \
    --multilingual_eval \
    --test_before_train --test_before_train_then_end_exp
```

## Post-training with LoRA

After pruning, a small amount of LoRA fine-tuning can recover some of the performance loss:

```bash
python post_training/post_training.py \
    --prune_model prune_log/my_experiment/pytorch_model.bin \
    --data_path /path/to/alpaca_data.json \
    --output_dir ./lora_checkpoint \
    --lora_r 16 --lora_alpha 32 \
    --batch_size 128 --micro_batch_size 4
```

## Downstream evaluation

For zero-shot tasks (Belebele, Global-MMLU), we use [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness):

```bash
pip install lm-eval>=0.4.0

# Belebele reading comprehension (8 languages)
bash scripts/eval_downstream.sh /path/to/hf_model my_run belebele

# Global-MMLU (7 languages)
bash scripts/eval_downstream.sh /path/to/hf_model my_run global_mmlu
```

## Efficiency benchmarking

```bash
python benchmark/benchmark_efficiency.py \
    --model_path prune_log/my_experiment/hf_model \
    --label my_experiment \
    --seq_len 128 --batch_size 1 \
    --output_path results/efficiency.json
```

Reports parameter count, MACs, inference latency, and peak GPU memory.

## Supported models

| Architecture | Models | Layers | Notes |
|-------------|--------|--------|-------|
| Cohere | Aya-Expanse-8B, Aya-Expanse-32B | 32 | Primary testbed |
| Qwen3 | Qwen3-8B (36L), Qwen3-14B (40L), Qwen3-32B | varies | Requires trust_remote_code |
| LLaMA | Llama-2-7B/13B, Llama-3-8B | 32 | Via the HF LLaMA adapter in `lib/models/hf_llama/` |

## Supported languages (25)

**Calibration (9):** Arabic (ar), Hebrew (iw), Czech (cs), Russian (ru), German (de), English (en), Spanish (es), Indonesian (id), Chinese (zh)

**Out-of-distribution evaluation (16):** Persian (fa), Urdu (ur), Amharic (am), Bulgarian (bg), Ukrainian (uk), Polish (pl), Dutch (nl), Swedish (sv), Danish (da), French (fr), Italian (it), Portuguese (pt), Burmese (my), Japanese (ja), Korean (ko), Vietnamese (vi)

All 25 languages are available for both calibration and evaluation. The OOD languages are never seen during importance estimation.

## Hardware requirements

- ~80GB GPU memory for Aya-Expanse-8B at 70% sparsity (A100/H100 recommended)
- ~40GB for Qwen3-8B
- One pruning run takes roughly 1 GPU-hour on an A100
- Multiple GPUs are supported via device_map="auto" in model loading

## Repository structure

```
LangPrune/
├── main.py                  # Entry point: pruning + PPL evaluation
├── lib/
│   ├── torch_pruning/       # Structured pruning engine (dependency graph, importance)
│   ├── pruner/              # Model-specific pruners (LLaMA, Qwen3 attention, RMSNorm)
│   ├── datasets/            # mC4 multilingual data loading
│   ├── models/              # HF LLaMA model definition (for architecture handling)
│   ├── peft/                # LoRA implementation (from PEFT)
│   ├── evaluator/           # Perplexity computation
│   ├── templates/           # Generation prompts
│   └── utils/               # Logger, prompter utilities
├── scripts/                 # Experiment scripts
├── post_training/           # LoRA post-training recovery
├── benchmark/               # Efficiency benchmarking (MACs, latency, memory)
├── utils/                   # Model export utilities
├── prune_log/               # Output directory (experiment results, .gitignored)
├── data/                    # mC4 data directory (.gitignored)
└── .cache/                  # HuggingFace dataset cache (.gitignored)
```

## Known limitations

- Evaluated primarily in one-shot pruning without extensive post-training recovery. LoRA helps but doesn't fully close the gap.
- Effectiveness depends on structured pruning paradigms — not applicable to unstructured methods (Wanda, SparseGPT).
- The calibration data quality matters: if a language has poor mC4 coverage, importance estimates will be noisy.
- Results vary by ±10-15% across random seeds. This is inherent to importance-estimation-based pruning.

## Citation

```bibtex
@inproceedings{langprune2026,
  title     = {Lang-Prune: Rethinking Pruning for Fairness and Efficiency in Multilingual LLMs},
  author    = {},
  booktitle = {Proceedings of the Conference on Language Modeling (COLM)},
  year      = {2026}
}
```

## License

MIT. See [LICENSE](LICENSE).

## Acknowledgments

Built on [LLM-Pruner](https://github.com/horseee/LLM-Pruner) and [Torch-Pruning](https://github.com/VainF/Torch-Pruning). The PEFT module is adapted from [huggingface/peft](https://github.com/huggingface/peft).
