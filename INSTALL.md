# Installation

## 1. Environment

```bash
conda create -n langprune python=3.10 -y
conda activate langprune
```

## 2. PyTorch

Pick your CUDA version:

```bash
# CUDA 12.1
pip install torch==2.3.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# CUDA 11.8
pip install torch==2.3.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Torch 2.0+ should work but we've tested with 2.3.0.

## 3. Python dependencies

```bash
pip install -r requirements.txt
```

Or install the package in development mode:

```bash
pip install -e .
```

## 4. Download models

```bash
# Aya-Expanse-8B
huggingface-cli download CohereForAI/aya-expanse-8b --local-dir ./models/aya-expanse-8b

# Qwen3-8B
huggingface-cli download Qwen/Qwen3-8B --local-dir ./models/Qwen3-8B
```

Or let the code download from HuggingFace Hub automatically by passing the model ID (e.g. `--base_model CohereForAI/aya-expanse-8b`).

## 5. Prepare mC4 data

Lang-Prune uses language-specific mC4 splits for calibration and evaluation. You need the validation splits for the 9 core languages (ar, iw, cs, ru, de, en, es, id, zh). For OOD evaluation, you also need the 16 additional languages.

### Option A: Download pre-split files

If you have access to pre-split mC4 files, place them under `./data/c4/` following the structure in `lib/datasets/languages.py`. Each language needs a train and validation file.

### Option B: Extract from HuggingFace

```python
# Example: extract Arabic mC4 validation
from datasets import load_dataset

# Load mC4 Arabic split
ds = load_dataset("mc4", "ar", split="validation", streaming=True)

# Save 1100 samples for PPL evaluation
samples = []
for i, example in enumerate(ds):
    if i >= 1100:
        break
    samples.append({"text": example["text"]})

import json
with open("./data/c4/multilingual/c4-ar-validation.json", "w") as f:
    for s in samples:
        f.write(json.dumps(s, ensure_ascii=False) + "\n")
```

The expected file paths follow this pattern (defined in `lib/datasets/languages.py`):

```
data/c4/
├── multilingual/
│   ├── c4-ar.tfrecord-00000-of-01024.json.gz
│   ├── c4-ar-validation.tfrecord-00000-of-00004.json.gz
│   ├── c4-zh.tfrecord-00000-of-01024.json.gz
│   ├── c4-zh-validation.tfrecord-00000-of-00002.json.gz
│   └── ...
├── en/
│   ├── c4-train.00000-of-01024.json.gz
│   └── c4-validation.00000-of-00008.json.gz
└── ...
```

The loader reads JSON Lines format (.json or .json.gz). Each line should be a JSON object with a `text` field.

## 6. Verify installation

```bash
# Test that the environment can import the library
python -c "import lib.torch_pruning; print('torch_pruning OK')"
python -c "from lib.datasets.loader import get_examples; print('datasets OK')"

# Test data loading (requires at least one language file)
python -c "
from transformers import AutoTokenizer
from lib.datasets.loader import get_examples
tokenizer = AutoTokenizer.from_pretrained('CohereForAI/aya-expanse-8b', trust_remote_code=True)
samples, _ = get_examples('c4', tokenizer, nsamples=5, seqlen=128, language='en')
print(f'Loaded {samples.shape[0]} samples, shape={samples.shape}')
"
```

## GPU memory requirements

| Model | Sparsity | Peak memory |
|-------|----------|-------------|
| Aya-Expanse-8B | 70% | ~80 GB |
| Aya-Expanse-8B | 50% | ~65 GB |
| Aya-Expanse-8B | 30% | ~55 GB |
| Qwen3-8B | 50% | ~45 GB |

A100 (80GB) or H100 is recommended for 70% sparsity. If you run out of memory, try reducing `--batch_size` or `--eval_batch_size`.
