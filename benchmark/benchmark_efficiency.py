"""
Efficiency benchmark for Lang-Prune.
Reports: parameter count, MACs, inference latency, peak GPU memory.
Usage: python benchmark_efficiency.py --model_path <hf_model_dir> --label <name>
"""
import os
import sys
import json
import time
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(__file__))


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def benchmark_throughput(model, tokenizer, device, seq_len=128, n_runs=50, batch_size=1):
    """Measure tokens/sec throughput."""
    model.eval()
    inp = torch.randint(0, tokenizer.vocab_size, (batch_size, seq_len)).to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(5):
            model(inp)
    torch.cuda.synchronize()

    # Reset memory stats
    torch.cuda.reset_peak_memory_stats(device)

    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_runs):
            model(inp)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    total_tokens = n_runs * batch_size * seq_len
    throughput = total_tokens / elapsed  # tokens/sec
    latency_ms = (elapsed / n_runs) * 1000  # ms per forward pass
    peak_mem_gb = torch.cuda.max_memory_allocated(device) / 1024**3

    return {
        "throughput_tokens_per_sec": round(throughput, 1),
        "latency_ms_per_forward": round(latency_ms, 2),
        "peak_gpu_memory_gb": round(peak_mem_gb, 3),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to HF-format model directory")
    parser.add_argument("--label", type=str, default="model",
                        help="Label for this model in the output")
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--n_runs", type=int, default=50)
    parser.add_argument("--output_path", type=str, default="results/efficiency.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading model from {args.model_path} ...")

    # Pruned models must be loaded via torch.load because save_pretrained writes
    # the original config.json dimensions; from_pretrained would rebuild the
    # original architecture and mismatch the pruned weights.
    # Detect: if pytorch_model.bin exists in the parent dir (prune_log layout),
    # load that; otherwise treat as a standard HF model directory.
    model_path = args.model_path.rstrip("/")
    parent_dir = os.path.dirname(model_path)
    bin_path = os.path.join(parent_dir, "pytorch_model.bin")
    if not os.path.exists(bin_path) and model_path.endswith("hf_model"):
        # Also check if the bin is directly named differently
        bins = [f for f in os.listdir(parent_dir) if f.endswith(".bin") or f.endswith(".pt")]
        if bins:
            bin_path = os.path.join(parent_dir, bins[0])

    if os.path.exists(bin_path):
        print(f"  (detected pruned model — loading via torch.load from {bin_path})")
        ckpt = torch.load(bin_path, map_location="cpu")
        model = ckpt["model"].to(device).half()
        tokenizer = ckpt["tokenizer"]
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            dtype=torch.float16,
            trust_remote_code=True,
        ).to(device)
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model.eval()

    # 1. Parameter count
    n_params = count_parameters(model)
    print(f"Parameters: {n_params:,} ({n_params/1e9:.3f}B)")

    # 2. Throughput and memory
    print(f"Benchmarking throughput (seq_len={args.seq_len}, batch={args.batch_size}, runs={args.n_runs})...")
    perf = benchmark_throughput(model, tokenizer, device,
                                seq_len=args.seq_len,
                                n_runs=args.n_runs,
                                batch_size=args.batch_size)

    result = {
        "label": args.label,
        "model_path": args.model_path,
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "n_parameters": n_params,
        "n_parameters_B": round(n_params / 1e9, 3),
        **perf,
    }

    print(json.dumps(result, indent=2))

    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    # Append to existing results file if it exists
    all_results = []
    if os.path.exists(args.output_path):
        with open(args.output_path) as f:
            all_results = json.load(f)
    all_results.append(result)
    with open(args.output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved to {args.output_path}")


if __name__ == "__main__":
    main()
