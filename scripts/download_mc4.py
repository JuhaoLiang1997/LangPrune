#!/usr/bin/env python
"""Download the mC4 shards Lang-Prune expects into ./data/c4/.

The file list comes from lib/datasets/languages.py, and the files are fetched
from the allenai/c4 dataset on the Hugging Face Hub, keeping the same relative
paths (multilingual/..., en/...), so the loader finds them without changes.

Examples:
    python scripts/download_mc4.py                       # 9 reference languages, train + validation
    python scripts/download_mc4.py --languages ar zh     # a subset
    python scripts/download_mc4.py --all --splits validation   # all languages, evaluation shards only

Note: each train shard is a few hundred MB; validation shards are smaller.
"""
import argparse
import importlib.util
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORE_LANGUAGES = ["ar", "iw", "cs", "ru", "de", "en", "es", "id", "zh"]


def load_language_table():
    # Load languages.py directly so this script does not need torch installed.
    path = os.path.join(ROOT, "lib", "datasets", "languages.py")
    spec = importlib.util.spec_from_file_location("languages", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.c4_multilingual_data


def main():
    table = load_language_table()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--languages", nargs="+", default=CORE_LANGUAGES, help="language codes from lib/datasets/languages.py")
    parser.add_argument("--all", action="store_true", help="download every language in languages.py")
    parser.add_argument("--splits", nargs="+", default=["train", "validation"], choices=["train", "validation"])
    parser.add_argument("--out", default=os.path.join(ROOT, "data", "c4"), help="output directory (default: ./data/c4)")
    args = parser.parse_args()

    from huggingface_hub import hf_hub_download

    languages = list(table) if args.all else args.languages
    unknown = [lang for lang in languages if lang not in table]
    if unknown:
        parser.error(f"unknown language code(s): {unknown}. Available: {sorted(table)}")

    for lang in languages:
        for split in args.splits:
            filename = table[lang][split]
            target = os.path.join(args.out, filename)
            if os.path.exists(target):
                print(f"[skip] {filename}")
                continue
            print(f"[get ] {filename}")
            hf_hub_download(repo_id="allenai/c4", repo_type="dataset", filename=filename, local_dir=args.out)
    print(f"Done. Files are under {args.out}")


if __name__ == "__main__":
    main()
