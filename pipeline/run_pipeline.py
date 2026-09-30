"""
CLI entry point: runs BM25 (baseline vs refined), Matryoshka dense retrieval
(base model at several truncation dims, fine-tuned model, instruction-prefix
A/B), and RRF fusion (k sweep + weighted sweep) over every synthetic hirer
against every synthetic provider, dumping ranked results to
pipeline/results/*.json for notebooks/evaluation.ipynb to consume.

Keeps the slow steps (embedding all provider/hirer text, once per model
variant) to a single pass each here, so the notebook's metric/tuning sweeps
stay fast.

Run: python3 pipeline/run_pipeline.py
"""
import argparse
import json
import time
from pathlib import Path

from corpus import provider_text, hirer_text
from retrieval_bm25 import BM25Retriever
from retrieval_dense import (
    encode_docs, encode_queries, rank_from_raw,
    BASE_MODEL_NAME, QUERY_PREFIX_GENERIC, QUERY_PREFIX_DOMAIN,
)
from retrieval_rrf import rrf_fuse

BASE = Path(__file__).parent
DATA_DIR = BASE / "data"
RESULTS_DIR = BASE / "results"
MODELS_DIR = BASE / "models"


def set_dataset(tag: str):
    """Namespace data/results/models by dataset so a data_sat run never
    overwrites the frozen synthetic-data baseline in results/."""
    global DATA_DIR, RESULTS_DIR, MODELS_DIR, FINETUNED_MODEL_PATH
    DATA_DIR = BASE / tag
    if tag == "data":
        RESULTS_DIR, MODELS_DIR = BASE / "results", BASE / "models"
    else:
        RESULTS_DIR, MODELS_DIR = BASE / f"results_{tag}", BASE / f"models_{tag}"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FINETUNED_MODEL_PATH = str(MODELS_DIR / "mxbai-finetuned")


FINETUNED_MODEL_PATH = str(BASE / "models" / "mxbai-finetuned")

TRUNCATE_DIMS = [None, 512, 256, 128, 64]
RRF_KS = [10, 60, 100]
RRF_WEIGHT_COMBOS = [(1.0, 1.0), (1.0, 1.5), (0.7, 1.3), (1.5, 1.0)]  # (sparse_weight, dense_weight)


def load_json(name):
    return json.loads((DATA_DIR / name).read_text())


def ranked_ids_only(scored: list[tuple[int, float]]) -> list[int]:
    return [pid for pid, _score in scored]


def run_dense_model(label, providers, hirers, provider_ids, model_name, dims, query_prefix=QUERY_PREFIX_GENERIC):
    print(f"Embedding with {label} ({model_name})...")
    t0 = time.time()
    doc_raw = encode_docs([provider_text(p) for p in providers], model_name=model_name)
    query_raw = encode_queries([hirer_text(h) for h in hirers], model_name=model_name, query_prefix=query_prefix)
    print(f"  embedding done in {time.time() - t0:.1f}s")

    scored_by_dim = {}
    for dim in dims:
        dim_label = str(dim) if dim is not None else "full"
        results = {}
        scored_by_hirer = {}
        for i, h in enumerate(hirers):
            scored = rank_from_raw(doc_raw, query_raw[i], provider_ids, truncate_dim=dim)
            scored_by_hirer[h["hire_id"]] = scored
            results[str(h["hire_id"])] = ranked_ids_only(scored)
        scored_by_dim[dim_label] = scored_by_hirer
        out_name = f"dense_{label}_{dim_label}.json" if label != "base" else f"dense_{dim_label}.json"
        (RESULTS_DIR / out_name).write_text(json.dumps(results, indent=2))
        print(f"  {out_name} written")
    return scored_by_dim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data",
                    help="dataset folder under pipeline/ (e.g. data_sat); outputs are namespaced accordingly")
    args = ap.parse_args()
    set_dataset(args.data_dir)

    providers = load_json("providers.json")
    hirers = load_json("hirers.json")
    provider_ids = [p["provider_id"] for p in providers]

    # ---------------- BM25: baseline vs refined ----------------
    print("Running BM25 (baseline)...")
    t0 = time.time()
    bm25_baseline = BM25Retriever(providers, refined=False)
    bm25_baseline_results = {}
    for h in hirers:
        scored = bm25_baseline.rank(hirer_text(h))
        bm25_baseline_results[str(h["hire_id"])] = ranked_ids_only(scored)
    (RESULTS_DIR / "bm25_baseline.json").write_text(json.dumps(bm25_baseline_results, indent=2))
    print(f"  done in {time.time() - t0:.1f}s")

    print("Running BM25 (refined: stemming + field weighting + synonyms)...")
    t0 = time.time()
    bm25_refined = BM25Retriever(providers, refined=True)
    bm25_refined_results = {}
    bm25_refined_scored_cache = {}
    for h in hirers:
        scored = bm25_refined.rank(hirer_text(h), query_title=h["hire_title"])
        bm25_refined_scored_cache[h["hire_id"]] = scored
        bm25_refined_results[str(h["hire_id"])] = ranked_ids_only(scored)
    (RESULTS_DIR / "bm25_refined.json").write_text(json.dumps(bm25_refined_results, indent=2))
    (RESULTS_DIR / "bm25.json").write_text(json.dumps(bm25_refined_results, indent=2))  # keep the plain name pointed at the better variant
    print(f"  done in {time.time() - t0:.1f}s")

    # ---------------- Dense: base model, Matryoshka truncation sweep ----------------
    base_scored_by_dim = run_dense_model(
        "base", providers, hirers, provider_ids, BASE_MODEL_NAME, TRUNCATE_DIMS,
    )
    full_dense_scored = base_scored_by_dim["full"]

    # ---------------- Dense: instruction-prefix A/B (base model, full dim) ----------------
    print("Embedding with domain-tuned instruction prefix (base model, full dim)...")
    t0 = time.time()
    doc_raw = encode_docs([provider_text(p) for p in providers], model_name=BASE_MODEL_NAME)
    query_raw_domain = encode_queries(
        [hirer_text(h) for h in hirers], model_name=BASE_MODEL_NAME, query_prefix=QUERY_PREFIX_DOMAIN,
    )
    domain_prefix_results = {}
    for i, h in enumerate(hirers):
        scored = rank_from_raw(doc_raw, query_raw_domain[i], provider_ids, truncate_dim=None)
        domain_prefix_results[str(h["hire_id"])] = ranked_ids_only(scored)
    (RESULTS_DIR / "dense_domainprefix_full.json").write_text(json.dumps(domain_prefix_results, indent=2))
    print(f"  done in {time.time() - t0:.1f}s")

    # ---------------- Dense: fine-tuned model (full + 256-dim default) ----------------
    finetuned_path = Path(FINETUNED_MODEL_PATH)
    if finetuned_path.exists():
        run_dense_model(
            "finetuned", providers, hirers, provider_ids, FINETUNED_MODEL_PATH, [None, 256],
        )
    else:
        print(f"Skipping fine-tuned dense run -- model not found at {FINETUNED_MODEL_PATH} "
              f"(run pipeline/finetune_embeddings.py first)")

    # ---------------- RRF: k sweep (refined BM25 + base full-dim dense) ----------------
    print("Running RRF k sweep...")
    for k in RRF_KS:
        rrf_results = {}
        for h in hirers:
            fused = rrf_fuse(bm25_refined_scored_cache[h["hire_id"]], full_dense_scored[h["hire_id"]], k=k)
            rrf_results[str(h["hire_id"])] = ranked_ids_only(fused)
        (RESULTS_DIR / f"rrf_k{k}.json").write_text(json.dumps(rrf_results, indent=2))
        print(f"  rrf[k={k}] written")

    # ---------------- RRF: weighted sweep (fixed k=60) ----------------
    print("Running weighted RRF sweep...")
    for sw, dw in RRF_WEIGHT_COMBOS:
        rrf_results = {}
        for h in hirers:
            fused = rrf_fuse(
                bm25_refined_scored_cache[h["hire_id"]], full_dense_scored[h["hire_id"]],
                k=60, sparse_weight=sw, dense_weight=dw,
            )
            rrf_results[str(h["hire_id"])] = ranked_ids_only(fused)
        (RESULTS_DIR / f"rrf_w_sw{sw}_dw{dw}.json").write_text(json.dumps(rrf_results, indent=2))
        print(f"  rrf[sparse_weight={sw}, dense_weight={dw}] written")

    print("\nAll results written to pipeline/results/")


if __name__ == "__main__":
    main()
