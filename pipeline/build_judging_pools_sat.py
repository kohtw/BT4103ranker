"""
Builds the LLM judging pools for the Scrape-and-Tag corpus (pipeline/data_sat/,
from import_scrape_and_tag.py) -- the (gig, provider) pairs that get graded 0-3.

Same TREC-style pooling as the synthetic ground truth (see llm_judgments.py /
judging_pools_v2.json): judge the union of what several retrievers put near the
top, plus a random sample, rather than the full gigs x providers cross-product.
A reranker only ever sees Stage 1's candidates, so that's where labels matter;
random pairs are almost all obviously irrelevant and teach it little.

Per gig, the pool is the union of
    RRF (k=60) top --rrf-top        the candidates the reranker will actually re-order
    refined BM25 top --method-top   lexical matches, incl. keyword-overlap near-misses
    dense (mxbai, full dim) top --method-top   semantic matches
    --random providers outside the above   so the labels aren't only retrieval-shaped

Writes to pipeline/data_sat/:
    judging_pools.json   {hire_id: [provider_id, ...]}
    pool_sources.json    {hire_id: {provider_id: ["rrf", "bm25", ...]}}  -- audit trail

Embeddings are cached under pipeline/cache/ (git-ignored), keyed by a hash of each
text, so a rerun with other pool sizes doesn't re-embed, and a re-import only embeds
the texts that changed (never reuses a vector for different text).

Run: python pipeline/build_judging_pools_sat.py [--rrf-top 20 --method-top 10 --random 5]
"""
import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np

from corpus import hirer_text, provider_text
from retrieval_bm25 import BM25Retriever
from retrieval_dense import BASE_MODEL_NAME, encode_docs, encode_queries, rank_from_raw
from retrieval_rrf import rrf_fuse

BASE = Path(__file__).parent
DATA_DIR = BASE / "data_sat"
CACHE_DIR = BASE / "cache"


def embed_cached(name: str, texts: list[str], encode) -> np.ndarray:
    """Embeddings for `texts`, reusing any text already embedded under `name`."""
    vec_path, key_path = CACHE_DIR / f"{name}.npy", CACHE_DIR / f"{name}.keys.json"
    known = {}
    if vec_path.exists() and key_path.exists():
        known = dict(zip(json.loads(key_path.read_text()), np.load(vec_path)))
    keys = [hashlib.sha1(t.encode("utf-8")).hexdigest() for t in texts]
    todo = {k: t for k, t in zip(keys, texts) if k not in known}
    print(f"  {name}: {len(texts) - len(todo)} cached, {len(todo)} to embed", flush=True)
    if todo:
        known.update(zip(todo, encode(list(todo.values()))))
        CACHE_DIR.mkdir(exist_ok=True)
        np.save(vec_path, np.stack(list(known.values())))
        key_path.write_text(json.dumps(list(known)))
    return np.stack([known[k] for k in keys])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rrf-top", type=int, default=20)
    ap.add_argument("--method-top", type=int, default=10)
    ap.add_argument("--random", type=int, default=5)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--data-dir", type=Path, default=DATA_DIR)
    ap.add_argument("--embed-only", action="store_true", help="fill the embedding cache and exit")
    args = ap.parse_args()
    data_dir = args.data_dir

    hirers = json.loads((data_dir / "hirers.json").read_text(encoding="utf-8"))
    providers = json.loads((data_dir / "providers.json").read_text(encoding="utf-8"))
    pids = [p["provider_id"] for p in providers]
    print(f"{len(hirers)} gigs x {len(providers)} providers")

    bm25 = BM25Retriever(providers, refined=True)
    print(f"embedding with {BASE_MODEL_NAME} (CPU is slow the first time; cached after)...", flush=True)
    doc_raw = embed_cached("sat_docs_mxbai_by_text", [provider_text(p) for p in providers], encode_docs)
    q_raw = embed_cached("sat_queries_mxbai_by_text", [hirer_text(h) for h in hirers], encode_queries)
    if args.embed_only:
        return

    rng = random.Random(args.seed)
    pools, sources = {}, {}
    for i, h in enumerate(hirers):
        sparse = bm25.rank(hirer_text(h), query_title=h["hire_title"])
        dense = rank_from_raw(doc_raw, q_raw[i], pids)
        fused = rrf_fuse(sparse, dense, k=60)
        src = {}
        for tag, ranked, n in [("rrf", fused, args.rrf_top), ("bm25", sparse, args.method_top),
                               ("dense", dense, args.method_top)]:
            for pid, _ in ranked[:n]:
                src.setdefault(pid, []).append(tag)
        outside = [p for p in pids if p not in src]
        for pid in rng.sample(outside, args.random):
            src[pid] = ["random"]
        hid = str(h["hire_id"])
        pools[hid] = sorted(src)
        sources[hid] = {str(p): s for p, s in sorted(src.items())}

    (data_dir / "judging_pools.json").write_text(json.dumps(pools, indent=1), encoding="utf-8")
    (data_dir / "pool_sources.json").write_text(json.dumps(sources, indent=1), encoding="utf-8")

    sizes = [len(v) for v in pools.values()]
    print(f"pool size per gig: min {min(sizes)}, mean {np.mean(sizes):.1f}, max {max(sizes)}")
    print(f"total pairs to judge: {sum(sizes)}")
    print(f"written to {data_dir}")


if __name__ == "__main__":
    main()
