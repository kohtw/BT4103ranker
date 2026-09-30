#!/usr/bin/env python3
"""Weighted reciprocal-rank fusion of two or more ranked-list files.

Why this exists: on data_sat the Stage-2 ranker and its predecessors are strong
on *different* metrics, and fusing their ranked lists beats either alone.

    system (1,023 queries)  NDCG@10    P@5     R@10     MRR
    noce (exp gain, raw)     0.6579   0.1533   0.8603   0.3375
    linz (linear + zscore)   0.6851   0.1413   0.8539   0.3331
    fuse w=0.7               0.6844   0.1462   0.8575   0.3413

fuse w=0.7 vs linz: NDCG@10 -0.0007 [-0.0034, +0.0020] (no loss),
P@5 +0.0049 [+0.0027, +0.0072] and MRR +0.0082 [+0.0040, +0.0126] (both
improved). It keeps the whole NDCG@10 gain over the old ranker while giving
back ~40% of the P@5 that the gain alignment costs.

RRF is used rather than score averaging because only ranked lists are persisted
by rerank_ltr.py, not the underlying scores.

Usage:
    python fuse_rankers.py --data-dir data_sat \\
        --inputs linz noce --weights 0.7 0.3 --out fused.json
"""
import argparse
import json
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
from evaluate import ndcg_at_k, precision_at_k, recall_at_k, reciprocal_rank  # noqa: E402

METRICS = [("NDCG@10", ndcg_at_k, 10), ("P@5", precision_at_k, 5),
           ("R@10", recall_at_k, 10), ("MRR", reciprocal_rank, None)]


def rrf_fuse(ranked_lists, weights, k=60):
    """weighted RRF: score(d) = sum_i w_i / (k + rank_i(d)), rank starting at 1."""
    fused = {}
    queries = ranked_lists[0].keys()
    for q in queries:
        scores = {}
        for lst, w in zip(ranked_lists, weights):
            for rank, pid in enumerate(lst.get(q, [])):
                scores[pid] = scores.get(pid, 0.0) + w / (k + rank + 1)
        fused[q] = [pid for pid, _ in sorted(scores.items(), key=lambda kv: -kv[1])]
    return fused


def evaluate(ranked, gt):
    out = {n: {} for n, _, _ in METRICS}
    for q, ids in ranked.items():
        if q not in gt:
            continue
        ids = [int(p) for p in ids]
        for n, fn, k in METRICS:
            v = fn(ids, gt[q], k) if k else fn(ids, gt[q])
            if v is not None:
                out[n][q] = v
    return out


def print_row(name, ms):
    print(f"{name:<28} " + "  ".join(f"{n}={sum(ms[n].values())/len(ms[n]):.4f}"
                                     for n, _, _ in METRICS))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=None,
                    help="dataset folder under pipeline/ (e.g. data_sat) -- sets the results dir")
    ap.add_argument("--inputs", nargs="+", required=True,
                    help="ranked-list names (without .json) inside the results dir")
    ap.add_argument("--weights", nargs="+", type=float,
                    help="one weight per input; default is uniform")
    ap.add_argument("--k", type=int, default=60, help="RRF constant (default 60)")
    ap.add_argument("--out", default="fused.json", help="output file inside the results dir")
    args = ap.parse_args()

    tag = args.data_dir
    data_dir = BASE / (tag or "data")
    results_dir = BASE / ("results" if tag in (None, "", "data") else f"results_{tag}")
    gt = json.loads((data_dir / "ground_truth_llm.json").read_text())

    weights = args.weights or [1.0] * len(args.inputs)
    if len(weights) != len(args.inputs):
        raise SystemExit(f"--weights has {len(weights)} entries for {len(args.inputs)} inputs")
    total = sum(weights)
    weights = [w / total for w in weights]

    lists = []
    for name in args.inputs:
        path = results_dir / f"{name}.json"
        if not path.exists():
            raise SystemExit(f"{path} not found")
        lists.append(json.loads(path.read_text()))

    print(f"inputs: {', '.join(f'{n} (w={w:.2f})' for n, w in zip(args.inputs, weights))}")
    for name, lst in zip(args.inputs, lists):
        print_row(name, evaluate(lst, gt))

    fused = rrf_fuse(lists, weights, k=args.k)
    print_row("fused", evaluate(fused, gt))
    out_path = results_dir / args.out
    out_path.write_text(json.dumps(fused))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
