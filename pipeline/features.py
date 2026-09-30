"""
Feature builder for Stage-2 re-ranking (cross-encoder + LambdaMART).

Produces, for each (hirer query, candidate provider) pair in the frozen RRF
top-K, the feature row a learned ranker needs -- text-retrieval signals
(bm25/dense/rrf scores and ranks) plus the *structured* signals that
embeddings structurally cannot see (budget fit, seniority fit, availability
immediacy). Writes plain CSVs so the ranker modules never need to re-embed.

Outputs (namespaced by --data-dir; shown here for the default "data")
-------
features/candidates_top{K}.csv : every top-K candidate for every hirer
features/train_pairs.csv       : the judged subset (label source for training)
cache/*.npy                    : cached provider/query embeddings (re-runnable)

Label policy
------------
`--label-policy judged` (default) keeps only pairs with a real LLM judgment
(grades 0-3). Unjudged pairs are NOT treated as negatives -- with only 3,769
of 13,520 pairs judged, silently scoring the rest as 0 injects label noise
that a GBDT will happily memorise. Pass `--label-policy all` to opt into that
(noisier) behaviour for a comparison run.

Train/serve skew warning
------------------------
budget_fit / seniority_fit / avail_immediacy need budget_lo, budget_hi,
seniority_needed, seniority, rate_per_hour, availability (and, for data_sat,
available_from / start_by). The synthetic corpus (--data-dir data) carries
these in a separate `_hirers_with_taxonomy.json` / `_providers_with_taxonomy.json`;
data_sat carries them inline on providers.json/hirers.json, and this script
falls back to those directly when no `_with_taxonomy.json` file exists.
Before shipping any ranker trained on them, confirm the production gig
payload actually carries these fields -- otherwise the model learns on
features that are absent (or constant) at serve time.

Run (from the repo root, in the fi-bench env):
    conda activate fi-bench
    python pipeline/features.py --top-k 50
    python pipeline/features.py --top-k 50 --data-dir data_sat
"""
import argparse
import csv
import json
from datetime import date
from pathlib import Path

import numpy as np

from corpus import provider_text, hirer_text
from retrieval_bm25 import BM25Retriever
from retrieval_dense import encode_docs, encode_queries, rank_from_raw, BASE_MODEL_NAME
from retrieval_rrf import rrf_fuse

BASE = Path(__file__).parent
DATA_DIR = BASE / "data"
FEAT_DIR = BASE / "features"
CACHE_DIR = BASE / "cache"


def set_dataset(tag: str):
    """Point every module-level path at `tag`'s data + a namespaced output
    tree, so a non-default dataset (e.g. data_sat) never overwrites the
    frozen synthetic-data baseline (features/, cache/) that RERANK_README.md
    documents."""
    global DATA_DIR, FEAT_DIR, CACHE_DIR
    DATA_DIR = BASE / tag
    if tag == "data":
        FEAT_DIR, CACHE_DIR = BASE / "features", BASE / "cache"
    else:
        FEAT_DIR, CACHE_DIR = BASE / f"features_{tag}", BASE / f"cache_{tag}"

SENIORITY_ORDER = {"mid": 0, "senior": 1, "expert": 2}

# Synthetic-data availability is free text; we only extract a coarse
# "how soon can they start" signal. Weak feature by design -- 6 distinct
# values over 104 providers -- the ranker is free to ignore it.
AVAIL_SOON = [
    ("immediately", 1.0),
    ("full-time capacity", 0.9),
    ("weekday evenings", 0.6),
    ("mondays to fridays", 0.5),
    ("3 days/week", 0.5),
    ("2 weeks' notice", 0.2),
]

AVAIL_DECAY_DAYS = 60.0  # data_sat: linear decay to 0 once availability trails start_by by this much


def load_json(name: str):
    return json.loads((DATA_DIR / name).read_text())


def budget_fit(budget_lo: float, budget_hi: float, rate: float) -> float:
    """1.0 when the provider's rate sits inside the hirer's band, decaying
    linearly outside it (normalised by the band width)."""
    lo, hi = float(budget_lo), float(budget_hi)
    if lo <= rate <= hi:
        return 1.0
    width = max(hi - lo, 1.0)
    gap = (lo - rate) if rate < lo else (rate - hi)
    return max(0.0, 1.0 - gap / width)


def seniority_fit(needed: str, has: str) -> float:
    """1.0 = exact match, 0.5 = one level off, 0.0 = two levels off."""
    a, b = SENIORITY_ORDER.get(needed), SENIORITY_ORDER.get(has)
    if a is None or b is None:
        return 0.5
    return 1.0 - abs(a - b) / 2.0


def _parse_avail_date(value: str | None, today: date) -> date | None:
    """data_sat uses the literal sentinels "now" (provider.available_from)
    and "asap" (hirer.start_by) instead of a date in ~45% / ~7% of records
    respectively -- both anchor to today's date. Anything else is ISO
    'YYYY-MM-DD' or unparseable (-> None, caller falls back to text)."""
    if value in ("now", "asap"):
        return today
    try:
        return date.fromisoformat(value) if value else None
    except ValueError:
        return None


def avail_immediacy(provider: dict, hirer: dict) -> float:
    """data_sat carries clean dates (`available_from` on the provider,
    `start_by` on the hirer) -- prefer those over the synthetic corpus's
    free-text `availability` field, whose fixed phrasings ("immediately",
    "3 days/week") never appear in data_sat's "Available from <date>, N
    days a week" strings and would otherwise silently collapse to the 0.5
    default for every real record."""
    today = date.today()
    d_avail = _parse_avail_date(provider.get("available_from"), today)
    d_start = _parse_avail_date(hirer.get("start_by"), today)
    if d_avail is not None and d_start is not None:
        gap_days = (d_avail - d_start).days
        return 1.0 if gap_days <= 0 else max(0.0, 1.0 - gap_days / AVAIL_DECAY_DAYS)
    t = (provider.get("availability") or "").lower()
    for needle, score in AVAIL_SOON:
        if needle in t:
            return score
    return 0.5


def grade_from_score(score: int) -> int:
    """ground_truth_llm.json stores 0/33/67/100; map back to the 0-3 grade."""
    return int(round(score * 3 / 100))


def dense_matrix(hirers, providers, model_name: str, dim, cache_tag: str):
    """Provider/query embeddings, cached to .npy so re-runs are free."""
    CACHE_DIR.mkdir(exist_ok=True)
    doc_path = CACHE_DIR / f"providers_{cache_tag}.npy"
    qry_path = CACHE_DIR / f"hirers_{cache_tag}.npy"
    if doc_path.exists() and qry_path.exists():
        print(f"  using cached embeddings ({cache_tag})")
        return np.load(doc_path), np.load(qry_path)
    print(f"  embedding {len(providers)} providers + {len(hirers)} hirers with {model_name} ...")
    docs = encode_docs([provider_text(p) for p in providers], model_name=model_name)
    queries = encode_queries([hirer_text(h) for h in hirers], model_name=model_name)
    np.save(doc_path, docs)
    np.save(qry_path, queries)
    return docs, queries


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-k", type=int, default=50,
                    help="candidate pool size per query (RRF top-K before re-ranking)")
    ap.add_argument("--dense-dim", type=int, default=1024,
                    help="Matryoshka truncation for the dense leg (1024 = full, matches the frozen baseline)")
    ap.add_argument("--dense-model", default=BASE_MODEL_NAME)
    ap.add_argument("--label-policy", choices=["judged", "all"], default="judged")
    ap.add_argument("--data-dir", default="data",
                    help="dataset folder under pipeline/ (e.g. data_sat); outputs are namespaced accordingly")
    args = ap.parse_args()
    set_dataset(args.data_dir)

    FEAT_DIR.mkdir(exist_ok=True)
    providers = load_json("providers.json")
    hirers = load_json("hirers.json")
    # Synthetic data keeps budget/seniority fields in a separate _with_taxonomy
    # file; data_sat already carries them inline on providers.json/hirers.json.
    prov_tax_path, hir_tax_path = DATA_DIR / "_providers_with_taxonomy.json", DATA_DIR / "_hirers_with_taxonomy.json"
    prov_tax = {p["provider_id"]: p for p in load_json("_providers_with_taxonomy.json")} \
        if prov_tax_path.exists() else {p["provider_id"]: p for p in providers}
    hir_tax = {h["hire_id"]: h for h in load_json("_hirers_with_taxonomy.json")} \
        if hir_tax_path.exists() else {h["hire_id"]: h for h in hirers}
    gt = load_json("ground_truth_llm.json")
    # Labels come from the RAW merged judgments (grades 0-3), not ground_truth_llm.json:
    # that file drops grade-0 pairs, so a ranker trained on it would have positives
    # only and no way to learn what to demote. gt stays for evaluation + diagnostics.
    grades = load_json("llm_judgments_merged.json")
    provider_ids = [p["provider_id"] for p in providers]

    print("Building sparse ranking (refined BM25) ...")
    bm25 = BM25Retriever(providers, refined=True)

    dim_tag = f"{args.dense_dim}"
    print(f"Building dense ranking (Matryoshka dim={dim_tag}) ...")
    doc_raw, query_raw = dense_matrix(hirers, providers, args.dense_model, args.dense_dim,
                                      cache_tag=f"{Path(args.dense_model).name}_{dim_tag}")

    rows = []
    skipped_out_of_pool = 0
    judged_in_pool = 0

    for i, h in enumerate(hirers):
        hid = str(h["hire_id"])
        bm25_ranked = bm25.rank(hirer_text(h), query_title=h["hire_title"])
        dense_ranked = rank_from_raw(doc_raw, query_raw[i], provider_ids, truncate_dim=args.dense_dim)
        fused = rrf_fuse(bm25_ranked, dense_ranked, k=60)

        bm25_score = dict(bm25_ranked)
        dense_score = dict(dense_ranked)
        bm25_pos = {pid: r + 1 for r, (pid, _) in enumerate(bm25_ranked)}
        dense_pos = {pid: r + 1 for r, (pid, _) in enumerate(dense_ranked)}

        gt_row = gt.get(hid, {})
        top = fused[: args.top_k]

        # how much of the genuinely relevant material is inside the pool?
        if gt_row:
            best_rank = {}
            for r, (pid, _) in enumerate(fused):
                best_rank[pid] = r + 1
            for pid_s, s in gt_row.items():
                if grade_from_score(s) >= 2 and best_rank.get(int(pid_s), 10 ** 9) > args.top_k:
                    skipped_out_of_pool += 1

        h_tax = hir_tax.get(h["hire_id"], {})
        grade_row = grades.get(hid, {})
        for rrf_pos, (pid, rrf_score) in enumerate(top, start=1):
            p_tax = prov_tax.get(pid, {})
            raw_grade = grade_row.get(str(pid))
            judged = raw_grade is not None
            if judged:
                judged_in_pool += 1
                label = int(raw_grade)
            elif args.label_policy == "all":
                label = 0          # documented trade-off: unjudged counted as negative
            else:
                label = ""         # excluded from training
            rows.append({
                "hire_id": hid,
                "provider_id": pid,
                "label": label,
                "judged": int(judged),
                "bm25_score": round(float(bm25_score.get(pid, 0.0)), 6),
                "bm25_rank": bm25_pos.get(pid, 0),
                "dense_cosine": round(float(dense_score.get(pid, 0.0)), 6),
                "dense_rank": dense_pos.get(pid, 0),
                "rrf_score": round(float(rrf_score), 8),
                "rrf_rank": rrf_pos,
                "budget_fit": round(budget_fit(h_tax.get("budget_lo", 0), h_tax.get("budget_hi", 0),
                                               p_tax.get("rate_per_hour", 0)), 4),
                "seniority_fit": round(seniority_fit(h_tax.get("seniority_needed", ""),
                                                     p_tax.get("seniority", "")), 4),
                "avail_immediacy": round(avail_immediacy(p_tax, h_tax), 4),
            })

    cand_path = FEAT_DIR / f"candidates_top{args.top_k}.csv"
    cols = list(rows[0].keys())
    with cand_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    train_rows = [r for r in rows if r["label"] != ""]
    train_path = FEAT_DIR / "train_pairs.csv"
    with train_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(train_rows)

    from collections import Counter
    dist = Counter(r["label"] for r in train_rows)
    n_unjudged_as_neg = sum(1 for r in train_rows if not r["judged"])
    print(f"\ncandidates_top{args.top_k}.csv : {len(rows)} rows ({len(hirers)} queries x <= {args.top_k})")
    print(f"train_pairs.csv               : {len(train_rows)} rows"
          + (f" (incl. {n_unjudged_as_neg} unjudged-as-negative, --label-policy all)" if n_unjudged_as_neg else ""))
    print(f"  label distribution (train)  : {dict(sorted(dist.items()))}")
    print(f"  judged pairs inside top-{args.top_k} : {judged_in_pool}")
    print(f"  grade>=2 pairs OUTSIDE top-{args.top_k} (recall loss, unfixable by re-ranking): {skipped_out_of_pool}")
    print("\nNext: score with the cross-encoder, then train the LTR ranker (see pipeline/RERANK_README.md)")


if __name__ == "__main__":
    main()
