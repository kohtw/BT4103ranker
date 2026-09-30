"""
LambdaMART learning-to-rank re-ranker (Stage 2b) for the Senseigigs matching search.

This is the component that turns *non-text* signals into ordering: budget fit,
seniority fit and availability are invisible to any embedding, yet they are
exactly what separates the same-category-wrong-speciality profiles that clog
ranks 2-5.

Training is mandatory here -- there is no pretrained LambdaMART, the model IS
the training. It is cheap: gradient-boosted trees over ~1-3k judged rows and
130 queries trains in seconds on CPU.

Method
------
- candidates from `features/candidates_topK.csv` (frozen RRF pool)
- labels from the LLM judgments (0-3) -- never the retired taxonomy formula
- GroupKFold by hire_id: every query is held out exactly once, so the reported
  metrics are out-of-fold (OOF) and no query is ever scored by a model that
  saw it. With 130 queries, a random split would leak and flatter the result.
- optional ce_score column merged from `features/ce_scores_<tag>.csv`

Run (fi-bench env, from the repo root):
    python pipeline/rerank_ltr.py --mode ablation
    python pipeline/rerank_ltr.py --mode ablation --ce-tag minilm
    python pipeline/rerank_ltr.py --mode train --ce-tag minilm --export-scores
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import xgboost as xgb
from sklearn.model_selection import GroupKFold

from evaluate import evaluate_all_hirers

BASE = Path(__file__).parent
DATA_DIR = BASE / "data"
FEAT_DIR = BASE / "features"
RESULTS_DIR = BASE / "results"
MODELS_DIR = BASE / "models"


def set_dataset(tag: str):
    """Namespace every path by dataset so data_sat runs never touch the
    frozen synthetic-data baseline (features/, results/, models/)."""
    global DATA_DIR, FEAT_DIR, RESULTS_DIR, MODELS_DIR
    DATA_DIR = BASE / tag
    if tag == "data":
        FEAT_DIR, RESULTS_DIR, MODELS_DIR = BASE / "features", BASE / "results", BASE / "models"
    else:
        FEAT_DIR = BASE / f"features_{tag}"
        RESULTS_DIR = BASE / f"results_{tag}"
        MODELS_DIR = BASE / f"models_{tag}"


ID_COLS = {"hire_id", "provider_id", "label", "judged"}
# Ranking objective params, shared by the CV models and the final model so the
# two can never drift apart. `rank:ndcg` uses EXPONENTIAL gain (2^label - 1)
# by default, but evaluate.py scores NDCG with LINEAR gain on the 0-100
# relevance scores. Exponential gain treats grade-3 as 7x grade-1; linear
# treats it as 3x -- so by default the model optimises a different ranking
# than the one being measured. --linear-gain sets ndcg_exp_gain=False to
# align training with evaluation.
MODEL_PARAMS = dict(
    objective="rank:ndcg", n_estimators=300, learning_rate=0.05, max_depth=4,
    subsample=0.9, colsample_bytree=0.9, min_child_weight=2, reg_lambda=1.0,
)
DEFAULT_FEATURES = ["bm25_score", "bm25_rank", "dense_cosine", "dense_rank",
                    "rrf_score", "rrf_rank", "budget_fit", "seniority_fit", "avail_immediacy"]
# Features whose raw scale drifts per query, so --normalize-scores rescales them
# within each query. The *_rank features are already monotone within a query, so
# normalising them changes nothing a tree can use.
SCORE_FEATURES = ["bm25_score", "dense_cosine", "rrf_score",
                  "budget_fit", "seniority_fit", "avail_immediacy"]


def normalize_by_query(rows, keys, how):
    """Rescale feature(s) WITHIN each query.

    The score features are the ones whose raw scale drifts per query: bm25_score
    depends on query length and term rarity, dense_cosine on the query's
    embedding neighbourhood, rrf_score on how many lists agreed. A tree splits
    on absolute values, so one threshold means a different thing for every
    query. Per-query normalisation makes a split mean the same thing everywhere:
    "how does this candidate rank among its siblings for the same gig".

    This also fixes a train/serve skew on `ce_score`: the OOF scores the ranker
    trains on come from 5 different fold models, while the saved serving model
    scores every query itself.

    Measured on data_sat: +0.006 to +0.010 NDCG@10 (paired bootstrap, 10k
    resamples, significant in both harnesses tried).
    """
    if how == "none":
        return
    if isinstance(keys, str):
        keys = [keys]
    by_query = {}
    for r in rows:
        by_query.setdefault(r["hire_id"], []).append(r)
    for group in by_query.values():
        for key in keys:
            vals = [r[key] for r in group if r[key] == r[key]]  # skip NaN
            if len(vals) < 2:
                continue
            if how == "zscore":
                mu = sum(vals) / len(vals)
                sd = (sum((v - mu) ** 2 for v in vals) / len(vals)) ** 0.5 or 1.0
                for r in group:
                    if r[key] == r[key]:
                        r[key] = (r[key] - mu) / sd
            elif how == "rank":
                # 1.0 for the query's best candidate, down to ~0.02 at rank 50;
                # ties receive the identical value.
                rank_of = {v: i + 1 for i, v in enumerate(sorted(set(vals), reverse=True))}
                for r in group:
                    if r[key] == r[key]:
                        r[key] = 1.0 / rank_of[r[key]]


def load_json(p):
    return json.loads(Path(p).read_text())


def load_candidates(top_k: int, ce_tag: str | None, ce_normalize: str = "none"):
    path = FEAT_DIR / f"candidates_top{top_k}.csv"
    if not path.exists():
        raise SystemExit(f"{path} not found -- run `python pipeline/features.py --top-k {top_k}` first")
    rows = list(csv.DictReader(path.open()))

    features = list(DEFAULT_FEATURES)
    if ce_tag:
        ce_path = FEAT_DIR / f"ce_scores_{ce_tag}.csv"
        if not ce_path.exists():
            raise SystemExit(f"{ce_path} not found -- run `python pipeline/rerank_crossencoder.py --mode score --tag {ce_tag}`")
        ce = {(r["hire_id"], r["provider_id"]): float(r["ce_score"]) for r in csv.DictReader(ce_path.open())}
        missing = 0
        for r in rows:
            key = (r["hire_id"], r["provider_id"])
            if key in ce:
                r["ce_score"] = ce[key]
            else:
                r["ce_score"] = float("nan")
                missing += 1
        if missing:
            print(f"  note: {missing} candidate rows had no cross-encoder score (filled with NaN)")
        if ce_normalize != "none":
            normalize_by_query(rows, "ce_score", ce_normalize)
            print(f"  ce_score normalised per query: {ce_normalize}")
        features.append("ce_score")

    for r in rows:
        for f in features:
            r[f] = float(r[f]) if r.get(f) not in ("", None) else float("nan")
        r["label"] = int(r["label"]) if r["label"] not in ("", None) else None
        r["hire_id"] = int(r["hire_id"])
        r["provider_id"] = int(r["provider_id"])
    return rows, features


def order_by(rows, key, descending=True):
    """Rank a query's candidates by `key`. Scores sort descending; rank
    columns (rrf_rank) sort ascending so rank 1 wins."""
    return [r["provider_id"] for r in sorted(rows, key=lambda r: (-r[key] if descending else r[key]))]


def full_lists(ranked_head, rrf_full):
    out = {}
    for hid, head in ranked_head.items():
        seen = set(head)
        out[str(hid)] = head + [pid for pid in rrf_full.get(str(hid), []) if pid not in seen]
    return out


def report(name, ranked_by_query, gt, rrf_full):
    m = evaluate_all_hirers(full_lists(ranked_by_query, rrf_full), gt, ks=(5, 10))
    print(f"  {name:34s} P@5={m['precision@5']:.3f} R@5={m['recall@5']:.3f} NDCG@5={m['ndcg@5']:.3f} "
          f"P@10={m['precision@10']:.3f} R@10={m['recall@10']:.3f} NDCG@10={m['ndcg@10']:.4f} MRR={m['mrr']:.3f}")
    return m


def train_oof(rows, features, folds, seed, gt, rrf_full):
    """Return out-of-fold LambdaMART scores keyed by (hire_id, provider_id)."""
    by_query = defaultdict(list)
    for r in rows:
        by_query[r["hire_id"]].append(r)
    queries = sorted(by_query)
    oof = {}
    importances = []
    fold_metrics = []

    gkf = GroupKFold(n_splits=folds)
    for fold, (tr_idx, te_idx) in enumerate(gkf.split(queries, groups=queries), start=1):
        tr_q = sorted(queries[i] for i in tr_idx)
        te_q = sorted(queries[i] for i in te_idx)

        tr_rows = [r for q in tr_q for r in by_query[q] if r["label"] is not None]
        if not tr_rows:
            continue
        X = np.array([[r[f] for f in features] for r in tr_rows], dtype=np.float32)
        y = np.array([r["label"] for r in tr_rows], dtype=np.float32)
        groups = [sum(1 for r in tr_rows if r["hire_id"] == q) for q in tr_q]

        model = xgb.XGBRanker(**MODEL_PARAMS, random_state=seed, n_jobs=4)
        model.fit(X, y, group=groups)

        te_rows = [r for q in te_q for r in by_query[q]]
        Xt = np.array([[r[f] for f in features] for r in te_rows], dtype=np.float32)
        preds = model.predict(Xt)
        for r, p in zip(te_rows, preds):
            oof[(r["hire_id"], r["provider_id"])] = float(p)

        importances.append(model.feature_importances_)
        # per-fold sanity metric (fold-local ranking, tail appended)
        fold_rank = {}
        for r, p in zip(te_rows, preds):
            fold_rank.setdefault(r["hire_id"], []).append((r["provider_id"], float(p)))
        ranked = {q: [pid for pid, _ in sorted(v, key=lambda x: -x[1])] for q, v in fold_rank.items()}
        fold_metrics.append(evaluate_all_hirers(full_lists(ranked, rrf_full), gt, ks=(10,))["ndcg@10"])
        print(f"  fold {fold}: {len(tr_q)} train q / {len(te_q)} held-out q  NDCG@10={fold_metrics[-1]:.4f}")

    imp = np.mean(importances, axis=0)
    print("\n  mean feature importance:")
    for f, v in sorted(zip(features, imp), key=lambda x: -x[1]):
        print(f"    {f:18s} {v:.4f}")
    if fold_metrics:
        print(f"  fold NDCG@10: mean={np.mean(fold_metrics):.4f} std={np.std(fold_metrics):.4f}")
    return oof


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["train", "ablation"], default="ablation")
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--ce-tag", default=None, help="include features/ce_scores_<tag>.csv as a feature")
    ap.add_argument("--ce-normalize", default="none", choices=["none", "zscore", "rank"],
                    help="rescale ce_score within each query. The raw logit's per-query mean "
                         "spans ~9 units, and OOF scores come from 5 fold models while serving "
                         "uses one -- so an un-normalised ce_score means different things per query")
    ap.add_argument("--normalize-scores", default="none", choices=["none", "zscore", "rank"],
                    help="rescale the score features within each query (recommended: zscore). "
                         "Their raw scale drifts per query, so an absolute tree split means "
                         "something different for every gig")
    ap.add_argument("--linear-gain", action="store_true",
                    help="rank:ndcg with ndcg_exp_gain=False, matching evaluate.py's linear-gain "
                         "NDCG instead of XGBoost's default exponential (2^label-1) gain")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--export-scores", action="store_true",
                    help="write results/ltr_<tag>.json ranked lists + 0-100 calibrated scores")
    ap.add_argument("--tag", default="ltr")
    ap.add_argument("--data-dir", default="data",
                    help="dataset folder under pipeline/ (e.g. data_sat); outputs are namespaced accordingly")
    args = ap.parse_args()
    set_dataset(args.data_dir)
    if args.linear_gain:
        MODEL_PARAMS["ndcg_exp_gain"] = False
        print("objective: rank:ndcg with LINEAR gain (training aligned with evaluate.py)")

    MODELS_DIR.mkdir(exist_ok=True)
    rrf_full = load_json(RESULTS_DIR / "rrf_k60.json")
    gt = load_json(DATA_DIR / "ground_truth_llm.json")

    rows, features = load_candidates(args.top_k, args.ce_tag, args.ce_normalize)
    if args.normalize_scores != "none":
        normalize_by_query(rows, SCORE_FEATURES, args.normalize_scores)
        print(f"score features normalised per query: {args.normalize_scores}")
    print(f"loaded {len(rows)} candidate rows / {len({r['hire_id'] for r in rows})} queries")
    print(f"features ({len(features)}): {features}\n")

    by_query = defaultdict(list)
    for r in rows:
        by_query[r["hire_id"]].append(r)

    # ---------------- baselines ------------------------------------------
    print("baselines (same candidate universe, full lists):")
    base_rrf = {q: order_by(rs, "rrf_rank", descending=False) for q, rs in by_query.items()}
    m_rrf = report("RRF (frozen baseline)", base_rrf, gt, rrf_full)
    if args.ce_tag:
        base_ce = {q: order_by(rs, "ce_score") for q, rs in by_query.items()}
        m_ce = report(f"cross-encoder ({args.ce_tag})", base_ce, gt, rrf_full)

    # ---------------- LambdaMART (out-of-fold) ---------------------------
    print("\nLambdaMART (GroupKFold by query, out-of-fold predictions):")
    oof = train_oof(rows, features, args.folds, args.seed, gt, rrf_full)

    ltr_rank = {}
    for q, rs in by_query.items():
        scored = [(r["provider_id"], oof.get((q, r["provider_id"]), float("-inf"))) for r in rs]
        ltr_rank[q] = [pid for pid, _ in sorted(scored, key=lambda x: -x[1])]
    m_ltr = report("LambdaMART (" + ", ".join(f for f in features if f != "rrf_rank") + ")", ltr_rank, gt, rrf_full)

    # Dump the OUT-OF-FOLD ranked lists so the reported row is traceable to an
    # artifact, exactly like results/rrf_k60.json and results/ce_<tag>.json.
    # This is the OOF ranking (not the full-data model's), so it stays honest.
    oof_path = RESULTS_DIR / f"{args.tag}.json"
    oof_path.write_text(json.dumps({str(q): [int(pid) for pid in rs] for q, rs in ltr_rank.items()}, indent=2))
    print(f"wrote out-of-fold ranked lists to {oof_path}")

    # ---------------- final trained model on all queries ------------------
    all_rows = sorted([r for r in rows if r["label"] is not None], key=lambda r: r["hire_id"])
    X = np.array([[r[f] for f in features] for r in all_rows], dtype=np.float32)
    y = np.array([r["label"] for r in all_rows], dtype=np.float32)
    all_q = sorted({r["hire_id"] for r in all_rows})
    groups = [sum(1 for r in all_rows if r["hire_id"] == q) for q in all_q]
    final = xgb.XGBRanker(**MODEL_PARAMS, random_state=args.seed, n_jobs=4)
    final.fit(X, y, group=groups)
    model_path = MODELS_DIR / f"{args.tag}_xgb.json"
    final.save_model(str(model_path))
    print(f"\nsaved full-data model to {model_path}")

    if args.export_scores:
        # NOTE: min-max per query -> 0..100 so the value can leave our service as a
        # semantic_score. Ranking is what matters; the absolute scale is a contract
        # detail and must be documented to the sponsor (fixed 0-100 normalisation).
        out = {}
        for q, rs in by_query.items():
            Xq = np.array([[r[f] for f in features] for r in rs], dtype=np.float32)
            preds = final.predict(Xq)
            order = np.argsort(-preds)
            pmin, pmax = float(preds.min()), float(preds.max())
            span = (pmax - pmin) or 1.0
            out[str(q)] = [
                {"provider_id": int(rs[i]["provider_id"]),
                 "semantic_score": round(100.0 * (float(preds[i]) - pmin) / span, 2),
                 "rank": int(np.where(order == i)[0][0]) + 1}
                for i in range(len(rs))
            ]
        path = RESULTS_DIR / f"{args.tag}_scores.json"
        path.write_text(json.dumps(out, indent=2))
        print(f"wrote calibrated 0-100 scores to {path}")

    print("\n--- ablation summary ---")
    print(f"  RRF baseline        NDCG@10={m_rrf['ndcg@10']:.4f}  P@5={m_rrf['precision@5']:.3f}  MRR={m_rrf['mrr']:.3f}")
    if args.ce_tag:
        print(f"  + cross-encoder     NDCG@10={m_ce['ndcg@10']:.4f}  P@5={m_ce['precision@5']:.3f}  MRR={m_ce['mrr']:.3f}")
    print(f"  + LambdaMART (OOF)  NDCG@10={m_ltr['ndcg@10']:.4f}  P@5={m_ltr['precision@5']:.3f}  MRR={m_ltr['mrr']:.3f}")
    # The ~0.02 figure comes from the 130-query synthetic corpus. It is NOT
    # valid at data_sat's scale: standard error scales as 1/sqrt(n), so at
    # 1,023 queries the bar is ~2.8x tighter (~0.007). Printing a flat 0.02
    # here caused a real NDCG@10 regression to be read as "flat vs RRF".
    n_q = len({r["hire_id"] for r in rows}) if rows else 0
    threshold = 0.02
    if n_q:
        threshold = 0.02 * (130.0 / n_q) ** 0.5
    print(f"\nReminder: {n_q or '?'} queries -- treat any delta under "
          f"~{threshold:.3f} as noise. Report paired bootstrap CIs before claiming "
          f"an improvement.")


if __name__ == "__main__":
    main()
