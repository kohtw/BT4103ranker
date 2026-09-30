"""
Cross-encoder re-ranker (Stage 2a) for the Senseigigs matching search.

Two jobs:
  score    -- zero-shot re-rank of the frozen RRF top-K, writing both the
              re-ranked result lists and a `ce_score` column the LambdaMART
              ranker can consume as a feature.
  finetune -- optional: train the cross-encoder on our own LLM judgments, under
              PER-QUERY K-FOLD CV so the scores it writes are out-of-fold.
              DO THIS LAST AND SUSPICIOUSLY. We have 173 grade-3 positives
              (389 at grade>=2) across 130 queries -- that is a very small
              fine-tuning set, and MS MARCO-pretrained rerankers are already
              out-of-domain for consulting/finance text. Zero-shot first;
              fine-tune only if the zero-shot gain is real but insufficient.
              The saved model is trained on all pairs (for serving); only the
              written ce_scores_<tag>.csv is OOF (for honest measurement).

GPU notes (RTX 3060 Laptop, 6 GB):
  - MiniLM-L6-v2 (~23M)   : batch 32+ fine, even with --fp16 off
  - gte-reranker-modernbert-base (149M): batch 16, max-length 512
  - bge-reranker-v2-m3 (568M) / mxbai-rerank-large (435M): batch 4-8, --fp16
    If you see NaN or loss going flat, drop --fp16 first -- that is the same
    class of numerical-stability bug that produced NaN embeddings on the
    embedding fine-tune (see finetune_embeddings.py).

Run (fi-bench env, from the repo root):
    python pipeline/rerank_crossencoder.py --mode score
    python pipeline/rerank_crossencoder.py --mode score --model BAAI/bge-reranker-v2-m3 --fp16 --tag bge
    python pipeline/rerank_crossencoder.py --mode finetune --model cross-encoder/ms-marco-MiniLM-L6-v2
"""
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import CrossEncoder, InputExample
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader

from corpus import provider_text, hirer_text
from evaluate import evaluate_all_hirers

BASE = Path(__file__).parent
DATA_DIR = BASE / "data"
FEAT_DIR = BASE / "features"
RESULTS_DIR = BASE / "results"
MODELS_DIR = BASE / "models"

DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"


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


def load_json(p):
    return json.loads(Path(p).read_text())


def pick_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  GPU: {name} ({vram:.1f} GB) -- device=cuda")
        return "cuda"
    print("  no CUDA device visible -- device=cpu")
    return "cpu"


def build_pairs(rows, providers_by_id, hirers_by_id):
    """rows: candidate CSV rows -> [(query_text, doc_text)] plus metadata."""
    pairs, meta = [], []
    for r in rows:
        h = hirers_by_id[int(r["hire_id"])]
        p = providers_by_id[int(r["provider_id"])]
        pairs.append((hirer_text(h).strip(), provider_text(p).strip()))
        meta.append(r)
    return pairs, meta


def rerank_lists(meta, scores, rrf_full):
    """Reorder each query's candidate pool by cross-encoder score, then append
    the untouched RRF tail so every ranked list stays a full permutation."""
    by_query = {}
    for r, s in zip(meta, scores):
        by_query.setdefault(str(r["hire_id"]), []).append((int(r["provider_id"]), float(s)))
    out = {}
    for hid, scored in by_query.items():
        seen = {pid for pid, _ in scored}
        head = [pid for pid, _ in sorted(scored, key=lambda x: -x[1])]
        tail = [pid for pid in rrf_full.get(hid, []) if pid not in seen]
        out[hid] = head + tail
    return out


def score(args):
    cand_path = FEAT_DIR / f"candidates_top{args.top_k}.csv"
    if not cand_path.exists():
        raise SystemExit(f"{cand_path} not found -- run `python pipeline/features.py --top-k {args.top_k}` first")

    rows = list(csv.DictReader(cand_path.open()))
    providers_by_id = {p["provider_id"]: p for p in load_json(DATA_DIR / "providers.json")}
    hirers_by_id = {h["hire_id"]: h for h in load_json(DATA_DIR / "hirers.json")}
    rrf_full = load_json(RESULTS_DIR / "rrf_k60.json")
    gt = load_json(DATA_DIR / "ground_truth_llm.json")

    device = pick_device(args.device)
    print(f"Loading cross-encoder {args.model} ...")
    ce = CrossEncoder(args.model, num_labels=1, device=device)
    if args.fp16 and device == "cuda":
        ce.model.half()
        print("  fp16 enabled")

    pairs, meta = build_pairs(rows, providers_by_id, hirers_by_id)
    print(f"Scoring {len(pairs)} (gig, showcase) pairs ...")
    t0 = time.time()
    scores = ce.predict(pairs, batch_size=args.batch_size, show_progress_bar=True)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if not np.isfinite(scores).all():
        raise SystemExit(f"non-finite cross-encoder scores ({int((~np.isfinite(scores)).sum())} of {len(scores)}) "
                         f"-- drop --fp16 and re-run")
    print(f"  scored in {time.time() - t0:.1f}s  (min={scores.min():.3f} max={scores.max():.3f})")

    # ce score column for the LTR ranker
    score_csv = FEAT_DIR / f"ce_scores_{args.tag}.csv"
    with score_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["hire_id", "provider_id", "ce_score"])
        for r, s in zip(meta, scores):
            w.writerow([r["hire_id"], r["provider_id"], f"{float(s):.8f}"])

    ranked = rerank_lists(meta, scores, rrf_full)
    out_json = RESULTS_DIR / f"ce_{args.tag}.json"
    out_json.write_text(json.dumps(ranked, indent=2))

    m = evaluate_all_hirers(ranked, gt, ks=(5, 10))
    print(f"\nwrote {out_json.name} + {score_csv.name}")
    print(f"  {args.tag:22s} P@5={m['precision@5']:.3f} R@5={m['recall@5']:.3f} "
          f"NDCG@5={m['ndcg@5']:.3f} P@10={m['precision@10']:.3f} R@10={m['recall@10']:.3f} "
          f"NDCG@10={m['ndcg@10']:.4f} MRR={m['mrr']:.3f}")


def make_examples(rows, train_q, providers_by_id, hirers_by_id, positive_grade):
    """(query, doc) pairs for the queries in `train_q`; returns (examples, n_pos, n_neg)."""
    ex, n_pos, n_neg = [], 0, 0
    for r in rows:
        if r["hire_id"] not in train_q:
            continue
        g = int(r["label"])
        if g >= positive_grade:
            label, n_pos = 1.0, n_pos + 1
        elif g <= 1:
            label, n_neg = 0.0, n_neg + 1
        else:
            continue                      # ambiguous middle grade: excluded
        h = hirers_by_id[int(r["hire_id"])]
        p = providers_by_id[int(r["provider_id"])]
        ex.append(InputExample(texts=[hirer_text(h).strip(), provider_text(p).strip()], label=label))
    return ex, n_pos, n_neg


def _fit_ce(args, examples, device):
    """Fresh cross-encoder fitted on `examples`; raises if the scores go non-finite."""
    ce = CrossEncoder(args.model, num_labels=1, device=device)
    if args.fp16 and device == "cuda":
        ce.model.half()
    loader = DataLoader(examples, shuffle=True, batch_size=args.batch_size)
    ce.fit(train_dataloader=loader, epochs=args.epochs,
           warmup_steps=max(10, len(loader) // 10),
           output_path=str(MODELS_DIR / "_ce_trainer_output"), show_progress_bar=not args.quiet)
    return ce


def finetune(args):
    """Fine-tune on our LLM judgments under **per-query K-fold cross-validation**.

    grade >= `positive_grade` is a positive, grade 0/1 a negative (grade 2 is
    ambiguous unless positive_grade=2). Folds are split by hirer, never randomly.

    The fine-tuned scores written to features/ce_scores_<tag>.csv are
    OUT-OF-FOLD: every query is scored by a model that never saw that query's
    pairs. The earlier single-split version scored 104/130 queries in-sample,
    which made the downstream LambdaMART numbers optimistic.

    A final model trained on ALL pairs is saved to models/ce-<tag> for serving --
    at serving time the query is unseen by construction, so nothing leaks there.
    Only the *measurement* needs the OOF scores.
    """
    cand_path = FEAT_DIR / "candidates_top50.csv"
    pair_path = FEAT_DIR / "train_pairs.csv"
    if not cand_path.exists() or not pair_path.exists():
        raise SystemExit("features/candidates_top50.csv not found -- run pipeline/features.py first")

    # Two distinct universes, and conflating them is a trap:
    #   cand_rows = every candidate the ranker will ever see  -> SCORING + metrics
    #   rows      = judged pairs only                         -> TRAINING
    # Scoring only the judged rows would shorten the candidate head and make the
    # reported metrics incomparable to the ablation (unjudged candidates would be
    # pushed into the RRF tail) and would leave 3,509 candidates without a
    # ce_score for the LambdaMART.
    cand_rows = list(csv.DictReader(cand_path.open()))
    rows = [r for r in cand_rows if r["label"] != ""]
    print(f"candidates to score: {len(cand_rows)}   training pairs: {len(rows)}")
    providers_by_id = {p["provider_id"]: p for p in load_json(DATA_DIR / "providers.json")}
    hirers_by_id = {h["hire_id"]: h for h in load_json(DATA_DIR / "hirers.json")}
    rrf_full = load_json(RESULTS_DIR / "rrf_k60.json")
    gt = load_json(DATA_DIR / "ground_truth_llm.json")

    queries = sorted({r["hire_id"] for r in cand_rows}, key=int)
    device = pick_device(args.device)
    print(f"cross-encoder: {args.model}")
    print(f"device: {device}   queries: {len(queries)}   folds: {args.cv_folds}   "
          f"positive grade: >= {args.positive_grade}")

    # ---- zero-shot reference over the SAME candidate universe (no training) ----
    base_ce = CrossEncoder(args.model, num_labels=1, device=device)
    pairs_all, meta_all = build_pairs(cand_rows, providers_by_id, hirers_by_id)
    zs_all = np.asarray(base_ce.predict(pairs_all, batch_size=args.batch_size,
                                       show_progress_bar=False)).reshape(-1)
    m_zs = evaluate_all_hirers(rerank_lists(meta_all, zs_all, rrf_full), gt, ks=(5, 10))
    print(f"  zero-shot   : P@5={m_zs['precision@5']:.3f} NDCG@10={m_zs['ndcg@10']:.4f} MRR={m_zs['mrr']:.3f}")
    del base_ce
    if device == "cuda":
        torch.cuda.empty_cache()

    # ---- per-query K-fold CV: every query scored by a model that never saw it ----
    oof, fold_report = {}, []
    for fold, (tr_idx, te_idx) in enumerate(GroupKFold(n_splits=args.cv_folds).split(queries, groups=queries), 1):
        train_q = {queries[i] for i in tr_idx}
        test_q = {queries[i] for i in te_idx}
        examples, n_pos, n_neg = make_examples(rows, train_q, providers_by_id, hirers_by_id, args.positive_grade)
        if not examples:
            raise SystemExit(f"fold {fold}: no training pairs")
        print(f"\n  fold {fold}/{args.cv_folds}: {len(examples)} pairs ({n_pos} pos / {n_neg} neg) "
              f"over {len(train_q)} queries -> scoring {len(test_q)} held-out queries")
        t0 = time.time()
        ce = _fit_ce(args, examples, device)

        test_rows = [r for r in cand_rows if r["hire_id"] in test_q]
        pairs, meta = build_pairs(test_rows, providers_by_id, hirers_by_id)
        sc = np.asarray(ce.predict(pairs, batch_size=args.batch_size,
                                   show_progress_bar=False)).reshape(-1)
        if not np.isfinite(sc.astype(np.float64)).all():
            raise SystemExit(f"fold {fold}: non-finite scores -- retrain without --fp16")
        for r, s in zip(meta, sc):
            oof[(r["hire_id"], r["provider_id"])] = float(s)

        m = evaluate_all_hirers(rerank_lists(meta, sc, rrf_full), gt, ks=(10,))
        fold_report.append(m["ndcg@10"])
        print(f"    done in {time.time() - t0:.1f}s   held-out NDCG@10={m['ndcg@10']:.4f}")
        del ce
        if device == "cuda":
            torch.cuda.empty_cache()

    # ---- OOF scoring over all queries: the honest number ----
    missing = [r for r in cand_rows if (r["hire_id"], r["provider_id"]) not in oof]
    if missing:
        raise SystemExit(f"{len(missing)} candidate pairs got no OOF score")

    score_csv = FEAT_DIR / f"ce_scores_{args.tag}.csv"
    with score_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["hire_id", "provider_id", "ce_score"])
        for r in cand_rows:
            w.writerow([r["hire_id"], r["provider_id"], f"{oof[(r['hire_id'], r['provider_id'])]:.8f}"])

    oof_scores = [oof[(r["hire_id"], r["provider_id"])] for r in cand_rows]
    ranked = rerank_lists(meta_all, oof_scores, rrf_full)
    out_json = RESULTS_DIR / f"ce_{args.tag}.json"
    out_json.write_text(json.dumps(ranked, indent=2))

    m_oof = evaluate_all_hirers(ranked, gt, ks=(5, 10))
    print(f"\n{'=' * 74}")
    print(f"  zero-shot     (all {len(queries)} queries) : P@5={m_zs['precision@5']:.3f} "
          f"NDCG@10={m_zs['ndcg@10']:.4f} MRR={m_zs['mrr']:.3f}")
    print(f"  fine-tuned OOF(all {len(queries)} queries) : P@5={m_oof['precision@5']:.3f} "
          f"NDCG@10={m_oof['ndcg@10']:.4f} MRR={m_oof['mrr']:.3f}   "
          f"(per-fold NDCG@10: {np.mean(fold_report):.4f} +- {np.std(fold_report):.4f})")
    delta = m_oof["ndcg@10"] - m_zs["ndcg@10"]
    print(f"  fine-tune delta: {delta:+.4f} NDCG@10   "
          f"{'keep fine-tuned' if delta > 0 else 'KEEP ZERO-SHOT (fine-tune did not help)'}")
    print(f"{'=' * 74}")
    print(f"wrote {score_csv.name} (OUT-OF-FOLD) + {out_json.name}")

    # ---- final model on all pairs, for serving ----
    all_examples, n_pos, n_neg = make_examples(rows, set(queries), providers_by_id, hirers_by_id,
                                               args.positive_grade)
    print(f"\ntraining final serving model on all {len(all_examples)} pairs "
          f"({n_pos} pos / {n_neg} neg) ...")
    final = _fit_ce(args, all_examples, device)
    probe = final.predict(pairs_all[:8], batch_size=8, show_progress_bar=False)
    if not np.isfinite(np.asarray(probe, dtype=np.float64)).all():
        raise SystemExit("final model produced NaN/inf -- retrain without --fp16")
    out_model = MODELS_DIR / f"ce-{args.tag}"
    final.save(str(out_model))
    print(f"saved serving model to {out_model}")
    print("\nNOTE: ce_scores_<tag>.csv is OOF (for honest measurement); the saved "
          "model is trained on all pairs (for serving).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["score", "finetune"], required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--tag", default="minilm", help="output name tag (results/ce_<tag>.json, features/ce_scores_<tag>.csv)")
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--cv-folds", type=int, default=5,
                    help="per-query folds; the written ce_scores_<tag>.csv is out-of-fold")
    ap.add_argument("--positive-grade", type=int, default=3, help="grade >= this counts as a positive")
    ap.add_argument("--quiet", action="store_true", help="suppress the training progress bars")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--data-dir", default="data",
                    help="dataset folder under pipeline/ (e.g. data_sat); outputs are namespaced accordingly")
    args = ap.parse_args()
    set_dataset(args.data_dir)
    MODELS_DIR.mkdir(exist_ok=True)
    ({"score": score, "finetune": finetune}[args.mode])(args)


if __name__ == "__main__":
    main()
