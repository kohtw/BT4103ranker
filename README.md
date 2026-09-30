# Senseigigs — Stage-2 re-ranking for the gig↔showcase matching search

Retrieval (Stage 1) returns candidates; this repo holds the **Stage-2 rankers** that order
them, plus the feature builder and evaluation tooling used to measure both. The pipeline is
trained and evaluated on **`pipeline/data_sat`** — real gig/provider data from the
Scrape-and-Tag pipeline (1,023 gigs × 2,165 providers, 22,289 LLM-graded pairs; see
`pipeline/label.md`).

Pipeline shape:

```
gig query → [refined BM25 ‖ mxbai dense] → RRF → top-50 candidates
          → LambdaMART over [text scores + business signals] (Stage 2)
          → fused with the previous ranker (RRF of the two lists, w=0.7)
          → 0-100 calibrated scores
```

The cross-encoder scores candidates during development but is **not** a ranker feature (see
Results — measured negative on real data), so it is absent from the serving path above.

## Layout

| Path | What |
|---|---|
| `pipeline/data_sat/` | Real corpus + LLM-graded relevance labels (the dataset the results below use) |
| `pipeline/label.md` | How `data_sat` was built and graded — pool construction, grader model, caveats |
| `pipeline/import_scrape_and_tag.py`, `pipeline/build_judging_pools_sat.py` | Build `data_sat/` from raw Scrape-and-Tag CSVs and generate its judging pools |
| `pipeline/features.py` | Feature builder: candidate pool + BM25/dense/RRF scores + budget/seniority/availability signals + labels |
| `pipeline/rerank_crossencoder.py` | Cross-encoder re-ranker: zero-shot scoring, and fine-tuning under **per-query K-fold CV** |
| `pipeline/rerank_ltr.py` | LambdaMART (XGBoost) ranker + ablation report. `--linear-gain` aligns the training gain with `evaluate.py`; `--normalize-scores zscore` rescales per query; `--ce-tag`/`--ce-normalize` optionally add cross-encoder scores |
| `pipeline/fuse_rankers.py` | Weighted reciprocal-rank fusion of ranked lists — how the shipped fused ranker is produced |
| `pipeline/finetune_embeddings.py` | Dense embedding fine-tune — Matryoshka-aware loss, CUDA, collapse guard |
| `pipeline/RERANK_README.md` | Runbook: exact commands, VRAM guidance, pitfalls, worked query examples |
| `pipeline/evaluate.py` | P@K / R@K / NDCG@K / MRR (linear gain, relevance bar = score ≥ 40 ⇔ grade ≥ 2) |
| `pipeline/retrieval_*.py` | Stage-1 retrievers (BM25, dense, RRF) |
| `pipeline/features_data_sat/`, `pipeline/results_data_sat/*`, `pipeline/models_data_sat/` | Built feature tables, result lists and trained model for `data_sat` |
| `pipeline/data/`, `pipeline/results/`, `pipeline/features/`, `pipeline/models/` | Archived synthetic-data baseline — kept for reference only, not covered below; see git history |

## Quickstart (CUDA)

```bash
conda create -n fi-bench python=3.12 && conda activate fi-bench   # torch with CUDA
pip install -r pipeline/requirements.txt
pip install -r pipeline/requirements-rerank.txt   # sentence-transformers, xgboost, scikit-learn, rank-bm25

python pipeline/run_pipeline.py --data-dir data_sat                       # Stage 1: BM25/dense/RRF
python pipeline/features.py --data-dir data_sat --top-k 50
python pipeline/rerank_crossencoder.py --mode score --data-dir data_sat --tag minilm   # zero-shot CE
python pipeline/rerank_ltr.py --mode ablation --data-dir data_sat --top-k 50 --linear-gain \
    --normalize-scores zscore --tag linz                                      # RRF → +LTR (recommended)
python pipeline/fuse_rankers.py --data-dir data_sat --inputs linz noce --weights 0.7 0.3 \
    --out fused_linz_noce.json                                                # → the shipped ranker
```

Fine-tune the cross-encoder (5-fold per-query CV, ~65 min on an RTX 3060 — completed for
`data_sat`; the OOF scores are kept for measurement, but the CE is deliberately **not** a ranker
feature, see Results):

```bash
python pipeline/rerank_crossencoder.py --mode finetune --data-dir data_sat --cv-folds 5 --epochs 2 --tag cecv --quiet
python pipeline/rerank_ltr.py --mode ablation --data-dir data_sat --ce-tag cecv
```

Every command above also runs against the archived synthetic corpus by omitting `--data-dir`
(defaults to `data`).

## Results

RRF top-50 → LambdaMART, evaluated out-of-fold (GroupKFold by query) against `data_sat`'s
1,023 LLM-graded queries:

| Stage | NDCG@10 | P@5 | R@5 | R@10 | MRR |
|---|---|---|---|---|---|
| RRF k=60 (Stage 1 baseline) | 0.671 | 0.116 | 0.531 | 0.774 | 0.278 |
| + cross-encoder, zero-shot (`ms-marco-MiniLM-L6-v2`) | 0.365 | 0.064 | 0.291 | 0.407 | 0.183 |
| + cross-encoder, fine-tuned (per-query 5-fold CV, out-of-fold) | 0.555 | 0.116 | 0.548 | 0.728 | 0.295 |
| + LambdaMART, exponential gain (previously shipped) | 0.658 | **0.153** | **0.682** | 0.860 | 0.338 |
| + LambdaMART with `ce_score` — trained, **rejected** | 0.633 | 0.154 | 0.705 | 0.854 | 0.363 |
| + LambdaMART, **linear gain + per-query-normalised features** | **0.685** | 0.141 | 0.640 | 0.854 | 0.333 |
| **Fused 0.7×new + 0.3×previous — the shipped ranker** | 0.684 | 0.146 | 0.657 | **0.858** | **0.341** |

Paired bootstrap over the 1,023 queries (10k resamples) — **shipped ranker (fused) vs RRF**:

| Metric | Δ | 95% CI | Verdict |
|---|---|---|---|
| NDCG@10 | **+0.0132** | [+0.0039, +0.0227] | improved |
| P@5 | **+0.0305** | [+0.0246, +0.0366] | improved |
| R@10 | **+0.0843** | [+0.0572, +0.1119] | improved |
| MRR | **+0.0644** | [+0.0509, +0.0782] | improved |

The shipped ranker now beats RRF on **all four** metrics. The previous ranker did not — it
*regressed* on NDCG@10 (−0.0133, [−0.0230, −0.0035]).

vs the **previous** ranker, the shipped model trades one metric for another, and the trade is
explicit: NDCG@10 **+0.0265** [+0.0211, +0.0318] improved, P@5 **−0.0070** [−0.0111, −0.0029]
worse, R@10 (0.0028) and MRR (+0.0038) unchanged. It buys the top-of-list quality that NDCG
measures and gives back a little precision-at-5.

**Two scoring bugs were found and fixed in this run** (both inherited, both now measured):

1. **The training objective disagreed with the evaluation metric.** `rerank_ltr.py` trains
   `rank:ndcg`, which by default uses *exponential* gain (2^label−1 → 0/1/3/7), while
   `evaluate.py` scores NDCG with *linear* gain on the 0-100 scores (0/33/66/100). Exponential
   gain makes grade-3 seven times grade-1; linear makes it three times — so the model was
   optimising a different ranking than the one being reported. `--linear-gain` sets
   `ndcg_exp_gain=False`: **NDCG@10 0.658 → 0.681** [+0.0206, +0.0336].
2. **Feature scales drift per query.** `bm25_score`, `dense_cosine` and `rrf_score` have
   query-dependent scales, and a tree splits on absolute values — so one threshold meant a
   different thing for every gig. `--normalize-scores zscore` rescales them within each query:
   a further **+0.006 to +0.010 NDCG@10**, significant in two independent harnesses.

- **The cross-encoder is fine-tuned on `data_sat`** (per-query 5-fold `GroupKFold`, 2 epochs,
  ~65 min on an RTX 3060, OOF scores only so it stays leak-free). It improves hugely over
  zero-shot — **0.365 → 0.555 NDCG@10 (+0.190)** — but is still below RRF standalone.
- **The CE stays out of the ranker even after the scaling fix.** Its raw logit has a per-query
  mean spanning ~9 units, and its OOF scores come from 5 different fold models while the serving
  model scores every query itself — a real train/serve skew worth fixing on principle. It was
  fixed (`--ce-normalize zscore`/`rank`) and the CE still lost: best CE variant 0.641 vs 0.681
  without it. The feature is genuinely redundant with BM25+dense here, not merely mis-scaled.
  Weights are kept in `models_data_sat/ce-cecv/` for the semantic-score role.
- **The fusion is the one place where two models beat either alone.** The linear-gain model and
  the previous model are strong on different metrics; weighted RRF at w=0.7 keeps the whole
  NDCG@10 gain while returning ~40% of the P@5 it would otherwise cost (P@5 +0.0049
  [+0.0027, +0.0072] and MRR +0.0082 [+0.0040, +0.0126] vs the new model alone, at no measurable
  NDCG loss). Reproduce with `pipeline/fuse_rankers.py --data-dir data_sat --inputs linz noce
  --weights 0.7 0.3`.
- **What did *not* help**, all measured and rejected: five new features from unused fields
  (industry match, rate ratio, budget under/over split, capacity fit) — no effect; per-feature
  *percentile* normalisation (0.671) and *rank* normalisation — worse than z-score; `rank:pairwise`
  (0.651); deeper trees (0.678); 600 trees @ lr 0.03 (0.686) and a 3-seed ensemble (0.687) — both
  within noise of 0.685; fusing the ranker with RRF — inconclusive.
- **One sharp trade-off worth knowing**: training on *binary* relevance (grade ≥2) with
  `rank:map` optimises exactly what P@5/R@10/MRR measure and gives the **best P@5 (0.158) and MRR
  (0.348) of anything tried**, but collapses NDCG@10 to 0.554. If the top-5 shortlist is what
  matters rather than graded ordering, that model is the one to use.
- **Zero-shot cross-encoder regresses hard** — confirmed out-of-domain for consulting/finance
  text (0.365 vs 0.671), so its score is not used anywhere downstream.

**Full write-up** — dataset stats, the fixes made to support real data (structured
budget/seniority/availability fields, a date-parsing bug found and fixed during review), feature
importances, worked examples, and caveats — is in `pipeline/RERANK_README.md`.

## Honest caveats

- **47% of gigs have no graded-≥2 provider _in their pool_.** 481/1,023 gigs have no provider
  graded ≥2 among the ~22 pooled for that gig. Only ~1% of the 2,165-provider catalog is graded
  for any given gig, so this is a **pool-coverage** statement, not a claim that the catalog lacks
  a strong match.
- **Recall is undefined for 47% of gigs.** `evaluate.recall_at_k` returns `None` when a gig has no
  relevant item at all, so every R@5/R@10 in the table above is averaged over only the **542
  answerable gigs**, not all 1,023. NDCG@10/P@5/MRR average over all 1,023.
- **The noise bar is ~0.007 on this corpus, not 0.02.** `rerank_ltr.py`'s printed reminder was
  calibrated for the *130-query synthetic* corpus; it is now dataset-aware and prints the correct
  figure for whatever corpus it is pointed at. At 1,023 queries the standard error is ~2.8×
  smaller than the synthetic corpus's, so deltas near 0.01 are real here — always quote the
  bootstrap CIs.
- **Grades cluster low.** 50.8% grade-0, 43.7% grade-1, 2.5% grade-2, 3.0% grade-3 (grader
  `qwen3.8:27b`, `pipeline/label.md`) — only 5.5% of judged pairs are ≥2, so most of the top-10
  for any query is "plausible, not excellent" by construction.
- **Zero-shot cross-encoder is a net negative** on this domain, and the fine-tuned one is a
  negative *as a ranker feature*. Both are measured, not assumed.
- **`budget_fit` penalizes below-budget rates as harshly as above-budget ones** — a cheaper-than-
  requested provider gets demoted the same as an over-budget one, which is a plausible source of
  some of LambdaMART's query-level regressions (see the worked example in `RERANK_README.md`).
  Not fixed in this run.
- **Pre-existing, not `data_sat`-specific**: `retrieval_bm25.py`'s query-side title weighting
  effectively 4×s the title instead of the documented 3× (a token-concatenation bug in
  `BM25Retriever._query_tokens`). Affects every BM25/RRF run in this repo. Flagged in
  `RERANK_README.md`, not fixed, since fixing it changes results for both datasets.
- **The fold split is fixed, not averaged.** Every number above comes from one 5-fold query
  split (`--seed 7`), so the *mean* is single-split; the CIs resample queries within that split.
  Re-running with a different split moves NDCG@10 by roughly ±0.008 — treat effects smaller than
  that as unproven until they replicate across splits.
- Model weights and embedding/results caches are git-ignored (regenerable, and large at this
  corpus scale); rebuild with the quickstart commands.
- An archived synthetic-data baseline and a Matryoshka-aware dense-encoder fine-tune validated
  against it are preserved in `pipeline/data/`, `results/`, `features/`, `models/` and
  `pipeline/finetune_embeddings.py` for reference — not documented here; see git history for
  their numbers and methodology.
