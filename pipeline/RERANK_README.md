# Stage-2 Re-ranking — cross-encoder + LambdaMART (runbook)

Search-engine shape: **retrieve wide (Stage 1, already built) → rank narrow (Stage 2, this).**

```
query → [refined BM25 ‖ mxbai dense] → RRF → top-50
      → cross-encoder over top-K         (relevance ordering)      rerank_crossencoder.py
      → LambdaMART over all features     (business/structured signals) rerank_ltr.py
      → calibrate → 0–100 JSON           (contract handoff)
```

Primary dataset: **`pipeline/data_sat`** — real gig/provider data from the Scrape-and-Tag
pipeline (1,023 hirers × 2,165 providers, 23,873 LLM-graded pairs; see `label.md`). Every
command below defaults to it via `--data-dir data_sat` and namespaces its outputs
(`features_data_sat/`, `results_data_sat/`, `models_data_sat/`, `cache_data_sat/`). An earlier,
smaller synthetic corpus is archived in `pipeline/data/` (omit `--data-dir`, or pass
`--data-dir data`) — its results are not covered here; see git history.

Current result to beat: **RRF k=60, NDCG@10 = 0.671, P@5 = 0.116, MRR = 0.278** (see "Results"
below for why these look low, and the worked examples for what's actually happening query by
query).

---

## 1. Environment

```bash
conda activate fi-bench
pip install -r pipeline/requirements-rerank.txt
python -c "import torch, xgboost, sentence_transformers; print(torch.cuda.is_available())"
```

`fi-bench` already has torch 2.11.0+cu130 and sees the RTX 3060 Laptop GPU (6.4 GB). An earlier
`data_sat` run was CPU/MPS-only, which is why cross-encoder fine-tuning (§6) was originally left
undone — it is slow without a GPU. **It has since been completed on the RTX 3060 (~65 min for
5 folds); §6 records the result, which is that the fine-tuned CE is NOT used as a ranker feature.**
If `import sentence_transformers` fails after install, it's a transformers-version clash —
`sentence-transformers` may pin `transformers<5`; check with `pip check` before assuming the env is fine.

## 2. Stage 1 (BM25 / dense / RRF)

```bash
python pipeline/run_pipeline.py --data-dir data_sat
```

Writes `results_data_sat/{bm25,dense_*,rrf_k*}.json` — the full ranked lists Stage 2 reranks.
This is a full sweep (multiple Matryoshka truncation dims, RRF k/weight combos) so it's the
slowest step (~6 minutes at `data_sat` scale on CPU/MPS); only `rrf_k60.json` is actually
consumed downstream.

## 3. Build the feature table

```bash
python pipeline/features.py --data-dir data_sat --top-k 50
```

Writes `features_data_sat/candidates_top50.csv` (all candidates), `features_data_sat/train_pairs.csv`
(judged only), and caches embeddings under `cache_data_sat/`. It also prints the recall-loss
line — how many grade≥2 providers fall **outside** the pool, which no re-ranker can recover (5
out of ~2,165 for `data_sat` — a tight pool, not a real recall problem).

## 4. Cross-encoder, zero-shot (Stage 2a)

```bash
# CPU-friendly default, safe starting point
python pipeline/rerank_crossencoder.py --mode score --data-dir data_sat --tag minilm
```

Writes `results_data_sat/ce_<tag>.json` (full ranked lists) and
`features_data_sat/ce_scores_<tag>.csv` (feature column for the ranker). At `data_sat` scale
(51,150 candidate pairs) this took ~14 minutes on CPU — budget accordingly, or move to GPU.

**On `data_sat`, zero-shot regresses hard** (NDCG@10 0.671 → 0.365) — confirmed out-of-domain
for consulting/finance text. Its score is **not** fed into LambdaMART below.

## 5. LambdaMART (Stage 2b) + ablation table

```bash
# RECOMMENDED — the configuration whose numbers are in the Results table
python pipeline/rerank_ltr.py --mode ablation --data-dir data_sat --top-k 50 \
    --linear-gain --normalize-scores zscore --tag linz

# baseline for comparison: exponential gain, raw feature scales
python pipeline/rerank_ltr.py --mode ablation --data-dir data_sat --top-k 50 --tag noce

# + cross-encoder score (measured negative on data_sat -- see the CE note below)
python pipeline/rerank_ltr.py --mode ablation --data-dir data_sat --top-k 50 --ce-tag cecv --ce-normalize zscore
```

Prints the ablation (RRF → [+CE] → +LTR), per-fold NDCG@10 with mean/std, and feature
importances. Predictions are **out-of-fold** (GroupKFold by query), so no query is scored by a
model that saw it.

### The two flags that matter

**`--linear-gain`** — `rank:ndcg` uses *exponential* gain by default (2^label−1 → 0/1/3/7) while
`evaluate.py` scores NDCG with *linear* gain on the 0-100 scores (0/33/66/100). Exponential gain
weights grade-3 seven times grade-1; linear weights it three times. Without this flag the model
optimises a different ranking than the one being reported. Worth **+0.023 NDCG@10**.

**`--normalize-scores zscore`** — `bm25_score`, `dense_cosine` and `rrf_score` have query-dependent
scales, and trees split on absolute values, so one threshold meant a different thing per gig.
Rescaling within each query makes a split mean "how does this candidate compare to its siblings
for this gig". Worth a further **+0.006 to +0.010 NDCG@10**. `rank` normalisation is available but
measured worse; *percentile* normalisation is worse still.

### Fusion: how the shipped ranker is built

The linear-gain model wins NDCG@10 but gives back some P@5 versus the exponential-gain model.
Fusing the two ranked lists with weighted RRF recovers that at no NDCG cost:

```bash
python pipeline/fuse_rankers.py --data-dir data_sat --inputs linz noce --weights 0.7 0.3 \
    --out fused_linz_noce.json
```

Prints each input's metrics and the fused row, and writes the fused list. Fusion at w=0.7 is a
strict Pareto point against `linz` alone (P@5 +0.0049 and MRR +0.0082 both significant, NDCG@10
unchanged within CI).

## 6. Optional: fine-tune the cross-encoder (do this last)

```bash
# 5-fold per-query CV: the ce_scores file it writes is OUT-OF-FOLD
python pipeline/rerank_crossencoder.py --mode finetune --data-dir data_sat \
    --model cross-encoder/ms-marco-MiniLM-L6-v2 --epochs 2 --cv-folds 5 --tag cecv --quiet
python pipeline/rerank_ltr.py --mode ablation --data-dir data_sat --ce-tag cecv     # re-run with the honest scores
```

**Completed for `data_sat` on an RTX 3060 (5 folds, 2 epochs, ~65 min).** Result — and the reason
the CE is **not** a ranker feature:

| CE variant (1,023 queries) | P@5 | NDCG@10 | MRR |
|---|---|---|---|
| zero-shot | 0.064 | 0.3649 | 0.183 |
| fine-tuned (OOF) | 0.116 | **0.5552** ± 0.0228 | 0.295 |

Fine-tuning lifts the CE by **+0.190 NDCG@10** — it is no longer catastrophic. But folding
`ce_score` into LambdaMART costs **−0.0246 NDCG@10 [−0.0349, −0.0141]** and buys only
**+0.0259 MRR [+0.0123, +0.0396]**, so the shipped ranker (`models_data_sat/noce_xgb.json`) omits
it. The OOF scores are still written for measurement, and the fine-tuned weights are kept for the
semantic-score role.

**Why CV and not a single split.** An earlier version trained on 80% of queries and evaluated on
the other 20%, then wrote *in-sample* scores for the training queries — which silently made every
downstream LambdaMART number optimistic. Now each fold trains on K−1 folds and scores only its
held-out fold, so **every query's `ce_score` comes from a model that never saw that query**.

**Two artefacts, two purposes.** `features_data_sat/ce_scores_<tag>.csv` is OOF → use it for
evaluation. `models_data_sat/ce-<tag>/` is trained on *all* pairs → use it for serving (a served
query is unseen by construction, so there is no leak there).

## GPU / VRAM guidance (6.4 GB)

| Model | Params | Suggested | Notes |
|---|---|---|---|
| `cross-encoder/ms-marco-MiniLM-L6-v2` | 23M | batch 32, no fp16 | the control; fast even on CPU |
| `Alibaba-NLP/gte-reranker-modernbert-base` | 149M | batch 16, `--fp16` | Apache-2.0, 8k input — best value |
| `mixedbread-ai/mxbai-rerank-large-v1` | 435M | batch 8, `--fp16` | sibling of your encoder; 512-token input |
| `BAAI/bge-reranker-v2-m3` | 568M | batch 4, `--fp16` | strongest permissive; heaviest |
| `jinaai/jina-reranker-v2-base-multilingual` | 278M | avoid | **CC-BY-NC-4.0 — non-commercial** |

If scores come back NaN, drop `--fp16` first — same failure class as the MPS NaN bug in
`finetune_embeddings.py`.

## Results

RRF top-50 → LambdaMART, out-of-fold, against `data_sat`'s 1,023 LLM-graded queries:

| Stage | NDCG@10 | P@5 | R@5 | R@10 | MRR |
|---|---|---|---|---|---|
| RRF k=60 (Stage 1 baseline) | 0.6710 | 0.116 | 0.532 | 0.774 | 0.278 |
| + cross-encoder, zero-shot (`ms-marco-MiniLM-L6-v2`) | 0.3649 | 0.064 | 0.291 | 0.407 | 0.183 |
| + LambdaMART (RRF/BM25/dense + budget/seniority/avail, no CE) | 0.6591 | 0.152 | 0.677 | 0.861 | 0.340 |

**Scale.** 1,023 hirers × 2,165 providers, 23,873 LLM-graded pairs total (grader `qwen3.8:27b`,
see `label.md`), 19,193 of those inside the top-50 RRF pool used for training. Grades cluster
low: 40.9% grade-0, 46.0% grade-1, 9.7% grade-2, 3.4% grade-3. Measured directly against this
run's `ground_truth_llm.json` (not `label.md`'s figure, which describes the earlier, superseded
774-gig `data_sat` before the data team backfilled budget/seniority/availability): **481/1,023
hirers (47.0%) have no provider graded ≥2 at all** (RELEVANCE_THRESHOLD=40 in `evaluate.py`).
`evaluate_all_hirers` excludes these from the recall average (recall is undefined with zero
relevant items) but scores them as a hard 0 in NDCG/MRR — so R@5/R@10 above are averaged over the
~542 hirers with a real match, while NDCG@10/MRR are not. A hard ceiling on P@K/NDCG@K/R@K is
expected and is a property of the corpus, not the ranker.

- **Zero-shot cross-encoder regresses hard** — confirmed out-of-domain for consulting/finance
  text. Per the project's own "zero-shot first, fine-tune only if the gain is real" rule, its
  score was **not** carried into LambdaMART as a feature (§6 above covers the fine-tuning status).
- **LambdaMART on structured + retrieval features alone**: NDCG@10 is flat vs. RRF (0.659 vs
  0.671 — inside the project's own ~0.02 noise threshold), but P@5/R@5/R@10/MRR all improve
  meaningfully. Read this as: the learned ranker isn't pulling more relevant items into the very
  top of a tied NDCG score, it's doing a better job spreading relevant items across the top-10 and
  reducing misses lower in the list.
- **Feature importance** (mean over 5 folds): `rrf_rank` 0.223, `rrf_score` 0.164, `dense_rank`
  0.154, `seniority_fit` 0.116, `avail_immediacy` 0.085, `budget_fit` 0.085, `dense_cosine` 0.070,
  `bm25_score` 0.056, `bm25_rank` 0.048. The three structured fields that needed real
  budget/seniority/availability data to compute (`seniority_fit`, `budget_fit`,
  `avail_immediacy`) rank 4th, 5th and 6th — real, non-trivial weight, which is the main payoff of
  those fields being backfilled by the data team.

**Fields backfilled.** `budget_lo/hi`, `seniority_needed` (hirers) and `rate_per_hour`,
`seniority`, `available_from`, `availability` (providers) are present inline on every record — an
earlier version of `data_sat` was missing these (see `label.md`), which paused this work until
the data team backfilled them. `data_sat` carries these fields directly (no separate
`_with_taxonomy.json` like the archived synthetic corpus), so `features.py` falls back to the
main `providers.json`/`hirers.json` when no `_with_taxonomy.json` file exists.

`avail_immediacy()` uses the structured `available_from` (provider) vs. `start_by` (hirer) date
gap, instead of matching fixed phrases like "immediately" against `data_sat`'s "Available from
`<date>`, N days a week" strings, which never match. A first version of this still had a bug
caught during review before push: `available_from`/`start_by` aren't always ISO dates — 966/2,165
providers use the literal sentinel `"now"` and 75/1,023 hirers use `"asap"` — and unhandled
`date.fromisoformat()` on those raised `ValueError`, falling through to the same non-matching
text matcher and landing on the 0.5 default for **47.5% of all candidate rows**, silently
defeating the fix. Both sentinels now resolve to `date.today()` before comparison; the
fallback-default rate dropped to 0.43%, and `avail_immediacy`'s feature importance moved from
0.054 (noise) to 0.085 once it carried real signal.

## Worked examples

Inspected directly against `results_data_sat/rrf_k60.json` / `results_data_sat/ltr_sat.json` /
`data_sat/ground_truth_llm.json`; regenerate `results_data_sat/` with
`python pipeline/run_pipeline.py --data-dir data_sat` to reproduce these.

- **Why the aggregate numbers look low even on a "success" query** (hire_id=1, "Design AI Agent
  Sales Support System for Life Insurer"): 3 grade-3 matches exist among 2,165 providers. Both
  RRF and LambdaMART surface all 3 within the top-6–9, but the other 7-of-10 slots are grade-1
  "in the neighborhood but not it" profiles — with only 3 truly-relevant providers in the whole
  catalog, P@5≈0.15 is close to the ceiling even when retrieval is working correctly.
- **A genuine LambdaMART win, and why**: hire_id=745 ("Design RCT for Refugee Cash Transfer
  Program") — RRF's top-5 is all grade-0, including two "Transfer *Pricing* Specialists": a pure
  BM25 lexical collision on the word "Transfer" (cash *Transfer* program vs. *Transfer* pricing),
  unrelated in meaning. LambdaMART's top-3 are all genuinely relevant (poverty/social-protection/
  policy-evaluation specialists) — dense embeddings + structured features override the bad
  lexical match. NDCG@10: 0.095 → 0.917.
- **A "loss" that is measurement noise, not a real regression**: hire_id=14 ("Assess Acoustic
  Performance for Staff Village Building") has exactly **one judged pair in its entire pool**
  (provider 1926, grade-1/score=33). RRF happened to rank it #1 (NDCG@10=1.0); LambdaMART ranked
  it #10 (NDCG@10=0.289). With a single graded item, NDCG@10 swings on one rank position — this
  is pool-sparsity variance, not evidence LambdaMART got the query wrong.
- **A genuine LambdaMART weakness**: hire_id=247 ("Consolidate Contract Playbooks into Single
  Storage", budget $160-215/hr) — RRF ranked grade-2 match provider 252 at #3; LambdaMART demoted
  it to #17. Provider 252's rate ($135/hr) sits *below* the hirer's budget floor, and `budget_fit`
  penalizes any rate outside `[budget_lo, budget_hi]` symmetrically — but a cheaper-than-budget
  provider is usually still a fine match, not a mismatch. `budget_fit`'s below-band penalty should
  likely be softer than its above-band penalty; this is a plausible explanation for the flat
  NDCG@10 in aggregate (LambdaMART fixes some BM25 failures like the case above while introducing
  some new ones like this). Not changed in this run — noted as a follow-up, not fixed silently.

## Guardrails (these decide whether the numbers are trustworthy)

1. **Split by query, never randomly.** 1,023 queries; a random split leaks and inflates.
2. **Labels come from `ground_truth_llm.json`**, never the retired taxonomy formula.
3. **Train/serve skew:** confirm the production gig payload actually carries `budget_lo/hi`,
   `seniority_needed`, `rate_per_hour`, `seniority`, `available_from`/`availability` before
   shipping a ranker that depends on them — otherwise the model learns on features that are
   missing at serve time. (Resolved for `data_sat` itself — see "Fields backfilled" above — this
   guardrail is about the *production* payload, a separate question.)
4. **Significance:** treat any delta under **~0.02** as noise. Only one seed/fold-split has been
   run so far — report paired bootstrap CIs before claiming a win.
5. **Calibration:** `--export-scores` writes min-max 0–100 per query for the JSON contract; not
   yet run for `data_sat`. The sponsor's merge (`0.6·taxonomy + 0.4·semantic`) uses magnitude,
   not just order — document the normalisation method, which is still an open item with the
   sponsor.
6. **Keep the plain RRF path.** The developer integrating this has no AI background; the service
   must always be able to return valid JSON without the learned components.
7. **Pre-existing bug, not `data_sat`-specific**: `retrieval_bm25.py`'s query-side field
   weighting effectively 4×-weights the title instead of the documented 3× (`title_weight=3`).
   `BM25Retriever._query_tokens` rebuilds query tokens via `hirer_bm25_tokens(h_like, ...)`, where
   `h_like["hire_description"]` is set to the *already-concatenated* `hirer_text(h)` (which itself
   includes the title) rather than the raw `hire_description` field — so the title's tokens are
   counted once via `title_tokens * title_weight` and once more inside `body_tokens`. This affects
   every BM25/RRF run in this repo, so it doesn't change the *relative* comparisons above, but it
   means "refined BM25" is stronger than its own documentation claims. Flagged, not fixed —
   fixing it moves results for both `data_sat` and the archived synthetic corpus, which needs its
   own decision.

## Outputs

| Path | What |
|---|---|
| `features_data_sat/candidates_top50.csv` | candidate pool + all features |
| `features_data_sat/train_pairs.csv` | judged subset (training labels) |
| `features_data_sat/ce_scores_<tag>.csv` | cross-encoder score per pair (LTR feature) |
| `results_data_sat/ce_<tag>.json` | CE-reranked lists |
| `results_data_sat/ltr_sat.json` | LambdaMART out-of-fold ranked lists (the "Results" table above) |
| `results_data_sat/ltr_<tag>_scores.json` | calibrated 0–100 semantic scores (with `--export-scores`) |
| `models_data_sat/ltr_sat_xgb.json`, `models_data_sat/ce-<tag>/` | trained models |
| `cache_data_sat/*.npy` | cached embeddings |

`results_data_sat/` and `cache_data_sat/` are git-ignored (regenerable, and large at this
corpus scale — the full Stage-1 sweep is ~340 MB); `features_data_sat/` and `models_data_sat/`
are committed, matching the archived synthetic corpus's convention.

An archived synthetic-data baseline (the same pipeline, run against `pipeline/data/` — a smaller,
synthetic corpus, `--data-dir data`) is preserved in `pipeline/data/`, `results/`, `features/`,
`models/` for reference; not covered here — see git history for its results and methodology.
