# LLM relevance labels for the Scrape-and-Tag corpus

The Stage-2 rerankers were built and evaluated on the synthetic corpus in `pipeline/data/`
(130 gigs × 104 providers). This is the labelled real-data set that replaces it: gigs and
provider profiles extracted by the BT4103-Scrape-and-Tag pipeline, enriched with budget,
seniority and availability fields, with every (gig, candidate provider) pair graded 0–3 by
an LLM.

The current set is in `pipeline/data_sat/`. The first labelled set (content fit only, on the
Sep 26 extraction) is archived in `pipeline/data_sat_v1/`; see [Archived v1 set](#archived-v1-set).

## Summary

| | |
|---|---|
| Gigs | 1,023 |
| Providers (search corpus) | 2,165 |
| Graded pairs | 22,289 (every pair in every gig's pool) |
| Pool size per gig | 15–28, median 22 |
| Grader | `qwen3.8:27b`, prompt `rubric_0_3.v2`, temperature 0 |
| Grades | 0: 11,315 (50.8%) · 1: 9,749 (43.7%) · 2: 555 (2.5%) · 3: 670 (3.0%) |
| Gigs with no provider graded ≥ 2 | 481 (47%) |
| Graded | 2026-09-29, 05:16–11:30, no errors |

**v2 grades are overall fit, not content fit.** The prompt grades what the provider has done
*and* whether the budget, seniority and availability terms work. A strong content match with
a rate far over budget is graded 1. See [Caveats](#caveats) before using these labels.

## Pipeline

```
BT4103-Scrape-and-Tag/docs/{hirers,providers}_enriched.csv
  │  pipeline/import_scrape_and_tag.py --enriched     filter + convert  → hirers.json, providers.json, id_map.json
  ▼
  │  pipeline/build_judging_pools_sat.py --rrf-top 12 --method-top 8 --random 3
  │                                                   pick pairs       → judging_pools.json, pool_sources.json
  ▼
  │  BT4103-Scrape-and-Tag/labeller/judge_pools.py --prompt v2       grade → judgments.jsonl
  │     (re-exports every N calls)                                          → llm_judgments_merged.json, ground_truth_llm.json
```

### 1. Import (`import_scrape_and_tag.py --enriched`)

Reads `hirers_enriched.csv` (1,216 gigs) and `providers_enriched.csv` (2,193 providers).

**Gigs** are kept only if they pass the same quality bar Scrape-and-Tag's `labeller/sample.py` uses:
the grounding review kept the extraction on the first pass (no repair); the description has a
`Deliverable:` line; the industry tag isn't `OTHER`; there are no additional notes; the title is
unique; and the description is 300–1,200 characters. `docs/extract_manifest.jsonl` has duplicate
entries for files written by overlapping extract runs, so the import uses the latest one per file.
The `Engagement duration: …` line is stripped from the description (its value is kept as
`duration_weeks`). 1,023 gigs pass.

**Providers** are kept if classified `PROVIDER` with a headline and an About section. 2,165 pass.

**Structured fields** are carried through, with numbers parsed and empty cells as `null`:

| Gigs | Providers |
|---|---|
| `budget_lo`, `budget_hi` (S$/hour) | `rate_per_hour` (S$/hour) |
| `seniority_needed` (`mid` / `senior` / `expert`) | `seniority` (same scale) |
| `start_by` (ISO date or `asap`) | `available_from` (ISO date or `now`) |
| `commitment` (days/week, 1–5) | `capacity` (days/week, 1–5) |
| `duration_weeks` (1,022 of 1,023 filled) | `availability` (display text) |

All are 100% filled apart from `duration_weeks`. **They are synthetic**: seniority, price tier,
urgency and commitment were inferred by an LLM from each record's text; budgets, rates, dates
and capacity were then generated from those categories by Scrape-and-Tag's
`enricher/rate_card.json` (seeded). Dates count from 2026-10-01. Details are in
Scrape-and-Tag's `ranker_input_spec.md` and its handover notes.

### 2. Judging pools (`build_judging_pools_sat.py`)

Each gig gets a TREC-style pool: the union of

| Source | Picks per gig |
|---|---|
| RRF (k=60) top 12 | 12 |
| refined BM25 top 8 | 8 |
| dense (mxbai, full dim) top 8 | 8 |
| random providers outside the above | 3 |

The pools are smaller than v1's (RRF 20 + 10 + 10 + 5, median 31 per gig). Grading had to fit
the 12:00 deadline at the API's ~1 call/s limit, which allowed about 22k calls. The overlap
between sources collapses the 31 picks to 15–28 providers per gig (median 22).

Embeddings are cached in `pipeline/cache/` by a hash of each text, so a re-import only embeds
new or changed text. On this machine's CPU, embedding ~2,200 providers takes about 1.5 hours.

### 3. Grading (`labeller/judge_pools.py --prompt v2`, in the Scrape-and-Tag repo)

The prompt (`labeller/prompts/rubric_0_3_v2.md`) shows the gig and profile text, followed by the
structured terms (e.g. "Budget: S$125-185 per hour", "Availability: from 13 Oct 2026, 3 days a
week"). It asks the model to judge content fit first, then check the terms:

| Term | Minor mismatch | Serious mismatch |
|---|---|---|
| Budget | rate up to ~15% over the top of the range | rate more than ~40% over |
| Seniority (`mid < senior < expert`) | one level apart | mid vs expert |
| Availability | starts up to 2 weeks late, or 1 day/week short | starts > 1 month late, or ≥ 2 days/week short |

| Grade | Meaning |
|---|---|
| 3 | expertise directly addresses the specific need, and at most one minor mismatch |
| 2 | relevant but not perfect content, with at most minor mismatches; or excellent content with several minor mismatches |
| 1 | only surface relevance; or relevant expertise with a serious mismatch |
| 0 | no genuine content relevance, whatever the terms |

Terms can only lower a grade. A rate below budget is fine.

**Scoring** is unchanged from v1. The grade is the argmax of the model's log-probabilities over
the four grade tokens. Each record keeps the full distribution (`probs`) and its mean
(`expected_grade`).

**Run.** Grading was ordered to fit the deadline:

1. **05:16.** The dense embeddings for the final data were still computing on CPU. Every gig's
   BM25 top 8 is in its final pool whatever the dense results, so those 8,184 pairs were graded
   first.
2. **06:19.** Full pools built, and grading switched to them. Pairs already graded were skipped.
3. Gigs were graded in seeded random order, and the label files re-exported every 300 calls, so
   a run stopped early would still have left a random sample of fully graded gigs.
4. **11:30.** All 22,289 pairs graded, with no errors and no rate-limit aborts.

### 4. Export

| File | Contents | Used for |
|---|---|---|
| `llm_judgments_merged.json` | `{hire_id: {provider_id: grade}}`, all pairs including zeros | training |
| `ground_truth_llm.json` | `{hire_id: {provider_id: 0/33/67/100}}`, grade > 0 only | evaluation |

All 1,023 gigs are fully graded and exported.

## What the labels look like

**By pool source.** A pair found by several sources counts under each.

| Source | Pairs | 0 | 1 | 2 | 3 |
|---|---|---|---|---|---|
| RRF | 12,276 | 37.9% | 53.8% | 3.7% | 4.7% |
| Dense | 8,184 | 35.4% | 55.6% | 3.7% | 5.2% |
| BM25 | 8,184 | 46.4% | 46.3% | 3.0% | 4.2% |
| Random | 3,069 | 91.3% | 8.5% | 0.1% | 0.0% |

**By term.** Share of pairs graded ≥ 2 (and graded 3):

| Budget | Pairs | ≥ 2 | 3 |
|---|---|---|---|
| rate within or under budget | 11,316 | 9.1% | 5.4% |
| ≤ 15% over | 2,113 | 5.7% | 2.1% |
| 15–40% over | 3,304 | 2.2% | 0.3% |
| > 40% over | 5,556 | 0.1% | 0.0% |

| Seniority | Pairs | ≥ 2 | 3 |
|---|---|---|---|
| same level | 8,101 | 11.2% | 6.9% |
| one level apart | 12,561 | 2.6% | 0.9% |
| mid vs expert | 1,627 | 0.0% | 0.0% |

| Start date | Pairs | ≥ 2 | 3 |
|---|---|---|---|
| on time | 16,508 | 6.5% | 3.7% |
| ≤ 14 days late | 2,601 | 4.7% | 1.8% |
| 15–30 days late | 2,228 | 1.4% | 0.3% |
| > 30 days late | 952 | 0.1% | 0.1% |

| Days per week | Pairs | ≥ 2 | 3 |
|---|---|---|---|
| enough | 13,410 | 8.2% | 4.8% |
| 1 day short | 5,632 | 2.2% | 0.4% |
| ≥ 2 days short | 3,247 | 0.1% | 0.0% |

The grader applies the thresholds consistently: serious mismatches almost never get a 2 or 3.

**Per gig: providers graded ≥ 2.**

| ≥ 2 per gig | 0 | 1 | 2 | 3 | 4 | 5 | 6+ |
|---|---|---|---|---|---|---|---|
| Gigs | 481 | 239 | 120 | 82 | 52 | 23 | 26 |

Mean 1.2, median 1. 654 gigs (64%) have no grade 3. 14 gigs have nothing above 0.

By industry, the share of gigs with no provider graded ≥ 2 ranges from 37% (Professional
Services) and 39% (Financial Services, Education) to 59% (Construction & Real Estate, 128 gigs).

**Confidence.** The median top-grade probability is 0.67, and 37% of pairs are below 0.6 (v1:
24%). Mean `expected_grade` by assigned grade: 0 → 0.20, 1 → 0.83, 2 → 1.47, 3 → 2.26.

## Caveats

- **Positives are scarce.** Only 5.5% of pairs are graded ≥ 2 (v1: 13%), and 47% of gigs have
  none. Those gigs give a reranker nothing strong to learn from, and NDCG/MRR on ≥ 2 relevance
  is undefined for them. Decide whether to drop them from evaluation, report them separately,
  or evaluate on ≥ 1.
- **Seniority does most of the damage.** 1,482 of 2,193 providers are experts and 949 of 1,216
  gigs want seniors, so 56% of pairs are one level apart, a "minor" mismatch that still cuts
  the ≥ 2 rate from 11.2% to 2.6%. Most of the drop from v1 comes from how the synthetic
  seniority levels are distributed, not from content. An expert applying to a senior gig is
  arguably not a mismatch at all; the rubric treats the gap as symmetric.
- **Grade 2 is under-used.** There are fewer 2s (555) than 3s (670). The v2 definition of 2 mixes
  "good content, terms fine" with "excellent content, several minor mismatches", and the model
  seems to resolve most such cases to 1 or 3.
- **The terms are synthetic.** Budgets, rates, dates and capacity were generated by code from
  LLM-inferred categories. The labels teach a ranker how those generated values interact, not
  how real clients trade them off. Features built from the same fields will partly reproduce
  the rubric's thresholds.
- **Not comparable with v1.** Different extraction, different corpus, different pools, and a
  different question (overall fit vs content fit). Don't mix the two sets.
- **Pool bias.** Only pooled pairs are graded, and the pools are smaller than v1's (median 22
  vs 31), so a relevant provider outside the top candidates is more likely to be missed.
- **LLM labels, not human labels.** No human spot-check has been done yet.

## Reproduce

```bash
# in ranker/
python pipeline/import_scrape_and_tag.py --enriched
python pipeline/build_judging_pools_sat.py --rrf-top 12 --method-top 8 --random 3

# in BT4103-Scrape-and-Tag/  (~6.2 h at 1 call/s; rerun to resume)
py -3 labeller/judge_pools.py --model qwen3.8:27b --prompt v2 --order random --export-every 300
```

`data_sat/phase1_bm25_pools.json` is the BM25-only pool file from step 1 of the run; nothing
reads it.

If the CSVs or pool settings change, the IDs and pairs change, and the grades already in
`judgments.jsonl` no longer line up with them. Treat that as a new labelling run.

## Archived v1 set

`pipeline/data_sat_v1/` holds the first labelled set, finished 2026-09-28:

- Sep 26 extraction (plain CSVs, no structured fields): 774 gigs, 1,696 providers.
- Pools of RRF top 20 + BM25 top 10 + dense top 10 + 5 random (median 31 per gig): 23,873 pairs.
- `qwen3.8:27b` with `rubric_0_3.v1`, **content fit only** (the budget and seniority clauses of
  the synthetic rubric were dropped because the data had no such fields).
- Grades: 0: 40.9% · 1: 46.0% · 2: 9.7% · 3: 3.4%. 262 gigs (34%) had no provider graded ≥ 2.
- A 20-gig pilot on `qwen3.6:35b` agreed exactly with `qwen3.8:27b` on only 45% of pairs, and
  graded far higher (54% ≥ 2 vs 15%). Grade scales are model-specific, so never mix models.

That set's extraction has since been replaced, so its IDs don't match the current CSVs.

## Next

- Point `features.py`, `rerank_crossencoder.py` and `rerank_ltr.py` at `data_sat/`, and read
  budget, seniority and availability from `hirers.json` / `providers.json` rather than the
  synthetic `_*_with_taxonomy.json` files. `seniority_fit` only recognises exact
  `mid` / `senior` / `expert`, which matches this data.
- Decide how to handle the 481 gigs with no positive before evaluating.
- Consider a content-only (v1 prompt) pass on the same pools if a reranker needs content fit
  separately from the terms. That's another ~6 hours of grading.
- Hand-grade a sample of pairs to check the LLM grades.
