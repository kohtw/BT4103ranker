"""
Taxonomy tower: deterministic skill matching over the SkillsFuture taxonomy -- the third
Stage-1 retriever alongside BM25 (lexical) and dense (semantic).

Inputs (tag_taxonomy_sat.py writes the tags, normalise_taxo.py the taxonomy):
    gig_tscs.json        the TSCs (skills) each gig needs, with tag similarity
    provider_roles.json  the job role(s) each provider fits, with tag similarity
    job_role_tsc.csv     which TSCs each job role uses (~22 per role)

A provider's skills are the union of the TSCs of their job roles. For one gig:

    score(provider) = sum over TSCs the gig needs AND the provider has of
                      rarity(tsc) * gig_weight(tsc) * provider_weight(tsc)

    rarity(tsc)          idf over providers: log(N / #providers who have the TSC). Sharing a
                         niche skill ("22KV Switchgear Systems Maintenance") counts for far more
                         than a common one ("Stakeholder Management").
    gig_weight(tsc)      the gig's tag similarity for that TSC / its best tag's, in (0, 1]
    provider_weight(tsc) the same for the provider's best role that carries the TSC, so a skill
                         from a weak 3rd-choice role counts less

Nothing about the gig's own TSC count enters the score -- every provider is scored against the
same gig, so normalising by it would never change the order. Providers sharing no TSC are left
out (the ranking is partial; RRF handles that). Ties -- common, since many providers share a
role -- break on how well the provider fits their top role, then provider_id.

Run (evaluates on the SAT labels, taxonomy alone and fused with BM25 + dense):
    python pipeline/retrieval_taxonomy.py [--no-provider-weight] [--taxo-weight 1.0]
"""
import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

BASE = Path(__file__).parent
DATA_DIR = BASE / "data_sat"
TAXO_DIR = BASE / "data_taxo"
RESULTS_DIR = BASE / "results"


def load_role_tscs(taxo_dir: Path = TAXO_DIR) -> dict[int, set[int]]:
    role_tscs = defaultdict(set)
    with open(taxo_dir / "job_role_tsc.csv", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            role_tscs[int(r["role_id"])].add(int(r["tsc_id"]))
    return role_tscs


class TaxonomyRetriever:
    def __init__(self, provider_roles: dict, role_tscs: dict[int, set[int]], provider_weight: bool = True):
        """provider_roles: {provider_id: [{"role_id", "sim", ...}, ...]} (best role first)."""
        self.provider_weight = provider_weight
        self.skills: dict[int, dict[int, float]] = {}  # provider -> {tsc: provider_weight}
        self.fit: dict[int, float] = {}                 # provider -> top role similarity (tie-break)
        for pid, roles in provider_roles.items():
            pid = int(pid)
            top = roles[0]["sim"]
            skills = {}
            for r in roles:
                w = r["sim"] / top
                for t in role_tscs.get(r["role_id"], ()):
                    skills[t] = max(skills.get(t, 0.0), w)
            self.skills[pid] = skills
            self.fit[pid] = top

        n = len(self.skills)
        df = defaultdict(int)
        for skills in self.skills.values():
            for t in skills:
                df[t] += 1
        self.idf = {t: math.log(n / c) for t, c in df.items()}

        self.holders = defaultdict(list)  # tsc -> providers who have it (inverted index)
        for pid, skills in self.skills.items():
            for t in skills:
                self.holders[t].append(pid)

    def rank(self, gig_tscs: list[dict]) -> list[tuple[int, float]]:
        """gig_tscs: [{"tsc_id", "sim"}, ...] best first -> [(provider_id, score), ...] descending."""
        if not gig_tscs:
            return []
        top = gig_tscs[0]["sim"]
        scores = defaultdict(float)
        for tag in gig_tscs:
            t = tag["tsc_id"]
            base = self.idf.get(t, 0.0) * tag["sim"] / top
            for pid in self.holders.get(t, ()):
                scores[pid] += base * (self.skills[pid][t] if self.provider_weight else 1.0)
        ranked = sorted(scores.items(), key=lambda x: (-x[1], -self.fit[x[0]], x[0]))
        return [(pid, s) for pid, s in ranked if s > 0]


def main():
    import numpy as np

    from build_judging_pools_sat import embed_cached
    from corpus import hirer_text, provider_text
    from evaluate import evaluate_all_hirers
    from retrieval_bm25 import BM25Retriever
    from retrieval_dense import encode_docs, encode_queries, rank_from_raw
    from retrieval_rrf import rrf_fuse_many

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-provider-weight", dest="provider_weight", action="store_false",
                    help="count every provider TSC fully (the taxonomy pool labels are for the weighted run)")
    ap.add_argument("--taxo-weight", type=float, default=1.0, help="RRF weight of the taxonomy list")
    ap.add_argument("--k", type=int, default=60, help="RRF k")
    ap.add_argument("--data-dir", type=Path, default=DATA_DIR)
    args = ap.parse_args()
    d = args.data_dir

    load = lambda name: json.loads((d / name).read_text(encoding="utf-8"))
    hirers, providers, gt = load("hirers.json"), load("providers.json"), load("ground_truth_llm.json")
    gig_tscs, provider_roles = load("gig_tscs.json"), load("provider_roles.json")
    pids = [p["provider_id"] for p in providers]

    taxo = TaxonomyRetriever(provider_roles, load_role_tscs(), provider_weight=args.provider_weight)
    bm25 = BM25Retriever(providers, refined=True)
    doc_raw = embed_cached("sat_docs_mxbai_by_text", [provider_text(p) for p in providers], encode_docs)
    q_raw = embed_cached("sat_queries_mxbai_by_text", [hirer_text(h) for h in hirers], encode_queries)

    runs = defaultdict(dict)
    returned = []
    for i, h in enumerate(hirers):
        hid = str(h["hire_id"])
        lists = {
            "bm25": bm25.rank(hirer_text(h), query_title=h["hire_title"]),
            "dense": rank_from_raw(doc_raw, q_raw[i], pids),
            "taxonomy": taxo.rank(gig_tscs.get(hid, [])),
        }
        returned.append(len(lists["taxonomy"]))
        lists["rrf_bm25_dense"] = rrf_fuse_many([lists["bm25"], lists["dense"]], k=args.k)
        lists["rrf_bm25_dense_taxonomy"] = rrf_fuse_many(
            [lists["bm25"], lists["dense"], lists["taxonomy"]], k=args.k, weights=[1.0, 1.0, args.taxo_weight])
        for name, ranked in lists.items():
            runs[name][hid] = [pid for pid, _ in ranked]

    print(f"{len(hirers)} gigs x {len(providers)} providers | taxonomy returns per gig: "
          f"median {int(np.median(returned))}, min {min(returned)}, max {max(returned)}, "
          f"empty {sum(n == 0 for n in returned)}")
    # Labels only exist for pooled pairs, and an unjudged pair scores as irrelevant -- judged@10
    # (share of the top 10 that was graded at all, incl. grade 0) shows how much that undercounts
    # a run. "graded-only" drops unjudged providers from each ranking before scoring.
    judged_pairs = load("llm_judgments_merged.json")  # every graded pair, incl. grade 0
    print(f"\n{'run':26s} {'ndcg@10':>8s} {'p@10':>6s} {'r@10':>6s} {'mrr':>6s} {'judged@10':>10s} "
          f"{'ndcg@10 graded-only':>20s}")
    for name, results in runs.items():
        m = evaluate_all_hirers(results, gt, ks=(10,))
        judged = np.mean([np.mean([str(p) in judged_pairs.get(hid, {}) for p in r[:10]]) if r else 0.0
                          for hid, r in results.items()])
        condensed = {hid: [p for p in r if str(p) in judged_pairs.get(hid, {})] for hid, r in results.items()}
        c = evaluate_all_hirers(condensed, gt, ks=(10,))
        print(f"{name:26s} {m['ndcg@10']:8.4f} {m['precision@10']:6.3f} {m['recall@10']:6.3f} "
              f"{m['mrr']:6.3f} {judged:10.1%} {c['ndcg@10']:20.4f}")
        (RESULTS_DIR / f"sat_{name}.json").write_text(json.dumps(results), encoding="utf-8")


if __name__ == "__main__":
    main()
