"""
One-time taxonomy tagging for the Scrape-and-Tag corpus (pipeline/data_sat/) -- the input to
the taxonomy tower (retrieval_taxonomy.py).

    gigs      -> the TSCs (skills) the gig needs
    providers -> the SkillsFuture job role(s) they fit; sector and track follow from the role

Pure embedding similarity, no LLM: BAAI/bge-m3 (run locally) embeds both sides and each item
keeps its nearest taxonomy entries. bge-m3 is deliberately a different model from the dense
tower's mxbai, so the two towers' errors are less correlated when fused.

Taxonomy texts (from pipeline/data_taxo/, see normalise_taxo.py):
    TSC       "<title>: <description>" -- one text per distinct description in
              tcs_descriptions.csv (272 titles are reworded per sector, up to 13 variants);
              a gig's similarity to a TSC is its best variant. The 2 TSCs with no description
              are embedded from the title alone.
    job role  "<role>. <track> track, <sector> sector. <description> Critical work functions: ..."

Selection, per item (sorted by cosine similarity):
    keep entries with sim >= top_sim - margin, then clamp the count to [min, max].
A tight margin keeps a vague gig to its few clear tags instead of padding it with noise;
the minimum makes sure every gig and provider gets tagged.

Writes to pipeline/data_sat/:
    gig_tscs.json        {hire_id: [{"tsc_id", "title", "sim"}, ...]}        best first
    provider_roles.json  {provider_id: [{"role_id", "role", "track", "sector", "sim"}, ...]}

Embeddings are cached under pipeline/cache/ keyed by text hash (same scheme as
build_judging_pools_sat.py), so re-tuning the thresholds doesn't re-embed.

Run: python pipeline/tag_taxonomy_sat.py [--gig-margin 0.05 --gig-min 3 --gig-max 8
                                          --role-margin 0.03 --role-min 1 --role-max 3] [--show 5]
"""
import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

from build_judging_pools_sat import embed_cached
from corpus import hirer_text, provider_text

BASE = Path(__file__).parent
DATA_DIR = BASE / "data_sat"
TAXO_DIR = BASE / "data_taxo"
MODEL_NAME = "BAAI/bge-m3"
MAX_SEQ_LENGTH = 512  # provider texts median ~300 tokens; the cap keeps CPU encoding tractable

_model = None


def encode(texts: list[str]) -> np.ndarray:
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(MODEL_NAME)
        _model.max_seq_length = MAX_SEQ_LENGTH
    return _model.encode(texts, batch_size=16, normalize_embeddings=True,
                         convert_to_numpy=True, show_progress_bar=True)


def read_csv(name: str) -> list[dict]:
    with open(TAXO_DIR / name, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def load_taxonomy():
    sectors = {r["sector_id"]: r["name"] for r in read_csv("sector.csv")}
    tracks = {r["track_id"]: (r["name"], sectors[r["sector_id"]]) for r in read_csv("track.csv")}
    profiles = {r["role_id"]: r for r in read_csv("job_role_profile.csv")}
    roles = []
    for r in read_csv("job_role.csv"):
        track, sector = tracks[r["track_id"]]
        prof = profiles.get(r["role_id"], {})
        text = f"{r['name']}. {track} track, {sector} sector. {prof.get('description', '')}"
        if prof.get("critical_work_functions"):
            text += f" Critical work functions: {prof['critical_work_functions']}."
        roles.append({"role_id": int(r["role_id"]), "role": r["name"], "track": track,
                      "sector": sector, "text": text})

    descs = defaultdict(dict)  # title -> {description: None}, first-seen order
    for r in read_csv("tcs_descriptions.csv"):
        d = r["TSC_CCS Description"].strip()
        if d:
            descs[r["TSC_CCS Title"].strip()][d] = None
    tscs, variants, variant_owner = [], [], []
    for i, r in enumerate(read_csv("tsc.csv")):
        tscs.append({"tsc_id": int(r["tsc_id"]), "title": r["title"]})
        for d in descs.get(r["title"]) or [None]:
            variants.append(f"{r['title']}: {d}" if d else r["title"])
            variant_owner.append(i)
    return tscs, variants, np.array(variant_owner), roles


def select(sims: np.ndarray, margin: float, k_min: int, k_max: int) -> list[int]:
    """Indices of the entries to keep for one item, best first."""
    order = np.argsort(-sims)[:k_max]
    keep = [j for j in order if sims[j] >= sims[order[0]] - margin]
    return [int(j) for j in order[:max(k_min, len(keep))]]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gig-margin", type=float, default=0.05)
    ap.add_argument("--gig-min", type=int, default=3)
    ap.add_argument("--gig-max", type=int, default=8)
    ap.add_argument("--role-margin", type=float, default=0.03)
    ap.add_argument("--role-min", type=int, default=1)
    ap.add_argument("--role-max", type=int, default=3)
    ap.add_argument("--show", type=int, default=0, help="print N random gigs and providers with their tags")
    ap.add_argument("--data-dir", type=Path, default=DATA_DIR)
    args = ap.parse_args()

    hirers = json.loads((args.data_dir / "hirers.json").read_text(encoding="utf-8"))
    providers = json.loads((args.data_dir / "providers.json").read_text(encoding="utf-8"))
    tscs, variants, variant_owner, roles = load_taxonomy()
    print(f"{len(hirers)} gigs, {len(providers)} providers | {len(tscs)} TSCs ({len(variants)} texts), "
          f"{len(roles)} job roles | {MODEL_NAME}")

    tsc_vecs = embed_cached("taxo_tsc_bgem3", variants, encode)
    role_vecs = embed_cached("taxo_role_bgem3", [r["text"] for r in roles], encode)
    gig_vecs = embed_cached("sat_queries_bgem3", [hirer_text(h) for h in hirers], encode)
    prov_vecs = embed_cached("sat_docs_bgem3", [provider_text(p) for p in providers], encode)

    # gig x TSC similarity = best matching description variant of that TSC
    var_sims = gig_vecs @ tsc_vecs.T
    gig_sims = np.full((len(hirers), len(tscs)), -1.0, dtype=np.float32)
    np.maximum.at(gig_sims.T, variant_owner, var_sims.T)
    role_sims = prov_vecs @ role_vecs.T

    gig_tags = {}
    for h, sims in zip(hirers, gig_sims):
        gig_tags[str(h["hire_id"])] = [
            {"tsc_id": tscs[j]["tsc_id"], "title": tscs[j]["title"], "sim": round(float(sims[j]), 4)}
            for j in select(sims, args.gig_margin, args.gig_min, args.gig_max)]
    prov_tags = {}
    for p, sims in zip(providers, role_sims):
        prov_tags[str(p["provider_id"])] = [
            {k: roles[j][k] for k in ("role_id", "role", "track", "sector")} | {"sim": round(float(sims[j]), 4)}
            for j in select(sims, args.role_margin, args.role_min, args.role_max)]

    (args.data_dir / "gig_tscs.json").write_text(json.dumps(gig_tags, indent=1, ensure_ascii=False), encoding="utf-8")
    (args.data_dir / "provider_roles.json").write_text(json.dumps(prov_tags, indent=1, ensure_ascii=False), encoding="utf-8")

    for name, tags in [("TSCs per gig", gig_tags), ("roles per provider", prov_tags)]:
        n = np.array([len(v) for v in tags.values()])
        top = np.array([v[0]["sim"] for v in tags.values()])
        print(f"{name}: mean {n.mean():.2f}, dist {np.bincount(n).tolist()} | top sim "
              f"p10 {np.percentile(top, 10):.3f} median {np.median(top):.3f} p90 {np.percentile(top, 90):.3f}")
    print(f"distinct TSCs used {len({t['tsc_id'] for v in gig_tags.values() for t in v})}, "
          f"distinct roles used {len({t['role_id'] for v in prov_tags.values() for t in v})}")
    print(f"written to {args.data_dir}")

    rng = random.Random(0)
    for h in rng.sample(hirers, args.show):
        print(f"\nGIG {h['hire_id']}: {h['hire_title']}")
        for t in gig_tags[str(h["hire_id"])]:
            print(f"   {t['sim']:.3f}  {t['title']}")
    for p in rng.sample(providers, args.show):
        print(f"\nPROVIDER {p['provider_id']}: {p['about_title']}")
        for t in prov_tags[str(p["provider_id"])]:
            print(f"   {t['sim']:.3f}  {t['role']}  [{t['track']} / {t['sector']}]")


if __name__ == "__main__":
    main()
