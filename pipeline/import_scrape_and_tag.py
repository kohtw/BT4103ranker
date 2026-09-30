"""
Imports the real gigs and provider profiles extracted by the BT4103-Scrape-and-Tag
pipeline (docs/hirers.csv, docs/providers.csv) into this repo's JSON format, so
the Stage-1 retrievers can pool candidates for LLM judging and the Stage-2
rankers can train on real data instead of the synthetic corpus.

Writes to pipeline/data_sat/ -- a separate folder, so the synthetic data/ and
every result built on it stay untouched:
    hirers.json     [{hire_id, hire_title, hire_description, hire_description_additional_notes, ...}]
    providers.json  [{provider_id, about_title, about_description, services_offered_title,
                      services_offered_description, relevant_experience, ...}]
    id_map.json     {"hirers": {id: source_file}, "providers": {id: source_file}}

Filters (the same quality bar Scrape-and-Tag's labeller/sample.py uses for gigs):
  gigs       grounding review kept it on the first pass (no repair), current prompt
             format (has a "Deliverable:" line), real industry tag (not OTHER), no
             additional notes, unique title, 300-1200 chars once the engagement line
             is stripped.
  providers  classified PROVIDER (not UNCERTAIN) with a headline and an About
             section. Profiles with some empty fields are kept: the provider side
             is the search corpus, so it should be as large and realistic as the
             data allows -- the per-gig judging pool, not the corpus, sets LLM cost.

"Engagement duration: ..." is stripped from every gig description: Scrape-and-Tag's
extract prompt appends it to every gig, so it says nothing about fit.

Structured fields (see Scrape-and-Tag's ranker_input_spec.md) are carried through when
the CSVs have them -- the *_enriched.csv files do, the plain ones don't. Numbers are
parsed; an empty cell becomes null, never 0:
  gigs       budget_lo, budget_hi (S$/hr), seniority_needed, start_by, commitment
             (days/week), duration_weeks
  providers  rate_per_hour (S$/hr), seniority, available_from, capacity (days/week),
             availability

Run: python pipeline/import_scrape_and_tag.py [--src ../BT4103-Scrape-and-Tag/docs]
     python pipeline/import_scrape_and_tag.py --enriched     # *_enriched.csv
"""
import argparse
import csv
import json
import re
from pathlib import Path

BASE = Path(__file__).parent
OUT_DIR = BASE / "data_sat"
DEFAULT_SRC = BASE.parent.parent / "BT4103-Scrape-and-Tag" / "docs"

ENGAGEMENT_RE = re.compile(r"\s*Engagement duration:[^\n]*$", re.IGNORECASE)
HIRER_FIELDS = ["hire_title", "hire_description", "hire_description_additional_notes"]
PROVIDER_FIELDS = ["about_title", "about_description", "services_offered_title",
                   "services_offered_description", "relevant_experience"]
# carried along for analysis only; nothing in the ranker reads them
META_FIELDS = ["industry", "secondary_industry"]
# structured fields from the enriched CSVs -> parser
HIRER_STRUCT = {"budget_lo": float, "budget_hi": float, "seniority_needed": str,
                "start_by": str, "commitment": int, "duration_weeks": float}
PROVIDER_STRUCT = {"rate_per_hour": float, "seniority": str, "available_from": str,
                   "capacity": int, "availability": str}

csv.field_size_limit(10**9)


def load_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def clean(v: str) -> str:
    s = (v or "").strip()
    if s[:1] == "'" and s[1:2] in ("=", "+", "-", "@", "\t", "\r"):
        s = s[1:]  # undo extract.py's CSV-formula guard
    return "" if s == "null" else s


def parse_struct(r: dict, spec: dict) -> dict:
    """The structured fields this CSV has; empty -> None, whole floats -> int."""
    out = {}
    for k, typ in spec.items():
        if k not in r:
            continue
        v = clean(r[k])
        if not v:
            out[k] = None
        elif typ is str:
            out[k] = v
        else:
            x = float(v)
            out[k] = int(x) if typ is int or x.is_integer() else x
    return out


def written_hirers(manifest: Path) -> dict:
    """source_file -> the extract_manifest entry of the attempt that wrote its row.
    Overlapping extract runs can log several "written" entries for one file; the
    latest timestamp is the one whose row is in the (deduplicated) CSV."""
    out = {}
    with manifest.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                if (r.get("entity_type") == "HIRER" and r.get("status") == "written"
                        and r.get("timestamp", "") >= out.get(r["file"], {}).get("timestamp", "")):
                    out[r["file"]] = r
    return out


def good_hirers(rows: list[dict], manifest: dict) -> list[dict]:
    seen, keep = set(), []
    for r in rows:
        m = manifest.get(r["source_file"], {})
        desc = ENGAGEMENT_RE.sub("", clean(r["hire_description"])).strip()
        title = clean(r["hire_title"])
        if ((m.get("review") or {}).get("decision") != "keep" or m.get("repaired")
                or "Deliverable:" not in desc
                or r.get("industry", "") in ("", "OTHER")
                or clean(r["hire_description_additional_notes"])
                or not 300 <= len(desc) <= 1200
                or title.lower() in seen):
            continue
        seen.add(title.lower())
        keep.append({**r, "hire_title": title, "hire_description": desc,
                     "hire_description_additional_notes": ""})
    return keep


def good_providers(rows: list[dict]) -> list[dict]:
    return [r for r in rows
            if r["classify_label"] == "PROVIDER"
            and clean(r["about_title"]) and clean(r["about_description"])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=DEFAULT_SRC,
                    help="Scrape-and-Tag docs/ folder (default: sibling checkout)")
    ap.add_argument("--enriched", action="store_true",
                    help="read hirers_enriched.csv / providers_enriched.csv (structured fields)")
    ap.add_argument("--out", type=Path, default=OUT_DIR, help="output folder (default: data_sat/)")
    args = ap.parse_args()

    suffix = "_enriched" if args.enriched else ""
    hirers = good_hirers(load_csv(args.src / f"hirers{suffix}.csv"),
                         written_hirers(args.src / "extract_manifest.jsonl"))
    providers = good_providers(load_csv(args.src / f"providers{suffix}.csv"))

    # ids follow source_file order, so a re-import of the same CSVs gives the same ids
    hirers.sort(key=lambda r: r["source_file"])
    providers.sort(key=lambda r: r["source_file"])
    h_out = [{"hire_id": i, **{k: clean(r[k]) for k in HIRER_FIELDS + META_FIELDS},
              **parse_struct(r, HIRER_STRUCT)}
             for i, r in enumerate(hirers, 1)]
    p_out = [{"provider_id": i, **{k: clean(r[k]) for k in PROVIDER_FIELDS + META_FIELDS},
              **parse_struct(r, PROVIDER_STRUCT)}
             for i, r in enumerate(providers, 1)]
    id_map = {"hirers": {i: r["source_file"] for i, r in enumerate(hirers, 1)},
              "providers": {i: r["source_file"] for i, r in enumerate(providers, 1)}}

    args.out.mkdir(exist_ok=True)
    for name, obj in [("hirers.json", h_out), ("providers.json", p_out), ("id_map.json", id_map)]:
        (args.out / name).write_text(json.dumps(obj, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"{len(h_out)} gigs, {len(p_out)} providers -> {args.out}")
    for name, rows, spec in [("gigs", h_out, HIRER_STRUCT), ("providers", p_out, PROVIDER_STRUCT)]:
        filled = {k: sum(r.get(k) is not None for r in rows) for k in spec if k in rows[0]}
        if filled:
            print(f"  {name} structured fields filled: " + ", ".join(f"{k} {n}/{len(rows)}" for k, n in filled.items()))


if __name__ == "__main__":
    main()
