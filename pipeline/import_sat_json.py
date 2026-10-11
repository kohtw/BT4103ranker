"""
Imports Scrape-and-Tag's schema v2 JSON (data/output/hirers.json, providers.json) into this
repo's data_sat-style format, so the Stage-1 retrievers can pool pairs for the v5 grader
(BT4103-Scrape-and-Tag/labeller/grade_pairs.py). Unlike import_scrape_and_tag.py, no
quality filter is applied and there are no structured budget/seniority/date fields: v5
works those out from the text.

id_map.json maps each integer id back to the record's source_file, which is the id
grade_pairs.py uses.

Run: python pipeline/import_sat_json.py [--src ../BT4103-Scrape-and-Tag/data/output] [--out pipeline/data_sat_v5]
"""
import argparse
import json
import re
from pathlib import Path

BASE = Path(__file__).parent
DEFAULT_SRC = BASE.parent.parent / "BT4103-Scrape-and-Tag" / "data" / "output"
ENGAGEMENT_RE = re.compile(r"\s*Engagement duration:[^\n]*$", re.IGNORECASE)


def hirer(i: int, r: dict) -> dict:
    desc = ENGAGEMENT_RE.sub("", (r.get("short_description") or "").strip()).strip()
    cats = (r.get("tags") or {}).get("categories") or []
    return {"hire_id": i, "hire_title": (r.get("gig_title") or "").strip(),
            "hire_description": desc,
            "hire_description_additional_notes": (r.get("additional_notes") or "").strip(),
            "industry": cats[0]["name"] if cats else ""}


def provider(i: int, r: dict) -> dict:
    services = r.get("services") or []
    about = " ".join(x for x in [r.get("about_headline"), r.get("about_bio")] if x)
    cats = (r.get("tags") or {}).get("categories") or []
    return {"provider_id": i, "about_title": (r.get("title") or "").strip(),
            "about_description": about,
            "services_offered_title": "; ".join(s.get("service_title", "") for s in services),
            "services_offered_description": " ".join(s.get("service_detail", "") for s in services),
            "relevant_experience": " ".join(r.get("achievements") or []),
            "industry": cats[0]["name"] if cats else ""}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=DEFAULT_SRC)
    ap.add_argument("--out", type=Path, default=BASE / "data_sat_v5")
    args = ap.parse_args()

    hirers = sorted(json.loads((args.src / "hirers.json").read_text(encoding="utf-8")),
                    key=lambda r: r["source_file"])
    providers = sorted(json.loads((args.src / "providers.json").read_text(encoding="utf-8")),
                       key=lambda r: r["source_file"])
    for name, rows in [("hirers", hirers), ("providers", providers)]:
        files = [r["source_file"] for r in rows]
        if len(set(files)) != len(files):
            raise SystemExit(f"duplicate source_file in {name}.json")
    h_out = [hirer(i, r) for i, r in enumerate(hirers, 1)]
    p_out = [provider(i, r) for i, r in enumerate(providers, 1)]
    id_map = {"hirers": {i: r["source_file"] for i, r in enumerate(hirers, 1)},
              "providers": {i: r["source_file"] for i, r in enumerate(providers, 1)}}

    args.out.mkdir(exist_ok=True)
    for name, obj in [("hirers.json", h_out), ("providers.json", p_out), ("id_map.json", id_map)]:
        (args.out / name).write_text(json.dumps(obj, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"{len(h_out)} gigs, {len(p_out)} providers -> {args.out}")


if __name__ == "__main__":
    main()
