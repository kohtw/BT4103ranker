"""
Normalises the flat Taxo.csv (Sector, Track, Job Role, TSC_CCS Title) into relational tables.

Writes to pipeline/data_taxo/ (ids are 1-based ints, assigned in order of first appearance):
    sector.csv          sector_id, name
    track.csv           track_id, sector_id, name            track   -> 1 sector
    job_role.csv        role_id, track_id, name              role    -> 1 track (many roles per track)
    tsc.csv             tsc_id, title                        one row per distinct TSC_CCS Title
    job_role_tsc.csv    role_id, tsc_id                      many-to-many: role <-> TSC
    job_role_profile.csv  role_id, description, critical_work_functions
                        from the SkillsFuture workbook (sheets Job Role_Description, Job Role_CWF_KT),
                        joined on (sector, track, role); critical work functions are "; "-joined in
                        workbook order. Skipped if the workbook isn't found (--workbook).

Identity rules (checked against the data):
    - Track names are not unique across sectors (9 are reused), so a track is keyed by (sector, track).
    - Job role names are not unique across tracks (82 are reused), and per the modelling decision a role
      belongs to exactly ONE track, so a role is keyed by (track, role) and the same name under two
      tracks becomes two role rows.
    - Exact duplicate source rows collapse to one junction row.
    - Leading/trailing whitespace is stripped from every cell.

Run: python pipeline/normalise_taxo.py [--src pipeline/data_sat_v1/Taxo.csv --out pipeline/data_taxo]
                                       [--workbook ../BT4103-Scrape-and-Tag/skillsfuture/...xlsx]
"""
import argparse
import csv
from pathlib import Path

HERE = Path(__file__).parent
DEFAULT_WORKBOOK = (HERE.parent.parent / "BT4103-Scrape-and-Tag" / "skillsfuture"
                    / "jobsandskills-skillsfuture-skills-framework-dataset.xlsx")


def _cell(v) -> str:
    return v.strip() if isinstance(v, str) else ("" if v is None else str(v))


def role_profiles(workbook: Path, role_key: dict) -> list[tuple]:
    """role_key: {(sector, track, role): role_id} -> [(role_id, description, critical_work_functions)]."""
    import openpyxl  # only needed for this table

    wb = openpyxl.load_workbook(workbook, read_only=True)

    def sheet(name):
        it = wb[name].iter_rows(values_only=True)
        header = next(it)
        for r in it:
            row = dict(zip(header, r))
            yield (_cell(row["Sector"]), _cell(row["Track"]), _cell(row["Job Role"])), row

    desc, cwfs = {}, {}
    for key, row in sheet("Job Role_Description"):
        if key in role_key:
            desc[role_key[key]] = _cell(row["Job Role Description"])
    for key, row in sheet("Job Role_CWF_KT"):
        if key in role_key:
            seen = cwfs.setdefault(role_key[key], {})
            seen[_cell(row["Critical Work Function"])] = None
    missing = len(role_key) - len(desc)
    if missing:
        print(f"  {missing} roles have no workbook description")
    return [(rid, desc.get(rid, ""), "; ".join(c for c in cwfs.get(rid, {}) if c))
            for rid in sorted(role_key.values())]


def normalise(src: Path, out: Path, workbook: Path | None = None) -> None:
    sectors, tracks, roles, tscs = {}, {}, {}, {}
    links = {}  # (role_id, tsc_id) -> None; dict keeps first-seen order

    with open(src, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            sector = row["Sector"].strip()
            track = row["Track"].strip()
            role = row["Job Role"].strip()
            title = row["TSC_CCS Title"].strip()

            sector_id = sectors.setdefault(sector, len(sectors) + 1)
            track_id = tracks.setdefault((sector_id, track), len(tracks) + 1)
            role_id = roles.setdefault((track_id, role), len(roles) + 1)
            tsc_id = tscs.setdefault(title, len(tscs) + 1)
            links[(role_id, tsc_id)] = None

    out.mkdir(parents=True, exist_ok=True)

    def write(name, header, rows):
        with open(out / name, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(rows)
        print(f"{name:18s} {len(rows):>6d} rows")

    write("sector.csv", ["sector_id", "name"], [(i, n) for n, i in sectors.items()])
    write("track.csv", ["track_id", "sector_id", "name"], [(i, s, n) for (s, n), i in tracks.items()])
    write("job_role.csv", ["role_id", "track_id", "name"], [(i, t, n) for (t, n), i in roles.items()])
    write("tsc.csv", ["tsc_id", "title"], [(i, n) for n, i in tscs.items()])
    write("job_role_tsc.csv", ["role_id", "tsc_id"], list(links))

    if workbook and workbook.exists():
        sector_name = {i: n for n, i in sectors.items()}
        track_of = {i: (sector_name[s], n) for (s, n), i in tracks.items()}
        role_key = {(*track_of[t], n): i for (t, n), i in roles.items()}
        write("job_role_profile.csv", ["role_id", "description", "critical_work_functions"],
              role_profiles(workbook, role_key))
    else:
        print(f"job_role_profile.csv skipped: workbook not found at {workbook}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, default=HERE / "data_sat_v1" / "Taxo.csv")
    ap.add_argument("--out", type=Path, default=HERE / "data_taxo")
    ap.add_argument("--workbook", type=Path, default=DEFAULT_WORKBOOK)
    args = ap.parse_args()
    normalise(args.src, args.out, args.workbook)
