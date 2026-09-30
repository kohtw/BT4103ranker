"""
Adds the taxonomy tower's top candidates to the SAT judging pools, so the tower can be
evaluated fairly: the v2 pools were built from RRF/BM25/dense only (build_judging_pools_sat.py),
so every provider only the taxonomy tower found was unjudged and scored as irrelevant.

Per gig, the taxonomy tower's top --top providers (results/sat_taxonomy.json, written by
retrieval_taxonomy.py) join the gig's pool; pool_sources.json records "taxonomy" for each of
them, including ones already in the pool from another source. Re-running is a no-op once
they're in.

Then grade the new pairs with the same model and prompt as the v2 labels -- judge_pools.py
skips the pairs already graded -- and re-export:
    py -3 ../BT4103-Scrape-and-Tag/labeller/judge_pools.py --model qwen3.8:27b --prompt v2 --order random
    py -3 ../BT4103-Scrape-and-Tag/labeller/judge_pools.py --model qwen3.8:27b --prompt v2 --export

Run: python pipeline/extend_pools_taxonomy.py [--top 10]
"""
import argparse
import json
from pathlib import Path

BASE = Path(__file__).parent


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--data-dir", type=Path, default=BASE / "data_sat")
    ap.add_argument("--run", type=Path, default=BASE / "results" / "sat_taxonomy.json")
    args = ap.parse_args()
    d = args.data_dir

    pools = json.loads((d / "judging_pools.json").read_text(encoding="utf-8"))
    sources = json.loads((d / "pool_sources.json").read_text(encoding="utf-8"))
    run = json.loads(args.run.read_text(encoding="utf-8"))

    added = 0
    for hid, ranked in run.items():
        pool, src = set(pools[hid]), sources[hid]
        for pid in ranked[:args.top]:
            tags = src.setdefault(str(pid), [])
            if "taxonomy" not in tags:
                tags.append("taxonomy")
            if pid not in pool:
                pool.add(pid)
                added += 1
        pools[hid] = sorted(pool)
        sources[hid] = dict(sorted(src.items(), key=lambda x: int(x[0])))

    (d / "judging_pools.json").write_text(json.dumps(pools, indent=1), encoding="utf-8")
    (d / "pool_sources.json").write_text(json.dumps(sources, indent=1), encoding="utf-8")
    print(f"added {added} pairs; pools now {sum(map(len, pools.values()))} pairs over {len(pools)} gigs")


if __name__ == "__main__":
    main()
