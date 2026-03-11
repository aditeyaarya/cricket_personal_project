"""
build_dataset.py  —  Multi-League Dataset Builder
==================================================
Combines IPL, T20I (filtered), and all domestic league data into two
master files used by the pipeline:

  data/all_matches.csv
  data/all_balls.csv      ← wave4b-enriched balls where available,
                             raw balls otherwise (pipeline auto-runs
                             wave4_leagues.py for missing enrichment)

Each row is tagged with a `source` column and an integer `league_id`
so the model can learn league-specific scoring patterns.

League configuration
--------------------
Edit LEAGUES below to add or remove competitions. Each entry defines:
  key         short name used as source tag and for filenames
  matches     path to parsed matches CSV
  balls       path to parsed balls CSV (wave4b preferred)
  balls_raw   fallback path if wave4b not yet computed
  gender      filter: "male", "female", or None (keep all)
  min_date    optional YYYY-MM-DD cutoff (extra guard; parser already
              applies SMAT cutoff but this is a belt-and-braces check)

Run order
---------
  1. python data_parser_leagues.py --league bbl --input data/raw/bbl
     (repeat for cpl, psl, sa20, smat)
  2. python filter_t20i.py          (if not already done)
  3. python build_dataset.py        (produces all_matches / all_balls)
  4. python pipeline.py             (auto-runs wave4_leagues.py if needed)

Usage
-----
  python build_dataset.py
  python build_dataset.py --output-dir data/
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path


# ════════════════════════════════════════════════════════════════════════════
# LEAGUE CONFIGURATION
# ════════════════════════════════════════════════════════════════════════════

LEAGUES = [
    {
        "key"      : "ipl",
        "matches"  : "data/ipl_matches.csv",
        "balls"    : "data/ipl_balls_wave4b.csv",
        "balls_raw": "data/ipl_balls_wave4.csv",
        "gender"   : "male",
        "min_date" : None,
    },
    {
        "key"      : "t20i",
        "matches"  : "data/t20i_matches_filtered.csv",
        "balls"    : "data/t20i_balls_wave4b.csv",
        "balls_raw": "data/t20i_balls_filtered.csv",
        "gender"   : "male",
        "min_date" : None,
    },
    {
        "key"      : "bbl",
        "matches"  : "data/bbl_matches.csv",
        "balls"    : "data/bbl_balls_wave4b.csv",
        "balls_raw": "data/bbl_balls.csv",
        "gender"   : "male",
        "min_date" : None,
    },
    {
        "key"      : "cpl",
        "matches"  : "data/cpl_matches.csv",
        "balls"    : "data/cpl_balls_wave4b.csv",
        "balls_raw": "data/cpl_balls.csv",
        "gender"   : "male",
        "min_date" : None,
    },
    {
        "key"      : "psl",
        "matches"  : "data/psl_matches.csv",
        "balls"    : "data/psl_balls_wave4b.csv",
        "balls_raw": "data/psl_balls.csv",
        "gender"   : "male",
        "min_date" : None,
    },
    {
        "key"      : "sa20",
        "matches"  : "data/sa20_matches.csv",
        "balls"    : "data/sa20_balls_wave4b.csv",
        "balls_raw": "data/sa20_balls.csv",
        "gender"   : "male",
        "min_date" : None,
    },
    {
        "key"      : "smat",
        "matches"  : "data/smat_matches.csv",
        "balls"    : "data/smat_balls_wave4b.csv",
        "balls_raw": "data/smat_balls.csv",
        "gender"   : "male",
        "min_date" : "2018-01-01",   # belt-and-braces; parser already filters
    },
]

# Stable integer encoding for the league_id feature
LEAGUE_ID = {cfg["key"]: i for i, cfg in enumerate(LEAGUES)}


# ════════════════════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════════════════════

def load_league(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, bool]:
    """
    Load matches + balls for one league.
    Returns (matches_df, balls_df, wave_features_present).
    Skips silently if no files exist yet.
    """
    key        = cfg["key"]
    m_path     = Path(cfg["matches"])
    b_path     = Path(cfg["balls"])
    b_raw_path = Path(cfg["balls_raw"])

    if not m_path.exists():
        print(f"  [{key.upper():5s}] matches file not found — skipping")
        return None, None, False

    matches = pd.read_csv(m_path, parse_dates=["date"])
    matches["match_id"] = matches["match_id"].astype(str)

    # Gender filter
    if cfg["gender"] and "gender" in matches.columns:
        before = len(matches)
        matches = matches[matches["gender"] == cfg["gender"]].copy()
        if len(matches) < before:
            print(f"  [{key.upper():5s}] gender filter: {before:,} → {len(matches):,} matches")

    # Date cutoff
    if cfg["min_date"]:
        cutoff = pd.Timestamp(cfg["min_date"])
        before = len(matches)
        matches = matches[matches["date"] >= cutoff].copy()
        if len(matches) < before:
            print(f"  [{key.upper():5s}] date cutoff {cfg['min_date']}: "
                  f"{before:,} → {len(matches):,} matches")

    valid_ids = set(matches["match_id"])

    # Prefer wave4b balls, fall back to raw
    wave_present = False
    if b_path.exists():
        balls = pd.read_csv(b_path)
        wave_present = True
    elif b_raw_path.exists():
        balls = pd.read_csv(b_raw_path)
        print(f"  [{key.upper():5s}] wave4b not found — using raw balls "
              f"(pipeline will enrich automatically)")
    else:
        print(f"  [{key.upper():5s}] no balls file found — skipping")
        return None, None, False

    balls["match_id"] = balls["match_id"].astype(str)
    balls = balls[balls["match_id"].isin(valid_ids)].copy()

    # Tag
    matches["source"]    = key
    matches["league_id"] = LEAGUE_ID[key]
    balls["source"]      = key
    balls["league_id"]   = LEAGUE_ID[key]

    print(f"  [{key.upper():5s}] {len(matches):,} matches  {len(balls):,} balls"
          f"  wave={'✓' if wave_present else '✗'}")

    return matches, balls, wave_present


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Multi-league dataset builder")
    parser.add_argument("--output-dir", default="data",
                        help="Output directory (default: data/)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("═" * 65)
    print("  BUILD DATASET — Multi-League Combiner")
    print("═" * 65)
    print()

    all_matches = []
    all_balls   = []
    missing_wave = []

    for cfg in LEAGUES:
        matches, balls, wave_present = load_league(cfg)
        if matches is None:
            continue
        all_matches.append(matches)
        all_balls.append(balls)
        if not wave_present:
            missing_wave.append(cfg["key"])

    if not all_matches:
        raise RuntimeError("No league data found. Run data_parser_leagues.py first.")

    # ── Combine ───────────────────────────────────────────────────────────
    matches_df = pd.concat(all_matches, ignore_index=True)
    balls_df   = pd.concat(all_balls,   ignore_index=True)
    matches_df = matches_df.sort_values("date").reset_index(drop=True)

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{'─'*65}")
    print(f"  Combined: {len(matches_df):,} matches  {len(balls_df):,} balls")
    print(f"\n  By league:")
    for key, grp in matches_df.groupby("source"):
        dates = pd.to_datetime(grp["date"])
        print(f"    {key.upper():6s}: {len(grp):,} matches  "
              f"({dates.min().date()} → {dates.max().date()})")

    if missing_wave:
        print(f"\n  Leagues missing wave4b enrichment: {', '.join(missing_wave).upper()}")
        print(f"  Pipeline will auto-run wave4_leagues.py for these.")

    # ── Save ──────────────────────────────────────────────────────────────
    matches_out = output_dir / "all_matches.csv"
    balls_out   = output_dir / "all_balls.csv"

    matches_df.to_csv(matches_out, index=False)
    balls_df.to_csv(balls_out,     index=False)

    print(f"\nSaved → {matches_out}")
    print(f"Saved → {balls_out}")
    print("Done.")


if __name__ == "__main__":
    main()