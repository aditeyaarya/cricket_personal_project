"""
wave2.py  —  Wave 2 Feature Engineering
========================================
Computes three player-level lineup features on top of ipl_balls.csv
and writes the enriched table to data/ipl_balls_wave2.csv.

Features added
--------------
  batting_strength_remaining   float   Sum of career SR for undismissed
                                       batsmen NOT currently at the crease.
  bowling_strength_remaining   float   Sum of (1/career_economy) × quota
                                       balls remaining for each active bowler.
  wickets_remaining_weighted   float   Sum of (career_SR / GLOBAL_AVG_SR) for
                                       every undismissed batter — units are
                                       "average-quality batters remaining".

Leakage prevention
------------------
  Career strike rate and economy are computed with an expanding window
  over all deliveries BEFORE the current match (sorted by date).  No
  information from the current match is ever used.  Players with fewer
  than MIN_BALLS (30) legal deliveries default to the global average.

Batting order
-------------
  Inferred per innings from the order in which batsmen first appear at
  the crease (batsman + non_striker columns in chronological order).

Usage
-----
  python wave2.py                          # uses default paths
  python wave2.py --balls data/ipl_balls.csv --matches data/ipl_matches.csv

Output
------
  data/ipl_balls_wave2.csv   (original columns + 3 new columns)
"""

import argparse
from pathlib import Path

import pandas as pd
import numpy as np
from tqdm import tqdm


# ── Tuneable constants ────────────────────────────────────────────────────────
GLOBAL_AVG_SR   = 130.0   # T20 global average strike rate
GLOBAL_AVG_ECON =   8.0   # T20 global average economy rate
MIN_BALLS       =  30     # minimum legal balls for a reliable career stat
BOWLER_QUOTA    =  24     # max legal deliveries per bowler per innings


def compute_career_stats(balls: pd.DataFrame,
                         match_dates: pd.DataFrame) -> tuple[dict, dict]:
    """
    Return two dicts:
      career_sr_by_match   : { match_id -> { player -> SR before this match } }
      career_econ_by_match : { match_id -> { player -> economy before this match } }
    """
    # Attach date, sort by date
    df = balls.merge(match_dates, on="match_id", how="left")
    df = df.sort_values(["date", "match_id", "innings", "over", "ball"]) \
           .reset_index(drop=True)

    # Accumulators
    bat_runs:   dict[str, int] = {}
    bat_balls:  dict[str, int] = {}
    bowl_runs:  dict[str, int] = {}
    bowl_balls: dict[str, int] = {}

    career_sr_by_match:   dict[str, dict] = {}
    career_econ_by_match: dict[str, dict] = {}

    # Iterate match-by-match in date order
    match_order = (
        df.groupby("match_id")["date"].first()
          .sort_values().index.tolist()
    )

    for mid in tqdm(match_order, desc="Career stats"):
        mdf = df[df["match_id"] == mid]

        # ── Snapshot BEFORE this match ────────────────────────────────────
        sr_snap  = {}
        eco_snap = {}

        for p in mdf["batsman"].unique():
            b = bat_balls.get(p, 0)
            r = bat_runs.get(p, 0)
            sr_snap[p] = (r / b * 100) if b >= MIN_BALLS else GLOBAL_AVG_SR

        for p in mdf["bowler"].unique():
            b = bowl_balls.get(p, 0)
            r = bowl_runs.get(p, 0)
            eco_snap[p] = (r / (b / 6)) if b >= MIN_BALLS else GLOBAL_AVG_ECON

        # Also snapshot for non-strikers who may never face a ball this match
        for p in mdf["non_striker"].unique():
            if p not in sr_snap:
                b = bat_balls.get(p, 0)
                r = bat_runs.get(p, 0)
                sr_snap[p] = (r / b * 100) if b >= MIN_BALLS else GLOBAL_AVG_SR

        career_sr_by_match[mid]   = sr_snap
        career_econ_by_match[mid] = eco_snap

        # ── Update accumulators AFTER snapshot ────────────────────────────
        for p, grp in mdf.groupby("batsman"):
            bat_runs[p]  = bat_runs.get(p, 0)  + int(grp["runs_batsman"].sum())
            bat_balls[p] = bat_balls.get(p, 0) + int(grp.loc[grp["is_legal"] == 1, "is_legal"].sum())

        for p, grp in mdf.groupby("bowler"):
            bowl_runs[p]  = bowl_runs.get(p, 0)  + int(grp["total_runs"].sum())
            bowl_balls[p] = bowl_balls.get(p, 0) + int(grp.loc[grp["is_legal"] == 1, "is_legal"].sum())

    print(f"  Career stats for {len(bat_runs)} batsmen, {len(bowl_runs)} bowlers")
    return career_sr_by_match, career_econ_by_match


def compute_lineup_features(balls: pd.DataFrame,
                            career_sr_by_match: dict,
                            career_econ_by_match: dict) -> pd.DataFrame:
    """
    Add three columns to balls:
      batting_strength_remaining
      bowling_strength_remaining
      wickets_remaining_weighted
    Returns the augmented DataFrame (original index preserved).
    """
    n = len(balls)
    bat_str  = np.zeros(n, dtype=np.float32)
    bowl_str = np.zeros(n, dtype=np.float32)
    wkt_wt   = np.zeros(n, dtype=np.float32)

    groups = balls.groupby(["match_id", "innings"], sort=False)

    for (mid, inn), grp in tqdm(groups, desc="Lineup features"):
        grp_sorted = grp.sort_values(["over", "ball"])
        idx_arr    = grp_sorted.index.values

        # ── Infer batting order from first appearance ─────────────────────
        batting_order = []
        seen = set()
        for _, row in grp_sorted.iterrows():
            for p in [row["batsman"], row["non_striker"]]:
                if p and p not in seen:
                    seen.add(p)
                    batting_order.append(p)

        sr_lookup   = career_sr_by_match.get(mid, {})
        econ_lookup = career_econ_by_match.get(mid, {})

        dismissed    = set()
        bowler_used  = {}  # bowler -> legal balls bowled so far

        for i, (_, row) in enumerate(grp_sorted.iterrows()):
            idx = idx_arr[i]
            current_bat = row["batsman"]
            bowler      = row["bowler"]

            # ── batting_strength_remaining ────────────────────────────────
            remaining = [p for p in batting_order
                         if p not in dismissed and p != current_bat]
            bat_str[idx] = sum(sr_lookup.get(p, GLOBAL_AVG_SR) for p in remaining)

            # ── wickets_remaining_weighted ────────────────────────────────
            undismissed = [p for p in batting_order if p not in dismissed]
            wkt_wt[idx] = sum(sr_lookup.get(p, GLOBAL_AVG_SR) / GLOBAL_AVG_SR
                              for p in undismissed)

            # ── bowling_strength_remaining ────────────────────────────────
            # Value BEFORE this delivery: sum over all bowlers who have
            # been used so far (plus current bowler) of
            #   (1/economy) × quota_balls_left
            total_bowl = 0.0
            for b_name, b_used in bowler_used.items():
                b_left = max(0, BOWLER_QUOTA - b_used)
                if b_left > 0:
                    eco = econ_lookup.get(b_name, GLOBAL_AVG_ECON)
                    total_bowl += (1.0 / eco) * b_left
            # Current bowler — if first ball, full quota
            if bowler not in bowler_used:
                eco = econ_lookup.get(bowler, GLOBAL_AVG_ECON)
                total_bowl += (1.0 / eco) * BOWLER_QUOTA
            bowl_str[idx] = total_bowl

            # ── Update state AFTER recording ──────────────────────────────
            if row["wicket"] == 1 and row["player_out"]:
                dismissed.add(row["player_out"])
            if row["is_legal"] == 1:
                bowler_used[bowler] = bowler_used.get(bowler, 0) + 1

    balls = balls.copy()
    balls["batting_strength_remaining"]  = bat_str
    balls["bowling_strength_remaining"]  = bowl_str
    balls["wickets_remaining_weighted"]  = wkt_wt
    return balls


def main():
    parser = argparse.ArgumentParser(description="Wave 2: lineup strength features")
    parser.add_argument("--balls",   default="data/ipl_balls.csv")
    parser.add_argument("--matches", default="data/ipl_matches.csv")
    parser.add_argument("--output",  default="data/ipl_balls_wave2.csv")
    args = parser.parse_args()

    print("═" * 65)
    print("  WAVE 2 — Lineup Strength Features")
    print("═" * 65)

    balls   = pd.read_csv(args.balls)
    matches = pd.read_csv(args.matches, parse_dates=["date"])

    balls["match_id"]   = balls["match_id"].astype(str)
    matches["match_id"] = matches["match_id"].astype(str)

    match_dates = matches[["match_id", "date"]].drop_duplicates("match_id")

    print(f"\nBalls loaded   : {len(balls):,}")
    print(f"Matches loaded : {len(matches):,}")

    # Step 1: career stats (expanding window, no leakage)
    career_sr, career_econ = compute_career_stats(balls, match_dates)

    # Step 2: per-delivery lineup features
    balls = compute_lineup_features(balls, career_sr, career_econ)

    # Sanity checks
    print("\n── Sanity checks ───────────────────────────────────────────────────")
    for col in ["batting_strength_remaining", "bowling_strength_remaining",
                "wickets_remaining_weighted"]:
        v = balls[col]
        print(f"  {col:35s}  mean={v.mean():.1f}  min={v.min():.1f}  max={v.max():.1f}")

    # First-ball values (should be high — full lineup available)
    first_balls = balls.groupby(["match_id", "innings"]).first()
    print(f"\n  First-ball batting_str avg : {first_balls['batting_strength_remaining'].mean():.1f}")
    print(f"  First-ball wkt_rem_wt avg  : {first_balls['wickets_remaining_weighted'].mean():.2f}")

    # Save
    balls.to_csv(args.output, index=False)
    print(f"\nSaved enriched balls to {args.output}  ({len(balls):,} rows)")
    print("Done.")


if __name__ == "__main__":
    main()