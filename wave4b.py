"""
wave4b.py  —  Wave 4b Feature Engineering
==========================================
Enriches ipl_balls_wave4.csv with player-attribute features from
player_attributes.csv and pace/spin split batting stats.

Features added
--------------
  FROM PLAYER ATTRIBUTES (per delivery):
    bowler_is_pace          1 if current bowler is pace, 0 if spin
    batsman_is_lhb          1 if current batsman is left-hand bat
    lhb_vs_spin             interaction: 1 when LHB facing spin bowler

  PACE/SPIN SPLIT (2yr rolling, leakage-free):
    batsman_sr_vs_pace      batsman's SR against pace in last 730 days
    batsman_sr_vs_spin      batsman's SR against spin in last 730 days
    pace_spin_sr_diff       = sr_vs_spin - sr_vs_pace (positive = prefers spin)

Usage
-----
  python wave4b.py
  # Reads: data/ipl_balls_wave4.csv, data/ipl_matches.csv, data/player_attributes.csv
  # Writes: data/ipl_balls_wave4b.csv

Then run pipeline.py which auto-detects wave4b.csv.
"""

import argparse
from pathlib import Path

import pandas as pd
import numpy as np
from tqdm import tqdm


WINDOW_DAYS   = 730
MIN_BALLS     = 15      # lower threshold for pace/spin splits (fewer samples per type)
GLOBAL_AVG_SR = 130.0


def compute_pace_spin_splits(balls: pd.DataFrame,
                              match_dates: pd.DataFrame,
                              bowler_group_map: dict) -> dict:
    """
    For each match, snapshot every batsman's 2yr rolling SR vs pace and vs spin.
    Returns: match_id -> { player -> {sr_vs_pace, sr_vs_spin} }
    """
    df = balls.merge(match_dates, on="match_id", how="left")
    df["bowler_group"] = df["bowler"].map(bowler_group_map).fillna("pace")
    df = df.sort_values(["date", "match_id", "innings", "over", "ball"]).reset_index(drop=True)

    # Pre-aggregate per match × batsman × bowler_group
    agg = (
        df.groupby(["match_id", "date", "batsman", "bowler_group"])
        .agg(
            runs  = ("runs_batsman", "sum"),
            legal = ("is_legal", "sum"),
        )
        .reset_index()
    )

    match_order = (
        df.groupby("match_id")["date"].first()
          .sort_values().reset_index()
    )

    splits_by_match = {}

    for _, mrow in tqdm(match_order.iterrows(), total=len(match_order),
                        desc="Pace/spin splits"):
        mid   = mrow["match_id"]
        mdate = mrow["date"]
        cutoff = mdate - pd.Timedelta(days=WINDOW_DAYS)

        window = agg[(agg["date"] >= cutoff) & (agg["date"] < mdate)]
        player_splits = {}

        for player, pg in window.groupby("batsman"):
            pace_rows = pg[pg["bowler_group"] == "pace"]
            spin_rows = pg[pg["bowler_group"] == "spin"]

            pace_legal = pace_rows["legal"].sum()
            pace_runs  = pace_rows["runs"].sum()
            spin_legal = spin_rows["legal"].sum()
            spin_runs  = spin_rows["runs"].sum()

            sr_pace = (pace_runs / pace_legal * 100) if pace_legal >= MIN_BALLS else GLOBAL_AVG_SR
            sr_spin = (spin_runs / spin_legal * 100) if spin_legal >= MIN_BALLS else GLOBAL_AVG_SR

            player_splits[player] = {"sr_vs_pace": sr_pace, "sr_vs_spin": sr_spin}

        splits_by_match[mid] = player_splits

    return splits_by_match


def main():
    parser = argparse.ArgumentParser(description="Wave 4b: player attributes + pace/spin splits")
    parser.add_argument("--balls",  default="data/ipl_balls_wave4.csv")
    parser.add_argument("--matches", default="data/ipl_matches.csv")
    parser.add_argument("--attrs",  default="data/player_attributes.csv")
    parser.add_argument("--output", default="data/ipl_balls_wave4b.csv")
    args = parser.parse_args()

    print("═" * 65)
    print("  WAVE 4b — Player Attributes + Pace/Spin Splits")
    print("═" * 65)

    balls   = pd.read_csv(args.balls)
    matches = pd.read_csv(args.matches, parse_dates=["date"])
    attrs   = pd.read_csv(args.attrs)

    balls["match_id"]   = balls["match_id"].astype(str)
    matches["match_id"] = matches["match_id"].astype(str)

    print(f"\nBalls loaded      : {len(balls):,}")
    print(f"Matches loaded    : {len(matches):,}")
    print(f"Player attributes : {len(attrs):,}")

    # ── Build lookup dicts from player_attributes ─────────────────────────
    bowler_group_map = dict(zip(attrs["player_name"], attrs["bowling_group"]))
    batting_hand_map = dict(zip(attrs["player_name"], attrs["batting_hand"]))

    # ── Static per-delivery features from attributes ──────────────────────
    print("\nAttaching static attribute features...")

    balls["bowler_is_pace"] = balls["bowler"].map(bowler_group_map).apply(
        lambda x: 1 if x == "pace" else 0).astype(np.int8)
    balls["batsman_is_lhb"] = balls["batsman"].map(batting_hand_map).apply(
        lambda x: 1 if x == "LHB" else 0).astype(np.int8)

    # LHB × spin interaction
    bowler_is_spin = balls["bowler"].map(bowler_group_map).apply(
        lambda x: 1 if x == "spin" else 0)
    balls["lhb_vs_spin"] = (balls["batsman_is_lhb"] * bowler_is_spin).astype(np.int8)

    print(f"  bowler_is_pace  : {balls['bowler_is_pace'].mean():.3f} (fraction pace)")
    print(f"  batsman_is_lhb  : {balls['batsman_is_lhb'].mean():.3f} (fraction LHB)")
    print(f"  lhb_vs_spin     : {balls['lhb_vs_spin'].mean():.3f} (fraction LHB facing spin)")

    # ── Pace/spin split SR (2yr rolling, leakage-free) ────────────────────
    match_dates = matches[["match_id", "date"]].drop_duplicates("match_id")

    splits = compute_pace_spin_splits(balls, match_dates, bowler_group_map)

    # Attach to balls
    n = len(balls)
    sr_vs_pace = np.full(n, GLOBAL_AVG_SR, dtype=np.float32)
    sr_vs_spin = np.full(n, GLOBAL_AVG_SR, dtype=np.float32)

    for i, row in tqdm(balls.iterrows(), total=n, desc="Attaching splits",
                       miniters=50000):
        mid = row["match_id"]
        bat = row.get("batsman", "")
        if mid in splits and bat in splits[mid]:
            ps = splits[mid][bat]
            sr_vs_pace[i] = ps["sr_vs_pace"]
            sr_vs_spin[i] = ps["sr_vs_spin"]

    balls["batsman_sr_vs_pace"]  = sr_vs_pace
    balls["batsman_sr_vs_spin"]  = sr_vs_spin
    balls["pace_spin_sr_diff"]   = (sr_vs_spin - sr_vs_pace).astype(np.float32)

    # ── Sanity checks ─────────────────────────────────────────────────────
    W4B_COLS = ["bowler_is_pace", "batsman_is_lhb", "lhb_vs_spin",
                "batsman_sr_vs_pace", "batsman_sr_vs_spin", "pace_spin_sr_diff"]

    print("\n── Sanity checks ───────────────────────────────────────────────────")
    for col in W4B_COLS:
        v = balls[col]
        print(f"  {col:25s}  mean={v.mean():8.3f}  std={v.std():7.3f}  "
              f"min={v.min():8.3f}  max={v.max():8.3f}")

    # ── Save ──────────────────────────────────────────────────────────────
    balls.to_csv(args.output, index=False)
    print(f"\nSaved to {args.output}  ({len(balls):,} rows, {len(balls.columns)} cols)")
    print("Done.")


if __name__ == "__main__":
    main()