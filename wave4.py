"""
wave4.py  —  Wave 4 Feature Engineering
========================================
Computes rolling player form features and advanced game-state features
on top of ipl_balls_wave2.csv.  Writes data/ipl_balls_wave4.csv.

Features added
--------------
  PER-DELIVERY PLAYER FORM (2-year rolling window, leakage-free):
    batsman_sr_2yr           Current batsman's strike rate over last 730 days
    batsman_boundary_pct_2yr Boundary frequency (4s+6s / legal balls)
    batsman_dot_pct_2yr      Dot ball percentage
    bowler_economy_2yr       Current bowler's economy over last 730 days
    bowler_dot_pct_2yr       Bowler's dot ball percentage

  GAME-STATE:
    momentum_index           Batting acceleration vs own 24-ball baseline
    over_par                 Runs ahead/behind venue median at this over
                             (innings 1 only — KEY first innings signal)

Leakage prevention
------------------
  Rolling stats use only deliveries within the 730-day window ENDING
  the day before the current match (same match excluded).
  over_par uses only training-set matches (dates < val cutoff).

Usage
-----
  python wave4.py                                # default paths
  python wave4.py --balls data/ipl_balls_wave2.csv --matches data/ipl_matches.csv

Output
------
  data/ipl_balls_wave4.csv  (all previous columns + 7 new columns)
"""

import argparse
from pathlib import Path
from collections import defaultdict

import pandas as pd
import numpy as np
from tqdm import tqdm


# ── Constants ─────────────────────────────────────────────────────────────────
WINDOW_DAYS     = 730      # 2 IPL seasons
MIN_BAT_BALLS   = 20       # minimum sample for rolling batsman stats
MIN_BOWL_BALLS  = 20       # minimum sample for rolling bowler stats
GLOBAL_AVG_SR   = 130.0
GLOBAL_AVG_ECON = 8.0
GLOBAL_DOT_PCT  = 0.42     # approximate T20 dot ball rate
GLOBAL_BDRY_PCT = 0.14     # approximate boundary rate


# ════════════════════════════════════════════════════════════════════════════
# 1.  ROLLING PLAYER STATS (2-year window, per-match snapshots)
# ════════════════════════════════════════════════════════════════════════════

def compute_rolling_player_stats(balls: pd.DataFrame,
                                  match_dates: pd.DataFrame) -> dict:
    """
    For each match, snapshot every player's 2yr rolling stats computed
    from deliveries in matches within (current_date - 730 days, current_date).
    Same-match deliveries are excluded (leakage-free).

    Returns dict:  match_id -> {
        'bat': { player -> {sr, bdry_pct, dot_pct} },
        'bowl': { player -> {econ, dot_pct} }
    }
    """
    df = balls.merge(match_dates, on="match_id", how="left")
    df = df.sort_values(["date", "match_id", "innings", "over", "ball"]).reset_index(drop=True)

    # Pre-aggregate per-match per-player stats for fast windowed lookup
    # Batsman aggregates
    bat_agg = (
        df.groupby(["match_id", "date", "batsman"])
        .agg(
            bat_runs    = ("runs_batsman", "sum"),
            bat_legal   = ("is_legal", "sum"),
            bat_dots    = ("runs_batsman", lambda x: (x == 0).sum()),   # dots while batting
            bat_fours   = ("runs_batsman", lambda x: (x == 4).sum()),
            bat_sixes   = ("runs_batsman", lambda x: (x == 6).sum()),
        )
        .reset_index()
    )
    # Count only legal balls for dot percentage
    legal_mask = df["is_legal"] == 1
    bat_dot_legal = (
        df[legal_mask]
        .groupby(["match_id", "date", "batsman"])
        .agg(bat_dots_legal = ("runs_batsman", lambda x: (x == 0).sum()))
        .reset_index()
    )
    bat_agg = bat_agg.merge(bat_dot_legal, on=["match_id", "date", "batsman"], how="left")
    bat_agg["bat_dots_legal"] = bat_agg["bat_dots_legal"].fillna(0).astype(int)

    # Bowler aggregates
    bowl_agg = (
        df.groupby(["match_id", "date", "bowler"])
        .agg(
            bowl_runs   = ("total_runs", "sum"),
            bowl_legal  = ("is_legal", "sum"),
        )
        .reset_index()
    )
    bowl_dot_legal = (
        df[legal_mask]
        .groupby(["match_id", "date", "bowler"])
        .agg(bowl_dots_legal = ("total_runs", lambda x: (x == 0).sum()))
        .reset_index()
    )
    bowl_agg = bowl_agg.merge(bowl_dot_legal, on=["match_id", "date", "bowler"], how="left")
    bowl_agg["bowl_dots_legal"] = bowl_agg["bowl_dots_legal"].fillna(0).astype(int)

    # Sort matches chronologically
    match_order = (
        df.groupby("match_id")["date"].first()
          .sort_values().reset_index()
    )

    # For each match, compute rolling window stats
    stats_by_match = {}

    for _, mrow in tqdm(match_order.iterrows(), total=len(match_order), desc="Rolling 2yr stats"):
        mid  = mrow["match_id"]
        mdate = mrow["date"]
        cutoff = mdate - pd.Timedelta(days=WINDOW_DAYS)

        # Batsman window: matches in [cutoff, mdate) — excludes current match
        bw = bat_agg[(bat_agg["date"] >= cutoff) & (bat_agg["date"] < mdate)]
        bat_stats = {}
        for player, pg in bw.groupby("batsman"):
            legal = pg["bat_legal"].sum()
            runs  = pg["bat_runs"].sum()
            fours = pg["bat_fours"].sum()
            sixes = pg["bat_sixes"].sum()
            dots  = pg["bat_dots_legal"].sum()

            if legal >= MIN_BAT_BALLS:
                bat_stats[player] = {
                    "sr":       runs / legal * 100,
                    "bdry_pct": (fours + sixes) / legal,
                    "dot_pct":  dots / legal,
                }
            else:
                bat_stats[player] = {
                    "sr": GLOBAL_AVG_SR, "bdry_pct": GLOBAL_BDRY_PCT, "dot_pct": GLOBAL_DOT_PCT
                }

        # Bowler window
        bwl = bowl_agg[(bowl_agg["date"] >= cutoff) & (bowl_agg["date"] < mdate)]
        bowl_stats = {}
        for player, pg in bwl.groupby("bowler"):
            legal = pg["bowl_legal"].sum()
            runs  = pg["bowl_runs"].sum()
            dots  = pg["bowl_dots_legal"].sum()

            if legal >= MIN_BOWL_BALLS:
                bowl_stats[player] = {
                    "econ":    runs / (legal / 6),
                    "dot_pct": dots / legal,
                }
            else:
                bowl_stats[player] = {"econ": GLOBAL_AVG_ECON, "dot_pct": GLOBAL_DOT_PCT}

        stats_by_match[mid] = {"bat": bat_stats, "bowl": bowl_stats}

    print(f"  Rolling stats computed for {len(match_order)} matches")
    return stats_by_match


# ════════════════════════════════════════════════════════════════════════════
# 2.  OVER-PAR FEATURE (venue × over median score, training set only)
# ════════════════════════════════════════════════════════════════════════════

def compute_over_par(balls: pd.DataFrame,
                     match_dates: pd.DataFrame,
                     train_cutoff_date,
                     matches_csv: str = "data/ipl_matches.csv") -> pd.DataFrame:
    """
    For each venue × over, compute median cumulative score at that over
    from training-set innings-1 data only.  Returns a lookup DataFrame.
    """
    df = balls[balls["innings"] == 1].copy()
    df["match_id"] = df["match_id"].astype(str)
    df = df.merge(match_dates, on="match_id", how="left")

    # Training set only (no leakage)
    df = df[df["date"] < train_cutoff_date].copy()

    # Need venue — merge from matches
    # Actually we need venue on balls. Get it from the match.
    # balls doesn't have venue, so we need to add it
    matches_venue = pd.read_csv(matches_csv)[["match_id", "venue"]]
    matches_venue["match_id"] = matches_venue["match_id"].astype(str)
    df = df.merge(matches_venue, on="match_id", how="left")

    # Compute cumulative runs per match-innings at each over boundary
    df["over_completed"] = df.groupby(["match_id"])["is_legal"].cumsum() // 6

    # Get cumulative score at each over for each match
    over_scores = (
        df.groupby(["match_id", "venue", "over_completed"])["total_runs"]
        .sum()
        .groupby(level=[0, 1]).cumsum()
        .reset_index()
        .rename(columns={"total_runs": "cum_runs"})
    )

    # Median by venue × over
    venue_over_median = (
        over_scores.groupby(["venue", "over_completed"])["cum_runs"]
        .median()
        .reset_index()
        .rename(columns={"cum_runs": "venue_median_at_over"})
    )

    # Also compute global median for unseen venues
    global_over_median = (
        over_scores.groupby("over_completed")["cum_runs"]
        .median()
        .reset_index()
        .rename(columns={"cum_runs": "global_median_at_over"})
    )

    return venue_over_median, global_over_median


# ════════════════════════════════════════════════════════════════════════════
# 3.  ATTACH ALL FEATURES TO BALLS
# ════════════════════════════════════════════════════════════════════════════

def attach_features(balls: pd.DataFrame,
                    stats_by_match: dict,
                    over_par_data: tuple,
                    match_dates: pd.DataFrame,
                    matches_csv: str = "data/ipl_matches.csv") -> pd.DataFrame:
    """Attach rolling player stats, momentum index, and over_par to balls."""

    df = balls.copy()
    df["match_id"] = df["match_id"].astype(str)
    n = len(df)

    # ── Player form features ──────────────────────────────────────────────
    batsman_sr       = np.full(n, GLOBAL_AVG_SR, dtype=np.float32)
    batsman_bdry_pct = np.full(n, GLOBAL_BDRY_PCT, dtype=np.float32)
    batsman_dot_pct  = np.full(n, GLOBAL_DOT_PCT, dtype=np.float32)
    bowler_econ      = np.full(n, GLOBAL_AVG_ECON, dtype=np.float32)
    bowler_dot_pct   = np.full(n, GLOBAL_DOT_PCT, dtype=np.float32)

    for i, row in tqdm(df.iterrows(), total=n, desc="Attaching player stats", miniters=50000):
        mid = row["match_id"]
        if mid not in stats_by_match:
            continue
        ms = stats_by_match[mid]

        bat = row.get("batsman", "")
        if bat in ms["bat"]:
            bs = ms["bat"][bat]
            batsman_sr[i]       = bs["sr"]
            batsman_bdry_pct[i] = bs["bdry_pct"]
            batsman_dot_pct[i]  = bs["dot_pct"]

        bowl = row.get("bowler", "")
        if bowl in ms["bowl"]:
            bls = ms["bowl"][bowl]
            bowler_econ[i]    = bls["econ"]
            bowler_dot_pct[i] = bls["dot_pct"]

    df["batsman_sr_2yr"]           = batsman_sr
    df["batsman_boundary_pct_2yr"] = batsman_bdry_pct
    df["batsman_dot_pct_2yr"]      = batsman_dot_pct
    df["bowler_economy_2yr"]       = bowler_econ
    df["bowler_dot_pct_2yr"]       = bowler_dot_pct

    # ── Momentum index ────────────────────────────────────────────────────
    # runs_last_6 / (runs_last_24 / 4), penalised by wickets in last 6
    print("Computing momentum index...")
    df = df.sort_values(["match_id", "innings", "over", "ball"]).reset_index(drop=True)

    runs_last_6  = df.groupby(["match_id","innings"])["total_runs"] \
        .transform(lambda x: x.rolling(6, min_periods=1).sum().shift(1, fill_value=0))
    runs_last_24 = df.groupby(["match_id","innings"])["total_runs"] \
        .transform(lambda x: x.rolling(24, min_periods=1).sum().shift(1, fill_value=0))
    wkts_last_6  = df.groupby(["match_id","innings"])["wicket"] \
        .transform(lambda x: x.rolling(6, min_periods=1).sum().shift(1, fill_value=0))

    baseline = np.maximum(runs_last_24 / 4.0, 1.0)
    raw_momentum = runs_last_6 / baseline
    penalty = 1.0 - 0.15 * wkts_last_6
    df["momentum_index"] = np.clip(raw_momentum * penalty, 0, 3).astype(np.float32)

    # ── Over-par (innings 1 only) ─────────────────────────────────────────
    print("Computing over_par...")
    venue_over_median, global_over_median = over_par_data

    # Need venue on balls
    matches_venue = pd.read_csv(matches_csv)[["match_id", "venue"]]
    matches_venue["match_id"] = matches_venue["match_id"].astype(str)
    df = df.merge(matches_venue, on="match_id", how="left")

    # Compute current over for each delivery
    df["current_over"] = (
        df.groupby(["match_id", "innings"])["is_legal"]
        .transform(lambda x: x.shift(1, fill_value=0).cumsum()) // 6
    ).clip(0, 19)

    # Compute cumulative runs (state BEFORE this delivery)
    df["cum_runs_before"] = (
        df.groupby(["match_id", "innings"])["total_runs"]
        .transform(lambda x: x.shift(1, fill_value=0).cumsum())
    )

    # Merge venue median
    df = df.merge(
        venue_over_median,
        left_on=["venue", "current_over"],
        right_on=["venue", "over_completed"],
        how="left"
    )
    df = df.merge(
        global_over_median,
        left_on=["current_over"],
        right_on=["over_completed"],
        how="left",
        suffixes=("", "_global")
    )

    # Use venue median where available, else global
    median_at_over = df["venue_median_at_over"].fillna(df["global_median_at_over"]).fillna(0)

    # over_par = actual cumulative runs - expected median at this over (inn 1 only)
    df["over_par"] = np.where(
        df["innings"] == 1,
        df["cum_runs_before"] - median_at_over,
        0.0
    ).astype(np.float32)

    # Clean up temp columns
    drop_cols = ["venue", "current_over", "cum_runs_before",
                 "venue_median_at_over", "over_completed",
                 "global_median_at_over", "over_completed_global"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    return df


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Wave 4: rolling player form + game state")
    parser.add_argument("--balls",   default="data/ipl_balls_wave2.csv")
    parser.add_argument("--matches", default="data/ipl_matches.csv")
    parser.add_argument("--output",  default="data/ipl_balls_wave4.csv")
    args = parser.parse_args()

    print("═" * 65)
    print("  WAVE 4 — Rolling Player Form + Advanced Game State")
    print("═" * 65)

    balls   = pd.read_csv(args.balls)
    matches = pd.read_csv(args.matches, parse_dates=["date"])
    balls["match_id"]   = balls["match_id"].astype(str)
    matches["match_id"] = matches["match_id"].astype(str)
    match_dates = matches[["match_id", "date"]].drop_duplicates("match_id")

    print(f"\nBalls loaded   : {len(balls):,}")
    print(f"Matches loaded : {len(matches):,}")

    # Step 1: rolling player stats
    stats = compute_rolling_player_stats(balls, match_dates)

    # Step 2: over-par lookup (training set only)
    matches_sorted = matches.sort_values("date").reset_index(drop=True)
    train_cutoff = matches_sorted.iloc[int(len(matches_sorted) * 0.70)]["date"]
    print(f"\nOver-par training cutoff: {train_cutoff.date()}")
    over_par_data = compute_over_par(balls, match_dates, train_cutoff,
                                     matches_csv=args.matches)

    # Step 3: attach all features
    balls = attach_features(balls, stats, over_par_data, match_dates,
                            matches_csv=args.matches)

    # Sanity checks
    WAVE4_COLS = ["batsman_sr_2yr", "batsman_boundary_pct_2yr", "batsman_dot_pct_2yr",
                  "bowler_economy_2yr", "bowler_dot_pct_2yr", "momentum_index", "over_par"]
    print("\n── Sanity checks ───────────────────────────────────────────────────")
    for col in WAVE4_COLS:
        v = balls[col]
        print(f"  {col:30s}  mean={v.mean():8.3f}  std={v.std():7.3f}  "
              f"min={v.min():8.3f}  max={v.max():8.3f}")

    # Save
    balls.to_csv(args.output, index=False)
    print(f"\nSaved to {args.output}  ({len(balls):,} rows)")
    print("Done.")


if __name__ == "__main__":
    main()