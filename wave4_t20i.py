"""
wave4_t20i.py  —  Wave 4 + 4b Feature Engineering for T20I balls
=================================================================
Computes the same rolling player-form and game-state features as
wave4.py / wave4b.py, but for T20I deliveries.

KEY DESIGN CHOICE — Combined ball history
------------------------------------------
Rolling stats (batsman SR, bowler economy, pace/spin splits) are
computed from the UNION of IPL and T20I balls. This means a batsman
who appeared in both formats gets credit for all his recent T20 form,
not just one competition. The 2-year window is the same as before.

Same-match exclusion is preserved — only deliveries from matches
strictly before the current match date are used.

Inputs (all under data/)
  ipl_balls_wave4b.csv     (or ipl_balls_wave4.csv if 4b not yet run)
  ipl_matches.csv
  t20i_balls_filtered.csv
  t20i_matches_filtered.csv
  player_attributes.csv

Output
  data/t20i_balls_wave4b.csv

Usage
-----
  python wave4_t20i.py
"""

import argparse
from pathlib import Path

import pandas as pd
import numpy as np
from tqdm import tqdm


# ── Constants (match wave4.py / wave4b.py) ────────────────────────────────────
WINDOW_DAYS     = 730
MIN_BAT_BALLS   = 20
MIN_BOWL_BALLS  = 20
MIN_BALLS_PS    = 15     # pace/spin split minimum
GLOBAL_AVG_SR   = 130.0
GLOBAL_AVG_ECON = 8.0
GLOBAL_DOT_PCT  = 0.42
GLOBAL_BDRY_PCT = 0.14


# ════════════════════════════════════════════════════════════════════════════
# 1.  ROLLING PLAYER STATS  (combined IPL + T20I history)
# ════════════════════════════════════════════════════════════════════════════

def compute_rolling_player_stats(all_balls: pd.DataFrame,
                                  match_dates: pd.DataFrame,
                                  target_match_ids: set) -> dict:
    """
    Compute per-match rolling 2yr player stats.
    all_balls  — combined IPL + T20I deliveries (used as the history pool)
    match_dates — all matches (both formats) with their dates
    target_match_ids — only build snapshots for T20I match IDs (saves time)
    """
    df = all_balls.merge(match_dates, on="match_id", how="left")
    df = df.sort_values(["date", "match_id", "innings", "over", "ball"]).reset_index(drop=True)
    legal_mask = df["is_legal"] == 1

    # Pre-aggregate per match × batsman
    bat_agg = (
        df.groupby(["match_id", "date", "batsman"])
        .agg(
            bat_runs  = ("runs_batsman", "sum"),
            bat_legal = ("is_legal",     "sum"),
            bat_fours = ("runs_batsman", lambda x: (x == 4).sum()),
            bat_sixes = ("runs_batsman", lambda x: (x == 6).sum()),
        )
        .reset_index()
    )
    bat_dot_legal = (
        df[legal_mask]
        .groupby(["match_id", "date", "batsman"])
        .agg(bat_dots_legal=("runs_batsman", lambda x: (x == 0).sum()))
        .reset_index()
    )
    bat_agg = bat_agg.merge(bat_dot_legal, on=["match_id", "date", "batsman"], how="left")
    bat_agg["bat_dots_legal"] = bat_agg["bat_dots_legal"].fillna(0).astype(int)

    # Pre-aggregate per match × bowler
    bowl_agg = (
        df.groupby(["match_id", "date", "bowler"])
        .agg(
            bowl_runs  = ("total_runs", "sum"),
            bowl_legal = ("is_legal",   "sum"),
        )
        .reset_index()
    )
    bowl_dot_legal = (
        df[legal_mask]
        .groupby(["match_id", "date", "bowler"])
        .agg(bowl_dots_legal=("total_runs", lambda x: (x == 0).sum()))
        .reset_index()
    )
    bowl_agg = bowl_agg.merge(bowl_dot_legal, on=["match_id", "date", "bowler"], how="left")
    bowl_agg["bowl_dots_legal"] = bowl_agg["bowl_dots_legal"].fillna(0).astype(int)

    # Only iterate over T20I matches (not all IPL matches)
    match_order = (
        df[df["match_id"].isin(target_match_ids)]
        .groupby("match_id")["date"].first()
        .sort_values().reset_index()
    )

    stats_by_match = {}
    for _, mrow in tqdm(match_order.iterrows(), total=len(match_order),
                        desc="Rolling 2yr stats (T20I)"):
        mid   = mrow["match_id"]
        mdate = mrow["date"]
        cutoff = mdate - pd.Timedelta(days=WINDOW_DAYS)

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
                    "sr": GLOBAL_AVG_SR, "bdry_pct": GLOBAL_BDRY_PCT,
                    "dot_pct": GLOBAL_DOT_PCT,
                }

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

    print(f"  Snapshots built for {len(stats_by_match)} T20I matches")
    return stats_by_match


# ════════════════════════════════════════════════════════════════════════════
# 2.  LINEUP STRENGTH FEATURES  (wave2 equivalent)
# ════════════════════════════════════════════════════════════════════════════

BOWLER_QUOTA = 24   # max legal deliveries per bowler per innings

def compute_career_stats(all_balls: pd.DataFrame,
                          match_dates: pd.DataFrame,
                          target_match_ids: set) -> tuple:
    """
    Expanding-window career SR and economy for every player in target matches.
    Uses combined IPL + T20I history as the lookback pool.
    Returns:
        career_sr_by_match   : { match_id -> { player -> SR } }
        career_econ_by_match : { match_id -> { player -> economy } }
    """
    df = all_balls.merge(match_dates, on="match_id", how="left")
    df = df.sort_values(["date", "match_id", "innings", "over", "ball"]).reset_index(drop=True)

    bat_runs:  dict = {}
    bat_balls: dict = {}
    bowl_runs:  dict = {}
    bowl_balls: dict = {}

    career_sr_by_match:   dict = {}
    career_econ_by_match: dict = {}

    match_order = (
        df.groupby("match_id")["date"].first()
          .sort_values().index.tolist()
    )

    for mid in tqdm(match_order, desc="Career stats (combined)"):
        mdf = df[df["match_id"] == mid]

        # Snapshot BEFORE this match
        if mid in target_match_ids:
            sr_snap, eco_snap = {}, {}
            for p in set(mdf["batsman"].tolist() + mdf["non_striker"].tolist()):
                b = bat_balls.get(p, 0)
                r = bat_runs.get(p, 0)
                sr_snap[p] = (r / b * 100) if b >= MIN_BAT_BALLS else GLOBAL_AVG_SR
            for p in mdf["bowler"].unique():
                b = bowl_balls.get(p, 0)
                r = bowl_runs.get(p, 0)
                eco_snap[p] = (r / (b / 6)) if b >= MIN_BOWL_BALLS else GLOBAL_AVG_ECON
            career_sr_by_match[mid]   = sr_snap
            career_econ_by_match[mid] = eco_snap

        # Update accumulators
        for p, grp in mdf.groupby("batsman"):
            bat_runs[p]  = bat_runs.get(p, 0)  + int(grp["runs_batsman"].sum())
            bat_balls[p] = bat_balls.get(p, 0) + int((grp["is_legal"] == 1).sum())
        for p, grp in mdf.groupby("bowler"):
            bowl_runs[p]  = bowl_runs.get(p, 0)  + int(grp["total_runs"].sum())
            bowl_balls[p] = bowl_balls.get(p, 0) + int((grp["is_legal"] == 1).sum())

    print(f"  Career stats for {len(bat_runs)} batsmen, {len(bowl_runs)} bowlers")
    return career_sr_by_match, career_econ_by_match


def compute_lineup_features(balls: pd.DataFrame,
                             career_sr_by_match: dict,
                             career_econ_by_match: dict) -> pd.DataFrame:
    """
    Add batting_strength_remaining, bowling_strength_remaining,
    and wickets_remaining_weighted to T20I balls.
    """
    n = len(balls)
    bat_str  = np.zeros(n, dtype=np.float32)
    bowl_str = np.zeros(n, dtype=np.float32)
    wkt_wt   = np.zeros(n, dtype=np.float32)

    groups = balls.groupby(["match_id", "innings"], sort=False)

    for (mid, inn), grp in tqdm(groups, desc="Lineup features (T20I)"):
        grp_sorted = grp.sort_values(["over", "ball"])
        idx_arr    = grp_sorted.index.values

        # Infer batting order from first appearance
        batting_order, seen = [], set()
        for _, row in grp_sorted.iterrows():
            for p in [row["batsman"], row["non_striker"]]:
                if p and p not in seen:
                    seen.add(p); batting_order.append(p)

        sr_lookup   = career_sr_by_match.get(mid,   {})
        econ_lookup = career_econ_by_match.get(mid,  {})
        dismissed   = set()
        bowler_used = {}

        for i, (_, row) in enumerate(grp_sorted.iterrows()):
            idx         = idx_arr[i]
            current_bat = row["batsman"]
            bowler      = row["bowler"]

            # batting_strength_remaining
            remaining = [p for p in batting_order
                         if p not in dismissed and p != current_bat]
            bat_str[idx] = sum(sr_lookup.get(p, GLOBAL_AVG_SR) for p in remaining)

            # wickets_remaining_weighted
            undismissed = [p for p in batting_order if p not in dismissed]
            wkt_wt[idx] = sum(sr_lookup.get(p, GLOBAL_AVG_SR) / GLOBAL_AVG_SR
                              for p in undismissed)

            # bowling_strength_remaining
            total_bowl = 0.0
            for b_name, b_used in bowler_used.items():
                b_left = max(0, BOWLER_QUOTA - b_used)
                if b_left > 0:
                    eco = econ_lookup.get(b_name, GLOBAL_AVG_ECON)
                    total_bowl += (1.0 / eco) * b_left
            if bowler not in bowler_used:
                eco = econ_lookup.get(bowler, GLOBAL_AVG_ECON)
                total_bowl += (1.0 / eco) * BOWLER_QUOTA
            bowl_str[idx] = total_bowl

            # Update state
            if row["wicket"] == 1 and row.get("player_out"):
                dismissed.add(row["player_out"])
            if row["is_legal"] == 1:
                bowler_used[bowler] = bowler_used.get(bowler, 0) + 1

    balls = balls.copy()
    balls["batting_strength_remaining"]  = bat_str
    balls["bowling_strength_remaining"]  = bowl_str
    balls["wickets_remaining_weighted"]  = wkt_wt
    return balls


# ════════════════════════════════════════════════════════════════════════════
# 3.  PACE / SPIN SPLIT STATS  (wave4b equivalent)
# ════════════════════════════════════════════════════════════════════════════

def compute_pace_spin_splits(all_balls: pd.DataFrame,
                              match_dates: pd.DataFrame,
                              bowler_group_map: dict,
                              target_match_ids: set) -> dict:
    """Snapshot batsman SR vs pace and vs spin for each T20I match."""
    df = all_balls.merge(match_dates, on="match_id", how="left")
    df["bowler_group"] = df["bowler"].map(bowler_group_map).fillna("pace")
    df = df.sort_values(["date", "match_id", "innings", "over", "ball"]).reset_index(drop=True)

    agg = (
        df.groupby(["match_id", "date", "batsman", "bowler_group"])
        .agg(runs=("runs_batsman", "sum"), legal=("is_legal", "sum"))
        .reset_index()
    )

    match_order = (
        df[df["match_id"].isin(target_match_ids)]
        .groupby("match_id")["date"].first()
        .sort_values().reset_index()
    )

    ps_by_match = {}
    for _, mrow in tqdm(match_order.iterrows(), total=len(match_order),
                        desc="Pace/spin splits (T20I)"):
        mid   = mrow["match_id"]
        mdate = mrow["date"]
        cutoff = mdate - pd.Timedelta(days=WINDOW_DAYS)

        window = agg[(agg["date"] >= cutoff) & (agg["date"] < mdate)]
        snap = {}
        for player, pg in window.groupby("batsman"):
            pace_rows = pg[pg["bowler_group"] == "pace"]
            spin_rows = pg[pg["bowler_group"] == "spin"]
            p_legal = pace_rows["legal"].sum()
            s_legal = spin_rows["legal"].sum()
            snap[player] = {
                "sr_vs_pace": (pace_rows["runs"].sum() / p_legal * 100
                               if p_legal >= MIN_BALLS_PS else GLOBAL_AVG_SR),
                "sr_vs_spin": (spin_rows["runs"].sum() / s_legal * 100
                               if s_legal >= MIN_BALLS_PS else GLOBAL_AVG_SR),
            }
        ps_by_match[mid] = snap

    return ps_by_match


# ════════════════════════════════════════════════════════════════════════════
# 3.  OVER-PAR  (T20I venues, training set only)
# ════════════════════════════════════════════════════════════════════════════

def compute_over_par(t20i_balls: pd.DataFrame,
                     match_dates: pd.DataFrame,
                     matches_venue: pd.DataFrame,
                     train_cutoff_date) -> tuple:
    df = t20i_balls[t20i_balls["innings"] == 1].copy()
    df["match_id"] = df["match_id"].astype(str)
    df = df.merge(match_dates, on="match_id", how="left")
    df = df[df["date"] < train_cutoff_date].copy()
    df = df.merge(matches_venue, on="match_id", how="left")

    df["over_completed"] = df.groupby("match_id")["is_legal"].cumsum() // 6
    over_scores = (
        df.groupby(["match_id", "venue", "over_completed"])["total_runs"]
        .sum().groupby(level=[0, 1]).cumsum().reset_index()
        .rename(columns={"total_runs": "cum_runs"})
    )
    venue_over_median = (
        over_scores.groupby(["venue", "over_completed"])["cum_runs"]
        .median().reset_index()
        .rename(columns={"cum_runs": "venue_median_at_over"})
    )
    global_over_median = (
        over_scores.groupby("over_completed")["cum_runs"]
        .median().reset_index()
        .rename(columns={"cum_runs": "global_median_at_over"})
    )
    return venue_over_median, global_over_median


# ════════════════════════════════════════════════════════════════════════════
# 4.  ATTACH ALL FEATURES TO T20I BALLS
# ════════════════════════════════════════════════════════════════════════════

def attach_features(t20i_balls: pd.DataFrame,
                    stats_by_match: dict,
                    ps_by_match: dict,
                    over_par_data: tuple,
                    match_dates: pd.DataFrame,
                    matches_venue: pd.DataFrame,
                    bowler_group_map: dict,
                    player_hand_map: dict) -> pd.DataFrame:

    df = t20i_balls.copy()
    df["match_id"] = df["match_id"].astype(str)
    n = len(df)

    # ── Wave 4: rolling player stats ──────────────────────────────────────
    batsman_sr       = np.full(n, GLOBAL_AVG_SR,   dtype=np.float32)
    batsman_bdry_pct = np.full(n, GLOBAL_BDRY_PCT, dtype=np.float32)
    batsman_dot_pct  = np.full(n, GLOBAL_DOT_PCT,  dtype=np.float32)
    bowler_econ      = np.full(n, GLOBAL_AVG_ECON, dtype=np.float32)
    bowler_dot_pct   = np.full(n, GLOBAL_DOT_PCT,  dtype=np.float32)

    # ── Wave 4b: pace/spin splits ─────────────────────────────────────────
    bowler_is_pace      = np.ones(n, dtype=np.float32)   # default pace
    batsman_is_lhb      = np.zeros(n, dtype=np.float32)
    sr_vs_pace          = np.full(n, GLOBAL_AVG_SR, dtype=np.float32)
    sr_vs_spin          = np.full(n, GLOBAL_AVG_SR, dtype=np.float32)

    for i, row in tqdm(df.iterrows(), total=n, desc="Attaching features", miniters=50000):
        mid = row["match_id"]

        # Wave 4
        if mid in stats_by_match:
            ms  = stats_by_match[mid]
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

        # Wave 4b
        bowler = row.get("bowler", "")
        bgroup = bowler_group_map.get(bowler, "pace")
        bowler_is_pace[i] = 1.0 if bgroup == "pace" else 0.0

        batsman = row.get("batsman", "")
        batsman_is_lhb[i] = 1.0 if player_hand_map.get(batsman) == "Left" else 0.0

        if mid in ps_by_match and batsman in ps_by_match[mid]:
            ps = ps_by_match[mid][batsman]
            sr_vs_pace[i] = ps["sr_vs_pace"]
            sr_vs_spin[i] = ps["sr_vs_spin"]

    df["batsman_sr_2yr"]           = batsman_sr
    df["batsman_boundary_pct_2yr"] = batsman_bdry_pct
    df["batsman_dot_pct_2yr"]      = batsman_dot_pct
    df["bowler_economy_2yr"]       = bowler_econ
    df["bowler_dot_pct_2yr"]       = bowler_dot_pct
    df["bowler_is_pace"]           = bowler_is_pace
    df["batsman_is_lhb"]           = batsman_is_lhb
    df["batsman_sr_vs_pace"]       = sr_vs_pace
    df["batsman_sr_vs_spin"]       = sr_vs_spin
    df["pace_spin_sr_diff"]        = df["batsman_sr_vs_spin"] - df["batsman_sr_vs_pace"]
    df["lhb_vs_spin"]              = ((df["batsman_is_lhb"] == 1) &
                                      (df["bowler_is_pace"]  == 0)).astype(np.float32)

    # ── Momentum index ────────────────────────────────────────────────────
    print("Computing momentum index...")
    df = df.sort_values(["match_id", "innings", "over", "ball"]).reset_index(drop=True)
    runs_last_6  = df.groupby(["match_id","innings"])["total_runs"] \
        .transform(lambda x: x.rolling(6, min_periods=1).sum().shift(1, fill_value=0))
    runs_last_24 = df.groupby(["match_id","innings"])["total_runs"] \
        .transform(lambda x: x.rolling(24, min_periods=1).sum().shift(1, fill_value=0))
    wkts_last_6  = df.groupby(["match_id","innings"])["wicket"] \
        .transform(lambda x: x.rolling(6, min_periods=1).sum().shift(1, fill_value=0))
    baseline = np.maximum(runs_last_24 / 4.0, 1.0)
    df["momentum_index"] = np.clip(
        (runs_last_6 / baseline) * (1.0 - 0.15 * wkts_last_6), 0, 3
    ).astype(np.float32)

    # ── Over-par ──────────────────────────────────────────────────────────
    print("Computing over_par...")
    venue_over_median, global_over_median = over_par_data
    df = df.merge(matches_venue, on="match_id", how="left")
    df["current_over"] = (
        df.groupby(["match_id", "innings"])["is_legal"]
        .transform(lambda x: x.shift(1, fill_value=0).cumsum()) // 6
    ).clip(0, 19)
    df["cum_runs_before"] = (
        df.groupby(["match_id", "innings"])["total_runs"]
        .transform(lambda x: x.shift(1, fill_value=0).cumsum())
    )
    df = df.merge(venue_over_median,
                  left_on=["venue", "current_over"],
                  right_on=["venue", "over_completed"], how="left")
    df = df.merge(global_over_median,
                  left_on=["current_over"], right_on=["over_completed"],
                  how="left", suffixes=("", "_global"))
    median_at_over = df["venue_median_at_over"].fillna(df["global_median_at_over"]).fillna(0)
    df["over_par"] = np.where(df["innings"] == 1,
                               df["cum_runs_before"] - median_at_over, 0.0).astype(np.float32)
    drop_cols = ["venue", "current_over", "cum_runs_before",
                 "venue_median_at_over", "over_completed",
                 "global_median_at_over", "over_completed_global"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    return df


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Wave 4 + 4b features for T20I balls")
    parser.add_argument("--t20i-balls",   default="data/t20i_balls_filtered.csv")
    parser.add_argument("--t20i-matches", default="data/t20i_matches_filtered.csv")
    parser.add_argument("--ipl-balls",    default="data/ipl_balls_wave4b.csv")
    parser.add_argument("--ipl-matches",  default="data/ipl_matches.csv")
    parser.add_argument("--player-attrs", default="data/player_attributes.csv")
    parser.add_argument("--output",       default="data/t20i_balls_wave4b.csv")
    args = parser.parse_args()

    # Fallback: use wave4 if wave4b not yet available
    ipl_balls_path = args.ipl_balls
    if not Path(ipl_balls_path).exists():
        ipl_balls_path = ipl_balls_path.replace("wave4b", "wave4")
        print(f"wave4b not found — falling back to {ipl_balls_path}")

    print("═" * 65)
    print("  WAVE 4 T20I — Combined IPL+T20I Rolling Player Form")
    print("═" * 65)

    # ── Load data ─────────────────────────────────────────────────────────
    t20i_balls   = pd.read_csv(args.t20i_balls)
    t20i_matches = pd.read_csv(args.t20i_matches, parse_dates=["date"])
    ipl_balls    = pd.read_csv(ipl_balls_path)
    ipl_matches  = pd.read_csv(args.ipl_matches,  parse_dates=["date"])
    attrs        = pd.read_csv(args.player_attrs)

    for df in [t20i_balls, t20i_matches, ipl_balls, ipl_matches]:
        df["match_id"] = df["match_id"].astype(str)

    # Men's T20I only
    if "gender" in t20i_matches.columns:
        t20i_matches = t20i_matches[t20i_matches["gender"] == "male"].copy()
        valid_ids = set(t20i_matches["match_id"])
        t20i_balls = t20i_balls[t20i_balls["match_id"].isin(valid_ids)].copy()

    print(f"T20I balls   : {len(t20i_balls):,}")
    print(f"IPL balls    : {len(ipl_balls):,}")

    # ── Player attribute maps ─────────────────────────────────────────────
    # Keep only columns we need from IPL balls to avoid merge bloat
    ipl_balls_slim = ipl_balls[["match_id", "innings", "over", "ball",
                                 "batsman", "non_striker", "bowler",
                                 "runs_batsman", "total_runs",
                                 "is_legal", "wicket", "player_out"]].copy()

    bowler_group_map = attrs.set_index("player_name")["bowling_group"].to_dict()
    player_hand_map  = attrs.set_index("player_name")["batting_hand"].to_dict()

    # ── Combined ball history for rolling stats ───────────────────────────
    t20i_balls_slim = t20i_balls[["match_id", "innings", "over", "ball",
                                   "batsman", "non_striker", "bowler",
                                   "runs_batsman", "total_runs",
                                   "is_legal", "wicket", "player_out"]].copy()

    all_matches = pd.concat([
        ipl_matches[["match_id", "date"]],
        t20i_matches[["match_id", "date"]]
    ], ignore_index=True).drop_duplicates("match_id")

    all_balls = pd.concat([ipl_balls_slim, t20i_balls_slim], ignore_index=True)
    all_balls["match_id"] = all_balls["match_id"].astype(str)

    match_dates = all_matches[["match_id", "date"]].copy()
    match_dates["match_id"] = match_dates["match_id"].astype(str)

    target_ids = set(t20i_balls["match_id"].astype(str))

    # ── Step 1: rolling stats ─────────────────────────────────────────────
    stats_by_match = compute_rolling_player_stats(all_balls, match_dates, target_ids)

    # ── Step 2: career stats + lineup features (wave2 equivalent) ────────
    career_sr, career_econ = compute_career_stats(all_balls, match_dates, target_ids)
    t20i_balls = compute_lineup_features(t20i_balls, career_sr, career_econ)

    # ── Step 3: pace/spin splits ──────────────────────────────────────────
    ps_by_match = compute_pace_spin_splits(all_balls, match_dates, bowler_group_map, target_ids)

    # ── Step 4: over-par ──────────────────────────────────────────────────
    t20i_sorted = t20i_matches.sort_values("date").reset_index(drop=True)
    train_cutoff = t20i_sorted.iloc[int(len(t20i_sorted) * 0.70)]["date"]
    print(f"\nOver-par training cutoff: {train_cutoff.date()}")
    t20i_match_dates = t20i_matches[["match_id", "date"]].copy()
    t20i_match_dates["match_id"] = t20i_match_dates["match_id"].astype(str)
    matches_venue = t20i_matches[["match_id", "venue"]].copy()
    matches_venue["match_id"] = matches_venue["match_id"].astype(str)
    over_par_data = compute_over_par(t20i_balls, t20i_match_dates, matches_venue, train_cutoff)

    # ── Step 5: attach remaining features ────────────────────────────────
    t20i_balls = attach_features(
        t20i_balls, stats_by_match, ps_by_match,
        over_par_data, t20i_match_dates, matches_venue,
        bowler_group_map, player_hand_map
    )

    # ── Sanity checks ─────────────────────────────────────────────────────
    WAVE_COLS = [
        "batting_strength_remaining", "bowling_strength_remaining",
        "wickets_remaining_weighted",
        "batsman_sr_2yr", "batsman_boundary_pct_2yr", "batsman_dot_pct_2yr",
        "bowler_economy_2yr", "bowler_dot_pct_2yr", "momentum_index", "over_par",
        "bowler_is_pace", "batsman_sr_vs_pace", "batsman_sr_vs_spin", "pace_spin_sr_diff",
    ]
    print("\n── Sanity checks ───────────────────────────────────────────────────")
    for col in WAVE_COLS:
        if col in t20i_balls.columns:
            v = t20i_balls[col]
            print(f"  {col:35s}  mean={v.mean():8.3f}  std={v.std():7.3f}")

    # ── Save ──────────────────────────────────────────────────────────────
    t20i_balls.to_csv(args.output, index=False)
    print(f"\nSaved → {args.output}  ({len(t20i_balls):,} rows)")
    print("Done.")


if __name__ == "__main__":
    main()