"""
wave4_leagues.py  —  Wave 4 + 4b Feature Engineering for All Non-IPL Leagues
=============================================================================
Generalised version of wave4_t20i.py that enriches any league's balls file
with the full suite of wave features, using a combined rolling history drawn
from ALL available leagues (IPL + T20I + BBL + CPL + PSL + SA20 + SMAT).

This means a player's 2-year rolling stats reflect everything they've played
across all formats — the richest possible signal.

Features computed
-----------------
  Wave 2 (lineup):
    batting_strength_remaining, bowling_strength_remaining,
    wickets_remaining_weighted

  Wave 4 (rolling form):
    batsman_sr_2yr, batsman_boundary_pct_2yr, batsman_dot_pct_2yr,
    bowler_economy_2yr, bowler_dot_pct_2yr,
    momentum_index, over_par

  Wave 4b (pace/spin):
    bowler_is_pace, batsman_is_lhb, batsman_sr_vs_pace,
    batsman_sr_vs_spin, pace_spin_sr_diff, lhb_vs_spin

Inputs
------
  data/all_balls.csv       — combined ball history for rolling lookback
  data/all_matches.csv     — combined match dates and venues
  data/player_attributes.csv
  data/<league>_balls.csv  — target league raw balls (one per league)
  data/<league>_matches.csv

Output
------
  data/<league>_balls_wave4b.csv  (one per league)

Usage
-----
  # Enrich a specific league
  python wave4_leagues.py --league bbl

  # Enrich all leagues that are missing wave4b files
  python wave4_leagues.py --all

  # Force re-enrichment even if file exists
  python wave4_leagues.py --league bbl --force
"""

import argparse
from pathlib import Path

import pandas as pd
import numpy as np
from tqdm import tqdm


# ── Constants ──────────────────────────────────────────────────────────────────
WINDOW_DAYS     = 730
MIN_BAT_BALLS   = 20
MIN_BOWL_BALLS  = 20
MIN_BALLS_PS    = 15
GLOBAL_AVG_SR   = 130.0
GLOBAL_AVG_ECON = 8.0
GLOBAL_DOT_PCT  = 0.42
GLOBAL_BDRY_PCT = 0.14
BOWLER_QUOTA    = 24

# Leagues to process when --all is passed
ALL_LEAGUES = ["t20i", "bbl", "cpl", "psl", "sa20", "smat"]


# ════════════════════════════════════════════════════════════════════════════
# 1.  ROLLING PLAYER STATS
# ════════════════════════════════════════════════════════════════════════════

def compute_rolling_player_stats(all_balls: pd.DataFrame,
                                  match_dates: pd.DataFrame,
                                  target_match_ids: set) -> dict:
    df = all_balls.merge(match_dates, on="match_id", how="left")
    df = df.sort_values(["date","match_id","innings","over","ball"]).reset_index(drop=True)
    legal = df["is_legal"] == 1

    bat_agg = (
        df.groupby(["match_id","date","batsman"])
        .agg(bat_runs=("runs_batsman","sum"), bat_legal=("is_legal","sum"),
             bat_fours=("runs_batsman", lambda x: (x==4).sum()),
             bat_sixes=("runs_batsman", lambda x: (x==6).sum()))
        .reset_index()
    )
    bat_dot = (df[legal].groupby(["match_id","date","batsman"])
               .agg(bat_dots_legal=("runs_batsman", lambda x: (x==0).sum()))
               .reset_index())
    bat_agg = bat_agg.merge(bat_dot, on=["match_id","date","batsman"], how="left")
    bat_agg["bat_dots_legal"] = bat_agg["bat_dots_legal"].fillna(0).astype(int)

    bowl_agg = (
        df.groupby(["match_id","date","bowler"])
        .agg(bowl_runs=("total_runs","sum"), bowl_legal=("is_legal","sum"))
        .reset_index()
    )
    bowl_dot = (df[legal].groupby(["match_id","date","bowler"])
                .agg(bowl_dots_legal=("total_runs", lambda x: (x==0).sum()))
                .reset_index())
    bowl_agg = bowl_agg.merge(bowl_dot, on=["match_id","date","bowler"], how="left")
    bowl_agg["bowl_dots_legal"] = bowl_agg["bowl_dots_legal"].fillna(0).astype(int)

    match_order = (df[df["match_id"].isin(target_match_ids)]
                   .groupby("match_id")["date"].first()
                   .sort_values().reset_index())

    stats = {}
    for _, mrow in tqdm(match_order.iterrows(), total=len(match_order),
                        desc="  Rolling 2yr stats"):
        mid, mdate = mrow["match_id"], mrow["date"]
        cutoff = mdate - pd.Timedelta(days=WINDOW_DAYS)

        bw  = bat_agg[(bat_agg["date"] >= cutoff) & (bat_agg["date"] < mdate)]
        bwl = bowl_agg[(bowl_agg["date"] >= cutoff) & (bowl_agg["date"] < mdate)]

        bat_stats = {}
        for player, pg in bw.groupby("batsman"):
            l, r = pg["bat_legal"].sum(), pg["bat_runs"].sum()
            f, s, d = pg["bat_fours"].sum(), pg["bat_sixes"].sum(), pg["bat_dots_legal"].sum()
            bat_stats[player] = ({"sr": r/l*100, "bdry_pct": (f+s)/l, "dot_pct": d/l}
                                  if l >= MIN_BAT_BALLS else
                                  {"sr": GLOBAL_AVG_SR, "bdry_pct": GLOBAL_BDRY_PCT,
                                   "dot_pct": GLOBAL_DOT_PCT})

        bowl_stats = {}
        for player, pg in bwl.groupby("bowler"):
            l, r, d = pg["bowl_legal"].sum(), pg["bowl_runs"].sum(), pg["bowl_dots_legal"].sum()
            bowl_stats[player] = ({"econ": r/(l/6), "dot_pct": d/l}
                                   if l >= MIN_BOWL_BALLS else
                                   {"econ": GLOBAL_AVG_ECON, "dot_pct": GLOBAL_DOT_PCT})

        stats[mid] = {"bat": bat_stats, "bowl": bowl_stats}

    return stats


# ════════════════════════════════════════════════════════════════════════════
# 2.  CAREER STATS + LINEUP FEATURES  (wave 2)
# ════════════════════════════════════════════════════════════════════════════

def compute_career_stats(all_balls: pd.DataFrame,
                          match_dates: pd.DataFrame,
                          target_match_ids: set) -> tuple:
    df = all_balls.merge(match_dates, on="match_id", how="left")
    df = df.sort_values(["date","match_id","innings","over","ball"]).reset_index(drop=True)

    bat_runs, bat_balls = {}, {}
    bowl_runs, bowl_balls = {}, {}
    career_sr:   dict = {}
    career_econ: dict = {}

    match_order = (df.groupby("match_id")["date"].first()
                   .sort_values().index.tolist())

    for mid in tqdm(match_order, desc="  Career stats"):
        mdf = df[df["match_id"] == mid]
        if mid in target_match_ids:
            sr_snap, eco_snap = {}, {}
            for p in set(mdf["batsman"].tolist() + mdf["non_striker"].tolist()):
                b, r = bat_balls.get(p,0), bat_runs.get(p,0)
                sr_snap[p] = (r/b*100) if b >= MIN_BAT_BALLS else GLOBAL_AVG_SR
            for p in mdf["bowler"].unique():
                b, r = bowl_balls.get(p,0), bowl_runs.get(p,0)
                eco_snap[p] = (r/(b/6)) if b >= MIN_BOWL_BALLS else GLOBAL_AVG_ECON
            career_sr[mid]   = sr_snap
            career_econ[mid] = eco_snap
        for p, grp in mdf.groupby("batsman"):
            bat_runs[p]  = bat_runs.get(p,0)  + int(grp["runs_batsman"].sum())
            bat_balls[p] = bat_balls.get(p,0) + int((grp["is_legal"]==1).sum())
        for p, grp in mdf.groupby("bowler"):
            bowl_runs[p]  = bowl_runs.get(p,0)  + int(grp["total_runs"].sum())
            bowl_balls[p] = bowl_balls.get(p,0) + int((grp["is_legal"]==1).sum())

    return career_sr, career_econ


def compute_lineup_features(balls: pd.DataFrame,
                             career_sr: dict,
                             career_econ: dict) -> pd.DataFrame:
    n = len(balls)
    bat_str = np.zeros(n, dtype=np.float32)
    bowl_str = np.zeros(n, dtype=np.float32)
    wkt_wt   = np.zeros(n, dtype=np.float32)

    for (mid, inn), grp in tqdm(balls.groupby(["match_id","innings"], sort=False),
                                 desc="  Lineup features"):
        grp_s   = grp.sort_values(["over","ball"])
        idx_arr = grp_s.index.values

        batting_order, seen = [], set()
        for _, row in grp_s.iterrows():
            for p in [row["batsman"], row["non_striker"]]:
                if p and p not in seen:
                    seen.add(p); batting_order.append(p)

        sr_lu   = career_sr.get(mid,   {})
        eco_lu  = career_econ.get(mid, {})
        dismissed   = set()
        bowler_used = {}

        for i, (_, row) in enumerate(grp_s.iterrows()):
            idx     = idx_arr[i]
            cur_bat = row["batsman"]
            bowler  = row["bowler"]

            remaining   = [p for p in batting_order if p not in dismissed and p != cur_bat]
            undismissed = [p for p in batting_order if p not in dismissed]

            bat_str[idx]  = sum(sr_lu.get(p, GLOBAL_AVG_SR) for p in remaining)
            wkt_wt[idx]   = sum(sr_lu.get(p, GLOBAL_AVG_SR)/GLOBAL_AVG_SR for p in undismissed)

            total_bowl = 0.0
            for bn, bu in bowler_used.items():
                bl = max(0, BOWLER_QUOTA - bu)
                if bl > 0:
                    total_bowl += (1.0 / eco_lu.get(bn, GLOBAL_AVG_ECON)) * bl
            if bowler not in bowler_used:
                total_bowl += (1.0 / eco_lu.get(bowler, GLOBAL_AVG_ECON)) * BOWLER_QUOTA
            bowl_str[idx] = total_bowl

            if row["wicket"] == 1 and row.get("player_out"):
                dismissed.add(row["player_out"])
            if row["is_legal"] == 1:
                bowler_used[bowler] = bowler_used.get(bowler, 0) + 1

    balls = balls.copy()
    balls["batting_strength_remaining"] = bat_str
    balls["bowling_strength_remaining"] = bowl_str
    balls["wickets_remaining_weighted"] = wkt_wt
    return balls


# ════════════════════════════════════════════════════════════════════════════
# 3.  PACE/SPIN SPLITS  (wave 4b)
# ════════════════════════════════════════════════════════════════════════════

def compute_pace_spin_splits(all_balls: pd.DataFrame,
                              match_dates: pd.DataFrame,
                              bowler_group_map: dict,
                              target_match_ids: set) -> dict:
    df = all_balls.merge(match_dates, on="match_id", how="left")
    df["bowler_group"] = df["bowler"].map(bowler_group_map).fillna("pace")
    df = df.sort_values(["date","match_id","innings","over","ball"]).reset_index(drop=True)

    agg = (df.groupby(["match_id","date","batsman","bowler_group"])
           .agg(runs=("runs_batsman","sum"), legal=("is_legal","sum"))
           .reset_index())

    match_order = (df[df["match_id"].isin(target_match_ids)]
                   .groupby("match_id")["date"].first()
                   .sort_values().reset_index())

    ps = {}
    for _, mrow in tqdm(match_order.iterrows(), total=len(match_order),
                        desc="  Pace/spin splits"):
        mid, mdate = mrow["match_id"], mrow["date"]
        cutoff = mdate - pd.Timedelta(days=WINDOW_DAYS)
        window = agg[(agg["date"] >= cutoff) & (agg["date"] < mdate)]
        snap = {}
        for player, pg in window.groupby("batsman"):
            pace_r = pg[pg["bowler_group"]=="pace"]
            spin_r = pg[pg["bowler_group"]=="spin"]
            pl, sl = pace_r["legal"].sum(), spin_r["legal"].sum()
            snap[player] = {
                "sr_vs_pace": pace_r["runs"].sum()/pl*100 if pl >= MIN_BALLS_PS else GLOBAL_AVG_SR,
                "sr_vs_spin": spin_r["runs"].sum()/sl*100 if sl >= MIN_BALLS_PS else GLOBAL_AVG_SR,
            }
        ps[mid] = snap
    return ps


# ════════════════════════════════════════════════════════════════════════════
# 4.  OVER-PAR
# ════════════════════════════════════════════════════════════════════════════

def compute_over_par(league_balls: pd.DataFrame,
                     match_dates: pd.DataFrame,
                     matches_venue: pd.DataFrame,
                     train_cutoff) -> tuple:
    df = league_balls[league_balls["innings"]==1].copy()
    df["match_id"] = df["match_id"].astype(str)
    df = df.merge(match_dates, on="match_id", how="left")
    df = df[df["date"] < train_cutoff].copy()
    df = df.merge(matches_venue, on="match_id", how="left")

    df["over_completed"] = df.groupby("match_id")["is_legal"].cumsum() // 6
    over_scores = (df.groupby(["match_id","venue","over_completed"])["total_runs"]
                   .sum().groupby(level=[0,1]).cumsum().reset_index()
                   .rename(columns={"total_runs":"cum_runs"}))
    venue_med = (over_scores.groupby(["venue","over_completed"])["cum_runs"]
                 .median().reset_index()
                 .rename(columns={"cum_runs":"venue_median_at_over"}))
    global_med = (over_scores.groupby("over_completed")["cum_runs"]
                  .median().reset_index()
                  .rename(columns={"cum_runs":"global_median_at_over"}))
    return venue_med, global_med


# ════════════════════════════════════════════════════════════════════════════
# 5.  ATTACH ALL FEATURES
# ════════════════════════════════════════════════════════════════════════════

def attach_features(balls: pd.DataFrame,
                    stats: dict,
                    ps: dict,
                    over_par_data: tuple,
                    match_dates: pd.DataFrame,
                    matches_venue: pd.DataFrame,
                    bowler_group_map: dict,
                    player_hand_map: dict) -> pd.DataFrame:

    df  = balls.copy()
    df["match_id"] = df["match_id"].astype(str)
    n   = len(df)

    bat_sr       = np.full(n, GLOBAL_AVG_SR,   dtype=np.float32)
    bat_bdry     = np.full(n, GLOBAL_BDRY_PCT, dtype=np.float32)
    bat_dot      = np.full(n, GLOBAL_DOT_PCT,  dtype=np.float32)
    bowl_econ    = np.full(n, GLOBAL_AVG_ECON, dtype=np.float32)
    bowl_dot     = np.full(n, GLOBAL_DOT_PCT,  dtype=np.float32)
    bow_is_pace  = np.ones(n,  dtype=np.float32)
    bat_is_lhb   = np.zeros(n, dtype=np.float32)
    sr_vs_pace   = np.full(n, GLOBAL_AVG_SR, dtype=np.float32)
    sr_vs_spin   = np.full(n, GLOBAL_AVG_SR, dtype=np.float32)

    for i, row in tqdm(df.iterrows(), total=n, desc="  Attaching features", miniters=50000):
        mid = row["match_id"]
        if mid in stats:
            ms  = stats[mid]
            bat = row.get("batsman","")
            if bat in ms["bat"]:
                bs = ms["bat"][bat]
                bat_sr[i]   = bs["sr"]
                bat_bdry[i] = bs["bdry_pct"]
                bat_dot[i]  = bs["dot_pct"]
            bowl = row.get("bowler","")
            if bowl in ms["bowl"]:
                bls = ms["bowl"][bowl]
                bowl_econ[i] = bls["econ"]
                bowl_dot[i]  = bls["dot_pct"]

        bowler  = row.get("bowler","")
        batsman = row.get("batsman","")
        bow_is_pace[i] = 1.0 if bowler_group_map.get(bowler,"pace") == "pace" else 0.0
        bat_is_lhb[i]  = 1.0 if player_hand_map.get(batsman) == "Left" else 0.0

        if mid in ps and batsman in ps[mid]:
            sr_vs_pace[i] = ps[mid][batsman]["sr_vs_pace"]
            sr_vs_spin[i] = ps[mid][batsman]["sr_vs_spin"]

    df["batsman_sr_2yr"]           = bat_sr
    df["batsman_boundary_pct_2yr"] = bat_bdry
    df["batsman_dot_pct_2yr"]      = bat_dot
    df["bowler_economy_2yr"]       = bowl_econ
    df["bowler_dot_pct_2yr"]       = bowl_dot
    df["bowler_is_pace"]           = bow_is_pace
    df["batsman_is_lhb"]           = bat_is_lhb
    df["batsman_sr_vs_pace"]       = sr_vs_pace
    df["batsman_sr_vs_spin"]       = sr_vs_spin
    df["pace_spin_sr_diff"]        = df["batsman_sr_vs_spin"] - df["batsman_sr_vs_pace"]
    df["lhb_vs_spin"]              = ((df["batsman_is_lhb"]==1) &
                                      (df["bowler_is_pace"]==0)).astype(np.float32)

    # Momentum index
    print("  Computing momentum index...")
    df = df.sort_values(["match_id","innings","over","ball"]).reset_index(drop=True)
    r6  = df.groupby(["match_id","innings"])["total_runs"] \
            .transform(lambda x: x.rolling(6, min_periods=1).sum().shift(1, fill_value=0))
    r24 = df.groupby(["match_id","innings"])["total_runs"] \
            .transform(lambda x: x.rolling(24, min_periods=1).sum().shift(1, fill_value=0))
    w6  = df.groupby(["match_id","innings"])["wicket"] \
            .transform(lambda x: x.rolling(6, min_periods=1).sum().shift(1, fill_value=0))
    df["momentum_index"] = np.clip(
        (r6 / np.maximum(r24/4, 1.0)) * (1.0 - 0.15*w6), 0, 3
    ).astype(np.float32)

    # Over-par
    print("  Computing over_par...")
    venue_med, global_med = over_par_data
    df = df.merge(matches_venue, on="match_id", how="left")
    df["current_over"] = (
        df.groupby(["match_id","innings"])["is_legal"]
        .transform(lambda x: x.shift(1, fill_value=0).cumsum()) // 6
    ).clip(0, 19)
    df["cum_runs_before"] = (
        df.groupby(["match_id","innings"])["total_runs"]
        .transform(lambda x: x.shift(1, fill_value=0).cumsum())
    )
    df = df.merge(venue_med,  left_on=["venue","current_over"],
                  right_on=["venue","over_completed"], how="left")
    df = df.merge(global_med, left_on=["current_over"],
                  right_on=["over_completed"], how="left", suffixes=("","_global"))
    med = df["venue_median_at_over"].fillna(df["global_median_at_over"]).fillna(0)
    df["over_par"] = np.where(df["innings"]==1, df["cum_runs_before"]-med, 0.0).astype(np.float32)

    drop = ["venue","current_over","cum_runs_before","venue_median_at_over",
            "over_completed","global_median_at_over","over_completed_global"]
    df = df.drop(columns=[c for c in drop if c in df.columns], errors="ignore")
    return df


# ════════════════════════════════════════════════════════════════════════════
# PROCESS ONE LEAGUE
# ════════════════════════════════════════════════════════════════════════════

def process_league(league: str,
                   all_balls: pd.DataFrame,
                   all_match_dates: pd.DataFrame,
                   attrs: pd.DataFrame,
                   force: bool = False):

    output_path = Path(f"data/{league}_balls_wave4b.csv")
    if output_path.exists() and not force:
        print(f"  [{league.upper()}] wave4b already exists — skipping "
              f"(use --force to recompute)")
        return

    balls_path   = Path(f"data/{league}_balls.csv")
    matches_path = Path(f"data/{league}_matches.csv")

    if not balls_path.exists() or not matches_path.exists():
        print(f"  [{league.upper()}] raw files not found — skipping")
        return

    print(f"\n{'═'*65}")
    print(f"  ENRICHING {league.upper()}")
    print(f"{'═'*65}")

    league_balls   = pd.read_csv(balls_path)
    league_matches = pd.read_csv(matches_path, parse_dates=["date"])
    league_balls["match_id"]   = league_balls["match_id"].astype(str)
    league_matches["match_id"] = league_matches["match_id"].astype(str)

    # Men's only
    if "gender" in league_matches.columns:
        league_matches = league_matches[league_matches["gender"]=="male"].copy()
    valid_ids = set(league_matches["match_id"])
    league_balls = league_balls[league_balls["match_id"].isin(valid_ids)].copy()

    target_ids = set(league_balls["match_id"])

    bowler_group_map = attrs.set_index("player_name")["bowling_group"].to_dict()
    player_hand_map  = attrs.set_index("player_name")["batting_hand"].to_dict()

    # Slim columns for history pool
    SLIM = ["match_id","innings","over","ball","batsman","non_striker",
            "bowler","runs_batsman","total_runs","is_legal","wicket","player_out"]
    slim_balls = all_balls[[c for c in SLIM if c in all_balls.columns]].copy()

    # Step 1: rolling stats
    rolling = compute_rolling_player_stats(slim_balls, all_match_dates, target_ids)

    # Step 2: career stats + lineup features
    career_sr, career_econ = compute_career_stats(slim_balls, all_match_dates, target_ids)
    league_balls = compute_lineup_features(league_balls, career_sr, career_econ)

    # Step 3: pace/spin splits
    ps = compute_pace_spin_splits(slim_balls, all_match_dates, bowler_group_map, target_ids)

    # Step 4: over-par
    league_sorted  = league_matches.sort_values("date").reset_index(drop=True)
    train_cutoff   = league_sorted.iloc[int(len(league_sorted)*0.70)]["date"]
    league_dates   = league_matches[["match_id","date"]].copy()
    league_dates["match_id"] = league_dates["match_id"].astype(str)
    matches_venue  = league_matches[["match_id","venue"]].copy()
    matches_venue["match_id"] = matches_venue["match_id"].astype(str)
    over_par_data  = compute_over_par(league_balls, league_dates, matches_venue, train_cutoff)

    # Step 5: attach everything
    league_balls = attach_features(
        league_balls, rolling, ps, over_par_data,
        league_dates, matches_venue, bowler_group_map, player_hand_map
    )

    # Sanity checks
    WAVE_COLS = ["batting_strength_remaining","bowling_strength_remaining",
                 "wickets_remaining_weighted","batsman_sr_2yr",
                 "batsman_boundary_pct_2yr","batsman_dot_pct_2yr",
                 "bowler_economy_2yr","bowler_dot_pct_2yr",
                 "momentum_index","over_par","bowler_is_pace",
                 "batsman_sr_vs_pace","batsman_sr_vs_spin","pace_spin_sr_diff"]
    print("\n  ── Sanity checks ──────────────────────────────────────────────")
    for col in WAVE_COLS:
        if col in league_balls.columns:
            v = league_balls[col]
            print(f"    {col:35s}  mean={v.mean():8.3f}  std={v.std():7.3f}")

    league_balls.to_csv(output_path, index=False)
    print(f"\n  Saved → {output_path}  ({len(league_balls):,} rows)")


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Wave 4+4b enrichment for all non-IPL leagues"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--league", type=str,
                       help=f"Single league to enrich: {', '.join(ALL_LEAGUES)}")
    group.add_argument("--all", action="store_true",
                       help="Enrich all leagues missing wave4b files")
    parser.add_argument("--force", action="store_true",
                        help="Re-enrich even if wave4b file already exists")
    parser.add_argument("--all-balls",   default="data/all_balls.csv")
    parser.add_argument("--all-matches", default="data/all_matches.csv")
    parser.add_argument("--player-attrs", default="data/player_attributes.csv")
    args = parser.parse_args()

    leagues_to_run = ALL_LEAGUES if args.all else [args.league.lower()]

    # Filter to only leagues with missing wave4b (unless --force)
    if not args.force:
        leagues_to_run = [
            lg for lg in leagues_to_run
            if not Path(f"data/{lg}_balls_wave4b.csv").exists()
        ]
        if not leagues_to_run:
            print("All wave4b files already exist. Use --force to recompute.")
            return

    print("═" * 65)
    print(f"  WAVE 4 LEAGUES — Enriching: {', '.join(l.upper() for l in leagues_to_run)}")
    print("═" * 65)

    # Load shared resources once
    print("\nLoading combined ball history...")
    all_balls = pd.read_csv(args.all_balls)
    all_balls["match_id"] = all_balls["match_id"].astype(str)

    all_matches = pd.read_csv(args.all_matches, parse_dates=["date"])
    all_matches["match_id"] = all_matches["match_id"].astype(str)
    all_match_dates = all_matches[["match_id","date"]].drop_duplicates("match_id")

    attrs = pd.read_csv(args.player_attrs)

    print(f"  Combined pool: {len(all_balls):,} balls across "
          f"{all_matches['source'].nunique()} leagues")

    # Process each league
    for league in leagues_to_run:
        process_league(league, all_balls, all_match_dates, attrs, force=args.force)

    print("\nAll done.")


if __name__ == "__main__":
    main()