"""
data_parser_leagues.py  —  Universal Cricsheet T20 League Parser
================================================================
Parses any Cricsheet YAML folder into matches + balls CSVs.
Handles all three wicket formats (old dict, old list, new list).
Extracts gender, city, and event fields where present.

Supported leagues (pass via --league):
  ipl, t20i, bbl, cpl, psl, sa20, smat, (or any string you choose)

Usage
-----
  # Parse BBL
  python data_parser_leagues.py --league bbl --input data/raw/bbl --output-dir data/

  # Parse CPL
  python data_parser_leagues.py --league cpl --input data/raw/cpl --output-dir data/

  # Parse PSL
  python data_parser_leagues.py --league psl --input data/raw/psl --output-dir data/

  # Parse SA20
  python data_parser_leagues.py --league sa20 --input data/raw/sa20 --output-dir data/

  # Parse SMAT (with 2018 cutoff applied automatically)
  python data_parser_leagues.py --league smat --input data/raw/smat --output-dir data/

Output
------
  data/<league>_matches.csv
  data/<league>_balls.csv

The output schema matches ipl_matches.csv / ipl_balls.csv exactly,
so all downstream scripts (build_dataset.py, wave4_leagues.py,
pipeline.py) work without modification.
"""

import os
import argparse
import yaml
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from datetime import date


# ── SMAT cutoff: only use matches from 2018 onwards ───────────────────────────
SMAT_CUTOFF_YEAR = 2018


def parse_league(data_path: str, league: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Parse all YAML files in data_path for the given league.
    Returns (matches_df, balls_df).
    """
    files = sorted([f for f in os.listdir(data_path) if f.endswith(".yaml")])
    if not files:
        raise FileNotFoundError(f"No YAML files found in {data_path}")

    matches = []
    balls   = []
    skipped = 0

    for file in tqdm(files, desc=f"Parsing {league.upper()}"):

        with open(os.path.join(data_path, file), "r") as f:
            data = yaml.safe_load(f)

        info     = data["info"]
        match_id = file.replace(".yaml", "")
        teams    = info["teams"]
        date_val = info["dates"][0]

        # ── SMAT cutoff ───────────────────────────────────────────────────────
        if league == "smat":
            cutoff = date(SMAT_CUTOFF_YEAR, 1, 1)
            match_date = date_val if isinstance(date_val, date) else \
                         date.fromisoformat(str(date_val))
            if match_date < cutoff:
                skipped += 1
                continue

        # ── Match-level fields ────────────────────────────────────────────────
        winner = None
        if "winner" in info.get("outcome", {}):
            winner = info["outcome"]["winner"]

        toss_winner   = info["toss"]["winner"]
        toss_decision = info["toss"]["decision"]
        venue         = info.get("venue", None)
        city          = info.get("city", None)
        gender        = info.get("gender", None)

        matches.append({
            "match_id"     : match_id,
            "team1"        : teams[0],
            "team2"        : teams[1],
            "winner"       : winner,
            "toss_winner"  : toss_winner,
            "toss_decision": toss_decision,
            "venue"        : venue,
            "city"         : city,
            "date"         : date_val,
            "gender"       : gender,
        })

        # ── Deliveries ────────────────────────────────────────────────────────
        for innings_num, inning in enumerate(data["innings"], start=1):

            inning_key  = list(inning.keys())[0]
            inning_data = inning[inning_key]
            batting_team = inning_data["team"]
            legal_ball_count = 0

            for delivery in inning_data["deliveries"]:

                delivery_data = list(delivery.values())[0]

                ball_key     = list(delivery.keys())[0]
                ball_str     = str(ball_key)
                over         = int(ball_str.split(".")[0])
                ball_in_over = int(ball_str.split(".")[1])

                extras      = delivery_data.get("extras", {})
                wide_runs   = extras.get("wides",   0)
                noball_runs = extras.get("noballs", 0)
                bye_runs    = extras.get("byes",    0) + extras.get("legbyes", 0)

                is_legal_delivery = (wide_runs == 0 and noball_runs == 0)

                legal_ball_before      = legal_ball_count
                balls_remaining_before = max(0, 120 - legal_ball_before)

                if is_legal_delivery:
                    legal_ball_count += 1

                runs_block   = delivery_data.get("runs", {})
                runs_batsman = runs_block.get("batsman", 0)
                runs_extras  = runs_block.get("extras",  0)
                total_runs   = runs_block.get("total",   runs_batsman + runs_extras)

                # Three wicket formats: new list, old list, old dict
                wicket_info = None
                raw = delivery_data.get("wickets") or delivery_data.get("wicket")
                if raw is not None:
                    if isinstance(raw, list):
                        wicket_info = raw[0] if raw else None
                    elif isinstance(raw, dict):
                        wicket_info = raw

                wicket      = 1 if wicket_info else 0
                wicket_kind = wicket_info.get("kind",       "") if wicket_info else ""
                player_out  = wicket_info.get("player_out", "") if wicket_info else ""

                balls.append({
                    "match_id"              : match_id,
                    "innings"               : innings_num,
                    "over"                  : over,
                    "ball"                  : ball_in_over,
                    "legal_ball_before"     : legal_ball_before,
                    "balls_remaining_before": balls_remaining_before,
                    "batting_team"          : batting_team,
                    "batsman"               : delivery_data.get("batsman",     ""),
                    "non_striker"           : delivery_data.get("non_striker", ""),
                    "bowler"                : delivery_data.get("bowler",      ""),
                    "runs_batsman"          : runs_batsman,
                    "runs_extras"           : runs_extras,
                    "total_runs"            : total_runs,
                    "wide_runs"             : wide_runs,
                    "noball_runs"           : noball_runs,
                    "bye_runs"              : bye_runs,
                    "is_legal"              : int(is_legal_delivery),
                    "wicket"                : wicket,
                    "wicket_kind"           : wicket_kind,
                    "player_out"            : player_out,
                })

    matches_df = pd.DataFrame(matches)
    balls_df   = pd.DataFrame(balls)

    if skipped:
        print(f"  Skipped {skipped:,} matches (SMAT pre-{SMAT_CUTOFF_YEAR} cutoff)")

    return matches_df, balls_df


def run_sanity_checks(matches_df: pd.DataFrame, balls_df: pd.DataFrame, league: str):
    print(f"\n{'='*60}")
    print(f"SANITY CHECKS — {league.upper()}")
    print(f"{'='*60}")
    print(f"  Matches : {len(matches_df):,}")
    print(f"  Balls   : {len(balls_df):,}")

    if balls_df.empty:
        print("  WARNING: no balls parsed")
        return

    # 1. Legal ball counts ≤ 120
    max_legal = balls_df.groupby(["match_id", "innings"])["legal_ball_before"].max()
    over_120  = (max_legal > 120).sum()
    print(f"\n[1] Max legal_ball_before : {max_legal.max()}  (must be ≤ 120)")
    print("    ✓ OK" if not over_120 else f"    WARNING: {over_120} innings exceeded 120")

    # 2. Wides and no-balls present
    n_wides   = (balls_df["wide_runs"]   > 0).sum()
    n_noballs = (balls_df["noball_runs"] > 0).sum()
    total     = len(balls_df)
    print(f"\n[2] Wides: {n_wides:,} ({n_wides/total*100:.1f}%)  "
          f"No-balls: {n_noballs:,} ({n_noballs/total*100:.1f}%)")
    print("    ✓ OK" if n_wides > 0 else "    WARNING: zero wides")

    # 3. Legal rate 85–95%
    pct_legal = balls_df["is_legal"].mean() * 100
    print(f"\n[3] Legal delivery rate: {pct_legal:.1f}%  (expected 85–95%)")
    print("    ✓ OK" if 85 <= pct_legal <= 95 else "    WARNING: outside expected range")

    # 4. Gender split
    if "gender" in matches_df.columns:
        print(f"\n[4] Gender split:\n{matches_df['gender'].value_counts().to_string()}")

    # 5. Date range
    if "date" in matches_df.columns:
        dates = pd.to_datetime(matches_df["date"])
        print(f"\n[5] Date range: {dates.min().date()} → {dates.max().date()}")

    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="Universal Cricsheet T20 league parser"
    )
    parser.add_argument("--league",     required=True,
                        help="League name tag, e.g. bbl, cpl, psl, sa20, smat")
    parser.add_argument("--input",      required=True,
                        help="Folder containing .yaml match files")
    parser.add_argument("--output-dir", default="data",
                        help="Output directory (default: data/)")
    args = parser.parse_args()

    league     = args.league.lower()
    data_path  = args.input
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("═" * 65)
    print(f"  LEAGUE PARSER — {league.upper()}")
    print("═" * 65)
    print(f"  Input  : {data_path}")
    print(f"  Output : {output_dir}/")

    matches_df, balls_df = parse_league(data_path, league)

    run_sanity_checks(matches_df, balls_df, league)

    matches_out = output_dir / f"{league}_matches.csv"
    balls_out   = output_dir / f"{league}_balls.csv"

    matches_df.to_csv(matches_out, index=False)
    balls_df.to_csv(balls_out,     index=False)

    print(f"\nSaved → {matches_out}  ({len(matches_df):,} matches)")
    print(f"Saved → {balls_out}  ({len(balls_df):,} balls)")
    print("Done.")


if __name__ == "__main__":
    main()