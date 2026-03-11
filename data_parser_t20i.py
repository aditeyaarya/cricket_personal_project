import os
import yaml
import pandas as pd
from tqdm import tqdm

DATA_PATH = "data/t20i"   # ← folder containing your T20I .yaml files

matches = []
balls = []

files = [f for f in os.listdir(DATA_PATH) if f.endswith(".yaml")]

for file in tqdm(files):

    with open(os.path.join(DATA_PATH, file), "r") as f:
        data = yaml.safe_load(f)

    info = data["info"]

    match_id = file.replace(".yaml", "")
    teams = info["teams"]

    winner = None
    if "winner" in info.get("outcome", {}):
        winner = info["outcome"]["winner"]

    toss_winner = info["toss"]["winner"]
    toss_decision = info["toss"]["decision"]

    venue = info.get("venue", None)
    date = info["dates"][0]

    # T20I-specific extra fields
    gender = info.get("gender", None)
    city   = info.get("city", None)

    matches.append({
        "match_id"     : match_id,
        "team1"        : teams[0],
        "team2"        : teams[1],
        "winner"       : winner,
        "toss_winner"  : toss_winner,
        "toss_decision": toss_decision,
        "venue"        : venue,
        "city"         : city,
        "date"         : date,
        "gender"       : gender,
    })

    # ── Parse deliveries ──────────────────────────────────────────────────────
    for innings_num, inning in enumerate(data["innings"], start=1):

        inning_key  = list(inning.keys())[0]
        inning_data = inning[inning_key]

        batting_team = inning_data["team"]

        # Cumulative count of LEGAL deliveries bowled so far in this innings.
        # Wides and no-balls must NOT increment this counter.
        legal_ball_count = 0

        for delivery in inning_data["deliveries"]:

            # ── Step 1: unwrap the delivery ──────────────────────────────────
            delivery_data = list(delivery.values())[0]

            # ── Step 2: over and ball_in_over ─────────────────────────────────
            ball_key     = list(delivery.keys())[0]
            ball_str     = str(ball_key)
            over         = int(ball_str.split(".")[0])
            ball_in_over = int(ball_str.split(".")[1])

            # ── Step 3: extras ────────────────────────────────────────────────
            extras      = delivery_data.get("extras", {})
            wide_runs   = extras.get("wides",   0)
            noball_runs = extras.get("noballs", 0)
            bye_runs    = extras.get("byes",    0) + extras.get("legbyes", 0)

            # ── Step 4: legality ──────────────────────────────────────────────
            is_legal_delivery = (wide_runs == 0 and noball_runs == 0)

            # ── Step 5: ball counter ──────────────────────────────────────────
            legal_ball_before       = legal_ball_count
            balls_remaining_before  = max(0, 120 - legal_ball_before)

            if is_legal_delivery:
                legal_ball_count += 1

            # ── Step 6: runs ──────────────────────────────────────────────────
            runs_block   = delivery_data.get("runs", {})
            runs_batsman = runs_block.get("batsman", 0)
            runs_extras  = runs_block.get("extras",  0)
            total_runs   = runs_block.get("total",   runs_batsman + runs_extras)

            # ── Step 7: wicket ────────────────────────────────────────────────
            # Three formats exist across Cricsheet vintages:
            #   A) "wicket":  { "kind": "...", "player_out": "..." }   old dict
            #   B) "wicket":  [ { "kind": "...", "player_out": "..." } ] old list
            #   C) "wickets": [ { "kind": "...", "player_out": "..." } ] new list
            # Normalise all three to a single dict (or None).
            wicket_info = None
            raw = delivery_data.get("wickets") or delivery_data.get("wicket")
            if raw is not None:
                if isinstance(raw, list):
                    wicket_info = raw[0] if raw else None   # formats B & C
                elif isinstance(raw, dict):
                    wicket_info = raw                       # format A

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

matches_df.to_csv("data/t20i_matches.csv", index=False)
balls_df.to_csv("data/t20i_balls.csv",   index=False)

print("Parsing complete")
print(f"Matches : {matches_df.shape}")
print(f"Balls   : {balls_df.shape}")

# ════════════════════════════════════════════════════════════════
# SANITY CHECKS
# ════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("SANITY CHECKS")
print("=" * 60)

# 1. Legal ball counts must never exceed 120 per innings
max_legal = (
    balls_df.groupby(["match_id", "innings"])["legal_ball_before"].max()
)
over_120 = (max_legal > 120).sum()
print(f"\n[1] Max legal_ball_before per innings : {max_legal.max()}  (must be ≤ 120)")
if over_120:
    print(f"    WARNING: {over_120} innings exceeded 120 — check parser")
else:
    print("    ✓ All innings within 120 legal deliveries")

# 2. Wides and no-balls must be non-zero across the dataset
n_wides   = (balls_df["wide_runs"]   > 0).sum()
n_noballs = (balls_df["noball_runs"] > 0).sum()
total     = len(balls_df)
print(f"\n[2] Wide deliveries  : {n_wides:,}  ({n_wides/total*100:.1f}% of all deliveries)")
print(f"    No-ball deliveries: {n_noballs:,}  ({n_noballs/total*100:.1f}% of all deliveries)")
if n_wides == 0 and n_noballs == 0:
    print("    WARNING: zero wides and no-balls — extras are NOT being parsed")
else:
    print("    ✓ Wides and no-balls are non-zero")

# 3. Legal rate should be roughly 85-92 %
pct_legal = balls_df["is_legal"].mean() * 100
print(f"\n[3] Legal delivery rate : {pct_legal:.1f}%  (expected 85–92%)")
if pct_legal < 85 or pct_legal > 95:
    print("    WARNING: legal rate outside expected range")
else:
    print("    ✓ Legal rate within expected range")

# 4. balls_remaining_before must be ≥ 0 everywhere
neg = (balls_df["balls_remaining_before"] < 0).sum()
print(f"\n[4] Negative balls_remaining_before : {neg}  (must be 0)")
print("    ✓ No negative values" if neg == 0 else "    WARNING: negative values found")

# 5. Gender split (T20I-specific)
print(f"\n[5] Gender split:\n{matches_df['gender'].value_counts().to_string()}")

print("\n" + "=" * 60)