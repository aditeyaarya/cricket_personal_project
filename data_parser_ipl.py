import os
import yaml
import pandas as pd
from tqdm import tqdm

DATA_PATH = "data/ipl"

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

    matches.append({
        "match_id": match_id,
        "team1": teams[0],
        "team2": teams[1],
        "winner": winner,
        "toss_winner": toss_winner,
        "toss_decision": toss_decision,
        "venue": venue,
        "date": date
    })

    # Parse deliveries
    for innings_num, inning in enumerate(data["innings"], start=1):

        inning_key = list(inning.keys())[0]
        inning_data = inning[inning_key]

        batting_team = inning_data["team"]

        # Cumulative count of LEGAL deliveries bowled so far in this innings.
        # Wides and no-balls must NOT increment this counter.
        # Reset to 0 at the start of every innings.
        legal_ball_count = 0

        for delivery in inning_data["deliveries"]:

            # ── Step 1: unwrap the delivery ──────────────────────────────────
            # Each element of inning_data["deliveries"] is a SINGLE-KEY dict:
            #   { 0.2: { "batsman": ..., "extras": { "wides": 1 }, ... } }
            # The key is the over.ball float (e.g. 0.2).
            # The value is the delivery payload.
            #
            # The original code used list(delivery.keys())[0] to get the key
            # and delivery[ball] to get the value — that part was correct.
            # The root bug was that nothing below ever READ delivery_data.get("extras").
            delivery_data = list(delivery.values())[0]   # unwrap outer key → inner dict

            # ── Step 2: over and ball_in_over ─────────────────────────────────
            # ball_key is a float (yaml.safe_load returns numeric YAML keys as floats).
            # Convert via string to eliminate floating-point rounding risk:
            #   str(3.4) → "3.4" → over=3, ball_in_over=4   (safe)
            #   (3.4 - 3) * 10  → 3.999...  (risky with round())
            ball_key     = list(delivery.keys())[0]          # e.g. 0.2 as float
            ball_str     = str(ball_key)                     # "0.2"
            over         = int(ball_str.split(".")[0])       # 0
            ball_in_over = int(ball_str.split(".")[1])       # 2

            # ── Step 3: extras — THE CRITICAL FIX ────────────────────────────
            # "extras" only appears in delivery_data when extras actually occurred.
            # .get("extras", {}) safely returns {} for a normal delivery so the
            # subsequent .get("wides", 0) calls always succeed without KeyError.
            extras      = delivery_data.get("extras", {})
            wide_runs   = extras.get("wides",   0)
            noball_runs = extras.get("noballs", 0)
            bye_runs    = extras.get("byes",    0) + extras.get("legbyes", 0)

            # ── Step 4: legality ──────────────────────────────────────────────
            # A delivery is legal (counts as one of the 120 balls in an innings)
            # if and only if it is NOT a wide AND NOT a no-ball.
            # Byes and leg-byes on an otherwise legal ball still count as legal.
            is_legal_delivery = (wide_runs == 0 and noball_runs == 0)

            # ── Step 5: ball counter ──────────────────────────────────────────
            # Record legal_ball_before BEFORE incrementing so it represents
            # "how many legal balls had been bowled before THIS delivery".
            # The pipeline uses this directly as legal_ball_before (no shift needed).
            legal_ball_before      = legal_ball_count
            balls_remaining_before = max(0, 120 - legal_ball_before)

            if is_legal_delivery:
                legal_ball_count += 1    # wides and no-balls do NOT increment

            # ── Step 6: runs ──────────────────────────────────────────────────
            runs_block   = delivery_data.get("runs", {})
            runs_batsman = runs_block.get("batsman", 0)
            runs_extras  = runs_block.get("extras",  0)
            total_runs   = runs_block.get("total",   runs_batsman + runs_extras)

            # ── Step 7: wicket ────────────────────────────────────────────────
            # Cricsheet has two formats depending on file vintage:
            #   OLD (pre-2023): "wicket":  { "kind": "caught", "player_out": "..." }
            #   NEW (2023+):    "wickets": [ { "kind": "caught", "player_out": "..." } ]
            # The original code only checked "wicket" so the new format silently
            # dropped wickets for recent seasons. We normalise both to one dict.
            wicket_info = None
            if "wicket" in delivery_data:
                wicket_info = delivery_data["wicket"]            # old: dict
            elif "wickets" in delivery_data:
                wlist = delivery_data["wickets"]                 # new: list
                if wlist:
                    wicket_info = wlist[0]

            wicket      = 1 if wicket_info else 0
            wicket_kind = wicket_info.get("kind",       "") if wicket_info else ""
            player_out  = wicket_info.get("player_out", "") if wicket_info else ""

            balls.append({
                "match_id"             : match_id,
                "innings"              : innings_num,
                "over"                 : over,
                "ball"                 : ball_in_over,
                "legal_ball_before"    : legal_ball_before,       # state BEFORE delivery
                "balls_remaining_before": balls_remaining_before, # state BEFORE delivery
                "batting_team"         : batting_team,
                "batsman"              : delivery_data.get("batsman",     ""),
                "non_striker"          : delivery_data.get("non_striker", ""),
                "bowler"               : delivery_data.get("bowler",      ""),
                "runs_batsman"         : runs_batsman,
                "runs_extras"          : runs_extras,
                "total_runs"           : total_runs,
                "wide_runs"            : wide_runs,
                "noball_runs"          : noball_runs,
                "bye_runs"             : bye_runs,
                "is_legal"             : int(is_legal_delivery),
                "wicket"               : wicket,
                "wicket_kind"          : wicket_kind,
                "player_out"           : player_out,
            })

matches_df = pd.DataFrame(matches)
balls_df = pd.DataFrame(balls)

matches_df.to_csv("data/ipl_matches.csv", index=False)
balls_df.to_csv("data/ipl_balls.csv", index=False)

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

print("\n" + "=" * 60)