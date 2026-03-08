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

        for delivery in inning_data["deliveries"]:

            ball = list(delivery.keys())[0]
            delivery_data = delivery[ball]

            over = int(ball)
            ball_in_over = int(round((ball - over) * 10))

            batsman = delivery_data["batsman"]
            bowler = delivery_data["bowler"]

            runs_batsman = delivery_data["runs"]["batsman"]
            runs_extras = delivery_data["runs"]["extras"]
            total_runs = delivery_data["runs"]["total"]

            wicket = 1 if "wicket" in delivery_data else 0

            balls.append({
                "match_id": match_id,
                "innings": innings_num,
                "over": over,
                "ball": ball_in_over,
                "batting_team": batting_team,
                "batsman": batsman,
                "bowler": bowler,
                "runs_batsman": runs_batsman,
                "runs_extras": runs_extras,
                "total_runs": total_runs,
                "wicket": wicket
            })

matches_df = pd.DataFrame(matches)
balls_df = pd.DataFrame(balls)

matches_df.to_csv("data/ipl_matches.csv", index=False)
balls_df.to_csv("data/ipl_balls.csv", index=False)

print("Parsing complete")
print("Matches:", matches_df.shape)
print("Balls:", balls_df.shape)