import pandas as pd
import requests
import time
import logging
from tqdm import tqdm
from rapidfuzz import process, fuzz
from pathlib import Path

# -------------------------------------------------------
# CONFIG
# -------------------------------------------------------

BALLS_PATH = "data/all1_balls.csv"
PEOPLE_PATH = "data/cricsheet_register/people.csv"
OUTPUT_PATH = "data/player_attributes.csv"

CRICINFO_API = "https://site.web.api.espn.com/apis/common/v3/sports/cricket/athletes/{}"

PLAYER_COLUMNS = [
    "batsman",
    "bowler",
    "non_striker",
    "player_dismissed"
]

# -------------------------------------------------------
# LOGGING SETUP
# -------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)

file_handler = logging.FileHandler("player_attribute_build.log")
file_handler.setFormatter(
    logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
)

logging.getLogger().addHandler(file_handler)

# -------------------------------------------------------
# NORMALIZATION FUNCTIONS
# -------------------------------------------------------

def normalize_batting_hand(raw):

    if not raw:
        return ""

    raw = raw.lower()

    if "right" in raw:
        return "RHB"

    if "left" in raw:
        return "LHB"

    return ""


def normalize_bowling_style(raw):

    if not raw:
        return "none", "none"

    raw = raw.lower()

    spin_terms = [
        "offbreak",
        "legbreak",
        "orthodox",
        "wrist",
        "chinaman"
    ]

    pace_terms = [
        "fast",
        "medium",
        "fast-medium",
        "medium-fast"
    ]

    if any(t in raw for t in spin_terms):
        return raw, "spin"

    if any(t in raw for t in pace_terms):
        return raw, "pace"

    return raw, "none"


def normalize_role(raw):

    if not raw:
        return ""

    raw = raw.lower()

    if "wicketkeeper" in raw:
        return "wk_batsman"

    if "allrounder" in raw:
        return "allrounder"

    if "bowler" in raw:
        return "bowler"

    if "batter" in raw or "batsman" in raw:
        return "batsman"

    return ""


# -------------------------------------------------------
# FETCH PLAYER ATTRIBUTES FROM CRICINFO API
# -------------------------------------------------------

def fetch_player_attributes(player_name, cricinfo_id):

    url = CRICINFO_API.format(cricinfo_id)

    delay = 2

    logging.info(f"Fetching attributes for {player_name} (Cricinfo ID: {cricinfo_id})")

    for attempt in range(5):

        try:

            resp = requests.get(url, timeout=15)

            if resp.status_code == 200:

                data = resp.json()
                athlete = data.get("athlete", {})

                batting = athlete.get("batStyle")
                if batting:
                    batting = batting[0]["description"]
                bowling = athlete.get("bowlStyle")
                if bowling:
                    bowling = bowling[0]["description"]
                role = athlete.get("position")
                if role:
                    role = role.get("name")

                logging.info(
                    f"SUCCESS {player_name} | Bat: {batting} | Bowl: {bowling} | Role: {role}"
                )

                return batting, bowling, role

            else:

                logging.warning(
                    f"Attempt {attempt+1}/3 failed for {player_name} | HTTP {resp.status_code}"
                )

        except Exception as e:

            logging.warning(
                f"Attempt {attempt+1}/3 exception for {player_name}: {str(e)}"
            )

        time.sleep(delay)
        delay *= 2

    logging.error(f"FAILED to fetch attributes for {player_name}")

    return None


# -------------------------------------------------------
# EXTRACT UNIQUE PLAYERS FROM BALL DATA
# -------------------------------------------------------

def extract_unique_players():

    df = pd.read_csv(BALLS_PATH)

    players = set()

    for col in PLAYER_COLUMNS:

        if col in df.columns:

            players.update(df[col].dropna().unique())

    players = sorted(players)

    logging.info(f"{len(players)} unique players found in dataset")

    return players


# -------------------------------------------------------
# LOAD CRICSHEET REGISTER
# -------------------------------------------------------

def load_register():

    people = pd.read_csv(PEOPLE_PATH, dtype=str).fillna("")

    logging.info(f"{len(people)} players loaded from Cricsheet register")

    return people


# -------------------------------------------------------
# MATCH PLAYER TO CRICSHEET
# -------------------------------------------------------

def match_player(name, people):

    logging.info(f"Matching player: {name}")

    exact = people[people["unique_name"] == name]

    if len(exact) > 0:
        logging.info(f"Exact unique_name match for {name}")
        return exact.iloc[0]

    exact = people[people["name"] == name]

    if len(exact) > 0:
        logging.info(f"Exact name match for {name}")
        return exact.iloc[0]

    choices = people["unique_name"].tolist()

    match = process.extractOne(name, choices, scorer=fuzz.WRatio)

    if match and match[1] > 90:

        logging.info(f"Fuzzy match {name} → {match[0]} ({match[1]})")

        row = people[people["unique_name"] == match[0]]

        if len(row) > 0:
            return row.iloc[0]

    logging.warning(f"No match found for {name}")

    return None


# -------------------------------------------------------
# MAIN PIPELINE
# -------------------------------------------------------

def build_dataset():

    players = extract_unique_players()

    people = load_register()

    existing_players = set()

    if Path(OUTPUT_PATH).exists():

        df_existing = pd.read_csv(OUTPUT_PATH)

        existing_players = set(df_existing["player_name"])

        logging.info(f"{len(existing_players)} players already in dataset")

    rows = []

    for player in tqdm(players, desc="Building player attributes"):

        logging.info(f"Processing player: {player}")

        if player in existing_players:

            logging.info(f"Skipping {player} (already processed)")
            continue

        row = match_player(player, people)

        if row is None:

            logging.warning(f"No Cricsheet mapping for {player}")

            rows.append({
                "player_name": player,
                "cricsheet_id": "",
                "batting_hand": "",
                "bowling_style_raw": "none",
                "bowling_group": "none",
                "role": ""
            })

            continue

        cricsheet_id = row["identifier"]
        cricinfo_id = row["key_cricinfo"]

        logging.info(
            f"{player} -> Cricsheet ID: {cricsheet_id} | Cricinfo ID: {cricinfo_id}"
        )

        batting_hand = ""
        bowling_style_raw = "none"
        bowling_group = "none"
        role = ""

        if cricinfo_id:

            attrs = fetch_player_attributes(player, cricinfo_id)

            if attrs:

                batting_raw, bowling_raw, role_raw = attrs

                batting_hand = normalize_batting_hand(batting_raw)
                bowling_style_raw, bowling_group = normalize_bowling_style(bowling_raw)
                role = normalize_role(role_raw)

                logging.info(
                    f"Normalized {player} | Batting: {batting_hand} | Bowling: {bowling_group} | Role: {role}"
                )

        rows.append({
            "player_name": player,
            "cricsheet_id": cricsheet_id,
            "batting_hand": batting_hand,
            "bowling_style_raw": bowling_style_raw,
            "bowling_group": bowling_group,
            "role": role
        })

        time.sleep(0.3)

    df_new = pd.DataFrame(rows)

    if Path(OUTPUT_PATH).exists():

        df_old = pd.read_csv(OUTPUT_PATH)

        df = pd.concat([df_old, df_new], ignore_index=True)

    else:

        df = df_new

    df.to_csv(OUTPUT_PATH, index=False)

    logging.info(f"Dataset saved to {OUTPUT_PATH}")
    logging.info(f"Total players in dataset: {len(df)}")


# -------------------------------------------------------
# RUN
# -------------------------------------------------------

if __name__ == "__main__":

    logging.info("Starting player attribute build")

    build_dataset()

    logging.info("Finished building dataset")