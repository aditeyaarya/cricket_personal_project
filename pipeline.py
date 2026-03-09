"""
T20 Cricket Win Probability Pipeline  —  v7 (Wave 3)
=====================================================
Changes vs v6
--------------
  WAVE 3a — Wicket-Conditional Projection (Model T overhaul):
    Instead of a single LightGBM regressor, Model T now trains THREE
    models targeting different quantiles:
      model_T_opt  (alpha=0.75) — optimistic scenario
      model_T_exp  (regression) — expected / median scenario
      model_T_pes  (alpha=0.25) — pessimistic scenario
    The blended projection weights optimistic more when few wickets are
    lost and pessimistic more when many are down.  This gives Model W a
    richer projected total that encodes wicket-state uncertainty.

  WAVE 3b — Temperature Scaling replaces Platt calibration:
    v6 showed uncalibrated (0.1726) beating Platt-calibrated (0.1744).
    Temperature scaling is a single-parameter method that divides logits
    by a learned T, minimising NLL on the val set.  Lighter touch than
    Platt's two-parameter sigmoid.

  WAVE 3c — Partnership features added to INN1_FEATURES:
    partnership_runs  — runs scored in the current partnership so far
    partnership_balls — legal balls faced in the current partnership
    These signal batting stability: a team at 50/1 after a 45-run stand
    is better placed than 50/1 where the wicket just fell.

Project layout expected
------------------------
  personal_project/
  ├── pipeline.py           ← this file
  ├── wave2.py              ← lineup strength features
  ├── fetch_weather.py      ← weather data
  ├── data/
  │   ├── ipl_matches.csv
  │   ├── ipl_balls.csv
  │   ├── ipl_balls_wave2.csv
  │   └── weather.csv
  └── output/
      ├── models/
      ├── plots/
      └── logs/
"""

import os, sys, textwrap
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize_scalar
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
import lightgbm as lgb
import matplotlib.pyplot as plt
import joblib, warnings
warnings.filterwarnings("ignore")


# ════════════════════════════════════════════════════════════════════════════
# 0.  OUTPUT FOLDER SETUP
# ════════════════════════════════════════════════════════════════════════════

OUT_ROOT   = Path("output")
OUT_MODELS = OUT_ROOT / "models"
OUT_PLOTS  = OUT_ROOT / "plots"
OUT_LOGS   = OUT_ROOT / "logs"

for folder in [OUT_MODELS, OUT_PLOTS, OUT_LOGS]:
    folder.mkdir(parents=True, exist_ok=True)

_log_path = OUT_LOGS / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
_log_file = open(_log_path, "w", buffering=1)

class _Tee:
    def __init__(self, *streams): self.streams = streams
    def write(self, data):
        for s in self.streams: s.write(data)
    def flush(self):
        for s in self.streams: s.flush()

sys.stdout = _Tee(sys.__stdout__, _log_file)

print(f"Output folder  : {OUT_ROOT.resolve()}")
print(f"Log file       : {_log_path}")


# ════════════════════════════════════════════════════════════════════════════
# 1.  LOAD DATA  (prefer wave2-enriched balls if available)
# ════════════════════════════════════════════════════════════════════════════

WAVE2_BALLS = Path("data/ipl_balls_wave2.csv")
RAW_BALLS   = Path("data/ipl_balls.csv")

matches = pd.read_csv("data/ipl_matches.csv", parse_dates=["date"])
matches = matches.sort_values("date").reset_index(drop=True)

if WAVE2_BALLS.exists():
    print(f"\nLoading Wave 2 enriched balls from {WAVE2_BALLS}")
    balls = pd.read_csv(WAVE2_BALLS)
    WAVE2_COLS = ["batting_strength_remaining", "bowling_strength_remaining",
                  "wickets_remaining_weighted"]
    missing = [c for c in WAVE2_COLS if c not in balls.columns]
    if missing:
        raise ValueError(f"Wave 2 columns missing from {WAVE2_BALLS}: {missing}\n"
                         f"Re-run wave2.py to regenerate.")
    print(f"  Wave 2 columns present: {WAVE2_COLS}")
else:
    print(f"\n{WAVE2_BALLS} not found — computing Wave 2 features inline...")
    balls = pd.read_csv(RAW_BALLS)
    sys.path.insert(0, str(Path(__file__).parent))
    from wave2 import compute_career_stats, compute_lineup_features
    balls["match_id"]   = balls["match_id"].astype(str)
    matches_tmp = matches.copy()
    matches_tmp["match_id"] = matches_tmp["match_id"].astype(str)
    match_dates = matches_tmp[["match_id", "date"]].drop_duplicates("match_id")
    career_sr, career_econ = compute_career_stats(balls, match_dates)
    balls = compute_lineup_features(balls, career_sr, career_econ)
    balls.to_csv(WAVE2_BALLS, index=False)
    print(f"  Saved enriched balls to {WAVE2_BALLS}")

print(f"Matches loaded : {len(matches)}")
print(f"Deliveries     : {len(balls)}")


# ════════════════════════════════════════════════════════════════════════════
# 2.  WEATHER — fetch or load
# ════════════════════════════════════════════════════════════════════════════

WEATHER_CSV = Path("data/weather.csv")

if WEATHER_CSV.exists():
    print(f"\nWeather data   : loading from {WEATHER_CSV}")
    weather_df = pd.read_csv(WEATHER_CSV)
else:
    print("\nWeather data   : not found — fetching from Open-Meteo (~5 min)...")
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        from fetch_weather import build_weather_table
        weather_df = build_weather_table(
            matches_csv = "data/ipl_matches.csv",
            output_csv  = str(WEATHER_CSV),
        )
    except ImportError:
        print("ERROR: fetch_weather.py not found next to pipeline.py.")
        sys.exit(1)

WEATHER_FEATURES = [
    "temp_max_c", "humidity_eve_pct",
    "cloud_cover_eve_pct", "dew_point_eve_c", "precipitation_mm",
]
weather_cols = ["match_id"] + [c for c in WEATHER_FEATURES if c in weather_df.columns]
weather_df   = weather_df[weather_cols].copy()
weather_df["match_id"] = weather_df["match_id"].astype(str)
matches["match_id"]    = matches["match_id"].astype(str)

before = len(matches)
matches = matches.merge(weather_df, on="match_id", how="left")
found   = matches[WEATHER_FEATURES[0]].notna().sum() if WEATHER_FEATURES[0] in matches.columns else 0
print(f"Weather merge  : {found}/{before} matches have real weather data")

for col in WEATHER_FEATURES:
    if col in matches.columns:
        matches[col] = matches[col].fillna(matches[col].median())
    else:
        matches[col] = np.nan

WEATHER_FEATURES = [c for c in WEATHER_FEATURES
                    if c in matches.columns and matches[c].std() > 0]
print(f"Weather features used: {WEATHER_FEATURES}")


# ════════════════════════════════════════════════════════════════════════════
# 3.  LEGAL BALL COUNTING
# ════════════════════════════════════════════════════════════════════════════

def add_legality_flag(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "wide_runs" not in df.columns or "noball_runs" not in df.columns:
        df["wide_runs"]   = 0
        df["noball_runs"] = 0
        print("WARNING: wide_runs / noball_runs not found — all balls treated as legal.")
    df["is_legal"] = ((df["wide_runs"] == 0) & (df["noball_runs"] == 0)).astype(int)
    return df

balls = add_legality_flag(balls)
balls = balls.sort_values(["match_id","innings","over","ball"]).reset_index(drop=True)
balls["legal_ball_number"] = (
    balls.groupby(["match_id","innings"])["is_legal"].cumsum().clip(0, 120)
)
balls["balls_remaining"] = (120 - balls["legal_ball_number"]).clip(0, 120)

print(f"\nLegal ball counts — min:{balls['legal_ball_number'].min()}  "
      f"max:{balls['legal_ball_number'].max()}  (both should be 0–120)")


# ════════════════════════════════════════════════════════════════════════════
# 3.7  PARTNERSHIP FEATURES  (Wave 3c)
# ════════════════════════════════════════════════════════════════════════════
# Track runs and legal balls in the current partnership.  Reset to 0
# whenever a wicket falls.  Values represent state BEFORE this delivery.

print("\n── Computing partnership features ───────────────────────────────────")

balls["partnership_runs"]  = 0
balls["partnership_balls"] = 0

for (mid, inn), grp in balls.groupby(["match_id", "innings"], sort=False):
    grp_sorted = grp.sort_values(["over", "ball"])
    idx_arr = grp_sorted.index.values
    p_runs = 0
    p_balls = 0
    for i, (_, row) in enumerate(grp_sorted.iterrows()):
        balls.at[idx_arr[i], "partnership_runs"]  = p_runs
        balls.at[idx_arr[i], "partnership_balls"] = p_balls
        p_runs  += row["total_runs"]
        p_balls += int(row["is_legal"])
        if row["wicket"] == 1:
            p_runs  = 0
            p_balls = 0

print(f"  partnership_runs   mean={balls['partnership_runs'].mean():.1f}  "
      f"max={balls['partnership_runs'].max()}")
print(f"  partnership_balls  mean={balls['partnership_balls'].mean():.1f}  "
      f"max={balls['partnership_balls'].max()}")


# ════════════════════════════════════════════════════════════════════════════
# 4.  CONDITIONS FEATURES
# ════════════════════════════════════════════════════════════════════════════

if "start_time" in matches.columns:
    def parse_hour(t):
        try: return int(str(t).split(":")[0])
        except: return 19
    matches["start_hour"] = matches["start_time"].apply(parse_hour)
else:
    matches["start_hour"] = 19
    print("INFO: 'start_time' not found — defaulting to evening (19:30).")

matches["is_day_match"] = (matches["start_hour"] < 16).astype(int)

SUBCONTINENT_KW = [
    "mumbai","delhi","chennai","kolkata","bangalore","hyderabad",
    "pune","rajkot","indore","nagpur","ahmedabad","chandigarh",
    "jaipur","cuttack","visakhapatnam","dharamsala","mohali",
    "lahore","karachi","dhaka","colombo","dubai","abu dhabi","sharjah"
]

def dew_flag(row):
    if row["is_day_match"]: return 0
    return int(any(kw in str(row.get("venue","")).lower() for kw in SUBCONTINENT_KW))

matches["dew_likely"]   = matches.apply(dew_flag, axis=1)
matches["season_phase"] = matches["date"].dt.month.apply(
    lambda m: 0 if m <= 4 else (1 if m <= 5 else 2)
)

inn1_totals_match = (
    balls[balls["innings"] == 1]
    .groupby("match_id")["total_runs"].sum()
    .rename("innings1_total_match").reset_index()
)
matches["match_id"] = matches["match_id"].astype(str)
inn1_totals_match["match_id"] = inn1_totals_match["match_id"].astype(str)
matches = matches.merge(inn1_totals_match, on="match_id", how="left")
matches = matches.sort_values("date").reset_index(drop=True)

global_mean_total = matches["innings1_total_match"].mean()
n          = len(matches)
val_start  = int(n * 0.70)
test_start = int(n * 0.80)

venue_avg_map = (
    matches.iloc[:val_start]
    .groupby("venue")["innings1_total_match"].mean().to_dict()
)
matches["venue_avg_total"] = (
    matches["venue"].map(venue_avg_map).fillna(global_mean_total)
)

print(f"\nVenue avg total (global mean={global_mean_total:.1f}  "
      f"std={matches['venue_avg_total'].std():.1f})")
print(matches[["venue","venue_avg_total"]]
      .drop_duplicates("venue")
      .sort_values("venue_avg_total", ascending=False).head(8).to_string())


# ════════════════════════════════════════════════════════════════════════════
# 5.  STATE TABLE
# ════════════════════════════════════════════════════════════════════════════

WAVE2_COLS = ["batting_strength_remaining", "bowling_strength_remaining",
              "wickets_remaining_weighted"]

def build_state_table(balls: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    df = balls.copy()
    df["match_id"] = df["match_id"].astype(str)

    df["runs_so_far"] = (
        df.groupby(["match_id","innings"])["total_runs"]
          .transform(lambda x: x.shift(1, fill_value=0).cumsum())
    )
    df["wickets_lost"] = (
        df.groupby(["match_id","innings"])["wicket"]
          .transform(lambda x: x.shift(1, fill_value=0).cumsum())
    )
    df["legal_ball_before"] = (
        df.groupby(["match_id","innings"])["is_legal"]
          .transform(lambda x: x.shift(1, fill_value=0).cumsum())
          .clip(0, 120)
    )
    df["balls_remaining_before"] = (120 - df["legal_ball_before"]).clip(0, 120)

    df["runs_last_12"] = (
        df.groupby(["match_id","innings"])["total_runs"]
          .transform(lambda x: x.rolling(12, min_periods=1).sum().shift(1, fill_value=0))
    )
    df["wickets_last_12"] = (
        df.groupby(["match_id","innings"])["wicket"]
          .transform(lambda x: x.rolling(12, min_periods=1).sum().shift(1, fill_value=0))
    )
    df["phase"] = pd.cut(df["legal_ball_before"],
                         bins=[-1,36,90,120], labels=[0,1,2]).astype(int)
    df["current_rr"] = np.where(
        df["legal_ball_before"] > 0,
        df["runs_so_far"] / (df["legal_ball_before"] / 6), 0.0
    )

    inn1_totals = (
        df[df["innings"]==1].groupby("match_id")["total_runs"]
        .sum().rename("innings1_total")
    )
    df = df.merge(inn1_totals, on="match_id", how="left")

    df["runs_needed"] = np.where(
        df["innings"]==2, (df["innings1_total"]+1) - df["runs_so_far"], np.nan)
    df["required_rr"] = np.where(
        (df["innings"]==2) & (df["balls_remaining_before"]>0),
        df["runs_needed"] / (df["balls_remaining_before"]/6), np.nan)
    df["wickets_remaining"] = 10 - df["wickets_lost"]
    df["rr_diff"] = np.where(
        df["innings"]==2, df["current_rr"] - df["required_rr"], np.nan)

    winner_map = matches.set_index("match_id")["winner"].to_dict()
    df["match_winner"] = df["match_id"].map(winner_map)
    df["y"] = (df["batting_team"] == df["match_winner"]).astype(int)

    meta_cols = (
        ["match_id","date","is_day_match","dew_likely","season_phase","venue_avg_total"]
        + WEATHER_FEATURES
    )
    meta_cols = [c for c in meta_cols if c in matches.columns]
    df = df.merge(
        matches[meta_cols].drop_duplicates("match_id"),
        on="match_id", how="left"
    )
    return df

state = build_state_table(balls, matches)
state = state.dropna(subset=["y"])

print(f"\nState table    : {state.shape}")
print(f"Win rate       : {state['y'].mean():.3f}  (expected ~0.50)")
brb = state["balls_remaining_before"]
print(f"balls_remaining_before — min:{brb.min():.0f}  max:{brb.max():.0f}")

for col in WAVE2_COLS + ["partnership_runs", "partnership_balls"]:
    assert col in state.columns, f"Column '{col}' missing from state table!"
print(f"Wave 2 + partnership columns in state: ✓")


# ════════════════════════════════════════════════════════════════════════════
# 6.  ELO PRE-MATCH
# ════════════════════════════════════════════════════════════════════════════

def compute_elo(matches, K=20.0, init=1500.0):
    elo, rows = {}, []
    for _, row in matches.iterrows():
        t1, t2 = row["team1"], row["team2"]
        e1, e2 = elo.get(t1, init), elo.get(t2, init)
        p1 = 1 / (1 + 10**( -(e1-e2)/400 ))
        s1 = 1.0 if row["winner"] == t1 else 0.0
        elo[t1] = e1 + K*(s1-p1)
        elo[t2] = e2 + K*((1-s1)-(1-p1))
        rows.append({"match_id": str(row["match_id"]),
                     "elo_team1_before": e1,
                     "elo_team2_before": e2,
                     "p_pre_match": p1})
    return matches.merge(pd.DataFrame(rows), on="match_id", how="left")

matches = compute_elo(matches)


# ════════════════════════════════════════════════════════════════════════════
# 7.  POST-TOSS LOGISTIC MODEL
# ════════════════════════════════════════════════════════════════════════════

def logit(p):
    return np.log(np.clip(p,1e-6,1-1e-6) / (1-np.clip(p,1e-6,1-1e-6)))

matches["logit_pre"]         = logit(matches["p_pre_match"])
matches["toss_winner_is_t1"] = (matches["toss_winner"]==matches["team1"]).astype(int)
matches["bats_first_is_t1"]  = (
    ((matches["toss_winner"]==matches["team1"]) & (matches["toss_decision"]=="bat")) |
    ((matches["toss_winner"]==matches["team2"]) & (matches["toss_decision"]=="field"))
).astype(int)
matches["y_match"] = (matches["winner"]==matches["team1"]).astype(int)

m_train = matches.iloc[:val_start]
m_val   = matches.iloc[val_start:test_start]
m_test  = matches.iloc[test_start:]

toss_feats = ["logit_pre","toss_winner_is_t1","bats_first_is_t1"]
lr_toss    = LogisticRegression(max_iter=500)
lr_toss.fit(m_train[toss_feats], m_train["y_match"])

p_post_test_match = lr_toss.predict_proba(m_test[toss_feats])[:,1]
print(f"\nPost-toss model")
print(f"  Brier Elo-only  : {brier_score_loss(m_test['y_match'], m_test['p_pre_match']):.4f}")
print(f"  Brier post-toss : {brier_score_loss(m_test['y_match'], p_post_test_match):.4f}")

matches["p_post_toss"] = lr_toss.predict_proba(matches[toss_feats])[:,1]
state = state.merge(
    matches[["match_id","p_post_toss"]].drop_duplicates("match_id"),
    on="match_id", how="left"
)


# ════════════════════════════════════════════════════════════════════════════
# 8.  TIME SPLIT
# ════════════════════════════════════════════════════════════════════════════

dates_sorted = matches["date"].sort_values()
val_date     = dates_sorted.iloc[val_start]
test_date    = dates_sorted.iloc[test_start]
print(f"\nTime split — train:<{val_date.date()}  "
      f"val:{val_date.date()}–{test_date.date()}  "
      f"test:>{test_date.date()}")

inn1_all = state[state["innings"]==1].copy()
train_i1 = inn1_all[inn1_all["date"] <  val_date].copy()
val_i1   = inn1_all[(inn1_all["date"] >= val_date) & (inn1_all["date"] < test_date)].copy()


# ════════════════════════════════════════════════════════════════════════════
# 9.  MODEL T — WICKET-CONDITIONAL PROJECTED TOTAL  (Wave 3a)
# ════════════════════════════════════════════════════════════════════════════
# Three quantile models blended by wickets_lost:
#   Few wickets  → weight optimistic more (batting team likely to accelerate)
#   Many wickets → weight pessimistic more (tail exposed, lower finish)
# Partnership features added (Wave 3c).

INN1_FEATURES = [
    "runs_so_far", "balls_remaining_before", "wickets_lost",
    "runs_last_12", "wickets_last_12", "phase",
    "venue_avg_total", "is_day_match", "season_phase",
    # ── Wave 2 ──
    "batting_strength_remaining", "bowling_strength_remaining",
    "wickets_remaining_weighted",
    # ── Wave 3c ──
    "partnership_runs", "partnership_balls",
] + WEATHER_FEATURES

train_i1 = train_i1.dropna(subset=INN1_FEATURES + ["innings1_total"]).copy()
val_i1   = val_i1.dropna(subset=INN1_FEATURES + ["innings1_total"]).copy()

# ── Optuna tuning for the EXPECTED model (median) ────────────────────────────

def _objective_T(trial):
    params = dict(
        objective         = "regression",
        metric            = "mae",
        num_leaves        = trial.suggest_int("num_leaves", 64, 512),
        min_child_samples = trial.suggest_int("min_child_samples", 10, 100),
        learning_rate     = trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        subsample         = trial.suggest_float("subsample", 0.6, 1.0),
        colsample_bytree  = trial.suggest_float("colsample_bytree", 0.6, 1.0),
        n_estimators      = trial.suggest_int("n_estimators", 400, 1000),
        random_state      = 42,
        verbose           = -1,
    )
    m = lgb.LGBMRegressor(**params)
    m.fit(
        train_i1[INN1_FEATURES], train_i1["innings1_total"],
        eval_set=[(val_i1[INN1_FEATURES], val_i1["innings1_total"])],
        callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)],
    )
    return np.mean(np.abs(m.predict(val_i1[INN1_FEATURES]) - val_i1["innings1_total"]))

print("\nModel T — Optuna search (50 trials) ...")
study_T = optuna.create_study(direction="minimize",
                               sampler=optuna.samplers.TPESampler(seed=42))
study_T.optimize(_objective_T, n_trials=50, show_progress_bar=False)

best_T = study_T.best_params
print(f"  Best val MAE : {study_T.best_value:.3f}")
print(f"  Best params  : {best_T}")

params_T_best = dict(
    objective         = "regression",
    metric            = "mae",
    num_leaves        = best_T["num_leaves"],
    min_child_samples = best_T["min_child_samples"],
    learning_rate     = best_T["learning_rate"],
    subsample         = best_T["subsample"],
    colsample_bytree  = best_T["colsample_bytree"],
    n_estimators      = best_T["n_estimators"],
    random_state      = 42,
    verbose           = -1,
)

# ── Train three Model T variants ─────────────────────────────────────────────
model_T_exp = lgb.LGBMRegressor(**params_T_best)
model_T_exp.fit(
    train_i1[INN1_FEATURES], train_i1["innings1_total"],
    eval_set=[(val_i1[INN1_FEATURES], val_i1["innings1_total"])],
    callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)],
)

params_T_q = {**params_T_best, "n_estimators": 800, "learning_rate": 0.03,
              "min_child_samples": 10}

model_T_opt = lgb.LGBMRegressor(**{**params_T_q, "objective": "quantile", "alpha": 0.75})
model_T_pes = lgb.LGBMRegressor(**{**params_T_q, "objective": "quantile", "alpha": 0.25})
model_T_opt.fit(train_i1[INN1_FEATURES], train_i1["innings1_total"],
                callbacks=[lgb.log_evaluation(0)])
model_T_pes.fit(train_i1[INN1_FEATURES], train_i1["innings1_total"],
                callbacks=[lgb.log_evaluation(0)])

# ── Wicket-conditional blending ──────────────────────────────────────────────

def blend_weights(wickets_lost):
    """
    Few wickets → weight optimistic more (team likely to accelerate).
    Many wickets → weight pessimistic more (tail exposed).
    """
    w_opt = np.clip(0.5 - 0.08 * wickets_lost, 0.0, 0.5)
    w_pes = np.clip(0.1 + 0.07 * wickets_lost, 0.0, 0.5)
    w_exp = 1.0 - w_opt - w_pes
    return w_opt, w_exp, w_pes

def wicket_conditional_predict(df, features):
    """Predict blended proj_total for a DataFrame."""
    p_opt = model_T_opt.predict(df[features])
    p_exp = model_T_exp.predict(df[features])
    p_pes = model_T_pes.predict(df[features])
    wl    = df["wickets_lost"].values
    w_opt, w_exp, w_pes = blend_weights(wl)
    return w_opt * p_opt + w_exp * p_exp + w_pes * p_pes

# ── Evaluate on val ──────────────────────────────────────────────────────────
val_i1["proj_total_single"]  = model_T_exp.predict(val_i1[INN1_FEATURES])
val_i1["proj_total_blended"] = wicket_conditional_predict(val_i1, INN1_FEATURES)

mae_single  = np.mean(np.abs(val_i1["proj_total_single"]  - val_i1["innings1_total"]))
mae_blended = np.mean(np.abs(val_i1["proj_total_blended"] - val_i1["innings1_total"]))

print(f"\n── Wicket-conditional projection (val) ─────────────────────────────")
print(f"  Single model MAE  : {mae_single:.3f}")
print(f"  Blended model MAE : {mae_blended:.3f}  (Δ = {mae_blended - mae_single:+.3f})")

# Use whichever is better
USE_BLENDED = mae_blended <= mae_single
proj_col = "proj_total_blended" if USE_BLENDED else "proj_total_single"
print(f"  Using: {'blended (wicket-conditional)' if USE_BLENDED else 'single (expected)'}")

val_i1["proj_total"] = val_i1[proj_col]
val_i1["over_number"] = (val_i1["legal_ball_before"]//6).clip(0,19)
mae_by_over = val_i1.groupby("over_number").apply(
    lambda g: np.mean(np.abs(g["proj_total"] - g["innings1_total"])))
print("\nMAE by over (val):")
print(mae_by_over.to_string())

# ── Model T feature importances ──────────────────────────────────────────────
print("\n── Model T feature importances (expected model) ────────────────────")
fi_T = pd.Series(model_T_exp.feature_importances_, index=INN1_FEATURES).sort_values(ascending=False)
for feat, imp in fi_T.items():
    bar = "█" * int(imp/fi_T.max()*30)
    print(f"  {feat:35s} {bar}")

# ── Quantile bands for uncertainty plots (alpha 0.05/0.95) ───────────────────
model_T_p10 = lgb.LGBMRegressor(**{**params_T_q, "objective": "quantile", "alpha": 0.05})
model_T_p90 = lgb.LGBMRegressor(**{**params_T_q, "objective": "quantile", "alpha": 0.95})
model_T_p10.fit(train_i1[INN1_FEATURES], train_i1["innings1_total"],
                callbacks=[lgb.log_evaluation(0)])
model_T_p90.fit(train_i1[INN1_FEATURES], train_i1["innings1_total"],
                callbacks=[lgb.log_evaluation(0)])

raw_p10 = np.minimum(model_T_p10.predict(val_i1[INN1_FEATURES]), val_i1["proj_total"].values)
raw_p90 = np.maximum(model_T_p90.predict(val_i1[INN1_FEATURES]), val_i1["proj_total"].values)
true_vals = val_i1["innings1_total"].values

def coverage_at_margin(m, true, p10, p90):
    return ((true >= p10-m) & (true <= p90+m)).mean()

margin = 0.0
for candidate in np.arange(0, 50, 0.5):
    if coverage_at_margin(candidate, true_vals, raw_p10, raw_p90) >= 0.75:
        margin = candidate
        break

print(f"\nQuantile band — additive margin: ±{margin:.1f} runs  "
      f"coverage={coverage_at_margin(margin, true_vals, raw_p10, raw_p90):.3f}")


# ════════════════════════════════════════════════════════════════════════════
# 10.  ATTACH proj_total + elo_diff → FINAL SPLIT
# ════════════════════════════════════════════════════════════════════════════

inn1_mask = state["innings"]==1
state["proj_total"] = np.nan

if USE_BLENDED:
    state.loc[inn1_mask, "proj_total"] = wicket_conditional_predict(
        state.loc[inn1_mask], INN1_FEATURES)
else:
    state.loc[inn1_mask, "proj_total"] = model_T_exp.predict(
        state.loc[inn1_mask, INN1_FEATURES].fillna(0))

state.loc[~inn1_mask, "proj_total"] = state.loc[~inn1_mask, "innings1_total"]

elo_lookup = matches[["match_id","team1","elo_team1_before","elo_team2_before"]].drop_duplicates("match_id")
for col in ["team1","elo_team1_before","elo_team2_before"]:
    if col in state.columns: state = state.drop(columns=[col])
state = state.merge(elo_lookup, on="match_id", how="left")
state["elo_diff"] = np.where(
    state["batting_team"]==state["team1"],
    state["elo_team1_before"] - state["elo_team2_before"],
    state["elo_team2_before"] - state["elo_team1_before"]
)

train_state = state[state["date"] <  val_date].copy()
val_state   = state[(state["date"] >= val_date) & (state["date"] < test_date)].copy()
test_state  = state[state["date"] >= test_date].copy()

for name, df in [("train",train_state),("val",val_state),("test",test_state)]:
    missing = [c for c in ["proj_total","elo_diff"] + WAVE2_COLS if c not in df.columns]
    if missing: raise ValueError(f"Split '{name}' missing: {missing}")
print(f"\nFinal split — train:{len(train_state):,}  val:{len(val_state):,}  test:{len(test_state):,}")


# ════════════════════════════════════════════════════════════════════════════
# 11.  LR BASELINE  (innings 2)
# ════════════════════════════════════════════════════════════════════════════

INN2_FEATS_LR = ["runs_needed","balls_remaining_before","wickets_remaining",
                 "required_rr","runs_last_12","p_post_toss"]

train_i2 = train_state[train_state["innings"]==2].dropna(subset=INN2_FEATS_LR+["y"]).copy()
val_i2   = val_state[val_state["innings"]==2].dropna(subset=INN2_FEATS_LR+["y"]).copy()
test_i2  = test_state[test_state["innings"]==2].dropna(subset=INN2_FEATS_LR+["y"]).copy()

scaler   = StandardScaler()
lr_base  = LogisticRegression(max_iter=500)
lr_base.fit(scaler.fit_transform(train_i2[INN2_FEATS_LR]), train_i2["y"])
p_lr_test = lr_base.predict_proba(scaler.transform(test_i2[INN2_FEATS_LR]))[:,1]

print(f"\nLR baseline — Brier:{brier_score_loss(test_i2['y'], p_lr_test):.4f}  "
      f"LogLoss:{log_loss(test_i2['y'], p_lr_test):.4f}")


# ════════════════════════════════════════════════════════════════════════════
# 12.  MODEL W — LightGBM WIN PROBABILITY
# ════════════════════════════════════════════════════════════════════════════

WIN_FEATURES = [
    "p_post_toss", "elo_diff", "innings",
    "runs_so_far", "balls_remaining_before",
    "wickets_lost", "wickets_remaining",
    "wickets_last_12",
    "current_rr", "proj_total",
    "runs_needed", "required_rr", "rr_diff",
    "season_phase", "venue_avg_total",
    # ── Wave 2 ──
    "batting_strength_remaining", "bowling_strength_remaining",
    "wickets_remaining_weighted",
] + WEATHER_FEATURES

def prep(df):
    return df.dropna(subset=["y","p_post_toss","proj_total","elo_diff"]).copy()

tr = prep(train_state)
vl = prep(val_state)
te = prep(test_state)
print(f"\nModel W sizes — tr:{len(tr):,}  vl:{len(vl):,}  te:{len(te):,}")
print(f"WIN_FEATURES ({len(WIN_FEATURES)}): {WIN_FEATURES}")

# ── Model W — Optuna tuning (60 trials) ──────────────────────────────────────

def _objective_W(trial):
    params = dict(
        objective         = "binary",
        metric            = "binary_logloss",
        num_leaves        = trial.suggest_int("num_leaves", 32, 512),
        min_child_samples = trial.suggest_int("min_child_samples", 10, 200),
        learning_rate     = trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
        subsample         = trial.suggest_float("subsample", 0.6, 1.0),
        colsample_bytree  = trial.suggest_float("colsample_bytree", 0.5, 1.0),
        reg_alpha         = trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        reg_lambda        = trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        n_estimators      = 1000,
        random_state      = 42,
        verbose           = -1,
    )
    m = lgb.LGBMClassifier(**params)
    m.fit(
        tr[WIN_FEATURES], tr["y"],
        eval_set=[(vl[WIN_FEATURES], vl["y"])],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )
    return brier_score_loss(vl["y"], m.predict_proba(vl[WIN_FEATURES])[:, 1])

print("\nModel W — Optuna search (60 trials) ...")
study_W = optuna.create_study(direction="minimize",
                               sampler=optuna.samplers.TPESampler(seed=42))
study_W.optimize(_objective_W, n_trials=60, show_progress_bar=False)

best_W = study_W.best_params
print(f"  Best val Brier : {study_W.best_value:.4f}")
print(f"  Best params    : {best_W}")

print("  Top 5 trials:")
trials_df = study_W.trials_dataframe().sort_values("value").head(5)
for _, row in trials_df.iterrows():
    print(f"    trial #{int(row['number']):3d}  Brier={row['value']:.4f}  "
          f"leaves={int(row['params_num_leaves'])}  "
          f"lr={row['params_learning_rate']:.4f}  "
          f"min_child={int(row['params_min_child_samples'])}")

model_W = lgb.LGBMClassifier(
    objective         = "binary",
    metric            = "binary_logloss",
    n_estimators      = 1000,
    random_state      = 42,
    verbose           = -1,
    **{k: v for k, v in best_W.items()},
)
model_W.fit(
    tr[WIN_FEATURES], tr["y"],
    eval_set=[(vl[WIN_FEATURES], vl["y"])],
    callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
)


# ════════════════════════════════════════════════════════════════════════════
# 12.5  TEMPERATURE SCALING  (Wave 3b — replaces Platt calibration)
# ════════════════════════════════════════════════════════════════════════════
# Single-parameter calibration: divide logits by learned temperature T.
# T > 1 → model was overconfident → soften predictions toward 0.5
# T < 1 → model was underconfident → sharpen predictions toward 0/1

print("\n── Temperature scaling calibration ─────────────────────────────────")

probs_val_raw = model_W.predict_proba(vl[WIN_FEATURES])[:, 1]
logits_val    = np.log(np.clip(probs_val_raw, 1e-7, 1-1e-7)
                       / (1 - np.clip(probs_val_raw, 1e-7, 1-1e-7)))

def nll_at_T(T):
    p_cal = 1 / (1 + np.exp(-logits_val / T))
    return log_loss(vl["y"], p_cal)

result = minimize_scalar(nll_at_T, bounds=(0.1, 10.0), method="bounded")
T_opt  = result.x

print(f"  Optimal temperature T = {T_opt:.4f}")
if T_opt > 1.0:
    print(f"  Interpretation: model was overconfident → softening predictions")
elif T_opt < 1.0:
    print(f"  Interpretation: model was underconfident → sharpening predictions")
else:
    print(f"  Interpretation: model was already perfectly calibrated")

# Brier comparison on val: raw vs Platt vs temperature
brier_val_raw = brier_score_loss(vl["y"], probs_val_raw)
p_val_temp    = 1 / (1 + np.exp(-logits_val / T_opt))
brier_val_temp = brier_score_loss(vl["y"], p_val_temp)

# Also fit Platt for comparison
model_W_platt = CalibratedClassifierCV(model_W, method="sigmoid", cv="prefit")
model_W_platt.fit(vl[WIN_FEATURES], vl["y"])
p_val_platt   = model_W_platt.predict_proba(vl[WIN_FEATURES])[:, 1]
brier_val_platt = brier_score_loss(vl["y"], p_val_platt)

print(f"\n  Val Brier — raw:         {brier_val_raw:.4f}")
print(f"  Val Brier — Platt:       {brier_val_platt:.4f}")
print(f"  Val Brier — temperature: {brier_val_temp:.4f}")

# Pick the best calibration method
cal_methods = {"raw": brier_val_raw, "platt": brier_val_platt, "temperature": brier_val_temp}
best_cal = min(cal_methods, key=cal_methods.get)
print(f"  Best calibration: {best_cal} (Brier={cal_methods[best_cal]:.4f})")

def apply_temperature(model, X, T):
    """Apply temperature scaling to model predictions."""
    probs = model.predict_proba(X)[:, 1]
    logits = np.log(np.clip(probs, 1e-7, 1-1e-7) / (1 - np.clip(probs, 1e-7, 1-1e-7)))
    return 1 / (1 + np.exp(-logits / T))


# ════════════════════════════════════════════════════════════════════════════
# 13.  EVALUATION REPORT  —  TEST SET
# ════════════════════════════════════════════════════════════════════════════

p_raw_test   = model_W.predict_proba(te[WIN_FEATURES])[:, 1]
p_platt_test = model_W_platt.predict_proba(te[WIN_FEATURES])[:, 1]
p_temp_test  = apply_temperature(model_W, te[WIN_FEATURES], T_opt)

# Use best calibration
if best_cal == "temperature":
    p_cal_test = p_temp_test
elif best_cal == "platt":
    p_cal_test = p_platt_test
else:
    p_cal_test = p_raw_test

te["p_cal"] = p_cal_test

brier_overall = brier_score_loss(te["y"], p_cal_test)
logloss_cal   = log_loss(te["y"], p_cal_test)
te_i2         = te[te["innings"]==2].copy()

print("\n" + "="*70)
print("EVALUATION REPORT  —  TEST SET  (v7 / Wave 3)")
print("="*70)

print("\n── Overall ──────────────────────────────────────────────────────────")
print(f"  Always-predict-50%      Brier : 0.2500  ← baseline to beat")
print(f"  LR baseline (inn.2)     Brier : {brier_score_loss(test_i2['y'], p_lr_test):.4f}")
print(f"  LGB uncalibrated        Brier : {brier_score_loss(te['y'], p_raw_test):.4f}")
print(f"  LGB Platt-calibrated    Brier : {brier_score_loss(te['y'], p_platt_test):.4f}")
print(f"  LGB temp-scaled (T={T_opt:.3f}) Brier : {brier_score_loss(te['y'], p_temp_test):.4f}")
print(f"  BEST ({best_cal:11s})     Brier : {brier_overall:.4f}")
print(f"  BEST                  LogLoss : {logloss_cal:.4f}")
print(f"  Improvement over 0.25 baseline: {(0.25-brier_overall)/0.25*100:.1f}%")
print(f"  v6 (Wave 2) Brier was         : 0.1744")
print(f"  Wave 3 ΔBrier                 : {0.1744 - brier_overall:+.4f}")

print("\n── By innings ───────────────────────────────────────────────────────")
for inn in [1,2]:
    sub = te[te["innings"]==inn]
    print(f"  Innings {inn}: Brier={brier_score_loss(sub['y'], sub['p_cal']):.4f}  (n={len(sub):,})")

print("\n── By phase ─────────────────────────────────────────────────────────")
for ph, name in {0:"Powerplay (ov 1-6) ",1:"Middle   (ov 7-15) ",2:"Death    (ov 16-20)"}.items():
    sub = te[te["phase"]==ph]
    if not len(sub): continue
    print(f"  {name}: Brier={brier_score_loss(sub['y'], sub['p_cal']):.4f}  (n={len(sub):,})")

print("\n── By Required Run Rate bucket (innings 2) ──────────────────────────")
te_i2["rrr_bucket"] = pd.cut(
    te_i2["required_rr"].fillna(0), bins=[0,6,8,10,50],
    labels=["Easy (<6)","Moderate (6-8)","Hard (8-10)","Very Hard (>10)"])
for bucket, grp in te_i2.groupby("rrr_bucket", observed=True):
    print(f"  {str(bucket):22s}: Brier={brier_score_loss(grp['y'], grp['p_cal']):.4f}  (n={len(grp):,})")

print("\n── Day vs Night ─────────────────────────────────────────────────────")
for flag, label in [(0,"Night"),(1,"Day  ")]:
    sub = te[te["is_day_match"]==flag]
    if not len(sub): continue
    print(f"  {label}: Brier={brier_score_loss(sub['y'], sub['p_cal']):.4f}  (n={len(sub):,})")

print("\n── Dew factor (innings 2) ───────────────────────────────────────────")
for flag, label in [(0,"Dew unlikely"),(1,"Dew likely  ")]:
    sub = te_i2[te_i2["dew_likely"]==flag]
    if not len(sub): continue
    print(f"  {label}: Brier={brier_score_loss(sub['y'], sub['p_cal']):.4f}  (n={len(sub):,})")

if WEATHER_FEATURES:
    print("\n── Weather feature breakdown ────────────────────────────────────────")
    for feat in WEATHER_FEATURES:
        if feat not in te.columns: continue
        q33, q67 = te[feat].quantile([0.33, 0.67])
        low  = te[te[feat] <= q33]
        high = te[te[feat] >= q67]
        b_low  = brier_score_loss(low["y"],  low["p_cal"])
        b_high = brier_score_loss(high["y"], high["p_cal"])
        print(f"  {feat:25s}  low tercile={b_low:.4f}  high tercile={b_high:.4f}")

print("\n── Feature importances (all features) ───────────────────────────────")
fi = pd.Series(model_W.feature_importances_, index=WIN_FEATURES).sort_values(ascending=False)
for feat, imp in fi.items():
    bar = "█" * int(imp/fi.max()*30)
    print(f"  {feat:35s} {bar}")

print("\n── Benchmark context ────────────────────────────────────────────────")
inn1_b = brier_score_loss(te[te["innings"]==1]["y"], te[te["innings"]==1]["p_cal"])
inn2_b = brier_score_loss(te[te["innings"]==2]["y"], te[te["innings"]==2]["p_cal"])
print(f"""
  ┌─────────────────────────────────────────────┬───────────┬────────────┐
  │ Model / Paper                               │   Brier   │  Setting   │
  ├─────────────────────────────────────────────┼───────────┼────────────┤
  │ Always predict 50% (dumb baseline)          │   0.2500  │  —         │
  │ Logistic regression (Bailey & Clarke 2006)  │ ~0.210    │  ODI/T20   │
  │ Random Forest (Sankaranarayanan et al 2014) │ ~0.195    │  T20       │
  │ YOUR model v5 (pre-Wave 2)                  │   0.1942  │  IPL T20   │
  │ YOUR model v6 (Wave 2)                      │   0.1744  │  IPL T20   │
  │ YOUR model v7 (Wave 3)                      │  {brier_overall:.4f}  │  IPL T20   │
  │ State-of-the-art (deep learning, all data)  │ ~0.165    │  T20 intl  │
  └─────────────────────────────────────────────┴───────────┴────────────┘
  Innings-1 Brier: {inn1_b:.4f}  (v6 was 0.2321, v5 was 0.2341)
  Innings-2 Brier: {inn2_b:.4f}  (v6 was 0.1116, v5 was 0.1508)
  Temperature T = {T_opt:.4f}  Calibration method: {best_cal}
  Irreducible T20 floor ≈ 0.15   Gap to close: {brier_overall-0.15:.3f}
""")
print("="*70)


# ════════════════════════════════════════════════════════════════════════════
# 14.  RELIABILITY DIAGRAMS  →  output/plots/
# ════════════════════════════════════════════════════════════════════════════

def plot_reliability(y_true, p_list, labels, title, filename):
    path = str(OUT_PLOTS / filename)
    fig, axes = plt.subplots(1, 2, figsize=(14,5))
    colors = ["#E05A3A","#0F7173","#D4821A","#6A4C93"]
    ax = axes[0]
    ax.plot([0,1],[0,1],"k--",lw=1.2,label="Perfect")
    for (p,lbl),col in zip(zip(p_list,labels),colors):
        pt, pp = calibration_curve(y_true, p, n_bins=10)
        ax.plot(pp, pt, "o-", color=col, lw=2, markersize=6, label=lbl)
    ax.set_xlabel("Mean predicted probability",fontsize=11)
    ax.set_ylabel("Observed win rate",fontsize=11)
    ax.set_title("Calibration curve",fontsize=11,fontweight="bold")
    ax.legend(fontsize=9); ax.set_xlim(0,1); ax.set_ylim(0,1); ax.grid(alpha=0.3)
    ax2 = axes[1]
    for (p,lbl),col in zip(zip(p_list,labels),colors):
        ax2.hist(p, bins=20, alpha=0.4, color=col, label=lbl, density=True)
    ax2.set_xlabel("Predicted probability",fontsize=11)
    ax2.set_ylabel("Density",fontsize=11)
    ax2.set_title("Distribution of predictions",fontsize=11,fontweight="bold")
    ax2.legend(fontsize=9); ax2.grid(alpha=0.3)
    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(path, dpi=150); plt.close()
    print(f"Saved: {path}")

p_raw_i2   = model_W.predict_proba(test_i2[WIN_FEATURES])[:,1]
p_temp_i2  = apply_temperature(model_W, test_i2[WIN_FEATURES], T_opt)
p_platt_i2 = model_W_platt.predict_proba(test_i2[WIN_FEATURES])[:,1]
assert len(test_i2)==len(p_lr_test)==len(p_raw_i2)

plot_reliability(test_i2["y"], [p_lr_test, p_raw_i2, p_platt_i2, p_temp_i2],
                 ["LR baseline","LGB raw","LGB Platt","LGB temp"],
                 "Reliability — Innings 2 (test season, v7)",
                 "reliability_innings2.png")
plot_reliability(te["y"], [p_raw_test, p_platt_test, p_temp_test],
                 ["LGB raw","LGB Platt","LGB temp"],
                 "Reliability — Both Innings (test season, v7)",
                 "reliability_both_innings.png")


# ════════════════════════════════════════════════════════════════════════════
# 15.  MATCH WIN PROBABILITY CURVE  →  output/plots/
# ════════════════════════════════════════════════════════════════════════════

def plot_match_curve(match_id, state, model, win_features, matches, T_opt, best_cal):
    mdf = state[state["match_id"]==match_id].copy()
    mdf = mdf.dropna(subset=["p_post_toss"])
    if mdf.empty: return
    if best_cal == "temperature":
        mdf["wp"] = apply_temperature(model, mdf[win_features].fillna(0), T_opt)
    else:
        mdf["wp"] = model.predict_proba(mdf[win_features].fillna(0))[:,1]
    row      = matches[matches["match_id"]==match_id].iloc[0]
    team_bat = mdf.iloc[0]["batting_team"]
    p_pre    = row["p_pre_match"] if team_bat==row["team1"] else 1-row["p_pre_match"]
    p_post   = row["p_post_toss"] if team_bat==row["team1"] else 1-row["p_post_toss"]

    inn1 = mdf[mdf["innings"]==1].copy(); inn1["bx"] = inn1["legal_ball_before"]+1
    inn2 = mdf[mdf["innings"]==2].copy(); inn2["bx"] = inn2["legal_ball_before"]+121

    fig, ax = plt.subplots(figsize=(13,5))
    ax.axhline(p_pre,  color="#aaa", ls=":",  lw=1.5, label=f"Pre-match {p_pre:.2f}")
    ax.axhline(p_post, color="#D4821A", ls="--", lw=1.5, label=f"Post-toss {p_post:.2f}")
    ax.axhline(0.5,    color="#ddd", ls="-",  lw=0.8)
    ax.plot(inn1["bx"], inn1["wp"], color="#1B3A5C", lw=2, label="Innings 1")
    ax.plot(inn2["bx"], inn2["wp"], color="#0F7173", lw=2, label="Innings 2")
    ax.axvline(120, color="#999", lw=0.8)
    ax.fill_between(inn1["bx"], inn1["wp"], 0.5, alpha=0.08, color="#1B3A5C")
    ax.fill_between(inn2["bx"], inn2["wp"], 0.5, alpha=0.08, color="#0F7173")
    for inn_df, color in [(inn1,"#1B3A5C"),(inn2,"#0F7173")]:
        if len(inn_df)<2: continue
        for idx in inn_df["wp"].diff().abs().nlargest(2).index:
            bx  = inn_df.loc[idx,"bx"]; wp = inn_df.loc[idx,"wp"]
            wkt = int(inn_df.loc[idx,"wickets_lost"])
            ax.annotate(f"wkt={wkt}", xy=(bx,wp), xytext=(bx+3,wp+0.07),
                        fontsize=7, color=color,
                        arrowprops=dict(arrowstyle="->",color=color,lw=0.8))
    weather_str = ""
    if WEATHER_FEATURES and "temp_max_c" in row.index:
        t = row.get("temp_max_c","?"); h = row.get("humidity_eve_pct","?")
        weather_str = f"  [{t:.0f}°C, {h:.0f}% humidity]" if t!="?" else ""
    dew_str  = "Dew likely" if row.get("dew_likely",0) else "No dew"
    time_str = "Day" if row.get("is_day_match",0) else "Night"
    ax.set_xlim(0,240); ax.set_ylim(0,1)
    ax.set_xlabel("Delivery (1–120: Inn 1 | 121–240: Inn 2)", fontsize=10)
    ax.set_ylabel(f"P(win) — {team_bat}", fontsize=10)
    ax.set_title(f"{row['team1']} vs {row['team2']}  |  Winner: {row['winner']}  "
                 f"[{time_str}, {dew_str}{weather_str}]",
                 fontsize=11, fontweight="bold")
    ax.set_xticks([1,36,72,108,120,156,192,228,240])
    ax.set_xticklabels(["Ov1","6","12","18","20|1","6","12","18","20"],fontsize=8)
    ax.legend(fontsize=9); ax.grid(alpha=0.2)
    plt.tight_layout()
    fpath = str(OUT_PLOTS / f"match_{match_id}.png")
    plt.savefig(fpath, dpi=150); plt.close()
    print(f"Saved: {fpath}")


# ════════════════════════════════════════════════════════════════════════════
# 16.  PROJECTED TOTAL CURVE  →  output/plots/
# ════════════════════════════════════════════════════════════════════════════

def plot_projected_total(match_id, state, model_med, model_p10, model_p90,
                         features, band_margin):
    mdf = state[(state["match_id"]==match_id) & (state["innings"]==1)].dropna(subset=features)
    if mdf.empty: return
    mdf = mdf.copy()
    med = model_med.predict(mdf[features])
    p10 = np.minimum(model_p10.predict(mdf[features]), med) - band_margin
    p90 = np.maximum(model_p90.predict(mdf[features]), med) + band_margin
    mdf["bx"] = mdf["legal_ball_before"]+1
    true_total = mdf["innings1_total"].iloc[0]
    venue_avg  = mdf["venue_avg_total"].iloc[0]
    fig, ax = plt.subplots(figsize=(9,4))
    ax.fill_between(mdf["bx"], p10, p90, alpha=0.2, color="#0F7173",
                    label=f"Uncertainty band (±{band_margin:.0f})")
    ax.plot(mdf["bx"], med, color="#0F7173", lw=2, label="Projected total")
    ax.axhline(true_total, color="#E05A3A", ls="--", lw=1.5,
               label=f"Actual: {int(true_total)}")
    ax.axhline(venue_avg, color="#D4821A", ls=":", lw=1.2,
               label=f"Venue avg: {venue_avg:.0f}")
    ax.set_xlabel("Delivery number",fontsize=11)
    ax.set_ylabel("Runs",fontsize=11)
    ax.set_title(f"Projected Innings-1 Total — Match {match_id}",
                 fontsize=12,fontweight="bold")
    ax.legend(fontsize=9); ax.grid(alpha=0.25)
    plt.tight_layout()
    fpath = str(OUT_PLOTS / f"proj_{match_id}.png")
    plt.savefig(fpath, dpi=150); plt.close()
    print(f"Saved: {fpath}")


test_match_ids = te["match_id"].unique()[:3]
for mid in test_match_ids:
    plot_match_curve(mid, te, model_W, WIN_FEATURES, matches, T_opt, best_cal)
    plot_projected_total(mid, te, model_T_exp, model_T_p10, model_T_p90,
                         INN1_FEATURES, margin)


# ════════════════════════════════════════════════════════════════════════════
# 17.  SAVE MODELS  →  output/models/
# ════════════════════════════════════════════════════════════════════════════

joblib.dump(model_T_exp,  str(OUT_MODELS/"model_T_expected.pkl"))
joblib.dump(model_T_opt,  str(OUT_MODELS/"model_T_optimistic.pkl"))
joblib.dump(model_T_pes,  str(OUT_MODELS/"model_T_pessimistic.pkl"))
joblib.dump(model_T_p10,  str(OUT_MODELS/"model_T_p10.pkl"))
joblib.dump(model_T_p90,  str(OUT_MODELS/"model_T_p90.pkl"))
joblib.dump(model_W,      str(OUT_MODELS/"model_win_prob_raw.pkl"))
joblib.dump(model_W_platt,str(OUT_MODELS/"model_win_prob_platt.pkl"))
joblib.dump(lr_toss,      str(OUT_MODELS/"model_toss.pkl"))
joblib.dump(scaler,       str(OUT_MODELS/"scaler_lr_baseline.pkl"))
joblib.dump({"band_margin": margin,
             "win_features": WIN_FEATURES,
             "inn1_features": INN1_FEATURES,
             "weather_features": WEATHER_FEATURES,
             "wave2_cols": WAVE2_COLS,
             "T_optimal": T_opt,
             "best_calibration": best_cal,
             "use_blended_T": USE_BLENDED},
            str(OUT_MODELS/"run_config.pkl"))

print(f"\nAll models saved to {OUT_MODELS}/")
print(f"All plots  saved to {OUT_PLOTS}/")
print(f"Run log    saved to {_log_path}")
print("Pipeline complete.")

# Close log file
sys.stdout = sys.__stdout__
_log_file.close()