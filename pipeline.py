"""
T20 Cricket Win Probability Pipeline  —  v10 (IPL + T20I Combined)
===================================================================
Changes vs v9
-------------
  DATA EXPANSION — Train on both IPL and filtered T20I data.

  Two datasets are loaded and concatenated:
    IPL   — existing ipl_matches.csv + ipl_balls_wave4b.csv (or wave4)
    T20I  — t20i_matches_filtered.csv + t20i_balls_wave4b.csv
            (men's only; filtered to 18-team list + top-9 vs anyone rule)

  Each dataset is tagged with a `source` column ("ipl" / "t20i").

  ELO — computed separately per format (IPL teams vs T20I teams are
        different competitive contexts). Both elo columns are merged
        back before the state table is built.

  WAVE FEATURES — rolling player stats (batsman SR, bowler economy etc.)
        are computed from the combined IPL + T20I ball history so that
        players who appear in both formats share a single 2yr window.
        Run wave4_t20i.py once to produce t20i_balls_wave4b.csv before
        running this pipeline.

  Everything else (Model T, Model W, calibration, evaluation) is
  unchanged — they operate on the larger combined dataset.
"""

import os, sys
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
# 1.  LOAD DATA  (IPL + T20I)
# ════════════════════════════════════════════════════════════════════════════

# ── IPL ───────────────────────────────────────────────────────────────────────
IPL_WAVE4B = Path("data/ipl_balls_wave4b.csv")
IPL_WAVE4  = Path("data/ipl_balls_wave4.csv")

ipl_matches = pd.read_csv("data/ipl_matches.csv", parse_dates=["date"])
ipl_matches["source"] = "ipl"

if IPL_WAVE4B.exists():
    print(f"\nIPL balls: loading from {IPL_WAVE4B}")
    ipl_balls = pd.read_csv(IPL_WAVE4B)
elif IPL_WAVE4.exists():
    print(f"\nIPL balls: loading from {IPL_WAVE4}")
    ipl_balls = pd.read_csv(IPL_WAVE4)
else:
    raise FileNotFoundError("Need data/ipl_balls_wave4b.csv or data/ipl_balls_wave4.csv")

ipl_balls["source"] = "ipl"
print(f"IPL  — matches: {len(ipl_matches):,}  balls: {len(ipl_balls):,}")

# ── T20I ──────────────────────────────────────────────────────────────────────
T20I_MATCHES = Path("data/t20i_matches_filtered.csv")
T20I_WAVE4B  = Path("data/t20i_balls_wave4b.csv")
T20I_BALLS   = Path("data/t20i_balls_filtered.csv")

if T20I_MATCHES.exists():
    t20i_matches = pd.read_csv(T20I_MATCHES, parse_dates=["date"])
    # Men's only
    if "gender" in t20i_matches.columns:
        t20i_matches = t20i_matches[t20i_matches["gender"] == "male"].copy()
    t20i_matches["source"] = "t20i"

    if T20I_WAVE4B.exists():
        print(f"T20I balls: loading from {T20I_WAVE4B}")
        t20i_balls = pd.read_csv(T20I_WAVE4B)
    elif T20I_BALLS.exists():
        # wave4b not yet computed — run wave4_t20i.py automatically
        print(f"\nT20I wave features not found — running wave4_t20i.py now...")
        print("(This only happens once; output is cached to data/t20i_balls_wave4b.csv)\n")
        import importlib.util, subprocess
        _wave4_script = Path(__file__).parent / "wave4_t20i.py"
        if not _wave4_script.exists():
            raise FileNotFoundError(
                "wave4_t20i.py not found alongside pipeline.py. "
                "Place wave4_t20i.py in the same directory and re-run."
            )
        _result = subprocess.run(
            [sys.executable, str(_wave4_script)],
            check=True
        )
        if not T20I_WAVE4B.exists():
            raise RuntimeError(
                "wave4_t20i.py ran but did not produce data/t20i_balls_wave4b.csv. "
                "Check wave4_t20i.py output above for errors."
            )
        print(f"\nT20I wave features computed. Loading from {T20I_WAVE4B}")
        t20i_balls = pd.read_csv(T20I_WAVE4B)
    else:
        raise FileNotFoundError(
            "Need data/t20i_balls_filtered.csv — run filter_t20i.py first."
        )

    t20i_balls["source"] = "t20i"
    # Keep only men's match IDs
    valid_t20i_ids = set(t20i_matches["match_id"].astype(str))
    t20i_balls["match_id"] = t20i_balls["match_id"].astype(str)
    t20i_balls = t20i_balls[t20i_balls["match_id"].isin(valid_t20i_ids)]
    print(f"T20I — matches: {len(t20i_matches):,}  balls: {len(t20i_balls):,}")
else:
    print("WARNING: T20I data not found — running on IPL only")
    t20i_matches = pd.DataFrame(columns=ipl_matches.columns)
    t20i_balls   = pd.DataFrame(columns=ipl_balls.columns)

# ── Combine ───────────────────────────────────────────────────────────────────
matches = pd.concat([ipl_matches, t20i_matches], ignore_index=True)
matches = matches.sort_values("date").reset_index(drop=True)
balls   = pd.concat([ipl_balls, t20i_balls],   ignore_index=True)

print(f"\nCombined — matches: {len(matches):,}  balls: {len(balls):,}")
print(f"  IPL  : {(matches['source']=='ipl').sum():,} matches")
print(f"  T20I : {(matches['source']=='t20i').sum():,} matches")


# ════════════════════════════════════════════════════════════════════════════
# 2.  WEATHER
# ════════════════════════════════════════════════════════════════════════════

WEATHER_CSV = Path("data/weather.csv")
if WEATHER_CSV.exists():
    weather_df = pd.read_csv(WEATHER_CSV)
else:
    sys.path.insert(0, str(Path(__file__).parent))
    from fetch_weather import build_weather_table
    weather_df = build_weather_table(matches_csv="data/ipl_matches.csv",
                                     output_csv=str(WEATHER_CSV))

WEATHER_FEATURES = ["temp_max_c", "humidity_eve_pct", "cloud_cover_eve_pct",
                    "dew_point_eve_c", "precipitation_mm"]
weather_cols = ["match_id"] + [c for c in WEATHER_FEATURES if c in weather_df.columns]
weather_df = weather_df[weather_cols].copy()
weather_df["match_id"] = weather_df["match_id"].astype(str)
matches["match_id"] = matches["match_id"].astype(str)

matches = matches.merge(weather_df, on="match_id", how="left")
for col in WEATHER_FEATURES:
    if col in matches.columns:
        matches[col] = matches[col].fillna(matches[col].median())
    else:
        matches[col] = np.nan
WEATHER_FEATURES = [c for c in WEATHER_FEATURES
                    if c in matches.columns and matches[c].std() > 0]
print(f"Weather features: {WEATHER_FEATURES}")


# ════════════════════════════════════════════════════════════════════════════
# 3.  LEGAL BALLS + PARTNERSHIP
# ════════════════════════════════════════════════════════════════════════════

def add_legality_flag(df):
    df = df.copy()
    if "wide_runs" not in df.columns or "noball_runs" not in df.columns:
        df["wide_runs"] = 0; df["noball_runs"] = 0
    df["is_legal"] = ((df["wide_runs"] == 0) & (df["noball_runs"] == 0)).astype(int)
    return df

balls = add_legality_flag(balls)
balls = balls.sort_values(["match_id", "innings", "over", "ball"]).reset_index(drop=True)
balls["legal_ball_number"] = balls.groupby(["match_id", "innings"])["is_legal"].cumsum().clip(0, 120)
balls["balls_remaining"] = (120 - balls["legal_ball_number"]).clip(0, 120)

if "partnership_runs" not in balls.columns:
    print("\nComputing partnership features...")
    balls["partnership_runs"] = 0; balls["partnership_balls"] = 0
    for (mid, inn), grp in balls.groupby(["match_id", "innings"], sort=False):
        gs = grp.sort_values(["over", "ball"]); ia = gs.index.values
        pr = pb = 0
        for i, (_, r) in enumerate(gs.iterrows()):
            balls.at[ia[i], "partnership_runs"] = pr
            balls.at[ia[i], "partnership_balls"] = pb
            pr += r["total_runs"]; pb += int(r["is_legal"])
            if r["wicket"] == 1: pr = pb = 0

print(f"\nLegal balls — min:{balls['legal_ball_number'].min()} max:{balls['legal_ball_number'].max()}")


# ════════════════════════════════════════════════════════════════════════════
# 4.  CONDITIONS
# ════════════════════════════════════════════════════════════════════════════

if "start_time" in matches.columns:
    matches["start_hour"] = matches["start_time"].apply(
        lambda t: int(str(t).split(":")[0]) if ":" in str(t) else 19)
else:
    matches["start_hour"] = 19; print("INFO: defaulting to evening.")
matches["is_day_match"] = (matches["start_hour"] < 16).astype(int)

SUBCONTINENT_KW = ["mumbai","delhi","chennai","kolkata","bangalore","hyderabad","pune",
    "rajkot","indore","nagpur","ahmedabad","chandigarh","jaipur","cuttack",
    "visakhapatnam","dharamsala","mohali","lahore","karachi","dhaka","colombo",
    "dubai","abu dhabi","sharjah"]
def dew_flag(row):
    if row["is_day_match"]: return 0
    return int(any(kw in str(row.get("venue","")).lower() for kw in SUBCONTINENT_KW))
matches["dew_likely"] = matches.apply(dew_flag, axis=1)
matches["season_phase"] = matches["date"].dt.month.apply(
    lambda m: 0 if m <= 4 else (1 if m <= 5 else 2))

inn1_tm = (balls[balls["innings"] == 1]
           .groupby("match_id")["total_runs"].sum()
           .rename("innings1_total_match").reset_index())
matches["match_id"] = matches["match_id"].astype(str)
inn1_tm["match_id"] = inn1_tm["match_id"].astype(str)
matches = matches.merge(inn1_tm, on="match_id", how="left")
matches = matches.sort_values("date").reset_index(drop=True)

gmt = matches["innings1_total_match"].mean()
n = len(matches); val_start = int(n * 0.70); test_start = int(n * 0.80)
vam = matches.iloc[:val_start].groupby("venue")["innings1_total_match"].mean().to_dict()
matches["venue_avg_total"] = matches["venue"].map(vam).fillna(gmt)
print(f"\nVenue avg total (global mean={gmt:.1f})")


# ════════════════════════════════════════════════════════════════════════════
# 5.  STATE TABLE
# ════════════════════════════════════════════════════════════════════════════

def build_state_table(balls, matches):
    df = balls.copy(); df["match_id"] = df["match_id"].astype(str)
    df["runs_so_far"] = df.groupby(["match_id","innings"])["total_runs"].transform(
        lambda x: x.shift(1, fill_value=0).cumsum())
    df["wickets_lost"] = df.groupby(["match_id","innings"])["wicket"].transform(
        lambda x: x.shift(1, fill_value=0).cumsum())
    df["legal_ball_before"] = df.groupby(["match_id","innings"])["is_legal"].transform(
        lambda x: x.shift(1, fill_value=0).cumsum()).clip(0, 120)
    df["balls_remaining_before"] = (120 - df["legal_ball_before"]).clip(0, 120)
    df["runs_last_12"] = df.groupby(["match_id","innings"])["total_runs"].transform(
        lambda x: x.rolling(12, min_periods=1).sum().shift(1, fill_value=0))
    df["wickets_last_12"] = df.groupby(["match_id","innings"])["wicket"].transform(
        lambda x: x.rolling(12, min_periods=1).sum().shift(1, fill_value=0))
    df["phase"] = pd.cut(df["legal_ball_before"], bins=[-1,36,90,120], labels=[0,1,2]).astype(int)
    df["current_rr"] = np.where(df["legal_ball_before"] > 0,
        df["runs_so_far"] / (df["legal_ball_before"] / 6), 0.0)

    i1t = df[df["innings"]==1].groupby("match_id")["total_runs"].sum().rename("innings1_total")
    df = df.merge(i1t, on="match_id", how="left")

    df["runs_needed"] = np.where(df["innings"]==2,
        (df["innings1_total"]+1) - df["runs_so_far"], np.nan)
    df["required_rr"] = np.where(
        (df["innings"]==2) & (df["balls_remaining_before"]>0),
        df["runs_needed"] / (df["balls_remaining_before"]/6), np.nan)
    df["wickets_remaining"] = 10 - df["wickets_lost"]
    df["rr_diff"] = np.where(df["innings"]==2,
        df["current_rr"] - df["required_rr"], np.nan)
    df["pressure_index"] = np.where(df["innings"]==2,
        np.clip(df["required_rr"].fillna(0)/12.0, 0, 2)
        * (1.0 + 0.10 * df["wickets_lost"])
        * (1.0 + 0.50 * (1 - df["balls_remaining_before"]/120)), np.nan)

    wm = matches.set_index("match_id")["winner"].to_dict()
    df["match_winner"] = df["match_id"].map(wm)
    df["y"] = (df["batting_team"] == df["match_winner"]).astype(int)

    mc = (["match_id","date","source","is_day_match","dew_likely","season_phase","venue_avg_total"]
          + WEATHER_FEATURES)
    mc = [c for c in mc if c in matches.columns]
    df = df.merge(matches[mc].drop_duplicates("match_id"), on="match_id", how="left")
    return df

state = build_state_table(balls, matches)
state = state.dropna(subset=["y"])
print(f"\nState table: {state.shape}  Win rate: {state['y'].mean():.3f}")


# ════════════════════════════════════════════════════════════════════════════
# 6.  ELO  (computed separately per format, then merged back)
# ════════════════════════════════════════════════════════════════════════════

def compute_elo(matches_df, K=20.0, init=1500.0):
    """
    Compute pre-match Elo ratings for each row in matches_df (sorted by date).
    Returns a DataFrame with match_id, elo_team1_before, elo_team2_before, p_pre_match.
    """
    elo, rows = {}, []
    for _, row in matches_df.sort_values("date").iterrows():
        t1, t2 = row["team1"], row["team2"]
        e1, e2 = elo.get(t1, init), elo.get(t2, init)
        p1 = 1 / (1 + 10 ** (-(e1 - e2) / 400))
        s1 = 1.0 if row["winner"] == t1 else 0.0
        elo[t1] = e1 + K * (s1 - p1)
        elo[t2] = e2 + K * ((1 - s1) - (1 - p1))
        rows.append({
            "match_id":          str(row["match_id"]),
            "elo_team1_before":  e1,
            "elo_team2_before":  e2,
            "p_pre_match":       p1,
        })
    return pd.DataFrame(rows)

# Run Elo independently for each format
ipl_elo  = compute_elo(matches[matches["source"] == "ipl"].copy())
t20i_elo = compute_elo(matches[matches["source"] == "t20i"].copy())
elo_df   = pd.concat([ipl_elo, t20i_elo], ignore_index=True)

matches["match_id"] = matches["match_id"].astype(str)
matches = matches.merge(elo_df, on="match_id", how="left")


# ════════════════════════════════════════════════════════════════════════════
# 7.  POST-TOSS
# ════════════════════════════════════════════════════════════════════════════

def logit(p):
    return np.log(np.clip(p, 1e-6, 1-1e-6) / (1 - np.clip(p, 1e-6, 1-1e-6)))

matches["logit_pre"] = logit(matches["p_pre_match"])
matches["toss_winner_is_t1"] = (matches["toss_winner"] == matches["team1"]).astype(int)
matches["bats_first_is_t1"] = (
    ((matches["toss_winner"]==matches["team1"]) & (matches["toss_decision"]=="bat")) |
    ((matches["toss_winner"]==matches["team2"]) & (matches["toss_decision"]=="field"))
).astype(int)
matches["y_match"] = (matches["winner"] == matches["team1"]).astype(int)

m_train = matches.iloc[:val_start]; m_test = matches.iloc[test_start:]
toss_feats = ["logit_pre", "toss_winner_is_t1", "bats_first_is_t1"]
lr_toss = LogisticRegression(max_iter=500)
lr_toss.fit(m_train[toss_feats], m_train["y_match"])
print(f"\nPost-toss Brier: {brier_score_loss(m_test['y_match'], lr_toss.predict_proba(m_test[toss_feats])[:,1]):.4f}")
matches["p_post_toss"] = lr_toss.predict_proba(matches[toss_feats])[:, 1]
state = state.merge(
    matches[["match_id","p_post_toss"]].drop_duplicates("match_id"),
    on="match_id", how="left")


# ════════════════════════════════════════════════════════════════════════════
# 8.  TIME SPLIT
# ════════════════════════════════════════════════════════════════════════════

ds = matches["date"].sort_values()
val_date = ds.iloc[val_start]; test_date = ds.iloc[test_start]
print(f"Split — train:<{val_date.date()}  val:{val_date.date()}–{test_date.date()}  test:>{test_date.date()}")

inn1_all = state[state["innings"] == 1].copy()
train_i1 = inn1_all[inn1_all["date"] < val_date].copy()
val_i1   = inn1_all[(inn1_all["date"] >= val_date) & (inn1_all["date"] < test_date)].copy()


# ════════════════════════════════════════════════════════════════════════════
# 9.  MODEL T — MEDIAN + QUANTILE MODELS
# ════════════════════════════════════════════════════════════════════════════
# Train 3 models: median (regression), p25 (quantile 0.25), p75 (quantile 0.75)
# All three outputs get fed into Model W — this is the key innings-1 change.

INN1_FEATURES = [
    "runs_so_far", "balls_remaining_before", "wickets_lost",
    "runs_last_12", "wickets_last_12", "phase",
    "venue_avg_total", "is_day_match", "season_phase",
    # Wave 2
    "batting_strength_remaining", "bowling_strength_remaining",
    "wickets_remaining_weighted",
    # Wave 3
    "partnership_runs", "partnership_balls",
    # Wave 4
    "batsman_sr_2yr", "batsman_boundary_pct_2yr", "batsman_dot_pct_2yr",
    "bowler_economy_2yr", "bowler_dot_pct_2yr", "momentum_index", "over_par",
    # Wave 4b
    "bowler_is_pace", "batsman_sr_vs_pace", "batsman_sr_vs_spin", "pace_spin_sr_diff",
] + WEATHER_FEATURES

# Filter to features that actually exist in the data
INN1_FEATURES = [f for f in INN1_FEATURES if f in state.columns]
print(f"\nINN1_FEATURES ({len(INN1_FEATURES)}): {INN1_FEATURES}")

train_i1 = train_i1.dropna(subset=INN1_FEATURES + ["innings1_total"]).copy()
val_i1   = val_i1.dropna(subset=INN1_FEATURES + ["innings1_total"]).copy()

# ── Optuna for median model ──────────────────────────────────────────────────

def _objective_T(trial):
    p = dict(
        objective="regression", metric="mae",
        num_leaves=trial.suggest_int("num_leaves", 64, 512),
        min_child_samples=trial.suggest_int("min_child_samples", 10, 100),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        subsample=trial.suggest_float("subsample", 0.6, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
        n_estimators=trial.suggest_int("n_estimators", 400, 1000),
        random_state=42, verbose=-1)
    m = lgb.LGBMRegressor(**p)
    m.fit(train_i1[INN1_FEATURES], train_i1["innings1_total"],
          eval_set=[(val_i1[INN1_FEATURES], val_i1["innings1_total"])],
          callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)])
    return np.mean(np.abs(m.predict(val_i1[INN1_FEATURES]) - val_i1["innings1_total"]))

print("\nModel T — Optuna (50 trials)...")
study_T = optuna.create_study(direction="minimize",
                               sampler=optuna.samplers.TPESampler(seed=42))
study_T.optimize(_objective_T, n_trials=50, show_progress_bar=False)
best_T = study_T.best_params
print(f"  Best MAE: {study_T.best_value:.3f}")

pT = dict(objective="regression", metric="mae", random_state=42, verbose=-1, **best_T)

# ── Train median model ───────────────────────────────────────────────────────
model_T = lgb.LGBMRegressor(**pT)
model_T.fit(train_i1[INN1_FEATURES], train_i1["innings1_total"],
            eval_set=[(val_i1[INN1_FEATURES], val_i1["innings1_total"])],
            callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)])

# ── Train quantile models (p25, p75 for spread; p05, p95 for uncertainty band) ─
pTq = {**pT, "n_estimators": 800, "learning_rate": 0.03, "min_child_samples": 10}

model_T_p25 = lgb.LGBMRegressor(**{**pTq, "objective": "quantile", "alpha": 0.25})
model_T_p75 = lgb.LGBMRegressor(**{**pTq, "objective": "quantile", "alpha": 0.75})
model_T_p05 = lgb.LGBMRegressor(**{**pTq, "objective": "quantile", "alpha": 0.05})
model_T_p95 = lgb.LGBMRegressor(**{**pTq, "objective": "quantile", "alpha": 0.95})

for m in [model_T_p25, model_T_p75, model_T_p05, model_T_p95]:
    m.fit(train_i1[INN1_FEATURES], train_i1["innings1_total"],
          callbacks=[lgb.log_evaluation(0)])

# ── Evaluate ─────────────────────────────────────────────────────────────────
val_i1["proj_total"]     = model_T.predict(val_i1[INN1_FEATURES])
val_i1["proj_total_p25"] = model_T_p25.predict(val_i1[INN1_FEATURES])
val_i1["proj_total_p75"] = model_T_p75.predict(val_i1[INN1_FEATURES])
val_i1["proj_total_spread"] = val_i1["proj_total_p75"] - val_i1["proj_total_p25"]

val_i1["over_number"] = (val_i1["legal_ball_before"] // 6).clip(0, 19)
mbo = val_i1.groupby("over_number").apply(
    lambda g: np.mean(np.abs(g["proj_total"] - g["innings1_total"])))
print("\nMAE by over (val):")
print(mbo.to_string())

# Show spread by over — this is the key diagnostic
spread_by_over = val_i1.groupby("over_number")["proj_total_spread"].mean()
print("\n── Quantile spread (p75-p25) by over ───────────────────────────────")
print(spread_by_over.to_string())

print("\n── Model T feature importances ─────────────────────────────────────")
fi_T = pd.Series(model_T.feature_importances_, index=INN1_FEATURES).sort_values(ascending=False)
for f, i in fi_T.items():
    print(f"  {f:35s} {'█' * int(i / fi_T.max() * 30)}")

# Uncertainty band for plots (p05/p95)
rp05 = np.minimum(model_T_p05.predict(val_i1[INN1_FEATURES]), val_i1["proj_total"].values)
rp95 = np.maximum(model_T_p95.predict(val_i1[INN1_FEATURES]), val_i1["proj_total"].values)
tv = val_i1["innings1_total"].values
margin = 0.0
for c in np.arange(0, 50, 0.5):
    if ((tv >= rp05-c) & (tv <= rp95+c)).mean() >= 0.75:
        margin = c; break
print(f"\nQuantile band margin (p05/p95): ±{margin:.1f}")


# ════════════════════════════════════════════════════════════════════════════
# 10.  ATTACH proj_total + quantile spread → FINAL SPLIT
# ════════════════════════════════════════════════════════════════════════════

i1m = state["innings"] == 1

# Median projection
state["proj_total"] = np.nan
state.loc[i1m, "proj_total"] = model_T.predict(
    state.loc[i1m, INN1_FEATURES].fillna(0))
state.loc[~i1m, "proj_total"] = state.loc[~i1m, "innings1_total"]

# Quantile projections (innings 1 only; innings 2 = known total)
state["proj_total_p25"] = np.nan
state["proj_total_p75"] = np.nan
state.loc[i1m, "proj_total_p25"] = model_T_p25.predict(
    state.loc[i1m, INN1_FEATURES].fillna(0))
state.loc[i1m, "proj_total_p75"] = model_T_p75.predict(
    state.loc[i1m, INN1_FEATURES].fillna(0))

# For innings 2: actual total is known → no uncertainty
state.loc[~i1m, "proj_total_p25"] = state.loc[~i1m, "innings1_total"]
state.loc[~i1m, "proj_total_p75"] = state.loc[~i1m, "innings1_total"]

# Spread = uncertainty width
state["proj_total_spread"] = (state["proj_total_p75"] - state["proj_total_p25"]).clip(0)

print(f"\n── Quantile spread stats ───────────────────────────────────────────")
for inn in [1, 2]:
    s = state[state["innings"] == inn]["proj_total_spread"]
    print(f"  Innings {inn}: mean={s.mean():.1f}  std={s.std():.1f}  "
          f"min={s.min():.1f}  max={s.max():.1f}")

# Elo
el = matches[["match_id","team1","elo_team1_before","elo_team2_before"]].drop_duplicates("match_id")
for c in ["team1","elo_team1_before","elo_team2_before"]:
    if c in state.columns: state = state.drop(columns=[c])
state = state.merge(el, on="match_id", how="left")
state["elo_diff"] = np.where(
    state["batting_team"] == state["team1"],
    state["elo_team1_before"] - state["elo_team2_before"],
    state["elo_team2_before"] - state["elo_team1_before"])

train_state = state[state["date"] < val_date].copy()
val_state   = state[(state["date"] >= val_date) & (state["date"] < test_date)].copy()
test_state  = state[state["date"] >= test_date].copy()
print(f"\nSplit — tr:{len(train_state):,}  vl:{len(val_state):,}  te:{len(test_state):,}")


# ════════════════════════════════════════════════════════════════════════════
# 11.  LR BASELINE
# ════════════════════════════════════════════════════════════════════════════

INN2_LR = ["runs_needed","balls_remaining_before","wickets_remaining",
           "required_rr","runs_last_12","p_post_toss"]
train_i2 = train_state[train_state["innings"]==2].dropna(subset=INN2_LR+["y"]).copy()
test_i2  = test_state[test_state["innings"]==2].dropna(subset=INN2_LR+["y"]).copy()
scaler = StandardScaler(); lr_base = LogisticRegression(max_iter=500)
lr_base.fit(scaler.fit_transform(train_i2[INN2_LR]), train_i2["y"])
p_lr_test = lr_base.predict_proba(scaler.transform(test_i2[INN2_LR]))[:, 1]
print(f"\nLR baseline Brier: {brier_score_loss(test_i2['y'], p_lr_test):.4f}")


# ════════════════════════════════════════════════════════════════════════════
# 12.  MODEL W — with quantile spread features
# ════════════════════════════════════════════════════════════════════════════
# Key change: proj_total_p25, proj_total_p75, proj_total_spread replace
# the zero-importance matchup features from v8b.

WIN_FEATURES = [
    "p_post_toss", "elo_diff", "innings",
    "wickets_lost", "wickets_remaining",
    "current_rr", "proj_total",
    "runs_needed", "required_rr", "rr_diff",
    "venue_avg_total",
    "batting_strength_remaining", "wickets_remaining_weighted",
    # Wave 4 (kept)
    "pressure_index",
    # v9: quantile spread from Model T
    "proj_total_p25", "proj_total_p75", "proj_total_spread",
] + WEATHER_FEATURES

def prep(df):
    return df.dropna(subset=["y","p_post_toss","proj_total","elo_diff"]).copy()

tr = prep(train_state); vl = prep(val_state); te = prep(test_state)
print(f"\nModel W — tr:{len(tr):,}  vl:{len(vl):,}  te:{len(te):,}")
print(f"WIN_FEATURES ({len(WIN_FEATURES)}): {WIN_FEATURES}")

# ── Optuna ───────────────────────────────────────────────────────────────────

def _objective_W(trial):
    p = dict(
        objective="binary", metric="binary_logloss",
        num_leaves=trial.suggest_int("num_leaves", 32, 512),
        min_child_samples=trial.suggest_int("min_child_samples", 10, 200),
        learning_rate=trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
        subsample=trial.suggest_float("subsample", 0.6, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        n_estimators=1000, random_state=42, verbose=-1)
    m = lgb.LGBMClassifier(**p)
    m.fit(tr[WIN_FEATURES], tr["y"],
          eval_set=[(vl[WIN_FEATURES], vl["y"])],
          callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
    return brier_score_loss(vl["y"], m.predict_proba(vl[WIN_FEATURES])[:, 1])

print("\nModel W — Optuna (60 trials)...")
study_W = optuna.create_study(direction="minimize",
                               sampler=optuna.samplers.TPESampler(seed=42))
study_W.optimize(_objective_W, n_trials=60, show_progress_bar=False)
best_W = study_W.best_params
print(f"  Best val Brier: {study_W.best_value:.4f}")
print("  Top 5:")
for _, row in study_W.trials_dataframe().sort_values("value").head(5).iterrows():
    print(f"    #{int(row['number']):3d} Brier={row['value']:.4f} "
          f"leaves={int(row['params_num_leaves'])} lr={row['params_learning_rate']:.4f}")

model_W = lgb.LGBMClassifier(
    objective="binary", metric="binary_logloss",
    n_estimators=1000, random_state=42, verbose=-1, **best_W)
model_W.fit(tr[WIN_FEATURES], tr["y"],
            eval_set=[(vl[WIN_FEATURES], vl["y"])],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])


# ════════════════════════════════════════════════════════════════════════════
# 12.5  CALIBRATION
# ════════════════════════════════════════════════════════════════════════════

print("\n── Calibration ─────────────────────────────────────────────────────")
pv = model_W.predict_proba(vl[WIN_FEATURES])[:, 1]
lv = np.log(np.clip(pv, 1e-7, 1-1e-7) / (1 - np.clip(pv, 1e-7, 1-1e-7)))
T_opt = minimize_scalar(
    lambda T: log_loss(vl["y"], 1/(1+np.exp(-lv/T))),
    bounds=(0.1, 10), method="bounded").x
print(f"  T = {T_opt:.4f}")

br = brier_score_loss(vl["y"], pv)
bt = brier_score_loss(vl["y"], 1/(1+np.exp(-lv/T_opt)))
model_W_platt = CalibratedClassifierCV(model_W, method="sigmoid", cv="prefit")
model_W_platt.fit(vl[WIN_FEATURES], vl["y"])
bp = brier_score_loss(vl["y"], model_W_platt.predict_proba(vl[WIN_FEATURES])[:, 1])
print(f"  Val — raw:{br:.4f}  Platt:{bp:.4f}  temp:{bt:.4f}")
cm = {"raw": br, "platt": bp, "temperature": bt}
best_cal = min(cm, key=cm.get)
print(f"  Best: {best_cal}")

def apply_cal(model, X):
    p = model.predict_proba(X)[:, 1]
    if best_cal == "temperature":
        l = np.log(np.clip(p, 1e-7, 1-1e-7) / (1 - np.clip(p, 1e-7, 1-1e-7)))
        return 1 / (1 + np.exp(-l / T_opt))
    elif best_cal == "platt":
        return model_W_platt.predict_proba(X)[:, 1]
    return p


# ════════════════════════════════════════════════════════════════════════════
# 13.  EVALUATION
# ════════════════════════════════════════════════════════════════════════════

p_raw_test = model_W.predict_proba(te[WIN_FEATURES])[:, 1]
p_cal_test = apply_cal(model_W, te[WIN_FEATURES])
te["p_cal"] = p_cal_test

brier_overall = brier_score_loss(te["y"], p_cal_test)
logloss_cal = log_loss(te["y"], p_cal_test)
te_i2 = te[te["innings"] == 2].copy()

print("\n" + "=" * 70)
print("EVALUATION REPORT — TEST SET (v10 / IPL + T20I Combined)")
print("=" * 70)

print(f"\n── Overall ──────────────────────────────────────────────────────────")
print(f"  Always-50%       Brier: 0.2500")
print(f"  LR baseline      Brier: {brier_score_loss(test_i2['y'], p_lr_test):.4f}")
print(f"  LGB raw          Brier: {brier_score_loss(te['y'], p_raw_test):.4f}")
print(f"  BEST ({best_cal:11s}) Brier: {brier_overall:.4f}  LogLoss: {logloss_cal:.4f}")
print(f"  vs 0.25 baseline      : {(0.25-brier_overall)/0.25*100:.1f}% better")
print(f"  v9 (Quantile Spread)  : update after v9 run")
print(f"  v8b (Wave 4b) was     : 0.1724")

print(f"\n── By innings ───────────────────────────────────────────────────────")
for inn in [1, 2]:
    s = te[te["innings"] == inn]
    print(f"  Inn {inn}: Brier={brier_score_loss(s['y'], s['p_cal']):.4f} (n={len(s):,})")

print(f"\n── By phase ─────────────────────────────────────────────────────────")
for ph, nm in {0:"PP (1-6)", 1:"Mid (7-15)", 2:"Death (16-20)"}.items():
    s = te[te["phase"] == ph]
    if len(s):
        print(f"  {nm:15s}: Brier={brier_score_loss(s['y'], s['p_cal']):.4f} (n={len(s):,})")

print(f"\n── RRR buckets (inn 2) ──────────────────────────────────────────────")
te_i2["rrr_bucket"] = pd.cut(te_i2["required_rr"].fillna(0), bins=[0,6,8,10,50],
    labels=["Easy (<6)","Mod (6-8)","Hard (8-10)","VHard (>10)"])
for b, g in te_i2.groupby("rrr_bucket", observed=True):
    print(f"  {str(b):15s}: Brier={brier_score_loss(g['y'], g['p_cal']):.4f} (n={len(g):,})")

print(f"\n── Feature importances (Model W) ────────────────────────────────────")
fi = pd.Series(model_W.feature_importances_, index=WIN_FEATURES).sort_values(ascending=False)
for f, i in fi.items():
    print(f"  {f:35s} {'█' * int(i / fi.max() * 30)}")

inn1_b = brier_score_loss(te[te["innings"]==1]["y"], te[te["innings"]==1]["p_cal"])
inn2_b = brier_score_loss(te[te["innings"]==2]["y"], te[te["innings"]==2]["p_cal"])
print(f"""
── Benchmark ────────────────────────────────────────────────────────
  ┌─────────────────────────────────────────────┬───────────┬────────────────┐
  │ Model                                       │   Brier   │  Setting       │
  ├─────────────────────────────────────────────┼───────────┼────────────────┤
  │ Always 50%                                  │   0.2500  │  —             │
  │ LR (Bailey & Clarke 2006)                   │ ~0.210    │  ODI/T20       │
  │ RF (Sankaranarayanan 2014)                  │ ~0.195    │  T20           │
  │ v5 (pre-Wave 2)                             │   0.1942  │  IPL T20       │
  │ v6 (Wave 2)                                 │   0.1744  │  IPL T20       │
  │ v7 (Wave 3)                                 │   0.1720  │  IPL T20       │
  │ v8b (Wave 4b)                               │   0.1724  │  IPL T20       │
  │ v9 (Quantile Spread)                        │  update   │  IPL T20       │
  │ v10 (IPL + T20I)                            │  {brier_overall:.4f}  │  IPL + T20I    │
  │ SOTA (deep learning)                        │ ~0.165    │  T20 intl      │
  └─────────────────────────────────────────────┴───────────┴────────────────┘
  Inn-1: {inn1_b:.4f}  (v8b=0.2308, v7=0.2297)
  Inn-2: {inn2_b:.4f}  (v8b=0.1088, v7=0.1093)
  Cal: {best_cal}  T={T_opt:.4f}   Gap: {brier_overall-0.15:.3f}
""")

# ── Per-source breakdown ──────────────────────────────────────────────────────
if "source" in te.columns:
    print("── By source ────────────────────────────────────────────────────────")
    for src in ["ipl", "t20i"]:
        s = te[te["source"] == src]
        if len(s):
            print(f"  {src.upper():5s}: Brier={brier_score_loss(s['y'], s['p_cal']):.4f} (n={len(s):,})")

print("=" * 70)


# ════════════════════════════════════════════════════════════════════════════
# 14-16.  PLOTS
# ════════════════════════════════════════════════════════════════════════════

def plot_reliability(yt, pl, lb, title, fn):
    path = str(OUT_PLOTS / fn)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    cols = ["#E05A3A", "#0F7173", "#D4821A", "#6A4C93"]
    ax = axes[0]
    ax.plot([0,1],[0,1], "k--", lw=1.2, label="Perfect")
    for (p, l), c in zip(zip(pl, lb), cols):
        pt, pp = calibration_curve(yt, p, n_bins=10)
        ax.plot(pp, pt, "o-", color=c, lw=2, markersize=6, label=l)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Observed")
    ax.legend(fontsize=9); ax.grid(alpha=0.3); ax.set_xlim(0,1); ax.set_ylim(0,1)
    ax2 = axes[1]
    for (p, l), c in zip(zip(pl, lb), cols):
        ax2.hist(p, bins=20, alpha=0.4, color=c, label=l, density=True)
    ax2.set_xlabel("Predicted"); ax2.legend(fontsize=9); ax2.grid(alpha=0.3)
    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()
    print(f"Saved: {path}")

pri2 = model_W.predict_proba(test_i2[WIN_FEATURES])[:, 1]
pci2 = apply_cal(model_W, test_i2[WIN_FEATURES])
plot_reliability(test_i2["y"], [p_lr_test, pri2, pci2],
                 ["LR", "LGB raw", f"LGB {best_cal}"],
                 "Reliability Inn2 (v10)", "reliability_innings2.png")
plot_reliability(te["y"], [p_raw_test, p_cal_test],
                 ["raw", best_cal],
                 "Reliability Both (v10)", "reliability_both_innings.png")

def plot_match(mid, state, model, wf, matches):
    mdf = state[state["match_id"]==mid].copy()
    mdf = mdf.dropna(subset=["p_post_toss"])
    if mdf.empty: return
    mdf["wp"] = apply_cal(model, mdf[wf].fillna(0))
    row = matches[matches["match_id"]==mid].iloc[0]
    i1 = mdf[mdf["innings"]==1].copy(); i1["bx"] = i1["legal_ball_before"]+1
    i2 = mdf[mdf["innings"]==2].copy(); i2["bx"] = i2["legal_ball_before"]+121
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.axhline(0.5, color="#ddd", lw=0.8)
    ax.plot(i1["bx"], i1["wp"], color="#1B3A5C", lw=2, label="Inn1")
    ax.plot(i2["bx"], i2["wp"], color="#0F7173", lw=2, label="Inn2")
    ax.axvline(120, color="#999", lw=0.8)
    ax.set_xlim(0, 240); ax.set_ylim(0, 1)
    ax.set_title(f"{row['team1']} vs {row['team2']} | Winner: {row['winner']}")
    ax.legend(); ax.grid(alpha=0.2); plt.tight_layout()
    plt.savefig(str(OUT_PLOTS / f"match_{mid}.png"), dpi=150); plt.close()

def plot_proj(mid, state, mT, mp05, mp95, feat, mg):
    mdf = state[(state["match_id"]==mid) & (state["innings"]==1)].dropna(subset=feat)
    if mdf.empty: return
    mdf = mdf.copy()
    med = mT.predict(mdf[feat])
    p05 = np.minimum(mp05.predict(mdf[feat]), med) - mg
    p95 = np.maximum(mp95.predict(mdf[feat]), med) + mg
    mdf["bx"] = mdf["legal_ball_before"] + 1
    tt = mdf["innings1_total"].iloc[0]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.fill_between(mdf["bx"], p05, p95, alpha=0.2, color="#0F7173")
    ax.plot(mdf["bx"], med, color="#0F7173", lw=2)
    ax.axhline(tt, color="#E05A3A", ls="--", lw=1.5, label=f"Actual:{int(tt)}")
    ax.legend(); ax.grid(alpha=0.25); plt.tight_layout()
    plt.savefig(str(OUT_PLOTS / f"proj_{mid}.png"), dpi=150); plt.close()

for mid in te["match_id"].unique()[:3]:
    plot_match(mid, te, model_W, WIN_FEATURES, matches)
    plot_proj(mid, te, model_T, model_T_p05, model_T_p95, INN1_FEATURES, margin)


# ════════════════════════════════════════════════════════════════════════════
# 17.  SAVE
# ════════════════════════════════════════════════════════════════════════════

joblib.dump(model_T,      str(OUT_MODELS / "model_T.pkl"))
joblib.dump(model_T_p25,  str(OUT_MODELS / "model_T_p25.pkl"))
joblib.dump(model_T_p75,  str(OUT_MODELS / "model_T_p75.pkl"))
joblib.dump(model_T_p05,  str(OUT_MODELS / "model_T_p05.pkl"))
joblib.dump(model_T_p95,  str(OUT_MODELS / "model_T_p95.pkl"))
joblib.dump(model_W,      str(OUT_MODELS / "model_W.pkl"))
joblib.dump(model_W_platt,str(OUT_MODELS / "model_W_platt.pkl"))
joblib.dump(lr_toss,      str(OUT_MODELS / "model_toss.pkl"))
joblib.dump(scaler,       str(OUT_MODELS / "scaler_lr.pkl"))
joblib.dump({
    "win_features": WIN_FEATURES,
    "inn1_features": INN1_FEATURES,
    "weather_features": WEATHER_FEATURES,
    "T_optimal": T_opt,
    "best_calibration": best_cal,
    "band_margin": margin,
}, str(OUT_MODELS / "run_config.pkl"))

print(f"\nModels → {OUT_MODELS}/")
print(f"Plots  → {OUT_PLOTS}/")
print(f"Log    → {_log_path}")
print("Pipeline complete.")

sys.stdout = sys.__stdout__
_log_file.close()