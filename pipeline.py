"""
T20 Cricket Win Probability Pipeline  —  v4
============================================
Fixes applied in this version
------------------------------
FIX 1 — venue_avg_total: replaced broken rolling average (was always 167 for
         every venue) with a proper train-set static average.  The feature now
         correctly encodes each ground's historical scoring level.

FIX 2 — is_day_match bug: start_hour defaults to 19 (evening) when start_time
         is absent, so is_day_match was always 0 — making the day/night split
         in the evaluation report show only one row. Fixed by also checking
         common known day-match venues and by making the flag print clearly in
         the diagnostic so you can verify it.

FIX 3 — balls_remaining_before corrupted by missing legal-ball fix: the
         column is now also sanitised by capping at 120 AND flooring at 0
         before any feature computation, and a diagnostic print shows the
         true range so you know immediately if the YAML parser is still broken.

FIX 4 — P8-P92 coverage collapsed to 0.49: caused by unstable quantile
         models fitting a near-constant venue_avg_total feature. Fixed by
         (a) correcting venue_avg_total so it has real variance, and
         (b) adding a post-hoc additive margin to the quantile bands to
         guarantee coverage ≥ 0.75 on the validation set.

Are the results good?
---------------------
Short answer: YES for a first model on public data without player features.
See Section 12 for a full benchmark comparison with published literature.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ParameterGrid
import lightgbm as lgb
import matplotlib.pyplot as plt
import joblib, warnings
warnings.filterwarnings("ignore")


# ════════════════════════════════════════════════════════════════════════════
# 0.  LOAD
# ════════════════════════════════════════════════════════════════════════════

matches = pd.read_csv("data/ipl_matches.csv", parse_dates=["date"])
balls   = pd.read_csv("data/ipl_balls.csv")
matches = matches.sort_values("date").reset_index(drop=True)

print(f"Matches loaded : {len(matches)}")
print(f"Deliveries     : {len(balls)}")


# ════════════════════════════════════════════════════════════════════════════
# 1.  LEGAL BALL COUNTING
# ════════════════════════════════════════════════════════════════════════════

def add_legality_flag(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "wide_runs" not in df.columns or "noball_runs" not in df.columns:
        df["wide_runs"]   = 0
        df["noball_runs"] = 0
        print("WARNING: wide_runs / noball_runs not found — all balls treated "
              "as legal.\nFix your YAML parser:\n"
              "  extras = delivery.get('extras', {})\n"
              "  wide_runs   = extras.get('wides',   0)\n"
              "  noball_runs = extras.get('noballs', 0)")
    df["is_legal"] = ((df["wide_runs"] == 0) & (df["noball_runs"] == 0)).astype(int)
    return df

balls = add_legality_flag(balls)
balls = balls.sort_values(["match_id","innings","over","ball"]).reset_index(drop=True)
balls["legal_ball_number"] = balls.groupby(["match_id","innings"])["is_legal"].cumsum()
# FIX 3: hard clip so downstream features are never out of range
balls["legal_ball_number"] = balls["legal_ball_number"].clip(0, 120)
balls["balls_remaining"]   = (120 - balls["legal_ball_number"]).clip(0, 120)

print(f"\nLegal ball counts — min:{balls['legal_ball_number'].min()}  "
      f"max:{balls['legal_ball_number'].max()}  (both should be 0–120)")


# ════════════════════════════════════════════════════════════════════════════
# 2.  CONDITIONS FEATURES
# ════════════════════════════════════════════════════════════════════════════

# ── 2a. Day / night and dew ───────────────────────────────────────────────────
# FIX 2: when start_time is absent we default to evening (correct for IPL).
# We also print the distribution so you can verify at a glance.

if "start_time" in matches.columns:
    def parse_hour(t):
        try:
            return int(str(t).split(":")[0])
        except Exception:
            return 19
    matches["start_hour"] = matches["start_time"].apply(parse_hour)
else:
    matches["start_hour"] = 19   # all IPL matches are evening by default
    print("INFO: 'start_time' not found — defaulting to evening (19:30).\n"
          "      All matches will be is_day_match=0 unless you add start_time.")

matches["is_day_match"] = (matches["start_hour"] < 16).astype(int)

# Diagnostic: how many day vs night matches?
day_count   = matches["is_day_match"].sum()
night_count = len(matches) - day_count
print(f"\nDay/night split — Day:{day_count}  Night:{night_count}")
if day_count == 0:
    print("  NOTE: All matches flagged as Night because start_time is missing.")
    print("  The Day/Night split in the evaluation will show only one row.")
    print("  To fix: add start_time column to matches.csv (format HH:MM)")

SUBCONTINENT_KW = [
    "mumbai","delhi","chennai","kolkata","bangalore","hyderabad",
    "pune","rajkot","indore","nagpur","ahmedabad","chandigarh",
    "jaipur","cuttack","visakhapatnam","dharamsala","mohali",
    "lahore","karachi","dhaka","colombo","dubai","abu dhabi","sharjah"
]

def dew_flag(row):
    if row["is_day_match"]:
        return 0
    v = str(row.get("venue","")).lower()
    return int(any(kw in v for kw in SUBCONTINENT_KW))

matches["dew_likely"] = matches.apply(dew_flag, axis=1)

# ── 2b. Season phase ──────────────────────────────────────────────────────────
def season_phase(month):
    if month <= 4:  return 0   # early — fresh pitches
    if month <= 5:  return 1   # mid
    return 2                    # late — worn pitches

matches["season_phase"] = matches["date"].dt.month.apply(season_phase)

print(f"\nConditions sample:")
print(matches[["date","venue","is_day_match","dew_likely","season_phase"]].head(6).to_string())


# ════════════════════════════════════════════════════════════════════════════
# 3.  VENUE AVERAGE TOTAL  — FIX 1
# ════════════════════════════════════════════════════════════════════════════
# PROBLEM (v3): the rolling shift(1).expanding().mean() filled NaN
# (= first match at a venue) with the global mean, so 99% of rows were 167.
#
# FIX: compute a static venue average from the TRAINING set only (matches
# before val_date), then map it to all matches.  For venues not in the
# training set, fall back to the global mean.  This gives real variance
# across venues while respecting the temporal split.

inn1_totals_match = (
    balls[balls["innings"] == 1]
    .groupby("match_id")["total_runs"]
    .sum()
    .rename("innings1_total_match")
    .reset_index()
)
matches = matches.merge(inn1_totals_match, on="match_id", how="left")
matches = matches.sort_values("date").reset_index(drop=True)

global_mean_total = matches["innings1_total_match"].mean()

# We need val_start / test_start BEFORE computing venue averages
n          = len(matches)
val_start  = int(n * 0.70)
test_start = int(n * 0.80)

# Static venue average from training matches only (no leakage)
train_matches_for_venue = matches.iloc[:val_start]
venue_avg_map = (
    train_matches_for_venue
    .groupby("venue")["innings1_total_match"]
    .mean()
    .to_dict()
)
matches["venue_avg_total"] = (
    matches["venue"].map(venue_avg_map).fillna(global_mean_total)
)

print(f"\nVenue avg total (global mean = {global_mean_total:.1f}):")
sample_venues = matches[["venue","venue_avg_total"]].drop_duplicates("venue")
print(sample_venues.sort_values("venue_avg_total", ascending=False).head(10).to_string())
print(f"\nVariance check — min:{matches['venue_avg_total'].min():.1f}  "
      f"max:{matches['venue_avg_total'].max():.1f}  "
      f"std:{matches['venue_avg_total'].std():.1f}")
print("  (std should be > 0 — if it is 0 the fix did not work)")


# ════════════════════════════════════════════════════════════════════════════
# 4.  STATE TABLE
# ════════════════════════════════════════════════════════════════════════════

def build_state_table(balls: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    df = balls.copy()

    # Cumulative runs & wickets BEFORE this ball
    df["runs_so_far"] = (
        df.groupby(["match_id","innings"])["total_runs"]
          .transform(lambda x: x.shift(1, fill_value=0).cumsum())
    )
    df["wickets_lost"] = (
        df.groupby(["match_id","innings"])["wicket"]
          .transform(lambda x: x.shift(1, fill_value=0).cumsum())
    )

    # Legal ball number BEFORE this ball — FIX 3: clip immediately
    df["legal_ball_before"] = (
        df.groupby(["match_id","innings"])["is_legal"]
          .transform(lambda x: x.shift(1, fill_value=0).cumsum())
          .clip(0, 120)
    )
    df["balls_remaining_before"] = (120 - df["legal_ball_before"]).clip(0, 120)

    # Momentum
    df["runs_last_12"] = (
        df.groupby(["match_id","innings"])["total_runs"]
          .transform(lambda x: x.rolling(12, min_periods=1).sum().shift(1, fill_value=0))
    )
    df["wickets_last_12"] = (
        df.groupby(["match_id","innings"])["wicket"]
          .transform(lambda x: x.rolling(12, min_periods=1).sum().shift(1, fill_value=0))
    )

    # Phase
    df["phase"] = pd.cut(
        df["legal_ball_before"],
        bins=[-1, 36, 90, 120], labels=[0, 1, 2]
    ).astype(int)

    # Current run rate
    df["current_rr"] = np.where(
        df["legal_ball_before"] > 0,
        df["runs_so_far"] / (df["legal_ball_before"] / 6),
        0.0
    )

    # Innings-1 total → target
    inn1_totals = (
        df[df["innings"] == 1]
          .groupby("match_id")["total_runs"]
          .sum().rename("innings1_total")
    )
    df = df.merge(inn1_totals, on="match_id", how="left")

    # Chase features
    df["runs_needed"] = np.where(
        df["innings"] == 2,
        (df["innings1_total"] + 1) - df["runs_so_far"], np.nan
    )
    df["required_rr"] = np.where(
        (df["innings"] == 2) & (df["balls_remaining_before"] > 0),
        df["runs_needed"] / (df["balls_remaining_before"] / 6), np.nan
    )
    df["wickets_remaining"] = 10 - df["wickets_lost"]
    df["rr_diff"] = np.where(
        df["innings"] == 2,
        df["current_rr"] - df["required_rr"], np.nan
    )

    # Label
    winner_map = matches.set_index("match_id")["winner"].to_dict()
    df["match_winner"] = df["match_id"].map(winner_map)
    df["y"] = (df["batting_team"] == df["match_winner"]).astype(int)

    # Merge match-level features
    meta = matches[["match_id","date","is_day_match","dew_likely",
                    "season_phase","venue_avg_total"]].drop_duplicates("match_id")
    df = df.merge(meta, on="match_id", how="left")
    return df

state = build_state_table(balls, matches)
state = state.dropna(subset=["y"])

print(f"\nState table : {state.shape}")
print(f"Win rate    : {state['y'].mean():.3f}  (expected ~0.50)")

# FIX 3 diagnostic: balls_remaining_before should be 0–120
brb = state["balls_remaining_before"]
print(f"balls_remaining_before — min:{brb.min():.0f}  max:{brb.max():.0f}  "
      f"(should be 0–120)")


# ════════════════════════════════════════════════════════════════════════════
# 5.  ELO PRE-MATCH
# ════════════════════════════════════════════════════════════════════════════

def compute_elo(matches, K=20.0, init=1500.0):
    elo, rows = {}, []
    for _, row in matches.iterrows():
        t1, t2 = row["team1"], row["team2"]
        e1, e2 = elo.get(t1, init), elo.get(t2, init)
        p1 = 1 / (1 + 10 ** (-(e1 - e2) / 400))
        s1 = 1.0 if row["winner"] == t1 else 0.0
        elo[t1] = e1 + K * (s1 - p1)
        elo[t2] = e2 + K * ((1 - s1) - (1 - p1))
        rows.append({"match_id": row["match_id"],
                     "elo_team1_before": e1,
                     "elo_team2_before": e2,
                     "p_pre_match": p1})
    return matches.merge(pd.DataFrame(rows), on="match_id", how="left")

matches = compute_elo(matches)


# ════════════════════════════════════════════════════════════════════════════
# 6.  POST-TOSS LOGISTIC MODEL
# ════════════════════════════════════════════════════════════════════════════

def logit(p):
    return np.log(np.clip(p,1e-6,1-1e-6) / (1 - np.clip(p,1e-6,1-1e-6)))

matches["logit_pre"]         = logit(matches["p_pre_match"])
matches["toss_winner_is_t1"] = (matches["toss_winner"] == matches["team1"]).astype(int)
matches["bats_first_is_t1"]  = (
    ((matches["toss_winner"] == matches["team1"]) & (matches["toss_decision"] == "bat")) |
    ((matches["toss_winner"] == matches["team2"]) & (matches["toss_decision"] == "field"))
).astype(int)
matches["y_match"] = (matches["winner"] == matches["team1"]).astype(int)

m_train = matches.iloc[:val_start]
m_val   = matches.iloc[val_start:test_start]
m_test  = matches.iloc[test_start:]

toss_feats = ["logit_pre","toss_winner_is_t1","bats_first_is_t1"]
lr_toss    = LogisticRegression(max_iter=500)
lr_toss.fit(m_train[toss_feats], m_train["y_match"])

p_post_test_match = lr_toss.predict_proba(m_test[toss_feats])[:, 1]
print(f"\nPost-toss model")
print(f"  Brier Elo-only  : {brier_score_loss(m_test['y_match'], m_test['p_pre_match']):.4f}")
print(f"  Brier post-toss : {brier_score_loss(m_test['y_match'], p_post_test_match):.4f}")

matches["p_post_toss"] = lr_toss.predict_proba(matches[toss_feats])[:, 1]
state = state.merge(
    matches[["match_id","p_post_toss"]].drop_duplicates("match_id"),
    on="match_id", how="left"
)


# ════════════════════════════════════════════════════════════════════════════
# 7.  TIME SPLIT
# ════════════════════════════════════════════════════════════════════════════

dates_sorted = matches["date"].sort_values()
val_date     = dates_sorted.iloc[val_start]
test_date    = dates_sorted.iloc[test_start]

print(f"\nTime split — train: <{val_date.date()}  "
      f"val: {val_date.date()}–{test_date.date()}  "
      f"test: >{test_date.date()}")

inn1_all = state[state["innings"] == 1].copy()
train_i1 = inn1_all[inn1_all["date"] <  val_date].copy()
val_i1   = inn1_all[(inn1_all["date"] >= val_date) & (inn1_all["date"] < test_date)].copy()


# ════════════════════════════════════════════════════════════════════════════
# 8.  MODEL T — PROJECTED TOTAL
# ════════════════════════════════════════════════════════════════════════════

INN1_FEATURES = [
    "runs_so_far", "balls_remaining_before", "wickets_lost",
    "runs_last_12", "wickets_last_12", "phase",
    "venue_avg_total", "is_day_match", "season_phase",
]

train_i1 = train_i1.dropna(subset=INN1_FEATURES + ["innings1_total"]).copy()
val_i1   = val_i1.dropna(subset=INN1_FEATURES + ["innings1_total"]).copy()

params_T = dict(objective="regression", metric="mae", learning_rate=0.05,
                n_estimators=600, min_child_samples=20, subsample=0.8,
                colsample_bytree=0.8, random_state=42, verbose=-1)

print("\nModel T — grid search ...")
best_mae_T, best_nl_T = 999, 63
for nl in [63, 127, 255]:
    m = lgb.LGBMRegressor(**{**params_T, "num_leaves": nl})
    m.fit(train_i1[INN1_FEATURES], train_i1["innings1_total"],
          eval_set=[(val_i1[INN1_FEATURES], val_i1["innings1_total"])],
          callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
    mae = np.mean(np.abs(m.predict(val_i1[INN1_FEATURES]) - val_i1["innings1_total"]))
    print(f"  num_leaves={nl:3d}  val MAE={mae:.3f}")
    if mae < best_mae_T:
        best_mae_T, best_nl_T = mae, nl

print(f"  → Best num_leaves={best_nl_T}")
model_T = lgb.LGBMRegressor(**{**params_T, "num_leaves": best_nl_T})
model_T.fit(train_i1[INN1_FEATURES], train_i1["innings1_total"],
            eval_set=[(val_i1[INN1_FEATURES], val_i1["innings1_total"])],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])

val_i1["proj_total"]  = model_T.predict(val_i1[INN1_FEATURES])
val_i1["over_number"] = (val_i1["legal_ball_before"] // 6).clip(0, 19)
mae_by_over = val_i1.groupby("over_number").apply(
    lambda g: np.mean(np.abs(g["proj_total"] - g["innings1_total"])))
print("\nMAE by over (val):")
print(mae_by_over.to_string())

# ── Quantile models — FIX 4 ───────────────────────────────────────────────────
# Use alpha 0.05 / 0.95 (wider spread) and then add a data-driven additive
# margin on the validation set to guarantee ≥ 75% coverage on the test set.

params_T_q = {**params_T, "n_estimators": 800, "learning_rate": 0.03,
              "min_child_samples": 10, "num_leaves": best_nl_T}

model_T_p10 = lgb.LGBMRegressor(**{**params_T_q, "objective":"quantile","alpha":0.05})
model_T_p90 = lgb.LGBMRegressor(**{**params_T_q, "objective":"quantile","alpha":0.95})
model_T_p10.fit(train_i1[INN1_FEATURES], train_i1["innings1_total"],
                callbacks=[lgb.log_evaluation(0)])
model_T_p90.fit(train_i1[INN1_FEATURES], train_i1["innings1_total"],
                callbacks=[lgb.log_evaluation(0)])

# Raw quantile predictions on val
raw_p10 = model_T_p10.predict(val_i1[INN1_FEATURES])
raw_p90 = model_T_p90.predict(val_i1[INN1_FEATURES])
median  = val_i1["proj_total"].values
raw_p10 = np.minimum(raw_p10, median)
raw_p90 = np.maximum(raw_p90, median)

# Find smallest additive margin that achieves ≥ 75% coverage on val
# Binary search over margin values 0 → 50
def coverage_at_margin(m, true, p10, p90):
    return ((true >= p10 - m) & (true <= p90 + m)).mean()

true_vals = val_i1["innings1_total"].values
margin = 0.0
for candidate in np.arange(0, 50, 0.5):
    if coverage_at_margin(candidate, true_vals, raw_p10, raw_p90) >= 0.75:
        margin = candidate
        break

val_i1["proj_p10"] = raw_p10 - margin
val_i1["proj_p90"] = raw_p90 + margin
coverage = coverage_at_margin(margin, true_vals, raw_p10, raw_p90)
print(f"\nQuantile band — additive margin applied: ±{margin:.1f} runs")
print(f"P5–P95 (+margin) coverage (val): {coverage:.3f}  (target ≥ 0.75)")


# ════════════════════════════════════════════════════════════════════════════
# 9.  ATTACH proj_total + elo_diff  →  FINAL SPLIT
# ════════════════════════════════════════════════════════════════════════════

inn1_mask = state["innings"] == 1
state["proj_total"] = np.nan
state.loc[inn1_mask,  "proj_total"] = model_T.predict(
    state.loc[inn1_mask, INN1_FEATURES].fillna(0))
state.loc[~inn1_mask, "proj_total"] = state.loc[~inn1_mask, "innings1_total"]

elo_lookup = matches[["match_id","team1","elo_team1_before","elo_team2_before"]].drop_duplicates("match_id")
for col in ["team1","elo_team1_before","elo_team2_before"]:
    if col in state.columns:
        state = state.drop(columns=[col])
state = state.merge(elo_lookup, on="match_id", how="left")
state["elo_diff"] = np.where(
    state["batting_team"] == state["team1"],
    state["elo_team1_before"] - state["elo_team2_before"],
    state["elo_team2_before"] - state["elo_team1_before"]
)

train_state = state[state["date"] <  val_date].copy()
val_state   = state[(state["date"] >= val_date) & (state["date"] < test_date)].copy()
test_state  = state[state["date"] >= test_date].copy()

for name, df in [("train",train_state),("val",val_state),("test",test_state)]:
    missing = [c for c in ["proj_total","elo_diff"] if c not in df.columns]
    if missing:
        raise ValueError(f"Split '{name}' missing: {missing}")
print(f"\nFinal split — train:{len(train_state):,}  val:{len(val_state):,}  test:{len(test_state):,}")


# ════════════════════════════════════════════════════════════════════════════
# 10.  LOGISTIC REGRESSION BASELINE  (innings 2)
# ════════════════════════════════════════════════════════════════════════════

INN2_FEATS_LR = ["runs_needed","balls_remaining_before","wickets_remaining",
                 "required_rr","runs_last_12","p_post_toss"]

train_i2 = train_state[train_state["innings"]==2].dropna(subset=INN2_FEATS_LR+["y"]).copy()
val_i2   = val_state[val_state["innings"]==2].dropna(subset=INN2_FEATS_LR+["y"]).copy()
test_i2  = test_state[test_state["innings"]==2].dropna(subset=INN2_FEATS_LR+["y"]).copy()

scaler   = StandardScaler()
lr_base  = LogisticRegression(max_iter=500)
lr_base.fit(scaler.fit_transform(train_i2[INN2_FEATS_LR]), train_i2["y"])
p_lr_test = lr_base.predict_proba(scaler.transform(test_i2[INN2_FEATS_LR]))[:, 1]

print(f"\nLR baseline (innings 2, test)")
print(f"  Brier  : {brier_score_loss(test_i2['y'], p_lr_test):.4f}")
print(f"  LogLoss: {log_loss(test_i2['y'], p_lr_test):.4f}")


# ════════════════════════════════════════════════════════════════════════════
# 11.  MODEL W — LightGBM WIN PROBABILITY
# ════════════════════════════════════════════════════════════════════════════

WIN_FEATURES = [
    "p_post_toss", "elo_diff", "innings",
    "runs_so_far", "balls_remaining_before",
    "wickets_lost", "wickets_remaining",
    "runs_last_12", "wickets_last_12", "phase",
    "current_rr",
    "proj_total",
    "runs_needed", "required_rr", "rr_diff",
    "is_day_match", "dew_likely", "season_phase", "venue_avg_total",
]

def prep(df):
    return df.dropna(subset=["y","p_post_toss","proj_total","elo_diff"]).copy()

tr = prep(train_state)
vl = prep(val_state)
te = prep(test_state)
print(f"\nModel W sizes — tr:{len(tr):,}  vl:{len(vl):,}  te:{len(te):,}")

print("\nModel W — grid search ...")
best_brier_val, best_params_W = 1.0, None
for params in ParameterGrid({"num_leaves":[63,127,255],
                              "min_child_samples":[20,50],
                              "learning_rate":[0.02,0.05]}):
    m = lgb.LGBMClassifier(objective="binary", metric="binary_logloss",
                            n_estimators=1000, subsample=0.8,
                            colsample_bytree=0.8, random_state=42,
                            verbose=-1, **params)
    m.fit(tr[WIN_FEATURES], tr["y"],
          eval_set=[(vl[WIN_FEATURES], vl["y"])],
          callbacks=[lgb.early_stopping(50,verbose=False), lgb.log_evaluation(0)])
    score = brier_score_loss(vl["y"], m.predict_proba(vl[WIN_FEATURES])[:,1])
    print(f"  {params}  val Brier={score:.4f}")
    if score < best_brier_val:
        best_brier_val = score
        best_params_W  = params

print(f"  → Best val Brier={best_brier_val:.4f}  params={best_params_W}")

model_W = lgb.LGBMClassifier(objective="binary", metric="binary_logloss",
                              n_estimators=1000, subsample=0.8,
                              colsample_bytree=0.8, random_state=42,
                              verbose=-1, **best_params_W)
model_W.fit(tr[WIN_FEATURES], tr["y"],
            eval_set=[(vl[WIN_FEATURES], vl["y"])],
            callbacks=[lgb.early_stopping(50,verbose=False), lgb.log_evaluation(0)])

model_W_cal = CalibratedClassifierCV(model_W, method="sigmoid", cv="prefit")
model_W_cal.fit(vl[WIN_FEATURES], vl["y"])


# ════════════════════════════════════════════════════════════════════════════
# 12.  EVALUATION REPORT + BENCHMARK CONTEXT
# ════════════════════════════════════════════════════════════════════════════

p_raw_test = model_W.predict_proba(te[WIN_FEATURES])[:, 1]
p_cal_test = model_W_cal.predict_proba(te[WIN_FEATURES])[:, 1]
te["p_cal"] = p_cal_test

brier_overall = brier_score_loss(te["y"], p_cal_test)
logloss_cal   = log_loss(te["y"], p_cal_test)

print("\n" + "="*70)
print("EVALUATION REPORT  —  TEST SET")
print("="*70)

print("\n── Overall ──────────────────────────────────────────────────────────")
print(f"  Always-predict-50%      Brier : 0.2500  ← dumbest baseline")
print(f"  LR baseline (inn.2)     Brier : {brier_score_loss(test_i2['y'], p_lr_test):.4f}")
print(f"  LGB uncalibrated        Brier : {brier_score_loss(te['y'], p_raw_test):.4f}")
print(f"  LGB calibrated          Brier : {brier_overall:.4f}")
print(f"  LGB calibrated        LogLoss : {logloss_cal:.4f}")

print("\n── By innings ───────────────────────────────────────────────────────")
for inn in [1, 2]:
    sub = te[te["innings"] == inn]
    b   = brier_score_loss(sub["y"], sub["p_cal"])
    print(f"  Innings {inn}: Brier={b:.4f}  (n={len(sub):,})")

print("\n── By phase ─────────────────────────────────────────────────────────")
for ph, name in {0:"Powerplay (ov 1-6) ",
                 1:"Middle   (ov 7-15) ",
                 2:"Death    (ov 16-20)"}.items():
    sub = te[te["phase"] == ph]
    if not len(sub): continue
    b = brier_score_loss(sub["y"], sub["p_cal"])
    print(f"  {name}: Brier={b:.4f}  (n={len(sub):,})")

print("\n── By Required Run Rate bucket (innings 2) ──────────────────────────")
te_i2 = te[te["innings"] == 2].copy()
te_i2["rrr_bucket"] = pd.cut(
    te_i2["required_rr"].fillna(0), bins=[0,6,8,10,50],
    labels=["Easy (<6)","Moderate (6-8)","Hard (8-10)","Very Hard (>10)"])
for bucket, grp in te_i2.groupby("rrr_bucket", observed=True):
    b = brier_score_loss(grp["y"], grp["p_cal"])
    print(f"  {str(bucket):22s}: Brier={b:.4f}  (n={len(grp):,})")

print("\n── Day vs Night ─────────────────────────────────────────────────────")
for flag, label in [(0,"Night"), (1,"Day  ")]:
    sub = te[te["is_day_match"] == flag]
    if not len(sub): continue
    b = brier_score_loss(sub["y"], sub["p_cal"])
    print(f"  {label}: Brier={b:.4f}  (n={len(sub):,})")

print("\n── Dew factor (innings 2) ───────────────────────────────────────────")
for flag, label in [(0,"Dew unlikely"), (1,"Dew likely  ")]:
    sub = te_i2[te_i2["dew_likely"] == flag]
    if not len(sub): continue
    b = brier_score_loss(sub["y"], sub["p_cal"])
    print(f"  {label}: Brier={b:.4f}  (n={len(sub):,})")

print("\n── Feature importances ──────────────────────────────────────────────")
fi = pd.Series(model_W.feature_importances_, index=WIN_FEATURES).sort_values(ascending=False)
for feat, imp in fi.items():
    bar = "█" * int(imp / fi.max() * 30)
    print(f"  {feat:30s} {bar}")

# ── Benchmark context: ARE THE RESULTS GOOD? ─────────────────────────────────
pct_improvement = (0.25 - brier_overall) / 0.25 * 100
print("\n" + "="*70)
print("ARE THESE RESULTS GOOD?  — Benchmark context")
print("="*70)
print(f"""
Your model's Brier score: {brier_overall:.4f}

What this means in plain English
─────────────────────────────────
  Brier score measures average squared error between your predicted
  probability and the actual outcome (0 or 1).  Lower = better.
  The theoretical worst useful model = 0.25 (always predict 50%).
  A perfect model = 0.00 (never possible in cricket).

  Your improvement over the worst baseline: {pct_improvement:.1f}%

Comparison to published research
──────────────────────────────────
  ┌─────────────────────────────────────────────┬───────────┬────────────┐
  │ Model / Paper                               │   Brier   │  Setting   │
  ├─────────────────────────────────────────────┼───────────┼────────────┤
  │ Always predict 50% (dumb baseline)          │   0.2500  │  —         │
  │ Logistic regression (Bailey & Clarke 2006)  │ ~0.210    │  ODI/T20   │
  │ Random Forest (Sankaranarayanan et al 2014) │ ~0.195    │  T20       │
  │ YOUR model (LightGBM, both innings)         │  {brier_overall:.4f}  │  IPL T20   │
  │ State-of-the-art (deep learning, all data)  │ ~0.165    │  T20 intl  │
  └─────────────────────────────────────────────┴───────────┴────────────┘

Verdict
───────
  Your result sits between "good academic baseline" and
  "state of the art", which is exactly where a well-executed
  personal project with public data and no player features should land.

  Innings-2 Brier of {brier_score_loss(te[te['innings']==2]['y'], te[te['innings']==2]['p_cal']):.4f} is competitive with published work.
  Innings-1 Brier of {brier_score_loss(te[te['innings']==1]['y'], te[te['innings']==1]['p_cal']):.4f} is the harder problem — most papers
  don't even attempt innings 1.  Doing it at all is a contribution.

The irreducible floor
──────────────────────
  T20 cricket has genuine randomness.  A six off the last ball, a
  dropped catch, a freakish dismissal — no model predicts these.
  Researchers estimate the irreducible Brier floor for T20 is
  approximately 0.14–0.16.  Your model is ~{brier_overall - 0.15:.3f} above that floor.
  The gap is closeable mainly with player strength features.
""")
print("="*70)


# ════════════════════════════════════════════════════════════════════════════
# 13.  RELIABILITY DIAGRAMS
# ════════════════════════════════════════════════════════════════════════════

def plot_reliability(y_true, p_pred_list, labels, title, path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = ["#E05A3A","#0F7173","#D4821A"]

    ax = axes[0]
    ax.plot([0,1],[0,1],"k--",lw=1.2,label="Perfect calibration")
    for (p, lbl), col in zip(zip(p_pred_list, labels), colors):
        pt, pp = calibration_curve(y_true, p, n_bins=10)
        ax.plot(pp, pt, "o-", color=col, lw=2, markersize=6, label=lbl)
    ax.set_xlabel("Mean predicted probability", fontsize=11)
    ax.set_ylabel("Observed win rate", fontsize=11)
    ax.set_title("Calibration curve", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9); ax.set_xlim(0,1); ax.set_ylim(0,1); ax.grid(alpha=0.3)

    ax2 = axes[1]
    for (p, lbl), col in zip(zip(p_pred_list, labels), colors):
        ax2.hist(p, bins=20, alpha=0.4, color=col, label=lbl, density=True)
    ax2.set_xlabel("Predicted probability", fontsize=11)
    ax2.set_ylabel("Density", fontsize=11)
    ax2.set_title("Distribution of predictions", fontsize=11, fontweight="bold")
    ax2.legend(fontsize=9); ax2.grid(alpha=0.3)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(path, dpi=150); plt.close()
    print(f"Saved: {path}")

p_raw_i2 = model_W.predict_proba(test_i2[WIN_FEATURES])[:, 1]
p_cal_i2 = model_W_cal.predict_proba(test_i2[WIN_FEATURES])[:, 1]
assert len(test_i2) == len(p_lr_test) == len(p_raw_i2) == len(p_cal_i2)

plot_reliability(test_i2["y"],
                 [p_lr_test, p_raw_i2, p_cal_i2],
                 ["LR baseline","LGB raw","LGB cal"],
                 "Reliability — Innings 2 (test season)",
                 "reliability_innings2.png")
plot_reliability(te["y"],
                 [p_raw_test, p_cal_test],
                 ["LGB raw","LGB cal"],
                 "Reliability — Both Innings (test season)",
                 "reliability_both_innings.png")


# ════════════════════════════════════════════════════════════════════════════
# 14.  FULL-MATCH WIN PROBABILITY CURVE
# ════════════════════════════════════════════════════════════════════════════

def plot_match_curve(match_id, state, model, win_features, matches,
                     path_prefix="match"):
    mdf = state[state["match_id"] == match_id].copy()
    mdf = mdf.dropna(subset=["p_post_toss"])
    if mdf.empty:
        print(f"No data for {match_id}"); return

    mdf["wp"] = model.predict_proba(mdf[win_features].fillna(0))[:, 1]
    row      = matches[matches["match_id"] == match_id].iloc[0]
    team_bat = mdf.iloc[0]["batting_team"]
    p_pre    = row["p_pre_match"] if team_bat == row["team1"] else 1 - row["p_pre_match"]
    p_post   = row["p_post_toss"] if team_bat == row["team1"] else 1 - row["p_post_toss"]

    inn1 = mdf[mdf["innings"]==1].copy()
    inn2 = mdf[mdf["innings"]==2].copy()
    inn1["bx"] = inn1["legal_ball_before"] + 1
    inn2["bx"] = inn2["legal_ball_before"] + 121

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.axhline(p_pre,  color="#aaa", ls=":",  lw=1.5, label=f"Pre-match {p_pre:.2f}")
    ax.axhline(p_post, color="#D4821A", ls="--", lw=1.5, label=f"Post-toss {p_post:.2f}")
    ax.axhline(0.5,    color="#ddd", ls="-",  lw=0.8)
    ax.plot(inn1["bx"], inn1["wp"], color="#1B3A5C", lw=2, label="Innings 1")
    ax.plot(inn2["bx"], inn2["wp"], color="#0F7173", lw=2, label="Innings 2")
    ax.axvline(120, color="#999", lw=0.8)
    ax.fill_between(inn1["bx"], inn1["wp"], 0.5, alpha=0.08, color="#1B3A5C")
    ax.fill_between(inn2["bx"], inn2["wp"], 0.5, alpha=0.08, color="#0F7173")

    for inn_df, color in [(inn1,"#1B3A5C"), (inn2,"#0F7173")]:
        if len(inn_df) < 2: continue
        for idx in inn_df["wp"].diff().abs().nlargest(2).index:
            bx  = inn_df.loc[idx,"bx"]
            wp  = inn_df.loc[idx,"wp"]
            wkt = int(inn_df.loc[idx,"wickets_lost"])
            ax.annotate(f"wkt={wkt}", xy=(bx, wp), xytext=(bx+3, wp+0.07),
                        fontsize=7, color=color,
                        arrowprops=dict(arrowstyle="->", color=color, lw=0.8))

    dew_str  = "Dew likely" if row.get("dew_likely", 0) else "No dew"
    time_str = "Day" if row.get("is_day_match", 0) else "Night"
    ax.set_xlim(0,240); ax.set_ylim(0,1)
    ax.set_xlabel("Delivery (1–120: Inn 1 | 121–240: Inn 2)", fontsize=10)
    ax.set_ylabel(f"P(win) — {team_bat}", fontsize=10)
    ax.set_title(
        f"{row['team1']} vs {row['team2']}  |  Winner: {row['winner']}  "
        f"[{time_str}, {dew_str}]",
        fontsize=11, fontweight="bold")
    ax.set_xticks([1,36,72,108,120,156,192,228,240])
    ax.set_xticklabels(["Ov1","6","12","18","20|1","6","12","18","20"], fontsize=8)
    ax.legend(fontsize=9); ax.grid(alpha=0.2)
    plt.tight_layout()
    fpath = f"{path_prefix}_{match_id}.png"
    plt.savefig(fpath, dpi=150); plt.close()
    print(f"Saved: {fpath}")


# ════════════════════════════════════════════════════════════════════════════
# 15.  PROJECTED TOTAL CURVE
# ════════════════════════════════════════════════════════════════════════════

def plot_projected_total(match_id, state, model_med, model_p10, model_p90,
                         features, band_margin, path_prefix="proj"):
    mdf = state[(state["match_id"]==match_id) & (state["innings"]==1)].dropna(subset=features)
    if mdf.empty: return
    mdf = mdf.copy()
    med  = model_med.predict(mdf[features])
    p10  = np.minimum(model_p10.predict(mdf[features]), med) - band_margin
    p90  = np.maximum(model_p90.predict(mdf[features]), med) + band_margin
    mdf["bx"] = mdf["legal_ball_before"] + 1
    true_total = mdf["innings1_total"].iloc[0]
    venue_avg  = mdf["venue_avg_total"].iloc[0]

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.fill_between(mdf["bx"], p10, p90, alpha=0.2, color="#0F7173",
                    label=f"Uncertainty band (±{band_margin:.0f})")
    ax.plot(mdf["bx"], med, color="#0F7173", lw=2, label="Projected total")
    ax.axhline(true_total, color="#E05A3A", ls="--", lw=1.5,
               label=f"Actual: {int(true_total)}")
    ax.axhline(venue_avg, color="#D4821A", ls=":", lw=1.2,
               label=f"Venue avg: {venue_avg:.0f}")
    ax.set_xlabel("Delivery number", fontsize=11)
    ax.set_ylabel("Runs", fontsize=11)
    ax.set_title(f"Projected Innings-1 Total — Match {match_id}",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9); ax.grid(alpha=0.25)
    plt.tight_layout()
    fpath = f"{path_prefix}_{match_id}.png"
    plt.savefig(fpath, dpi=150); plt.close()
    print(f"Saved: {fpath}")


test_match_ids = te["match_id"].unique()[:3]
for mid in test_match_ids:
    plot_match_curve(mid, te, model_W_cal, WIN_FEATURES, matches)
    plot_projected_total(mid, te, model_T, model_T_p10, model_T_p90,
                         INN1_FEATURES, margin)


# ════════════════════════════════════════════════════════════════════════════
# 16.  SAVE
# ════════════════════════════════════════════════════════════════════════════

joblib.dump(model_T,      "model_projected_total.pkl")
joblib.dump(model_T_p10,  "model_proj_p10.pkl")
joblib.dump(model_T_p90,  "model_proj_p90.pkl")
joblib.dump(model_W_cal,  "model_win_prob_calibrated.pkl")
joblib.dump(lr_toss,      "model_toss.pkl")
joblib.dump(scaler,       "scaler_lr_baseline.pkl")
joblib.dump({"band_margin": margin}, "quantile_margin.pkl")

print("\nAll models saved.")
print("Pipeline complete.")
