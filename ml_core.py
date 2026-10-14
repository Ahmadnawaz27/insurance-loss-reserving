"""
Shared ML building blocks for steps 3 and 4. Use as: import ml_core as ml.

Each training row is one accident year of one company at development lag j. The target is the log of cumulative paid growth to lag j+1.

Leakage rules:
  - only cells known at the 2007 valuation (is_known = 1) are used, filtered before the SQL window functions so LEAD() cannot reach a future cell
  - company-history features use only older accident years at the same lag

Contents: build_training_view, load_training_data, add_cl_features, SPECS and fit_spec (the model variants), rolling_eval, select_variant (the pre-declared selection rule), baselines and score."""

import sqlite3
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

DB_PATH = "reserving.db"
BENCHMARK_CSV = "multi_company_results.csv"   # written by step 2: the 95 benchmark companies

BASE = ["DevelopmentLag", "paid_to_premium", "log_prior_growth", "log_premium"]
FEATURES = BASE                                # name used by older scripts
CL1 = BASE + ["log_cl"]
CL3 = CL1 + ["cl_disp", "resid_prev", "cum_dev"]

# ------------------------------------------------------------------ SQL view
CREATE_VIEW = """
CREATE VIEW ml_training AS
WITH known AS (
    SELECT
        GRCODE, AccidentYear, DevelopmentLag, DevelopmentYear, CumPaidLoss, EarnedPremNet,
        LAG(CumPaidLoss)  OVER w AS prior_paid,
        LEAD(CumPaidLoss) OVER w AS next_paid,
        LEAD(DevelopmentYear) OVER w AS next_dev_year
    FROM claims_features
    WHERE is_known = 1
    WINDOW w AS (PARTITION BY GRCODE, AccidentYear ORDER BY DevelopmentLag)
)
SELECT
    GRCODE, AccidentYear, DevelopmentLag,
    CumPaidLoss * 1.0 / EarnedPremNet                       AS paid_to_premium,
    CASE WHEN prior_paid > 0 THEN CumPaidLoss * 1.0 / prior_paid END AS prior_growth,
    EarnedPremNet                                           AS premium,
    next_paid * 1.0 / CumPaidLoss                           AS target_growth,
    next_dev_year                                           AS target_dev_year
FROM known
WHERE next_paid IS NOT NULL
  AND CumPaidLoss > 0
  AND EarnedPremNet > 0;
"""


def build_training_view(conn: sqlite3.Connection) -> None:
    conn.execute("DROP VIEW IF EXISTS ml_training;")
    conn.execute(CREATE_VIEW)
    conn.commit()


# ------------------------------------------------------------------ features
def add_cl_features(df: pd.DataFrame) -> pd.DataFrame:
    """All features use only older accident years at the same lag (shift(1))."""
    d = df.copy()
    d["c_now"] = d["paid_to_premium"] * d["premium"]
    d["c_next"] = d["c_now"] * d["target_growth"]
    d = d.sort_values(["GRCODE", "DevelopmentLag", "AccidentYear"])
    g = d.groupby(["GRCODE", "DevelopmentLag"])
    num = g["c_next"].transform(lambda s: s.cumsum().shift(1))
    den = g["c_now"].transform(lambda s: s.cumsum().shift(1))
    d["log_cl"] = np.log(num / den)                       # company's chain-ladder factor, logged
    d["cl_disp"] = g["y"].transform(lambda s: s.shift(1).expanding().std())   # spread of past factors
    d["hist_growth"] = g["y"].transform(lambda s: s.shift(1).expanding().mean())
    d["hist_n"] = g.cumcount()                            # how many older years back the above
    d["resid"] = d["y"] - d["log_cl"]                     # how far this step was from chain-ladder
    d = d.sort_values(["GRCODE", "AccidentYear", "DevelopmentLag"])
    ga = d.groupby(["GRCODE", "AccidentYear"])
    d["resid_prev"] = ga["resid"].shift(1)                # this year's previous deviation
    d["cum_dev"] = ga["resid"].transform(lambda s: s.cumsum().shift(1))   # all deviations so far
    d.loc[d["DevelopmentLag"] == 1, "cum_dev"] = 0.0
    return d.sort_index()


def load_training_data(db_path: str = DB_PATH, csv_path: str = BENCHMARK_CSV) -> pd.DataFrame:
    codes = pd.read_csv(csv_path)["GRCODE"].unique()
    conn = sqlite3.connect(db_path)
    if not conn.execute("SELECT 1 FROM sqlite_master WHERE name = 'ml_training'").fetchone():
        build_training_view(conn)
    df = pd.read_sql("SELECT * FROM ml_training", conn)
    conn.close()
    df = df[df["GRCODE"].isin(codes)].copy()
    df["log_premium"] = np.log(df["premium"])
    df["log_prior_growth"] = np.log(df["prior_growth"])    # stays empty at lag 1
    df["y"] = np.log(df["target_growth"])
    return add_cl_features(df.reset_index(drop=True))


# --------------------------------------------------------------------- model
def make_model() -> HistGradientBoostingRegressor:
    # absolute_error loss: a few wild rows (growth of 100x off a tiny payment)
    # should not drag the whole model around
    return HistGradientBoostingRegressor(
        loss="absolute_error", learning_rate=0.05, max_iter=300,
        max_depth=4, min_samples_leaf=20, random_state=0,
    )


# Variants. "direct" predicts log growth; "resid" predicts the correction to
# the chain-ladder log factor; "bias" is chain-ladder plus an average per-lag
# correction.
SPECS = {
    "CL one-step (baseline)":       dict(kind="cl"),
    "ML base features":             dict(kind="direct", feats=BASE),
    "ML + history: hist+n":         dict(kind="direct", feats=BASE + ["hist_growth", "hist_n"]),
    "ML + history: hist only":      dict(kind="direct", feats=BASE + ["hist_growth"]),
    "Stack S3: + CL deviations":    dict(kind="direct", feats=CL3),
    "Residual R1: + CL factor":     dict(kind="resid", feats=CL1),
    "Residual R3: + CL deviations": dict(kind="resid", feats=CL3),
    "CL + lag bias":                dict(kind="bias"),
}
REFERENCES = ["CL one-step (baseline)", "ML base features", "ML + history: hist+n", "ML + history: hist only"]
CHOSEN = "Residual R3: + CL deviations"   # what select_variant picks; 03 asserts it still does


def fit_spec(spec: dict, train: pd.DataFrame):
    """Returns a function: feature table -> predicted log growth."""
    kind = spec["kind"]
    if kind == "cl":
        return lambda X: X["log_cl"].to_numpy()
    f = spec.get("feats")
    if kind == "direct":
        m = make_model().fit(train[f], train["y"])
        return lambda X: m.predict(X[f])
    t = train[train["resid"].notna()]
    if kind == "resid":
        m = make_model().fit(t[f], t["resid"])
        return lambda X: X["log_cl"].to_numpy() + m.predict(X[f])
    if kind == "bias":
        bias = t.groupby("DevelopmentLag")["resid"].mean().reindex(range(1, 10)).ffill().fillna(0.0)
        return lambda X: X["log_cl"].to_numpy() + X["DevelopmentLag"].map(bias).to_numpy()
    raise ValueError(kind)


# ------------------------------------------------------------ tests and scores
def per_lag_average(train: pd.DataFrame) -> pd.Series:
    """
    Average log growth for each lag 1-9. Lag 9 (the step into lag 10) has no
    training rows in the 2006 split, because the first such step happens in
    2007, the test year. Those lags reuse the nearest earlier lag's average.
    """
    return train.groupby("DevelopmentLag")["y"].mean().reindex(range(1, 10)).ffill()


def pooled_baseline(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    return test["DevelopmentLag"].map(per_lag_average(train)).to_numpy()


def per_company_baseline(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    per_co = train.groupby(["GRCODE", "DevelopmentLag"])["y"].mean()
    pooled = per_lag_average(train)
    return np.array([per_co.get((g, lag), pooled[lag])      # fall back if no history
                     for g, lag in zip(test["GRCODE"], test["DevelopmentLag"])])


def score(y_true: pd.Series, y_pred: np.ndarray, lags: pd.Series) -> dict:
    err = np.abs(y_true.to_numpy() - y_pred) * 100       # approx % error on next payment
    bucket = pd.cut(lags, [0, 2, 5, 9], labels=["lag 1-2", "lag 3-5", "lag 6-9"])
    out = {"all (mean)": err.mean(), "all (median)": np.median(err)}
    for b in ["lag 1-2", "lag 3-5", "lag 6-9"]:
        out[b + " (mean)"] = err[(bucket == b).to_numpy()].mean()
    return out


def rolling_eval(data: pd.DataFrame, years=(2005, 2006, 2007)) -> pd.DataFrame:
    """Train on targets before Y, test on the diagonal Y, for each Y.
    Scored only on steps where a chain-ladder factor exists (not accident year 1998)."""
    rows = {}
    for name, spec in SPECS.items():
        for Y in years:
            train = data[data["target_dev_year"] < Y]
            test = data[(data["target_dev_year"] == Y) & data["log_cl"].notna()]
            assert train["target_dev_year"].max() < Y
            pred = fit_spec(spec, train)(test)
            rows[(name, Y)] = {**score(test["y"], pred, test["DevelopmentLag"]), "n": len(test)}
    return pd.DataFrame(rows).T


def select_variant(res: pd.DataFrame) -> str:
    """Pre-declared rule: lowest average (over the 3 folds) mean one-step miss,
    among everything that is not a reference or baseline."""
    avg = res["all (mean)"].unstack().mean(axis=1)
    return avg.drop(REFERENCES).idxmin()