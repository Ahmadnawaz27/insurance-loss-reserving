"""
Step 4: project every accident year to its final paid amount with each method, then score against what actually happened.

Methods: CL (chain-ladder), BF, ML (base features), ML_HIST (adds company history), CORR (chain-ladder plus an ML correction, the variant chosen in step 3). Robustness checks: ML_HIST_NO_N and CL_BIAS.

An ML projection starts from the latest known paid amount, predicts the next growth step, and repeats to lag 10. Training uses only steps known at the 2007 valuation, and features for future steps come from older accident years only.

Accident year 1998 is already settled, so it is not scored (see scoring.py). A built-in check confirms the projector reproduces step 2's chain-ladder results.

Writes ultimate_results.csv, projection_paths.csv and method_vs_cl_by_lag.csv to app_data/.
"""

import os
import numpy as np
import pandas as pd
import chain_ladder as cl
import ml_core as ml
import scoring

DB_PATH = "reserving.db"
BENCHMARK_CSV = "multi_company_results.csv"
APP_DIR = "app_data"
LAST_LAG = 10

METHODS = {                                   # the ML-based methods; CL and BF come from step 2
    "ML":           ml.SPECS["ML base features"],
    "ML_HIST":      ml.SPECS["ML + history: hist+n"],
    "CORR":         ml.SPECS[ml.CHOSEN],
    "ML_HIST_NO_N": ml.SPECS["ML + history: hist only"],
    "CL_BIAS":      ml.SPECS["CL + lag bias"],
}
ORDER = ["CL", "BF"] + list(METHODS)
PATH_METHODS = ["CL", "ML", "ML_HIST", "CORR"]          # the methods the app draws as curves
DIFF_METHODS = ["ML", "ML_HIST", "CORR"]                # compared with chain-ladder, step by step


# ------------------------------------------------------------ projection engine
def starting_states(db_path: str, codes) -> pd.DataFrame:
    """One row per company and accident year: where it stands at the 2007 valuation."""
    rows = []
    for g in codes:
        tri = cl.load_triangle(db_path, g)
        prem = cl.load_premium(db_path, g)
        for ay in tri.index:
            known = tri.loc[ay].dropna()
            lag = int(known.index[-1])
            rows.append({
                "GRCODE": g, "AccidentYear": int(ay), "lag": lag,
                "paid": float(known.iloc[-1]),
                "prev_paid": float(tri.loc[ay, lag - 1]) if lag > 1 else np.nan,
                "premium": float(prem[ay]),
            })
    return pd.DataFrame(rows)


def build_lookup(data):
    """Per (company, lag): the known steps, used to build features for future steps."""
    lk = {}
    for (c, j), g in data.groupby(["GRCODE", "DevelopmentLag"]):
        g = g.sort_values("AccidentYear")
        lk[(c, j)] = (g["AccidentYear"].to_numpy(), g["c_now"].to_numpy(),
                      g["c_next"].to_numpy(), g["y"].to_numpy())
    return lk


def step_feats(lk, c, j, a):
    """Same as ml.add_cl_features, for a future step: older accident years only."""
    t = lk.get((c, j))
    if t is None:
        return np.nan, np.nan, np.nan, 0
    ay, cn, cx, y = t
    m = ay < a
    n = int(m.sum())
    if n == 0:
        return np.nan, np.nan, np.nan, 0
    return (np.log(cx[m].sum() / cn[m].sum()),
            y[m].std(ddof=1) if n >= 2 else np.nan, y[m].mean(), n)


def project(state, lk, resid_by_ay, predictor):
    """Roll every row forward one lag at a time until lag 10.
    Returns the final paid amounts and one record per step taken: which row of
    `state`, the lag it started from, the correction against chain-ladder's
    log factor, and the paid amount after the step."""
    G = state["GRCODE"].to_numpy(); A = state["AccidentYear"].to_numpy()
    lag = state["lag"].to_numpy().copy(); paid = state["paid"].to_numpy().copy()
    prev = state["prev_paid"].to_numpy().copy(); prem = state["premium"].to_numpy()
    rp = np.full(len(state), np.nan); cd = np.zeros(len(state))
    for i in range(len(state)):                       # deviations already observed
        r = resid_by_ay.get((G[i], A[i]), np.array([]))
        rp[i] = r[-1] if len(r) else np.nan
        cd[i] = r.sum() if len(r) else 0.0
    steps = []
    while (lag < LAST_LAG).any():
        idx = np.where(lag < LAST_LAG)[0]
        f = np.array([step_feats(lk, G[i], lag[i], A[i]) for i in idx])
        X = pd.DataFrame({
            "DevelopmentLag": lag[idx], "paid_to_premium": paid[idx] / prem[idx],
            "log_prior_growth": np.log(paid[idx] / np.where(prev[idx] > 0, prev[idx], np.nan)),
            "log_premium": np.log(prem[idx]), "log_cl": f[:, 0], "cl_disp": f[:, 1],
            "hist_growth": f[:, 2], "hist_n": f[:, 3], "resid_prev": rp[idx], "cum_dev": cd[idx],
        })
        assert X["log_cl"].notna().all(), "a projected step has no chain-ladder factor"
        lg = np.asarray(predictor(X))
        resid = lg - X["log_cl"].to_numpy()
        prev[idx] = paid[idx]; paid[idx] = paid[idx] * np.exp(lg)
        steps.append(pd.DataFrame({"row": idx, "lag": lag[idx], "corr": resid, "paid_after": paid[idx]}))
        rp[idx] = resid; cd[idx] = cd[idx] + resid; lag[idx] += 1
    return paid, pd.concat(steps, ignore_index=True)


# -------------------------------------------------------------- what gets saved
def build_paths(state, steps_by_method) -> pd.DataFrame:
    """Long table of lag-by-lag paid amounts: each method's curve starts at the latest
    known cell (the same point for every method) and runs to lag 10."""
    frames = []
    for m, st in steps_by_method.items():
        start = pd.DataFrame({"GRCODE": state["GRCODE"], "AccidentYear": state["AccidentYear"],
                              "method": m, "DevelopmentLag": state["lag"], "paid": state["paid"]})
        who = state.iloc[st["row"].to_numpy()]
        later = pd.DataFrame({"GRCODE": who["GRCODE"].to_numpy(), "AccidentYear": who["AccidentYear"].to_numpy(),
                              "method": m, "DevelopmentLag": st["lag"].to_numpy() + 1,
                              "paid": st["paid_after"].to_numpy()})
        frames += [start, later]
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(["GRCODE", "AccidentYear", "method", "DevelopmentLag"]).reset_index(drop=True)


def method_vs_cl_by_lag(steps_by_method) -> pd.DataFrame:
    """For each ML method and each step (lag j to j+1): how much its predicted growth
    factor differs from chain-ladder's for the same company and step, in % of the factor."""
    rows = []
    for m in DIFF_METHODS:
        st = steps_by_method[m].assign(pct=lambda d: (np.exp(d["corr"]) - 1) * 100)
        g = st.groupby("lag")["pct"]
        rows.append(pd.DataFrame({
            "method": m, "from_lag": g.mean().index, "mean_pct": g.mean().to_numpy(),
            "median_pct": g.median().to_numpy(), "p10_pct": g.quantile(0.1).to_numpy(),
            "p90_pct": g.quantile(0.9).to_numpy(), "n": g.size().to_numpy()}))
    return pd.concat(rows, ignore_index=True)


if __name__ == "__main__":
    pd.set_option("display.width", 200); pd.set_option("display.max_columns", 30)
    os.makedirs(APP_DIR, exist_ok=True)
    res = pd.read_csv(BENCHMARK_CSV)[["GRCODE", "AccidentYear", "latest_lag", "paid_to_date",
                                      "actual", "CL", "BF", "CL_err", "BF_err"]]

    data = ml.load_training_data()
    assert data["target_dev_year"].max() <= 2007, "future data in the training table"
    print(f"trained on {len(data)} steps from {data['GRCODE'].nunique()} companies "
          f"(latest target year {data['target_dev_year'].max()})")
    lk = build_lookup(data)
    resid_by_ay = {k: g.sort_values("DevelopmentLag")["resid"].to_numpy()
                   for k, g in data.groupby(["GRCODE", "AccidentYear"])}
    state0 = starting_states(DB_PATH, res["GRCODE"].unique())
    key = state0[["GRCODE", "AccidentYear"]]

    # state0 (built from the database) must agree with the starting point step 2 recorded
    chk = state0.merge(res, on=["GRCODE", "AccidentYear"])
    assert len(chk) == len(res) == 950
    assert (chk["lag"] == chk["latest_lag"]).all() and np.allclose(chk["paid"], chk["paid_to_date"])

    def run(spec):
        return project(state0, lk, resid_by_ay, ml.fit_spec(spec, data))

    # chain-ladder through this projector must reproduce step 2 exactly
    u_cl, steps_cl = run(ml.SPECS["CL one-step (baseline)"])
    chk = key.assign(x=u_cl).merge(res, on=["GRCODE", "AccidentYear"])
    d = ((chk["x"] - chk["CL"]).abs() / chk["CL"]).max()
    print(f"check: projector with chain-ladder factors vs step 2 chain-ladder, max relative difference {d:.2e}")
    assert d < 1e-9

    steps = {"CL": steps_cl}
    for tag, spec in METHODS.items():
        u, st = run(spec)
        res = res.merge(key.assign(**{tag: u}), on=["GRCODE", "AccidentYear"])
        res[f"{tag}_err"] = (res[tag] - res["actual"]) / res["actual"] * 100
        steps[tag] = st
    assert len(res) == 950
    res = res.merge(state0[["GRCODE", "AccidentYear", "premium"]], on=["GRCODE", "AccidentYear"])
    res = res[["GRCODE", "AccidentYear", "premium", "latest_lag", "paid_to_date", "actual"]
              + ORDER + [f"{m}_err" for m in ORDER]]
    res.to_csv(f"{APP_DIR}/ultimate_results.csv", index=False)

    paths = build_paths(state0, {m: steps[m] for m in PATH_METHODS})
    ends = paths.loc[paths.groupby(["GRCODE", "AccidentYear", "method"])["DevelopmentLag"].idxmax()]
    ends = ends.pivot(index=["GRCODE", "AccidentYear"], columns="method", values="paid")
    check = res.set_index(["GRCODE", "AccidentYear"])
    for m in PATH_METHODS:                                   # every curve ends exactly at its ultimate
        assert np.allclose(ends[m], check.loc[ends.index, m], rtol=1e-12), m
    paths.to_csv(f"{APP_DIR}/projection_paths.csv", index=False)
    method_vs_cl_by_lag(steps).to_csv(f"{APP_DIR}/method_vs_cl_by_lag.csv", index=False)

    # ----------------------------------------------------------------- scoring
    sc, per = scoring.scorecard(res, ORDER)
    info = scoring.scope(res)
    print(f"\n=== {info['projections']} projections from {info['companies']} companies, scored once "
          f"({info['left_out']} already-paid rows of accident year 1998 left out) ===")
    tab = sc[["mean_company_miss_pct", "median_company_miss_pct", "worst_company_miss_pct",
              "beats_chain_ladder_pct", "dollar_weighted_miss_pct", "total_error_pct", "net_reserve_error_pct"]]
    tab.columns = ["mean co. miss", "median co. miss", "worst co.", "beats CL in %",
                   "dollar miss", "total error", "net reserve error"]
    print(tab.rename(index=scoring.NAMES).round(2))

    print("\nMedian miss % by accident-year age:")
    print(sc[[f"{a}_years_median_miss_pct" for a in scoring.AGE_ORDER]].set_axis(scoring.AGE_ORDER, axis=1).round(2))
    print("\nMedian signed error % for young years (bias; positive = over-projects):")
    print(sc["young_years_bias_pct"].round(2).to_string())

    print("\nPaired bootstrap over companies, difference in mean company miss vs chain-ladder "
          "(pts, negative = better):")
    for m in ORDER[1:]:
        r = sc.loc[m]
        print(f"  {m:13s} {r['vs_chain_ladder_pts']:+.3f}  95% CI [{r['vs_cl_ci_low']:+.3f}, {r['vs_cl_ci_high']:+.3f}]")
    print(f"\nwrote {APP_DIR}/ultimate_results.csv, projection_paths.csv, method_vs_cl_by_lag.csv")
