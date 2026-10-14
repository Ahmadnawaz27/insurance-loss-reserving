"""
Step 3: test the ML variants one step ahead, before any projection.

A. Calendar split: train on targets up to 2006, test on the 2007 diagonal.
B. Rolling origin: test diagonals 2005, 2006 and 2007. The variant used downstream (ml_core.CHOSEN) is picked from this table by a rule fixed in advance, never from step 4 scores.

Writes app_data/one_step_results.csv and app_data/feature_importance.csv.
"""

import os
import sqlite3
import numpy as np
import pandas as pd
import ml_core as ml

APP_DIR = "app_data"
IMPORTANCE_MODELS = ["ML base features", "ML + history: hist+n", ml.CHOSEN]

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)


def build_and_check_view() -> None:
    conn = sqlite3.connect(ml.DB_PATH)
    ml.build_training_view(conn)
    n, n_co = conn.execute("SELECT COUNT(*), COUNT(DISTINCT GRCODE) FROM ml_training").fetchone()
    print(f"ml_training: {n} rows from {n_co} companies")
    first = conn.execute("SELECT COUNT(*) FROM ml_training WHERE prior_growth IS NULL").fetchone()[0]
    print(f"rows with no prior_growth (development lag 1, nothing before it): {first}")
    latest = conn.execute("SELECT MAX(target_dev_year) FROM ml_training").fetchone()[0]
    print(f"leakage check, latest year any target cell comes from: {latest} (must be 2007)")
    assert latest <= 2007, "a future cell leaked into the training table"
    print("\naverage next-step growth by development lag (what the model has to learn):")
    for lag, rows, g in conn.execute("""
        SELECT DevelopmentLag, COUNT(*), ROUND(AVG(target_growth), 4)
        FROM ml_training GROUP BY DevelopmentLag ORDER BY DevelopmentLag"""):
        print(f"  lag {lag}: {rows} rows, {g}")
    conn.close()


def permutation_importance(spec: dict, data: pd.DataFrame, year: int = 2007,
                           repeats: int = 20, seed: int = 0) -> pd.DataFrame:
    """How much worse does the one-step miss get if one input is shuffled?
    Train on targets before `year`, test on diagonal `year`. For the correction models
    only the model's inputs are shuffled; the chain-ladder factor it corrects stays put."""
    kind, feats = spec["kind"], spec["feats"]
    train = data[data["target_dev_year"] < year]
    test = data[(data["target_dev_year"] == year) & data["log_cl"].notna()]
    t = train if kind == "direct" else train[train["resid"].notna()]
    model = ml.make_model().fit(t[feats], t["y" if kind == "direct" else "resid"])
    base = test["log_cl"].to_numpy() if kind == "resid" else 0.0
    y = test["y"].to_numpy()

    def miss(X):                                     # mean one-step miss, % of next payment
        return np.abs(y - (base + model.predict(X))).mean() * 100

    ref = miss(test[feats])
    rng = np.random.default_rng(seed)
    rows = []
    for col in feats:
        inc = []
        for _ in range(repeats):
            X = test[feats].copy()
            X[col] = rng.permutation(X[col].to_numpy())
            inc.append(miss(X) - ref)
        rows.append({"feature": col, "importance": np.mean(inc), "std": np.std(inc), "baseline_miss": ref})
    return pd.DataFrame(rows)


def feasibility() -> None:
    print("\nWhy the correction is learned one step at a time, not on the final amount:")
    print("  only accident year 1998 has a settled final amount (lag 10) by 2007, and")
    print("  chain-ladder's last factor (lag 9 to 10) is first observed in 2007, so a")
    print("  chain-ladder ULTIMATE cannot be rebuilt as of 2006. Nothing to train on.")


if __name__ == "__main__":
    build_and_check_view()
    data = ml.load_training_data()
    assert data["target_dev_year"].max() <= 2007

    # ---- A: calendar split ----
    train = data[data["target_dev_year"] <= 2006]
    test = data[data["target_dev_year"] == 2007]
    assert train["target_dev_year"].max() < test["target_dev_year"].min()
    print(f"\n== A. Calendar split: train {len(train)} rows (targets up to 2006), "
          f"test {len(test)} rows (2007 diagonal) ==")
    model = ml.make_model().fit(train[ml.FEATURES], train["y"])
    preds = {
        "pooled average": ml.pooled_baseline(train, test),
        "per-company (chain-ladder idea)": ml.per_company_baseline(train, test),
        "ML model": model.predict(test[ml.FEATURES]),
    }
    print("Average miss on next-step payment, % (lower is better):")
    print(pd.DataFrame({k: ml.score(test["y"], p, test["DevelopmentLag"])
                        for k, p in preds.items()}).round(2))

    # ---- B: rolling origin ----
    res = ml.rolling_eval(data)
    tbl = res["all (mean)"].unstack()
    tbl["average"] = tbl.mean(axis=1)
    print("\n== B. Rolling-origin test, mean one-step miss % by test diagonal ==")
    print(tbl.round(2).sort_values("average"))
    print("\n2007 diagonal detail:")
    print(res.xs(2007, level=1)[["all (mean)", "all (median)", "lag 1-2 (mean)",
                                 "lag 3-5 (mean)", "lag 6-9 (mean)", "n"]].round(2))
    picked = ml.select_variant(res)
    print("\nSelected by the pre-declared rule:", picked)
    assert picked == ml.CHOSEN, f"update ml_core.CHOSEN to {picked!r}"

    # ---- C. which inputs do the models use? (permutation test on the 2007 diagonal) ----
    imp = []
    for name in IMPORTANCE_MODELS:
        one = permutation_importance(ml.SPECS[name], data)
        ref_2007 = res.loc[(name, 2007), "all (mean)"]
        assert abs(one["baseline_miss"].iloc[0] - ref_2007) < 1e-9, "importance model differs from rolling_eval"
        imp.append(one.assign(model=name))
    imp = pd.concat(imp, ignore_index=True)
    print("\n== C. Rise in one-step miss (percentage points) when one input is shuffled ==")
    print(imp.pivot(index="feature", columns="model", values="importance").round(3))

    os.makedirs(APP_DIR, exist_ok=True)
    out = res.reset_index().rename(columns={"level_0": "variant", "level_1": "test_year"})
    out.to_csv(f"{APP_DIR}/one_step_results.csv", index=False)
    imp.to_csv(f"{APP_DIR}/feature_importance.csv", index=False)
    print(f"\nwrote {APP_DIR}/one_step_results.csv and {APP_DIR}/feature_importance.csv")
    feasibility()