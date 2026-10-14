"""
Step 2: chain-ladder and Bornhuetter-Ferguson (BF) for every company with a full 10x10 history, scored against the actual final paid amount.

Accident year 1998 is already fully paid at the 2007 valuation, so it stays in multi_company_results.csv (latest_lag = 10) but is not scored.
"""

import sqlite3
import numpy as np
import pandas as pd
import chain_ladder as cl

DB_PATH = "reserving.db"


def full_history_companies(db_path: str) -> list:
    conn = sqlite3.connect(db_path)
    rows = conn.execute("""
        SELECT GRCODE FROM claims GROUP BY GRCODE HAVING COUNT(*) = 100
    """).fetchall()
    conn.close()
    return [r[0] for r in rows]


def run_company(db_path: str, grcode: int):
    """Returns a per-accident-year DataFrame, or None if the company can't be modelled."""
    triangle = cl.load_triangle(db_path, grcode)
    premium = cl.load_premium(db_path, grcode)
    actual = cl.load_actual_ultimates(db_path, grcode)

    # Skip companies where the maths would divide by zero or go negative
    if (premium <= 0).any() or (actual <= 0).any() or (triangle.iloc[:, 0] <= 0).any():
        return None
    factors = cl.age_to_age_factors(triangle)
    if not all(np.isfinite(v) and v > 0 for v in factors.values()):
        return None

    projected = cl.project_triangle(triangle, factors)
    last_lag = max(triangle.columns)
    cl_ult = projected[last_lag]
    cdf = cl.cumulative_dev_factors(factors, last_lag)
    elr = cl.expected_loss_ratio(cl_ult, premium, cl.N_OLDEST_FOR_ELR)
    bf_ult = cl.bornhuetter_ferguson(triangle, cdf, premium, elr)

    out = pd.DataFrame({"CL": cl_ult, "BF": bf_ult, "actual": actual})
    out["latest_lag"] = triangle.notna().sum(axis=1)            # lags known at the 2007 valuation
    out["paid_to_date"] = cl.reserve_summary(triangle, projected)["latest_known"]
    out["CL_err"] = (out["CL"] - out["actual"]) / out["actual"] * 100
    out["BF_err"] = (out["BF"] - out["actual"]) / out["actual"] * 100
    out["GRCODE"] = grcode
    out.index.name = "AccidentYear"
    return out.reset_index()


if __name__ == "__main__":
    codes = full_history_companies(DB_PATH)
    frames, skipped = [], 0
    for g in codes:
        res = run_company(DB_PATH, g)
        if res is None:
            skipped += 1
        else:
            frames.append(res)
    allres = pd.concat(frames, ignore_index=True)
    allres = allres[["GRCODE", "AccidentYear", "latest_lag", "paid_to_date", "actual",
                     "CL", "BF", "CL_err", "BF_err"]]
    allres.to_csv("multi_company_results.csv", index=False)
    scored = allres[allres["latest_lag"] < 10]          # 1998 is already fully paid: nothing to project

    n_comp = allres["GRCODE"].nunique()
    print(f"Companies with full history: {len(codes)}; modelled: {n_comp}; skipped: {skipped}")
    print(f"Company-years: {len(allres)}; scored: {len(scored)} "
          f"(accident year 1998 is already fully paid at the 2007 valuation, so its "
          f"{len(allres) - len(scored)} rows are not predictions)\n")

    per_co = scored.groupby("GRCODE").agg(
        CL_mape=("CL_err", lambda s: s.abs().mean()),
        BF_mape=("BF_err", lambda s: s.abs().mean()),
    )
    print("Average miss per year, per company (absolute %):")
    print(per_co.describe().loc[["mean", "50%", "max"]].round(2))
    print(f"\nBF beats CL in {(per_co['BF_mape'] < per_co['CL_mape']).mean() * 100:.0f}% of companies")

    scored = scored.assign(age=np.where(scored["AccidentYear"] >= 2005, "young (2005-07)",
                           np.where(scored["AccidentYear"] <= 2002, "old (1999-02)", "middle (2003-04)")))
    by_age = scored.groupby("age").agg(
        CL_median_abs=("CL_err", lambda s: s.abs().median()),
        BF_median_abs=("BF_err", lambda s: s.abs().median()),
        CL_bias=("CL_err", "median"),
        BF_bias=("BF_err", "median"),
    )
    print("\nBy accident-year age (median abs error %, and median signed error % = bias):")
    print(by_age.round(2))