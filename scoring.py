"""
Scoring helpers shared by 05_scorecard.py and the app. Plain pandas and numpy only.

Only accident years that still had money to pay at the 2007 valuation are scored. Accident year 1998 is already settled, so every method scores a perfect 0 on it and including it would flatter all of them. Set SCORE_SETTLED_YEAR = True to score all 950 rows.

Definitions (about the projected final paid amount at lag 10):
  miss                 absolute % error of one projection
  company miss         a company's average miss over its scored accident years
  average / typical    mean / median of the company misses
  dollar-weighted      sum of absolute dollar errors / sum of actual final amounts
  total error          sum of projected finals / sum of actual finals - 1, signed
  net reserve error    (projected finals - actual finals) / money still to be paid
                       (actual final - paid to date); negative means too little reserved
"""

import numpy as np
import pandas as pd

LAST_LAG = 10
SCORE_SETTLED_YEAR = False

METHODS = ["CL", "BF", "ML", "ML_HIST", "CORR"]          # the main five
ROBUSTNESS = ["ML_HIST_NO_N", "CL_BIAS"]                 # reference rows, not part of the headline
NAMES = {
    "CL": "Chain-ladder", "BF": "Bornhuetter-Ferguson", "ML": "ML, base features",
    "ML_HIST": "ML + company history", "CORR": "Chain-ladder + ML correction",
    "ML_HIST_NO_N": "ML + history, no hist_n", "CL_BIAS": "Chain-ladder + lag bias",
}
SHORT = {
    "CL": "Chain-ladder", "BF": "Born.-Ferguson", "ML": "ML, base", "ML_HIST": "ML + history",
    "CORR": "CL + ML correction", "ML_HIST_NO_N": "ML + history (no n)", "CL_BIAS": "CL + lag bias",
}
AGE_ORDER = ["old", "middle", "young"]
AGE_LABELS = {"old": "Old years (1999-2002)", "middle": "Middle years (2003-2004)",
              "young": "Young years (2005-2007)"}


def scored(res: pd.DataFrame) -> pd.DataFrame:
    """The rows that count as predictions (see the note at the top). Safe to call twice."""
    if SCORE_SETTLED_YEAR:
        return res
    if "latest_lag" not in res:
        raise KeyError("results need a latest_lag column (re-run 02 and 04)")
    return res[res["latest_lag"] < LAST_LAG]


def scope(res: pd.DataFrame) -> dict:
    """Counts for the captions: companies, scored projections, rows left out."""
    s = scored(res)
    return {"companies": int(s["GRCODE"].nunique()), "projections": len(s), "left_out": len(res) - len(s)}


def age_group(accident_year) -> np.ndarray:
    ay = np.asarray(accident_year)
    return np.where(ay >= 2005, "young", np.where(ay <= 2002, "old", "middle"))


def with_age(res: pd.DataFrame) -> pd.DataFrame:
    out = scored(res).copy()
    out["age"] = age_group(out["AccidentYear"])
    return out


def company_miss(res: pd.DataFrame, methods) -> pd.DataFrame:
    """One row per company: average absolute % error over its scored accident years, per method."""
    return scored(res).groupby("GRCODE").agg(**{m: (f"{m}_err", lambda s: s.abs().mean()) for m in methods})


def bootstrap_vs_cl(per: pd.DataFrame, methods, n_boot: int = 5000, seed: int = 0) -> pd.DataFrame:
    """Paired bootstrap over companies of (method miss - chain-ladder miss), mean company miss.
    One random stream, methods in the given order (so adding methods at the end changes nothing)."""
    rng = np.random.default_rng(seed)
    rows = {}
    for m in methods:
        d = (per[m] - per["CL"]).to_numpy()
        bs = [rng.choice(d, len(d)).mean() for _ in range(n_boot)]
        rows[m] = (d.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5))
    return pd.DataFrame(rows, index=["diff", "lo", "hi"]).T


def bootstrap_pair(per: pd.DataFrame, a: str, b: str, n_boot: int = 5000, seed: int = 0):
    """Paired bootstrap over companies of (a miss - b miss): (difference, ci_low, ci_high)."""
    rng = np.random.default_rng(seed)
    d = (per[a] - per[b]).to_numpy()
    bs = [rng.choice(d, len(d)).mean() for _ in range(n_boot)]
    return d.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def net_reserve_error(res: pd.DataFrame, methods) -> pd.Series:
    """Whole-book net reserve error, % of the money still to be paid. NaN if paid_to_date is absent."""
    res = scored(res)
    if "paid_to_date" not in res:
        return pd.Series(np.nan, index=methods)
    need = (res["actual"] - res["paid_to_date"]).sum()
    return pd.Series({m: ((res[m] - res["actual"]).sum() / need) * 100 for m in methods})


def still_to_pay(res: pd.DataFrame) -> float:
    res = scored(res)
    return float((res["actual"] - res["paid_to_date"]).sum())


def scorecard(res: pd.DataFrame, methods=METHODS, n_boot: int = 5000, seed: int = 0):
    """Every headline number, one row per method. Returns (scorecard, per-company misses)."""
    assert "CL" in methods
    res = with_age(res)
    per = company_miss(res, methods)
    boot = bootstrap_vs_cl(per, methods, n_boot, seed) if n_boot else None      # n_boot=0 skips the intervals
    reserve = net_reserve_error(res, methods)
    rows = []
    for m in methods:
        row = {
            "method": m,
            "mean_company_miss_pct": per[m].mean(),
            "median_company_miss_pct": per[m].median(),
            "worst_company_miss_pct": per[m].max(),
            "beats_chain_ladder_pct": (per[m] < per["CL"]).mean() * 100 if m != "CL" else np.nan,
            "dollar_weighted_miss_pct": (res[m] - res.actual).abs().sum() / res.actual.sum() * 100,
            "total_error_pct": (res[m].sum() / res.actual.sum() - 1) * 100,
            "net_reserve_error_pct": reserve[m],
            "vs_chain_ladder_pts": (per[m] - per["CL"]).mean() if m != "CL" else np.nan,
            "vs_cl_ci_low": boot.loc[m, "lo"] if (m != "CL" and boot is not None) else np.nan,
            "vs_cl_ci_high": boot.loc[m, "hi"] if (m != "CL" and boot is not None) else np.nan,
        }
        for a in AGE_ORDER:
            row[f"{a}_years_median_miss_pct"] = res.loc[res["age"] == a, f"{m}_err"].abs().median()
        row["young_years_bias_pct"] = res.loc[res["age"] == "young", f"{m}_err"].median()
        rows.append(row)
    return pd.DataFrame(rows).set_index("method"), per


def error_by_accident_year(res: pd.DataFrame, methods, stat: str = "bias") -> pd.DataFrame:
    """Median signed error (bias) or median absolute error (miss), accident year x method."""
    cols = {m: f"{m}_err" for m in methods}
    g = scored(res).groupby("AccidentYear")[list(cols.values())]
    out = g.median() if stat == "bias" else g.agg(lambda s: s.abs().median())
    return out.rename(columns={v: k for k, v in cols.items()})


def error_spread(res: pd.DataFrame, methods, qs=(0.1, 0.25, 0.5, 0.75, 0.9)) -> pd.DataFrame:
    """Quantiles of the signed % error by accident-year age group and method (long format)."""
    res = with_age(res)
    rows = []
    for a in AGE_ORDER:
        sub = res[res["age"] == a]
        for m in methods:
            e = sub[f"{m}_err"]
            rows.append({"age": a, "method": m, "n": len(e),
                         **{f"q{int(q * 100)}": e.quantile(q) for q in qs}})
    return pd.DataFrame(rows)


def company_table(res: pd.DataFrame, companies: pd.DataFrame = None, methods=METHODS) -> pd.DataFrame:
    """Per-company average miss for each method, with size tier, average premium and the best method.
    With `companies` (GRCODE, name, size_tier, avg_premium) the name, tier and premium come from it, which keeps
    the tiers fixed when `res` is only a slice of the data. Without it they are worked out from `res`."""
    per = company_miss(res, methods).reset_index()
    if companies is not None:
        out = companies[["GRCODE", "name", "size_tier", "avg_premium"]].merge(per, on="GRCODE")
    else:
        out = per.merge(size_tiers(res).rename("size_tier").reset_index(), on="GRCODE")
        out = out.merge(res.groupby("GRCODE")["premium"].mean().rename("avg_premium").reset_index(), on="GRCODE")
    out["best_method"] = out[list(methods)].idxmin(axis=1)
    return out


def size_table(res: pd.DataFrame, methods=METHODS, companies: pd.DataFrame = None) -> pd.DataFrame:
    """Mean company miss by company size tier (small / medium / large), one row per tier."""
    t = company_table(res, companies, methods)
    t["size_tier"] = t["size_tier"].astype(str)
    g = t.groupby("size_tier")
    out = g[list(methods)].mean()
    out["companies"] = g.size()
    return out.reindex([x for x in ["small", "medium", "large"] if x in out.index])


def size_tiers(res: pd.DataFrame) -> pd.Series:
    """small / medium / large: terciles of each company's average yearly premium,
    over all ten accident years. Needs the unfiltered results (the premium column)."""
    avg = res.groupby("GRCODE")["premium"].mean()
    return pd.qcut(avg, 3, labels=["small", "medium", "large"])