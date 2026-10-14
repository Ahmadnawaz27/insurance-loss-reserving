"""
Chain-ladder, its validation against actual outcomes, and Bornhuetter-Ferguson (BF).

Chain-ladder: age-to-age factors from the known triangle, project the missing cells, reserve = projected final minus paid to date.
BF: paid so far + premium x expected loss ratio x share still unpaid. The expected loss ratio is the pooled chain-ladder ultimate over earned premium for the five oldest accident years, so nothing from the future leaks in.

Use as: import chain_ladder as cl. Run directly for a single-company demo (Secura, GRCODE 965).
"""

import sqlite3
import pandas as pd

DB_PATH = "reserving.db"
GRCODE = 965                # Secura Insurance Company, used for the demo
N_OLDEST_FOR_ELR = 5


# ---------------------------------------------------------------- chain-ladder
def load_triangle(db_path: str, grcode: int) -> pd.DataFrame:
    """Cumulative paid loss known at year-end 2007: accident year x development lag."""
    conn = sqlite3.connect(db_path)
    df = pd.read_sql(f"""
        SELECT AccidentYear, DevelopmentLag, CumPaidLoss
        FROM claims_features
        WHERE GRCODE = {grcode} AND is_known = 1;
    """, conn)
    conn.close()
    return df.pivot(index="AccidentYear", columns="DevelopmentLag", values="CumPaidLoss")


def age_to_age_factors(triangle: pd.DataFrame) -> dict:
    factors = {}
    cols = sorted(triangle.columns)
    for j, j_next in zip(cols[:-1], cols[1:]):
        both_known = triangle[j].notna() & triangle[j_next].notna()
        factors[j] = triangle[j_next][both_known].sum() / triangle[j][both_known].sum()
    return factors


def project_triangle(triangle: pd.DataFrame, factors: dict) -> pd.DataFrame:
    projected = triangle.copy()
    cols = sorted(triangle.columns)
    for ay in projected.index:
        for j, j_next in zip(cols[:-1], cols[1:]):
            if pd.isna(projected.loc[ay, j_next]) and not pd.isna(projected.loc[ay, j]):
                projected.loc[ay, j_next] = projected.loc[ay, j] * factors[j]
    return projected


def reserve_summary(triangle: pd.DataFrame, projected: pd.DataFrame) -> pd.DataFrame:
    last_col = max(triangle.columns)
    latest_known = triangle.apply(lambda row: row.dropna().iloc[-1], axis=1)
    ultimate = projected[last_col]
    return pd.DataFrame({
        "latest_known": latest_known,
        "projected_ultimate": ultimate,
        "reserve": ultimate - latest_known,
    })


# ------------------------------------------------------------------ validation
def load_actual_ultimates(db_path: str, grcode: int) -> pd.Series:
    """The true final cumulative paid loss (lag 10), from the full uncensored data."""
    conn = sqlite3.connect(db_path)
    df = pd.read_sql(f"""
        SELECT AccidentYear, CumPaidLoss
        FROM claims_features
        WHERE GRCODE = {grcode} AND DevelopmentLag = 10;
    """, conn)
    conn.close()
    return df.set_index("AccidentYear")["CumPaidLoss"]


def compare(projected_ultimate: pd.Series, actual_ultimate: pd.Series) -> pd.DataFrame:
    out = pd.DataFrame({"projected_ultimate": projected_ultimate,
                        "actual_ultimate": actual_ultimate})
    out["error"] = out["projected_ultimate"] - out["actual_ultimate"]
    out["pct_error"] = (out["error"] / out["actual_ultimate"] * 100).round(2)
    return out


# --------------------------------------------------------- Bornhuetter-Ferguson
def load_premium(db_path: str, grcode: int) -> pd.Series:
    conn = sqlite3.connect(db_path)
    df = pd.read_sql(f"""
        SELECT AccidentYear, EarnedPremNet
        FROM claims_features
        WHERE GRCODE = {grcode} AND DevelopmentLag = 1;
    """, conn)
    conn.close()
    return df.set_index("AccidentYear")["EarnedPremNet"]


def cumulative_dev_factors(factors: dict, last_lag: int) -> dict:
    """Factor from each lag to final: multiply that lag's factor and all later ones."""
    cdf = {last_lag: 1.0}
    for lag in range(last_lag - 1, 0, -1):
        cdf[lag] = factors[lag] * cdf[lag + 1]
    return cdf


def expected_loss_ratio(projected_ultimate: pd.Series, premium: pd.Series, n_oldest: int) -> float:
    oldest = sorted(premium.index)[:n_oldest]
    return projected_ultimate[oldest].sum() / premium[oldest].sum()


def bornhuetter_ferguson(triangle, cdf, premium, elr) -> pd.Series:
    out = {}
    for ay in triangle.index:
        known = triangle.loc[ay].dropna()
        latest_lag, latest_paid = known.index[-1], known.iloc[-1]
        share_unpaid = 1 - 1 / cdf[latest_lag]
        out[ay] = latest_paid + premium[ay] * elr * share_unpaid
    return pd.Series(out)


# ------------------------------------------------------------------- demo (965)
if __name__ == "__main__":
    triangle = load_triangle(DB_PATH, GRCODE)
    factors = age_to_age_factors(triangle)
    projected = project_triangle(triangle, factors)
    last_lag = max(triangle.columns)
    cl_ultimate = projected[last_lag]

    print("== 1. Chain-ladder ==")
    print("Development factors:", {k: round(float(v), 5) for k, v in factors.items()})
    summary = reserve_summary(triangle, projected)
    print("\nReserve summary:\n", summary.round(0))
    print(f"\nTotal chain-ladder reserve: {summary['reserve'].sum():.0f}")

    print("\n== 2. Validation against what actually happened ==")
    actual = load_actual_ultimates(DB_PATH, GRCODE)
    comparison = compare(cl_ultimate, actual)
    print(comparison.round(0))
    total_pct = (comparison["projected_ultimate"].sum() / comparison["actual_ultimate"].sum() - 1) * 100
    print(f"\nTotal projected ultimate: {comparison['projected_ultimate'].sum():,.0f}")
    print(f"Total actual ultimate:    {comparison['actual_ultimate'].sum():,.0f}")
    print(f"Overall error: {total_pct:.2f}%")
    worst = comparison["pct_error"].abs().idxmax()
    print(f"Worst single-year miss: accident year {worst} ({comparison.loc[worst, 'pct_error']:.1f}%)")

    print("\n== 3. Bornhuetter-Ferguson vs chain-ladder ==")
    premium = load_premium(DB_PATH, GRCODE)
    cdf = cumulative_dev_factors(factors, last_lag)
    elr = expected_loss_ratio(cl_ultimate, premium, N_OLDEST_FOR_ELR)
    print(f"Expected loss ratio (from the {N_OLDEST_FOR_ELR} oldest years): {elr:.3f}\n")
    bf_ultimate = bornhuetter_ferguson(triangle, cdf, premium, elr)
    table = pd.DataFrame({"CL": cl_ultimate, "BF": bf_ultimate, "actual": actual})
    table["CL_err_%"] = ((table["CL"] - table["actual"]) / table["actual"] * 100).round(1)
    table["BF_err_%"] = ((table["BF"] - table["actual"]) / table["actual"] * 100).round(1)
    print(table.round(0))
    print("\nTotal error:  CL {:.2f}%   BF {:.2f}%".format(
        (table["CL"].sum() / table["actual"].sum() - 1) * 100,
        (table["BF"].sum() / table["actual"].sum() - 1) * 100))
    print("Average miss per year (absolute %):  CL {:.2f}%   BF {:.2f}%".format(
        table["CL_err_%"].abs().mean(), table["BF_err_%"].abs().mean()))