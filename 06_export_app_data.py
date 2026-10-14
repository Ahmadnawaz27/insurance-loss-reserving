"""
Step 6: export the data the Streamlit app needs, so it runs from app_data/ alone (no database, no scikit-learn). Run after step 4.

Writes companies.csv, triangles.csv and run_info.json. Amounts are in $000, as in the NAIC file.
"""

import json
import sqlite3
import pandas as pd
import ml_core as ml
import scoring

DB_PATH = "reserving.db"
APP_DIR = "app_data"


if __name__ == "__main__":
    res = pd.read_csv(f"{APP_DIR}/ultimate_results.csv")
    codes = sorted(res["GRCODE"].unique())
    conn = sqlite3.connect(DB_PATH)

    names = pd.read_sql("SELECT DISTINCT GRCODE, GRNAME AS name FROM claims", conn)
    names["name"] = names["name"].str.strip()
    avg_premium = res.groupby("GRCODE")["premium"].mean().rename("avg_premium")
    companies = (names[names["GRCODE"].isin(codes)]
                 .merge(avg_premium.reset_index(), on="GRCODE")
                 .merge(scoring.size_tiers(res).rename("size_tier").reset_index(), on="GRCODE")
                 .sort_values("name").reset_index(drop=True))
    assert len(companies) == 95 and companies["GRCODE"].is_unique
    companies.to_csv(f"{APP_DIR}/companies.csv", index=False)

    tri = pd.read_sql(f"""
        SELECT GRCODE, AccidentYear, DevelopmentLag, DevelopmentYear, CumPaidLoss, EarnedPremNet, is_known
        FROM claims_features
        WHERE GRCODE IN ({",".join(str(int(c)) for c in codes)})
        ORDER BY GRCODE, AccidentYear, DevelopmentLag
    """, conn)
    assert len(tri) == 95 * 100 and tri.groupby("GRCODE").size().eq(100).all()
    assert tri["is_known"].sum() == 95 * 55                  # 55 known cells per company at year-end 2007
    tri.to_csv(f"{APP_DIR}/triangles.csv", index=False)

    scope = scoring.scope(res)
    info = {
        "valuation_year": 2007,
        "accident_years": [int(res["AccidentYear"].min()), int(res["AccidentYear"].max())],
        "companies_in_file": int(conn.execute("SELECT COUNT(DISTINCT GRCODE) FROM claims").fetchone()[0]),
        "companies_full_history": int(conn.execute(
            "SELECT COUNT(*) FROM (SELECT GRCODE FROM claims GROUP BY GRCODE HAVING COUNT(*) = 100)").fetchone()[0]),
        "companies_modelled": scope["companies"],
        "projections_scored": scope["projections"],
        "settled_rows_left_out": scope["left_out"],
        "training_steps": int(len(ml.load_training_data())),
        "still_to_pay_thousands": scoring.still_to_pay(res),
        "units": "thousands of US dollars",
    }
    conn.close()
    with open(f"{APP_DIR}/run_info.json", "w") as f:
        json.dump(info, f, indent=2)
    print(json.dumps(info, indent=2))
    print(f"\nwrote {APP_DIR}/companies.csv ({len(companies)} rows), triangles.csv ({len(tri)} rows), run_info.json")
