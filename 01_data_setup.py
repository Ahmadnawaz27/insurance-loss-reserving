"""Step 1: load the CSV into SQLite, run data-quality checks, build the claims_features view.
 
Output: reserving.db (table claims, view claims_features). Safe to re-run.
"""

import sqlite3
import pandas as pd

CSV_PATH = "ppauto_98-07.csv"
DB_PATH = "reserving.db"

CHECKS = {
    "row count per company (flags anyone without the full 100 rows)": """
        SELECT GRCODE, GRNAME, COUNT(*) AS row_count
        FROM claims
        GROUP BY GRCODE, GRNAME
        HAVING row_count <> 100
        ORDER BY row_count
        LIMIT 10;
    """,
    "nulls in the key numeric columns": """
        SELECT
            SUM(CASE WHEN CumPaidLoss IS NULL THEN 1 ELSE 0 END) AS null_paid,
            SUM(CASE WHEN IncurredLosses IS NULL THEN 1 ELSE 0 END) AS null_incurred,
            SUM(CASE WHEN EarnedPremNet IS NULL THEN 1 ELSE 0 END) AS null_premium
        FROM claims;
    """,
    "duplicate (company, accident year, development lag) rows": """
        SELECT GRCODE, AccidentYear, DevelopmentLag, COUNT(*) AS cnt
        FROM claims
        GROUP BY GRCODE, AccidentYear, DevelopmentLag
        HAVING cnt > 1
        LIMIT 10;
    """,
    "paid loss exceeding incurred loss (a sanity flag, not always an error)": """
        SELECT COUNT(*) AS flagged_rows
        FROM claims
        WHERE CumPaidLoss > IncurredLosses;
    """,
    "confirm Secura (965) has the full history": """
        SELECT COUNT(*) AS secura_rows
        FROM claims
        WHERE GRCODE = 965;
    """,
}

# is_known: would this row have been known at year-end 2007?
# The file holds the full square (development out to 2016), so this flag
# is what stops later steps from peeking at the future.
CREATE_FEATURES_VIEW = """
CREATE VIEW claims_features AS
SELECT
    GRCODE,
    GRNAME,
    AccidentYear,
    DevelopmentYear,
    DevelopmentLag,
    IncurredLosses,
    CumPaidLoss,
    BulkLoss,
    EarnedPremNet,
    PostedReserves2007,
    CASE WHEN DevelopmentYear <= 2007 THEN 1 ELSE 0 END AS is_known
FROM claims;
"""


def load_csv(conn: sqlite3.Connection, csv_path: str = CSV_PATH) -> None:
    df = pd.read_csv(csv_path)
    df.to_sql("claims", conn, if_exists="replace", index=False)
    n = conn.execute("SELECT COUNT(*) FROM claims").fetchone()[0]
    print(f"Loaded {n} rows into table 'claims'")
    print("Columns:", [c[1] for c in conn.execute("PRAGMA table_info(claims)").fetchall()])


def run_checks(conn: sqlite3.Connection) -> None:
    for label, sql in CHECKS.items():
        print(f"\n-- {label} --")
        cur = conn.execute(sql)
        rows = cur.fetchall()
        print([d[0] for d in cur.description])
        for r in rows:
            print(r)


def build_features_view(conn: sqlite3.Connection) -> None:
    conn.execute("DROP VIEW IF EXISTS claims_features;")
    conn.execute(CREATE_FEATURES_VIEW)

    print("\n-- Secura, accident year 1998, first rows --")
    for r in conn.execute("""
        SELECT AccidentYear, DevelopmentLag, DevelopmentYear, CumPaidLoss, is_known
        FROM claims_features
        WHERE GRCODE = 965 AND AccidentYear = 1998
        ORDER BY DevelopmentLag LIMIT 4;
    """):
        print(r)

    split = dict(conn.execute("""
        SELECT is_known, COUNT(*) FROM claims_features WHERE GRCODE = 965 GROUP BY is_known;
    """).fetchall())
    print(f"\nSecura known vs future rows: {split}  (should be 55 known, 45 future)")
    assert split == {0: 45, 1: 55}, "known/future split is not 55/45"


if __name__ == "__main__":
    conn = sqlite3.connect(DB_PATH)
    load_csv(conn)
    run_checks(conn)
    build_features_view(conn)
    conn.commit()
    conn.close()