"""Convert the Stanek & Ones (2023) database xlsx into the pickle the pipeline reads.

Download the database from the Data Repository for the University of Minnesota
(https://doi.org/10.13020/7YS8-ET75; CC BY-NC-ND 3.0 US) and place the file
"Stanek and Ones - Personality Traits and Cognitive Abilities Relations Database 2023-05-16.xlsx"
in database/. The database is not redistributed with this repository.

Output: database/db.pkl (60,690 rows x 20 columns, sheet "Public Database").
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DB_DIR = ROOT / "database"


def main() -> None:
    xlsx = sorted(DB_DIR.glob("Stanek and Ones - Personality Traits and Cognitive Abilities Relations Database*.xlsx"))
    xlsx = [p for p in xlsx if "Legend" not in p.name]
    if not xlsx:
        raise SystemExit("Place the Stanek & Ones database xlsx in database/ (see database/README.md).")
    db = pd.read_excel(xlsx[0], sheet_name="Public Database")
    db.to_pickle(DB_DIR / "db.pkl")
    print(f"Read {xlsx[0].name}: {db.shape[0]} effect sizes x {db.shape[1]} columns -> database/db.pkl")


if __name__ == "__main__":
    main()
