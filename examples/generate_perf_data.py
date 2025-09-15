#!/usr/bin/env python3
"""
Generate synthetic epidemiological-like datasets for performance testing.

Outputs CSV files with configurable number of rows and a schema resembling
`tests/test_data/sample_messy_data.csv`, including:
- study_id (string with prefix/suffix)
- date_of_birth (mixed formats, invalids, and missing values)
- patient_age (ints, strings, and NA markers)
- test_result (categorical with NA markers)
- empty_column (mostly empty to simulate constants)
- constant_value (constant/near-constant column)
- duplicate_group (to allow duplicate rows)

Usage:
  python examples/generate_perf_data.py --rows 1000 --out tests/test_data/perf_1000.csv
  python examples/generate_perf_data.py --rows 4000 --out tests/test_data/perf_4000.csv
"""

import argparse
import csv
import random
import sys
from datetime import datetime, timedelta

RANDOM_SEED = 42

DOB_FORMATS = [
    "%Y-%m-%d",
    "%d/%m/%Y",
    "%b %d, %Y",
]

NA_STRINGS = ["-99", "N/A", "", "NULL", "missing", "unknown"]
TEST_RESULTS = ["positive", "negative", "inconclusive", NA_STRINGS[0], NA_STRINGS[1]]


def random_date(start: datetime, end: datetime) -> str:
    delta = end - start
    days = random.randrange(delta.days + 1)
    dt = start + timedelta(days=days)
    fmt = random.choice(DOB_FORMATS)
    return dt.strftime(fmt)


def maybe_na(value: str, prob: float = 0.05) -> str:
    return random.choice(NA_STRINGS) if random.random() < prob else value


def random_age(prob_na: float = 0.05) -> str:
    r = random.random()
    if r < prob_na:
        return random.choice(NA_STRINGS)
    if r < 0.15:
        return random.choice(["twenty", "thirty", "forty", "fifty"])
    return str(random.randint(0, 100))


def generate_rows(n: int):
    start = datetime(1950, 1, 1)
    end = datetime(2010, 12, 31)

    rows = []
    for i in range(1, n + 1):
        prefix = "PS"
        num = f"{i:03d}"
        suffix = "P2"
        study_id = f"{prefix}{num}{suffix}"

        # Some duplicates: repeat every 200th record from the previous row
        if i % 200 == 0 and rows:
            rows.append(rows[-1].copy())
            continue

        dob = random_date(start, end)
        # Introduce some invalid dates
        if i % 123 == 0:
            dob = "31/02/2000"
        dob = maybe_na(dob, prob=0.03)

        age = random_age(prob_na=0.04)
        result = maybe_na(random.choice(TEST_RESULTS), prob=0.03)

        empty_col = "" if random.random() < 0.85 else random.choice(NA_STRINGS)
        constant_value = "same"

        rows.append(
            {
                "Study ID": study_id,
                "Date of Birth": dob,
                "Patient Age": age,
                "Test Result": result,
                "Empty Column": empty_col,
                "Constant Value": constant_value,
                "duplicate_group": (i // 50),
            }
        )
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, required=True, help="Number of rows to generate")
    parser.add_argument(
        "--out", type=str, required=True, help="Output CSV path (e.g., tests/test_data/perf_1000.csv)"
    )
    args = parser.parse_args(argv)

    random.seed(RANDOM_SEED)

    rows = generate_rows(args.rows)
    fieldnames = [
        "Study ID",
        "Date of Birth",
        "Patient Age",
        "Test Result",
        "Empty Column",
        "Constant Value",
        "duplicate_group",
    ]

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"Wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    sys.exit(main())
