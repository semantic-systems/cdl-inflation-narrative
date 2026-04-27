import pandas as pd
from bs4 import BeautifulSoup
import re
import os
from tqdm import tqdm
from pathlib import Path


INPUT_PATTERN = "../data/DJN/samples/no_filter/no_filter_{year}.csv"
OUTPUT_DIR    = "../data/DJN/samples/random_samples"
YEARS         = range(1984, 2024)          # 1984 – 2023 inclusive
SAMPLE_N      = 30
RANDOM_SEED   = 42


def strip_html(raw: str) -> str:
    """Remove all HTML tags and decode HTML entities."""
    if not isinstance(raw, str):
        return ""
    return BeautifulSoup(raw, "html.parser").get_text(separator=" ").strip()


def process_year(year: int):
    path = INPUT_PATTERN.format(year=year)
    if not os.path.exists(path):
        print(f"[SKIP] {path} not found")
        return None

    df = pd.read_csv(path, index_col=False)
    num_sample = len(df)

    # drop duplicates
    df = df.drop_duplicates(subset=['body'])
    num_deduplicated_sample = len(df)
    print(f"Number of duplicated sample: {num_sample-num_deduplicated_sample}")
    
    # drop docs with exclude code
    df = add_column_has_exclude_code(df)
    # Strip HTML from body
    df["body_clean"] = df["body"].apply(strip_html)

    # Concatenate headline + cleaned body
    df["text"] = (
        df["headline"].fillna("").str.strip()
        + " "
        + df["body_clean"]
    ).str.strip()

    # extract date from raw body
    df["date"] = pd.to_datetime(
        df["accession_number"].astype(str).str[:8], format="%Y%m%d", errors="coerce"
    )
    df.drop(columns=["body", "body_clean"], inplace=True)
    return df


def sample_by_day(df: pd.DataFrame, n: int = SAMPLE_N, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Draw up to n rows per calendar day; take all available if fewer than n."""
    if df["date"].isna().all():
        # No dates extracted — fall back to a flat sample
        return df.sample(min(n, len(df)), random_state=seed)

    groups = []
    for day, grp in df.groupby(df["date"].dt.date):
        groups.append(grp.sample(min(n, len(grp)), random_state=seed))

    return pd.concat(groups, ignore_index=True)

    
def add_column_has_exclude_code(df):
    include_codes = ["DJIB", "DJG", "GPRW", "DJAN", "AWSJ", "WSJE", "PREL", "NRG", "DJBN", "AWP", "BRNS", "JNL", "WAL",
                     "WLS", "WSJ"]
    exclude_codes = ["TAB", "TAN", "FTH", "CAL", "PRL"]
    has_exclude_code_list = []
    for i in range(len(df)):
        include = False
        for code in include_codes:
            if code in df["subject_code"].values[i]:
                include = True
        for code in exclude_codes:
            if code in df["subject_code"].values[i]:
                include = False
        has_exclude_code_list.append(include)
    df["has_exclude_code"] = has_exclude_code_list
    return df


def main():
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    for year in tqdm(YEARS):
        df = process_year(year)
        if df is None:
            continue

        sampled = sample_by_day(df)

        out_path = os.path.join(OUTPUT_DIR, f"{year}.csv")
        sampled.to_csv(out_path, index=False)
        print(f"[OK] {year}: {len(sampled):,} rows")


if __name__ == "__main__":
    main()