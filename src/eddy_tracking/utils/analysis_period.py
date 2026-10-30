from typing import Any, cast

import pandas as pd


# First and last day of base.time.analysis_date_range, both inclusive
def read_analysis_period(cfg: dict[str, Any]) -> tuple[pd.Timestamp, pd.Timestamp]:
    start, end = cfg["base"]["time"]["analysis_date_range"]
    return cast(pd.Timestamp, pd.Timestamp(start)), cast(pd.Timestamp, pd.Timestamp(end))


# A table with birth_date and death_date keeps each eddy whose track overlaps the period. A table with date keeps only the rows dated inside it. A table with both does both, so a long-lived eddy stays but loses its composites outside the period.
def restrict_to_analysis_period(table: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    start, end = read_analysis_period(cfg)
    keep = pd.Series(True, index=table.index)
    if {"birth_date", "death_date"} <= set(table.columns):
        keep &= table["birth_date"].le(end) & table["death_date"].ge(start)
    if "date" in table.columns:
        keep &= table["date"].between(start, end)
    return table.loc[keep].copy()
