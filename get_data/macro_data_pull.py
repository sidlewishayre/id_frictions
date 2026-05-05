import os
import sys
import pandas as pd

CWD = os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir))

sys.path.append(CWD)
from settings import MIN_DATE

sys.path.append("/home/sidlh/Documents/reusable_code")
from database_management.fred_pull import get_fred_series_many
from other_utils.datetime_utils import quarter_collapse


def get_macro_data():
    df = get_fred_series_many(
        series_list=["GDPC1", "DFF", "VIXCLS"],
        pivot=True,
        series_names=["gdp", "ffr", "vix"],
    )
    df = df.reset_index()
    df = df.rename(columns={"asofdate": "date"})
    df = quarter_collapse(df.sort_values("date"), date_col="date", fn="last")
    df = df[df.index >= pd.to_datetime(MIN_DATE)]
    df = df.dropna()
    return df


if __name__ == "__main__":
    df = get_macro_data()
    df.to_csv(os.path.join(CWD, "data", "macro_data.csv"))
