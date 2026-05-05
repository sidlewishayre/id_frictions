import os
import sys
import pandas as pd

CWD = os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir))

sys.path.append("/home/sidlh/Documents/reusable_code")
from other_utils.pandas_utils import clean_col_names

intermediary_map = {
    "banks": [6021, 6022, 6029, 6035, 6036, 6099],
    "broker_dealers": [6211],
    "non_banks": [
        6111,
        6141,
        6153,
        6159,
        6162,
        6163,
        6172,
        6189,
        6199,
        6282,
        6311,
        6321,
        6324,
        6331,
        6351,
        6361,
        6399,
        6411,
        # 6500,
        # 6510,
        # 6512,
        # 6513,
        # 6519,
        # 6531,
        # 6532,
        # 6552,
        # 6770,
        # 6792,
        # 6794,
        # 6795,
        # 6798,
        # 6799,
        # 8880,
        6221,
        6200,
    ],
}


def get_sic_classification():
    intermediary_map_rev = {
        code: category for category, codes in intermediary_map.items() for code in codes
    }
    df = pd.read_excel(os.path.join(CWD, "get_data", "sic_codes.xlsx"))
    df = clean_col_names(df)
    df["intermediary_classification"] = df["sic_code"].map(intermediary_map_rev)
    assert (~df["intermediary_classification"].isna()).sum() == len(
        intermediary_map_rev
    )
    finance_bool = df["office"].str.lower().str.find("finance") != -1
    assert df[finance_bool]["intermediary_classification"].isnull().sum() == 0
    assert df[(~finance_bool) & df["intermediary_classification"].notna()].shape[0] == 2
    return df[df["intermediary_classification"].notnull()].set_index("sic_code")[
        "intermediary_classification"
    ]
