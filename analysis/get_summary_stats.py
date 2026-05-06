import os
import sys
import pandas as pd

CWD = os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir))

sys.path.append(CWD)
from settings import PROD_DATA, SUMMARY_STATS

sys.path.append("/home/sidlh/Documents/reusable_code")
from summary_stats import run_summary_stats

# df = pd.read_csv(PROD_DATA)

run_summary_stats(
    folder=SUMMARY_STATS,
    data=PROD_DATA,
    panels={
        "Balance Sheet": ["assets", "debt", "equity", "return"],
        "Constructed Variables": [
            "euler_equation",
            "leverage",
            "net_worth",
            "var_pct",
        ],
    },
    fns=["mean", "std", "min", "q25", "q50", "q75", "max"],
    fn_names=[
        "Mean",
        "Std Dev",
        "Min",
        "25th Pctile",
        "50th Pctile",
        "75th Pctile",
        "Max",
    ],
    # group_var="bank",
    var_map={
        "assets": "Assets (\\$)",
        "debt": "Debt (\\$)",
        "equity": "Equity (\\$)",
        "return": "Return (\\%)",
        "euler_equation": "Euler Equation",
        "leverage": "Leverage",
        "net_worth": "Net Worth",
        "var_pct": "Value at Risk",
    },
    open_pdf=True,
    # all_name="Intermediary Type",
)
