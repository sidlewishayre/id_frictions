import os
import sys
import pandas as pd

CWD = os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir))

sys.path.append(CWD)
from settings import PROD_DATA, SUMMARY_STATS_SIMPLE, SUMMARY_STATS_GROUP

sys.path.append("/home/sidlh/Documents/reusable_code")
from summary_stats import run_summary_stats

# df = pd.read_csv(PROD_DATA)

simple_kwargs = dict(
    folder=SUMMARY_STATS_SIMPLE,
)

group_kwargs = dict(
    folder=SUMMARY_STATS_GROUP,
    fns=["mean", "std", "q25", "q75"],
    fn_names=["Mean", "Std Dev", "25th Percentile", "75th Percentile"],
    group_var="bank",
    # all_name="Intermediary Type",
    latex_kwargs={
        "page_dimensions": [15, 5],
    },
)

kwargs = dict(
    data=PROD_DATA,
    panels=[
        {
            "id": "balance_sheet",
            "title": "Balance Sheet",
            "variables": ["assets", "debt", "equity", "return"],
        },
        {
            "id": "con_vars",
            "title": "Constructed Variables",
            "variables": [
                "euler_equation",
                "leverage",
                "net_worth",
                "var_pct",
            ],
        },
    ],
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
    # open_pdf=True,
)

for this_kwargs in [simple_kwargs, group_kwargs]:
    run_summary_stats(
        **{**kwargs, **this_kwargs},
    )
