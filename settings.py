import os

CWD = os.path.dirname(__file__)


# MODEL SETTINGS
BETA = 0.96
GAMMA = 2
FINANCIAL_FRICTIONS = ["leverage", "net_worth", "var_pct"]

# DATA SETTINGS
WRDS_USERNAME = "lucanadig"

ASSETS_COL_COMPUSTAT = ["atq"]
DEBT_COL_COMPUSTAT = ["dlcq", "dlttq"]
RETURN_COLS_CRSP = ["ret", "prc", "shrout"]
EXTRA_COLS = ["ceqq", "ltq"]
MIN_DATE = "1998-01-01"

# FOLDER SETTINGS
DATA_FOLDER = os.path.join(CWD, "data")
RESULTS_FOLDER = os.path.join(CWD, "results")

# data files
WRDS_DATA = os.path.join(DATA_FOLDER, "wrds_panel.csv")
MACRO_DATA = os.path.join(DATA_FOLDER, "macro_data.csv")
PROD_DATA = os.path.join(DATA_FOLDER, "panel_data.csv")
SIC_DATA = os.path.join(DATA_FOLDER, "sic_classification.csv")

# results files
SUMMARY_STATS_SIMPLE = os.path.join(RESULTS_FOLDER, "summary_stats_simple")
SUMMARY_STATS_GROUP = os.path.join(RESULTS_FOLDER, "summary_stats_group")

# MAKE FOLDERS
os.makedirs(DATA_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)
