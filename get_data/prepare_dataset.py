import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import norm

CWD = os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir))

sys.path.append(CWD)
from settings import WRDS_DATA, MACRO_DATA, PROD_DATA, SIC_DATA
from settings import BETA, GAMMA
from settings import FINANCIAL_FRICTIONS

sys.path.append("/home/sidlh/Documents/reusable_code")
from timeseries_utils.group_apply import group_transform

var_std = lambda x: x.rolling(20, min_periods=5).std()

wrds = pd.read_csv(WRDS_DATA)
wrds["date"] = pd.to_datetime(wrds["date"])
macro_data = pd.read_csv(MACRO_DATA)
macro_data["date"] = pd.to_datetime(macro_data["date"])
macro_data = macro_data.sort_values("date")
macro_data["gdp_next"] = macro_data["gdp"].shift(-1)
macro_data["vix_std"] = var_std(macro_data["vix"])


df = wrds.merge(macro_data, on="date", how="left")

# check no observations dropped and that macro data is complete
assert df[macro_data.columns].isnull().sum().sum() == 0
assert len(df) == len(wrds)


######################
### DATA FILTERING ###
######################

for col in ["assets", "debt"]:
    df[col] *= 100 / df["gdp_deflator"]
df = df[df.groupby("gvkey")["assets"].transform("mean") > 1000].copy()

#####################
### DATA CLEANING ###
#####################

# cap returns to 100%
df["return"] = df["return"].clip(upper=1)

#############################
### VARIABLE CONSTRUCTION ###
#############################

# specify necceary variables
df["return_next"] = group_transform(
    df,
    apply_fn=lambda x: x.shift(-1),
    panel_var=["gvkey"],
    time_var="date",
    transform_var="return",
    return_series=True,
)
df["euler_equation"] = (
    BETA
    * (df["gdp"] / df["gdp_next"]) ** (GAMMA)
    * ((df["return_next"] + 1) / (1 + df["ffr"] / 100))
    - 1
)

# specifying financial frictions
df["equity"] = df["assets"] - df["debt"]
df["leverage"] = df["assets"] / df["equity"]
df["net_worth"] = 1 / df["leverage"]

# adding Value at Risk
df["return_std"] = group_transform(
    df,
    apply_fn=var_std,
    panel_var=["gvkey"],
    time_var="date",
    transform_var="return",
    return_series=True,
)
df["port_std"] = df["vix"] * (df["return_std"] / df["vix_std"])
df["var_sigma"] = df["assets"] / (df["port_std"] * df["equity"])
df["var_pct"] = norm.cdf(df["var_sigma"])

# financial friction derivatives
df["d_leverage"] = -df["debt"] / (df["equity"] ** 2)
df["d_net_worth"] = df["debt"] / (df["assets"] ** 2)
df["d_var_pct"] = df["d_leverage"] * norm.pdf(df["var_sigma"]) / df["port_std"]

######################
### SAVING DATASET ###
######################

final_df = df[
    ["gvkey", "date", "euler_equation", "assets", "debt", "equity", "return"]
    + FINANCIAL_FRICTIONS
    + [f"d_{ff}" for ff in FINANCIAL_FRICTIONS]
].copy()
final_df = final_df.dropna()
# TODO: make sure data includes enough consecutive observations per intermediary

# add in intermediary classification
sic_data = pd.read_csv(SIC_DATA)
sic_data["bank"] = (sic_data["intermediary_classification"] == "banks").map(
    {True: "Bank", False: "Non-Bank"}
)
assert (
    sic_data["gvkey"].value_counts().max() == 1
), "gvkey appears multiple times in sic data"
len_bef = len(final_df)
final_df = final_df.merge(sic_data[["gvkey", "bank"]], on="gvkey", how="left")
assert len(final_df) == len_bef, "merge added/dropped observations"
assert final_df["bank"].isnull().sum() == 0, "some gvkeys missing bank classification"
assert (final_df["bank"] == "Bank").sum() > 0, "no banks found"

final_df = final_df.replace([np.inf, -np.inf], np.nan)

final_df.to_csv(PROD_DATA, index=False)
