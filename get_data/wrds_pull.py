import os
import sys
import wrds
import pandas as pd

CWD = os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir))

sys.path.append(CWD)
from settings import (
    WRDS_USERNAME,
    MIN_DATE,
    ASSETS_COL_COMPUSTAT,
    DEBT_COL_COMPUSTAT,
    RETURN_COLS_CRSP,
    WRDS_DATA,
    SIC_DATA,
)
from get_data.get_sic_classification import get_sic_classification

sys.path.append("/home/sidlh/Documents/reusable_code")
from timeseries_utils.group_apply import group_transform
from other_utils.datetime_utils import quarter_collapse
from wrds_management.pull_wrds import get_wrds_connection
from sql_management.sql_utils import get_sql_list


def get_data():

    con = get_wrds_connection(username=WRDS_USERNAME)
    sic_list = get_sic_classification()

    ######################
    ### get GVKEY list ###
    ######################

    gvkey_query = f"""
    SELECT DISTINCT gvkey, sic
    FROM comp.company
    WHERE sic IN ({get_sql_list(sic_list.index, quotes=True)});
    """
    gvkey_list = con.raw_sql(gvkey_query)

    ##########################
    ### Get COMPUSTAT data ###
    ##########################

    compustat_query = f"""
    SELECT gvkey, datadate, {get_sql_list(ASSETS_COL_COMPUSTAT + DEBT_COL_COMPUSTAT, quotes=False)}
    FROM comp.fundq
    WHERE datadate >= '{MIN_DATE}'
    AND gvkey IN ({get_sql_list(gvkey_list.gvkey, quotes=True)})
    """
    df = con.raw_sql(compustat_query)
    df["datadate"] = pd.to_datetime(df["datadate"])
    # forward fill to end of quarter
    df = group_transform(
        df,
        apply_fn=lambda x: x.ffill(limit=3),  # TODO make sure only fill within quarter
        panel_var=["gvkey"],
        time_var="datadate",
        return_series=False,
    )
    q_df = quarter_collapse(df, date_col="datadate", group_cols=["gvkey"], fn="last")

    ##########################
    ### Get CRSP crosswalk ###
    ##########################

    crsp_permno_query = f"""
    SELECT gvkey, lpermno
    FROM crsp.ccmxpf_linktable
    WHERE gvkey IN ({get_sql_list(gvkey_list.gvkey, quotes=True)});
    """
    crsp_permno = con.raw_sql(crsp_permno_query)
    crsp_permno = crsp_permno.dropna(subset=["lpermno"]).copy()
    crsp_permno["lpermno"] = crsp_permno["lpermno"].astype(int)

    # NOTE: each gvkey has more than one permno and vice versa
    # crsp_permno['gvkey'].value_counts()
    # crsp_permno['lpermno'].value_counts()

    assert crsp_permno["gvkey"].isin(gvkey_list.gvkey).all()

    ##########################
    ### Get CRSP data ###
    ##########################

    crsp_query = f"""
    SELECT permno, date, {get_sql_list(RETURN_COLS_CRSP, quotes=False)}
    FROM crsp.msf
    WHERE permno IN ({get_sql_list(crsp_permno.lpermno, quotes=True)})
    AND date >= '{MIN_DATE}'
    """
    return_df = con.raw_sql(crsp_query)
    assert return_df[["permno", "date"]].value_counts().max() == 1
    return_df["market_cap"] = return_df["prc"].abs() * return_df["shrout"]
    return_df["gross_ret"] = return_df["ret"] + 1
    # TODO deal with missing quarterly reports
    ret_df_q = quarter_collapse(
        return_df.sort_values("date"),
        date_col="date",
        group_cols=["permno"],
        fn="last",
        overwrite_dict={"ret": "prod"},
    )
    ret_df_q["ret"] = ret_df_q["gross_ret"] ** 4 - 1
    ret_df_q["ret"].describe()
    ret_df_q = ret_df_q.reset_index()

    #################################
    ### merege CRSP and COMPUSTAT ###
    #################################

    id_ret_df = ret_df_q.merge(
        crsp_permno.rename(columns={"lpermno": "permno"}), on="permno"
    )
    # TODO: watch out for multiple entries and missing entries
    df = q_df.reset_index().rename(columns={"datadate": "date"}).merge(id_ret_df)

    df = df.groupby(["gvkey", "date"]).apply("mean").reset_index()
    # TODO do this more carefully (could get two companies with same gvkey)

    df["debt"] = df[DEBT_COL_COMPUSTAT].sum(axis=1)
    df["assets"] = df[ASSETS_COL_COMPUSTAT].sum(axis=1)
    df = df.rename(columns={"ret": "return"})

    final_df = df[["gvkey", "date", "debt", "assets", "return"]].copy()

    # make sic_data
    sic_list_df = sic_list.reset_index().rename(columns={"sic_code": "sic"})
    sic_list_df["sic"] = sic_list_df["sic"].astype(str)
    sic_data = (
        final_df[["gvkey"]].drop_duplicates().merge(gvkey_list).merge(sic_list_df)
    )
    assert final_df["gvkey"].isin(sic_data.gvkey).all()

    return final_df, sic_data


if __name__ == "__main__":
    df, sic_data = get_data()
    df.to_csv(WRDS_DATA, index=False)
    sic_data.to_csv(SIC_DATA, index=False)
