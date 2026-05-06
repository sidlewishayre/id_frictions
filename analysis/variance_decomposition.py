import numpy as np
import pandas as pd


# ============================================================
# Settings
# ============================================================

FRICTIONS = ["leverage", "net_worth", "var_pct"]

Y_COL = "euler_equation"
ID_COL = "gvkey"
DATE_COL = "date"
ASSET_COL = "assets"


# ============================================================
# Core construction
# ============================================================

def add_B_components(
    df: pd.DataFrame,
    beta_hat: dict,
    frictions: list[str] = FRICTIONS,
    y_col: str = Y_COL,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Construct B^j_it = beta_j * f^j_it * df^j_it.

    Parameters
    ----------
    df:
        Panel dataframe with columns:
        - euler_equation
        - leverage, net_worth, var_pct
        - d_leverage, d_net_worth, d_var_pct

    beta_hat:
        Dictionary mapping friction name to estimated beta/mu.
        Example:
            beta_hat = {
                "leverage": 0.12,
                "net_worth": -0.05,
                "var_pct": 0.03,
            }

    Returns
    -------
    df_out:
        Dataframe with added B columns.

    B_cols:
        List of B-column names.
    """

    df_out = df.copy()
    B_cols = []

    for f in frictions:
        d_col = f"d_{f}"
        B_col = f"B_{f}"

        if f not in beta_hat:
            raise ValueError(f"Missing beta_hat for friction: {f}")

        if f not in df_out.columns:
            raise ValueError(f"Missing friction column in dataframe: {f}")

        if d_col not in df_out.columns:
            raise ValueError(f"Missing derivative column in dataframe: {d_col}")

        df_out[B_col] = beta_hat[f] * df_out[f] * df_out[d_col]
        B_cols.append(B_col)

    df_out["R_target_residual"] = df_out[y_col] - df_out[B_cols].sum(axis=1)

    return df_out, B_cols


def covariance_shares(
    data: pd.DataFrame,
    y_col: str,
    component_cols: list[str],
    include_residual: bool = True,
    residual_col: str = "R_target_residual",
    min_obs: int = 10,
) -> pd.Series:
    """
    Compute covariance shares:

        s_j = Cov(y, component_j) / Var(y)

    If include_residual=True, also computes

        s_R = Cov(y, R) / Var(y)

    where R = y - sum_j component_j.

    The shares sum to one when residual is included,
    up to numerical precision and missing-value handling.
    """

    cols = [y_col] + component_cols
    if include_residual:
        cols = cols + [residual_col]

    d = data[cols].replace([np.inf, -np.inf], np.nan).dropna()

    out = {}

    if len(d) < min_obs:
        for c in component_cols:
            out[f"share_{c}"] = np.nan
        if include_residual:
            out[f"share_{residual_col}"] = np.nan
        out["var_y"] = np.nan
        out["n_obs"] = len(d)
        out["sum_shares"] = np.nan
        return pd.Series(out)

    y = d[y_col]
    var_y = y.var(ddof=1)

    if var_y <= 0 or np.isnan(var_y):
        for c in component_cols:
            out[f"share_{c}"] = np.nan
        if include_residual:
            out[f"share_{residual_col}"] = np.nan
        out["var_y"] = var_y
        out["n_obs"] = len(d)
        out["sum_shares"] = np.nan
        return pd.Series(out)

    share_cols = []

    for c in component_cols:
        s = y.cov(d[c]) / var_y
        out[f"share_{c}"] = s
        share_cols.append(f"share_{c}")

    if include_residual:
        s_R = y.cov(d[residual_col]) / var_y
        out[f"share_{residual_col}"] = s_R
        share_cols.append(f"share_{residual_col}")

    out["var_y"] = var_y
    out["n_obs"] = len(d)
    out["sum_shares"] = np.nansum([out[c] for c in share_cols])

    return pd.Series(out)


# ============================================================
# 1. Firm-level rolling decomposition
# ============================================================

def firm_rolling_decomposition(
    df: pd.DataFrame,
    B_cols: list[str],
    window: int = 20,
    min_obs: int = 12,
    id_col: str = ID_COL,
    date_col: str = DATE_COL,
    y_col: str = Y_COL,
) -> pd.DataFrame:
    """
    For each firm, compute rolling covariance shares over time.

    Output has one row per firm-date.
    """

    df = df.sort_values([id_col, date_col]).copy()
    results = []

    for firm_id, g in df.groupby(id_col, sort=False):
        g = g.sort_values(date_col).reset_index(drop=True)

        for end in range(len(g)):
            start = max(0, end - window + 1)
            window_df = g.iloc[start:end + 1]

            shares = covariance_shares(
                window_df,
                y_col=y_col,
                component_cols=B_cols,
                include_residual=True,
                min_obs=min_obs,
            )

            row = {
                id_col: firm_id,
                date_col: g.loc[end, date_col],
            }
            row.update(shares.to_dict())
            results.append(row)

    return pd.DataFrame(results)


def summarize_firm_rolling(
    firm_roll: pd.DataFrame,
    date_col: str = DATE_COL,
    percentiles: tuple[float, ...] = (0.10, 0.25, 0.50, 0.75, 0.90),
) -> pd.DataFrame:
    """
    Summarize the cross-firm distribution of firm-level rolling shares.

    Returns one row per date-component.
    """

    share_cols = [c for c in firm_roll.columns if c.startswith("share_")]

    long = firm_roll.melt(
        id_vars=[date_col],
        value_vars=share_cols,
        var_name="component",
        value_name="share",
    ).dropna(subset=["share"])

    def summarize_group(x):
        out = {
            "mean": x.mean(),
            "std": x.std(ddof=1),
            "n_firms": x.shape[0],
        }
        for p in percentiles:
            out[f"p{int(100*p):02d}"] = x.quantile(p)
        return pd.Series(out)

    summary = (
        long.groupby([date_col, "component"])["share"]
        .apply(summarize_group)
        .reset_index()
    )

    summary = summary.pivot_table(
        index=[date_col, "component"],
        columns="level_2",
        values="share",
    ).reset_index()

    return summary


# ============================================================
# 2. Asset-weighted aggregate time-series decomposition
# ============================================================

def asset_weighted_aggregate_series(
    df: pd.DataFrame,
    B_cols: list[str],
    date_col: str = DATE_COL,
    y_col: str = Y_COL,
    asset_col: str = ASSET_COL,
) -> pd.DataFrame:
    """
    Construct asset-weighted aggregate y_t and B^j_t.

        y_t = sum_i w_it y_it
        B^j_t = sum_i w_it B^j_it
        w_it = assets_it / sum_i assets_it
    """

    cols = [date_col, y_col, asset_col] + B_cols
    d = df[cols].replace([np.inf, -np.inf], np.nan).dropna().copy()

    d = d[d[asset_col] > 0].copy()

    d["weight"] = d[asset_col] / d.groupby(date_col)[asset_col].transform("sum")

    agg_dict = {y_col: lambda x: np.nan}
    rows = []

    for date, g in d.groupby(date_col):
        row = {date_col: date}
        row[y_col] = np.sum(g["weight"] * g[y_col])
        for c in B_cols:
            row[c] = np.sum(g["weight"] * g[c])
        rows.append(row)

    out = pd.DataFrame(rows).sort_values(date_col)
    out["R_target_residual"] = out[y_col] - out[B_cols].sum(axis=1)

    return out


def rolling_time_series_decomposition(
    ts_df: pd.DataFrame,
    B_cols: list[str],
    window: int = 20,
    min_obs: int = 12,
    date_col: str = DATE_COL,
    y_col: str = Y_COL,
) -> pd.DataFrame:
    """
    Rolling covariance shares for a time-series dataframe.

    Used after asset-weighting across firms.
    """

    ts_df = ts_df.sort_values(date_col).reset_index(drop=True).copy()
    results = []

    for end in range(len(ts_df)):
        start = max(0, end - window + 1)
        window_df = ts_df.iloc[start:end + 1]

        shares = covariance_shares(
            window_df,
            y_col=y_col,
            component_cols=B_cols,
            include_residual=True,
            min_obs=min_obs,
        )

        row = {date_col: ts_df.loc[end, date_col]}
        row.update(shares.to_dict())
        results.append(row)

    return pd.DataFrame(results)


def asset_weighted_rolling_decomposition(
    df: pd.DataFrame,
    B_cols: list[str],
    window: int = 20,
    min_obs: int = 12,
    date_col: str = DATE_COL,
    y_col: str = Y_COL,
    asset_col: str = ASSET_COL,
) -> pd.DataFrame:
    """
    Full asset-weighted rolling decomposition:
    1. asset-weight firms into aggregate time series;
    2. compute rolling covariance shares over time.
    """

    agg = asset_weighted_aggregate_series(
        df,
        B_cols=B_cols,
        date_col=date_col,
        y_col=y_col,
        asset_col=asset_col,
    )

    roll = rolling_time_series_decomposition(
        agg,
        B_cols=B_cols,
        window=window,
        min_obs=min_obs,
        date_col=date_col,
        y_col=y_col,
    )

    return roll


# ============================================================
# 3. Pooled decompositions
# ============================================================

def pooled_full_sample_decomposition(
    df: pd.DataFrame,
    B_cols: list[str],
    y_col: str = Y_COL,
    min_obs: int = 50,
) -> pd.Series:
    """
    Pool all firm-date observations and compute one covariance decomposition.
    """

    return covariance_shares(
        df,
        y_col=y_col,
        component_cols=B_cols,
        include_residual=True,
        min_obs=min_obs,
    )


def pooled_cross_sectional_by_date(
    df: pd.DataFrame,
    B_cols: list[str],
    date_col: str = DATE_COL,
    y_col: str = Y_COL,
    min_obs: int = 20,
) -> pd.DataFrame:
    """
    At each date, pool firms cross-sectionally and compute covariance shares.

    This gives one decomposition per date using cross-firm variation at that date.
    """

    results = []

    for date, g in df.groupby(date_col):
        shares = covariance_shares(
            g,
            y_col=y_col,
            component_cols=B_cols,
            include_residual=True,
            min_obs=min_obs,
        )
        row = {date_col: date}
        row.update(shares.to_dict())
        results.append(row)

    return pd.DataFrame(results).sort_values(date_col)


def pooled_rolling_panel_decomposition(
    df: pd.DataFrame,
    B_cols: list[str],
    window: int = 20,
    min_obs: int = 100,
    date_col: str = DATE_COL,
    y_col: str = Y_COL,
) -> pd.DataFrame:
    """
    Rolling pooled panel decomposition.

    At each date t, use all firm-date observations in the last `window`
    dates and compute Cov(y, B^j) / Var(y).

    This gives one decomposition per date using both time and cross-section
    variation within the rolling window.
    """

    d = df.sort_values(date_col).copy()
    dates = np.array(sorted(d[date_col].dropna().unique()))

    results = []

    for end_idx, date in enumerate(dates):
        start_idx = max(0, end_idx - window + 1)
        window_dates = dates[start_idx:end_idx + 1]

        window_df = d[d[date_col].isin(window_dates)]

        shares = covariance_shares(
            window_df,
            y_col=y_col,
            component_cols=B_cols,
            include_residual=True,
            min_obs=min_obs,
        )

        row = {
            date_col: date,
            "start_date": window_dates[0],
            "end_date": window_dates[-1],
        }
        row.update(shares.to_dict())
        results.append(row)

    return pd.DataFrame(results)


# ============================================================
# Example usage
# ============================================================

if __name__ == "__main__":

    # Load prepared data
    df = pd.read_csv("data/panel_data.csv")
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])

    # Replace these with your actual estimated mu/beta values
    beta_hat = {
        "leverage": 0.10,
        "net_worth": 0.05,
        "var_pct": 0.02,
    }

    # Construct B^j_it components
    df, B_cols = add_B_components(df, beta_hat)

    # --------------------------------------------------------
    # 1. Firm-level rolling decomposition
    # --------------------------------------------------------

    firm_roll = firm_rolling_decomposition(
        df,
        B_cols=B_cols,
        window=20,
        min_obs=12,
    )

    firm_roll_summary = summarize_firm_rolling(firm_roll)

    firm_roll.to_csv("results/firm_rolling_variance_decomposition.csv", index=False)
    firm_roll_summary.to_csv(
        "results/firm_rolling_variance_decomposition_summary.csv",
        index=False,
    )

    # --------------------------------------------------------
    # 2. Asset-weighted aggregate rolling decomposition
    # --------------------------------------------------------

    asset_weighted_roll = asset_weighted_rolling_decomposition(
        df,
        B_cols=B_cols,
        window=20,
        min_obs=12,
    )

    asset_weighted_roll.to_csv(
        "results/asset_weighted_rolling_variance_decomposition.csv",
        index=False,
    )

    # --------------------------------------------------------
    # 3a. Pooled full-sample decomposition
    # --------------------------------------------------------

    pooled_full = pooled_full_sample_decomposition(df, B_cols=B_cols)
    pooled_full.to_frame("value").to_csv(
        "results/pooled_full_sample_variance_decomposition.csv"
    )

    # --------------------------------------------------------
    # 3b. Pooled cross-sectional decomposition, one per date
    # --------------------------------------------------------

    pooled_by_date = pooled_cross_sectional_by_date(
        df,
        B_cols=B_cols,
        min_obs=20,
    )

    pooled_by_date.to_csv(
        "results/pooled_cross_sectional_by_date_variance_decomposition.csv",
        index=False,
    )

    # --------------------------------------------------------
    # 3c. Pooled rolling panel decomposition
    # --------------------------------------------------------

    pooled_roll = pooled_rolling_panel_decomposition(
        df,
        B_cols=B_cols,
        window=20,
        min_obs=100,
    )

    pooled_roll.to_csv(
        "results/pooled_rolling_panel_variance_decomposition.csv",
        index=False,
    )