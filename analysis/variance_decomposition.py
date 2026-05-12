# variance_decomposition.py

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# File paths
# ============================================================

DATA_DIR = "."
PANEL_FILE = "data/panel_data.csv"
SIC_FILE = "data/sic_classification.csv"

RESULTS_DIR = "results_variance_decomposition"
PLOTS_DIR = "plots_variance_decomposition"

if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

if not os.path.exists(PLOTS_DIR):
    os.makedirs(PLOTS_DIR)


# ============================================================
# User-input estimated mus / betas from STANDARDIZED regression
# ============================================================

# Replace these with the mus from the standardized regression.
# These coefficients multiply x_j_std, where:
# x_j_std = (f_j * df_j) / sd(f_j * df_j)
BETA_HAT = {
    "leverage": -0.137,
    "net_worth": -0.0209,
    "var_pct": 0.0166,
}


# ============================================================
# Column settings
# ============================================================

ID_COL = "gvkey"
DATE_COL = "date"
Y_COL = "euler_equation"
ASSET_COL = "assets"

FRICTIONS = ["leverage", "net_worth", "var_pct"]

DERIVATIVE_COLS = {
    "leverage": "d_leverage",
    "net_worth": "d_net_worth",
    "var_pct": "d_var_pct",
}

X_COLS = {
    "leverage": "x_leverage",
    "net_worth": "x_net_worth",
    "var_pct": "x_var_pct",
}


# ============================================================
# Load and prepare data
# ============================================================

def load_data(data_dir=DATA_DIR):
    panel_path = os.path.join(data_dir, PANEL_FILE)

    if not os.path.exists(panel_path):
        raise FileNotFoundError("Could not find {}".format(panel_path))

    df = pd.read_csv(panel_path)

    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values([ID_COL, DATE_COL]).reset_index(drop=True)

    sic_path = os.path.join(data_dir, SIC_FILE)
    if os.path.exists(sic_path):
        sic = pd.read_csv(sic_path)
        df = df.merge(sic, on=ID_COL, how="left")

    return df


def clean_panel(df):
    needed_cols = [ID_COL, DATE_COL, Y_COL, ASSET_COL]
    needed_cols += FRICTIONS
    needed_cols += [DERIVATIVE_COLS[f] for f in FRICTIONS]

    missing = [c for c in needed_cols if c not in df.columns]
    if len(missing) > 0:
        raise ValueError("Missing required columns: {}".format(missing))

    df = df.copy()
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=needed_cols)
    df = df[df[ASSET_COL] > 0].copy()

    return df


# ============================================================
# Standardization matching your friend's code
# ============================================================

def prepare_data_standardized(df, frictions=FRICTIONS):
    """
    Matches the friend's standardization:

    1. Construct:
        x_j = f_j * d_j

    2. Standardize f_j, d_j, and x_j separately:
        variable_std = variable / sd(variable)

    Therefore:
        x_j_std = (f_j_raw * d_j_raw) / sd(f_j_raw * d_j_raw)

    Not:
        (f_j_raw / sd(f_j_raw)) * (d_j_raw / sd(d_j_raw))
    """

    df = df.copy()

    # Save raw variables and construct raw x variables
    for f in frictions:
        d_col = DERIVATIVE_COLS[f]
        x_col = X_COLS[f]

        df[f + "_raw"] = df[f]
        df[d_col + "_raw"] = df[d_col]

        df[x_col] = df[f] * df[d_col]
        df[x_col + "_raw"] = df[x_col]

    # Standardize f, d, and x separately
    reg_vars = []
    reg_vars += frictions
    reg_vars += [DERIVATIVE_COLS[f] for f in frictions]
    reg_vars += [X_COLS[f] for f in frictions]

    scales = {}

    for var in reg_vars:
        sd = df[var].std(ddof=1)

        if sd == 0 or np.isnan(sd):
            raise ValueError("Cannot standardize {} because std is {}.".format(var, sd))

        scales[var] = sd
        df[var] = df[var] / sd

    scales = pd.Series(scales, name="standardization_sd")

    return df, scales


# ============================================================
# Construct B terms using standardized x variables
# ============================================================

def add_B_components_standardized(df, beta_hat):
    """
    Correct if beta_hat comes from standardized regression:

        B^j_it = beta_hat_j * x^j_std_it

    where x^j_std_it is the standardized x variable.
    """

    df = df.copy()
    beta_hat = pd.Series(beta_hat)

    B_cols = []

    for f in FRICTIONS:
        x_col = X_COLS[f]
        B_col = "B_{}".format(f)

        if f not in beta_hat.index:
            raise ValueError("Missing beta/mu for friction: {}".format(f))

        if x_col not in df.columns:
            raise ValueError("Missing standardized x column: {}".format(x_col))

        df[B_col] = beta_hat.loc[f] * df[x_col]
        B_cols.append(B_col)

    df["R_target_residual"] = df[Y_COL] - df[B_cols].sum(axis=1)

    return df, B_cols


# ============================================================
# Covariance-share calculation
# ============================================================

def covariance_shares(
    data,
    y_col,
    component_cols,
    residual_col="R_target_residual",
    include_residual=True,
    min_obs=10,
):
    """
    Computes:
        s_j = Cov(y, B_j) / Var(y)

    If residual is included:
        s_R = Cov(y, R) / Var(y)

    Since:
        y = sum_j B_j + R

    shares should sum to one up to numerical precision.
    """

    cols = [y_col] + component_cols
    if include_residual:
        cols.append(residual_col)

    d = data[cols].replace([np.inf, -np.inf], np.nan).dropna()

    out = {}

    if len(d) < min_obs:
        for c in component_cols:
            out["share_{}".format(c)] = np.nan
        if include_residual:
            out["share_{}".format(residual_col)] = np.nan
        out["var_y"] = np.nan
        out["n_obs"] = len(d)
        out["sum_shares"] = np.nan
        return pd.Series(out)

    y = d[y_col]
    var_y = y.var(ddof=1)

    if var_y <= 0 or np.isnan(var_y):
        for c in component_cols:
            out["share_{}".format(c)] = np.nan
        if include_residual:
            out["share_{}".format(residual_col)] = np.nan
        out["var_y"] = var_y
        out["n_obs"] = len(d)
        out["sum_shares"] = np.nan
        return pd.Series(out)

    share_names = []

    for c in component_cols:
        share_name = "share_{}".format(c)
        out[share_name] = y.cov(d[c]) / var_y
        share_names.append(share_name)

    if include_residual:
        share_name = "share_{}".format(residual_col)
        out[share_name] = y.cov(d[residual_col]) / var_y
        share_names.append(share_name)

    out["var_y"] = var_y
    out["n_obs"] = len(d)
    out["sum_shares"] = np.nansum([out[s] for s in share_names])

    return pd.Series(out)


# ============================================================
# 1. Firm-level rolling decomposition
# ============================================================

def firm_rolling_decomposition(df, B_cols, window=20, min_obs=12):
    """
    For each firm, compute rolling time-series covariance shares:

        s^j_{i,t} = Cov_tau(y_i, B^j_i) / Var_tau(y_i)

    using the last `window` observations for firm i.
    """

    df = df.sort_values([ID_COL, DATE_COL]).copy()
    results = []

    for firm_id, g in df.groupby(ID_COL):
        g = g.sort_values(DATE_COL).reset_index(drop=True)

        for end in range(len(g)):
            start = max(0, end - window + 1)
            window_df = g.iloc[start:end + 1]

            shares = covariance_shares(
                window_df,
                y_col=Y_COL,
                component_cols=B_cols,
                include_residual=True,
                min_obs=min_obs,
            )

            row = {
                ID_COL: firm_id,
                DATE_COL: g.loc[end, DATE_COL],
            }

            if "bank" in g.columns:
                row["bank"] = g.loc[end, "bank"]

            if "intermediary_classification" in g.columns:
                row["intermediary_classification"] = g.loc[
                    end, "intermediary_classification"
                ]

            row.update(shares.to_dict())
            results.append(row)

    return pd.DataFrame(results)


def summarize_firm_rolling(
    firm_roll,
    percentiles=(0.10, 0.25, 0.50, 0.75, 0.90),
):
    """
    At each date, summarize the cross-firm distribution of firm-level rolling shares.
    """

    share_cols = [c for c in firm_roll.columns if c.startswith("share_")]

    long = firm_roll.melt(
        id_vars=[DATE_COL],
        value_vars=share_cols,
        var_name="component",
        value_name="share",
    ).dropna(subset=["share"])

    rows = []

    for key, g in long.groupby([DATE_COL, "component"]):
        date, component = key
        x = g["share"]

        row = {
            DATE_COL: date,
            "component": component,
            "mean": x.mean(),
            "std": x.std(ddof=1),
            "n_firms": x.shape[0],
        }

        for p in percentiles:
            row["p{:02d}".format(int(100 * p))] = x.quantile(p)

        rows.append(row)

    return pd.DataFrame(rows).sort_values([DATE_COL, "component"])


# ============================================================
# 2. Asset-weighted aggregate decomposition
# ============================================================

def asset_weighted_aggregate_series(df, B_cols):
    """
    Constructs asset-weighted aggregate series:

        y_t = sum_i w_it y_it
        B^j_t = sum_i w_it B^j_it

    with:
        w_it = assets_it / sum_i assets_it
    """

    cols = [DATE_COL, ID_COL, Y_COL, ASSET_COL] + B_cols
    d = df[cols].replace([np.inf, -np.inf], np.nan).dropna().copy()

    d = d[d[ASSET_COL] > 0].copy()
    d["weight"] = d[ASSET_COL] / d.groupby(DATE_COL)[ASSET_COL].transform("sum")

    rows = []

    for date, g in d.groupby(DATE_COL):
        row = {DATE_COL: date}
        row[Y_COL] = np.sum(g["weight"] * g[Y_COL])
        row[ASSET_COL] = g[ASSET_COL].sum()
        row["n_firms"] = g[ID_COL].nunique()

        for c in B_cols:
            row[c] = np.sum(g["weight"] * g[c])

        rows.append(row)

    agg = pd.DataFrame(rows).sort_values(DATE_COL).reset_index(drop=True)
    agg["R_target_residual"] = agg[Y_COL] - agg[B_cols].sum(axis=1)

    return agg


def rolling_time_series_decomposition(ts_df, B_cols, window=20, min_obs=12):
    """
    Rolling covariance shares over time for an aggregate time series.
    """

    ts_df = ts_df.sort_values(DATE_COL).reset_index(drop=True).copy()
    results = []

    for end in range(len(ts_df)):
        start = max(0, end - window + 1)
        window_df = ts_df.iloc[start:end + 1]

        shares = covariance_shares(
            window_df,
            y_col=Y_COL,
            component_cols=B_cols,
            include_residual=True,
            min_obs=min_obs,
        )

        row = {
            DATE_COL: ts_df.loc[end, DATE_COL],
            "start_date": ts_df.loc[start, DATE_COL],
            "end_date": ts_df.loc[end, DATE_COL],
        }

        row.update(shares.to_dict())
        results.append(row)

    return pd.DataFrame(results)


def asset_weighted_rolling_decomposition(df, B_cols, window=20, min_obs=12):
    agg = asset_weighted_aggregate_series(df, B_cols)

    roll = rolling_time_series_decomposition(
        agg,
        B_cols=B_cols,
        window=window,
        min_obs=min_obs,
    )

    return agg, roll


# ============================================================
# 3. Pooled decompositions
# ============================================================

def pooled_full_sample_decomposition(df, B_cols, min_obs=50):
    """
    One pooled decomposition using all firm-date observations.
    """

    return covariance_shares(
        df,
        y_col=Y_COL,
        component_cols=B_cols,
        include_residual=True,
        min_obs=min_obs,
    )


def pooled_cross_sectional_by_date(df, B_cols, min_obs=20):
    """
    One cross-sectional decomposition per date using variation across firms.
    """

    results = []

    for date, g in df.groupby(DATE_COL):
        shares = covariance_shares(
            g,
            y_col=Y_COL,
            component_cols=B_cols,
            include_residual=True,
            min_obs=min_obs,
        )

        row = {
            DATE_COL: date,
            "n_firms_raw": g[ID_COL].nunique(),
        }

        row.update(shares.to_dict())
        results.append(row)

    return pd.DataFrame(results).sort_values(DATE_COL)


def pooled_rolling_panel_decomposition(df, B_cols, window=20, min_obs=100):
    """
    At each date t, use all firm-date observations in the last `window` dates.
    """

    d = df.sort_values(DATE_COL).copy()
    dates = np.array(sorted(d[DATE_COL].dropna().unique()))

    results = []

    for end_idx, date in enumerate(dates):
        start_idx = max(0, end_idx - window + 1)
        window_dates = dates[start_idx:end_idx + 1]

        window_df = d[d[DATE_COL].isin(window_dates)]

        shares = covariance_shares(
            window_df,
            y_col=Y_COL,
            component_cols=B_cols,
            include_residual=True,
            min_obs=min_obs,
        )

        row = {
            DATE_COL: date,
            "start_date": window_dates[0],
            "end_date": window_dates[-1],
            "n_dates": len(window_dates),
        }

        row.update(shares.to_dict())
        results.append(row)

    return pd.DataFrame(results).sort_values(DATE_COL)


# ============================================================
# Plotting
# ============================================================

def clean_component_name(name):
    name = name.replace("share_B_", "")
    name = name.replace("leverage", "Leverage")
    name = name.replace("net_worth", "Net worth")
    name = name.replace("var_pct", "VaR")
    return name


def friction_share_columns(df):
    """
    Returns only friction share columns, excluding residual share.
    """
    cols = [c for c in df.columns if c.startswith("share_B_")]
    return cols


def set_adaptive_ylim(ax, data_values, pad_frac=0.15, min_pad=1e-6):
    """
    Sets y-limits tightly around the plotted data.

    pad_frac adds some breathing room around min/max.
    min_pad prevents zero-height axes when variation is tiny.
    """

    values = pd.Series(np.asarray(data_values).ravel())
    values = values.replace([np.inf, -np.inf], np.nan).dropna()

    if len(values) == 0:
        return

    y_min = values.min()
    y_max = values.max()

    if y_min == y_max:
        pad = max(abs(y_min) * pad_frac, min_pad)
    else:
        pad = max((y_max - y_min) * pad_frac, min_pad)

    ax.set_ylim(y_min - pad, y_max + pad)


def plot_file(filename, title, ylabel, outname):
    path = os.path.join(RESULTS_DIR, filename)

    if not os.path.exists(path):
        print("Missing file:", path)
        return

    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])

    # Only plot friction components. Omit residual share.
    share_cols = friction_share_columns(df)

    if len(share_cols) == 0:
        print("No friction share columns found in:", filename)
        return

    fig, ax = plt.subplots(figsize=(11, 6))

    plotted_values = []

    for c in share_cols:
        ax.plot(df["date"], df[c], label=clean_component_name(c))
        plotted_values.append(df[c].values)

    ax.axhline(0, linewidth=0.8)

    # Adaptive y-axis based only on plotted friction components.
    set_adaptive_ylim(ax, np.concatenate(plotted_values))

    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel(ylabel)
    ax.legend()
    fig.tight_layout()

    out = os.path.join(PLOTS_DIR, outname)
    fig.savefig(out, dpi=300)
    plt.close(fig)

    print("Saved plot:", out)


def plot_firm_rolling_summary():
    path = os.path.join(RESULTS_DIR, "firm_rolling_decomposition_summary.csv")

    if not os.path.exists(path):
        print("Missing file:", path)
        return

    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])

    # Only plot B components. Omit R_target_residual.
    components = sorted([
        c for c in df["component"].dropna().unique()
        if c.startswith("share_B_")
    ])

    for comp in components:
        g = df[df["component"] == comp].sort_values("date")

        fig, ax = plt.subplots(figsize=(11, 6))

        ax.plot(g["date"], g["mean"], label="Mean")
        ax.plot(g["date"], g["p50"], linestyle="--", label="Median")

        plotted_values = [g["mean"].values, g["p50"].values]

        if "p25" in g.columns and "p75" in g.columns:
            ax.fill_between(
                g["date"],
                g["p25"],
                g["p75"],
                alpha=0.25,
                label="25th-75th percentile",
            )
            plotted_values.extend([g["p25"].values, g["p75"].values])

        if "p10" in g.columns and "p90" in g.columns:
            ax.fill_between(
                g["date"],
                g["p10"],
                g["p90"],
                alpha=0.15,
                label="10th-90th percentile",
            )
            plotted_values.extend([g["p10"].values, g["p90"].values])

        ax.axhline(0, linewidth=0.8)

        # Adaptive y-axis based on the component being plotted.
        set_adaptive_ylim(ax, np.concatenate(plotted_values))

        ax.set_title("Firm-Level Rolling Shares: {}".format(clean_component_name(comp)))
        ax.set_xlabel("Date")
        ax.set_ylabel("Covariance share")
        ax.legend()
        fig.tight_layout()

        safe_name = comp.replace("share_", "").replace("/", "_")
        out = os.path.join(PLOTS_DIR, "firm_rolling_summary_{}.png".format(safe_name))
        fig.savefig(out, dpi=300)
        plt.close(fig)

        print("Saved plot:", out)


def make_all_plots():
    plot_file(
        "asset_weighted_rolling_decomposition.csv",
        "Asset-Weighted Rolling Variance Decomposition",
        "Covariance share",
        "asset_weighted_rolling_decomposition.png",
    )

    plot_file(
        "pooled_cross_sectional_by_date.csv",
        "Pooled Cross-Sectional Variance Decomposition by Date",
        "Cross-sectional covariance share",
        "pooled_cross_sectional_by_date.png",
    )

    plot_file(
        "pooled_rolling_panel_decomposition.csv",
        "Pooled Rolling Panel Variance Decomposition",
        "Pooled rolling covariance share",
        "pooled_rolling_panel_decomposition.png",
    )

    plot_firm_rolling_summary()

# ============================================================
# Main execution
# ============================================================

def main():
    print("Loading data...")
    df = load_data(DATA_DIR)
    df = clean_panel(df)

    print("Standardizing variables...")
    df, scales = prepare_data_standardized(df)

    scales.to_csv(os.path.join(RESULTS_DIR, "standardization_scales.csv"))

    beta_hat = pd.Series(BETA_HAT, name="beta_hat_standardized")
    beta_hat.to_csv(os.path.join(RESULTS_DIR, "beta_hat_mu_used_standardized.csv"))

    print("\nUsing standardized beta/mu values:")
    print(beta_hat)

    print("\nStandardization scales:")
    print(scales)

    print("\nConstructing B terms...")
    df, B_cols = add_B_components_standardized(df, beta_hat)

    df.to_csv(os.path.join(RESULTS_DIR, "panel_with_standardized_B_terms.csv"), index=False)

    print("\nConstructed B columns:")
    print(B_cols)

    print("\nRunning firm-level rolling decomposition...")
    firm_roll = firm_rolling_decomposition(
        df,
        B_cols=B_cols,
        window=20,
        min_obs=12,
    )

    firm_roll_summary = summarize_firm_rolling(firm_roll)

    firm_roll.to_csv(
        os.path.join(RESULTS_DIR, "firm_rolling_decomposition.csv"),
        index=False,
    )

    firm_roll_summary.to_csv(
        os.path.join(RESULTS_DIR, "firm_rolling_decomposition_summary.csv"),
        index=False,
    )

    print("Running asset-weighted aggregate decomposition...")
    agg_ts, asset_weighted_roll = asset_weighted_rolling_decomposition(
        df,
        B_cols=B_cols,
        window=20,
        min_obs=12,
    )

    agg_ts.to_csv(
        os.path.join(RESULTS_DIR, "asset_weighted_aggregate_series.csv"),
        index=False,
    )

    asset_weighted_roll.to_csv(
        os.path.join(RESULTS_DIR, "asset_weighted_rolling_decomposition.csv"),
        index=False,
    )

    print("Running pooled full-sample decomposition...")
    pooled_full = pooled_full_sample_decomposition(
        df,
        B_cols=B_cols,
        min_obs=50,
    )

    pooled_full.to_frame("value").to_csv(
        os.path.join(RESULTS_DIR, "pooled_full_sample_decomposition.csv")
    )

    print("Running pooled cross-sectional-by-date decomposition...")
    pooled_by_date = pooled_cross_sectional_by_date(
        df,
        B_cols=B_cols,
        min_obs=20,
    )

    pooled_by_date.to_csv(
        os.path.join(RESULTS_DIR, "pooled_cross_sectional_by_date.csv"),
        index=False,
    )

    print("Running pooled rolling panel decomposition...")
    pooled_roll = pooled_rolling_panel_decomposition(
        df,
        B_cols=B_cols,
        window=20,
        min_obs=100,
    )

    pooled_roll.to_csv(
        os.path.join(RESULTS_DIR, "pooled_rolling_panel_decomposition.csv"),
        index=False,
    )

    print("\nPooled full-sample decomposition:")
    print(pooled_full)

    print("\nAverage sum of shares:")
    print("Firm rolling:", firm_roll["sum_shares"].mean(skipna=True))
    print("Asset weighted rolling:", asset_weighted_roll["sum_shares"].mean(skipna=True))
    print("Pooled by date:", pooled_by_date["sum_shares"].mean(skipna=True))
    print("Pooled rolling:", pooled_roll["sum_shares"].mean(skipna=True))

    print("\nSaved CSV outputs to:", RESULTS_DIR)

    print("\nMaking plots without residual line...")
    make_all_plots()

    print("\nSaved plots to:", PLOTS_DIR)
    print("\nDone.")


if __name__ == "__main__":
    main()