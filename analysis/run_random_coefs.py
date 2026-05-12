import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import norm
import statsmodels.api as sm

CWD = os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir))

sys.path.append(CWD)
from settings import PROD_DATA, FINANCIAL_FRICTIONS, DATA_FOLDER

sys.path.append("/home/sidlh/Documents/reusable_code")
from latex_utils.reg_to_table import coefs_to_table


def within_operator(y, x, within_transform=True, param_estimate=False):
    """
    Computes the within (projection) operator:
        Q = I - X (X'X)^(-1) X'
    Returns Qx and/or beta = (X'X)^(-1) X'y.
    """

    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    Qy = y - x @ beta

    if within_transform and not param_estimate:
        return Qy
    if param_estimate and not within_transform:
        return beta
    if within_transform and param_estimate:
        return Qy, beta

    raise ValueError("Must specify residuals=True or params=True")


def random_coefs_individual(y, z, x):
    Qy = within_operator(y, x)
    Qz = within_operator(z, x)
    beta_denom = z.T @ Qz
    beta_nume = z.T @ Qy
    return {"beta_denom": beta_denom, "beta_nume": beta_nume}


def full_rank(x):
    return np.linalg.matrix_rank(x) == x.shape[1]


def group_full_rank(x, z):
    return full_rank(x) and full_rank(z)


def random_coefs_individual_gamma(y, z, x, beta):
    v = y - z @ beta
    Qv, gamma = within_operator(v, x, within_transform=True, param_estimate=True)
    sigma = v.T @ Qv
    x_prod = np.linalg.inv(x.T @ x)
    return {
        "gamma": gamma,
        "sigma": sigma,
        "x_prod": x_prod,
    }


def random_coefs_individual_residuals(y, z, x, beta):
    Qz = within_operator(z, x)
    v = y - z @ beta
    Qv = within_operator(v, x)
    Omega = z.T @ Qv
    G = z.T @ Qz / len(x)
    return {
        "G": G,
        "Omega": Omega,
    }


def split_by_id(ids):
    uniq, inv = np.unique(ids, return_inverse=True)
    groups = [np.where(inv == i)[0] for i in range(len(uniq))]
    return groups


def random_coefs(df, y_var, x_vars, group_vars, z_vars=None):
    groups = split_by_id(df[group_vars])

    y = df[y_var].values
    x = df[x_vars].values
    include_z_vars = z_vars is not None
    if include_z_vars:
        z = df[z_vars].values
    else:
        z = np.repeat(0, len(y)).reshape(-1, 1)

    groups = [
        g
        for g in groups
        if (group_full_rank(x[g], z[g]) if include_z_vars else full_rank(x[g]))
    ]
    beta_results = []
    T_hat = np.mean([len(g) for g in groups])

    if include_z_vars:
        for g in groups:
            beta_results.append(random_coefs_individual(y=y[g], x=x[g], z=z[g]))
        beta_results = pd.DataFrame(beta_results)
        beta_nume = np.sum(beta_results["beta_nume"], axis=0)
        beta_denom = np.sum(beta_results["beta_denom"], axis=0)
        beta = np.linalg.solve(beta_denom, beta_nume)
    else:
        beta = np.array([0])

    gamma_info = []
    beta_stds = []
    for g in groups:
        gamma_info.append(
            random_coefs_individual_gamma(y=y[g], x=x[g], z=z[g], beta=beta)
        )
        if include_z_vars:
            beta_stds.append(
                random_coefs_individual_residuals(y=y[g], x=x[g], z=z[g], beta=beta)
            )
    if include_z_vars:
        beta_stds = pd.DataFrame(beta_stds)
    gamma_info = pd.DataFrame(gamma_info)
    gammas = np.array(gamma_info["gamma"].apply(lambda x: x.reshape(-1)).to_list())
    gamma = gammas.mean(axis=0)
    x_prod = np.mean(gamma_info["x_prod"])
    N = len(groups)
    sigma = gamma_info["sigma"].mean() / (T_hat - len(x_vars))

    if include_z_vars:
        G_inv = np.linalg.inv(beta_stds["G"].mean(axis=0))
        Omega = np.array(beta_stds["Omega"].to_list()).T
        beta_var = (1 / (N * T_hat) ** 2) * G_inv @ (Omega @ Omega.T) @ G_inv.T

    gamma_gap = np.expand_dims((gammas - gamma), axis=-1)
    gamma_var_first_term = (gamma_gap @ gamma_gap.swapaxes(-1, -2)).mean(axis=0)
    gamma_var = gamma_var_first_term - sigma * x_prod

    if include_z_vars:
        return beta, beta_var, gamma, gamma_var

    return gamma, gamma_var


frictions = FINANCIAL_FRICTIONS
d_frictions = ["d_" + var for var in frictions]
z_vars = ["x_" + var for var in frictions]


def prepare_data(df):
    for z_var, var, d_var in zip(z_vars, frictions, d_frictions):
        df[z_var] = df[var] * df[d_var]

    reg_vars = frictions + d_frictions + z_vars
    for var in reg_vars:
        df[var] = df[var] / df[var].std()
    return df


df = pd.read_csv(PROD_DATA)
df = prepare_data(df)

# option 1: in proposal

beta, beta_var, gamma_init, gamma_var_init = random_coefs(
    df,
    y_var="euler_equation",
    z_vars=z_vars,
    x_vars=d_frictions,
    group_vars=["gvkey"],
)

# option 2: everything heterogeneous

gamma_all, gamma_var_all = random_coefs(
    df,
    y_var="euler_equation",
    x_vars=z_vars + d_frictions,
    group_vars=["gvkey"],
)

# option 3: completely ignore derivative

gamma, gamma_var = random_coefs(
    df,
    y_var="euler_equation",
    x_vars=frictions[1:],
    group_vars=["gvkey"],
)

# option 4: baseline OLS

model = sm.OLS(df["euler_equation"], sm.add_constant(df[frictions]))
results = model.fit()
results.params
results.tvalues


def data_to_reg_table(b, se, pvalue=None, names=None):
    if pvalue is None:
        pvalue = 2 * (1 - norm.cdf(np.abs(b / se)))
    if names is not None:
        b = pd.Series(b, index=names)
        se = pd.Series(se, index=names)
        pvalue = pd.Series(pvalue, index=names)
    return pd.DataFrame(
        [b, se, pvalue], index=pd.Series(["b", "se", "pvalue"], name="row_names")
    )


reg_data = [
    {"b": beta, "se": np.diagonal(beta_var), "names": z_vars},
    # {"b": gamma_all[:3], "t": np.diagonal(gamma_var_all)[:3], "names": z_vars},
    {"b": gamma, "se": np.diagonal(gamma_var), "names": frictions[1:]},
    {"b": results.params, "se": results.bse, "pvalue": results.pvalues},
]

result_df = [data_to_reg_table(**reg) for reg in reg_data]

table = coefs_to_table(
    result_df,
    end_cols=[],
    var_map={
        "x_leverage": "Leverage",
        "x_net_worth": "Net Worth",
        "x_var_pct": "Value at Risk",
        "leverage": "Leverage",
        "net_worth": "Net Worth",
        "var_pct": "Value at Risk",
        "const": "Constant",
    },
    display_se=True,
)
table = table.iloc[:-1]
table.index = table.index.fillna("")

print(table.fillna("").to_latex())


stata_df = df.merge(
    df.groupby("gvkey")
    .apply(lambda x: full_rank(x[frictions].values) & (x[frictions].std().min() > 0))
    .replace(False, None)
    .dropna()
    .reset_index()
    .drop(columns=[0])
)
stata_df["date"] = stata_df["date"].pipe(pd.to_datetime)
stata_df.to_stata(
    os.path.join(DATA_FOLDER, "panel_data.dta"), convert_dates={"date": "tq"}
)
