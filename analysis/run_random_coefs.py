import os
import sys
import numpy as np
import pandas as pd
import statsmodels.api as sm

CWD = os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir))

sys.path.append(CWD)
from settings import PROD_DATA, FINANCIAL_FRICTIONS


def within_operator(y, x, within_transform=True, param_estimate=False):
    """
    Computes the within (projection) operator:
        Q = I - Z (Z'Z)^(-1) Z'
    Returns Qx and/or beta = (Z'Z)^(-1) Z'x.
    """

    # Ensure 2D
    y = np.atleast_2d(y)
    if y.shape[0] < y.shape[1]:
        y = y.T

    x = np.atleast_2d(x)
    if x.shape[0] < x.shape[1]:
        x = x.T

    T = x.shape[0]
    I = np.eye(T)

    # Compute (Z'Z)^(-1) Z'
    XtX_inv = np.linalg.inv(x.T @ x)
    P = x @ (XtX_inv @ x.T)
    Q = I - P

    Qx = Q @ y
    beta = XtX_inv @ (x.T @ y)

    if within_transform and not param_estimate:
        return Qx
    if param_estimate and not within_transform:
        return beta
    if within_transform and param_estimate:
        return Qx, beta

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
    Qv = within_operator(y - z @ beta, x)
    Omega = z.T @ Qv
    G = z.T @ Qz
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
        beta_nume = np.mean(beta_results["beta_nume"], axis=0)
        beta_denom = np.mean(beta_results["beta_denom"], axis=0)
        beta = (np.linalg.inv(beta_denom) @ beta_nume).T[0]
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
    gamma = gammas.sum(axis=0)
    x_prod = np.sum(gamma_info["x_prod"])
    sigma = gamma_info["sigma"].mean() * (
        len(groups) / (len(groups) * (T_hat - len(x_vars)))
    )

    if include_z_vars:
        G_inv = np.linalg.inv(beta_stds["G"].mean(axis=0))
        Omega = beta_stds["Omega"].mean(axis=0)
        beta_var = G_inv @ (Omega * Omega.T) @ G_inv.T

    gamma_gap = np.expand_dims((gammas - gamma), axis=-1)
    gamma_var_first_term = (gamma_gap @ gamma_gap.swapaxes(-1, -2)).mean(axis=0)
    gamma_var = gamma_var_first_term - sigma * x_prod

    if include_z_vars:
        return beta, beta_var, gamma, gamma_var

    return gamma, gamma_var


df = pd.read_csv(PROD_DATA)
frictions = FINANCIAL_FRICTIONS
d_frictions = ["d_" + var for var in frictions]
z_vars = ["x_" + var for var in frictions]

for z_var, var, d_var in zip(z_vars, frictions, d_frictions):
    df[z_var] = df[var] * df[d_var]


# option 1: in proposal

beta, beta_var, gamma, gamma_var = random_coefs(
    df,
    y_var="euler_equation",
    z_vars=z_vars,
    x_vars=d_frictions,
    group_vars=["gvkey"],
)

# option 2: everything heterogeneous

gamma, gamma_var = random_coefs(
    df,
    y_var="euler_equation",
    x_vars=z_vars + d_frictions,
    group_vars=["gvkey"],
)

# option 3: drop constant term

gamma, gamma_var = random_coefs(
    df,
    y_var="euler_equation",
    x_vars=z_vars,
    group_vars=["gvkey"],
)

# option 4: completely ignore derivative

gamma, gamma_var = random_coefs(
    df,
    y_var="euler_equation",
    x_vars=frictions,
    group_vars=["gvkey"],
)


# model = sm.OLS(df["euler_equation"], df[frictions])
# results = model.fit()
# results.params
# results.summary()

df[frictions].corr()
df[z_vars].corr()
df[frictions + d_frictions].corr()
