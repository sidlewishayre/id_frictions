import os
import sys
import numpy as np
import pandas as pd
import statsmodels.api as sm

CWD = os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir))

sys.path.append(CWD)
from settings import PROD_DATA, FINANCIAL_FRICTIONS


def within_operator(x, z, residuals=True, params=False):
    model = sm.OLS(x, z)
    results = model.fit()
    if residuals:
        residuals_results = results.resid
    if residuals and not params:
        return residuals_results
    if not residuals and params:
        return results.params
    if residuals and params:
        return residuals_results, results.params
    raise ValueError("Must specify residuals=True or params=True")


def random_coefs_individual(y, x, z):
    Qy = within_operator(y, z)
    Qx = within_operator(x, z)
    beta_denom = x.T @ Qx
    beta_nume = x.T @ Qy
    return {"beta_denom": beta_denom, "beta_nume": beta_nume}


def full_rank(x):
    return np.linalg.matrix_rank(x) == x.shape[1]


def group_full_rank(x, z):
    return full_rank(x) and full_rank(z)


def random_coefs_individual_gamma(y, x, z, beta):
    v = y - x @ beta
    Qv, gamma = within_operator(v, z, residuals=True, params=True)
    sigma = v.T @ Qv
    z_prod = np.linalg.inv(z.T @ z)
    return {"gamma": gamma, "sigma": sigma, "z_prod": z_prod}


def random_coefs_individual_residuals(y, x, z, beta):
    Qx = within_operator(x, z)  # TODO this is redundant
    Qv = within_operator(y - x @ beta, z)  # TODO this is redundant
    Omega = x.T @ Qv
    G = x.T @ Qx
    return {"G": G, "Omega": Omega}


def split_by_id(ids):
    uniq, inv = np.unique(ids, return_inverse=True)
    groups = [np.where(inv == i)[0] for i in range(len(uniq))]
    return groups


def random_coefs(df, y_var, x_vars, z_vars, group_vars=["gvkey"]):

    groups = split_by_id(df[group_vars])
    y = df[y_var].values
    x = df[x_vars].values
    z = df[z_vars].values

    groups = [g for g in groups if group_full_rank(x[g], z[g])]

    beta_results = []
    T_hat = np.mean([len(g) for g in groups])

    for g in groups:
        beta_results.append(random_coefs_individual(y=y[g], x=x[g], z=z[g]))
    beta_results = pd.DataFrame(beta_results)

    beta_nume = np.mean(beta_results["beta_nume"], axis=0)
    beta_denom = np.mean(beta_results["beta_denom"], axis=0)
    beta = np.linalg.solve(beta_denom, beta_nume)

    gamma_info = []
    beta_stds = []
    for g in groups:
        gamma_info.append(
            random_coefs_individual_gamma(y=y[g], x=x[g], z=z[g], beta=beta)
        )
        beta_stds.append(random_coefs_individual_residuals(y[g], x[g], z[g], beta))
    beta_stds = pd.DataFrame(beta_stds)
    gamma_info = pd.DataFrame(gamma_info)
    gammas = np.array(gamma_info["gamma"].to_list())
    gamma = gammas.mean(axis=0)
    z_prod = np.mean(gamma_info["z_prod"])
    sigma = gamma_info["sigma"].mean() * (
        len(groups) / len(groups) * (T_hat - len(z_vars))
    )

    G_inv = np.linalg.inv(beta_stds["G"].mean(axis=0))
    Omega = np.expand_dims(beta_stds["Omega"].mean(axis=0), axis=0)
    beta_var = G_inv @ (Omega * Omega.T) @ G_inv

    gamma_gap = np.expand_dims(gammas - gamma, axis=-1)
    gamma_var_first_term = (gamma_gap @ gamma_gap.swapaxes(-1, -2)).mean(axis=0)
    gamma_var = gamma_var_first_term - sigma * z_prod

    return beta, beta_var, gamma, gamma_var


df = pd.read_csv(PROD_DATA)
frictions = FINANCIAL_FRICTIONS
d_frictions = ["d_" + var for var in frictions]
x_vars = ["x_" + var for var in frictions]

for x_var, var, d_var in zip(x_vars, frictions, d_frictions):
    df[x_var] = df[var] * df[d_var]

beta, beta_var, gamma, gamma_var = random_coefs(
    df,
    y_var="euler_equation",
    x_vars=x_vars,
    z_vars=d_frictions,
)
