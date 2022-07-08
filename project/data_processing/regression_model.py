import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.style as style
from tqdm import tqdm
import os
from data_processing import helper_functions, find_fips_for_depts
import pymc as pm


def fit_model(df, regressor_names, dependent_variable_name, n_samples=2000):
    n_betas = len(regressor_names)
    dept_idxs, depts = pd.factorize(df.department_name)
    with pm.Model() as hierarchical_model:
        mu_betas = pm.Normal("mu_betas", 0, 5, shape=n_betas)
        var_betas = pm.InverseGamma('var_betas', alpha=0.5, beta=0.5, shape=n_betas)
        betas_offset = pm.Normal('betas_offset', mu=0, sigma=1, shape=(len(depts), n_betas))
        betas = pm.Deterministic('betas', mu_betas + betas_offset * np.sqrt(var_betas))
        log_mean = (df[regressor_names].to_numpy() * betas[dept_idxs, :]).sum(axis=1)
        alpha = pm.Exponential("alpha", lam=0.1, shape=len(depts))
        data_like = pm.NegativeBinomial("y", mu=np.exp(log_mean) + 1e-6, alpha=alpha[dept_idxs],
                                        observed=df[dependent_variable_name])
    with hierarchical_model:
        hierarchical_trace = pm.sample(n_samples, tune=500, chains=4, init='advi', return_inferencedata=False)
    return hierarchical_model, hierarchical_trace

