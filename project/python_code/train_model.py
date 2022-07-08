import numpy as np
import pandas as pd
import os
from sklearn.decomposition import PCA
import pymc as pm
from scipy.spatial import distance_matrix
from scipy.linalg import block_diag
from scipy.stats import multivariate_normal, norm
import pickle

df = pd.read_csv('../formatted_data/transformed_data.csv')
df = df[df['department_name'].isin(np.unique(df['department_name']))]
grouped_df = df.groupby('department_name')
groups = list(grouped_df.groups.keys())
X_depts = list(grouped_df.apply(lambda x: np.array(x[['intercept', 'component_1', 'component_2']])))
y_depts = list(grouped_df.apply(lambda x: np.array(x[['avg_change_baseline_incidents']]).flatten()))
df = df.sort_values(['department_name', 'day_of_the_year'])

dist = distance_matrix(df['day_of_the_year'].to_numpy().reshape(-1, 1), df['day_of_the_year'].to_numpy().reshape(-1, 1))

dept_idxs, dept_names = pd.factorize(df.department_name)
n_depts = len(df.groupby('department_name').groups)
group_sizes = df.groupby('department_name').size().to_numpy()
mats = [np.ones([group_sizes[i], group_sizes[i]]) for i in range(len(group_sizes))]
boolean_block_mat = block_diag(*mats)

with pm.Model() as model:
    mus_betas = pm.Normal('mus_beta', mu=0.0, sigma=10, shape=3)
    sigmas_sq_betas = pm.InverseGamma('sigmas_squared_betas', alpha=0.5, beta=0.5, shape=3)
    betas_offset = pm.Normal('betas_offset', mu=0, sigma=1, shape=(len(dept_names), 3))
    betas = pm.Deterministic('betas', mus_betas + betas_offset * np.sqrt(sigmas_sq_betas))
    means = (df[['intercept', 'component_1', 'component_2']].to_numpy() * betas[dept_idxs, :]).sum(axis=1)

    tau_s1 = pm.InverseGamma("tau_s1", alpha=0.5, beta=0.5, shape=len(dept_names))
    b_mean = pm.TruncatedNormal('b_mean', mu=20, sigma=50, lower=0)
    b_var = pm.InverseGamma('b_var', alpha=0.5, beta=0.5)
    bandwidth = pm.TruncatedNormal('b', mu=b_mean, sigma=np.sqrt(b_var), lower=0, shape=n_depts)
    full_mat = tau_s1[dept_idxs] * np.exp(-dist ** 2 / bandwidth[dept_idxs] ** 2) + np.eye(len(dept_idxs)) * 1e-6
    cov = full_mat * boolean_block_mat
    y_ = pm.MvNormal('y', mu=means, cov=cov, shape=dist.shape[0], observed=df['avg_change_baseline_incidents'])

with model:
    trace = pm.sample(2500, tune=500, init='advi+adapt_diag', target_accept=.99, chains=4, return_inferencedata=False)

var_list = ['mus_beta', 'sigmas_squared_betas', 'betas', 'tau_s1', 'b_mean', 'b_var', 'b']
traces_dict = [{var: trace.get_values(var)} for var in var_list]

with open('traces.pickle', 'wb') as handle:
    pickle.dump(traces_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
