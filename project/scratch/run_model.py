import pandas as pd
import numpy as np
from tqdm import tqdm
import new_gibbs
from sklearn.decomposition import PCA
import pickle
import sys

def run_chain(chain_number):
    gb = new_gibbs.GibbsSampler(X_department, y_department, time_vectors, n_iter=100000, burn=0, bandwidth_start=5, tau_sq_1_start=1000, f_start=5)
    gb.fit()
    with open(f'traces{chain_number}.pickle', 'wb') as handle:
        pickle.dump(gb.traces, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    chain = sys.argv[1]
    pca = PCA(n_components=2)
    df = pd.read_csv('../notebooks/all_data.csv')
    df = df.drop(columns='transit_stations_percent_change_from_baseline')
    df = df.dropna()
    X_transformed = pca.fit_transform(df[df.columns[np.r_[1:6]]])
    df['component_1'] = X_transformed[:, 0]
    df['component_2'] = X_transformed[:, 1]
    df['intercept'] = 1
    grouped_df = df.groupby('department_name')
    groups = list(grouped_df.groups.keys())
    X_department, y_department = [], []
    time_vectors = []
    for n, dept in enumerate(groups):
        df_dept = grouped_df.get_group(dept)
        df_dept = df_dept.sort_values('date')
        X = df_dept[['intercept', 'component_1', 'component_2']].to_numpy()
        y = df_dept['incidents'].to_numpy()
        X_department.append(X)
        y_department.append(y.flatten())
        df_dept['day_of_the_year'] = pd.to_datetime(df_dept['date']).dt.day_of_year
        time_vectors.append(df_dept['day_of_the_year'].to_numpy())

    run_chain(chain)



