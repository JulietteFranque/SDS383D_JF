from scipy.stats import nbinom
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def add_dummies_to_df(df, dropped_cols=None):
    """
    add dummies to df and drop two reference columns
    """
    if dropped_cols is None:
        dropped_cols = ['Monday', 'February']
    df_with_dummies = pd.concat(
        [df, pd.get_dummies(df['day_of_week']), pd.get_dummies(df['month_name'])], axis=1)
    df_with_dummies = df_with_dummies.drop(dropped_cols, axis=1)
    return df_with_dummies


def clean_raw_data(df):
    """
    initial data clean on raw data
    """
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day_of_week'] = df['date'].apply(lambda x: x.day_name())
    df = df[df['date'] >= pd.to_datetime('2019-1-1')]
    df = df[df['date'] <= pd.to_datetime('2021-1-1')]
    df['month_name'] = df['date'].apply(lambda x: x.month_name())
    df['department_name'] = df['department_name'].str.replace(r" \(.*\)", '', regex=True)
    df = df.groupby('department_name').filter(lambda x: np.sum(x['count']) > 0)
    df = df[df['department_name'] != 'Fairfax County Fire and Rescue Department']
    return df


def remove_outliers(df_count, sig=.05):
    """
    fits binomial distribution to data using moment matching and removes outliers
    """
    mean = np.mean(df_count['count'])
    var = np.var(df_count['count'])
    if var <= mean:
        var = mean + 10 ** -6
    p = mean / var
    n = mean * p / (1 - p)
    dist = nbinom(n, p)
    lower_interval_bound = dist.ppf(sig / 2)
    upper_interval_bound = dist.ppf(1 - sig / 2)
    df_count = df_count[(df_count['count'] > lower_interval_bound) & (df_count['count'] < upper_interval_bound)]
    return df_count


def average_baseline_change_weekly(df):
    """
    returns df containing average change from baseline for incidents
    """
    df = df.sort_values('date')
    df['avg_change_baseline_incidents'] = df['change_from_baseline'].rolling(window=7, min_periods=1).mean(skipna=True)
    first_day_of_week = 'Monday'
    weekly_df = df[df['day_of_week'] == first_day_of_week]
    return weekly_df[['date', 'avg_change_baseline_incidents']].sort_values('date')


def average_mobility_data(df):
    """
    returns df containing average change from baseline for google data
    """
    first_day_of_week = 'Monday'
    types = df.columns[2:7]
    new_col_names = [name + '_avg' for name in types]
    df[new_col_names] = df[types].rolling(window=7, min_periods=1).mean(skipna=True)
    weekly_df = df[df['day_of_week'] == first_day_of_week]
    return weekly_df[['date'] + new_col_names]


def calculate_baseline(df):
    """
    calculates baseline for incidents
    """
    baseline_month_day = df[df['year'] == 2019].groupby(['month', 'day_of_week'])['count'].median().reset_index(
        name='base_count')
    return baseline_month_day


def apply_baseline_to_df(df):
    """
    applies baseline to df
    """
    baseline_month_day = calculate_baseline(df)
    merged_df = pd.merge(df, baseline_month_day, how='inner', on=['month', 'day_of_week'])
    merged_df['change_from_baseline'] = (merged_df['count'] - merged_df['base_count']) / merged_df['base_count'] * 100
    return merged_df.sort_values(['date'])


def transform_data_with_pca(df):
    """
    returns df with first two principal components
    """
    pca = PCA(n_components=2)
    mobility_columns = df.columns[2:7]
    X_transformed = pca.fit_transform(df[mobility_columns])
    df['component_1'] = X_transformed[:, 0]
    df['component_2'] = X_transformed[:, 1]
    df['intercept'] = 1
    return df[['date', 'department_name', 'intercept', 'component_1', 'component_2', 'avg_change_baseline_incidents']]


def remove_gaps(input_df, gap_threshold=4):
    """
    Takes in a dataframe of dates/counts and returns one without gaps,
    i.e. intervals of more than gap_threshold consecutive zero-incident
    days.
    Parameters
    ----------
    input_df: pandas.DataFrame
            Input dataframe with dates and incident counts
    gap_threshold:	int
            The number of consecutive zero call days required to be identified
            as a gap
    Returns
    -------
    result_df: pandas.DataFrame
            The input_df with all gaps removed
    """
    df = input_df.copy()
    df['zero_mask'] = (df['count'] == 0).astype(int)
    df['group'] = (df['zero_mask'].diff(1) == -1).astype('int').cumsum()
    zero_df = df[df['count'] == 0].reset_index(drop=True)
    group_results = zero_df.groupby('group').size().reset_index(
        name='size')
    groups_to_exclude = group_results[
        group_results['size'] >= gap_threshold]['group'].values
    dates_to_exclude = zero_df[zero_df['group'].isin(
        groups_to_exclude)]['date']
    result_df = input_df[~input_df['date'].isin(dates_to_exclude
                                                )].reset_index(drop=True)
    return result_df
