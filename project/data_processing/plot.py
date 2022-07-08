import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np

plt.rcParams.update({'axes.labelsize': 16})
plt.rcParams.update({'axes.titlesize': 16})
plt.rcParams.update({'legend.fontsize': 16})
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16
plt.rcParams['lines.linewidth'] = 4
plt.style.use('ggplot')


def plot_outlier_removal(df, path):
    plt.figure(figsize=(20, 170), facecolor='white')
    departments = df['department_name'].unique()
    for n, dept in enumerate(departments):
        plt.subplot(len(departments), 2, 2 * n + 1)
        df_not_avg = df_no_outliers[df_no_outliers['department_name'] == dept]
        df_raw = df[df['department_name'] == dept]
        plt.scatter(df_raw['date'], df_raw['count'], alpha=.5, label='raw number of incidents')
        plt.xlim(pd.to_datetime('2019-1-1'), pd.to_datetime('2021-1-1'))
        plt.title(dept, fontsize=16)
        plt.ylim(df_raw['count'].min(), df_raw['count'].max())
        if n == 0:
            plt.legend()
        plt.subplot(len(departments), 2, 2 * n + 2)
        plt.scatter(df_not_avg['date'], df_not_avg['count'], alpha=.5, label='cleaned number of incidents',
                    color='dodgerblue')
        plt.xlim(pd.to_datetime('2019-1-1'), pd.to_datetime('2021-1-1'))
        plt.title(dept, fontsize=16)
        plt.ylim(df_raw['count'].min(), df_raw['count'].max())
        if n == 0:
            plt.legend()
        plt.tight_layout()
        plt.savefig(path, dpi=300, bbox_to_inches='tight')


def plot_alpha_traces(traces, depts, path):
    fig, ax = plt.subplots(nrows=traces.get_values('alpha').shape[1], figsize=(8, 100), facecolor='white')
    for i, dept in enumerate(depts):
        ax[i].hist(traces.get_values('alpha')[:, i], density=True, alpha=.5, bins=100)
        ax[i].set_ylabel(fr'$\alpha$')
        ax[i].set_title(dept)
        ax[i].set_xlim(0, 300)
    fig.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches='tight')


def plot_mu_beta_traces(traces, path, regressors):
    n_betas = traces.get_values('mu_betas').shape[1]
    fig, ax = plt.subplots(nrows=n_betas, figsize=(8, 30), facecolor='white')
    for i in range(n_betas):
        ax[i].hist(traces.get_values('mu_betas')[:, i], density=True, alpha=.5, bins=100)
        ax[i].set_ylabel(fr'$\beta${regressors[i]}')
    ax[0].set_title('posterior means of coefficients')
    fig.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches='tight')


def plot_fit(df, sample_posterior_predictive, path):
    mean_ppc = sample_posterior_predictive['y'].mean(axis=0)
    df_samples = pd.DataFrame(sample_posterior_predictive['y']).T
    dept_idxs, depts = pd.factorize(df.department_name)
    df_samples.insert(0, 'department', depts[dept_idxs])
    df['map'] = mean_ppc
    fig, ax = plt.subplots(nrows=5, ncols=13, figsize=(60, 18), facecolor='white')
    for n, dept in enumerate(depts):
        minn, maxx = df[df['department_name'] == depts[n]]['count'].min(), \
                     df[df['department_name'] == dept]['count'].max()
        df_dept = df[df['department_name'] == dept]
        ax.flatten()[n].plot([minn, maxx], [minn, maxx], color='black')
        ax.flatten()[n].scatter(df_dept['count'], df_dept['map'])
        ax.flatten()[n].set_xlabel('data')
        ax.flatten()[n].set_ylabel('mean of simulated data')
        ax.flatten()[n].set_title(dept)
    fig.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches='tight')


def plot_samples(df, sample_posterior_predictive, path):
    mean_ppc = sample_posterior_predictive['y'].mean(axis=0)
    df['map'] = mean_ppc
    fig, ax = plt.subplots(nrows=13, ncols=5, figsize=(40, 60), facecolor='white')
    df_samples = pd.DataFrame(sample_posterior_predictive['y']).T
    dept_idxs, depts = pd.factorize(df.department_name)
    df_samples.insert(0, 'department', depts[dept_idxs])
    for n, dept in enumerate(depts):
        df_dept = df[df['department_name'] == dept]
        df_samples_dept = df_samples[df_samples['department'] == dept]
        ax.flatten()[n].scatter(df_dept['date'], df_dept['count'], zorder=3, label='data')
        ax.flatten()[n].plot(df_dept['date'], df_dept['map'], color='black', zorder=3, label='mean of samples')
        ax.flatten()[n].plot(df_dept['date'], df_samples_dept[range(0, 20)], color='dodgerblue', alpha=.1)
        ax.flatten()[n].set_xlabel('time')
        ax.flatten()[n].set_ylabel('counts')
        ax.flatten()[n].set_title(dept)
        ax.flatten()[n].xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        ax.flatten()[n].legend()
    fig.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches='tight')


def plot_betas(betas, depts, regressors, path):
    n_betas = betas.shape[1]
    fig, ax = plt.subplots(nrows=n_betas, figsize=(40, 60), facecolor='white')
    dept_nums = np.arange(0, len(depts), 1)
    for beta in range(n_betas):
        neg_bool = betas[:, beta] > 0
        # plt.xticks(depts, fontsize=14)
        ax[beta].axhline(y=0, color='black', alpha=.3)
        if beta == n_betas - 1:
            ax[beta].set_xticks(range(len(depts)), depts, rotation=90)
        else:
            ax[beta].set_xticks(range(len(depts)), depts, rotation=90, color='white')
        ax[beta].scatter(dept_nums[neg_bool], betas[:, beta][neg_bool], s=200, zorder=3)
        ax[beta].scatter(dept_nums[~neg_bool], betas[:, beta][~neg_bool], color='dodgerblue', edgecolor='dodgerblue',
                         s=200, zorder=3)
        _, stemlines, baseline = ax[beta].stem(dept_nums, betas[:, beta])
        plt.setp(stemlines, 'linewidth', 3, 'color', 'black')
        ax[beta].set_ylabel(regressors[beta], fontsize=20)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())


def make_box_plot(trace, regressors, depts, path):
    n_betas = trace.get_values('betas').shape[2]
    fig, ax = plt.subplots(nrows=n_betas, figsize=(40, 60), facecolor='white')
    for beta in range(n_betas):
        ax[beta].axhline(y=0, color='red', alpha=.5, lw=2)
        ax[beta].boxplot(trace.get_values('betas')[:, :, beta], vert=True, showfliers=False,
                         showmeans=True, meanline=True)
        ax[beta].set_ylabel(regressors[beta], fontsize=20)
        if beta == n_betas - 1:
            ax[beta].set_xticks(range(len(depts)), depts, rotation=90)
        else:
            ax[beta].set_xticks([], color='white')
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
