import os
import sys
sys.path.insert(1, '../')
import numpy as np
import folktables
import pdb
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import seaborn as sns
import pandas as pd

import scipy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import LogLocator
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import NullFormatter

def get_data(year,features,outcome, randperm=True):
    # Predict income and regress to time to work
    data_source = folktables.ACSDataSource(survey_year=year, horizon='1-Year', survey='person')
    acs_data = data_source.get_data(states=["CA"], download=True)
    income_features = acs_data[features].fillna(-1)
    income = acs_data[outcome].fillna(-1)
    employed = np.isin(acs_data['COW'], np.array([1,2,3,4,5,6,7]))
    if randperm:
        shuffler = np.random.permutation(income.shape[0])
        income_features, income, employed = income_features.iloc[shuffler], income.iloc[shuffler], employed[shuffler]
    return income_features, income, employed

def transform_features(features, ft, enc=None):
    c_features = features.T[ft == "c"].T.astype(str)
    if enc is None:
        enc = OneHotEncoder(handle_unknown='ignore', drop='if_binary', sparse=False)
        enc.fit(c_features)
    c_features = enc.transform(c_features)
    features = scipy.sparse.csc_matrix(np.concatenate([features.T[ft == "q"].T.astype(float), c_features], axis=1))
    return features, enc

def ols(features, outcome):
    ols_coeffs = np.linalg.pinv(features).dot(outcome)
    return ols_coeffs

def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])


def make_width_coverage_plot(df, estimand_title, filename, theta_true, alpha = 0.1, n_l = 0, n_u = np.inf, num_trials = 100, n_example_ind = 0, finetuning=False, more_precision=False, less_precision=False):
    num_ints = 5
    inds = np.random.choice(num_trials, num_ints)
    ns = df["$n_b$"].unique()
    estimators = df["estimator"].unique()
    n_example = ns[n_example_ind]
    ints = [ [] for _ in range(len(estimators)) ]
    widths = np.zeros((len(estimators), len(ns)))

    # compute example intervals and average widths
    for i in range(len(estimators)):
        for j in range(len(ns)):
            widths[i,j] = df[(df.estimator == estimators[i]) & (df["$n_b$"] == ns[j])]['interval width'].mean()
    
        for j in range(num_ints):
            ind = inds[j]
            ints[i].append([df[(df.estimator == estimators[i]) & (df['$n_b$'] == n_example)].iloc[ind].lb, df[(df.estimator == estimators[i]) & (df['$n_b$'] == n_example)].iloc[ind].ub])

    n_l = n_l
    n_u = n_u
    inds_n = np.where((ns>n_l) & (ns<n_u))[0] # budget indices that will be plotted
    x_ticks = np.logspace(np.log10(min(df['$n_b$'][(df['$n_b$'] > n_l)])), np.log10(max(df['$n_b$'][(df['$n_b$'] < n_u)])), num=5) # adjust 'num' for more/less ticks
    x_ticks = [int(x) for x in x_ticks]
    y_ticks = np.logspace(np.log10(np.min(widths[:,inds_n[-1]])), np.log10(np.max(widths[:,inds_n[0]])), num=5) # adjust 'num' for more/less ticks

    # plotting params
    gap = 0.03
    start1 = 0.5
    start2 = 0.35
    start3 = 0.2
    linewidth_inner = 5
    linewidth_outer = 7
    col = [sns.color_palette("pastel")[1], sns.color_palette("pastel")[2], sns.color_palette("pastel")[0]]
    sns.set_theme(font_scale=1.5, style='white', palette=col, rc={'lines.linewidth': 3})
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(15,3.3))

    sns.lineplot(ax=axs[1],data=df[(df['$n_b$'] > n_l) & (df['$n_b$'] < n_u)], x='$n_b$', y='interval width', hue='estimator', alpha=0.8)
    sns.lineplot(ax=axs[2],data=df[(df['$n_b$'] > n_l) & (df['$n_b$'] < n_u)], x='$n_b$', y='coverage', hue='estimator', alpha=0.8, errorbar=None)

    axs[0].axvline(theta_true, color='gray', linestyle='dashed')
    for i in reversed(range(num_ints)):
        if i == 0:
            axs[0].plot([ints[0][i][0] , ints[0][i][1] ],[start1+i*gap,start1+i*gap], linewidth=linewidth_inner, color=lighten_color(col[0],0.6), path_effects=[pe.Stroke(linewidth=linewidth_outer, offset=(-1,0), foreground=col[0]), pe.Stroke(linewidth=linewidth_outer, offset=(1,0), foreground=col[0]), pe.Normal()],  solid_capstyle='butt')
            axs[0].plot([ints[1][i][0] , ints[1][i][1] ],[start2+i*gap, start2+i*gap], linewidth=linewidth_inner, color=lighten_color(col[1],0.6), path_effects=[pe.Stroke(linewidth=linewidth_outer, offset=(-1,0), foreground=col[1]), pe.Stroke(linewidth=linewidth_outer, offset=(1,0), foreground=col[1]), pe.Normal()],  solid_capstyle='butt')
            axs[0].plot([ints[2][i][0] , ints[2][i][1] ],[start3+i*gap, start3+i*gap], linewidth=linewidth_inner, color=lighten_color(col[2],0.6), path_effects=[pe.Stroke(linewidth=linewidth_outer, offset=(-1,0), foreground=col[2]), pe.Stroke(linewidth=linewidth_outer, offset=(1,0), foreground=col[2]), pe.Normal()],  solid_capstyle='butt')
        if i > 0:
            axs[0].plot([ints[0][i][0], ints[0][i][1]],[start1+i*gap,start1+i*gap], linewidth=linewidth_inner, color= lighten_color(col[0],0.6), path_effects=[pe.Stroke(linewidth=linewidth_outer, offset=(-1,0), foreground=col[0]), pe.Stroke(linewidth=linewidth_outer, offset=(1,0), foreground=col[0]), pe.Normal()], solid_capstyle='butt')
            axs[0].plot([ints[1][i][0] , ints[1][i][1]],[start2+i*gap, start2+i*gap], linewidth=linewidth_inner, color=lighten_color(col[1],0.6), path_effects=[pe.Stroke(linewidth=linewidth_outer, offset=(-1,0), foreground=col[1]), pe.Stroke(linewidth=linewidth_outer, offset=(1,0), foreground=col[1]), pe.Normal()], solid_capstyle='butt')
            axs[0].plot([ints[2][i][0] , ints[2][i][1]],[start3+i*gap, start3+i*gap], linewidth=linewidth_inner, color=lighten_color(col[2],0.6), path_effects=[pe.Stroke(linewidth=linewidth_outer, offset=(-1,0), foreground=col[2]), pe.Stroke(linewidth=linewidth_outer, offset=(1,0), foreground=col[2]), pe.Normal()], solid_capstyle='butt')
    axs[0].set_xlabel(estimand_title, fontsize=16)
    axs[0].set_yticks([])
    
    axs[1].get_legend().remove()
    axs[1].set(xscale='log', yscale='log')
    axs[1].set_xticks(x_ticks)
    axs[1].set_yticks(y_ticks)
    axs[1].xaxis.set_minor_formatter(NullFormatter())
    axs[1].yaxis.set_minor_formatter(NullFormatter())
    axs[1].get_xaxis().set_major_formatter(ScalarFormatter())
    axs[1].get_yaxis().set_major_formatter(ScalarFormatter())
    axs[1].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    if more_precision:
        axs[1].yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    if less_precision:
        axs[1].yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    axs[1].grid(True)
    
    axs[2].axhline(1-alpha, color="#888888", linestyle='dashed', zorder=1, alpha=0.8)
    handles, labels = axs[2].get_legend_handles_labels()
    if finetuning: # slighly nicer ordering in fine-tuning plots
        handles[1:3] = handles[2], handles[1]
        labels[1:3] = labels[2], labels[1]
    axs[2].legend(handles=handles, labels=labels, loc='lower right')
    axs[2].set_ylim([0.6,1])
    axs[2].grid(True)
    
    sns.despine(top=True, right=True)
    plt.tight_layout()
    
    # save plot
    plt.savefig(filename)
    plt.show()


def make_budget_plot(df, plot_title, filename, finetuning=False):
    ns = df["$n_b$"].unique()
    estimators = df["estimator"].unique()
    widths = np.zeros((len(estimators), len(ns)))

    # compute average widths
    for i in range(len(estimators)):
        for j in range(len(ns)):
            widths[i,j] = df[(df.estimator == estimators[i]) & (df["$n_b$"] == ns[j])]['interval width'].mean() 
            
    save1 = []
    save2 = []

    ns_large1 = ns[np.where(widths[0,0] > widths[1,:])]
    for n in ns_large1:
        target_width = df[(df.estimator == estimators[1]) & (df["$n_b$"] == n)]['interval width'].mean()
        active_0 = np.where(np.array(widths[0,:]) > target_width)[0][-1]
        active_1 = active_0 + 1
        # linearly interpolate:
        active_n = round((widths[0,active_0] - target_width)/(widths[0,active_0] - widths[0,active_1])*(ns[active_1] - ns[active_0]) + ns[active_0])
        save1.append((n - active_n)/n*100)

    ns_large2 = ns[np.where(widths[0,0] > widths[2,:])]
    for n in ns_large2:
        target_width = df[(df.estimator == estimators[2]) & (df["$n_b$"] == n)]['interval width'].mean()
        active_0 = np.where(np.array(widths[0,:]) > target_width)[0][-1]
        active_1 = active_0 + 1
        # linearly interpolate:
        active_n = round((widths[0,active_0] - target_width)/(widths[0,active_0] - widths[0,active_1])*(ns[active_1] - ns[active_0]) + ns[active_0])
        save2.append((n - active_n)/n*100)


    col = [sns.color_palette("pastel")[1]]
    sns.set_theme(font_scale=1.7, style='white', palette=col, rc={'lines.linewidth': 3})
    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(6,6))
    
    axs[0].set_title(plot_title)
    axs[0].plot(ns_large2, save2)
    axs[0].set_xlabel('$n_b$')
    if finetuning:
        axs[0].set_ylabel('budget save over\n no fine-tuning (%)')
    else:
        axs[0].set_ylabel('budget save\n over classical (%)')
    axs[0].get_yaxis().set_major_formatter(FormatStrFormatter('%.0f'))
    axs[0].grid(True)

    axs[1].plot(ns_large1, save1)
    axs[1].set_xlabel('$n_b$')
    axs[1].set_ylabel('budget save\n over uniform (%)')
    axs[1].get_yaxis().set_major_formatter(FormatStrFormatter('%.0f'))
    axs[1].grid(True)

    sns.despine(top=True, right=True)
    plt.tight_layout()
    
    # save plot
    plt.savefig(filename)
    plt.show()