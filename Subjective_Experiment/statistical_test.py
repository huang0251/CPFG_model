import os
import pandas as pd
from scipy.stats import kruskal, mannwhitneyu
from statsmodels.stats.multitest import multipletests
import itertools
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

file_path = 'results.csv'
df = pd.read_csv(file_path)
df['Score'] = pd.to_numeric(df['Score'], errors='coerce')

def stars(p):
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    else:
        return 'ns'

systems = sorted(df['System'].unique())

for metric in df['Metric'].unique():
    sub = df[df['Metric'] == metric]
    print(f"\n===== Kruskal-Wallis for {metric} =====")
    groups = [sub[sub['System']==s]['Score'] for s in systems]
    H, p_kw = kruskal(*groups)
    print(f"H = {H:.4f}, p = {p_kw:.4e}")

    pairs = list(itertools.combinations(systems, 2))
    pvals = []
    for s1, s2 in pairs:
        u, p = mannwhitneyu(sub[sub['System']==s1]['Score'],
                            sub[sub['System']==s2]['Score'],
                            alternative='two-sided')
        pvals.append(p)

    adj = multipletests(pvals, method='bonferroni')[1]
    pair_results = dict(zip(pairs, adj))

    print(f"\nPairwise Mann-Whitney for {metric}:")
    for (s1, s2), p_adj in pair_results.items():
        print(f"{s1} vs {s2}: p_adj = {p_adj:.4e} ({stars(p_adj)})")

    data = [sub[sub['System']==s]['Score'] for s in systems]
    plt.figure()
    plt.boxplot(data, labels=systems)
    plt.title(f"{metric} Scores by System")
    plt.ylabel('Score')

    y_max = sub['Score'].max()
    y_min = sub['Score'].min()
    h = (y_max - y_min) * 0.05
    for i, (s1, s2) in enumerate(pairs):
        p_adj = pair_results[(s1, s2)]
        star = stars(p_adj)
        if star != 'ns':
            x1, x2 = systems.index(s1)+1, systems.index(s2)+1
            y = y_max + h * (i+1)
            plt.plot([x1, x1, x2, x2], [y, y+h/2, y+h/2, y], lw=1.5)
            plt.text((x1+x2)*0.5, y+h/2, star, ha='center', va='bottom')
    plt.ylim(bottom=y_min - h, top=y + h*1.5)
    plt.show()

def plot_bar_with_significance(df, metric_name='CN'):
    df_metric = df[df['Metric'] == metric_name]

    plt.figure(figsize=(6, 4))

    # mean + ci
    sns.barplot(data=df_metric, x='System', y='Score', errorbar=('ci', 95),
                capsize=0.1, errwidth=1.5, palette='deep')


    plt.ylim(1, 6)
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    # calculate Mann–Whitney U + Bonferroni
    systems = df_metric['System'].unique()
    pairs = list(itertools.combinations(systems, 2))

    p_vals = []
    for s1, s2 in pairs:
        group1 = df_metric[df_metric['System'] == s1]['Score']
        group2 = df_metric[df_metric['System'] == s2]['Score']
        _, p = mannwhitneyu(group1, group2, alternative='two-sided')
        p_vals.append(p)

    # Bonferroni
    reject, pvals_corrected, _, _ = multipletests(p_vals, method='bonferroni')

    # stars
    for i, ((s1, s2), p_corr) in enumerate(zip(pairs, pvals_corrected)):
        if p_corr < 0.05:
            star = ''
            if p_corr < 0.001:
                star = '***'
            elif p_corr < 0.01:
                star = '**'
            else:
                star = '*'

            x1 = systems.tolist().index(s1)
            x2 = systems.tolist().index(s2)
            y = df_metric['Score'].max() - 0.3 + i * 0.3

            plt.plot([x1, x1, x2, x2], [y, y + 0.1, y + 0.1, y],
                     lw=1.5, color='black')
            plt.text((x1 + x2) / 2, y + 0.1, star,
                     ha='center', va='bottom', color='black', fontsize=12)

    plt.tight_layout()
    plt.show()
