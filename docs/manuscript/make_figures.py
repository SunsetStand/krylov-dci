#!/usr/bin/env python
"""Figures for docs/manuscript/self_consistent_schmidt_downfolding.md.

Reads only small committed JSON under results/. Run from the repository root:
    python docs/manuscript/make_figures.py
"""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter, NullFormatter  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, 'docs', 'manuscript', 'figures')
os.makedirs(OUT, exist_ok=True)

# Reference categorical palette, light mode, fixed order (dataviz references/palette.md).
SLOT = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100']
INK, INK2, GRID = '#0b0b0b', '#52514e', '#d9d8d4'
MARK = ['o', 's', '^', 'D']
STATES = ['S0 $A_g$', 'S1 $B_{1u}$', 'S2 $B_{2g}$', 'S3 $B_{3g}$']
CHEM = 1.6

plt.rcParams.update({
    'font.size': 9, 'axes.labelsize': 9, 'axes.titlesize': 9,
    'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8,
    'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2,
    'ytick.color': INK2, 'text.color': INK, 'axes.linewidth': 0.8,
    'savefig.dpi': 300, 'savefig.bbox': 'tight',
})


def load(path):
    with open(os.path.join(ROOT, path)) as handle:
        return json.load(handle)


def style(ax):
    ax.grid(True, which='major', color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)


def figure_threshold_curves():
    default = load('results/symm_seed/n2_local_default_seed_curve.json')['four_state']
    symm = [p for p in load('results/symm_seed/n2_cluster_job20910.json')['points']
            if p['slurm_job'].startswith('20910') and p['steps'] == 10]
    symm = sorted(symm, key=lambda p: p['D'])
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9), sharey=True)
    panels = [(axes[0], default, '(a) default seed, lanczos'),
              (axes[1], symm, '(b) irrep-complete seed, lanczos_symm')]
    for ax, rows, title in panels:
        style(ax)
        ax.axhline(CHEM, color=INK2, linewidth=1.0, linestyle='--', zorder=1)
        D = [r['D'] for r in rows]
        for k in range(4):
            err = [r['per_state_error_mH'][k] for r in rows]
            ax.plot(D, err, color=SLOT[k], linewidth=1.8, zorder=2,
                    label=STATES[k])
            for d, e, r in zip(D, err, rows):
                filled = r['converged']
                ax.plot(d, e, marker=MARK[k], markersize=6, zorder=3,
                        color=SLOT[k],
                        markerfacecolor=SLOT[k] if filled else 'white',
                        markeredgecolor=SLOT[k], markeredgewidth=1.3)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:,.0f}'))
        ax.set_xlabel('embedded dimension $D$')
        ax.set_title(title, loc='left', color=INK)
    axes[0].set_ylabel('error against exact CASCI (mH)')
    axes[0].set_xticks([200, 500, 1000, 2000, 5000])
    axes[1].set_xticks([5000, 7000, 10000])
    axes[0].text(225, CHEM * 1.12, 'chemical accuracy, 1.6 mH',
                 color=INK2, fontsize=7.5)
    # direct labels at the right end; the degenerate pair is labelled once
    last_a = default[-1]['per_state_error_mH']
    last_b = symm[-1]['per_state_error_mH']
    for ax, rows, last in ((axes[0], default, last_a), (axes[1], symm, last_b)):
        x = rows[-1]['D']
        for y, text in ((last[0], 'S0'), (last[1], 'S1'),
                        ((last[2] * last[3]) ** 0.5, '$^3\\Pi_g$ pair')):
            ax.annotate(text, (x, y), xytext=(7, 0), textcoords='offset points',
                        color=INK, fontsize=7.5, va='center',
                        annotation_clip=False)
    fig.subplots_adjust(bottom=0.30, wspace=0.18, right=0.90)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 0.04))
    fig.text(0.5, 0.0, 'open markers: outer loop not converged',
             ha='center', color=INK2, fontsize=7.5)
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(OUT, f'fig_threshold_curves.{ext}'))
    plt.close(fig)


def figure_lanczos():
    data = load('results/symm_seed/lanczos_orthogonality_and_cost.json')
    steps = data['steps']
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7))
    fig.subplots_adjust(wspace=0.42)
    ax = axes[0]
    style(ax)
    ax.plot(steps, data['gram_error'], color=SLOT[0], linewidth=1.8,
            marker='o', markersize=6)
    ax.set_yscale('log')
    ax.set_xlabel('Lanczos steps')
    ax.set_ylabel(r'$\max|K^{\mathsf{T}}K - I|$')
    ax.set_title('(a) Krylov basis orthogonality', loc='left', color=INK)
    ax = axes[1]
    style(ax)
    ax.axhline(0.0, color=INK2, linewidth=1.0, linestyle='--')
    # orthonormalized drawn first and larger, so coincident points show both
    ax.plot(steps, data['lowest_ritz_minus_exact_mH_orthonormalized'],
            color=SLOT[1], linewidth=1.8, marker='s', markersize=7.5,
            label='orthonormalized first', zorder=2)
    ax.plot(steps, data['lowest_ritz_minus_exact_mH_single_pass'],
            color=SLOT[0], linewidth=1.8, marker='o', markersize=4.5,
            label='single pass', zorder=3)
    ax.set_yscale('symlog', linthresh=0.01)
    ax.set_xlabel('Lanczos steps')
    ax.set_ylabel('lowest Ritz value minus exact (mH)')
    ax.set_title('(b) lowest Ritz value', loc='left', color=INK)
    ax.text(12.2, 0.02, 'exact ground state', color=INK2, fontsize=7.5)
    ax.legend(frameon=False, loc='lower left')
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(OUT, f'fig_lanczos_orthogonality.{ext}'))
    plt.close(fig)


def figure_dynamics():
    path = os.path.join(ROOT, 'results', 'h3_redo', 'n2_h3_redo.json')
    if not os.path.exists(path):
        print('skip dynamics figure: results/h3_redo/n2_h3_redo.json not present')
        return
    order = [('sc_no_enrichment', 'no enrichment'),
             ('sc_constant_0.5', r'constant $\lambda = 0.5$'),
             ('sc_annealed_0.5_d0.5', r'annealed, $d = 0.5$'),
             ('sc_annealed_0.5_d0.3', r'annealed, $d = 0.3$')]
    by_label = {r['label']: r for r in load('results/h3_redo/n2_h3_redo.json')
                if r['label'].startswith('sc_')}
    rows = [by_label[label] for label, _ in order]
    names = dict(order)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
    fig.subplots_adjust(wspace=0.32)
    for k, r in enumerate(rows):
        it = list(range(1, len(r['D_trajectory']) + 1))
        axes[0].plot(it, r['D_trajectory'], color=SLOT[k], linewidth=1.8,
                     marker=MARK[k], markersize=5, label=names[r['label']])
        proj = [(i, p) for i, p in zip(it, r['projector_distance'])
                if p is not None and p > 0]
        if proj:
            axes[1].plot([i for i, _ in proj], [p for _, p in proj],
                         color=SLOT[k], linewidth=1.8, marker=MARK[k],
                         markersize=5, label=names[r['label']])
    style(axes[0])
    axes[0].set_xlabel('outer iteration')
    axes[0].set_ylabel('embedded dimension $D$')
    axes[0].set_title('(a) retained dimension', loc='left', color=INK)
    axes[0].legend(frameon=False, loc='center right', fontsize=7.5,
                   bbox_to_anchor=(1.0, 0.42))
    style(axes[1])
    axes[1].set_yscale('log')
    axes[1].set_xlabel('outer iteration')
    axes[1].set_ylabel('Schmidt projector distance')
    axes[1].set_title('(b) basis change per iteration', loc='left', color=INK)
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(OUT, f'fig_outer_map_dynamics.{ext}'))
    plt.close(fig)


if __name__ == '__main__':
    figure_threshold_curves()
    figure_lanczos()
    figure_dynamics()
    print('figures written to', OUT)
