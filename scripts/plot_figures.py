# -*- coding: utf-8 -*-
"""Paper figure set v5 - journal style: continuous-cell heatmaps, frameless
annotations, paired panels, Okabe-Ito palette.

Run from anywhere:  python scripts/plot_figures.py [fig1 ... fig7]
Reads JSON data from <repo>/data/ and writes PNG+PDF into <repo>/manuscript/.
"""
import json, os, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.path as mpath
from matplotlib.patches import PathPatch, Rectangle, FancyBboxPatch, Circle
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import numpy as np
import scienceplots

plt.style.use('nature')

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), 'data')
OUT = os.path.join(os.path.dirname(HERE), 'manuscript')
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'mathtext.fontset': 'stixsans',
    'font.size': 8,
    'axes.linewidth': 0.7,
    'axes.edgecolor': '#333333',
    'axes.labelsize': 8.5,
    'xtick.labelsize': 7.5,
    'ytick.labelsize': 7.5,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'xtick.major.width': 0.7,
    'ytick.major.width': 0.7,
    'xtick.major.size': 3.0,
    'ytick.major.size': 3.0,
    'legend.fontsize': 7,
    'figure.dpi': 300,
    'savefig.dpi': 600,
    'savefig.bbox': 'tight',
    'savefig.facecolor': 'white',
    'pdf.fonttype': 42,
})

# Okabe-Ito colorblind-safe palette
BLUE   = '#0072B2'
ORANGE = '#E69F00'
GREEN  = '#009E73'
VERM   = '#D55E00'
GRAY   = '#7F7F7F'
DARK   = '#333333'
FLOW_OK = '#d8d4cc'   # warm light gray: correct flow

REP_COL = {'plain': BLUE, 'env': ORANGE, 'order': GREEN}
REP_MK  = {'plain': 'o', 'env': '^', 'order': 'D'}

def style_ax(ax, grid_axis=None):
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    if grid_axis:
        ax.grid(axis=grid_axis, color='0.92', lw=0.45, zorder=0)
        ax.set_axisbelow(True)

def panel_label(ax, text, dx=-0.12, dy=1.05):
    ax.text(dx, dy, text, transform=ax.transAxes, fontsize=9,
            fontweight='bold', va='top', ha='left')

def frame_matrix(ax, w, h):
    """Thin dark frame around the whole heatmap, replacing white gridlines."""
    ax.add_patch(Rectangle((0, 0), w, h, fill=False, ec=DARK, lw=0.9,
                           zorder=8, clip_on=False))

def save(fig, name):
    fig.savefig(os.path.join(OUT, name + '.png'))
    fig.savefig(os.path.join(OUT, name + '.pdf'))
    plt.close(fig)

def load_unified():
    with open(os.path.join(DATA, 'unified_p3_final.json'), encoding='utf-8') as f:
        return json.load(f)

DOMAINS = ['0.014@0', '0.014@1', '0.014@2', '0.014@3',
           '0.021@0', '0.021@1', '0.021@2', '0.021@3']
DOMAIN_LABELS = [d + 'HP' for d in DOMAINS]
REPS = ('plain', 'env', 'order')

# ================================================================= Fig.1
def fig_framework():
    fig, ax = plt.subplots(figsize=(6.3, 3.1))
    ax.set_xlim(0, 10); ax.set_ylim(0, 5.9); ax.axis('off')

    BOX_FC = '#f5f5f5'; BOX_EC = '#555555'
    def box(x, y, w, h, text, fc=BOX_FC, ec=BOX_EC, fs=7.0, bold=False, tc=DARK):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                     boxstyle='round,pad=0.02,rounding_size=0.08',
                     fc=fc, ec=ec, lw=0.7))
        ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=fs,
                fontweight='bold' if bold else 'normal', linespacing=1.35, color=tc)

    def arrow(x1, y1, x2, y2):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='-|>', lw=0.7, color='#555555',
                                    shrinkA=0, shrinkB=0))

    def stage(n, y):
        ax.add_patch(Circle((0.16, y), 0.13, fc=DARK, ec='none', zorder=5))
        ax.text(0.16, y, str(n), fontsize=6.4, color='white',
                ha='center', va='center', zorder=6)

    box(2.7, 5.05, 4.6, 0.66, 'Raw vibration signal\n2048 pts, 12 kHz, CWRU drive end',
        bold=True)
    stage(1, 5.38)

    bw, bh = 2.85, 0.92
    xs = [0.75, 3.575, 6.4]
    for x, c, t in [(xs[0], BLUE,   'plain\nCWT (Morlet, scales 1-99)'),
                    (xs[1], ORANGE, 'env\nHilbert envelope + CWT'),
                    (xs[2], GREEN,  'order\nspeed resampling + env CWT')]:
        box(x, 3.55, bw, bh, t, fc='white', ec=c, tc=c)
    stage(2, 4.01)
    centres = [x + bw/2 for x in xs]
    ax.plot([centres[0], centres[-1]], [4.81, 4.81], color='#555555', lw=0.7)
    arrow(5.0, 5.05, 5.0, 4.81)
    for xc in centres:
        arrow(xc, 4.81, xc, 4.47)

    box(0.75, 2.25, 8.5, 0.62,
        'Time-frequency image: 224 x 224, log1p + min-max to 0-255, 3-channel')
    stage(3, 2.56)
    for x in xs:
        arrow(x + bw/2, 3.55, x + bw/2, 2.87)

    box(0.75, 1.15, 8.5, 0.62,
        'EfficientNet-B0 (ImageNet pretrained) + 4-class head\nNormal / Inner race / Ball / Outer race', bold=True)
    stage(4, 1.46)
    arrow(5.0, 2.25, 5.0, 1.77)

    for x, t in [(0.75, 'P1  cross-load\n0.007@0HP to {1,2,3}HP'),
                 (3.575, 'P2  cross-SNR\n-5 / 0 / 10 dB'),
                 (6.4, 'P3  cross-severity\n0.007 to 0.014 / 0.021')]:
        box(x, 0.10, 2.85, 0.58, t, fc='white', fs=6.7)
    stage(5, 0.39)
    arrow(5.0, 1.15, 5.0, 0.68)
    save(fig, 'fig1_framework')

# ================================================================= Fig.2
def fig_p2_snr():
    snr = ['Clean', '10 dB', '0 dB', '-5 dB']
    x = np.arange(4)
    clean = [('plain', [100.0, 99.0, 65.0, 26.5]),
             ('env',   [100.0, 100.0, 89.0, 30.5])]
    noise = [('plain', [100.0, 100.0, 100.0, 100.0]),
             ('env',   [98.0, 100.0, 100.0, 97.5])]

    fig, axes = plt.subplots(1, 2, figsize=(6.2, 2.5), sharey=True)
    fig.subplots_adjust(left=0.075, right=0.985, top=0.87, bottom=0.30,
                        wspace=0.10)

    for ax, data, ttl in [(axes[0], clean, 'clean training'),
                          (axes[1], noise, 'noise training')]:
        ax.axhline(25, color=GRAY, lw=0.7, ls=':', zorder=1)
        for r, vals in data:
            ax.plot(x, vals, '-', color=REP_COL[r], marker=REP_MK[r],
                    ms=4.0, lw=1.2, mfc=REP_COL[r], mew=0.9, zorder=4)
        ax.set_xticks(x); ax.set_xticklabels(snr)
        ax.set_xlim(-0.25, 3.25); ax.set_ylim(15, 106)
        ax.set_title(ttl, fontsize=7.6, pad=4)
        ax.set_xlabel('Test SNR')
        style_ax(ax, 'y')

    axes[0].set_yticks([25, 50, 75, 100])
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].text(2.95, 27.5, 'chance (25%)', fontsize=6.0, color=GRAY,
                 ha='right', va='bottom')
    axes[0].annotate('', xy=(2, 88.4), xytext=(2, 65.6),
                     arrowprops=dict(arrowstyle='<|-|>', lw=0.8, color=DARK))
    axes[0].text(2.10, 77, '+24 pts', fontsize=6.4, color=DARK, va='center')

    panel_label(axes[0], '(a)', dx=-0.20, dy=1.10)
    panel_label(axes[1], '(b)', dx=-0.10, dy=1.10)

    handles = [Line2D([0], [0], color=REP_COL[r], marker=REP_MK[r],
                      ms=4.0, lw=1.2, label=r) for r in ('plain', 'env')]
    fig.legend(handles=handles, loc='lower center', ncol=2, frameon=False,
               fontsize=6.6, bbox_to_anchor=(0.5, -0.01),
               handlelength=1.6, handletextpad=0.5, columnspacing=1.4)
    save(fig, 'fig2_p2_snr')

# ================================================================= Fig.3
def fig_p3_domains():
    uni = load_unified()
    mat = np.array([[uni['%s|42' % r][d]['acc'] for r in REPS] for d in DOMAINS])

    plain_seeds = None
    ms_path = os.path.join(DATA, 'multiseed_recall.json')
    if os.path.exists(ms_path):
        with open(ms_path, encoding='utf-8') as f:
            ms = json.load(f)
        plain_seeds = {d: np.array(ms[d]['acc'], dtype=float)
                       for d in DOMAINS if d in ms}

    fig = plt.figure(figsize=(6.4, 5.3))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.0], hspace=0.42,
                          left=0.11, right=0.90, top=0.94, bottom=0.19)

    # ---- (a) viridis heatmap: continuous color cells ----
    ax = fig.add_subplot(gs[0])
    ax.pcolormesh(np.arange(4), np.arange(9), mat, cmap='viridis',
                  vmin=40, vmax=80)
    for i in range(8):
        for j in range(3):
            v = mat[i, j]
            ax.text(j + 0.5, i + 0.5, '%.1f' % v, ha='center', va='center',
                    fontsize=7, color='white' if v <= 56 else DARK)
    frame_matrix(ax, 3, 8)
    ax.axhline(4, color=DARK, lw=0.9)
    ax.text(3.14, 2, '0.014 in', fontsize=6.8, va='center', rotation=90, color=DARK)
    ax.text(3.14, 6, '0.021 in', fontsize=6.8, va='center', rotation=90, color=DARK)
    ax.set_xticks(np.arange(3) + 0.5); ax.set_xticklabels(list(REPS), fontsize=7.5)
    ax.xaxis.set_ticks_position('top'); ax.tick_params(length=0)
    ax.set_yticks(np.arange(8) + 0.5); ax.set_yticklabels(DOMAIN_LABELS, fontsize=7)
    ax.set_ylim(8, 0); ax.set_xlim(0, 3)
    for s in ax.spines.values(): s.set_visible(False)

    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(40, 80))
    cb = fig.colorbar(sm, ax=ax, pad=0.10, fraction=0.030, aspect=16,
                      ticks=[40, 50, 60, 70, 80])
    cb.set_label('Accuracy (%)', fontsize=7); cb.ax.tick_params(labelsize=6.5)
    cb.outline.set_linewidth(0.5)
    panel_label(ax, '(a)', dx=-0.17, dy=1.10)

    # ---- (b) dumbbell ----
    ax2 = fig.add_subplot(gs[1])
    ys = np.arange(8)[::-1]
    ax2.axvline(62.52, color=GRAY, lw=0.7, ls='--', zorder=1)
    ax2.text(62.9, 7.55, 'plain 3-seed mean', fontsize=6.0, color=GRAY)

    for i, d in enumerate(DOMAINS):
        y = ys[i]
        vals = mat[i]
        ax2.plot([vals.min(), vals.max()], [y, y], color='0.8', lw=0.9, zorder=2)
        for j, r in enumerate(REPS):
            ax2.plot(vals[j], y, REP_MK[r], color=REP_COL[r], ms=4.2, mew=0.9,
                     mfc=REP_COL[r] if r == 'plain' else 'white', zorder=4)
        if plain_seeds and d in plain_seeds:
            sv = plain_seeds[d]
            ax2.plot(sv, np.full_like(sv, y) + 0.24, '|', color=BLUE,
                     ms=4.5, mew=0.8, zorder=5)
            ax2.errorbar(sv.mean(), y - 0.24, xerr=sv.std(), fmt='none',
                         ecolor=BLUE, elinewidth=0.8, capsize=2, zorder=5)

    ax2.set_yticks(ys); ax2.set_yticklabels(DOMAIN_LABELS, fontsize=7)
    ax2.set_xlim(38, 82); ax2.set_ylim(-0.7, 8.0)
    ax2.set_xlabel('Accuracy (%)')
    style_ax(ax2, 'x')
    handles = [Line2D([0], [0], marker=REP_MK[r], color=REP_COL[r], lw=0,
                      ms=4.4, mew=0.9,
                      mfc=REP_COL[r] if r == 'plain' else 'white', label=r)
               for r in REPS]
    handles.append(Line2D([0], [0], marker='|', color=BLUE, lw=0, ms=5,
                          label='plain seeds (n=3)'))
    ax2.legend(handles=handles, loc='upper center', frameon=False, fontsize=6.4,
               ncol=4, bbox_to_anchor=(0.5, -0.26), handletextpad=0.3,
               columnspacing=1.0)
    panel_label(ax2, '(b)', dx=-0.17, dy=1.05)
    save(fig, 'fig3_p3_domains')

# ================================================================= Fig.4 confusion matrix
def fig_confusion():
    uni = load_unified()
    order = [2, 0, 1, 3]

    def get_conf(dom, fallback, acc_fb):
        key = 'plain|42'
        if key in uni and 'conf' in uni[key][dom]:
            c = np.array(uni[key][dom]['conf'])
            return c[np.ix_(order, order)], uni[key][dom]['acc']
        return np.array(fallback), acc_fb

    conf1, acc1 = get_conf('0.014@0',
        [[0,50,0,0],[0,0,0,50],[50,0,0,0],[0,0,0,50]], 50.00)
    conf2, acc2 = get_conf('0.021@0',
        [[0,0,0,50],[0,50,0,0],[50,0,0,0],[0,0,0,50]], 75.00)

    labels = ['N', 'IR', 'OR', 'B']

    fig, axes = plt.subplots(1, 2, figsize=(5.9, 2.95))
    fig.subplots_adjust(left=0.085, right=0.985, top=0.86, bottom=0.17,
                        wspace=0.38)

    for ax, conf, tag, acc in [(axes[0], conf1, '(a) 0.014@0HP', acc1),
                               (axes[1], conf2, '(b) 0.021@0HP', acc2)]:
        ax.imshow(conf, cmap='Blues', vmin=0, vmax=50, aspect='equal')
        for k in range(5):
            ax.axhline(k - 0.5, color='white', lw=1.6)
            ax.axvline(k - 0.5, color='white', lw=1.6)
        for i in range(4):
            for j in range(4):
                v = conf[i, j]
                off = i != j
                ax.text(j, i, '%d' % v, ha='center', va='center', fontsize=8.5,
                        fontweight='bold' if (off and v > 0) else 'normal',
                        color='white' if v >= 25 else DARK)
        ax.set_xticks(range(4)); ax.set_xticklabels(labels, fontsize=8)
        ax.set_yticks(range(4)); ax.set_yticklabels(labels, fontsize=8)
        ax.set_title('%s, accuracy %.2f%%' % (tag, acc), fontsize=8.2, pad=5)
        ax.set_xlabel('Predicted class')
        ax.tick_params(length=0)
        for s in ax.spines.values(): s.set_visible(False)
    axes[0].set_ylabel('True class')
    save(fig, 'fig4_confusion')

# ================================================================= Fig.5
def fig_fisher():
    bands = ['0-0.5', '0.5-1', '1-1.5', '1.5-2', '2-2.5', '2.5-3', '3-3.5', '3.5-4']
    centers = np.array([0.25, 0.75, 1.25, 1.75, 2.25, 2.75, 3.25, 3.75])
    with open(os.path.join(DATA, 'fisher_12domains.json'), encoding='utf-8') as f:
        data = json.load(f)
    doms = [(s, l) for s in ('0.007', '0.014', '0.021') for l in range(4)]
    labels = ['%s@%dHP' % (s, l) for s, l in doms]
    mat = np.array([data['%s@%d' % (s, l)] for s, l in doms])
    peaks = mat.argmax(axis=1)

    fig = plt.figure(figsize=(6.4, 5.1))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.35, 1.0], hspace=0.32,
                          left=0.11, right=0.90, top=0.94, bottom=0.08)

    # ---- (a) heatmap: continuous color cells ----
    ax = fig.add_subplot(gs[0])
    ax.pcolormesh(np.arange(9), np.arange(13), mat, cmap='viridis',
                  vmin=0.5, vmax=10)
    for i in range(12):
        for j in range(8):
            v = mat[i, j]
            ax.text(j + 0.5, i + 0.5, '%.1f' % v, ha='center', va='center',
                    fontsize=6.4, color='white' if v <= 5.6 else DARK)
        # peak marker: dark dot with white edge (avoids merged-box look)
        ax.plot(peaks[i] + 0.80, i + 0.20, 'o', ms=2.8, mfc=DARK,
                mec='white', mew=0.6, zorder=6)
    ax.add_patch(Rectangle((2, 0), 1, 12, fill=False, ec='white', lw=0.9,
                           ls=(0, (4, 2)), zorder=6))
    frame_matrix(ax, 8, 12)
    for y in (4, 8):
        ax.axhline(y, color=DARK, lw=0.9)
    for yc, lab in [(2, '0.007 in'), (6, '0.014 in'), (10, '0.021 in')]:
        ax.text(8.14, yc, lab, fontsize=6.8, va='center', rotation=90, color=DARK)
    ax.set_xticks(np.arange(8) + 0.5); ax.set_xticklabels(bands, fontsize=6.8)
    ax.set_yticks(np.arange(12) + 0.5); ax.set_yticklabels(labels, fontsize=6.8)
    ax.set_ylim(12, 0); ax.set_xlim(0, 8)
    ax.tick_params(length=0)
    for s in ax.spines.values(): s.set_visible(False)
    ax.set_xlabel('Frequency band (kHz)')
    ax.xaxis.set_label_position('top')
    ax.xaxis.set_ticks_position('top')

    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(0.5, 10))
    cb = fig.colorbar(sm, ax=ax, pad=0.10, fraction=0.028, aspect=18,
                      ticks=[1, 3, 5, 7, 9])
    cb.set_label('Fisher ratio $F$ (IR vs OR)', fontsize=7)
    cb.ax.tick_params(labelsize=6.5); cb.outline.set_linewidth(0.5)
    panel_label(ax, '(a)', dx=-0.17, dy=1.04)

    # ---- (b) peak-band migration ----
    ax2 = fig.add_subplot(gs[1])
    loads = np.arange(4)
    size_style = [('0.007 in', BLUE, 'o'), ('0.014 in', ORANGE, '^'),
                  ('0.021 in', GREEN, 'D')]
    ax2.axhline(1.25, color=GRAY, lw=0.7, ls='--', zorder=0)
    ax2.text(0.06, 1.40, 'source band 1-1.5 kHz', fontsize=6.0, color=GRAY)
    for k, (name, col, mk) in enumerate(size_style):
        yv = centers[peaks[k*4:(k+1)*4]]
        ax2.plot(loads, yv, '-', color=col, lw=1.1, zorder=3)
        ax2.plot(loads, yv, mk, color=col, ms=4.4, mfc='white', mew=1.0,
                 zorder=4, label=name)
    ax2.set_xticks(loads); ax2.set_xticklabels(['0 HP', '1 HP', '2 HP', '3 HP'])
    ax2.set_ylim(0, 4.25); ax2.set_xlim(-0.3, 3.5)
    ax2.set_yticks(centers); ax2.set_yticklabels(bands, fontsize=6.4)
    ax2.set_ylabel('Peak band centre (kHz)'); ax2.set_xlabel('Load')
    style_ax(ax2, 'y')
    ax2.legend(loc='upper right', frameon=False, fontsize=6.4,
               handletextpad=0.4)
    panel_label(ax2, '(b)', dx=-0.17, dy=1.05)
    save(fig, 'fig5_fisher')

# ================================================================= Fig.6
def fig_remedies():
    remedies = [
        ('Fixed 4-4.8 kHz band',      4.63),
        ('Band-masking training',     1.87),
        ('Multi-source augmentation', 0.25),
        ('Entropy TTA',              -3.13),
        ('Envelope demodulation',    -5.50),
        ('Order normalisation',      -5.56),
        ('Entropy band selection',   -9.56),
        ('BN adaptation',           -15.87),
    ]
    fig, ax = plt.subplots(figsize=(5.4, 3.1))
    fig.subplots_adjust(left=0.30, right=0.96, top=0.97, bottom=0.15)

    ys = np.arange(len(remedies))[::-1]
    for (name, d), y in zip(remedies, ys):
        col = GREEN if d >= 0 else VERM
        ax.barh(y, d, height=0.62, color=col, zorder=3)
        ax.text(d + (0.35 if d >= 0 else -0.35), y, '%+.2f' % d,
                va='center', ha='left' if d >= 0 else 'right',
                fontsize=6.8, color=DARK)

    ax.axvline(0, color=DARK, lw=0.9, zorder=4)
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in remedies], fontsize=7)
    ax.set_xlim(-18.5, 8.5)
    ax.set_xlabel('Change vs plain baseline 62.25% (accuracy points)')
    style_ax(ax, 'x')
    save(fig, 'fig6_remedies')

# ================================================================= Fig.7
def fig_datasets():
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.8),
                             gridspec_kw={'width_ratios': [1.0, 1.25]})
    fig.subplots_adjust(left=0.08, right=0.98, top=0.88, bottom=0.16, wspace=0.42)

    # ---- (a) slope ----
    ax = axes[0]
    rows = [('CWRU cross-severity', 100.00, 62.25, BLUE,    3.2, -4.5),
            ('JNU cross-speed',      99.89, 73.83, ORANGE, -4.4,  2.4),
            ('Paderborn cross-size', 48.22, 53.56, GREEN,   0.0, -7.5)]
    ax.axhline(25, color=GRAY, lw=0.6, ls=':', zorder=0)
    ax.text(0.98, 23.0, 'chance 25%', fontsize=5.8, color=GRAY,
            ha='right', va='top')
    for name, a, b, col, dya, dyb in rows:
        ax.annotate('', xy=(1, b), xytext=(0, a),
                    arrowprops=dict(arrowstyle='-|>', color=col, lw=1.3,
                                    shrinkA=3, shrinkB=3), zorder=3)
        ax.plot(0, a, 'o', color=col, ms=4.4, zorder=4)
        ax.plot(1, b, 'o', mfc='white', mec=col, mew=1.1, ms=4.4, zorder=4)
        ax.text(-0.06, a + dya, '%.2f' % a, ha='right', va='center',
                fontsize=6.2, color=col)
        ax.text(1.07, b + dyb, '%.2f\n%s' % (b, name), ha='left', va='center',
                fontsize=6.2, color=col, linespacing=1.25)
    ax.set_xticks([0, 1]); ax.set_xticklabels(['In-domain', 'Transfer'])
    ax.set_xlim(-0.55, 1.62); ax.set_ylim(15, 110)
    ax.set_yticks([25, 50, 75, 100])
    ax.set_ylabel('Accuracy (%)')
    style_ax(ax, 'y')
    panel_label(ax, '(a)', dx=-0.42, dy=1.09)

    # ---- (b) recall heatmap: continuous color cells ----
    ax2 = axes[1]
    tasks = ['600→800', '600→1000', '800→600',
             '800→1000', '1000→600', '1000→800']
    rec = np.array([
        [17.33, 100.00, 43.33, 96.00],
        [3.33,  100.00, 48.67, 96.00],
        [54.67, 100.00, 54.00, 98.67],
        [50.67, 100.00, 100.00, 100.00],
        [0.00,   99.33,  20.00, 100.00],
        [91.33,  98.67, 100.00, 100.00],
    ])
    ax2.pcolormesh(np.arange(5), np.arange(7), rec, cmap='viridis',
                   vmin=0, vmax=100)
    for i in range(6):
        for j in range(4):
            v = rec[i, j]
            ax2.text(j + 0.5, i + 0.5, '%.1f' % v, ha='center', va='center',
                     fontsize=6.4, color='white' if v <= 55 else DARK)
    frame_matrix(ax2, 4, 6)
    ax2.set_xticks(np.arange(4) + 0.5)
    ax2.set_xticklabels(['Normal', 'Inner race', 'Outer race', 'Ball'], fontsize=6.4)
    ax2.xaxis.set_ticks_position('top')
    ax2.set_yticks(np.arange(6) + 0.5); ax2.set_yticklabels(tasks, fontsize=6.8)
    ax2.set_ylim(6, 0); ax2.set_xlim(0, 4)
    ax2.tick_params(length=0)
    for s in ax2.spines.values(): s.set_visible(False)
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(0, 100))
    cb = fig.colorbar(sm, ax=ax2, pad=0.02, fraction=0.035, aspect=14,
                      ticks=[0, 25, 50, 75, 100])
    cb.set_label('Recall (%)', fontsize=6.8); cb.ax.tick_params(labelsize=6.2)
    cb.outline.set_linewidth(0.5)
    ax2.set_title('JNU cross-speed recall by class', fontsize=7.5, pad=3)
    panel_label(ax2, '(b)', dx=-0.16, dy=1.09)
    save(fig, 'fig7_datasets')

FIGURES = {'fig1': fig_framework, 'fig2': fig_p2_snr, 'fig3': fig_p3_domains,
           'fig4': fig_confusion, 'fig5': fig_fisher, 'fig6': fig_remedies,
           'fig7': fig_datasets}

if __name__ == '__main__':
    keys = sys.argv[1:] or list(FIGURES)
    for k in keys:
        FIGURES[k]()
        print('written', k, flush=True)
    print('figures written to', OUT)
