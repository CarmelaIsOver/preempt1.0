#!/usr/bin/env python3
"""
图3: RQ4 前缀注入攻击防御 — 柱状图 + 折线叠加
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

plt.rcParams.update({
    'font.family': 'Noto Sans CJK JP',
    'axes.unicode_minus': False,
    'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'axes.linewidth': 0.8,
    'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
})

OUT = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(os.path.dirname(OUT), 'figures')
os.makedirs(FIGS, exist_ok=True)

BLUE  = '#2c7bb6'
RED   = '#d7191c'
GREEN = '#1a9641'
GRAY  = '#555555'

levels     = ['None\n(Raw Data)', 'T1\n(Name+FF3)', 'T1+T2\n(+Num DP)', 'T1+T3\n(+Med Term)', 'Full\n(Anon)']
extraction = [100.0, 3.3, 0.0, 3.3, 0.0]
colors     = [RED, '#f4a460', GREEN, '#f4a460', BLUE]

x = np.arange(len(levels))

fig, ax = plt.subplots(figsize=(7, 4.5), facecolor='white')

# ---- 柱状图 ----
bars = ax.bar(x, extraction, width=0.5, color=colors, edgecolor='white', lw=1.0, zorder=2)

# ---- 折线叠加: 下降趋势 (折线点在上方标注) ----
line_x = x + 0.0
ax.plot(line_x, extraction, 'o-', color=GRAY, lw=2.5, ms=11,
        mfc='white', mew=2.5, zorder=5)

for i, val in enumerate(extraction):
    yoff = 7 if i < 2 else (8 if i == 3 else 7)
    ax.annotate(f'{val:.0f}%',
                xy=(line_x[i], extraction[i]),
                xytext=(0, yoff), textcoords='offset points',
                fontsize=9.5, color=GRAY, ha='center', fontweight='bold')

# ---- 种子数据散点 ----
seed_data = [
    [16.7, 10.0, 0.0],
    [0.0, 3.3, 6.7],
    [6.7, 3.3, 0.0],
    [0.0, 0.0, 0.0],
    [3.3, 0.0, 0.0],
]
for i, seeds in enumerate(seed_data):
    xi = np.full(len(seeds), x[i]) + np.linspace(-0.14, 0.14, len(seeds))
    ax.scatter(xi, seeds, s=18, c=GRAY, alpha=0.5, zorder=6, marker='_', linewidths=1.2)

# ---- 轴 ----
ax.set_xticks(x)
ax.set_xticklabels(levels, fontsize=10)
ax.set_ylabel('PII Extraction Rate (%)', fontsize=11, color=GRAY)
ax.set_title('RQ4: Prefix Injection Attack Defense (5-Level Ablation)', fontsize=12, fontweight='bold', pad=10)
ax.set_ylim(0, 115)
ax.grid(axis='y', alpha=0.12, lw=0.4)
ax.tick_params(labelsize=10)
ax.set_xlim(-0.6, 4.6)

# ---- 图例 ----
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=RED, label='No defense'),
    Patch(facecolor='#f4a460', label='Partial defense'),
    Patch(facecolor=BLUE, label='Full defense'),
    Line2D([0], [0], color=GRAY, lw=2.5, marker='o', mfc='white', mew=2, ms=8, label='Trend line'),
]
ax.legend(handles=legend_elements, fontsize=8, frameon=True,
          loc='upper right', edgecolor='#ddd', fancybox=False, ncol=2)

fig.tight_layout(pad=1.5)
path = os.path.join(FIGS, 'fig3_rq4_prefix_defense')
fig.savefig(path + '.svg', bbox_inches='tight', pad_inches=0.1, facecolor='white')
fig.savefig(path + '.png', bbox_inches='tight', pad_inches=0.1, facecolor='white', dpi=150)
plt.close(fig)
print(f'✅ {path}.svg')
