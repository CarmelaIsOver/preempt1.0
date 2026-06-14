#!/usr/bin/env python3
"""
图10: 跨样本假名一致性 — 3方法 × 4实体类型
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

plt.rcParams.update({
    'font.family': 'Noto Sans CJK JP', 'axes.unicode_minus': False,
    'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'axes.linewidth': 0.8, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
})
OUT = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(os.path.dirname(OUT), 'figures')
os.makedirs(FIGS, exist_ok=True)

BLUE  = '#2c7bb6'
ORANGE = '#fdae61'
RED   = '#d7191c'
GRAY  = '#555555'

entities = ['PERSON', 'PHONE', 'EMAIL', 'ID']
anon_vals   = [100, 100, 100, 100]
regex_vals  = [100, 100, 100, 100]
random_vals = [0, 0, 0, 0]

x = np.arange(len(entities))
width = 0.25

fig, ax = plt.subplots(figsize=(7, 4.2), facecolor='white')

b1 = ax.bar(x - width, anon_vals, width, color=BLUE, edgecolor='white', lw=0.8,
            label='Anon (Deterministic Hash)')
b2 = ax.bar(x, regex_vals, width, color=ORANGE, edgecolor='white', lw=0.8,
            label='Regex-Only (Placeholder)')
b3 = ax.bar(x + width, random_vals, width, color=RED, edgecolor='white', lw=0.8,
            label='Random Replacement')

# Anon advantage note
ax.annotate('Anon: 100% consistent\nyet preserves entity\ndiscriminability',
            xy=(0, 100), xytext=(1.8, 72),
            fontsize=8, color=BLUE, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color=BLUE, lw=0.8),
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec=BLUE, alpha=0.8))

ax.annotate('Regex: 100% consistent\nbut loses identity\n(all names→[PERSON])',
            xy=(0.25, 100), xytext=(2.8, 42),
            fontsize=8, color=ORANGE,
            arrowprops=dict(arrowstyle='->', color=ORANGE, lw=0.8),
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec=ORANGE, alpha=0.8))

ax.set_xticks(x)
ax.set_xticklabels(entities, fontsize=11)
ax.set_ylabel('Consistent Mapping Rate (%)', fontsize=11, color=GRAY)
ax.set_title('Cross-Sample Pseudonym Consistency', fontsize=12, fontweight='bold', pad=10)
ax.legend(fontsize=8.5, frameon=True, loc='lower left', edgecolor='#ddd')
ax.set_ylim(0, 125)
ax.grid(axis='y', alpha=0.12, lw=0.4)
ax.tick_params(labelsize=10)

fig.tight_layout(pad=1.5)
path = os.path.join(FIGS, 'fig10_cross_sample_consistency')
fig.savefig(path + '.svg', bbox_inches='tight', pad_inches=0.1, facecolor='white')
fig.savefig(path + '.png', bbox_inches='tight', pad_inches=0.1, facecolor='white', dpi=150)
plt.close(fig)
print(f'✅ {path}.svg')
