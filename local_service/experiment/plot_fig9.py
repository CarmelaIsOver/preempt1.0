#!/usr/bin/env python3
"""
图9: 数据效用气泡图 — BLEU-4 × Semantic Similarity, bubble size = Constraint Satisfaction
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

RED   = '#d7191c'
BLUE  = '#2c7bb6'
GREEN = '#1a9641'
GRAY  = '#555555'

methods = [
    ('Regex-Only',   45.2, 0.623,   0, RED, 'Placeholder Replacement'),
    ('Presidio',     58.7, 0.718,   0, '#fdae61', 'Generic De-id'),
    ('Anon (ε=1.0)', 82.7, 0.910, 99.6, BLUE, 'Anon Full Pipeline'),
    ('Original Text',100.0, 1.000, 100, GREEN, 'Raw Data (Reference)'),
]

fig, ax = plt.subplots(figsize=(7.5, 5), facecolor='white')

for name, bleu, sim, constr, color, label in methods:
    size = max(constr, 20) * 7
    ax.scatter(bleu, sim, s=size, c=color, alpha=0.55, edgecolors=color, lw=2, zorder=5)
    offset_y = -4 if 'Anon' in name else 3
    ax.annotate(name, (bleu, sim), textcoords='offset points',
                xytext=(0, offset_y), ha='center', fontsize=9, fontweight='bold', color=color)

ax.set_xlabel('BLEU-4 (Text Naturalness, %)', fontsize=11, color=GRAY)
ax.set_ylabel('Semantic Similarity', fontsize=11, color=GRAY)
ax.set_title('Data Utility Assessment (Comprehensive)', fontsize=12, fontweight='bold', pad=10)
ax.grid(True, alpha=0.12, lw=0.4)
ax.set_xlim(25, 115)
ax.set_ylim(0.50, 1.10)
ax.tick_params(labelsize=9)

# Bubble size legend
for size_val, label in [(800, 'Constraint ≥99.6%'), (150, 'Constraint <50%'), (1, 'No Constraint')]:
    ax.scatter([], [], s=size_val, c='gray', alpha=0.3, label=label)
ax.legend(fontsize=8, frameon=True, loc='lower right', edgecolor='#ddd', title='Bubble = Constraint')

# Arrow: Anon advantage
ax.annotate('Anon: BLEU 82.7%\nSim 0.910\nConstraint 99.6%',
            xy=(82.7, 0.910), xytext=(60, 0.70),
            fontsize=8, color=BLUE, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color=BLUE, lw=1.0),
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec=BLUE, alpha=0.8))

fig.tight_layout(pad=1.5)
path = os.path.join(FIGS, 'fig9_utility_bubble')
fig.savefig(path + '.svg', bbox_inches='tight', pad_inches=0.1, facecolor='white')
fig.savefig(path + '.png', bbox_inches='tight', pad_inches=0.1, facecolor='white', dpi=150)
plt.close(fig)
print(f'✅ {path}.svg')
