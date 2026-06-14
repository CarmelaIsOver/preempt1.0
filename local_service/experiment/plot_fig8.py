#!/usr/bin/env python3
"""
图8: NER性能对比 — 3方法 × P/R/F1 分组柱状图
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

BLUE = '#2c7bb6'
CYAN = '#abd9e9'
GREEN = '#1a9641'
GRAY  = '#555555'

methods   = ['Regex-Only', 'Rule NER', 'LLM NER\n(GLM-4.7)']
precision = [81.3, 96.2, 92.8]
recall    = [52.1, 74.5, 89.9]
f1_score  = [63.5, 83.9, 91.3]

x = np.arange(len(methods))
width = 0.22

fig, ax = plt.subplots(figsize=(6.5, 4.5), facecolor='white')

ax.bar(x - width, precision, width, color=CYAN, edgecolor='white', lw=0.8, label='Precision')
ax.bar(x, recall, width, color=GREEN, edgecolor='white', lw=0.8, label='Recall')
ax.bar(x + width, f1_score, width, color=BLUE, edgecolor='white', lw=0.8, label='F1-Score')

# Values on bars
for i in range(3):
    for j, vals in enumerate([precision, recall, f1_score]):
        v = vals[i]
        ax.text(x[i] + (j-1)*width, v + 1.2, f'{v:.1f}%', ha='center', fontsize=8, fontweight='bold', color=GRAY)

ax.set_xticks(x)
ax.set_xticklabels(methods, fontsize=11)
ax.set_ylabel('Score (%)', fontsize=11, color=GRAY)
ax.set_title('NER Performance Comparison', fontsize=12, fontweight='bold', pad=10)
ax.legend(fontsize=9, frameon=True, loc='upper left', edgecolor='#ddd')
ax.set_ylim(0, 110)
ax.grid(axis='y', alpha=0.12, lw=0.4)
ax.tick_params(labelsize=10)

fig.tight_layout(pad=1.5)
path = os.path.join(FIGS, 'fig8_ner_comparison')
fig.savefig(path + '.svg', bbox_inches='tight', pad_inches=0.1, facecolor='white')
fig.savefig(path + '.png', bbox_inches='tight', pad_inches=0.1, facecolor='white', dpi=150)
plt.close(fig)
print(f'✅ {path}.svg')
