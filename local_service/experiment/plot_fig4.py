#!/usr/bin/env python3
"""
图4: RQ4 ε-LDP对MIA的理论防御保证 — 双面板
左: ε vs AUC上界 (散点连线)
右: 攻击者优势 (柱状图)
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

epsilon    = np.array([0.1, 0.5, 1.0, 2.0, 5.0])
advantage  = np.array([0.025, 0.123, 0.231, 0.381, 0.493])
auc_upper  = np.array([0.525, 0.623, 0.731, 0.881, 0.993])
defense    = ['Very Strong', 'Strong', 'Moderate ★', 'Weak', 'Weak']

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2), facecolor='white')

# ---- 左: AUC 理论上界 ----
ax1.plot(epsilon, auc_upper, 'o-', color=BLUE, lw=2.5, ms=10,
         mfc='white', mew=2.2, zorder=5)
ax1.fill_between(epsilon, 0.5, auc_upper, color=BLUE, alpha=0.08)
ax1.axhline(y=0.5, color=GRAY, lw=1, ls='--', alpha=0.6, label='Random Guess (AUC=0.5)')
ax1.axhline(y=0.731, color=RED, lw=1, ls=':', alpha=0.7,
            label='Recommended ε=1.0 (AUC≤0.731)')

ax1.set_xlabel('ε (Privacy Budget)', fontsize=11, color=GRAY)
ax1.set_ylabel('MIA AUC Upper Bound', fontsize=11, color=GRAY)
ax1.set_title('Theoretical AUC Upper Bound vs ε', fontsize=12, fontweight='bold', pad=8)
ax1.legend(fontsize=8, frameon=True, loc='lower right', edgecolor='#ddd')
ax1.grid(True, alpha=0.12, lw=0.4)
ax1.set_ylim(0.45, 1.05)
ax1.tick_params(labelsize=9)

# 标注 ε=1.0
idx = 2
ax1.annotate(f'ε=1.0\nAUC≤{auc_upper[idx]:.3f}',
             xy=(epsilon[idx], auc_upper[idx]), xytext=(epsilon[idx], 0.69),
             fontsize=8.5, color=RED, ha='center',
             arrowprops=dict(arrowstyle='->', color=RED, lw=0.8),
             bbox=dict(boxstyle='round,pad=0.2', fc='white', ec=RED, alpha=0.8))

# ---- 右: 攻击者优势柱状图 ----
bar_colors = ['#1a5f8a', BLUE, GREEN, '#fdae61', RED]
bars = ax2.bar(range(len(epsilon)), advantage, width=0.55, color=bar_colors,
               edgecolor='white', lw=1.0, zorder=3)

for i, (bar, val, e) in enumerate(zip(bars, advantage, epsilon)):
    ax2.text(bar.get_x() + bar.get_width()/2, val + 0.012,
             f'{val:.3f}', ha='center', fontsize=9, fontweight='bold', color=GRAY)

ax2.set_xticks(range(len(epsilon)))
ax2.set_xticklabels([f'ε={e}' for e in epsilon], fontsize=10)
ax2.set_ylabel('Attacker Advantage Upper Bound', fontsize=11, color=GRAY)
ax2.set_title('Attacker Advantage vs ε', fontsize=12, fontweight='bold', pad=8)
ax2.grid(axis='y', alpha=0.12, lw=0.4)
ax2.set_ylim(0, 0.6)
ax2.tick_params(labelsize=9)

# ε=0.1 标注 (极强防御)
ax2.annotate(f'ε=0.1\nNearly\nrandom',
             xy=(0, advantage[0]), xytext=(0.4, 0.12),
             fontsize=8, color='#1a5f8a', ha='center',
             arrowprops=dict(arrowstyle='->', color='#1a5f8a', lw=0.7),
             bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='#1a5f8a', alpha=0.7))

fig.suptitle('RQ4: ε-Differential Privacy — Theoretical MIA Defense Guarantee',
             fontsize=13, fontweight='bold', y=1.02)

plt.tight_layout(pad=2.0)
path = os.path.join(FIGS, 'fig4_rq4_mia_theory')
fig.savefig(path + '.svg', bbox_inches='tight', pad_inches=0.1, facecolor='white')
fig.savefig(path + '.png', bbox_inches='tight', pad_inches=0.1, facecolor='white', dpi=150)
plt.close(fig)
print(f'✅ {path}.svg')
