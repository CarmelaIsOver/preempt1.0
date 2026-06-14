#!/usr/bin/env python3
"""
图2: T3 医疗实体语义扰动隐私-效用权衡 — 双轴散点连线图
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

BLUE   = '#2c7bb6'
RED    = '#d7191c'
ORANGE = '#f4a460'
GRAY   = '#555555'

epsilon       = np.array([0.1, 0.5, 1.0, 2.0, 5.0, 10.0])
st_sim        = np.array([0.731, 0.734, 0.738, 0.745, 0.768, 0.801])
consistency   = np.array([67.4, 66.3, 65.4, 65.1, 63.3, 60.8])
m3e_sim       = np.array([0.790, 0.790, 0.791, 0.793, 0.799, 0.811])
random_consis = np.array([5.0, 5.0, 5.0, 5.0, 5.0, 5.0])

fig, ax1 = plt.subplots(figsize=(7.5, 4.8), facecolor='white')
ax2 = ax1.twinx()

# ---- 左轴: 语义相似度 ----
ax1.plot(epsilon, st_sim, 'o-', color=BLUE, lw=2.2, ms=8,
         mfc='white', mew=1.8, label='ST Similarity (Anon)', zorder=5)
ax1.plot(epsilon, m3e_sim, 's--', color='#abd5e8', lw=1.8, ms=7,
         mfc='white', mew=1.5, label='M3E Similarity', zorder=4)
ax1.set_xscale('log')
ax1.set_xlabel('ε (Privacy Budget)', fontsize=11, color=GRAY)
ax1.set_ylabel('Semantic Similarity', fontsize=11, color=BLUE)
ax1.tick_params(axis='y', labelcolor=BLUE, labelsize=9)
ax1.set_ylim(0.69, 0.85)
ax1.grid(True, alpha=0.12, lw=0.4)

# ---- 右轴: 领域一致性 ----
ax2.plot(epsilon, consistency, 'D-', color=RED, lw=2.2, ms=8,
         mfc='white', mew=1.8, label='Domain Consistency (Anon)', zorder=5)
ax2.plot(epsilon, random_consis, 'v:', color=ORANGE, lw=1.5, ms=6,
         mfc='white', mew=1.2, label='Random-T3 Baseline', zorder=3)
ax2.set_ylabel('Domain Consistency (%)', fontsize=11, color=RED)
ax2.tick_params(axis='y', labelcolor=RED, labelsize=9)
ax2.set_ylim(-2, 80)

# ---- 图例放在图内左下角 ----
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
leg = ax1.legend(lines1 + lines2, labels1 + labels2,
                 fontsize=8, frameon=True, ncol=1,
                 loc='lower left', edgecolor='#ddd', fancybox=False)

# ---- 标注 (靠近数据点) ----
idx = 2  # ε=1.0

ax1.annotate(f'ε=1.0  sim={st_sim[idx]:.3f}',
             xy=(1.0, st_sim[idx]), xytext=(1.0, 0.724),
             fontsize=8, color=BLUE, ha='center',
             arrowprops=dict(arrowstyle='->', color=BLUE, lw=0.7),
             bbox=dict(boxstyle='round,pad=0.2', fc='white', ec=BLUE, alpha=0.8))

ax2.annotate(f'{consistency[idx]:.1f}% (13× baseline)',
             xy=(1.0, consistency[idx]), xytext=(1.0, 60.5),
             fontsize=8, color=RED, ha='center',
             arrowprops=dict(arrowstyle='->', color=RED, lw=0.7),
             bbox=dict(boxstyle='round,pad=0.2', fc='white', ec=RED, alpha=0.8))

ax1.set_title('T3 Medical Semantic Perturbation: Privacy-Utility Tradeoff',
              fontsize=12, fontweight='bold', pad=10)

fig.tight_layout(pad=1.5)

path = os.path.join(FIGS, 'fig2_t3_tradeoff')
try:
    fig.savefig(path + '.pdf', bbox_inches='tight', pad_inches=0.1, facecolor='white')
except PermissionError:
    pass
fig.savefig(path + '.png', bbox_inches='tight', pad_inches=0.1, facecolor='white', dpi=150)
fig.savefig(path + '.svg', bbox_inches='tight', pad_inches=0.1, facecolor='white')
plt.close(fig)
print(f'✅ {path}.png')
