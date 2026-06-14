#!/usr/bin/env python3
"""
图6: RQ7 下游任务效用消融评测 — 分组柱状图
5 ablation levels × 3 retention metrics
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
GREEN  = '#1a9641'
ORANGE = '#fdae61'
RED    = '#d7191c'
GRAY   = '#555555'

levels      = ['None\n(Raw Data)', 'T1\n(Name+FF3)', 'T1+T2\n(+Num DP)', 'T1+T3\n(+Med Term)', 'Full\n(All Anon)']
ppl_ret     = [100, 72, 72, 72, 72]
compl_ret   = [100, 68, 68, 68, 68]
knowl_ret   = [93,  93, 93, 90, 93]

x = np.arange(len(levels))
width = 0.22

fig, ax = plt.subplots(figsize=(9, 5), facecolor='white')

b1 = ax.bar(x - width, ppl_ret, width, color=BLUE, edgecolor='white', lw=0.8,
            label='PPL Retention (LM Quality)', zorder=3)
b2 = ax.bar(x, compl_ret, width, color=GREEN, edgecolor='white', lw=0.8,
            label='Completion Retention (Query Understanding)', zorder=3)
b3 = ax.bar(x + width, knowl_ret, width, color=ORANGE, edgecolor='white', lw=0.8,
            label='Knowledge Retention (Medical Facts)', zorder=3)

# 柱顶数值
for bars in [b1, b2, b3]:
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 1.0, f'{h:.0f}%',
                ha='center', va='bottom', fontsize=7.5, fontweight='bold', color=GRAY)

ax.set_xticks(x)
ax.set_xticklabels(levels, fontsize=10)
ax.set_ylabel('Retention Rate (%)', fontsize=11, color=GRAY)
ax.set_title('Downstream Utility — Ablation Study (Qwen2.5-0.5B + LoRA)',
             fontsize=12, fontweight='bold', pad=10)
ax.set_ylim(0, 115)
ax.grid(axis='y', alpha=0.12, lw=0.4)
ax.tick_params(labelsize=10)
ax.set_xlim(-0.5, 4.5)

# 图例
ax.legend(fontsize=8, frameon=True, loc='lower left', edgecolor='#ddd')

# 核心洞察标注
ax.annotate('T1 = Sole utility cost\n(PPL from 100% → 72%)',
            xy=(1.25, 72), xytext=(3.8, 82),
            fontsize=8.5, color=RED, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color=RED, lw=1.0),
            bbox=dict(boxstyle='round,pad=0.3', fc='#fff5f5', ec=RED, alpha=0.85))

ax.annotate('T2 + T3:\nZero marginal cost',
            xy=(2.75, 72), xytext=(3.8, 58),
            fontsize=8.5, color=GREEN, fontweight='bold', ha='center',
            arrowprops=dict(arrowstyle='->', color=GREEN, lw=1.0),
            bbox=dict(boxstyle='round,pad=0.3', fc='#f5fff5', ec=GREEN, alpha=0.85))

fig.tight_layout(pad=1.5)
path = os.path.join(FIGS, 'fig6_rq7_downstream')
fig.savefig(path + '.svg', bbox_inches='tight', pad_inches=0.1, facecolor='white')
fig.savefig(path + '.png', bbox_inches='tight', pad_inches=0.1, facecolor='white', dpi=150)
plt.close(fig)
print(f'✅ {path}.svg')
