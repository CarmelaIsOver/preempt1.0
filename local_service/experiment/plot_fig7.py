#!/usr/bin/env python3
"""
图7: 训练收敛曲线 — 5模型 × 3 epochs 时序折线图
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

BLUE   = '#2c7bb6'
RED    = '#d7191c'
GREEN  = '#1a9641'
ORANGE = '#fdae61'
PURPLE = '#7b3294'
GRAY   = '#555555'

epochs = [1, 2, 3]
models = {
    'None (Raw)':       ([2.8525, 2.5743, 2.5097], BLUE),
    'T1 (Name+FF3)':    ([3.0521, 2.6929, 2.5991], RED),
    'T1+T2 (+Num DP)':  ([3.0524, 2.6933, 2.5995], GREEN),
    'T1+T3 (+Med Term)':([3.0558, 2.6970, 2.6040], ORANGE),
    'Full (Anon)':      ([3.0550, 2.6963, 2.6031], PURPLE),
}
final_ppl = {'None (Raw)': 12.3, 'T1 (Name+FF3)': 13.5, 'T1+T2 (+Num DP)': 13.5,
             'T1+T3 (+Med Term)': 13.5, 'Full (Anon)': 13.5}

fig, ax = plt.subplots(figsize=(7.5, 4.8), facecolor='white')

for label, (losses, color) in models.items():
    ax.plot(epochs, losses, 'o-', color=color, lw=2.2, ms=9,
            mfc='white', mew=2, label=label, zorder=4)

ax.set_xlabel('Epoch', fontsize=11, color=GRAY)
ax.set_ylabel('Training Loss', fontsize=11, color=GRAY)
ax.set_title('Training Convergence (Qwen2.5-0.5B + LoRA, r=16)',
             fontsize=12, fontweight='bold', pad=10)
ax.legend(fontsize=8, frameon=True, loc='upper right', edgecolor='#ddd', ncol=2)
ax.grid(True, alpha=0.12, lw=0.4)
ax.set_xticks(epochs)
ax.tick_params(labelsize=9)
ax.set_xlim(0.8, 3.2)

# Final PPL annotation
ax.text(3.05, 2.61, f'Final PPL\nNone: 12.3\nOthers: 13.5', fontsize=7.5,
        color=GRAY, ha='left', va='center', fontstyle='italic',
        bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))

fig.tight_layout(pad=1.5)
path = os.path.join(FIGS, 'fig7_training_loss')
fig.savefig(path + '.svg', bbox_inches='tight', pad_inches=0.1, facecolor='white')
fig.savefig(path + '.png', bbox_inches='tight', pad_inches=0.1, facecolor='white', dpi=150)
plt.close(fig)
print(f'✅ {path}.svg')
