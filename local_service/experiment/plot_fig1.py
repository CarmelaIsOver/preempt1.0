#!/usr/bin/env python3
"""
图1: T2 数值脱敏隐私-效用权衡 — 双面板散点连线图
ε vs MRE (左) + ε vs 约束满足率 (右)
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as tkr
import numpy as np
import os

# ===== 全局样式 =====
plt.rcParams.update({
    'font.family': 'Noto Sans CJK JP',
    'axes.unicode_minus': False,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.linewidth': 1.0,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
})

OUT = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(os.path.dirname(OUT), 'figures')
os.makedirs(FIGS, exist_ok=True)

# ===== 配色 =====
BLUE   = '#2c7bb6'
RED    = '#d7191c'
GRAY   = '#555555'
LIGHT  = '#f0f0f0'

# ===== 数据 =====
epsilon    = np.array([0.1, 0.5, 1.0, 2.0, 5.0, 10.0])
dag_mre    = np.array([6.85, 3.94, 2.79, 1.57, 0.45, 0.09])
nodag_mre  = np.array([10.66, 5.97, 4.63, 1.76, 0.08, 0.00])
dag_con    = np.array([99.6, 99.6, 99.6, 99.6, 99.6, 99.6])
nodag_con  = np.array([50.9, 54.3, 54.3, 66.8, 89.4, 90.1])

# ===== 创建画布 =====
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.8), facecolor='white')

# ==================== 左面板: MRE ====================
ax1.plot(epsilon, dag_mre, 'o-', color=BLUE, lw=2.8, ms=10,
         mfc='white', mew=2.2, label='DAG-Constrained (Anon)', zorder=5)
ax1.plot(epsilon, nodag_mre, 's--', color=RED, lw=2.5, ms=10,
         mfc='white', mew=2.2, label='Independent (No-DAG)', zorder=4)
ax1.fill_between(epsilon, dag_mre, nodag_mre, color=RED, alpha=0.08)

ax1.set_xscale('log')
ax1.set_xlabel('ε (Privacy Budget)', fontsize=13, fontweight='bold', color=GRAY)
ax1.set_ylabel('Mean Relative Error MRE (%)', fontsize=13, fontweight='bold', color=GRAY)
ax1.set_title('T2 Numerical Perturbation Accuracy', fontsize=14, fontweight='bold', pad=12)
ax1.legend(fontsize=9.5, frameon=False, loc='upper right')
ax1.grid(True, alpha=0.2, lw=0.5)
ax1.set_xlim(0.08, 12)
ax1.set_ylim(-1, 13)
ax1.tick_params(labelsize=10)

# 标注 ε=1.0 推荐点
idx = 2  # epsilon=1.0
ax1.annotate(f'Recommended: ε=1.0\nMRE={dag_mre[idx]:.2f}%',
             xy=(1.0, dag_mre[idx]), xytext=(2.8, 5.5),
             fontsize=9.5, fontweight='bold', color=BLUE,
             arrowprops=dict(arrowstyle='->', color=BLUE, lw=1.3),
             bbox=dict(boxstyle='round,pad=0.4', fc='white', ec=BLUE, alpha=0.85))

# No-DAG collapse annotation
ax1.annotate('No-DAG:\nMRE spikes to 10.66%\nat low ε',
             xy=(0.1, nodag_mre[0]), xytext=(0.35, 9.0),
             fontsize=8.5, color=RED, fontstyle='italic',
             arrowprops=dict(arrowstyle='->', color=RED, lw=1.0, alpha=0.7))

# ==================== 右面板: 约束满足率 ====================
ax2.plot(epsilon, dag_con, 'o-', color=BLUE, lw=2.8, ms=10,
         mfc='white', mew=2.2, label='DAG-Constrained', zorder=5)
ax2.plot(epsilon, nodag_con, 's--', color=RED, lw=2.5, ms=10,
         mfc='white', mew=2.2, label='Independent (No-DAG)', zorder=4)
ax2.fill_between(epsilon, nodag_con, dag_con, color=RED, alpha=0.08)

ax2.set_xscale('log')
ax2.set_xlabel('ε (Privacy Budget)', fontsize=13, fontweight='bold', color=GRAY)
ax2.set_ylabel('Constraint Satisfaction Rate (%)', fontsize=13, fontweight='bold', color=GRAY)
ax2.set_title('DAG Constraint Preservation', fontsize=14, fontweight='bold', pad=12)
ax2.legend(fontsize=9.5, frameon=False, loc='lower right')
ax2.grid(True, alpha=0.2, lw=0.5)
ax2.set_ylim(30, 108)
ax2.tick_params(labelsize=10)

# 99.6% reference line
ax2.axhline(y=99.6, color=BLUE, lw=0.8, ls=':', alpha=0.6)
ax2.text(3.5, 100.8, 'Stable at 99.6%', fontsize=9, color=BLUE, fontweight='bold',
         bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.8))

# No-DAG collapse at low ε
ax2.annotate('ε=0.1\nConstraint drops\nto 50.9%',
             xy=(0.1, nodag_con[0]), xytext=(0.55, 58),
             fontsize=9, color=RED, fontstyle='italic',
             arrowprops=dict(arrowstyle='->', color=RED, lw=1.0, alpha=0.7),
             bbox=dict(boxstyle='round,pad=0.3', fc='white', ec=RED, alpha=0.7))

# Gain from constraint propagation (text only, no arrow)
ax2.text(0.12, 80, 'Gain from\nconstraint\npropagation',
         fontsize=10, color=GRAY, fontweight='bold',
         ha='left', va='center')

# ===== 全局标题 =====
fig.suptitle('RQ2: T2 Numerical De-identification — Privacy-Utility Tradeoff under $\\varepsilon$-LDP',
             fontsize=15, fontweight='bold', y=1.02)

plt.tight_layout(pad=2.0)
path = os.path.join(FIGS, 'fig1_t2_tradeoff.pdf')
try:
    fig.savefig(path, bbox_inches='tight', pad_inches=0.15, facecolor='white')
except PermissionError:
    pass
# 同时保存 PNG 预览
png_path = path.replace('.pdf', '.png')
fig.savefig(png_path, bbox_inches='tight', pad_inches=0.15, facecolor='white', dpi=150)
svg_path = path.replace('.pdf', '.svg')
fig.savefig(svg_path, bbox_inches='tight', pad_inches=0.15, facecolor='white')
plt.close(fig)
print(f'✅ PDF: {path}')
print(f'✅ PNG: {png_path}')
print(f'   文件路径 (Windows): D:\\Desktop\\mytask\\local_service\\figures\\fig1_t2_tradeoff.png')
