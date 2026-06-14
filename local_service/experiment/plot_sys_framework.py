#!/usr/bin/env python3
"""
CISCN 功能框架图: Anon 前端系统模块架构
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import os

plt.rcParams.update({
    'font.family': 'Noto Sans CJK JP', 'axes.unicode_minus': False,
    'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
})

OUT = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(os.path.dirname(OUT), 'figures')
os.makedirs(FIGS, exist_ok=True)

BLACK = '#1a1a1a'
DARK  = '#333333'
GRAY  = '#666666'
LGRAY = '#999999'
BG    = '#f8f8f8'
WHITE = '#ffffff'
BLUE  = '#1a56db'
RED   = '#c81e1e'
GREEN = '#047857'
ACCENT = '#f0f0f0'

def box(ax, x, y, w, h, text, color=BLACK, bg=WHITE, fontsize=9, bold=False, alpha=1.0, lw=1.2):
    """Draw a rounded box with text"""
    fc = bg if bg else WHITE
    rect = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.15',
                          fc=fc, ec=color, lw=lw, alpha=alpha, zorder=3)
    ax.add_patch(rect)
    weight = 'bold' if bold else 'normal'
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=fontsize,
            color=color, fontweight=weight, zorder=4)

def arrow(ax, x1, y1, x2, y2, color=GRAY, lw=1.0, style='->'):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw))

def dashed_box(ax, x, y, w, h, label, color=GRAY):
    rect = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.15',
                          fc='none', ec=color, lw=1.2, ls='--', zorder=1, alpha=0.6)
    ax.add_patch(rect)
    ax.text(x + 0.2, y + h - 0.2, label, fontsize=7.5, color=color, fontstyle='italic', va='top')

fig, ax = plt.subplots(figsize=(16, 10), facecolor='white')
ax.set_xlim(0, 16)
ax.set_ylim(0, 10)
ax.axis('off')

# ===== TITLE =====
ax.text(8, 9.55, 'Anon Frontend — System Functional Framework',
        fontsize=16, fontweight='bold', ha='center', color=BLACK)
ax.text(8, 9.15, 'React 18 + TypeScript + Vite + Tailwind CSS + Recharts + Framer Motion',
        fontsize=9, ha='center', color=GRAY)

# ===== LAYER 0: Entry =====
box(ax, 6.5, 8.3, 3.0, 0.55, 'Browser Entry\nindex.html → main.tsx → App.tsx',
    color=DARK, bg=ACCENT, fontsize=7.5, lw=1.0)
arrow(ax, 8, 8.3, 8, 7.9, DARK, 0.8)

# ===== LAYER 1: Router + Layout =====
box(ax, 6.2, 7.3, 3.6, 0.6, 'BrowserRouter + AnimatePresence\nLayout (Navbar + Content)',
    color=BLUE, bg='#eef2ff', fontsize=8, bold=True, lw=1.2)
arrow(ax, 8, 8.3, 8, 7.9, DARK, 0.8)

# ===== LAYER 2: 5 Pages (Core) =====
pages = [
    (0.3, 5.2, 'Dashboard\n仪表盘', '#1a56db', 'Hero + KPIs\nCharts + Metrics'),
    (3.2, 5.2, 'DemoPage\n交互演示', '#047857', 'Text Input + ε Slider\nNER + Sanitize'),
    (6.1, 5.2, 'AttackSim\n攻击防御', '#c81e1e', 'Prefix Injection\nMIA ROC Curve'),
    (9.0, 5.2, 'Comparison\n方法对比', '#7c3aed', 'Radar Chart\nComparison Table'),
    (11.9, 5.2, 'Architecture\n系统架构', '#d97706', 'Pipeline\nAlgorithms'),
]
for x, y, title, color, desc in pages:
    box(ax, x, y, 2.7, 1.9, '', color=color, bg=WHITE, lw=1.5)
    ax.text(x + 1.35, y + 1.55, title, fontsize=9, fontweight='bold', ha='center', color=color)
    ax.text(x + 1.35, y + 0.9, desc, fontsize=7, ha='center', color=GRAY)
    # Route label
    routes = ['/', '/demo', '/attack', '/comparison', '/architecture']
    ax.text(x + 1.35, y + 0.25, routes[pages.index((x, y, title, color, desc))],
            fontsize=6.5, ha='center', color=LGRAY, fontfamily='monospace')

# Arrows from Layout to pages
for x, y, _, _, _ in pages:
    arrow(ax, 8, 7.3, x + 1.35, y + 1.9, GRAY, 0.6)

# ===== LAYER 3: Shared Components =====
dashed_box(ax, 0.3, 2.7, 5.5, 2.1, 'Reusable Components', GRAY)
box(ax, 0.5, 3.3, 2.5, 0.55, 'Card / MetricCard / Badge', color=DARK, bg=WHITE, fontsize=7, lw=0.8)
box(ax, 3.2, 3.3, 2.5, 0.55, 'RadarChart / ComparisonTable', color=DARK, bg=WHITE, fontsize=7, lw=0.8)
box(ax, 0.5, 2.85, 2.5, 0.4, 'Framer Motion Animations', color=LGRAY, bg=WHITE, fontsize=6.5, lw=0.6)
box(ax, 3.2, 2.85, 2.5, 0.4, 'Recharts Data Visualization', color=LGRAY, bg=WHITE, fontsize=6.5, lw=0.6)

# ===== LAYER 4: Data Layer =====
dashed_box(ax, 6.5, 2.7, 5.5, 2.1, 'Data & State Layer', GRAY)
box(ax, 6.7, 3.3, 2.5, 0.55, 'mockData.ts\n(sample texts + entities)', color=DARK, bg=WHITE, fontsize=7, lw=0.8)
box(ax, 9.4, 3.3, 2.5, 0.55, 'constants.ts\n(types + epsilon presets)', color=DARK, bg=WHITE, fontsize=7, lw=0.8)
box(ax, 6.7, 2.85, 5.2, 0.4, 'useSimulation.ts — sanitization simulation engine', color=GREEN, bg='#ecfdf5', fontsize=7.5, bold=True, lw=1.0)

# Arrows from pages to data
for x, y, _, _, _ in pages:
    arrow(ax, x + 1.35, y, 8, 4.8, GRAY, 0.4, '-')

# ===== LAYER 5: Deployment =====
box(ax, 3.0, 1.5, 10.0, 0.7,
    'Deployment: Vite Build → Static Assets  |  Target: CISCN 2025 Demo Showcase  |  No Backend — Pure Frontend SPA',
    color=DARK, bg=ACCENT, fontsize=8, lw=1.0)
arrow(ax, 8, 2.7, 8, 2.2, DARK, 0.8)

# ===== RIGHT: Tech Stack Sidebar =====
tech_x = 13.2
ax.text(tech_x, 6.5, 'Tech Stack', fontsize=9, fontweight='bold', color=BLACK, ha='center')
tech_items = [
    ('React 18', BLUE), ('TypeScript', BLUE), ('Vite 6', DARK),
    ('Tailwind CSS', DARK), ('Framer Motion', GRAY), ('Recharts', GRAY),
    ('react-router-dom', LGRAY),
]
for i, (name, color) in enumerate(tech_items):
    box(ax, tech_x - 0.7, 6.0 - i * 0.42, 2.8, 0.35, name, color=color, bg=WHITE, fontsize=7, lw=0.7)

# ===== BOTTOM: Legend =====
ax.text(8, 1.05, 'Pages (5 routes)  |  Reusable Components (6)  |  Data Store (3 files)  |  Simulation Engine (1 hook)',
        fontsize=7.5, ha='center', color=LGRAY)

fig.tight_layout(pad=0.5)
path = os.path.join(FIGS, 'fig_sys_framework')
fig.savefig(path + '.svg', bbox_inches='tight', pad_inches=0.1, facecolor='white')
fig.savefig(path + '.png', bbox_inches='tight', pad_inches=0.1, facecolor='white', dpi=150)
plt.close(fig)
print(f'✅ {path}.svg')
