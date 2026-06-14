#!/usr/bin/env python3
"""
CISCN 用户交互时序图: Demo 页面脱敏流程
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
LGRAY = '#aaaaaa'
WHITE = '#ffffff'
BLUE  = '#1a56db'
GREEN = '#047857'
RED   = '#c81e1e'
ACCENT = '#f5f5f5'

def lifeline(ax, x, y0, y1, color=GRAY, lw=1.2):
    ax.plot([x, x], [y0, y1], color=color, lw=lw, ls='--', dashes=(6, 4), zorder=1)

def box(ax, x, y, w, h, text, color=BLACK, bg=WHITE, fs=8, bold=False, alpha=1.0, lw=1.0):
    rect = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.12',
                          fc=bg, ec=color, lw=lw, alpha=alpha, zorder=3)
    ax.add_patch(rect)
    weight = 'bold' if bold else 'normal'
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=fs,
            color=color, fontweight=weight, zorder=4)

def arrow_h(ax, x1, y, x2, color=GRAY, lw=1.0, label='', fs=7):
    ax.annotate('', xy=(x2, y), xytext=(x1, y),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw), zorder=5)
    if label:
        mid = (x1 + x2) / 2
        ax.text(mid, y + 0.12, label, fontsize=fs, color=color, ha='center',
                bbox=dict(boxstyle='round,pad=0.1', fc='white', alpha=0.9))

def arrow_self(ax, x, y, dx, color=GRAY, lw=1.0, label='', fs=7):
    """Self-loop arrow going right then back"""
    ax.annotate('', xy=(x + 0.15, y - 0.05), xytext=(x + 0.15, y + 0.35),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw,
                               connectionstyle='arc3,rad=0.4'), zorder=5)
    if label:
        ax.text(x + 0.65, y + 0.15, label, fontsize=fs, color=color,
                bbox=dict(boxstyle='round,pad=0.1', fc='white', alpha=0.9))

fig, ax = plt.subplots(figsize=(16, 11), facecolor='white')
ax.set_xlim(0, 16)
ax.set_ylim(0, 11)
ax.axis('off')

# ===== TITLE =====
ax.text(8, 10.7, 'Anon Frontend — User Interaction Sequence (Demo Page)',
        fontsize=15, fontweight='bold', ha='center', color=BLACK)
ax.text(8, 10.3, 'Interactive Sanitization Demo: Text Input → NER → Tiered Sanitization → Result Display',
        fontsize=8.5, ha='center', color=GRAY)

# ===== PARTICIPANTS (Top headers) =====
participants = [
    (1.5, 'User', BLUE),
    (5.0, 'Navbar/\nRouter', DARK),
    (8.0, 'DemoPage\nComponent', GREEN),
    (11.0, 'useSimulation\nHook', RED),
    (14.0, 'mockData\n(Static)', GRAY),
]
for x, name, color in participants:
    box(ax, x - 1.1, 9.6, 2.2, 0.65, name, color=color, bg=WHITE, fs=8, bold=True, lw=1.2)
    lifeline(ax, x, 9.6, 1.0, color, 1.0)

# ===== STEP 1: User visits /demo =====
y = 9.2
arrow_h(ax, 1.5, y, 5.0, BLUE, 1.2, '1. Navigate to /demo', 7.5)
box(ax, 4.4, 8.7, 1.2, 0.45, 'Route\nMatch', color=DARK, bg=ACCENT, fs=6.5, lw=0.8)

# ===== STEP 2: Component mounts =====
y = 8.4
arrow_h(ax, 5.0, y, 8.0, DARK, 1.0, '2. Render DemoPage', 7.5)

# ===== STEP 3: Load sample texts =====
y = 8.0
arrow_h(ax, 8.0, y, 14.0, GREEN, 0.9, '3. Load SAMPLE_TEXTS', 7)
arrow_h(ax, 14.0, 7.7, 8.0, GRAY, 0.7, '  return 5 medical texts', 6.5)

# ===== STEP 4: User selects sample / edits text =====
y = 7.2
arrow_h(ax, 1.5, y, 8.0, BLUE, 1.0, '4. Select sample / Edit input text', 7.5)
arrow_self(ax, 8.0, 6.9, 0.6, GREEN, 0.8, 'setInputText()', 6.5)

# ===== STEP 5: Adjust epsilon =====
y = 6.5
arrow_h(ax, 1.5, y, 8.0, BLUE, 1.0, '5. Adjust ε slider (0.1–10.0)', 7.5)
arrow_self(ax, 8.0, 6.2, 0.6, GREEN, 0.8, 'setEpsilon()', 6.5)

# ===== STEP 6: Click Run =====
y = 5.8
arrow_h(ax, 1.5, y, 8.0, BLUE, 1.2, '6. Click "Execute Sanitization"', 7.5)

# ===== STEP 7: Trigger simulation =====
y = 5.4
arrow_h(ax, 8.0, y, 11.0, GREEN, 1.0, '7. runSanitization()', 7.5)
box(ax, 10.4, 5.1, 1.2, 0.4, 'setIsProcessing\n(true)', color=RED, bg='#fef2f2', fs=6, lw=0.7)

# ===== STEP 8: Simulation engine =====
y = 4.8
arrow_h(ax, 11.0, y, 14.0, RED, 1.0, '8. Load SAMPLE_ENTITIES + SAMPLE_DAG_EDGES', 7)
arrow_h(ax, 14.0, 4.5, 11.0, GRAY, 0.7, '  return entities[] + dagEdges[]', 6.5)

# ===== STEP 9: Epsilon-based perturbation =====
y = 4.2
arrow_self(ax, 11.0, 4.0, 0.6, RED, 0.8, 'simulateWithEpsilon(ε, entities)', 6.5)
box(ax, 11.0, 3.5, 2.0, 0.8, 'T1: FF3 encrypt\nT2: mLDP perturb\nT3: ST+FT replace',
    color=RED, bg='#fef2f2', fs=6.5, lw=1.0)

# ===== STEP 10: Build result =====
y = 3.2
arrow_self(ax, 11.0, 2.8, 0.6, RED, 0.8, 'buildSanitizedText()', 6.5)
bar_h = 0.3
box(ax, 10.7, 2.5, 2.6, 0.35, 'buildHighlightedText() × 2', color=RED, bg='#fef2f2', fs=6.5, lw=0.7)

# ===== STEP 11: Return result =====
y = 2.2
arrow_h(ax, 11.0, y, 8.0, RED, 1.0, '9. Return SanitizationResult', 7.5)
box(ax, 8.2, 1.75, 2.8, 0.8,
    'entities[] + dagEdges[]\nmetrics + sanitizedText\nProcessing Time: ~1.2s',
    color=GREEN, bg='#ecfdf5', fs=7, bold=True, lw=1.0)

# ===== STEP 12: Render results =====
y = 1.6
arrow_h(ax, 8.0, y, 5.0, GREEN, 0.8, '10. Update UI (Framer Motion)', 7)
arrow_h(ax, 5.0, 1.3, 1.5, DARK, 0.8, '11. Display to User', 7)

# ===== Result panels shown =====
result_box_x = 0.8
box(ax, result_box_x, 0.3, 6.5, 0.85,
    'Display: Original vs Sanitized (side-by-side) | Entity Table | Metrics Panel | DAG Edges | Privacy-Utility Scatter',
    color=BLUE, bg='#eef2ff', fs=7, bold=True, lw=1.2)

# ===== Time axis (right side) =====
ax.text(15.5, 6.0, 'Total:\n~1.2s', fontsize=8, color=GRAY, ha='center',
        fontstyle='italic', fontfamily='monospace',
        bbox=dict(boxstyle='round,pad=0.3', fc='white', ec=LGRAY, alpha=0.8))

# ===== Legend =====
legend_y = 0.1
legend_items = [
    (BLUE, 'User Action'), (GREEN, 'Component State'), (RED, 'Simulation Logic'), (GRAY, 'Static Data')
]
for i, (color, label) in enumerate(legend_items):
    ax.plot([1.5 + i * 3.5], [legend_y], 's', color=color, markersize=8, mec=color, mfc=WHITE, mew=1.5)
    ax.text(1.5 + i * 3.5 + 0.2, legend_y - 0.03, label, fontsize=7, color=color, va='center')

fig.tight_layout(pad=0.3)
path = os.path.join(FIGS, 'fig_user_sequence')
fig.savefig(path + '.svg', bbox_inches='tight', pad_inches=0.1, facecolor='white')
fig.savefig(path + '.png', bbox_inches='tight', pad_inches=0.1, facecolor='white', dpi=150)
plt.close(fig)
print(f'✅ {path}.svg')
