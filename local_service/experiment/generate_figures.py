#!/usr/bin/env python3
"""
为 Anon 报告生成所有图表：散点图、柱状图、热力图、时序图、弦图、威胁模型图
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import json
import os

# ============================================================
# 全局设置 — 使用 Noto Sans CJK JP 支持中文
# ============================================================
plt.rcParams['font.family'] = 'Noto Sans CJK JP'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(OUTPUT_DIR, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

# 学术配色
C1 = '#2c7bb6'  # 蓝
C2 = '#d7191c'  # 红
C3 = '#fdae61'  # 橙
C4 = '#1a9641'  # 绿
C5 = '#abd9e9'  # 浅蓝
C6 = '#7b3294'  # 紫
COLORS_5 = ['#2c7bb6', '#fdae61', '#1a9641', '#d7191c', '#7b3294']
COLORS_4 = ['#d7191c', '#fdae61', '#2c7bb6', '#1a9641']


def save(fig, name):
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, bbox_inches='tight', pad_inches=0.1)
    print(f'  -> {path}')
    plt.close(fig)


# ============================================================
# 图 1: RQ2 — T2 隐私-效用权衡散点连线图
# ============================================================
def fig_rq2_t2_tradeoff():
    """ε vs MRE，DAG vs No-DAG 对比"""
    epsilons = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    dag_mre   = [6.85, 3.94, 2.79, 1.57, 0.45, 0.09]
    nodag_mre = [10.66, 5.97, 4.63, 1.76, 0.08, 0.00]
    dag_constraint   = [99.6, 99.6, 99.6, 99.6, 99.6, 99.6]
    nodag_constraint = [50.9, 54.3, 54.3, 66.8, 89.4, 90.1]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    # 左: MRE
    ax1.plot(epsilons, dag_mre, 'o-', color=C1, linewidth=2, markersize=8, label='DAG 约束传播 (Anon)')
    ax1.plot(epsilons, nodag_mre, 's--', color=C2, linewidth=2, markersize=8, label='独立扰动 (No-DAG)')
    ax1.fill_between(epsilons, dag_mre, nodag_mre, alpha=0.12, color=C2)
    ax1.set_xscale('log')
    ax1.set_xlabel('ε (隐私预算)', fontsize=12)
    ax1.set_ylabel('均值相对误差 MRE (%)', fontsize=12)
    ax1.set_title('T2 数值扰动精度对比', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=9, frameon=False)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0.08, 12)
    ax1.annotate('约束保持\nMRE=2.79%', xy=(1.0, 2.79), xytext=(2.5, 5.5),
                 arrowprops=dict(arrowstyle='->', color='gray'), fontsize=9, color=C1)

    # 右: 约束满足率
    ax2.plot(epsilons, dag_constraint, 'o-', color=C1, linewidth=2, markersize=8, label='DAG 约束传播')
    ax2.plot(epsilons, nodag_constraint, 's--', color=C2, linewidth=2, markersize=8, label='独立扰动')
    ax2.fill_between(epsilons, nodag_constraint, dag_constraint, alpha=0.12, color=C2)
    ax2.set_xscale('log')
    ax2.set_xlabel('ε (隐私预算)', fontsize=12)
    ax2.set_ylabel('约束满足率 (%)', fontsize=12)
    ax2.set_title('DAG 约束满足率对比', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=9, frameon=False)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 105)

    fig.suptitle('RQ2：T2 数值脱敏隐私-效用权衡 ($\\varepsilon$-LDP)', fontsize=14,
                 fontweight='bold', y=1.02)
    save(fig, 'fig_rq2_t2_tradeoff.pdf')


# ============================================================
# 图 2: RQ2 — T3 语义扰动散点图 (双纵轴)
# ============================================================
def fig_rq2_t3_tradeoff():
    """ε vs 语义相似度 + 领域一致性"""
    epsilons = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    st_sim   = [0.731, 0.734, 0.738, 0.745, 0.768, 0.801]
    # 一致性数据来自 report 中的报告值
    consistency = [67.4, 66.3, 65.4, 65.1, 63.3, 60.8]
    m3e_sim = [0.790, 0.790, 0.791, 0.793, 0.799, 0.811]

    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    ax2 = ax1.twinx()

    # 左轴: 语义相似度
    ax1.plot(epsilons, st_sim, 'o-', color=C1, linewidth=2, markersize=8, label='ST 相似度 (Anon)')
    ax1.plot(epsilons, m3e_sim, 's--', color=C5, linewidth=2, markersize=7, label='M3E 相似度')
    ax1.set_xscale('log')
    ax1.set_xlabel('ε (隐私预算)', fontsize=12)
    ax1.set_ylabel('语义相似度', fontsize=12, color=C1)
    ax1.tick_params(axis='y', labelcolor=C1)
    ax1.set_ylim(0.70, 0.85)

    # 右轴: 领域一致性
    ax2.plot(epsilons, consistency, 'D-', color=C2, linewidth=2, markersize=8, label='领域一致性')
    ax2.set_ylabel('领域一致性 (%)', fontsize=12, color=C2)
    ax2.tick_params(axis='y', labelcolor=C2)
    ax2.set_ylim(55, 75)

    # 合并图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9, frameon=False, loc='lower right')

    ax1.grid(True, alpha=0.3)
    ax1.set_title('RQ2：T3 医疗实体语义扰动隐私-效用权衡', fontsize=13, fontweight='bold')

    # 标注推荐点
    ax1.annotate('推荐配置\nε=1.0, sim=0.738\n一致性=65.4%',
                 xy=(1.0, 0.738), xytext=(2.5, 0.77),
                 arrowprops=dict(arrowstyle='->', color='gray'),
                 fontsize=9, ha='center')

    save(fig, 'fig_rq2_t3_tradeoff.pdf')


# ============================================================
# 图 3: RQ4 — 前缀注入攻击防御柱状图
# ============================================================
def fig_rq4_prefix_defense():
    """5 消融层级的 PII 提取率对比"""
    levels = ['None\n(原始数据)', 'T1\n(姓名+FF3)', 'T1+T2\n(+数值DP)', 'T1+T3\n(+医疗术语)', 'Full\n(全链路)']
    extraction = [100, 3.3, 0.0, 3.3, 0.0]
    colors = [C2, C4, C1, C4, C1]
    hatches = ['', '', '', '///', '///']

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(levels, extraction, color=colors, edgecolor='white', linewidth=1.2, width=0.6)
    for bar, val in zip(bars, extraction):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                f'{val}%' if val > 0 else '0%',
                ha='center', va='bottom', fontsize=12, fontweight='bold',
                color=C2 if val > 0 else C1)

    # 添加防御效果箭头标注
    ax.annotate('', xy=(1, 50), xytext=(0, 97),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))
    ax.text(0.5, 70, '↓96.7%', fontsize=11, fontweight='bold', color=C1, ha='center')
    ax.annotate('', xy=(2, 10), xytext=(1, 40),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))
    ax.text(1.5, 25, '↓100%', fontsize=11, fontweight='bold', color=C1, ha='center')

    ax.set_ylabel('PII 提取率 (%)', fontsize=12)
    ax.set_title('RQ4：前缀注入攻击防御效果（消融实验）', fontsize=13, fontweight='bold')
    ax.set_ylim(0, 115)
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=0, color='black', linewidth=0.8)

    # 底部图例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=C1, label='全链路防御 (提取率 0%)'),
        Patch(facecolor=C4, label='部分防御'),
        Patch(facecolor=C2, label='无防御 (原始数据)'),
    ]
    ax.legend(handles=legend_elements, fontsize=9, frameon=False, loc='upper right')

    save(fig, 'fig_rq4_prefix_defense.pdf')


# ============================================================
# 图 4: RQ4 — ε-LDP 对 MIA 的理论防御散点图
# ============================================================
def fig_rq4_mia_theory():
    """ε vs AUC 上界 vs 攻击者优势"""
    epsilons = [0.1, 0.5, 1.0, 2.0, 5.0]
    advantage = [0.025, 0.123, 0.231, 0.381, 0.493]
    auc_upper = [0.525, 0.623, 0.731, 0.881, 0.993]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))

    # 左: AUC 上界
    ax1.plot(epsilons, auc_upper, 'o-', color=C1, linewidth=2, markersize=10, markerfacecolor='white')
    ax1.fill_between(epsilons, 0.5, auc_upper, alpha=0.15, color=C1)
    ax1.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, label='随机猜测 (AUC=0.5)')
    ax1.axhline(y=0.731, color=C2, linestyle=':', linewidth=1.5, label='推荐配置 ε=1.0 (AUC≤0.731)')
    ax1.set_xlabel('ε (隐私预算)', fontsize=12)
    ax1.set_ylabel('MIA AUC 理论上界', fontsize=12)
    ax1.set_title('MIA 防御理论上界 vs ε', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=8, frameon=False)
    ax1.grid(True, alpha=0.3)

    # 右: 攻击者优势
    ax2.bar(range(len(epsilons)), advantage, color=[C1, C1, C4, '#fdae61', C2], edgecolor='white')
    ax2.set_xticks(range(len(epsilons)))
    ax2.set_xticklabels([f'ε={e}' for e in epsilons])
    ax2.set_ylabel('攻击者优势上界', fontsize=12)
    ax2.set_title('攻击者优势 vs ε', fontsize=12, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    for i, (e, a) in enumerate(zip(epsilons, advantage)):
        ax2.text(i, a + 0.01, f'{a:.3f}', ha='center', fontsize=10, fontweight='bold')
    ax2.axhline(y=0.231, color=C4, linestyle=':', linewidth=1.5, label='ε=1.0 上界')
    ax2.legend(fontsize=9, frameon=False)

    fig.suptitle('RQ4：ε-差分隐私对成员推断攻击的理论防御保证', fontsize=13, fontweight='bold', y=1.02)
    save(fig, 'fig_rq4_mia_theory.pdf')


# ============================================================
# 图 5: RQ6 — 模块消融热力图
# ============================================================
def fig_rq6_ablation_heatmap():
    """4 配置 × 5 指标的消融热力图"""
    configs = ['Regex-Only\n(纯正则)', 'No-DAG\n(无约束)', 'Random-T3\n(随机替换)', 'Anon\n(全链路)']
    metrics = ['实体 F1', 'BLEU-4\n(自然度)', 'T2 约束\n满足率', 'T3 领域\n一致性', '安全特性\n评分']

    # 数据矩阵 (每行一个配置)
    # F1, BLEU-4, T2约束, T3一致性, 安全评分(5分制)
    data = np.array([
        [63.5, 81.5,  0,    0,   1.0],   # Regex-Only
        [91.3, 79.4, 54.3,  0,   2.0],   # No-DAG
        [91.3, 86.3,  0,    5.0, 2.0],   # Random-T3
        [91.3, 82.2, 99.6, 65.4, 5.0],   # Anon
    ])

    # 归一化到0-1用于颜色映射 (每列独立)
    data_norm = np.zeros_like(data, dtype=float)
    for j in range(data.shape[1]):
        col = data[:, j]
        if col.max() > col.min():
            data_norm[:, j] = (col - col.min()) / (col.max() - col.min())
        else:
            data_norm[:, j] = 0.5

    fig, ax = plt.subplots(figsize=(10, 4.5))
    im = ax.imshow(data_norm, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)

    # 标注每个 cell
    for i in range(len(configs)):
        for j in range(len(metrics)):
            val = data[i, j]
            if val == 0:
                text = '---'
            elif j == 4:
                text = f'{val:.1f}★'
            elif j == 0 or j == 2 or j == 3:
                text = f'{val:.1f}%'
            else:
                text = f'{val:.1f}%'
            color = 'white' if data_norm[i, j] < 0.3 or data_norm[i, j] > 0.7 else 'black'
            text_color = 'white' if (data_norm[i, j] < 0.35 or data_norm[i, j] > 0.7) else 'black'
            ax.text(j, i, text, ha='center', va='center', fontsize=10,
                    fontweight='bold', color=text_color)

    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(metrics, fontsize=10)
    ax.set_yticks(range(len(configs)))
    ax.set_yticklabels(configs, fontsize=10)

    ax.set_title('RQ6：模块消融分析热力图', fontsize=13, fontweight='bold', pad=15)
    cbar = plt.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label('归一化得分 (0=最低, 1=最高)', fontsize=10)

    # 高亮 Anon 行
    ax.axhline(y=2.5, color=C1, linewidth=2.5, linestyle='-')
    ax.annotate('  ← 全链路最优', xy=(4.5, 3.0), fontsize=10, color=C1, fontweight='bold',
                va='center')

    save(fig, 'fig_rq6_ablation_heatmap.pdf')


# ============================================================
# 图 6: RQ7 — 下游任务效用的分组柱状图
# ============================================================
def fig_rq7_downstream():
    """5 消融层级 × 3 指标的分组柱状图"""
    levels = ['None\n(未脱敏)', 'T1\n(仅FF3)', 'T1+T2\n(+数值DP)', 'T1+T3\n(+医疗)', 'Full\n(全链路)']

    ppl_retention    = [100, 72, 72, 72, 72]
    completion_ret   = [100, 68, 68, 68, 68]
    knowledge_ret    = [93,  93, 93, 90, 93]

    x = np.arange(len(levels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars1 = ax.bar(x - width, ppl_retention, width, color=C1, edgecolor='white',
                   label='PPL 保留率 (语言建模质量)')
    bars2 = ax.bar(x, completion_ret, width, color=C4, edgecolor='white',
                   label='补全保留率 (查询理解)')
    bars3 = ax.bar(x + width, knowledge_ret, width, color=C3, edgecolor='white',
                   label='知识保留率 (医学事实)')

    # 标注数值
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.8, f'{h:.0f}%',
                    ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(levels, fontsize=10)
    ax.set_ylabel('保留率 (%)', fontsize=12)
    ax.set_title('RQ7：下游任务效用消融评测 — 各层级脱敏对模型训练质量的影响',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, frameon=False, loc='lower left')
    ax.set_ylim(0, 115)
    ax.grid(axis='y', alpha=0.3)

    # 标注关键洞察
    ax.annotate('T1 = 唯一效用代价源\n(T1→T1+T2→T1+T3→Full\nPPL 始终锁定 72%)',
                xy=(1, 72), xytext=(3.5, 60),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.2),
                fontsize=9, ha='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))

    ax.annotate('T2+T3 零附加代价',
                xy=(4, 72), xytext=(3.2, 85),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.2),
                fontsize=9, ha='center', color=C1, fontweight='bold')

    save(fig, 'fig_rq7_downstream.pdf')


# ============================================================
# 图 7: 训练收敛曲线 (时序折线图)
# ============================================================
def fig_training_loss():
    """5 模型的训练 loss 收敛曲线"""
    epochs = [1, 2, 3]
    models = {
        'None (未脱敏)':  [2.8525, 2.5743, 2.5097],
        'T1 (仅FF3)':     [3.0521, 2.6929, 2.5991],
        'T1+T2 (+DP)':    [3.0524, 2.6933, 2.5995],
        'T1+T3 (+医疗)':  [3.0558, 2.6970, 2.6040],
        'Full (全链路)':  [3.0550, 2.6963, 2.6031],
    }

    fig, ax = plt.subplots(figsize=(8, 5))
    markers = ['o', 's', 'D', '^', 'v']
    for (label, losses), marker, color in zip(models.items(), markers, COLORS_5):
        ax.plot(epochs, losses, marker=marker, color=color, linewidth=2,
                markersize=9, label=label, markerfacecolor='white')

    # 标注 PPL
    ppl_map = {1: 15.1, 3: 12.3}
    for epoch, ppl in ppl_map.items():
        ax.annotate(f'PPL={ppl}', xy=(epoch, 2.51), fontsize=9, color='gray')

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Training Loss', fontsize=12)
    ax.set_title('RQ7：模型训练收敛曲线 (Qwen2.5-0.5B + LoRA)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, frameon=False)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(epochs)

    # 小图：最终 PPL 对比
    axins = ax.inset_axes([0.55, 0.55, 0.4, 0.4])
    final_ppls = [12.3, 13.5, 13.5, 13.5, 13.5]
    labels_short = ['None', 'T1', 'T1+T2', 'T1+T3', 'Full']
    bar_colors = [C1, C2, C2, C2, C2]
    xi = np.arange(len(labels_short))
    axins.bar(xi, final_ppls, color=bar_colors, edgecolor='white')
    axins.set_xticks(xi)
    axins.set_xticklabels(labels_short, fontsize=6)
    axins.set_ylabel('Final PPL', fontsize=8)
    axins.set_title('最终 PPL 对比', fontsize=9, fontweight='bold')
    axins.tick_params(labelsize=7)
    axins.grid(axis='y', alpha=0.3)
    # 标注 T1 代价
    for i, (l, p) in enumerate(zip(labels_short, final_ppls)):
        axins.text(i, p + 0.15, str(p), ha='center', fontsize=7, fontweight='bold')
    axins.axhline(y=12.3, color=C1, linestyle='--', linewidth=0.8, alpha=0.6)

    save(fig, 'fig_training_loss.pdf')


# ============================================================
# 图 8: NER 性能对比柱状图
# ============================================================
def fig_ner_comparison():
    """3 方法 × P/R/F1"""
    methods = ['Regex-Only\n(纯正则)', '规则 NER\n(正则+词典)', '大模型 NER\n(GLM-4.7)']
    precision = [81.3, 96.2, 92.8]
    recall    = [52.1, 74.5, 89.9]
    f1        = [63.5, 83.9, 91.3]

    x = np.arange(len(methods))
    width = 0.25

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(x - width, precision, width, color=C5, edgecolor='white', label='Precision')
    ax.bar(x, recall, width, color=C4, edgecolor='white', label='Recall')
    ax.bar(x + width, f1, width, color=C1, edgecolor='white', label='F1-Score')

    for i in range(3):
        for j, vals in enumerate([precision, recall, f1]):
            v = vals[i]
            ax.text(x[i] + (j-1)*width, v + 1, f'{v:.1f}%', ha='center', fontsize=8, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=10)
    ax.set_ylabel('分数 (%)', fontsize=12)
    ax.set_title('RQ1：实体识别性能对比', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, frameon=False, loc='upper left')
    ax.set_ylim(0, 110)
    ax.grid(axis='y', alpha=0.3)

    save(fig, 'fig_ner_comparison.pdf')


# ============================================================
# 图 9: 威胁模型示意图 (DoS 攻击路径图)
# ============================================================
def fig_threat_model():
    """训练数据隐私威胁模型示意图"""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis('off')
    ax.set_aspect('equal')

    # === 左侧: 攻击者 ===
    attacker_box = plt.Rectangle((0.2, 3.5), 2.0, 1.2, fc=C2, alpha=0.15, ec=C2, lw=2, ls='--')
    ax.add_patch(attacker_box)
    ax.text(1.2, 4.4, '攻击者', fontsize=12, fontweight='bold', ha='center', color=C2)
    ax.text(1.2, 4.0, '• 查询权限\n• 前缀获取\n• 统计分析', fontsize=8, ha='center', color=C2)

    # 攻击类型
    attacks = [
        ('前缀注入攻击\n(Prefix Injection)', (0.2, 1.0), C2),
        ('成员推断攻击\n(MIA / LiRA)', (0.2, 5.8), C2),
    ]
    for text, pos, color in attacks:
        box = plt.Rectangle(pos, 2.0, 1.0, fc=color, alpha=0.08, ec=color, lw=1.5)
        ax.add_patch(box)
        ax.text(pos[0]+1.0, pos[1]+0.55, text, fontsize=8, ha='center', color=color, fontweight='bold')

    # 攻击箭头
    ax.annotate('', xy=(3.5, 4.7), xytext=(2.2, 4.2),
                arrowprops=dict(arrowstyle='->', color=C2, lw=2.5, connectionstyle='arc3,rad=0.2'))
    ax.annotate('', xy=(3.5, 2.0), xytext=(2.2, 1.5),
                arrowprops=dict(arrowstyle='->', color=C2, lw=2.5, connectionstyle='arc3,rad=-0.2'))
    ax.text(3.0, 3.3, '① 攻击\n训练模型', fontsize=9, ha='center', color=C2, fontweight='bold')

    # === 中间: 训练管线 ===
    pipeline_box = plt.Rectangle((3.5, 2.0), 3.5, 4.0, fc=C1, alpha=0.08, ec=C1, lw=2)
    ax.add_patch(pipeline_box)
    ax.text(5.25, 5.6, 'LLM 训练管线', fontsize=11, fontweight='bold', ha='center', color=C1)

    pipeline_steps = ['原始\n提示词', 'Anon\n脱敏', '模型\n训练', '部署\n模型']
    for i, step in enumerate(pipeline_steps):
        x = 4.1 + i * 0.8
        box = plt.Rectangle((x, 3.7), 0.7, 0.9, fc='white', ec=C1, lw=1.2)
        ax.add_patch(box)
        ax.text(x+0.35, 4.1, step, fontsize=7, ha='center', va='center')
        if i < 3:
            ax.annotate('', xy=(x+0.75, 4.15), xytext=(x+0.7, 4.15),
                        arrowprops=dict(arrowstyle='->', color=C1, lw=1))

    # === Anon 防御标注 ===
    defense_box = plt.Rectangle((4.7, 2.3), 1.2, 1.2, fc=C4, alpha=0.15, ec=C4, lw=2)
    ax.add_patch(defense_box)
    ax.text(5.3, 3.1, 'Anon\n防御层', fontsize=9, fontweight='bold', ha='center', color=C4)
    ax.text(5.3, 2.6, 'T1 FF3\nT2 mLDP+DAG\nT3 ST指数机制', fontsize=7, ha='center', color=C4)

    # 防御箭头
    ax.annotate('', xy=(8.5, 3.5), xytext=(7.0, 3.5),
                arrowprops=dict(arrowstyle='->', color=C4, lw=2.5))
    ax.text(7.8, 3.8, '② 脱敏后\n训练', fontsize=9, ha='center', color=C4, fontweight='bold')

    # === 右侧: 防御效果 ===
    result_box = plt.Rectangle((8.5, 2.0), 3.2, 4.0, fc=C4, alpha=0.06, ec=C4, lw=2)
    ax.add_patch(result_box)
    ax.text(10.1, 5.6, '防御效果', fontsize=11, fontweight='bold', ha='center', color=C4)

    results = [
        ('SR1: 前缀注入\n提取率 100% → 0%', 3.8, C4),
        ('SR2: MIA AUC\n$\\leq$0.731 (实测≈0.5)', 2.9, C1),
        ('SR3: ε-LDP\n形式化保证', 2.1, C1),
    ]
    for text, y, color in results:
        box = plt.Rectangle((8.7, y-0.25), 2.8, 0.7, fc=color, alpha=0.06, ec=color, lw=1.2)
        ax.add_patch(box)
        ax.text(10.1, y+0.1, text, fontsize=8, ha='center', color=color)

    ax.set_title('威胁模型与防御示意图', fontsize=14, fontweight='bold', pad=10)
    save(fig, 'fig_threat_model.pdf')


# ============================================================
# 图 10: T3 疾病-药物关系弦图
# ============================================================
def fig_t3_chord():
    """展示 T3 疾病→药物类比推理关系的简化弦图"""
    # 用简化方式画弦图: 疾病(左) → 药物(右) 的连接
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.2, 1.2)
    ax.axis('off')
    ax.set_aspect('equal')

    diseases = [
        ('高血压', 0), ('冠心病', 36), ('糖尿病', 72), ('抑郁症', 108),
        ('胃溃疡', 144), ('哮喘', 180), ('骨质疏松', 216), ('心房颤动', 252),
        ('甲亢', 288), ('COPD', 324),
    ]
    drugs = [
        ('硝苯地平', 18), ('阿司匹林', 54), ('二甲双胍', 90), ('氟西汀', 126),
        ('奥美拉唑', 162), ('沙丁胺醇', 198), ('阿仑膦酸钠', 234), ('华法林', 270),
        ('甲巯咪唑', 306), ('异丙托溴铵', 342),
    ]

    # 外环: 疾病
    for name, angle in diseases:
        rad = np.deg2rad(angle)
        x, y = 0.95 * np.cos(rad), 0.95 * np.sin(rad)
        ax.plot([1.05*x, 0.8*x], [1.05*y, 0.8*y], color=C2, lw=0.5, alpha=0.3)
        ax.text(1.15*x, 1.15*y, name, fontsize=8, ha='center', va='center',
                color=C2, fontweight='bold')

    # 外环: 药物
    for name, angle in drugs:
        rad = np.deg2rad(angle)
        x, y = 0.95 * np.cos(rad), 0.95 * np.sin(rad)
        ax.text(1.15*x, 1.15*y, name, fontsize=7, ha='center', va='center',
                color=C1)

    # 连接曲线 (疾病→药物): 用贝塞尔曲线
    pairs = [
        (0, 18, C2), (36, 54, C2), (72, 90, C2), (108, 126, C2),
        (144, 162, C2), (180, 198, C2), (216, 234, C2), (252, 270, C2),
        (288, 306, C2), (324, 342, C2),
        # 类比推理连接 (不同疾病的药物替换)
        (0, 54, C4),   # 高血压药→冠心病药 (类比)
        (72, 306, C4), # 糖尿病→甲亢
        (108, 270, C4), # 抑郁症→心房颤动
    ]
    for d_angle, r_angle, color in pairs:
        d_rad, r_rad = np.deg2rad(d_angle), np.deg2rad(r_angle)
        dx, dy = 0.75*np.cos(d_rad), 0.75*np.sin(d_rad)
        rx, ry = 0.75*np.cos(r_rad), 0.75*np.sin(r_rad)
        # 贝塞尔曲线
        mid_angle = (d_rad + r_rad) / 2
        mx, my = 0.3*np.cos(mid_angle), 0.3*np.sin(mid_angle)
        curve_x = [dx, mx, rx]
        curve_y = [dy, my, ry]
        alpha = 0.5 if color == C2 else 0.3
        lw = 1.5 if color == C2 else 1.0
        ls = '-' if color == C2 else '--'
        ax.plot(curve_x, curve_y, color=color, alpha=alpha, lw=lw, linestyle=ls)

    # 中心圆
    circle = plt.Circle((0, 0), 0.25, fc='white', ec='gray', lw=1.5)
    ax.add_patch(circle)
    ax.text(0, 0.07, 'T3 类比\n推理', fontsize=9, ha='center', va='center', fontweight='bold', color=C1)
    ax.text(0, -0.08, 'Anon', fontsize=8, ha='center', va='center', color='gray')

    # 图例
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=C2, lw=2, label='同域内关联 (同一疾病-药物对)'),
        Line2D([0], [0], color=C4, lw=1.5, ls='--', label='跨域类比推理 (疾病扰动后的药物对齐)'),
    ]
    ax.legend(handles=legend_elements, fontsize=8, frameon=False,
              loc='lower center', bbox_to_anchor=(0.5, -0.1), ncol=2)

    ax.set_title('T3 疾病-药物关系弦图：语义类比推理的向量空间关系', fontsize=13,
                 fontweight='bold', pad=5)
    save(fig, 'fig_t3_chord.pdf')


# ============================================================
# 图 11: 数据效用气泡图 (RQ2+3 综合)
# ============================================================
def fig_utility_bubble():
    """气泡图: x=Bleu, y=语义相似度, bubble size=约束满足率"""
    fig, ax = plt.subplots(figsize=(8, 5.5))

    methods = [
        ('Regex-Only', 45.2, 0.623, 0, C2, '占位符替换'),
        ('Presidio', 58.7, 0.718, 0, C3, '通用脱敏'),
        ('Anon\n($\\varepsilon$=1.0)', 82.7, 0.910, 99.6, C1, 'Anon 全链路'),
        ('原始文本\n(无脱敏)', 100.0, 1.000, 100, C4, '原始数据 (参考)'),
    ]

    for name, bleu, sim, constraint, color, label in methods:
        size = max(constraint, 15) * 8
        ax.scatter(bleu, sim, s=size, c=color, alpha=0.6, edgecolors=color,
                   linewidth=2, zorder=5)
        offset = 2 if 'Anon' in name else (-3 if 'Regex' in name else 2)
        ax.annotate(name, (bleu, sim), textcoords='offset points',
                    xytext=(0, offset), ha='center', fontsize=9, fontweight='bold',
                    color=color)

    ax.set_xlabel('BLEU-4 (文本自然度, %)', fontsize=12)
    ax.set_ylabel('语义相似度', fontsize=12)
    ax.set_title('RQ3：脱敏数据质量 — 综合效用气泡图', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(30, 110)
    ax.set_ylim(0.50, 1.08)

    # 气泡大小图例
    for size_val, label in [(800, '约束满足率 ≥99.6%'), (200, '约束满足率 <50%'), (0.1, '约束满足率 = 0')]:
        ax.scatter([], [], s=size_val, c='gray', alpha=0.3, label=label)
    ax.legend(fontsize=8, frameon=False, loc='lower right', title='气泡大小')

    save(fig, 'fig_utility_bubble.pdf')


# ============================================================
# 图 12: 跨样本一致性对比图
# ============================================================
def fig_cross_sample_consistency():
    """柱状图: 各实体类型的一致性"""
    entity_types = ['PERSON', 'PHONE', 'EMAIL', 'ID']
    methods_data = {
        'Anon (确定性哈希)': [100, 100, 100, 100],
        'Regex-Only\n(占位符替换)': [100, 100, 100, 100],
        '随机替换': [0, 0, 0, 0],
    }
    colors_map = {'Anon (确定性哈希)': C1, 'Regex-Only\n(占位符替换)': C3, '随机替换': C2}

    x = np.arange(len(entity_types))
    width = 0.25
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, (method, values) in enumerate(methods_data.items()):
        ax.bar(x + i*width, values, width, color=colors_map[method],
               edgecolor='white', label=method, alpha=0.85)

    ax.set_xticks(x + width)
    ax.set_xticklabels(entity_types, fontsize=11)
    ax.set_ylabel('一致映射率 (%)', fontsize=12)
    ax.set_title('RQ3：跨样本假名一致性 — 各实体类型对比', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, frameon=False, loc='lower left')
    ax.set_ylim(0, 120)
    ax.grid(axis='y', alpha=0.3)

    # 标注 Anon 的优势
    ax.annotate('Anon: 确定性的同时保留\n实体区分能力\n(不同人名→不同假名)',
                xy=(0.25, 100), xytext=(2, 80),
                arrowprops=dict(arrowstyle='->', color=C1, lw=1.2),
                fontsize=8.5, color=C1,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.6))
    ax.annotate('Regex-Only: 一致但丧失\n区分能力\n(所有人名→同一占位符)',
                xy=(1.25, 100), xytext=(3, 50),
                arrowprops=dict(arrowstyle='->', color=C3, lw=1.2),
                fontsize=8.5, color='#8B6914')

    save(fig, 'fig_cross_sample_consistency.pdf')


# ============================================================
# 主函数
# ============================================================
if __name__ == '__main__':
    print('生成报告图表...')
    print(f'输出目录: {FIG_DIR}')

    print('\n[1/11] RQ2 T2 隐私-效用权衡散点图...')
    fig_rq2_t2_tradeoff()

    print('[2/11] RQ2 T3 语义扰动双轴散点图...')
    fig_rq2_t3_tradeoff()

    print('[3/11] RQ4 前缀注入防御柱状图...')
    fig_rq4_prefix_defense()

    print('[4/11] RQ4 MIA 理论防御散点图...')
    fig_rq4_mia_theory()

    print('[5/11] RQ6 模块消融热力图...')
    fig_rq6_ablation_heatmap()

    print('[6/11] RQ7 下游效用分组柱状图...')
    fig_rq7_downstream()

    print('[7/11] 训练收敛时序折线图...')
    fig_training_loss()

    print('[8/11] NER 性能对比柱状图...')
    fig_ner_comparison()

    print('[9/11] 威胁模型示意图...')
    fig_threat_model()

    print('[10/11] T3 疾病-药物弦图...')
    fig_t3_chord()

    print('[11/11] 数据效用气泡图...')
    fig_utility_bubble()

    print('[_/_] 跨样本一致性对比图...')
    fig_cross_sample_consistency()

    print(f'\n✅ 全部完成! 共生成 12 张图表到 {FIG_DIR}')
