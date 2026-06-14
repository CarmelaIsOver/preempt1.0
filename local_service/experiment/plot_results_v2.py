"""
绘制实验2 v2 图表（基于 data.json 真实数据的结果）
"""

import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

for font in ["SimHei", "WenQuanYi Micro Hei", "Noto Sans CJK SC", "DejaVu Sans"]:
    try:
        matplotlib.font_manager.findfont(font)
        plt.rcParams["font.sans-serif"] = [font, "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
        break
    except Exception:
        continue

os.makedirs("experiment/results", exist_ok=True)

# 加载数据
csv_path = "experiment/results/experiment_results_v2.csv"
if not os.path.exists(csv_path):
    print(f"错误：找不到 {csv_path}，请先运行 run_experiment_v2.py")
    exit(1)

result_df = pd.read_csv(csv_path)
epsilons = result_df["epsilon"].values

COLOR_DAG = "#2196F3"
COLOR_NODAG = "#FF5722"
MARKER_DAG, MARKER_NODAG = "o", "s"
ALPHA_FILL = 0.15
FIGSIZE = (9, 5.5)


def save_and_close(fig, name):
    path = f"experiment/results/{name}"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  保存: {path}")
    plt.close(fig)


# ========================================================================
# 图 1：MRE 对比（核心）
# ========================================================================
print("绘制图 1: MRE 对比...")
fig, ax = plt.subplots(figsize=FIGSIZE)

ax.errorbar(epsilons, result_df["dag_mre_mean"], yerr=result_df["dag_mre_std"],
            marker=MARKER_DAG, ls="-", color=COLOR_DAG, lw=2, ms=8, capsize=4,
            label="DAG+mLDP (ours)")
ax.fill_between(epsilons,
                result_df["dag_mre_mean"] - result_df["dag_mre_std"],
                result_df["dag_mre_mean"] + result_df["dag_mre_std"],
                color=COLOR_DAG, alpha=ALPHA_FILL)

ax.errorbar(epsilons, result_df["nodag_mre_mean"], yerr=result_df["nodag_mre_std"],
            marker=MARKER_NODAG, ls="--", color=COLOR_NODAG, lw=2, ms=8, capsize=4,
            label="No-DAG (independent)")
ax.fill_between(epsilons,
                result_df["nodag_mre_mean"] - result_df["nodag_mre_std"],
                result_df["nodag_mre_mean"] + result_df["nodag_mre_std"],
                color=COLOR_NODAG, alpha=ALPHA_FILL)

ax.set_xscale("log")
ax.set_xlabel("Privacy Budget ε (log scale)", fontsize=12)
ax.set_ylabel("Mean Relative Error (%)", fontsize=12)
ax.set_title("Privacy-Utility Tradeoff (data.json, 200 real samples)", fontsize=14, fontweight="bold")
ax.legend(fontsize=10, loc="upper right")
ax.grid(True, alpha=0.3)
ax.set_xlim(0.08, 12)
save_and_close(fig, "mre_v2.png")

# ========================================================================
# 图 2：约束满足率
# ========================================================================
print("绘制图 2: 约束满足率...")
fig, ax = plt.subplots(figsize=FIGSIZE)

ax.errorbar(epsilons, result_df["dag_constraint_mean"], yerr=result_df["dag_constraint_std"],
            marker=MARKER_DAG, ls="-", color=COLOR_DAG, lw=2, ms=8, capsize=4, label="DAG+mLDP")
ax.errorbar(epsilons, result_df["nodag_constraint_mean"], yerr=result_df["nodag_constraint_std"],
            marker=MARKER_NODAG, ls="--", color=COLOR_NODAG, lw=2, ms=8, capsize=4, label="No-DAG")
ax.fill_between(epsilons,
                result_df["nodag_constraint_mean"] - result_df["nodag_constraint_std"],
                result_df["nodag_constraint_mean"] + result_df["nodag_constraint_std"],
                color=COLOR_NODAG, alpha=ALPHA_FILL)

ax.set_xscale("log")
ax.axhline(y=100, color=COLOR_DAG, ls=":", alpha=0.5)
ax.set_xlabel("Privacy Budget ε (log scale)", fontsize=12)
ax.set_ylabel("Constraint Satisfaction Rate (%)", fontsize=12)
ax.set_title("Constraint Preservation (data.json)", fontsize=14, fontweight="bold")
ax.legend(fontsize=10, loc="lower right")
ax.grid(True, alpha=0.3)
ax.set_ylim(-5, 110)
ax.set_xlim(0.08, 12)
save_and_close(fig, "constraint_v2.png")

# ========================================================================
# 图 3：约束违反幅度
# ========================================================================
print("绘制图 3: 约束违反幅度...")
fig, ax = plt.subplots(figsize=FIGSIZE)
ax.errorbar(epsilons, result_df["nodag_violation_mean"], yerr=result_df["nodag_violation_std"],
            marker=MARKER_NODAG, ls="-", color=COLOR_NODAG, lw=2, ms=8, capsize=4,
            label="No-DAG Violation")
ax.fill_between(epsilons,
                result_df["nodag_violation_mean"] - result_df["nodag_violation_std"],
                result_df["nodag_violation_mean"] + result_df["nodag_violation_std"],
                color=COLOR_NODAG, alpha=ALPHA_FILL)
ax.axhline(y=0, color=COLOR_DAG, ls="--", alpha=0.7)
ax.text(0.12, 1.5, "DAG: always 0%", fontsize=9, color=COLOR_DAG, alpha=0.8)
ax.set_xscale("log")
ax.set_xlabel("Privacy Budget ε (log scale)", fontsize=12)
ax.set_ylabel("Constraint Violation Magnitude (%)", fontsize=12)
ax.set_title("Constraint Violation: No-DAG Method (data.json)", fontsize=14, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim(0.08, 12)
save_and_close(fig, "violation_v2.png")

# ========================================================================
# 图 4：综合仪表板
# ========================================================================
print("绘制图 4: 综合仪表板...")
fig, axes = plt.subplots(2, 3, figsize=(18, 11))

# MRE
ax = axes[0, 0]
ax.errorbar(epsilons, result_df["dag_mre_mean"], yerr=result_df["dag_mre_std"],
            marker=MARKER_DAG, ls="-", color=COLOR_DAG, lw=2, ms=6, capsize=3, label="DAG")
ax.errorbar(epsilons, result_df["nodag_mre_mean"], yerr=result_df["nodag_mre_std"],
            marker=MARKER_NODAG, ls="--", color=COLOR_NODAG, lw=2, ms=6, capsize=3, label="No-DAG")
ax.set_xscale("log"); ax.set_xlabel("ε"); ax.set_ylabel("MRE (%)")
ax.set_title("Mean Relative Error"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# MAE
ax = axes[0, 1]
ax.errorbar(epsilons, result_df["dag_mae_mean"], yerr=result_df["dag_mae_std"],
            marker=MARKER_DAG, ls="-", color=COLOR_DAG, lw=2, ms=6, capsize=3, label="DAG")
ax.errorbar(epsilons, result_df["nodag_mae_mean"], yerr=result_df["nodag_mae_std"],
            marker=MARKER_NODAG, ls="--", color=COLOR_NODAG, lw=2, ms=6, capsize=3, label="No-DAG")
ax.set_xscale("log"); ax.set_xlabel("ε"); ax.set_ylabel("MAE")
ax.set_title("Mean Absolute Error"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# RMSE
ax = axes[0, 2]
ax.errorbar(epsilons, result_df["dag_rmse_mean"], yerr=result_df["dag_rmse_std"],
            marker=MARKER_DAG, ls="-", color=COLOR_DAG, lw=2, ms=6, capsize=3, label="DAG")
ax.errorbar(epsilons, result_df["nodag_rmse_mean"], yerr=result_df["nodag_rmse_std"],
            marker=MARKER_NODAG, ls="--", color=COLOR_NODAG, lw=2, ms=6, capsize=3, label="No-DAG")
ax.set_xscale("log"); ax.set_xlabel("ε"); ax.set_ylabel("RMSE")
ax.set_title("Root Mean Square Error"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# Constraint
ax = axes[1, 0]
ax.errorbar(epsilons, result_df["dag_constraint_mean"], yerr=result_df["dag_constraint_std"],
            marker=MARKER_DAG, ls="-", color=COLOR_DAG, lw=2, ms=6, capsize=3, label="DAG")
ax.errorbar(epsilons, result_df["nodag_constraint_mean"], yerr=result_df["nodag_constraint_std"],
            marker=MARKER_NODAG, ls="--", color=COLOR_NODAG, lw=2, ms=6, capsize=3, label="No-DAG")
ax.set_xscale("log"); ax.axhline(y=100, color=COLOR_DAG, ls=":", alpha=0.5)
ax.set_xlabel("ε"); ax.set_ylabel("Rate (%)"); ax.set_ylim(-5, 110)
ax.set_title("Constraint Satisfaction"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# Violation
ax = axes[1, 1]
ax.errorbar(epsilons, result_df["nodag_violation_mean"], yerr=result_df["nodag_violation_std"],
            marker=MARKER_NODAG, ls="-", color=COLOR_NODAG, lw=2, ms=6, capsize=3)
ax.set_xscale("log"); ax.axhline(y=0, color=COLOR_DAG, ls="--", alpha=0.5)
ax.set_xlabel("ε"); ax.set_ylabel("Violation (%)")
ax.set_title("Constraint Violation (No-DAG only)"); ax.grid(True, alpha=0.3)

# Endpoint
ax = axes[1, 2]
ax.errorbar(epsilons, result_df["dag_endpoint_mean"], yerr=result_df["dag_endpoint_std"],
            marker=MARKER_DAG, ls="-", color=COLOR_DAG, lw=2, ms=6, capsize=3, label="DAG")
ax.errorbar(epsilons, result_df["nodag_endpoint_mean"], yerr=result_df["nodag_endpoint_std"],
            marker=MARKER_NODAG, ls="--", color=COLOR_NODAG, lw=2, ms=6, capsize=3, label="No-DAG")
ax.set_xscale("log"); ax.set_xlabel("ε"); ax.set_ylabel("Error (%)")
ax.set_title("Max Relative Error"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

fig.suptitle("DAG+mLDP vs No-DAG: Comprehensive Analysis (data.json, 200 real samples)",
             fontsize=16, fontweight="bold", y=1.01)
plt.tight_layout()
save_and_close(fig, "dashboard_v2.png")

# ========================================================================
# 打印结果
# ========================================================================
print(f"\n{'='*80}")
print("FINAL RESULTS (data.json)")
print(f"{'='*80}")
print(f"{'ε':<8} {'DAG MRE%':<12} {'NoDAG MRE%':<12} {'DAG Constr%':<14} {'NoDAG Constr%':<14} {'NoDAG Viol%':<14}")
print(f"{'-'*75}")
for _, row in result_df.iterrows():
    print(f"{row['epsilon']:<8.1f} "
          f"{row['dag_mre_mean']:<12.2f} "
          f"{row['nodag_mre_mean']:<12.2f} "
          f"{row['dag_constraint_mean']:<14.1f} "
          f"{row['nodag_constraint_mean']:<14.1f} "
          f"{row['nodag_violation_mean']:<14.2f}")

print(f"\n所有图像已保存到 experiment/results/")
print("  - mre_v2.png, constraint_v2.png, violation_v2.png, dashboard_v2.png")
