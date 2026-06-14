"""
RQ2 T3 隐私-效用权衡补充：类比推理关系保持率 vs ε
"""

import json, math, sys, os
import numpy as np
from statistics import mean
from sentence_transformers import SentenceTransformer

OUT_DIR = os.path.join(os.path.dirname(__file__), "results")

# 疾病-药物-症状 三元组（医学上已知的正确关系）
TRIPLES = [
    ("高血压", "硝苯地平", "头痛", "氨氯地平"),
    ("糖尿病", "二甲双胍", "乏力", "阿卡波糖"),
    ("冠心病", "阿司匹林", "胸闷", "氯吡格雷"),
    ("哮喘", "沙丁胺醇", "咳嗽", "特布他林"),
    ("抑郁症", "氟西汀", "失眠", "帕罗西汀"),
    ("胃溃疡", "奥美拉唑", "腹痛", "泮托拉唑"),
    ("甲亢", "甲巯咪唑", "心悸", "丙硫氧嘧啶"),
    ("骨质疏松", "阿仑膦酸钠", "关节疼痛", "唑来膦酸"),
    ("心房颤动", "华法林", "心悸", "达比加群"),
    ("过敏性鼻炎", "氯雷他定", "咳嗽", "西替利嗪"),
]

# ============================================================================
# ST 类比推理：疾病→新疾病，药物跟随
# ============================================================================
def st_analogy(st, disease_orig, disease_pert, drug_orig, drug_pool):
    """ST 类比推理：找到最合适的药物"""
    v_do = st.encode(disease_orig)
    v_dp = st.encode(disease_pert)
    v_drug = st.encode(drug_orig)
    delta = v_do - v_dp
    v_result = v_drug - delta

    pool = [d for d in drug_pool if d != drug_orig]
    best = max(pool, key=lambda d: float(
        np.dot(st.encode(d), v_result) /
        (np.linalg.norm(st.encode(d)) * np.linalg.norm(v_result) + 1e-8)
    ))
    return best


# ============================================================================
# 指数机制下的效用评估
# ============================================================================
def evaluate_utility(st, epsilon, n_trials=100):
    """评估 T3 在给定 ε 下的三重效用指标"""
    # 收集所有药物作为候选池
    all_drugs = set()
    for _, drug, _, _ in TRIPLES:
        all_drugs.add(drug)
    all_drugs = list(all_drugs)

    rel_preserved = 0  # 关系保持率
    sims = []           # ST 相似度
    total = 0

    for disease_orig, drug_orig, symptom_orig, expected_drug in TRIPLES:
        # 随机扰动疾病（模拟指数机制：ε越大越倾向于近邻，ε越小越随机）
        diseases = ["低血压","冠心病","心力衰竭","心律失常","糖尿病",
                    "高胰岛素血症","脑梗塞","胃炎","肝炎","肾炎",
                    "COPD","支气管炎","焦虑症","心境恶劣","慢性胃炎"]
        v_do = st.encode(disease_orig)
        disease_sims = []
        for d in diseases:
            v_d = st.encode(d)
            sim = float(np.dot(v_d, v_do) / (np.linalg.norm(v_d) * np.linalg.norm(v_do) + 1e-8))
            disease_sims.append((d, sim))

        # 指数机制采样疾病
        sensitivity = 0.5
        for _ in range(n_trials):
            arr = np.array([s for _, s in disease_sims])
            log_p = epsilon * arr / (2 * sensitivity)
            log_p -= log_p.max()
            probs = np.exp(log_p)
            probs /= probs.sum()
            idx = np.random.choice(len(diseases), p=probs)
            disease_pert = diseases[idx]

            # ST 类比推理找对应药物
            predicted_drug = st_analogy(st, disease_orig, disease_pert,
                                        drug_orig, all_drugs)

            # 评估
            if predicted_drug == expected_drug:
                rel_preserved += 1
            v_pred = st.encode(predicted_drug)
            v_exp = st.encode(expected_drug)
            sim = float(np.dot(v_pred, v_exp) / (np.linalg.norm(v_pred) * np.linalg.norm(v_exp) + 1e-8))
            sims.append(sim)
            total += 1

    return {
        "relationship_preservation": round(rel_preserved / total * 100, 1),
        "avg_similarity": round(mean(sims), 4),
    }


def main():
    print("加载 ST...")
    st = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')

    epsilons = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    print(f"\n{'='*60}")
    print("RQ2 T3 隐私-效用权衡：类比推理关系保持率")
    print(f"{'='*60}")
    print(f"{'ε':<8} {'关系保持率':<12} {'ST相似度':<10}")
    print("-" * 35)

    results = []
    for eps in epsilons:
        r = evaluate_utility(st, eps, n_trials=50)
        results.append({"epsilon": eps, **r})
        marker = " ←" if eps == 1.0 else ""
        print(f"{eps:<8.1f} {r['relationship_preservation']:<12.1f}% {r['avg_similarity']:<10.4f}{marker}")

    # 保存
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "rq2_t3_tradeoff.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n保存: {out_path}")

    # 生成 Markdown 表格
    print(f"\n### 表 6.5b T3 隐私-效用权衡（类比推理关系保持）")
    print(f"| ε | 关系保持率 | ST 相似度 |")
    print(f"|----|----------|----------|")
    for r in results:
        print(f"| {r['epsilon']} | {r['relationship_preservation']}% | {r['avg_similarity']} |")


if __name__ == "__main__":
    main()
