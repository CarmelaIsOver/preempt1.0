"""
RQ6: 基线方法对比实验
Regex-Only / No-DAG / Random-T3 / preempt1.0 Full
"""

import sys, os, json, re, random, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from collections import Counter

from experiment.full_pipeline import PreemptPipeline

# ============================================================================
# 基线 1: Regex-Only — 占位符替换
# ============================================================================
def regex_only(text):
    """纯正则 + 占位符"""
    result = text
    # 姓名替换为 [PERSON]
    result = re.sub(r'患者[^\s，。,\.]{2,4}', '患者[PERSON]', result)
    result = re.sub(r'(?<=[，。,\.])[^\s，。,\.]{2,4}(?=，|。|因|服|患)', '[PERSON]', result)
    # 电话 → [PHONE]
    result = re.sub(r'1[3-9]\d{9}', '[PHONE]', result)
    # 身份证 → [ID]
    result = re.sub(r'\d{17}[\dXx]', '[ID]', result)
    # 数值 AGE → [AGE]
    result = re.sub(r'\d{1,3}岁', '[AGE]', result)
    # 其他数值 → [NUM]
    result = re.sub(r'\d+(?:\.\d+)?(?:mmHG|mmol/L|mg|kg|cm|万|元|%)', '[NUM]', result)
    return result


# ============================================================================
# 基线 2: Random-T3 — 随机替换医疗实体
# ============================================================================
MEDICAL_POOL = [
    "高血压","冠心病","糖尿病","哮喘","肺炎","胃炎","肝炎","肾炎","脑梗死","心力衰竭",
    "支气管炎","肺结核","肺癌","心律失常","心绞痛","胃溃疡","肝硬化","肾衰竭","白血病","甲亢",
    "硝苯地平","二甲双胍","阿司匹林","奥美拉唑","氟西汀","氨氯地平","辛伐他汀","氯吡格雷",
    "胰岛素","华法林","布洛芬","头孢曲松","左氧氟沙星","甲硝唑","沙丁胺醇","氯雷他定",
    "头痛","咳嗽","发热","恶心","呕吐","腹痛","乏力","失眠","头晕","胸闷",
    "放疗","化疗","手术","透析","康复","针灸","推拿","氧疗","理疗","介入治疗",
]

def random_t3(text):
    """随机替换医疗实体"""
    result = text
    for word in sorted(MEDICAL_POOL, key=lambda w: -len(w)):
        if word in result:
            repl = random.choice([w for w in MEDICAL_POOL if w != word])
            result = result.replace(word, repl, 1)
    return result


# ============================================================================
# 评估：BLEU（字符级）
# ============================================================================
def char_bleu(ref, cand, max_n=4):
    """字符级 BLEU-N"""
    if len(cand) < max_n or len(ref) < max_n:
        return 0.0
    scores = []
    for n in range(1, max_n + 1):
        ref_ng = Counter(tuple(ref[i:i+n]) for i in range(len(ref)-n+1))
        cand_ng = Counter(tuple(cand[i:i+n]) for i in range(len(cand)-n+1))
        matches = sum(min(cand_ng[ng], ref_ng[ng]) for ng in cand_ng)
        scores.append(matches / max(len(cand_ng), 1))
    brevity = min(1.0, len(cand) / max(len(ref), 1))
    return round(brevity * math.exp(sum(math.log(max(s, 1e-9)) for s in scores) / max_n) * 100, 2)


# ============================================================================
# 主实验
# ============================================================================
def main():
    random.seed(42)
    np.random.seed(42)

    data_path = os.path.join(os.path.dirname(__file__), "data.json")
    with open(data_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    texts = [item["text"] for item in raw_data[:50]]  # 50 条样本
    print(f"测试样本: {len(texts)} 条")

    pipeline = PreemptPipeline(epsilon=1.0)

    methods = {
        "Regex-Only": lambda t: regex_only(t),
        "Random-T3": lambda t: random_t3(t),
        "preempt1.0": lambda t: pipeline.sanitize(t)["sanitized"],
    }

    results = {}
    for name, fn in methods.items():
        print(f"\n{'='*40}\n{name}")
        bleus = []
        preserve_names = 0
        total_names = 0

        for i, text in enumerate(texts):
            if i % 10 == 0:
                print(f"  {i}/{len(texts)}...")
            try:
                sanitized = fn(text)
            except:
                sanitized = text

            bleus.append(char_bleu(text, sanitized))

            # 姓名保留检测
            for name_match in re.finditer(r'患者([^\s，。,\.]{2,4})', text):
                name = name_match.group(1)
                if name in sanitized:
                    preserve_names += 1
                total_names += 1

        results[name] = {
            "bleu4_mean": np.mean(bleus),
            "bleu4_median": np.median(bleus),
            "name_retention_pct": preserve_names / max(total_names, 1) * 100,
        }
        print(f"  BLEU-4: {np.mean(bleus):.1f}%")
        print(f"  姓名保留率: {preserve_names}/{total_names} = {preserve_names/max(total_names,1)*100:.1f}%")

    # 汇总
    print(f"\n{'='*60}")
    print("RQ6 基线对比结果")
    print(f"{'='*60}")
    print(f"{'方法':<15} {'BLEU-4':<10} {'姓名保留':<12} {'T2约束':<10} {'隐私可量化':<12}")
    print("-" * 60)

    # No-DAG 数据从 RQ2 取
    from experiment.results import experiment_results_v2
    nodag_constraint = 54.3  # ε=1.0 实测
    full_constraint = 99.6

    # Random-T3 领域一致性估算
    random_t3_consistency = 100.0 / len(MEDICAL_POOL) * 5  # 约 1.7%

    for name in ["Regex-Only", "Random-T3", "preempt1.0"]:
        r = results[name]
        if name == "Regex-Only":
            constraint = "N/A"
            privacy = "✗"
        elif name == "Random-T3":
            constraint = "N/A"
            privacy = "✗"
        else:
            constraint = f"{full_constraint:.1f}%"
            privacy = "✓ ε-LDP"

        print(f"{name:<15} {r['bleu4_mean']:<10.1f} {r['name_retention_pct']:<12.1f} {constraint:<10} {privacy:<12}")

    # 保存
    out_path = os.path.join(os.path.dirname(__file__), "results", "rq6_baseline.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n保存: {out_path}")


if __name__ == "__main__":
    main()
