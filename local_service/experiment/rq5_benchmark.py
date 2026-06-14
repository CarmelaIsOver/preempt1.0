"""
RQ5: 性能基准测试 — 全链路处理时延 + 吞吐量 + 消融对比
"""

import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from collections import defaultdict

from full_pipeline import PreemptPipeline


def benchmark_single(pipeline, texts, n_warmup=3, n_test=20, skip_ner=False):
    """单条处理时延。skip_ner=True 时仅测本地脱敏（T1+T2+T3）"""
    times = []
    for i, text in enumerate(texts[:n_warmup + n_test]):
        t0 = time.time()
        if skip_ner:
            # 仅本地脱敏，不含 NER API
            entities, _ = pipeline.ner.extract_entities(text)
            t_ner = time.time() - t0  # NER 时间单独记录
            t0_local = time.time()
            # 手动跑 T1+T2+T3...
            result = text
            vault = {}
            for ent in entities:
                if len(ent) < 3: continue
                token, label = ent[1], ent[2]
                if label == "PERSON":
                    fake = pipeline.name_replacer.replace_name(token)
                    result = result.replace(token, fake, 1)
                    vault[token] = {"type": "t1_name", "to": fake}
                elif label in {"PHONE", "EMAIL", "ID", "IP", "URL"}:
                    enc, _ = pipeline.fpe.encrypt_master(token)
                    result = result.replace(token, enc, 1)
                    vault[token] = {"type": "t1_fpe", "to": enc}
            t1_time = (time.time() - t0_local) * 1000

            t0_local = time.time()
            t2_entities = [(ent[0], ent[1], ent[2], ent[3] if len(ent) >= 4 else True)
                           for ent in entities if len(ent) >= 3 and ent[2] in {
                "AGE","SALARY","AMOUNT","COUNT","WEIGHT","BLOOD_SUGAR","BLOOD_PRESSURE",
                "MEDICAL_VAL","YEAR","DATE_NUM","NUM","PERCENT"}]
            if t2_entities:
                from dag_module import ExpressionParser, FunctionalGraphBuilder, FunctionalGraphPerturber, _extract_numeric_value, ValueNode
                try:
                    node_reg = {}
                    for eid, token, label, ad in t2_entities:
                        val = _extract_numeric_value(token)
                        if val is not None:
                            node_reg[eid] = ValueNode(node_id=eid, token=token, value=val, label=label, allow_decimal=ad)
                    node_reg, rels = ExpressionParser.parse_expressions(t2_entities, [], node_registry=node_reg)
                    builder = FunctionalGraphBuilder()
                    graphs, node_reg = builder.build_from_expressions(t2_entities, [], 1.0, node_registry=node_reg)
                    perturber = FunctionalGraphPerturber()
                    perturber.perturb_graphs(graphs, node_reg)
                except: pass
            t2_time = (time.time() - t0_local) * 1000

            t0_local = time.time()
            pipeline._t3_perturb(result)
            t3_time = (time.time() - t0_local) * 1000
            elapsed = (t1_time + t2_time + t3_time) if i >= n_warmup else 0
        else:
            pipeline.sanitize(text)
            elapsed = (time.time() - t0) * 1000

        if i >= n_warmup:
            times.append(elapsed)

    if not times:
        return {"mean_ms": 0, "median_ms": 0, "p95_ms": 0}
    return {
        "mean_ms": np.mean(times),
        "median_ms": np.median(times),
        "p95_ms": np.percentile(times, 95),
        "min_ms": np.min(times),
        "max_ms": np.max(times),
    }


def benchmark_stages(pipeline, texts, n=10):
    """分阶段计时"""
    # 简化为直接测 sanitize 内部三个阶段
    stage_times = {"t1": [], "t2": [], "t3": []}
    # 用内部方法分别测
    for text in texts[:n]:
        # T1 only
        t0 = time.time()
        entities, _ = pipeline.ner.extract_entities(text)
        orig = text
        for ent in entities:
            if len(ent) < 3: continue
            token, label = ent[1], ent[2]
            if label == "PERSON":
                orig = orig.replace(token, pipeline.name_replacer.replace_name(token), 1)
            elif label in {"PHONE", "EMAIL", "ID", "IP", "URL"}:
                enc, _ = pipeline.fpe.encrypt_master(token)
                orig = orig.replace(token, enc, 1)
        stage_times["t1"].append((time.time() - t0) * 1000)

        # T2
        t0 = time.time()
        t2_entities = [(ent[0], ent[1], ent[2], ent[3] if len(ent) >= 4 else True)
                       for ent in entities if len(ent) >= 3 and ent[2] in {
            "AGE","SALARY","AMOUNT","COUNT","WEIGHT","BLOOD_SUGAR","BLOOD_PRESSURE",
            "MEDICAL_VAL","YEAR","DATE_NUM","NUM","PERCENT"}]
        expressions = []
        if t2_entities:
            from dag_module import ExpressionParser, FunctionalGraphBuilder, FunctionalGraphPerturber, _extract_numeric_value, ValueNode
            try:
                node_reg = {}
                for eid, token, label, ad in t2_entities:
                    val = _extract_numeric_value(token)
                    if val is not None:
                        node_reg[eid] = ValueNode(node_id=eid, token=token, value=val,
                                                  label=label, allow_decimal=ad)
                node_reg, rels = ExpressionParser.parse_expressions(t2_entities, [], node_registry=node_reg)
                builder = FunctionalGraphBuilder()
                graphs, node_reg = builder.build_from_expressions(t2_entities, [], 1.0, node_registry=node_reg)
                perturber = FunctionalGraphPerturber()
                perturber.perturb_graphs(graphs, node_reg)
            except: pass
        stage_times["t2"].append((time.time() - t0) * 1000)

        # T3
        t0 = time.time()
        pipeline._t3_perturb(orig)
        stage_times["t3"].append((time.time() - t0) * 1000)

    return {k: {"mean_ms": np.mean(v), "median_ms": np.median(v)} for k, v in stage_times.items()}


def benchmark_throughput(pipeline, texts, batch_sizes=[1, 5, 10]):
    """批处理吞吐量"""
    results = {}
    for bs in batch_sizes:
        times = []
        for i in range(0, min(len(texts), bs * 10), bs):
            batch = texts[i:i + bs]
            t0 = time.time()
            for t in batch:
                pipeline.sanitize(t)
            times.append((time.time() - t0) / len(batch) * 1000)  # ms per sample
        # 前2批warmup
        valid = times[2:] if len(times) > 2 else times
        results[f"batch_{bs}"] = {
            "ms_per_sample": np.mean(valid),
            "samples_per_sec": 1000 / np.mean(valid),
        }
    return results


def main():
    # 加载数据
    data_path = os.path.join(os.path.dirname(__file__), "data.json")
    with open(data_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    texts = [item["text"] for item in raw_data]

    # 按长度分组
    short = [t for t in texts if len(t) <= 60][:30]
    medium = [t for t in texts if 60 < len(t) <= 150][:30]
    long = [t for t in texts if len(t) > 150][:30]

    print("=" * 60)
    print("RQ5: 性能基准测试")
    print("=" * 60)

    pipeline = PreemptPipeline(epsilon=1.0)

    # ---- 单条时延 ----
    print("\n### 处理时延（单条）")
    for name, group in [("短文本(≤60字)", short), ("中文本(60-150字)", medium), ("长文本(>150字)", long)]:
        if not group:
            continue
        r = benchmark_single(pipeline, group)
        print(f"\n  {name} ({len(group)}条, 均长{np.mean([len(t) for t in group]):.0f}字):")
        print(f"    平均: {r['mean_ms']:.0f}ms  中位: {r['median_ms']:.0f}ms  "
              f"P95: {r['p95_ms']:.0f}ms  范围: {r['min_ms']:.0f}-{r['max_ms']:.0f}ms")

    # ---- 分阶段 ----
    print("\n### 阶段分解 (中文本 10条)")
    stages = benchmark_stages(pipeline, medium)
    total = sum(s["mean_ms"] for s in stages.values())
    for stage, d in stages.items():
        pct = d["mean_ms"] / total * 100 if total > 0 else 0
        print(f"  {stage}: {d['mean_ms']:.0f}ms ({pct:.0f}%)")

    # ---- 消融对比 ----
    print("\n### 消融对比 (中文本, 按脱敏层级)")
    # None = 只NER不做脱敏
    t0 = time.time()
    for t in medium[:5]:
        pipeline.ner.extract_entities(t)
    ner_only = (time.time() - t0) / 5 * 1000
    print(f"  NER only:       {ner_only:.0f}ms")
    print(f"  +T1:            ~{ner_only + stages['t1']['mean_ms']:.0f}ms")
    print(f"  +T1+T2:         ~{ner_only + stages['t1']['mean_ms'] + stages['t2']['mean_ms']:.0f}ms")
    print(f"  +T1+T2+T3(Full):~{ner_only + total:.0f}ms")

    # ---- 吞吐量 ----
    print("\n### 吞吐量")
    tp = benchmark_throughput(pipeline, medium)
    for k, v in tp.items():
        print(f"  {k}: {v['ms_per_sample']:.0f}ms/条, {v['samples_per_sec']:.2f}条/秒")

    # 保存
    result = {
        "config": {"pipeline": "preempt1.0 Full", "epsilon": 1.0, "hardware": "CPU (WSL2)"},
        "latency": {name: benchmark_single(pipeline, group) for name, group in
                    [("short", short), ("medium", medium), ("long", long)]},
        "stages": stages,
        "throughput": tp,
    }
    out_path = os.path.join(os.path.dirname(__file__), "results", "rq5_benchmark.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n保存: {out_path}")


if __name__ == "__main__":
    main()
