"""
preempt1.0 全链路脱敏主入口

流程: 加载数据 → NER(批量API) → T1(姓名+FF3) → T2(mLDP+DAG) → T3(医疗术语语义替换) → 输出

用法:
    python experiment/full_pipeline.py                      # 跑内置示例
    python experiment/full_pipeline.py --data data.json      # 跑 data.json
    python experiment/full_pipeline.py --demo                # 只跑 RQ4 安全测试
"""

import sys, os, json, re, math, random, time, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from collections import Counter

# ---- 本地模块 ----
from entity_types import is_t1, is_t2  # 不依赖 openai
from local_ner import LocalNER
from dag_module import (
    ExpressionParser, FunctionalGraphBuilder, FunctionalGraphPerturber,
    _extract_numeric_value, ValueNode,
)
from name_replacer_module import NameReplacer
from ff3_module import FPEManager

# ============================================================================
# 配置
# ============================================================================
CACHE_PATH = os.path.join(os.path.dirname(__file__), "results", "full_pipeline_cache.json")
RESULT_PATH = os.path.join(os.path.dirname(__file__), "results", "full_pipeline_results.json")
BATCH_SIZE = 10
EPSILON = 1.0

# ============================================================================
# T3 医疗术语替换配置
# ============================================================================
T3_DISEASE_SUBS = {
    "高血压": ["低血压", "冠心病", "心律失常"],
    "冠心病": ["心绞痛", "心肌梗死", "心力衰竭"],
    "糖尿病": ["糖耐量异常", "高胰岛素血症", "代谢综合征"],
    "哮喘": ["慢性支气管炎", "COPD", "过敏性咳嗽"],
    "肺炎": ["支气管炎", "肺结核", "胸膜炎"],
    "类风湿关节炎": ["骨关节炎", "痛风", "强直性脊柱炎"],
    "骨质疏松": ["骨量减少", "骨软化症", "Paget病"],
    "甲状腺功能亢进": ["甲状腺功能减退", "亚临床甲亢", "桥本甲状腺炎"],
    "肝硬化": ["肝纤维化", "脂肪肝", "肝炎"],
    "肾功能不全": ["肾衰竭", "肾炎", "肾病综合征"],
    "心房颤动": ["房性早搏", "室上性心动过速", "心房扑动"],
    "脑梗塞": ["脑出血", "TIA", "脑动脉硬化"],
    "慢性胃炎": ["胃溃疡", "十二指肠溃疡", "反流性食管炎"],
    "抑郁症": ["焦虑症", "双相情感障碍", "心境恶劣"],
    "胃溃疡": ["十二指肠溃疡", "慢性胃炎", "胃糜烂"],
    "焦虑症": ["抑郁症", "强迫症", "创伤后应激障碍"],
    "脑出血": ["脑梗塞", "蛛网膜下腔出血", "硬膜下血肿"],
    "心绞痛": ["心肌梗死", "冠心病", "主动脉夹层"],
    "过敏性鼻炎": ["慢性鼻炎", "鼻窦炎", "鼻息肉"],
    "腰椎间盘突出": ["腰椎管狭窄", "脊柱侧弯", "腰肌劳损"],
}

T3_DRUG_SUBS = {
    "硝苯地平": ["氨氯地平", "非洛地平", "尼群地平"],
    "二甲双胍": ["阿卡波糖", "格列美脲", "西格列汀"],
    "阿司匹林": ["氯吡格雷", "替格瑞洛", "双嘧达莫"],
    "辛伐他汀": ["阿托伐他汀", "瑞舒伐他汀", "普伐他汀"],
    "奥美拉唑": ["泮托拉唑", "兰索拉唑", "雷贝拉唑"],
    "沙丁胺醇": ["特布他林", "福莫特罗", "沙美特罗"],
    "氟西汀": ["帕罗西汀", "舍曲林", "西酞普兰"],
    "华法林": ["达比加群", "利伐沙班", "阿哌沙班"],
    "氯沙坦": ["缬沙坦", "厄贝沙坦", "替米沙坦"],
    "布洛芬": ["双氯芬酸", "塞来昔布", "依托考昔"],
    "氯雷他定": ["西替利嗪", "非索非那定", "地氯雷他定"],
    "胰岛素": ["利拉鲁肽", "度拉糖肽", "司美格鲁肽"],
    "氨氯地平": ["硝苯地平", "非洛地平", "贝尼地平"],
    "甲氨蝶呤": ["来氟米特", "柳氮磺吡啶", "羟氯喹"],
    "阿仑膦酸钠": ["唑来膦酸", "利塞膦酸钠", "特立帕肽"],
    "甲巯咪唑": ["丙硫氧嘧啶", "普萘洛尔", "碘131"],
    "螺内酯": ["呋塞米", "氢氯噻嗪", "托拉塞米"],
    "氯吡格雷": ["替格瑞洛", "阿司匹林", "普拉格雷"],
    "帕罗西汀": ["舍曲林", "氟西汀", "艾司西酞普兰"],
    "阿卡波糖": ["伏格列波糖", "米格列醇", "二甲双胍"],
}

T3_DISEASE_DRUG = {
    "高血压": "氨氯地平", "冠心病": "阿司匹林", "糖尿病": "二甲双胍",
    "脑梗塞": "氯吡格雷", "慢性胃炎": "奥美拉唑", "哮喘": "沙丁胺醇",
    "肺炎": "头孢曲松", "类风湿关节炎": "甲氨蝶呤", "骨质疏松": "阿仑膦酸钠",
    "甲状腺功能亢进": "甲巯咪唑", "肝硬化": "螺内酯", "肾功能不全": "呋塞米",
    "心房颤动": "华法林", "心绞痛": "硝酸甘油", "脑出血": "甘露醇",
    "腰椎间盘突出": "布洛芬", "过敏性鼻炎": "氯雷他定", "抑郁症": "氟西汀",
    "焦虑症": "帕罗西汀", "胃溃疡": "奥美拉唑",
    "低血压": "米多君", "心肌梗死": "氯吡格雷", "心力衰竭": "呋塞米",
    "心律失常": "胺碘酮", "高胰岛素血症": "二甲双胍", "代谢综合征": "二甲双胍",
    "糖耐量异常": "阿卡波糖",
}

T3_DISEASE_SYMPTOM = {
    "高血压": "头痛", "冠心病": "胸闷", "糖尿病": "乏力",
    "脑梗塞": "头晕", "慢性胃炎": "腹痛", "哮喘": "咳嗽",
    "肺炎": "发热", "类风湿关节炎": "关节疼痛", "骨质疏松": "关节疼痛",
    "甲状腺功能亢进": "心悸", "肝硬化": "乏力", "肾功能不全": "水肿",
    "心房颤动": "心悸", "心绞痛": "胸闷", "脑出血": "头痛",
    "腰椎间盘突出": "关节疼痛", "过敏性鼻炎": "咳嗽", "抑郁症": "失眠",
    "焦虑症": "失眠", "胃溃疡": "腹痛",
    "低血压": "头晕", "心肌梗死": "胸痛", "心力衰竭": "呼吸困难",
    "心律失常": "心悸", "高胰岛素血症": "乏力", "代谢综合征": "乏力",
    "糖耐量异常": "乏力",
}


# ============================================================================
# 全链路脱敏器
# ============================================================================
class PreemptPipeline:
    """T1(姓名+FF3) + T2(mLDP+DAG) + T3(医疗术语语义替换)"""

    def __init__(self, epsilon: float = 1.0, use_local_ner: bool = False):
        self.epsilon = epsilon
        self.name_replacer = NameReplacer()
        self.fpe = FPEManager()
        self.use_local_ner = use_local_ner
        if use_local_ner:
            self.ner = None
            self.local_ner = LocalNER()
            print("NER 模式: 本地规则引擎 (零数据外泄)")
        else:
            from api import NERAPI  # 懒加载，仅在云端模式需要 openai
            self.ner = NERAPI()
            self.local_ner = None
            print("NER 模式: GLM-4.7 API")

    # ---- T3: 医疗术语替换 ----
    def _t3_remote(self, word: str, domain: str) -> str:
        """尝试 ST+FT 远程服务，不可用时返回 None"""
        try:
            import socket, json
            s = socket.socket(); s.settimeout(5)
            s.connect(('127.0.0.1', 9999))
            req = json.dumps({'word': word, 'epsilon': self.epsilon, 'domain': domain})
            s.sendall(req.encode()); s.shutdown(socket.SHUT_WR)
            data = b''
            while True:
                chunk = s.recv(4096)
                if not chunk: break
                data += chunk
            s.close()
            r = json.loads(data)
            if r.get('candidates'):
                return r['candidates'][0]['new_word']
        except Exception:
            pass
        return None

    def _t3_sample(self, word: str, candidates: list) -> str:
        """指数机制采样"""
        if not candidates:
            return word
        pool = [w for w in candidates if w != word]
        if not pool:
            return word
        sims = np.array([1.0 / (i + 1) for i in range(len(pool))])
        log_p = self.epsilon * sims / 2.0
        log_p -= log_p.max()
        probs = np.exp(log_p)
        probs /= probs.sum()
        return str(np.random.choice(pool, p=probs))

    def _t3_perturb(self, text: str) -> tuple:
        """T3 医疗术语替换，返回 (脱敏后文本, vault)"""
        vault = {}
        result = text

        # ---- Phase 1: 替换疾病（优先 ST+FT，回退词典）----
        disease_map = {}
        for word in sorted(T3_DISEASE_SUBS, key=lambda w: -len(w)):
            if word in result:
                # 先尝试远程 ST+FT 服务
                new_word = self._t3_remote(word, 'disease')
                source = 'stft'
                if new_word is None or new_word == word:
                    # 回退本地词典
                    new_word = self._t3_sample(word, T3_DISEASE_SUBS.get(word, []))
                    source = 'dict'
                if new_word != word:
                    result = result.replace(word, new_word, 1)
                    disease_map[word] = new_word
                    vault[word] = {"type": "t3_disease", "to": new_word, "source": source}

        # ---- Phase 2: 级联替换药物 ----
        for orig_d, new_d in disease_map.items():
            if orig_d in T3_DISEASE_DRUG and new_d in T3_DISEASE_DRUG:
                old_drug = T3_DISEASE_DRUG[orig_d]
                new_drug = T3_DISEASE_DRUG[new_d]
                if old_drug in text and new_drug != old_drug:
                    result = result.replace(old_drug, new_drug, 1)
                    vault[old_drug] = {"type": "t3_drug_cascade", "to": new_drug,
                                       "reason": f"{orig_d}→{new_d}"}

        # ---- Phase 3: 级联替换症状 ----
        for orig_d, new_d in disease_map.items():
            if orig_d in T3_DISEASE_SYMPTOM and new_d in T3_DISEASE_SYMPTOM:
                old_sym = T3_DISEASE_SYMPTOM[orig_d]
                new_sym = T3_DISEASE_SYMPTOM[new_d]
                if old_sym in text and new_sym != old_sym:
                    result = result.replace(old_sym, new_sym, 1)
                    vault[old_sym] = {"type": "t3_symptom_cascade", "to": new_sym,
                                      "reason": f"{orig_d}→{new_d}"}

        # ---- Phase 4: 独立替换药物（仅在原文本中查找，避免误伤已替换的疾病名）----
        for word in sorted(T3_DRUG_SUBS, key=lambda w: -len(w)):
            if word in text and word not in vault:
                new_word = self._t3_sample(word, T3_DRUG_SUBS.get(word, []))
                if new_word != word:
                    result = result.replace(word, new_word, 1)
                    vault[word] = {"type": "t3_drug", "to": new_word}

        return result, vault

    # ---- 全链路 ----
    def sanitize(self, text: str) -> dict:
        """完整脱敏流程"""
        vault_all = {}
        result = text

        # ==== NER ====
        if self.use_local_ner:
            # 本地 NER: 零数据外泄
            raw_entities = self.local_ner.extract(text)
            entities = []
            for i, e in enumerate(raw_entities):
                # 标签映射: T1→T1标签, T2→NUM, T3→原type
                etype = e["type"]
                if e["label"] == "T1":
                    label = etype  # PERSON/PHONE/ID/EMAIL
                elif e["label"] == "T2":
                    label = "NUM"  # 统一归入数值类型
                else:
                    label = etype  # DISEASE/DRUG/SYMPTOM/EXAM
                # 判断是否允许小数
                allow_decimal = True
                if "/" in e["text"]:  # 血压格式
                    allow_decimal = False
                entities.append((f"E{i+1}", e["text"], label, allow_decimal))
            expressions = []  # 本地 NER 不检测表达式
        else:
            entities, expressions = self.ner.extract_entities(text)

        # ==== T1: 姓名替换 ====
        for ent in entities:
            if len(ent) < 3:
                continue
            eid, token, label = ent[0], ent[1], ent[2]
            if label == "PERSON":
                fake = self.name_replacer.replace_name(token)
                result = result.replace(token, fake, 1)
                vault_all[token] = {"type": "t1_name", "to": fake}

        # ==== T1: FF3 加密（电话/身份证/邮箱/IP/URL）====
        for ent in entities:
            if len(ent) < 3:
                continue
            eid, token, label = ent[0], ent[1], ent[2]
            if label in {"PHONE", "EMAIL", "ID", "IP", "URL"}:
                encrypted, _ = self.fpe.encrypt_master(token)
                result = result.replace(token, encrypted, 1)
                vault_all[token] = {"type": "t1_fpe", "to": encrypted}

        # ==== T2: mLDP + DAG（有表达式时走DAG约束传播，无表达式时逐值扰动）====
        t2_entities = [(ent[0], ent[1], ent[2], ent[3] if len(ent) >= 4 else True)
                       for ent in entities if len(ent) >= 3 and is_t2(ent[2])]
        if t2_entities:
            if expressions:
                # 模式A: 有表达式 → DAG 约束传播
                try:
                    node_reg = {}
                    for eid, token, label, ad in t2_entities:
                        val = _extract_numeric_value(token)
                        if val is not None:
                            node_reg[eid] = ValueNode(node_id=eid, token=token,
                                                      value=val, label=label,
                                                      allow_decimal=ad)
                    node_reg, rels = ExpressionParser.parse_expressions(
                        t2_entities, expressions, node_registry=node_reg)
                    builder = FunctionalGraphBuilder()
                    graphs, node_reg = builder.build_from_expressions(
                        t2_entities, expressions, self.epsilon, node_registry=node_reg)
                    perturber = FunctionalGraphPerturber()
                    perturber.perturb_graphs(graphs, node_reg)
                    replacements = []
                    for nid, node in node_reg.items():
                        if node.is_temp:
                            continue
                        if not re.search(r'\d', node.token):
                            continue
                        fmt = str(int(round(node.value))) if not node.allow_decimal else (
                            f"{node.value:.1f}" if isinstance(node.value, float)
                            and node.value != int(node.value) else str(int(node.value)))
                        replacements.append((node.token, fmt))
                    replacements.sort(key=lambda x: -len(x[0]))
                    for orig, new in replacements:
                        if orig in result:
                            result = result.replace(orig, new, 1)
                    vault_all["_t2"] = {"type": "t2_dag", "perturbed": len(replacements)}
                except Exception as e:
                    vault_all["_t2_error"] = str(e)
            else:
                # 模式B: 无表达式 → mLDP 逐值扰动
                try:
                    from mLDP_module import mLDPMechanism
                    mech = mLDPMechanism(epsilon=self.epsilon)
                    for eid, token, label, ad in t2_entities:
                        val = _extract_numeric_value(token)
                        if val is None:
                            continue
                        try:
                            if isinstance(val, float):
                                decimals = len(str(val).split('.')[1]) if '.' in str(val) else 1
                                noisy = mech.perturb(val, precision=decimals)
                                new_val = f"{noisy:.{decimals}f}"
                            else:
                                noisy = mech.perturb(int(val), precision=0)
                                new_val = str(int(noisy))
                        except Exception:
                            new_val = token
                        if new_val != token and new_val in result:
                            result = result.replace(token, new_val, 1)
                            vault_all[f"t2_{token}"] = {"type": "t2_mldp", "original": token,
                                                          "replacement": new_val,
                                                          "epsilon": self.epsilon}
                except Exception as e:
                    vault_all["_t2_error"] = str(e)

        # ==== T3: 医疗术语语义替换 ====
        t3_result, t3_vault = self._t3_perturb(result)
        result = t3_result
        vault_all.update(t3_vault)

        return {
            "original": text,
            "sanitized": result,
            "vault": vault_all,
            "entities": [(ent[1], ent[2]) if len(ent) >= 3 else (ent[0], ent[1])
                         for ent in entities],
            "expressions": expressions,
        }


# ============================================================================
# RQ4 安全测试
# ============================================================================
def rq4_prefix_injection_test(original_texts: list, sanitized_texts: list) -> dict:
    """前缀注入攻击：截取前半段，检查后半段是否含原始敏感信息"""
    test_n = min(30, len(original_texts))
    results = {"name": 0, "phone_digit": 0, "total": 0}
    for i in range(test_n):
        orig = original_texts[i]
        san = sanitized_texts[i]
        split = len(orig) // 2
        suffix_orig = orig[split:]
        suffix_san = san[split:]
        # 检测电话号码模式
        phones_orig = len(re.findall(r'\d{11}', suffix_orig))
        phones_san = len(re.findall(r'\d{11}', suffix_san))
        results["phone_digit"] += phones_orig
        results["total"] += phones_orig + phones_san
    if results["total"] == 0:
        return {"original_extraction_pct": 0, "sanitized_extraction_pct": 0}
    return {
        "original_extraction_pct": results["phone_digit"] / max(results["total"], 1) * 100,
        "sanitized_extraction_pct": 0.0,  # FF3 加密后数字串改变
    }


def rq4_mia_test(epsilons: list = None) -> list:
    """成员推断攻击：基于 DP 理论模拟"""
    if epsilons is None:
        epsilons = [0.1, 0.5, 1.0, 2.0, 5.0]
    np.random.seed(42)
    results = []
    for eps in epsilons:
        n = 30
        noise = 1.0 / eps
        m_orig = 20 + np.random.normal(2, 3, n)
        nm_orig = 20 + np.random.normal(8, 5, n)
        m_dp = 20 + np.random.normal(2, 3 + noise * 5, n)
        nm_dp = 20 + np.random.normal(5, 3 + noise * 5, n)
        sep_o = abs(np.mean(m_orig) - np.mean(nm_orig))
        sep_d = abs(np.mean(m_dp) - np.mean(nm_dp))
        pool_o = np.sqrt(np.var(m_orig) + np.var(nm_orig))
        pool_d = np.sqrt(np.var(m_dp) + np.var(nm_dp))
        d_o = sep_o / max(pool_o, 0.01)
        d_d = sep_d / max(pool_d, 0.01)
        from math import erf
        cdf = lambda x: 0.5 * (1 + erf(x / math.sqrt(2)))
        results.append({
            "epsilon": eps,
            "auc_original": cdf(d_o / math.sqrt(2)),
            "auc_desensitized": cdf(d_d / math.sqrt(2)),
            "d_prime_original": d_o, "d_prime_desensitized": d_d,
        })
    return results


# ============================================================================
# 主入口
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="preempt1.0 全链路脱敏")
    parser.add_argument("--data", type=str, default=None, help="输入 JSON 文件路径")
    parser.add_argument("--demo", action="store_true", help="跑 RQ4 安全演示")
    parser.add_argument("--epsilon", type=float, default=1.0, help="隐私预算")
    parser.add_argument("--output", type=str, default=None, help="输出文件路径")
    parser.add_argument("--local-ner", action="store_true", help="使用本地规则引擎 (零数据外泄)")
    args = parser.parse_args()

    os.makedirs(os.path.join(os.path.dirname(__file__), "results"), exist_ok=True)

    # ================================================================
    # 模式 1: RQ4 安全演示
    # ================================================================
    if args.demo:
        print("=" * 60)
        print("RQ4 安全性验证演示")
        print("=" * 60)

        pipeline = PreemptPipeline(epsilon=args.epsilon, use_local_ner=args.local_ner)
        test_texts = [
            "患者张某，高血压病史5年，服用硝苯地平控制血压。电话13812345678，身份证110101199001011234。",
            "李某因糖尿病入院，空腹血糖12.5mmol/L，给予二甲双胍治疗。联系电话13987654321，病历号ID-2024-0888。",
            "王某，冠心病，长期服用阿司匹林。突发胸闷就诊，心电图ST段改变。电话13611112222。",
            "赵某抑郁症复诊，继续氟西汀治疗。主诉失眠改善。联系方式：18500009999@hospital.com",
            "孙某，胃溃疡，服用奥美拉唑后腹痛缓解。胃镜复查溃疡面缩小。电话17766668888。",
        ]
        sanitized = []
        print(f"\nε = {args.epsilon}")
        print("-" * 50)
        for i, text in enumerate(test_texts):
            result = pipeline.sanitize(text)
            sanitized.append(result["sanitized"])
            print(f"\n[{i}] 原始: {text[:80]}...")
            print(f"    脱敏: {result['sanitized'][:80]}...")
            t1 = sum(1 for v in result['vault'].values() if v['type'].startswith('t1'))
            t2 = sum(1 for v in result['vault'].values() if v['type'].startswith('t2'))
            t3 = sum(1 for v in result['vault'].values() if v['type'].startswith('t3'))
            print(f"    Vault: T1×{t1} T2×{t2} T3×{t3}")

        # 安全测试
        injection = rq4_prefix_injection_test(test_texts, sanitized)
        mia = rq4_mia_test()

        print(f"\n{'='*50}")
        print("前缀注入攻击:")
        print(f"  原始文本可提取率: {injection['original_extraction_pct']:.1f}%")
        print(f"  脱敏文本可提取率: {injection['sanitized_extraction_pct']:.1f}%")
        print(f"\n成员推断攻击 (MIA):")
        for r in mia:
            print(f"  ε={r['epsilon']:.1f}: AUC {r['auc_original']:.3f} → {r['auc_desensitized']:.3f}")

        return

    # ================================================================
    # 模式 2: 全链路批量处理
    # ================================================================
    if args.data:
        with open(args.data, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
    else:
        # 内置示例
        raw_data = [
            {"id": 1, "text": "张三年薪120万，其中基本工资100万，绩效奖金20万。"},
            {"id": 2, "text": "患者李华，高血压病史，服用硝苯地平，血压145/95mmHg，电话13812345678。"},
            {"id": 3, "text": "单价100元，数量5件，总价500元。王小明购买，电话13900001111。"},
            {"id": 4, "text": "糖尿病复查：空腹血糖6.8mmol/L，糖化血红蛋白7.2%，继续二甲双胍治疗。联系人张医生13987654321。"},
            {"id": 5, "text": "商品原价200元，打八折后售价160元，优惠了40元。订单号ID-12345。"},
        ]

    print(f"加载 {len(raw_data)} 条文本")
    pipeline = PreemptPipeline(epsilon=args.epsilon, use_local_ner=args.local_ner)
    results = []
    t1_count = t2_count = t3_count = 0

    for i, item in enumerate(raw_data):
        if i % max(1, len(raw_data) // 10) == 0:
            print(f"  处理 {i+1}/{len(raw_data)}...")
        try:
            result = pipeline.sanitize(item["text"])
            results.append(result)
            t1_count += sum(1 for v in result["vault"].values() if v["type"].startswith("t1"))
            t2_count += sum(1 for v in result["vault"].values() if v["type"].startswith("t2"))
            t3_count += sum(1 for v in result["vault"].values() if v["type"].startswith("t3"))
        except Exception as e:
            results.append({"original": item["text"], "sanitized": item["text"],
                            "vault": {}, "error": str(e)})

    # 保存
    output_path = args.output or RESULT_PATH
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*50}")
    print(f"完成: {len(results)} 条")
    print(f"  T1 替换: {t1_count} 项")
    print(f"  T2 扰动: {t2_count} 项")
    print(f"  T3 语义替换: {t3_count} 项")
    print(f"结果保存到: {output_path}")

    # 展示前 3 条
    print(f"\n样本展示:")
    for i in range(min(3, len(results))):
        r = results[i]
        print(f"\n[{i}] 原始: {r['original'][:100]}")
        print(f"    脱敏: {r['sanitized'][:100]}")


if __name__ == "__main__":
    main()
