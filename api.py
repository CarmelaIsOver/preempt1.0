"""
api.py

单次 NER 调用完成两项任务：
  1. NER：细粒度实体识别（t1/t2 分类）
  2. DAG 边：t2 数值依赖关系（支持多父节点聚合边）

使用通义千问官方 API (DashScope)
"""

import json
import os
import re
from typing import List, Tuple, Dict, Optional

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

# 千问系统提示词
QWEN_SYSTEM_PROMPT = """你是实体识别、数值关系分析专家。对用户输入，只输出JSON，禁止输出<think>标签或任何思考过程以及任何文字描述。verify字段必须写出完整验证算式，结果为False的边必须丢弃。

【任务一】提取实体
t1类 ：PERSON/ORG/LOCATION/PHONE/EMAIL/ID/MEDICAL_ID/IP/URL/ENTITY
t2类 （数值）：AGE/SALARY/AMOUNT/COUNT/WEIGHT/BLOOD_SUGAR/BLOOD_PRESSURE/MEDICAL_VAL/YEAR/DATE_NUM/NUM/PERCENT
t3类 （医疗）：MEDICINE（药物）/DISEASE（疾病）/SYMPTOM（症状）/THERAPY（疗法）/FORBIDDEN_FOOD（忌口食物）/SIDE_EFFECT（副作用）/GENE（基因）/ORGAN（器官）/FUNCTION（功能）
注意：t3类实体只能使用上述9个标签，不得使用其他标签名。
【任务二】识别数值关系

Step1 语义判断（必须最先执行，结果写入mode字段）：
- 含 平均/均值/average → mode=average
- 含 总计/合计/总共/sum/total 
    检查已提取t2类token，若存在A×B=C（C÷A=B且B在已提取token中）→ mode=simple(强制执行判断)
    否则 → mode=sum
- 含 增长/上涨/下降/减少/grew/decreased → mode=growth
- 含 占比/占/proportion → mode=proportion
- 含 加权 → mode=weighted_avg
- 否则 → mode=simple

Step2a mode=simple时：
只对文本中有明确数值关系描述的token对建边，禁止穷举所有组合。
公式：
- multiply: child ÷ parent = param
- add:      child - parent = param
- percent:  child ÷ parent × 100 = param
- copy:     child = parent，param=null

Step2b mode为复合运算时：
注意：每一步的child必须是_tmp节点，原始token只能作为parent或param出现。
严格按以下伪代码生成edge_checks，条数由步骤数决定不多不少。

mode=average，输入[x1,...,xn]，结果D：
_tmp1=x1+x2 (add,param=x2,parent=x1,child=_tmp1)
_tmp(k)=_tmp(k-1)+x(k+2) (add,param=x(k+2),child=_tmp(k)) ← k=1到n-2
D=_tmp(n-1)×(1/n) (multiply,param=1/n,parent=_tmp(n-1),child=D)

mode=sum，输入[x1,...,xn]，结果D：
_tmp1=x1+x2 (add,param=x2,parent=x1,child=_tmp1)
_tmp(k)=_tmp(k-1)+x(k+2) (add,param=x(k+2),child=_tmp(k)) ← k=1到n-2
D=_tmp(n-1) (copy,parent=_tmp(n-1),child=D)

mode=growth，输入基数B，比例p，结果C：
_tmp1=B×p/100 (percent,parent=B,param=p,child=_tmp1)
C=B+_tmp1 (add,param=_tmp1值,parent=B,child=C)

mode=proportion，输入整体B，比例p，结果A：
A=B×p/100 (percent,parent=B,param=p,child=A)

mode=weighted_avg，输入[x1..xn],[w1..wn]，结果D：
_tmp(2i-1)=xi×wi (multiply,param=wi,child=_tmp(2i-1)) ← i=1到n
_tmp(2i)=_tmp(2i-2)+_tmp(2i-1) (add,child=_tmp(2i)) ← i=2到n
D=_tmp(2n-2) (copy,child=D)

【输出格式】
{
  "entities": [{"token":"值","label":"类型"}],
  "mode_reason": "触发词或判断依据",
  "mode": "simple|average|sum|growth|proportion|weighted_avg",
  "edge_checks": [
    {
      "parent": "父节点",
      "child": "子节点(_tmp或结果token)",
      "rel_type": "multiply|add|percent|copy",
      "calc": "完整算式及计算结果",
      "param_candidate": 数值,
      "calc_ok": true/false,
      "param_legal": true/false,
      "semantic_ok": true/false,
      "pass": true/false
    }
  ],
  "edges": [
    {
      "parent_tokens": ["单父节点"],
      "child_token": "单子节点",
      "rel_type": "multiply|add|percent|copy",
      "param": 数值
    }
  ]
}"""


class NERAPI:
    def __init__(
            self,
            api_key: Optional[str] = "sk-fravrtqnmmjbetfhpjfkzmrtpqsllvdfprfseajojuxdaxxc",
            model: str = "Pro/zai-org/GLM-4.7",  
            base_url: str = "https://api.siliconflow.cn/v1",
            temperature: float = 0.1,
            top_p: float = 0.8,
            max_tokens: int = 2048,
            enable_fallback: bool = True,
    ):
        """
        初始化通义千问 API 客户端

        Parameters
        ----------
        api_key : str
            DashScope API Key，若不传则从环境变量 DASHSCOPE_API_KEY 读取
        model : str
            模型名称，可选：qwen-turbo, qwen-plus, qwen-max, qwen-max-longcontext
        base_url : str
            DashScope 服务的 base_url
        temperature : float
            温度参数，控制随机性 (0.0-2.0)
        top_p : float
            核采样参数 (0.0-1.0)
        max_tokens : int
            最大生成 token 数
        """
        # 获取 API Key [citation:9]
        self.api_key = (
            api_key
            or os.getenv("SILICONFLOW_API_KEY")
            or os.getenv("DASHSCOPE_API_KEY")
        )
        self.model = model
        self.base_url = base_url
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.enable_fallback = enable_fallback

        # 创建 OpenAI 兼容客户端 [citation:3]
        self.client = None
        if self.api_key and OpenAI is not None:
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )

    def extract_entities(
            self, user_prompt: str
    ) -> Tuple[List[Tuple[str, str]], List[Dict]]:
        """
        调用通义千问 API 进行实体识别和关系抽取

        Returns
        -------
        entities : [(token, label), ...]
        edges    : [{"parent_tokens", "child_token", "rel_type", "param"}, ...]
        """
        if self.client is None:
            return self._fallback_extract_entities(user_prompt)

        try:
            # 调用通义千问 API [citation:9]
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": QWEN_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=self.max_tokens
            )

            # 获取响应内容
            content = completion.choices[0].message.content

        except Exception as e:
            print(f"API 调用失败: {e}")
            return self._fallback_extract_entities(user_prompt)

        # 清理可能的 think 标签
        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)

        # 解析 JSON 响应
        entities = []
        edges = []

        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            try:
                result = json.loads(json_match.group(0))

                # 解析 entities
                for e in result.get("entities", []):
                    token = e.get("token", "")
                    label = e.get("label", "T")
                    if token:
                        entities.append((token, canonical_label(label)))

                # 解析 edges
                raw_edge_list = result.get("edges", [])
                for ed in raw_edge_list:
                    edges.append({
                        "parent_tokens": ed.get("parent_tokens", []),
                        "child_token": ed.get("child_token", ""),
                        "rel_type": ed.get("rel_type", ""),
                        "param": ed.get("param", 1.0),
                    })

            except json.JSONDecodeError as e:
                print(f"JSON 解析失败: {e}")
                print(f"原始内容: {content}")

        if not entities and self.enable_fallback:
            return self._fallback_extract_entities(user_prompt)

        print(f"NERAPI entities: {entities}")
        print(f"NERAPI edges: {edges}")
        return entities, edges

    def _fallback_extract_entities(
            self, user_prompt: str
    ) -> Tuple[List[Tuple[str, str]], List[Dict]]:
        if not self.enable_fallback:
            return [], []

        entities = _local_extract_entities(user_prompt)
        print(f"NERAPI fallback entities: {entities}")
        return entities, []

    def __call__(
            self, user_prompt: str
    ) -> Tuple[List[Tuple[str, str]], List[Dict]]:
        return self.extract_entities(user_prompt)


_LOCAL_T3_TERMS = {
    "MEDICINE": {
        "阿司匹林", "布洛芬", "二甲双胍", "硝苯地平", "硝苯地平控释片",
        "头孢曲松", "头孢", "奥美拉唑", "青霉素", "胰岛素", "沙丁胺醇",
        "对乙酰氨基酚", "氯吡格雷", "氨氯地平", "美托洛尔",
    },
    "DISEASE": {
        "肺炎", "右肺肺炎", "高血压", "高血压病", "糖尿病", "哮喘",
        "冠心病", "胃炎", "肝炎", "肾炎", "感冒", "感染", "肿瘤",
        "癌症", "脑梗死", "心力衰竭",
    },
    "SYMPTOM": {
        "咳嗽", "发热", "头痛", "头晕", "恶心", "呕吐", "腹痛",
        "胸痛", "乏力", "气促", "呼吸困难", "咳痰", "皮疹", "水肿",
    },
    "THERAPY": {
        "治疗", "抗感染治疗", "放疗", "化疗", "手术", "静脉滴注",
        "输液", "透析", "康复", "介入治疗", "免疫治疗",
    },
    "FORBIDDEN_FOOD": {
        "海鲜", "酒精", "辛辣食物", "高糖食物", "油腻食物", "甜食",
        "饮酒", "咖啡", "浓茶",
    },
    "SIDE_EFFECT": {
        "恶心", "呕吐", "腹泻", "皮疹", "头晕", "嗜睡", "肝损伤",
        "肾损伤", "过敏", "胃肠道反应",
    },
    "ORGAN": {
        "肺", "右肺", "左肺", "胸部", "心脏", "肝脏", "肾脏", "胃",
        "脑", "血液", "白细胞", "血管", "胰腺", "甲状腺",
    },
    "FUNCTION": {
        "免疫", "代谢", "凝血", "呼吸", "循环", "消化", "吸收",
        "排泄", "白细胞计数", "血常规", "体温",
    },
}

_LOCAL_NAME_RE = re.compile(r"(?:患者|病人|姓名[:：]?)([\u4e00-\u9fff]{2,4})(?=[，,。；;、\s]|因|男|女|$)")
_LOCAL_GENE_RE = re.compile(r"(?<![A-Za-z0-9-])[A-Z]{2,}[A-Z0-9-]*\d[A-Z0-9-]*(?![A-Za-z0-9-])")
_LOCAL_NUM_PATTERNS = (
    (re.compile(r"(\d{1,3})岁"), "AGE"),
    (re.compile(r"(\d+(?:\.\d+)?)度"), "MEDICAL_VAL"),
    (re.compile(r"(\d+(?:\.\d+)?)(?:mg|g|ml|mmHg|mmol/L)\b", re.IGNORECASE), "MEDICAL_VAL"),
)


def _local_extract_entities(text: str) -> List[Tuple[str, str]]:
    spans: List[Tuple[int, int, str, str]] = []

    def add_span(start: int, end: int, token: str, label: str) -> None:
        if token and start >= 0 and end > start:
            spans.append((start, end, token, canonical_label(label)))

    for match in _LOCAL_NAME_RE.finditer(text):
        add_span(match.start(1), match.end(1), match.group(1), "PERSON")

    for match in _LOCAL_GENE_RE.finditer(text):
        add_span(match.start(), match.end(), match.group(0), "GENE")

    for pattern, label in _LOCAL_NUM_PATTERNS:
        for match in pattern.finditer(text):
            add_span(match.start(1), match.end(1), match.group(1), label)

    for label, terms in _LOCAL_T3_TERMS.items():
        for term in sorted(terms, key=len, reverse=True):
            for match in re.finditer(re.escape(term), text):
                actual_label = label
                context = text[max(0, match.start() - 8):match.start()]
                if (
                    label == "SYMPTOM"
                    and term in _LOCAL_T3_TERMS["SIDE_EFFECT"]
                    and ("副作用" in context or "不良反应" in context)
                ):
                    actual_label = "SIDE_EFFECT"
                add_span(match.start(), match.end(), match.group(0), actual_label)

    for match in re.finditer(r"(?:服用|予|给予|使用|口服|注射)([\u4e00-\u9fff]{2,12}(?:片|胶囊|颗粒|注射液|控释片|缓释片)?)", text):
        add_span(match.start(1), match.end(1), match.group(1), "MEDICINE")

    for match in re.finditer(r"(?:禁食|忌食|避免|不能吃)([\u4e00-\u9fff]{2,10})", text):
        add_span(match.start(1), match.end(1), match.group(1), "FORBIDDEN_FOOD")

    spans.sort(key=lambda item: (item[0], -(item[1] - item[0])))
    accepted: List[Tuple[int, int, str, str]] = []
    occupied: List[Tuple[int, int]] = []
    for start, end, token, label in spans:
        if any(start < old_end and end > old_start for old_start, old_end in occupied):
            continue
        accepted.append((start, end, token, label))
        occupied.append((start, end))

    return [(token, label) for _, _, token, label in accepted]


# ---------------------------------------------------------------------------
# 类型判断工具函数
# ---------------------------------------------------------------------------

T1_LABELS = {
    "PERSON", "ORG", "LOCATION", "PHONE", "EMAIL",
    "ID", "MEDICAL_ID", "IP", "URL",
}

# ENTITY 是通用标签，不应该被姓名替换
T1_NAME_LABELS = {
    "PERSON",  # 只有 PERSON 才需要姓名替换
}

T2_LABELS = {
    "AGE", "SALARY", "AMOUNT", "COUNT", "WEIGHT",
    "BLOOD_SUGAR", "BLOOD_PRESSURE", "MEDICAL_VAL",
    "YEAR", "DATE_NUM", "NUM", "PERCENT",
}

T3_LABELS = {
    "MEDICINE", "DISEASE", "SYMPTOM", "THERAPY",
    "FORBIDDEN_FOOD", "SIDE_EFFECT", "GENE", "ORGAN", "FUNCTION",
}

_LABEL_ALIASES = {
    "MEDICINE": "MEDICINE",
    "DISEASE": "DISEASE",
    "SYMPTOM": "SYMPTOM",
    "SYSMPTOM": "SYMPTOM",
    "THERAPY": "THERAPY",
    "FORBIDDEN_FOOD": "FORBIDDEN_FOOD",
    "FORBIDDENFOOD": "FORBIDDEN_FOOD",
    "SIDE_EFFECT": "SIDE_EFFECT",
    "SIDEEFFECT": "SIDE_EFFECT",
    "GENE": "GENE",
    "ORGAN": "ORGAN",
    "FUNCTION": "FUNCTION",
}


def canonical_label(label: str) -> str:
    raw = str(label or "").strip()
    if not raw:
        return raw

    key = re.sub(r"[^0-9A-Za-z]+", "_", raw).strip("_").upper()
    if key.startswith("T3_"):
        key = key[3:]
    elif key == "T3":
        return "T3"
    elif key.startswith("T1_") or key.startswith("T2_"):
        return key

    compact = key.replace("_", "")
    return _LABEL_ALIASES.get(key, _LABEL_ALIASES.get(compact, key))


def is_t1(label: str) -> bool:
    label = canonical_label(label)
    return label in T1_LABELS or label.lower().startswith("t1")


def is_t1_name(label: str) -> bool:
    """判断是否需要姓名替换（只有 PERSON）"""
    return canonical_label(label) in T1_NAME_LABELS


def is_t1_fpe(label: str) -> bool:
    """判断是否需要 FPE 加密（除了 PERSON 的其他 t1 标签）"""
    return is_t1(label) and not is_t1_name(label)


def is_t2(label: str) -> bool:
    label = canonical_label(label)
    return label in T2_LABELS or label.lower().startswith("t2")


def is_t3(label: str) -> bool:
    label = canonical_label(label)
    return label in T3_LABELS or label.lower().startswith("t3")
