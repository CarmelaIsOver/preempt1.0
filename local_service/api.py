"""
api.py

单次 NER 调用完成两项任务：
  1. NER：细粒度实体识别（t1/t2 分类）
  2. Relations：识别完整数值表达式及其相互引用关系

架构升级
--------
旧版：LLM 负责拆解链式 _tmp 步骤 → 复杂、易错、难维护
新版：LLM 输出完整表达式 + 表达式间引用 → sympy 自动解析建图

LLM 输出极简化：
  - entities: 实体列表
  - expressions: 每个关系一个对象，含 token / expression / depends_on / output_token
  - 表达式间引用通过 depends_on 显式声明，expression 中用 [token] 引用
  - 完全不拆 _tmp，复杂函数（sin/cos/log/条件）由 Python 原生表达式承载
"""

import json
import os
import re
from collections import deque
from typing import List, Tuple, Dict, Optional

from openai import OpenAI


QWEN_SYSTEM_PROMPT = """你是实体识别、数值关系分析专家。对用户输入，只输出 JSON，禁止输出 <think> 标签或任何思考过程。

【任务一】提取实体
t1 类：PERSON / ORG / LOCATION / PHONE / EMAIL / ID / MEDICAL_ID / IP / URL / ENTITY
t2 类（数值）：AGE / SALARY / AMOUNT / COUNT / WEIGHT / BLOOD_SUGAR / BLOOD_PRESSURE / MEDICAL_VAL / YEAR / DATE_NUM / NUM / PERCENT
  ★ 每个 t2 实体必须包含 id 字段：E1, E2, E3... 按出现顺序编号
  ★ allow_decimal: true（允许小数）或 false（年龄/数量/血压/年份必须整数）

【任务二】识别数值表达式及其引用关系

核心概念：
- 一句话中可能有多个数值计算关系，表达式之间可能存在相互引用：
  * 输入引用：[5000] 引用实体 "5000" —— 实体 token 必须出现在 entities 列表中
  * 结果引用：[total] 引用另一个表达式 "total" 的输出 —— total 必须是某个表达式的 output_token
- 每个表达式用 depends_on 字段明确列出它直接依赖的所有 token
- **重要**：depends_on 中的 token 必须与 expression 中 [方括号] 内引用的 token 完全一致，不能多也不能少

规则：
1. 为每个表达式分配唯一 token（如 "R1", "R2", "bonus", "total", "net"），不要与实体 token 重名
2. expression 字段用 [E1][E2] 引用 t2 实体，用 [token] 引用表达式输出
3. depends_on 列出 expression 中所有引用的 id/token，必须一一对应
4. output_token 默认为表达式的 token；若结果恰好对应某个实体，则用该实体 token
5. 支持标准数学函数：sin / cos / tan / log / sqrt / exp / abs / pow / max / min
6. 支持条件表达式：value_if_true if condition else value_if_false（如 [salary] * 0.2 if [salary] > 5000 else [salary] * 0.1）
7. 百分数自动转小数：20% → 0.2，增长率 10% → 0.1
8. 表达式按依赖顺序排列：被依赖的先写，依赖者后写（确保阅读时从上到下可执行）
9. **关键**：如果一句话中有链式计算（A 算 B，B 算 C），必须为每一步创建独立表达式，通过 [token] 串联

【示例 1】两个表达式共享输入
输入："工资5000元，奖金是工资的20%，个人所得税是工资的5%"
输出：
{
  "entities": [
    {"token": "5000", "label": "SALARY", "allow_decimal": true},
    {"token": "20%", "label": "PERCENT", "allow_decimal": true},
    {"token": "5%", "label": "PERCENT", "allow_decimal": true}
  ],
  "expressions": [
    {
      "token": "bonus",
      "expression": "[5000] * 0.2",
      "depends_on": ["5000"],
      "output_token": "bonus",
      "description": "奖金 = 工资 * 20%"
    },
    {
      "token": "tax",
      "expression": "[5000] * 0.05",
      "depends_on": ["5000"],
      "output_token": "tax",
      "description": "个税 = 工资 * 5%"
    }
  ]
}

【示例 2】链式引用（表达式间相互引用）
输入："工资5000，奖金1000，总收入是工资加奖金，税后收入是总收入减300"
输出：
{
  "entities": [
    {"token": "5000", "label": "SALARY", "allow_decimal": true},
    {"token": "1000", "label": "AMOUNT", "allow_decimal": true},
    {"token": "300", "label": "AMOUNT", "allow_decimal": true}
  ],
  "expressions": [
    {
      "token": "total",
      "expression": "[5000] + [1000]",
      "depends_on": ["5000", "1000"],
      "output_token": "total",
      "description": "总收入"
    },
    {
      "token": "net",
      "expression": "[total] - [300]",
      "depends_on": ["total", "300"],
      "output_token": "net",
      "description": "税后收入"
    }
  ]
}
注意：表达式 "net" 引用了表达式 "total" 的输出 → depends_on 包含 "total"，expression 中用 [total] 引用

【示例 3】三层链式引用
输入："单价100，数量5，总价是单价乘数量，税额是总价的13%，最终价格是总价加税额"
输出：
{
  "entities": [
    {"token": "100", "label": "AMOUNT", "allow_decimal": true},
    {"token": "5", "label": "COUNT", "allow_decimal": false},
    {"token": "13%", "label": "PERCENT", "allow_decimal": true}
  ],
  "expressions": [
    {
      "token": "total",
      "expression": "[100] * [5]",
      "depends_on": ["100", "5"],
      "output_token": "total",
      "description": "总价 = 单价 × 数量"
    },
    {
      "token": "tax_amount",
      "expression": "[total] * 0.13",
      "depends_on": ["total"],
      "output_token": "tax_amount",
      "description": "税额 = 总价 × 13%"
    },
    {
      "token": "final",
      "expression": "[total] + [tax_amount]",
      "depends_on": ["total", "tax_amount"],
      "output_token": "final",
      "description": "最终价格 = 总价 + 税额"
    }
  ]
}

【示例 4】条件表达式
输入："月薪8000元，绩效评分85分，绩效系数是1.2如果评分大于80否则1.0，最终薪资是月薪乘绩效系数"
输出：
{
  "entities": [
    {"token": "8000", "label": "SALARY", "allow_decimal": true},
    {"token": "85", "label": "NUM", "allow_decimal": true}
  ],
  "expressions": [
    {
      "token": "rate",
      "expression": "1.2 if [85] > 80 else 1.0",
      "depends_on": ["85"],
      "output_token": "rate",
      "description": "绩效系数"
    },
    {
      "token": "final_salary",
      "expression": "[8000] * [rate]",
      "depends_on": ["8000", "rate"],
      "output_token": "final_salary",
      "description": "最终薪资"
    }
  ]
}

【示例 5】复利计算
输入："本金10000元，年利率3.5%，存5年，复利计算本息"
输出：
{
  "entities": [
    {"token": "10000", "label": "AMOUNT", "allow_decimal": true},
    {"token": "3.5%", "label": "PERCENT", "allow_decimal": true},
    {"token": "5", "label": "YEAR", "allow_decimal": false}
  ],
  "expressions": [
    {
      "token": "R1",
      "expression": "[10000] * (1 + 0.035) ** [5]",
      "depends_on": ["10000", "5"],
      "output_token": "R1",
      "description": "复利本息和"
    }
  ]
}

【示例 6】分合关系（总分/分解 — 关键场景）
输入："张三年薪120万，其中基本工资100万，绩效奖金20万"
分析：120万 = 100万 + 20万，这是隐含的求和约束
输出：
{
  "entities": [
    {"token": "张三", "label": "PERSON", "allow_decimal": false},
    {"token": "120万", "label": "SALARY", "allow_decimal": true},
    {"token": "100万", "label": "SALARY", "allow_decimal": true},
    {"token": "20万", "label": "SALARY", "allow_decimal": true}
  ],
  "expressions": [
    {
      "token": "R1",
      "expression": "[100万] + [20万]",
      "depends_on": ["100万", "20万"],
      "output_token": "120万",
      "description": "年薪 = 基本工资 + 绩效奖金"
    }
  ]
}

【示例 7】增长率关系
输入："季度总收入为80万，同比增长25%，增长额为20万"
分析：增长额 = 总收入 × 增长率，即 20 = 80 × 0.25
输出：
{
  "entities": [
    {"token": "80万", "label": "AMOUNT", "allow_decimal": true},
    {"token": "25%", "label": "PERCENT", "allow_decimal": true},
    {"token": "20万", "label": "AMOUNT", "allow_decimal": true}
  ],
  "expressions": [
    {
      "token": "growth",
      "expression": "[80万] * 0.25",
      "depends_on": ["80万"],
      "output_token": "20万",
      "description": "增长额 = 总收入 × 增长率"
    }
  ]
}

【示例 8】混合运算（含函数）
输入："圆的半径为5cm，面积为78.5平方厘米，圆周长为31.4cm"
输出：
{
  "entities": [
    {"token": "5", "label": "NUM", "allow_decimal": true},
    {"token": "78.5", "label": "NUM", "allow_decimal": true},
    {"token": "31.4", "label": "NUM", "allow_decimal": true}
  ],
  "expressions": [
    {
      "token": "area",
      "expression": "3.14 * [5] ** 2",
      "depends_on": ["5"],
      "output_token": "78.5",
      "description": "面积 = π × r²"
    },
    {
      "token": "circumference",
      "expression": "2 * 3.14 * [5]",
      "depends_on": ["5"],
      "output_token": "31.4",
      "description": "周长 = 2πr"
    }
  ]
}

输出格式严格为：
{
  "entities": [...],
  "expressions": [...]
}

若输入中没有数值计算关系，expressions 返回空数组 []。
"""


BATCH_SYSTEM_PROMPT = """你是实体识别、数值关系分析专家。只输出 JSON 数组，禁止输出任何解释。

对每条输入文本，输出一个 JSON 对象，包含 entities 和 expressions。

【实体分类】
t1：PERSON/ORG/LOCATION/PHONE/EMAIL/ID/MEDICAL_ID/IP/URL/ENTITY（不需要 id）
t2：AGE/SALARY/AMOUNT/COUNT/WEIGHT/BLOOD_SUGAR/BLOOD_PRESSURE/MEDICAL_VAL/YEAR/DATE_NUM/NUM/PERCENT
  ★ 每个 t2 实体必须有 id 字段：E1, E2, E3... 按出现顺序编号
  ★ allow_decimal: true（允许小数）或 false（年龄/数量/血压/年份必须整数）

【表达式规则 — 用实体 id 引用，不用 token 文本】
- expression 中用 [E1], [E2] 等引用 t2 实体
- 引用表达式输出时用 [R1], [total] 等 token
- depends_on 列出所有引用的 id/token
- output_token 默认为表达式 token，若结果对应某个 t2 实体则填该实体的 id
- 条件表达式：x if cond else y
- 百分数转小数：20% → 0.2

【示例】
输入："工资5000，奖金1000，总收入是工资加奖金，税后是总收入减300"
输出：
{"entities":[
  {"id":"E1","token":"5000","label":"SALARY","allow_decimal":true},
  {"id":"E2","token":"1000","label":"AMOUNT","allow_decimal":true},
  {"id":"E3","token":"300","label":"AMOUNT","allow_decimal":true}
],
"expressions":[
  {"token":"total","expression":"[E1] + [E2]","depends_on":["E1","E2"],"output_token":"total"},
  {"token":"net","expression":"[total] - [E3]","depends_on":["total","E3"],"output_token":"net"}
]}

输入："张三年薪120万，基本工资100万，绩效20万"
输出：
{"entities":[
  {"id":"E1","token":"张三","label":"PERSON"},
  {"id":"E2","token":"120万","label":"SALARY","allow_decimal":true},
  {"id":"E3","token":"100万","label":"SALARY","allow_decimal":true},
  {"id":"E4","token":"20万","label":"SALARY","allow_decimal":true}
],
"expressions":[
  {"token":"R1","expression":"[E3] + [E4]","depends_on":["E3","E4"],"output_token":"E2"}
]}

输出格式：严格 JSON 数组，每条一个对象。若某条无表达式则 expressions 为空数组。
"""


class NERAPI:
    def __init__(
        self,
        api_key: str = "a3cba7253c98456e85c785b91e6b4fc6.wh7yHcxeW6bxLCAI",
        model: str = "glm-4-plus",
        base_url: str = "https://open.bigmodel.cn/api/paas/v4/",
        temperature: float = 0.1,
        top_p: float = 0.8,
        max_tokens: int = 4096
    ):
        self.api_key = api_key or os.getenv("PREEMPT_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
        if not self.api_key:
            raise ValueError("请设置 DASHSCOPE_API_KEY 环境变量或传入 api_key")
        self.model = model
        self.base_url = base_url
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def extract_entities(
        self, user_prompt: str
    ) -> Tuple[List[Tuple[str, str]], List[Dict]]:
        """调用 API，返回 (entities, expressions) — 已做规范化处理"""
        try:
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
            content = completion.choices[0].message.content
        except Exception as e:
            print(f"API 调用失败: {e}")
            return [], []

        # 清理 think 标签
        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)

        entities: List[Tuple[str, str]] = []
        expressions: List[Dict] = []

        # 更鲁棒的 JSON 提取：尝试匹配最外层大括号
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if not json_match:
            print(f"未找到 JSON 对象，原始内容: {content[:200]}")
            return [], []

        try:
            result = json.loads(json_match.group(0))
        except json.JSONDecodeError as e:
            # 回退：尝试修复常见格式问题后再解析
            print(f"JSON 解析失败: {e}，尝试修复...")
            result = self._repair_and_parse(json_match.group(0))
            if result is None:
                print(f"修复失败，原始内容: {content[:500]}")
                return [], []

        for i, e in enumerate(result.get("entities", [])):
            if not isinstance(e, dict) or "token" not in e:
                continue
            allow_decimal = e.get("allow_decimal", True)
            eid = e.get("id", f"E{i+1}")
            entities.append((eid, e["token"], e["label"], allow_decimal))
        for expr in result.get("expressions", []):
            expressions.append({
                "token": expr["token"],
                "expression": expr["expression"],
                "depends_on": expr.get("depends_on", []),
                "output_token": expr.get("output_token", expr["token"]),
                "description": expr.get("description", ""),
            })

        # 后处理：规范化 expressions
        expressions = self._normalize_expressions(entities, expressions)

        print(f"NERAPI entities: {entities}")
        print(f"NERAPI expressions: {expressions}")
        return entities, expressions

    # ------------------------------------------------------------------
    # 内部方法：JSON 修复
    # ------------------------------------------------------------------
    @staticmethod
    def _repair_and_parse(json_str: str) -> Optional[Dict]:
        """尝试修复常见 LLM JSON 格式问题"""
        # 1. 去除尾逗号
        repaired = re.sub(r',\s*([}\]])', r'\1', json_str)
        # 2. 单引号转双引号（简单情况）
        # 3. 去除注释
        repaired = re.sub(r'//.*?\n', '\n', repaired)
        repaired = re.sub(r'/\*.*?\*/', '', repaired, flags=re.DOTALL)
        try:
            return json.loads(repaired)
        except json.JSONDecodeError:
            return None

    # ------------------------------------------------------------------
    # 内部方法：表达式规范化
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_expressions(
        entities: List[Tuple[str, str]],
        expressions: List[Dict],
    ) -> List[Dict]:
        """
        规范化表达式列表：
        1. 从 expression 的 [token] 中自动提取 depends_on（若缺失或不一致）
        2. 验证引用 token 是否存在于 entities 或 expressions 的 output_token 中
        3. 拓扑排序（依赖先于被依赖）
        4. 检测并报告循环引用
        """
        if not expressions:
            return expressions

        # 构建所有可引用的名称：entity ID (E1,E2) + token 文本 (300万,500万)
        entity_tokens = set()
        for ent in entities:
            entity_tokens.add(ent[0])  # entity ID
            if len(ent) >= 2:
                entity_tokens.add(ent[1])  # token 文本
        expr_outputs: Dict[str, int] = {}
        normalized: List[Dict] = []

        # ---- Pass 1: 修正 depends_on 与 output_token ----
        for i, expr in enumerate(expressions):
            e = dict(expr)  # 浅拷贝
            expression = e.get("expression", "")
            raw_depends = list(e.get("depends_on", []))

            # 从 expression 的 [token] 提取引用
            refs = re.findall(r'\[([^\]]+)\]', expression)

            if refs:
                # 以 [token] 引用为准
                if set(refs) != set(raw_depends):
                    if raw_depends:
                        print(f"  规范化: 表达式 '{e['token']}' depends_on {raw_depends} → 修正为 {refs}")
                    else:
                        print(f"  规范化: 表达式 '{e['token']}' 自动提取 depends_on = {refs}")
                    e["depends_on"] = refs
            elif raw_depends:
                # expression 中没有 [token] 但有 depends_on → 保留
                # 尝试在 expression 文本中直接找 token
                print(f"  规范化: 表达式 '{e['token']}' 无 [token] 引用，保留 depends_on = {raw_depends}")
            # else: 无引用也无 depends_on → 空依赖，可能是常量表达式

            # 确保 output_token 存在
            if "output_token" not in e:
                e["output_token"] = e.get("token", f"R{i}")

            normalized.append(e)

        # ---- Pass 2: 验证引用目标存在 ----
        # 收集所有可引用的 token（实体 + 表达式输出）
        for expr in normalized:
            expr_outputs[expr["output_token"]] = expr_outputs.get(expr["output_token"], 0) + 1
        all_valid_refs = entity_tokens | set(expr_outputs.keys())

        for expr in normalized:
            valid_deps = []
            for dep in expr.get("depends_on", []):
                if dep in all_valid_refs:
                    valid_deps.append(dep)
                else:
                    print(f"  警告: 表达式 '{expr['token']}' 引用了不存在的 token '{dep}'，已移除")
            expr["depends_on"] = valid_deps

            # 去重 depends_on（保持顺序）
            seen = set()
            deduped = []
            for dep in expr["depends_on"]:
                if dep not in seen:
                    seen.add(dep)
                    deduped.append(dep)
            expr["depends_on"] = deduped

        # ---- Pass 3: 拓扑排序 ----
        if len(normalized) > 1:
            # output_token → 表达式在 normalized 列表中的原始索引
            out_to_idx: Dict[str, int] = {}
            for i, expr in enumerate(normalized):
                out_to_idx[expr["output_token"]] = i

            in_deg = [0] * len(normalized)
            children: Dict[int, List[int]] = {i: [] for i in range(len(normalized))}

            for i, expr in enumerate(normalized):
                for dep in expr.get("depends_on", []):
                    if dep in out_to_idx and out_to_idx[dep] != i:
                        parent_idx = out_to_idx[dep]
                        in_deg[i] += 1
                        children[parent_idx].append(i)

            queue = deque([i for i, d in enumerate(in_deg) if d == 0])
            sorted_indices = []
            while queue:
                cur = queue.popleft()
                sorted_indices.append(cur)
                for child in children.get(cur, []):
                    in_deg[child] -= 1
                    if in_deg[child] == 0:
                        queue.append(child)

            if len(sorted_indices) == len(normalized):
                normalized = [normalized[i] for i in sorted_indices]
            else:
                print(f"  警告: 表达式间存在循环依赖 ({len(normalized)} 个表达式, {len(sorted_indices)} 个可排序)，保持原始顺序")

        return normalized

    # ------------------------------------------------------------------
    # 批量处理
    # ------------------------------------------------------------------
    def extract_entities_batch(
        self,
        texts: List[Dict],           # [{"id": ..., "text": ...}, ...]
        batch_size: int = 100,
    ) -> List[Dict]:
        """
        批量调用 API，一次处理多条文本（默认100条），大幅节省 system prompt token。

        返回：[{"id": ..., "text": ..., "entities": [...], "expressions": [...]}, ...]
        """
        results = []
        for batch_start in range(0, len(texts), batch_size):
            batch = texts[batch_start:batch_start + batch_size]
            print(f"  批量处理 [{batch_start+1}-{batch_start+len(batch)}]/{len(texts)} ({len(batch)} 条)...")

            batch_prompt = self._build_batch_prompt(batch)
            try:
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": BATCH_SYSTEM_PROMPT},
                        {"role": "user", "content": batch_prompt}
                    ],
                    temperature=self.temperature,
                    top_p=self.top_p,
                    max_tokens=16384  # 大批量输出需要更多 token
                )
                content = completion.choices[0].message.content
            except Exception as e:
                print(f"  批量 API 调用失败: {e}")
                # 回退到逐条处理
                for item in batch:
                    entities, expressions = self.extract_entities(item["text"])
                    results.append({
                        "id": item["id"], "text": item["text"],
                        "entities": entities, "expressions": expressions,
                    })
                continue

            # 清理
            content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)

            # 解析 JSON 数组
            batch_results = self._parse_batch_response(content, batch)
            results.extend(batch_results)

        return results

    @staticmethod
    def _build_batch_prompt(texts: List[Dict]) -> str:
        """构建批量处理的用户提示词"""
        lines = ["请分析以下文本，对每条文本输出一个 JSON 对象。最后将所有结果放在一个 JSON 数组中返回。\n"]
        for item in texts:
            lines.append(f"[ID:{item['id']}] {item['text']}")
        return "\n".join(lines)

    @staticmethod
    def _parse_batch_response(content: str, batch: List[Dict]) -> List[Dict]:
        """解析批量 API 返回的 JSON 数组"""
        results = []

        # 尝试提取 JSON 数组
        json_match = re.search(r'\[.*\]', content, re.DOTALL)
        if not json_match:
            print(f"  批量响应中未找到 JSON 数组，回退逐条处理...")
            return NERAPI._fallback_single(batch)

        try:
            arr = json.loads(json_match.group(0))
        except json.JSONDecodeError as e:
            print(f"  批量 JSON 解析失败: {e}，回退逐条处理...")
            return NERAPI._fallback_single(batch)

        if not isinstance(arr, list):
            print(f"  批量响应不是数组，回退逐条处理...")
            return NERAPI._fallback_single(batch)

        # 按 id 或顺序匹配结果
        for i, item in enumerate(batch):
            entities = []
            expressions = []
            # 尝试从 arr 中找到匹配项
            result_obj = None
            if i < len(arr):
                result_obj = arr[i]
            # 也可以按 id 匹配
            for obj in arr:
                if isinstance(obj, dict) and str(obj.get("id", "")) == str(item["id"]):
                    result_obj = obj
                    break

            if result_obj and isinstance(result_obj, dict):
                for j, e in enumerate(result_obj.get("entities", [])):
                    if not isinstance(e, dict) or "token" not in e:
                        continue  # 跳过格式不正确的实体
                    allow_decimal = e.get("allow_decimal", True)
                    eid = e.get("id", f"E{j+1}")
                    entities.append((eid, e["token"], e["label"], allow_decimal))
                for expr in result_obj.get("expressions", []):
                    if not isinstance(expr, dict) or "expression" not in expr:
                        continue
                    expressions.append({
                        "token": expr["token"],
                        "expression": expr["expression"],
                        "depends_on": expr.get("depends_on", []),
                        "output_token": expr.get("output_token", expr["token"]),
                        "description": expr.get("description", ""),
                    })
                # 规范化
                expressions = NERAPI._normalize_expressions(entities, expressions)

            results.append({
                "id": item["id"], "text": item["text"],
                "entities": entities, "expressions": expressions,
            })

        return results

    @staticmethod
    def _fallback_single(batch: List[Dict]) -> List[Dict]:
        """逐条处理回退 —— 自动把表达式里的文本 token 引用替换为实体 ID"""
        ner = NERAPI()
        results = []
        for item in batch:
            entities, expressions = ner.extract_entities(item["text"])
            # 构建 token → entity_id 映射
            token_to_id = {}
            for eid, token, label, ad in entities:
                token_to_id[token] = eid
            # 修正表达式：把所有 [token文本] 替换为 [entity_id]
            for expr in expressions:
                raw_expr = expr["expression"]
                # 从 expression 中提取所有 [xxx] 引用
                all_refs = re.findall(r'\[([^\]]+)\]', raw_expr)
                new_deps = list(expr.get("depends_on", []))
                for ref in all_refs:
                    if ref in token_to_id:
                        eid = token_to_id[ref]
                        raw_expr = raw_expr.replace(f"[{ref}]", f"[{eid}]")
                        if eid not in new_deps:
                            new_deps.append(eid)
                    elif ref not in new_deps:
                        new_deps.append(ref)
                expr["expression"] = raw_expr
                expr["depends_on"] = new_deps
                # output_token 也替换
                out = expr.get("output_token", "")
                if out in token_to_id:
                    expr["output_token"] = token_to_id[out]
            results.append({
                "id": item["id"], "text": item["text"],
                "entities": entities, "expressions": expressions,
            })
        return results

    def __call__(self, user_prompt: str):
        return self.extract_entities(user_prompt)


# ---------------------------------------------------------------------------
# 类型判断工具函数
# ---------------------------------------------------------------------------
T1_LABELS = {"PERSON","ORG","LOCATION","PHONE","EMAIL","ID","MEDICAL_ID","IP","URL","ENTITY"}
T2_LABELS = {"AGE","SALARY","AMOUNT","COUNT","WEIGHT","BLOOD_SUGAR","BLOOD_PRESSURE","MEDICAL_VAL","YEAR","DATE_NUM","NUM","PERCENT"}

def is_t1(label: str) -> bool:
    return label in T1_LABELS or label.lower().startswith("t1")

def is_t2(label: str) -> bool:
    return label in T2_LABELS or label.lower().startswith("t2")