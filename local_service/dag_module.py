"""
dag_module.py

可执行函数超图（Executable Functional HyperGraph）
基于 sympy 的表达式解析与自动建图

核心对象
--------
  ValueNode        : 全局唯一值节点
  FunctionalRelation : 可执行函数关系（多输入→单输出）
  FunctionalGraph  : 关系超图
  ExpressionParser : 用 sympy 解析表达式，自动提取符号依赖并建图
"""

from __future__ import annotations

import re
import math
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import sympy


# ---------------------------------------------------------------------------
# ValueNode
# ---------------------------------------------------------------------------
@dataclass
class ValueNode:
    node_id: str
    token: str
    value: Union[int, float]
    label: str = "NUM"
    precision: int = 0
    is_temp: bool = False
    allow_decimal: bool = True   # True=允许小数, False=必须整数（如年龄、件数）
    producer_relation: Optional[str] = None
    consumer_relations: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.is_temp and self.precision == 0 and isinstance(self.value, float):
            s = str(self.value)
            if '.' in s:
                self.precision = len(s.split('.')[1])

    def to_stored(self) -> int:
        if self.precision > 0:
            return int(round(self.value * (10 ** self.precision)))
        return int(round(self.value))

    @classmethod
    def from_stored(cls, node_id: str, token: str, stored: int, precision: int,
                    label: str = "NUM") -> "ValueNode":
        value = stored / (10 ** precision) if precision > 0 else stored
        return cls(node_id=node_id, token=token, value=value, label=label, precision=precision)


# ---------------------------------------------------------------------------
# FunctionalRelation
# ---------------------------------------------------------------------------
@dataclass
class FunctionalRelation:
    relation_id: str
    input_node_ids: List[str]
    output_node_id: str
    raw_expression: str = ""
    executable_expr: str = ""       # 可执行 Python 代码
    variable_mapping: Dict[str, str] = field(default_factory=dict)  # var -> node_id
    confidence: float = 1.0

    def execute(self, node_registry: Dict[str, ValueNode]) -> float:
        if not self.executable_expr:
            raise ValueError(f"[{self.relation_id}] executable_expr 为空")
        local_vars = {}
        for var, nid in self.variable_mapping.items():
            if nid not in node_registry:
                raise KeyError(f"[{self.relation_id}] 输入节点 '{nid}' 不在注册表中")
            local_vars[var] = node_registry[nid].value
        try:
            allowed = {
                "sin": math.sin, "cos": math.cos, "tan": math.tan,
                "log": math.log, "sqrt": math.sqrt, "exp": math.exp,
                "abs": abs, "pow": pow, "pi": math.pi, "e": math.e,
                "min": min, "max": max, "floor": math.floor, "ceil": math.ceil,
                "round": round, "int": int, "float": float,
                "True": True, "False": False
            }
            # 修复 LLM 可能误用的运算符：^ → **（Python 中 ^ 是 XOR，不是幂）
            safe_expr = self.executable_expr.replace("^", "**")
            result = eval(safe_expr, {"__builtins__": {}}, {**allowed, **local_vars})
        except Exception as e:
            raise RuntimeError(f"[{self.relation_id}] 执行失败: {e}")
        return float(result)


# ---------------------------------------------------------------------------
# FunctionalGraph
# ---------------------------------------------------------------------------
@dataclass
class FunctionalGraph:
    graph_id: int
    node_ids: List[str] = field(default_factory=list)
    relations: List[FunctionalRelation] = field(default_factory=list)
    roots: List[str] = field(default_factory=list)
    epsilon: float = 0.0

    def propagate(self, node_registry: Dict[str, ValueNode]) -> None:
        producer = {r.output_node_id: r for r in self.relations}
        # 计算入度：每个输入节点 → 输出节点 是一条边
        in_deg: Dict[str, int] = {nid: 0 for nid in self.node_ids}
        # 同时建立 consumer map 加速查找
        consumers: Dict[str, List[str]] = defaultdict(list)  # input_node -> [output_node (可能重复)]
        for r in self.relations:
            for inp in r.input_node_ids:
                in_deg[r.output_node_id] = in_deg.get(r.output_node_id, 0) + 1
                consumers[inp].append(r.output_node_id)
        queue = deque(nid for nid in self.node_ids if in_deg.get(nid, 0) == 0)
        visited = set(queue)
        while queue:
            cur = queue.popleft()
            if cur in producer:
                rel = producer[cur]
                node_registry[cur].value = rel.execute(node_registry)
            # 沿 consumer map 传播（每个下游节点可能因 cur 出现多次而多次减入度）
            for output_nid in consumers.get(cur, []):
                in_deg[output_nid] -= 1
                if in_deg[output_nid] == 0 and output_nid not in visited:
                    visited.add(output_nid)
                    queue.append(output_nid)


# ---------------------------------------------------------------------------
# ExpressionParser
# ---------------------------------------------------------------------------
class ExpressionParser:
    SYMPY_FUNCS = {
        'sin': sympy.sin, 'cos': sympy.cos, 'tan': sympy.tan,
        'log': sympy.log, 'sqrt': sympy.sqrt, 'exp': sympy.exp,
        'abs': sympy.Abs, 'pow': sympy.Pow
    }

    @classmethod
    def parse_expressions(
        cls,
        entities: List[Tuple[str, str]],
        expressions: List[Dict],
        node_registry: Optional[Dict[str, ValueNode]] = None,
    ) -> Tuple[Dict[str, ValueNode], List[FunctionalRelation]]:
        """
        从实体和表达式列表构建节点注册表和关系列表。
        若提供 node_registry，则复用已有节点（保留额外属性），仅补充缺失的临时节点。
        """
        if node_registry is None:
            node_registry = {}

        # 1. 从实体创建节点（使用实体 id 作为 node_id，token 仅为显示文本）
        for ent in entities:
            if len(ent) == 3:
                # 旧格式: (token, label, allow_decimal) → 自动分配 id
                token, label, allow_decimal = ent
                eid = token
            elif len(ent) == 4:
                # 新格式: (id, token, label, allow_decimal)
                eid, token, label, allow_decimal = ent
            else:
                # 最旧格式: (token, label)
                token, label = ent[0], ent[1]
                allow_decimal = True
                eid = token

            if eid not in node_registry:
                value = _extract_numeric_value(token)
                if value is None:
                    continue
                node_registry[eid] = ValueNode(
                    node_id=eid, token=token, value=value, label=label,
                    allow_decimal=allow_decimal
                )

        # 2. 为表达式输出创建临时节点（若未在注册表中）
        for expr in expressions:
            out_token = expr.get("output_token", expr["token"])
            if out_token not in node_registry:
                node_registry[out_token] = ValueNode(
                    node_id=out_token,
                    token=out_token,
                    value=0.0,
                    is_temp=True,
                    label="_temp"
                )

        # ---- 拓扑排序表达式：确保依赖先于被依赖处理 ----
        sorted_expressions = _topo_sort_expressions(expressions, node_registry)

        relations = []
        for i, expr in enumerate(sorted_expressions):
            rid = f"rel_{i}"
            token = expr["token"]
            expression = expr["expression"]
            depends_on = list(expr.get("depends_on", []))
            out_token = expr.get("output_token", token)

            # 1. 从 expression 的 [token] 中提取引用，以引用为准
            refs = re.findall(r'\[([^\]]+)\]', expression)
            if not refs and depends_on:
                # expression 中没有 [token] 引用但声明了依赖 → 自动补充
                refs = list(depends_on)
            if set(refs) != set(depends_on):
                if refs:
                    print(f"警告：表达式 {token} 引用 {refs} 与 depends_on {depends_on} 不一致，以引用为准")
                    depends_on = list(refs)
                else:
                    print(f"警告：表达式 {token} 无 [token] 引用，保留 depends_on: {depends_on}")

            # 2. 检查依赖节点是否存在，尝试模糊匹配 + 动态创建
            resolved_deps = []
            for dep in depends_on:
                if dep in node_registry:
                    resolved_deps.append(dep)
                    continue
                # 尝试：dep 的数值部分匹配某个实体 token
                dep_val = _extract_numeric_value(dep)
                matched = None
                if dep_val is not None:
                    # 查找 node_registry 中数值相同的 token
                    for nid, node in node_registry.items():
                        if abs(node.value - dep_val) < 1e-6:
                            matched = nid
                            break
                if matched is None:
                    # 尝试：dep 是某个实体 token 的前缀（如 "300" 匹配 "300万元"）
                    for nid in node_registry:
                        if nid.startswith(dep) and _extract_numeric_value(nid) is not None:
                            matched = nid
                            break
                if matched is not None:
                    print(f"  模糊匹配: [{dep}] → [{matched}]")
                    # 同步更新 expression 中的引用
                    expression = expression.replace(f"[{dep}]", f"[{matched}]")
                    resolved_deps.append(matched)
                else:
                    # 尝试直接从 token 创建节点（如 "半" → 0.5）
                    if dep_val is not None:
                        node_registry[dep] = ValueNode(
                            node_id=dep, token=dep, value=dep_val,
                            label="NUM", is_temp=True
                        )
                        print(f"  动态创建节点: [{dep}] = {dep_val}")
                        resolved_deps.append(dep)
                    else:
                        print(f"警告：依赖 '{dep}' 不在注册表中且无法解析，表达式 {token} 可能有问题")

            # 用解析后的依赖替换
            depends_on = resolved_deps
            missing = [d for d in depends_on if d not in node_registry]
            if missing:
                print(f"警告：依赖 {missing} 仍不在注册表中，跳过表达式 {token}")
                continue

            # 3. 构建节点→变量映射（相同节点映射到同一个变量，避免重复计数）
            node_to_var: Dict[str, str] = {}
            var_idx = 1
            for dep in depends_on:
                if dep not in node_to_var:
                    node_to_var[dep] = f"x{var_idx}"
                    var_idx += 1

            # 去重后的输入节点列表（保持拓扑序中的首次出现）
            seen_inputs: set = set()
            deduped_inputs: List[str] = []
            for dep in depends_on:
                if dep not in seen_inputs:
                    seen_inputs.add(dep)
                    deduped_inputs.append(dep)

            # 4. 替换 expression 中的 [token] → 变量名
            #    先替换长 token（避免子串匹配问题）
            sorted_deps = sorted(node_to_var.items(), key=lambda kv: len(kv[0]), reverse=True)
            expr_for_sympy = expression
            executable_expr = expression
            for dep_token, var_name in sorted_deps:
                placeholder = f"[{dep_token}]"
                expr_for_sympy = expr_for_sympy.replace(placeholder, var_name)
                executable_expr = executable_expr.replace(placeholder, var_name)

            # 5. sympy 解析验证
            try:
                sympy_expr = sympy.sympify(expr_for_sympy, locals=cls.SYMPY_FUNCS)
                free_syms = sympy_expr.free_symbols
                extracted_vars = {str(s) for s in free_syms}
                used_vars = set(node_to_var.values())
                extra_syms = extracted_vars - used_vars
                if extra_syms:
                    print(f"警告：sympy 解析出额外符号 {extra_syms}（表达式: {expr_for_sympy}）")
            except Exception as e:
                print(f"sympy 解析失败（将使用正则提取）: {e} （转换后: {expr_for_sympy}）")

            # 6. 构建 variable_mapping: var_name → node_id
            var_map: Dict[str, str] = {v: k for k, v in node_to_var.items()}

            # 7. 按变量名自然排序 input_node_ids（保证执行时一致性）
            sorted_vars = sorted(var_map.keys(), key=_natural_sort_key)

            relations.append(FunctionalRelation(
                relation_id=rid,
                input_node_ids=deduped_inputs,
                output_node_id=out_token,
                raw_expression=expression,
                executable_expr=executable_expr,
                variable_mapping={v: var_map[v] for v in sorted_vars},
                confidence=1.0
            ))

        # 注册 producer / consumer
        for rel in relations:
            out_node = node_registry.get(rel.output_node_id)
            if out_node:
                out_node.producer_relation = rel.relation_id
            for nid in rel.input_node_ids:
                in_node = node_registry.get(nid)
                if in_node and rel.relation_id not in in_node.consumer_relations:
                    in_node.consumer_relations.append(rel.relation_id)

        return node_registry, relations


# ---------------------------------------------------------------------------
# FunctionalGraphBuilder
# ---------------------------------------------------------------------------
class FunctionalGraphBuilder:
    def build_from_expressions(
        self,
        entities: List[Tuple[str, str]],
        expressions: List[Dict],
        total_epsilon: float,
        node_registry: Optional[Dict[str, ValueNode]] = None,
    ) -> Tuple[List[FunctionalGraph], Dict[str, ValueNode]]:
        # 解析
        node_registry, relations = ExpressionParser.parse_expressions(
            entities, expressions, node_registry=node_registry
        )
        if not node_registry:
            return [], {}

        # 去环
        all_node_ids = list(node_registry.keys())
        valid_relations = _remove_cycles_relations(relations, all_node_ids)

        # 并查集分连通分量
        uf = {nid: nid for nid in all_node_ids}
        def find(x):
            while uf[x] != x:
                uf[x] = uf[uf[x]]
                x = uf[x]
            return x
        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                uf[rb] = ra

        for rel in valid_relations:
            for inp in rel.input_node_ids:
                union(inp, rel.output_node_id)

        groups = defaultdict(list)
        for nid in all_node_ids:
            groups[find(nid)].append(nid)
        group_relations = defaultdict(list)
        for rel in valid_relations:
            group_relations[find(rel.output_node_id)].append(rel)

        graphs = []
        for gid, (rep, node_ids) in enumerate(groups.items()):
            g_rels = group_relations[rep]
            output_ids = {r.output_node_id for r in g_rels}
            roots = [nid for nid in node_ids if nid not in output_ids]
            graphs.append(FunctionalGraph(
                graph_id=gid,
                node_ids=node_ids,
                relations=g_rels,
                roots=roots,
                epsilon=0.0
            ))

        total_roots = sum(len(g.roots) for g in graphs)
        eps_per_root = total_epsilon / max(total_roots, 1)
        for g in graphs:
            g.epsilon = eps_per_root * len(g.roots)

        return graphs, node_registry


# ---------------------------------------------------------------------------
# FunctionalGraphPerturber
# ---------------------------------------------------------------------------
class FunctionalGraphPerturber:
    def __init__(self):
        import mLDP_module as mLDP
        self._mLDP = mLDP
        self._mldp_cache = {}

    def _get_mldp(self, epsilon):
        key = str(epsilon)
        if key not in self._mldp_cache:
            self._mldp_cache[key] = self._mLDP.mLDPMechanism(epsilon=epsilon)
        return self._mldp_cache[key]

    def _perturb_root(self, node, epsilon):
        stored = node.to_stored()
        noisy_stored = self._get_mldp(epsilon).perturb(stored)
        if node.precision > 0:
            return noisy_stored / (10 ** node.precision)
        return float(noisy_stored)

    def perturb_graphs(self, graphs, node_registry):
        noisy_map = {}
        for graph in graphs:
            eps_per_root = graph.epsilon / max(len(graph.roots), 1)
            for root_id in graph.roots:
                if root_id in node_registry:
                    noisy_val = self._perturb_root(node_registry[root_id], eps_per_root)
                    noisy_map[root_id] = noisy_val
                    node_registry[root_id].value = noisy_val
            producer = {r.output_node_id: r for r in graph.relations}
            # 使用 consumer map 避免重复依赖的入度计算错误
            in_deg: Dict[str, int] = {nid: 0 for nid in graph.node_ids}
            consumers: Dict[str, List[str]] = defaultdict(list)
            for r in graph.relations:
                for inp in r.input_node_ids:
                    in_deg[r.output_node_id] = in_deg.get(r.output_node_id, 0) + 1
                    consumers[inp].append(r.output_node_id)
            queue = deque(nid for nid in graph.node_ids if in_deg.get(nid, 0) == 0)
            visited = set(queue)
            while queue:
                cur = queue.popleft()
                if cur in producer and cur not in noisy_map:
                    rel = producer[cur]
                    try:
                        result = rel.execute(node_registry)
                        noisy_map[cur] = result
                        node_registry[cur].value = result
                    except Exception as e:
                        print(f"[Perturber] {rel.relation_id} 执行失败: {e}")
                for output_nid in consumers.get(cur, []):
                    in_deg[output_nid] -= 1
                    if in_deg[output_nid] == 0 and output_nid not in visited:
                        visited.add(output_nid)
                        queue.append(output_nid)
            for nid in graph.node_ids:
                if nid not in noisy_map and nid in node_registry:
                    node = node_registry[nid]
                    if not node.is_temp:
                        noisy_val = self._perturb_root(node, eps_per_root)
                        noisy_map[nid] = noisy_val
                        node_registry[nid].value = noisy_val
        return noisy_map


# ---------------------------------------------------------------------------
# T2FunctionalProcessor（统一入口，支持外部 node_registry）
# ---------------------------------------------------------------------------
class T2FunctionalProcessor:
    def __init__(self, total_epsilon: float):
        self.total_epsilon = total_epsilon
        self.builder = FunctionalGraphBuilder()
        self.perturber = FunctionalGraphPerturber()

    def process_expressions(
        self,
        entities: List[Tuple[str, str]],
        expressions: List[Dict],
        node_registry: Optional[Dict[str, ValueNode]] = None,
    ) -> Tuple[Dict[str, Union[int, float]], Dict]:
        graphs, node_registry = self.builder.build_from_expressions(
            entities, expressions, self.total_epsilon,
            node_registry=node_registry
        )
        noisy_map = self.perturber.perturb_graphs(graphs, node_registry)
        graph_info = self._build_graph_info(graphs, noisy_map)
        return noisy_map, graph_info

    def _build_graph_info(self, graphs, noisy_map):
        return {
            "graphs": [
                {
                    "graph_id": g.graph_id,
                    "nodes": g.node_ids,
                    "roots": g.roots,
                    "epsilon": g.epsilon,
                    "relations": [
                        {
                            "relation_id": r.relation_id,
                            "inputs": r.input_node_ids,
                            "output": r.output_node_id,
                            "raw_expression": r.raw_expression,
                            "executable_expr": r.executable_expr,
                            "variable_mapping": r.variable_mapping,
                        }
                        for r in g.relations
                    ],
                }
                for g in graphs
            ],
            "noisy_map": noisy_map,
        }


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------
def _natural_sort_key(s: str):
    """自然排序键：x1, x2, x10 而非 x1, x10, x2"""
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', s)]


def _topo_sort_expressions(
    expressions: List[Dict],
    node_registry: Dict[str, ValueNode],
) -> List[Dict]:
    """
    拓扑排序表达式，确保被依赖的表达式的输出节点先于依赖它的表达式存在。
    若存在环，按原顺序返回并打印警告。
    """
    if len(expressions) <= 1:
        return list(expressions)

    # 构建表达式间的依赖图：expr_idx -> set of expr indices it depends on (via output_token)
    expr_outputs: Dict[str, int] = {}  # output_token -> expr_index
    for idx, expr in enumerate(expressions):
        out_token = expr.get("output_token", expr.get("token", ""))
        expr_outputs[out_token] = idx

    in_degree = [0] * len(expressions)
    children: Dict[int, List[int]] = defaultdict(list)  # parent_idx -> [child_idx]

    for idx, expr in enumerate(expressions):
        depends_on = expr.get("depends_on", [])
        refs = re.findall(r'\[([^\]]+)\]', expr.get("expression", ""))
        all_deps = list(set(depends_on) | set(refs))
        for dep_token in all_deps:
            if dep_token in expr_outputs and expr_outputs[dep_token] != idx:
                parent_idx = expr_outputs[dep_token]
                in_degree[idx] += 1
                children[parent_idx].append(idx)

    # Kahn 算法
    queue = deque(i for i, d in enumerate(in_degree) if d == 0)
    sorted_indices = []
    while queue:
        cur = queue.popleft()
        sorted_indices.append(cur)
        for child in children.get(cur, []):
            in_degree[child] -= 1
            if in_degree[child] == 0:
                queue.append(child)

    if len(sorted_indices) != len(expressions):
        print(f"警告：表达式间存在循环依赖，按原始顺序处理")
        return list(expressions)

    return [expressions[i] for i in sorted_indices]


# ---------------------------------------------------------------------------
# 中文数值词映射
# ---------------------------------------------------------------------------
_CN_DIGIT_MAP = {
    '零': 0, '一': 1, '二': 2, '两': 2, '三': 3, '四': 4,
    '五': 5, '六': 6, '七': 7, '八': 8, '九': 9, '十': 10,
    '半': 0.5,
}

_CN_DISCOUNT_PATTERN = re.compile(
    r'^([一二两三四五六七八九]|十)折$'
)

_CN_PERCENT_PATTERN = re.compile(
    r'^([一二两三四五六七八九]|十)成$'
)


def _parse_cn_numeric(token: str) -> Optional[float]:
    """尝试解析中文数值词，返回数值或 None"""
    # 匹配 "X折" → X/10
    m = _CN_DISCOUNT_PATTERN.match(token)
    if m:
        digit_cn = m.group(1)
        digit = _CN_DIGIT_MAP.get(digit_cn, 0)
        return digit / 10.0

    # 匹配 "X成" → X/10
    m = _CN_PERCENT_PATTERN.match(token)
    if m:
        digit_cn = m.group(1)
        digit = _CN_DIGIT_MAP.get(digit_cn, 0)
        return digit / 10.0

    # "半" → 0.5
    if token in ('半',):
        return 0.5

    # 纯中文数字 "一"→1, "二"→2, ...
    if token in _CN_DIGIT_MAP:
        return float(_CN_DIGIT_MAP[token])

    # "X天" → X, "X月" → X, "X年" → X
    m = re.match(r'^([一两二三四五六七八九十\d]+)\s*([天月年])$', token)
    if m:
        cn_or_num = m.group(1)
        if cn_or_num in _CN_DIGIT_MAP:
            return float(_CN_DIGIT_MAP[cn_or_num])
        try:
            return float(cn_or_num)
        except ValueError:
            pass

    # "千分之X" → X/1000（X 可以是数字或中文数字，如 千分之三 → 3/1000）
    m = re.match(r'^千分之([\d.]+|[一二两三四五六七八九]+)$', token)
    if m:
        num_str = m.group(1)
        if num_str in _CN_DIGIT_MAP:
            return float(_CN_DIGIT_MAP[num_str]) / 1000
        try:
            return float(num_str) / 1000
        except ValueError:
            return None

    # "万分之X" → X/10000
    m = re.match(r'^万分之([\d.]+|[一二两三四五六七八九]+)$', token)
    if m:
        num_str = m.group(1)
        if num_str in _CN_DIGIT_MAP:
            return float(_CN_DIGIT_MAP[num_str]) / 10000
        try:
            return float(num_str) / 10000
        except ValueError:
            return None

    # "X倍" → X
    m = re.match(r'^([\d.]+)倍$', token)
    if m:
        return float(m.group(1))

    return None


def _extract_numeric_value(token: str) -> Optional[float]:
    """
    从 token 中提取数值。
    若 token 不含任何可解析的数字，返回 None，避免对非数值 token 的误扰动。
    """
    # 1. 尝试中文数值
    cn_val = _parse_cn_numeric(token)
    if cn_val is not None:
        return cn_val

    # 2. 判断是否为百分比（% 或 ‰）
    is_percent = '%' in token or '％' in token
    is_permille = '‰' in token

    # 3. 去除百分号、逗号、中文后缀后解析数值
    stripped = token.replace("%", "").replace("％", "").replace("‰", "").replace(",", "").replace("，", "").strip()
    stripped = re.sub(r'[^\d.]+$', '', stripped)
    try:
        val = float(stripped)
        if val > 0:
            if is_percent:
                return val / 100.0   # "45%" → 0.45
            if is_permille:
                return val / 1000.0  # "3‰" → 0.003
            return val
    except ValueError:
        pass

    # 4. 仅当 token 以数字开头时，提取数值部分
    if re.match(r'^\d', token):
        nums = re.findall(r'[\d.]+', token)
        if nums:
            try:
                val = float(nums[0])
                if is_percent:
                    return val / 100.0
                if is_permille:
                    return val / 1000.0
                return val
            except ValueError:
                pass

    return None

def _remove_cycles_relations(relations, all_node_ids):
    adj = defaultdict(list)
    in_deg = {nid: 0 for nid in all_node_ids}
    for r in relations:
        for inp in r.input_node_ids:
            adj[inp].append(r.output_node_id)
            in_deg[r.output_node_id] = in_deg.get(r.output_node_id, 0) + 1
    queue = deque(nid for nid in all_node_ids if in_deg.get(nid, 0) == 0)
    topo = []
    while queue:
        n = queue.popleft()
        topo.append(n)
        for nb in adj[n]:
            in_deg[nb] -= 1
            if in_deg[nb] == 0:
                queue.append(nb)
    topo_set = set(topo)
    return [r for r in relations if all(inp in topo_set for inp in r.input_node_ids) and r.output_node_id in topo_set]


# 向后兼容别名
T2DAGProcessor = T2FunctionalProcessor
