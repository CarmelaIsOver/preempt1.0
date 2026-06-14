"""
实验指标计算
新增：MAE, RMSE, Constraint Violation Magnitude
"""

import numpy as np
from collections import defaultdict


def calc_mre(original_values: dict, noisy_values: dict, min_value: float = 1.0) -> float:
    """
    Mean Relative Error (%)
    MRE = mean(|noisy - orig| / |orig|) × 100%
    min_value: 跳过绝对值小于此阈值的值（防止小数值如折扣率0.2拉高MRE）
    """
    errors = []
    skipped = 0
    for key in original_values:
        orig = abs(original_values[key])
        if orig < min_value:
            skipped += 1
            continue
        if key not in noisy_values:
            continue
        errors.append(abs(noisy_values[key] - original_values[key]) / orig)
    if not errors:
        return 0.0
    return np.mean(errors) * 100


def calc_mae(original_values: dict, noisy_values: dict, min_value: float = 0.0) -> float:
    """Mean Absolute Error"""
    errors = []
    for key in original_values:
        orig = abs(original_values[key])
        if orig < min_value:
            continue
        if key not in noisy_values:
            continue
        errors.append(abs(noisy_values[key] - original_values[key]))
    if not errors:
        return 0.0
    return np.mean(errors)


def calc_rmse(original_values: dict, noisy_values: dict, min_value: float = 0.0) -> float:
    """Root Mean Square Error"""
    errors = []
    for key in original_values:
        orig = abs(original_values[key])
        if orig < min_value:
            continue
        if key not in noisy_values:
            continue
        errors.append((noisy_values[key] - original_values[key]) ** 2)
    if not errors:
        return 0.0
    return np.sqrt(np.mean(errors))


def calc_endpoint_error(original_values: dict, noisy_values: dict, min_value: float = 1.0) -> float:
    """最大相对误差 (%)"""
    errors = []
    for key in original_values:
        orig = abs(original_values[key])
        if orig < min_value:
            continue
        if key not in noisy_values:
            continue
        errors.append(abs(noisy_values[key] - original_values[key]) / orig)
    if not errors:
        return 0.0
    return max(errors) * 100


def check_constraint_satisfaction(expressions: list, noisy_values: dict) -> float:
    """
    验证 DAG 约束是否满足
    对每个表达式，使用 perturbed 的输入值重新计算，对比 perturbed 的输出值

    返回约束满足率 (%)
    """
    if not expressions:
        return 100.0

    satisfied = 0
    total = 0

    for expr in expressions:
        depends_on = expr.get("depends_on", [])
        expression = expr.get("expression", "")
        output_token = expr.get("output_token", expr.get("token", ""))

        # 收集扰动后的输入值
        input_vals = {}
        for dep in depends_on:
            if dep in noisy_values:
                input_vals[dep] = noisy_values[dep]

        if output_token not in noisy_values:
            continue

        perturbed_output = noisy_values[output_token]

        # 尝试计算约束值
        try:
            # 用原始表达式和扰动输入值计算期望输出
            exec_expr = expression
            # 按长度降序替换 [token]
            sorted_deps = sorted(input_vals.items(), key=lambda kv: len(kv[0]), reverse=True)
            for dep_token, val in sorted_deps:
                exec_expr = exec_expr.replace(f"[{dep_token}]", str(val))

            # 安全 eval
            allowed = {"sin": np.sin, "cos": np.cos, "tan": np.tan,
                       "log": np.log, "sqrt": np.sqrt, "exp": np.exp,
                       "abs": abs, "pow": pow, "pi": np.pi, "e": np.e,
                       "True": True, "False": False}
            expected = eval(exec_expr, {"__builtins__": {}}, allowed)
            expected = float(expected)

            total += 1
            if abs(expected - perturbed_output) < 1e-6:
                satisfied += 1
        except Exception:
            continue

    if total == 0:
        return 100.0
    return satisfied / total * 100


def calc_constraint_violation_magnitude(
    expressions: list, noisy_values: dict, min_expected: float = 1.0
) -> float:
    """
    约束违反幅度：对每个表达式，计算 |perturbed_output - recomputed_output| / |recomputed_output|
    取平均 (%)。跳过 expected 值过小的表达式。
    """
    if not expressions:
        return 0.0

    violations = []

    for expr in expressions:
        depends_on = expr.get("depends_on", [])
        expression = expr.get("expression", "")
        output_token = expr.get("output_token", expr.get("token", ""))

        input_vals = {}
        for dep in depends_on:
            if dep in noisy_values:
                input_vals[dep] = noisy_values[dep]

        if output_token not in noisy_values:
            continue

        try:
            exec_expr = expression
            sorted_deps = sorted(input_vals.items(), key=lambda kv: len(kv[0]), reverse=True)
            for dep_token, val in sorted_deps:
                exec_expr = exec_expr.replace(f"[{dep_token}]", str(val))

            allowed = {"sin": np.sin, "cos": np.cos, "tan": np.tan,
                       "log": np.log, "sqrt": np.sqrt, "exp": np.exp,
                       "abs": abs, "pow": pow, "pi": np.pi, "e": np.e,
                       "True": True, "False": False}
            expected = float(eval(exec_expr, {"__builtins__": {}}, allowed))

            if abs(expected) >= min_expected:
                violation = abs(noisy_values[output_token] - expected) / abs(expected)
                violations.append(violation)
        except Exception:
            continue

    if not violations:
        return 0.0
    return np.mean(violations) * 100
