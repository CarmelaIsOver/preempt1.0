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

# 不允许小数的实体标签（扰动后必须整数化）
INTEGER_ONLY_LABELS = {
    "AGE",            # 年龄：58岁，不能是58.3岁
    "COUNT",          # 数量：5件，不能是5.2件
    "YEAR",           # 年份/年数：5年
    "DATE_NUM",       # 日期数字
    "BLOOD_PRESSURE", # 血压：130，不能是130.5
}


def is_t1(label: str) -> bool:
    return label in T1_LABELS or label.lower().startswith("t1")


def is_t1_name(label: str) -> bool:
    """判断是否需要姓名替换（只有 PERSON）"""
    return label in T1_NAME_LABELS


def is_t1_fpe(label: str) -> bool:
    """判断是否需要 FPE 加密（除了 PERSON 的其他 t1 标签）"""
    return is_t1(label) and not is_t1_name(label)


def is_t2(label: str) -> bool:
    return label in T2_LABELS or label.lower().startswith("t2")
