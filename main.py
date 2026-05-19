"""
main.py  —  医疗提示词三标签脱敏系统

调用 Sanitizer 对医疗提示词文本进行脱敏处理。

三标签体系
----------
t1 : 个人身份信息
     - t1_name (PERSON) → 随机假名替换
     - t1_fpe  (其他)    → FF3 格式保持加密
t2 : 数值型实体
     - mLDP 局部差分隐私扰动 + DAG 关系保持
t3 : 医疗实体（新增）
     - MEDICINE / DISEASE / SYMPTOM / THERAPY
     - FORBIDDEN_FOOD / SIDE_EFFECT / GENE / ORGAN / FUNCTION
     - 通过 perturb_word 的 ST+FT MLDP 服务进行语义扰动
"""

import json
import os
from typing import List, Dict

from sanitizer_module import Sanitizer


def load_texts_from_file(file_path: str) -> List[str]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")

    with open(file_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    if not lines:
        raise ValueError(f"文件为空: {file_path}")

    return lines


def save_results(
        original_texts: List[str],
        sanitized_texts: List[str],
        session_infos: List[Dict],
        output_dir: str = "./output"
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    with open(f"{output_dir}/sanitized_output.txt", 'w', encoding='utf-8') as f:
        for i, text in enumerate(sanitized_texts):
            f.write(f"[{i + 1}] {text}\n")
            f.write("-" * 50 + "\n")

    with open(f"{output_dir}/comparison.txt", 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("原始文本 vs 脱敏文本 对照表\n")
        f.write("=" * 80 + "\n\n")

        for i, (orig, san) in enumerate(zip(original_texts, sanitized_texts)):
            f.write(f"\n[{i + 1}] 原始文本:\n")
            f.write(f"{orig}\n\n")
            f.write(f"[{i + 1}] 脱敏文本:\n")
            f.write(f"{san}\n")
            f.write("-" * 80 + "\n")

    simplified_infos = []
    for info in session_infos:
        simplified = {
            "eps": info.get("eps_t2"),
            "ner_result": info.get("ner_result"),
            "dag_info": info.get("dag_info"),
            "vault_keys": list(info.get("vault", {}).keys()) if info.get("vault") else [],
        }
        simplified_infos.append(simplified)

    with open(f"{output_dir}/session_infos.json", 'w', encoding='utf-8') as f:
        json.dump(simplified_infos, f, ensure_ascii=False, indent=2)

    print(f"\n结果已保存到: {output_dir}/")
    print(f"  - sanitized_output.txt : 脱敏后文本")
    print(f"  - comparison.txt       : 原始/脱敏对照")
    print(f"  - session_infos.json   : 会话信息（简化版）")


def run_sanitizer_demo(
        file_path: str = "data.txt",
        epsilon: float = 1.0,
        output_dir: str = "./output"
) -> None:
    print(f"正在读取文件: {file_path}")
    texts = load_texts_from_file(file_path)
    print(f"共读取 {len(texts)} 条文本\n")

    print("初始化 Sanitizer...")
    print(f"隐私预算 epsilon = {epsilon}\n")
    sanitizer = Sanitizer(epsilon=epsilon)

    sanitized_texts = []
    session_infos = []

    for idx, text in enumerate(texts):
        print(f"\n{'=' * 60}")
        print(f"处理第 {idx + 1}/{len(texts)} 条:")
        print(f"原始: {text}")

        try:
            san_text, session_info = sanitizer.sanitizer(text)
            print(f"脱敏: {san_text}")

            sanitized_texts.append(san_text)
            session_infos.append(session_info)

            vault = session_info.get("vault", {})
            for enc, rec in vault.items():
                if rec["type"] == "t1_name":
                    print(f"  [t1_name] {enc!r} → {rec['original']!r}")
                elif rec["type"] == "t1_fpe":
                    print(f"  [t1_fpe]  {enc!r} → {rec['original']!r}")
                elif rec["type"] == "t2_perturb":
                    print(f"  [t2]      {rec['original']!r} → {rec['noisy_val']} (扰动)")
                elif rec["type"] == "t3_perturb":
                    print(f"  [t3]      [{rec['label']}] {rec['original']!r} → {enc!r} (sim={rec['similarity']:.3f})")

        except Exception as e:
            print(f"处理失败: {e}")
            sanitized_texts.append(f"[ERROR: {e}]")
            session_infos.append({})

    save_results(texts, sanitized_texts, session_infos, output_dir)

    # 反脱敏演示（只对第一条）
    if sanitized_texts and session_infos[0]:
        print(f"\n{'=' * 60}")
        print("反脱敏演示（还原 t1 和 t3 实体）:")

        restored, audit = sanitizer.desanitizer(sanitized_texts[0], session_infos[0])
        print(f"原始文本:     {texts[0]}")
        print(f"脱敏文本:     {sanitized_texts[0]}")
        print(f"反脱敏后:     {restored}")
        print(f"恢复率:       {audit['recovery_rate']}")
        if audit.get('missing_t1'):
            print(f"未出现的实体: {audit['missing_t1']}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="医疗提示词三标签脱敏工具")
    parser.add_argument(
        "--input", "-i",
        type=str,
        default="data.txt",
        help="输入文件路径（默认: data.txt）"
    )
    parser.add_argument(
        "--epsilon", "-e",
        type=float,
        default=1.0,
        help="隐私预算 epsilon，越小保护越强（默认: 1.0）"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="./output",
        help="输出目录（默认: ./output）"
    )

    args = parser.parse_args()

    run_sanitizer_demo(
        file_path=args.input,
        epsilon=args.epsilon,
        output_dir=args.output
    )


if __name__ == "__main__":
    main()
