"""
t3_module.py  —  T3 医疗实体扰动模块

通过调用 perturb_word 中的 ST+FT MLDP 服务（端口 9999）对医疗词汇进行差分扰动。
同时支持 BCV 类比推理服务（端口 9997）进行语义偏移。

T3 子标签:
  MEDICINE, DISEASE, SYMPTOM, THERAPY, FORBIDDEN_FOOD,
  SIDE_EFFECT, GENE, ORGAN, FUNCTION
"""

import json
import socket
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Optional


_LABEL_FALLBACK_TERMS = {
    "MEDICINE": [
        "阿司匹林", "布洛芬", "二甲双胍", "硝苯地平", "硝苯地平控释片",
        "头孢曲松", "奥美拉唑", "青霉素", "胰岛素", "沙丁胺醇",
        "氨氯地平", "美托洛尔", "氯吡格雷",
    ],
    "DISEASE": [
        "肺炎", "高血压", "高血压病", "糖尿病", "哮喘", "冠心病",
        "胃炎", "肝炎", "肾炎", "感冒", "感染", "脑梗死", "心力衰竭",
    ],
    "SYMPTOM": [
        "咳嗽", "发热", "头痛", "头晕", "恶心", "呕吐", "腹痛",
        "胸痛", "乏力", "气促", "呼吸困难", "咳痰", "皮疹", "水肿",
    ],
    "THERAPY": [
        "治疗", "抗感染治疗", "放疗", "化疗", "手术", "静脉滴注",
        "输液", "透析", "康复", "介入治疗", "免疫治疗",
    ],
    "FORBIDDEN_FOOD": [
        "海鲜", "酒精", "辛辣食物", "高糖食物", "油腻食物", "甜食",
        "咖啡", "浓茶",
    ],
    "SIDE_EFFECT": [
        "恶心", "呕吐", "腹泻", "皮疹", "头晕", "嗜睡", "肝损伤",
        "肾损伤", "过敏", "胃肠道反应",
    ],
    "GENE": ["BRCA1", "BRCA2", "EGFR", "TP53", "ALK", "KRAS", "HER2"],
    "ORGAN": [
        "肺", "右肺", "左肺", "胸部", "心脏", "肝脏", "肾脏", "胃",
        "脑", "血液", "白细胞", "血管", "胰腺", "甲状腺",
    ],
    "FUNCTION": [
        "免疫", "代谢", "凝血", "呼吸", "循环", "消化", "吸收",
        "排泄", "白细胞计数", "血常规", "体温", "血压", "血糖",
    ],
}


def _normalize_label(label: str) -> str:
    key = "".join(ch if ch.isalnum() else "_" for ch in str(label or "").strip())
    key = key.strip("_").upper()
    if key.startswith("T3_"):
        key = key[3:]
    aliases = {
        "SYSMPTOM": "SYMPTOM",
        "FORBIDDENFOOD": "FORBIDDEN_FOOD",
        "SIDE_EFFECT": "SIDE_EFFECT",
        "SIDEEFFECT": "SIDE_EFFECT",
    }
    return aliases.get(key, aliases.get(key.replace("_", ""), key))


def _recv_all(sock: socket.socket) -> bytes:
    data = b''
    while True:
        chunk = sock.recv(4096)
        if not chunk:
            break
        data += chunk
    return data


class T3Perturber:
    """
    T3 医疗实体扰动器。

    策略：优先使用 MLDP（端口 9999）对单个医疗词进行差分扰动；
    若 MLDP 返回空或不可用，回退到保留原词。
    """

    def __init__(
        self,
        host: str = '127.0.0.1',
        port_mlpd: int = 9999,
        port_mldp: Optional[int] = None,
        enable_local_fallback: bool = True,
    ):
        self.host = host
        if port_mldp is not None:
            port_mlpd = port_mldp
        self.port_mlpd = port_mlpd
        self.enable_local_fallback = enable_local_fallback
        self._service_failed = False
        self._fallback_words: Optional[List[str]] = None

    def perturb_mlpd(
        self,
        word: str,
        epsilon: float = 5.0,
        domain: str = 'medical',
        threshold_lower: float = 0.3,
        threshold_upper: float = 0.95,
    ) -> Optional[Dict]:
        """
        调用 ST+FT MLDP 服务扰动单个医疗词汇。

        返回 {'original', 'perturbed', 'similarity', 'epsilon'} 或 None。
        """
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(15)
            s.connect((self.host, self.port_mlpd))
            req = json.dumps({
                'word': word,
                'domain': domain,
                'epsilon': epsilon,
                'threshold_lower': threshold_lower,
                'threshold_upper': threshold_upper,
            })
            s.sendall(req.encode('utf-8'))
            s.shutdown(socket.SHUT_WR)
            data = _recv_all(s)
            s.close()

            result = json.loads(data)
            if 'error' in result or not result.get('candidates'):
                return None

            top = result['candidates'][0]
            return {
                'original': word,
                'perturbed': top['new_word'],
                'similarity': top.get('st_sim', 0),
                'epsilon': epsilon,
            }
        except Exception as e:
            print(f"[T3] MLDP perturbation failed for '{word}': {e}")
            self._service_failed = True
            return None

    def _load_fallback_words(self) -> List[str]:
        if self._fallback_words is not None:
            return self._fallback_words

        dict_dir = Path(__file__).resolve().parent / "perturb_word" / "dict"
        words: List[str] = []

        npy_path = dict_dir / "medical_words.npy"
        if npy_path.exists():
            try:
                import numpy as np

                words = [str(w) for w in np.load(npy_path, allow_pickle=True)]
            except Exception as e:
                print(f"[T3] Failed to load fallback npy dictionary: {e}")

        if not words:
            txt_path = dict_dir / "THUOCL_medical.txt"
            if txt_path.exists():
                try:
                    with open(txt_path, "r", encoding="utf-8") as f:
                        words = [
                            line.strip().split("\t")[0]
                            for line in f
                            if line.strip()
                        ]
                except Exception as e:
                    print(f"[T3] Failed to load fallback txt dictionary: {e}")

        seen = set()
        self._fallback_words = [
            w for w in words
            if w and not (w in seen or seen.add(w))
        ]
        return self._fallback_words

    def perturb_local(
        self,
        word: str,
        label: str,
        epsilon: float = 5.0,
        top_k: int = 50,
    ) -> Optional[Dict]:
        label = _normalize_label(label)
        label_words = _LABEL_FALLBACK_TERMS.get(label, [])
        words = label_words or self._load_fallback_words()
        if not words:
            return None

        scored = []
        max_len_delta = max(3, len(word))
        for candidate in words:
            if candidate == word or abs(len(candidate) - len(word)) > max_len_delta:
                continue
            sim = SequenceMatcher(None, word, candidate).ratio()
            if 0.15 <= sim < 0.98:
                scored.append((sim, candidate))

        if not scored and label_words:
            scored = [
                (SequenceMatcher(None, word, candidate).ratio(), candidate)
                for candidate in label_words
                if candidate != word
            ]

        if not scored:
            scored = [
                (0.0, candidate)
                for candidate in words
                if candidate != word
            ]

        if not scored:
            return None

        scored.sort(key=lambda item: (-item[0], abs(len(item[1]) - len(word)), item[1]))
        candidate_limit = 5 if label_words else top_k
        pool = scored[:max(1, min(candidate_limit, len(scored)))]
        chosen_sim, chosen_word = pool[0]

        return {
            'original': word,
            'perturbed': chosen_word,
            'similarity': chosen_sim,
            'epsilon': epsilon,
            'label': label,
        }

    def perturb(
        self,
        word: str,
        label: str,
        epsilon: float = 5.0,
    ) -> Dict:
        """
        对单个 t3 医疗实体执行扰动。

        参数
        ----
        word  : 原始医疗词汇
        label : t3 子标签（MEDICINE/DISEASE/SYMPTOM/...）
        epsilon : 隐私预算

        返回
        ----
        {'original', 'perturbed', 'similarity', 'label', 'epsilon'}
        如果扰动失败，perturbed == original，similarity == 1.0
        """
        label = _normalize_label(label)
        result = None
        if not self._service_failed:
            result = self.perturb_mlpd(word, epsilon=epsilon)
        if result is None and self.enable_local_fallback:
            result = self.perturb_local(word, label, epsilon=epsilon)
        if result is None:
            return {
                'original': word,
                'perturbed': word,
                'similarity': 1.0,
                'label': label,
                'epsilon': epsilon,
            }
        result['label'] = label
        return result

    def perturb_batch(
        self,
        items: List[tuple],
        epsilon: float = 5.0,
    ) -> Dict[str, Dict]:
        """
        批量扰动。

        参数
        ----
        items   : [(word, label), ...]
        epsilon : 隐私预算

        返回
        ----
        {original_word: {perturbed, similarity, label, epsilon}, ...}
        """
        results = {}
        for word, label in items:
            results[word] = self.perturb(word, label, epsilon)
        return results


# ---------------------------------------------------------------------------
# 自测
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    perturber = T3Perturber()

    test_words = [
        ("阿司匹林", "MEDICINE"),
        ("肺炎", "DISEASE"),
        ("咳嗽", "SYMPTOM"),
        ("放疗", "THERAPY"),
        ("海鲜", "FORBIDDEN_FOOD"),
        ("恶心", "SIDE_EFFECT"),
        ("BRCA1", "GENE"),
        ("肝脏", "ORGAN"),
        ("代谢", "FUNCTION"),
    ]

    print("=" * 60)
    print("T3 医疗实体扰动测试（需先启动 model_server.py :9999）")
    print("=" * 60)

    for word, label in test_words:
        result = perturber.perturb(word, label, epsilon=5.0)
        if result['perturbed'] != word:
            print(f"[{label}] {word} → {result['perturbed']} (sim={result['similarity']:.3f})")
        else:
            print(f"[{label}] {word} → (扰动失败，保留原词)")
