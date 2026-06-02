"""
build_merged_npy.py
将 disease_merged.txt / drug_merged.txt / symptom_merged.txt 编码为 npy
同时生成全量 merged 词典（三个类别合并去重）
"""
import os
os.environ['HF_HOME'] = '/root/.cache/huggingface_local'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

import numpy as np
from sentence_transformers import SentenceTransformer

DICT_DIR = 'dict'

print("加载 ST...")
st = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2', local_files_only=True)

# ===== 1. 加载三个 merged txt =====
print("\n[1/4] 加载 merged txt...")
categories = {}
all_words_set = set()

for cat in ['disease', 'drug', 'symptom']:
    path = f'{DICT_DIR}/{cat}_merged.txt'
    if not os.path.exists(path):
        print(f"  ⚠️  {path} 不存在，跳过")
        continue
    with open(path, 'r', encoding='utf-8') as f:
        words = [line.strip() for line in f if line.strip()]
    words = [w for w in words if len(w) >= 2]
    categories[cat] = words
    all_words_set.update(words)
    print(f"  {cat}: {len(words)} 词")

# ===== 2. 生成全量词典 =====
all_words = sorted(all_words_set)
print(f"\n[2/4] 全量 merged 词典: {len(all_words)} 词")

# ===== 3. 编码全量词典 =====
print(f"\n[3/4] 编码全量词典 ({len(all_words)} 词)...")
all_vecs = st.encode(all_words, batch_size=64, show_progress_bar=True)

np.save(f'{DICT_DIR}/medical_words.npy', np.array(all_words))
np.save(f'{DICT_DIR}/medical_vecs.npy', all_vecs)
print(f"✅ 保存: medical_words.npy + medical_vecs.npy ({len(all_words)}词)")

# ===== 4. 分别编码各类别 =====
print(f"\n[4/4] 编码分类词典...")
word_to_idx = {w: i for i, w in enumerate(all_words)}

for cat, words in categories.items():
    indices = [word_to_idx[w] for w in words if w in word_to_idx]
    cat_vecs = all_vecs[indices]
    np.save(f'{DICT_DIR}/{cat}_words.npy', np.array(words))
    np.save(f'{DICT_DIR}/{cat}_vecs.npy', cat_vecs)
    print(f"  ✅ {cat}: {len(words)}词")

print("\n全部完成。")
