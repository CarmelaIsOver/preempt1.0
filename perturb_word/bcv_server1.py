"""BioConceptVec 服务 - 一步到位 + 方向约束 + 均匀取样 + 指数机制采样"""
import os
import sys
import socket
import json
import re
import time
import numpy as np
from gensim.models import KeyedVectors
from openai import OpenAI

LLM_API_KEY = "821a645b91c04ba4ab1b2db01b050f32.Zfw4xlprX362QxIJ"
llm = OpenAI(api_key=LLM_API_KEY, base_url="https://open.bigmodel.cn/api/paas/v4")

print("加载 BioConceptVec...")
bv = KeyedVectors.load_word2vec_format(
    os.path.expanduser('~/bioconceptvec_word2vec_skipgram.bin'),
    binary=True
)
print("READY")


def is_valid_bcv_word(en_word):
    """过滤BCV英文词表中的噪音词"""
    if not en_word:
        return False
    if '=' in en_word:
        return False
    if '/' in en_word:
        return False
    if re.search(r'\d', en_word):
        return False
    if en_word[0].isdigit():
        return False
    if re.search(r'[a-z]\.[a-z]', en_word):
        return False
    if len(en_word) < 3:
        return False
    if en_word.isupper() and len(en_word) > 5:
        return False
    return True


def is_valid_medical_term(zh_word):
    """过滤翻译后的中文噪音词"""
    if not zh_word:
        return False
    if re.search(r'[A-Za-z]_?\d{3,}', zh_word):
        return False
    if re.search(r'\d', zh_word):
        return False
    if '网状' in zh_word or '化学网' in zh_word:
        return False
    if '无法翻译' in zh_word or '诱导的' in zh_word:
        return False
    if '/' in zh_word:
        return False
    if len(zh_word) > 20:
        return False
    return True


def map_to_bcv(zh_word):
    """中文→多候选英文→BCV词表验证"""
    try:
        resp = llm.chat.completions.create(
            model="glm-4-flash",
            messages=[{"role": "user", "content": f"""将中文医学术语"{zh_word}"翻译成英文。
要求：
1. 输出6个候选翻译，覆盖通用名、商品名、缩写、同义词，每行一个，只输出英文
2. 示例：
   硝苯地平：
   nifedipine
   adalat
   nifedipine tablet
   calcium channel blocker
   ccb
   dihydropyridine
3. 术语：{zh_word}"""}],
            temperature=0.1,
            max_tokens=128
        )
        candidates = resp.choices[0].message.content.strip().split('\n')
        candidates = [c.strip().lower() for c in candidates if c.strip()]

        for c in candidates:
            if c in bv:
                return c
            if c.replace(' ', '_') in bv:
                return c.replace(' ', '_')

        for c in candidates:
            first_word = c.split()[0]
            if len(first_word) < 4:
                continue
            matches = [w for w in bv.key_to_index
                       if w.lower().startswith(first_word)
                       and len(w) < 30
                       and is_valid_bcv_word(w)
                       and w.replace('-', '').replace('_', '').isalpha()]
            if matches:
                return min(matches, key=len)

        return None
    except Exception as e:
        print(f"  [LLM映射失败] {zh_word}: {e}", file=sys.stderr, flush=True)
        return None


def batch_translate_en_zh(words):
    try:
        text = "\n".join(words)
        resp = llm.chat.completions.create(
            model="glm-4-flash",
            messages=[{"role": "user", "content": f"将以下医学术语逐行翻译成中文，每行只输出中文翻译：\n{text}"}],
            temperature=0.1,
            max_tokens=512
        )
        lines = resp.choices[0].message.content.strip().split('\n')
        return [l.strip() for l in lines if l.strip()]
    except Exception as e:
        print(f"  [批量翻译失败]: {e}", file=sys.stderr, flush=True)
        return []


def analogical_perturb(root_orig_zh, root_pert_zh, child_zh,
                       top_k=30, epsilon=None, uniform_sample=True,
                       alpha=0.6):
    """
    BCV类比推理 + 方向约束

    参数:
        alpha: 余弦相似度权重（1-alpha为方向一致性权重）
    """
    t_start = time.time()

    root_orig_en = map_to_bcv(root_orig_zh)
    root_pert_en = map_to_bcv(root_pert_zh)
    child_en = map_to_bcv(child_zh)

    failed = []
    if not root_orig_en: failed.append(root_orig_zh)
    if not root_pert_en: failed.append(root_pert_zh)
    if not child_en: failed.append(child_zh)
    if failed:
        return {'error': '映射失败', 'failed': failed}

    print(f"  [映射] {root_orig_zh}→{root_orig_en}, "
          f"{root_pert_zh}→{root_pert_en}, "
          f"{child_zh}→{child_en}", file=sys.stderr, flush=True)

    # ===== 类比推理 =====
    v_root_orig = bv[root_orig_en]
    v_root_pert = bv[root_pert_en]
    v_child = bv[child_en]
    delta = v_child - v_root_orig
    v_final = v_root_pert + delta

    # ===== 搜索候选 =====
    search_k = top_k * 8
    neighbors = bv.similar_by_vector(v_final, topn=search_k)

    # 过滤BCV噪音词
    neighbors = [(w, s) for w, s in neighbors if is_valid_bcv_word(w)]
    print(f"  [BCV过滤后] {len(neighbors)}个有效候选", file=sys.stderr, flush=True)

    if not neighbors:
        return {'error': 'BCV过滤后无有效候选'}

    # ===== 方向约束评分 =====
    delta_norm = delta / (np.linalg.norm(delta) + 1e-8)

    scored_neighbors = []
    for word, sim in neighbors:
        v_candidate = bv[word]
        candidate_offset = v_candidate - v_root_pert
        offset_norm = candidate_offset / (np.linalg.norm(candidate_offset) + 1e-8)
        direction_score = float(np.dot(offset_norm, delta_norm))
        # 综合得分：余弦相似度 + 方向一致性
        combined = alpha * sim + (1 - alpha) * direction_score
        scored_neighbors.append((word, sim, combined, direction_score))

    # 按综合得分排序
    scored_neighbors.sort(key=lambda x: -x[2])
    print(f"  [方向约束] Top3方向得分: "
          f"{[f'{w}(dir={d:.3f})' for w, _, _, d in scored_neighbors[:3]]}",
          file=sys.stderr, flush=True)

    # ===== 均匀取样 =====
    if uniform_sample and len(scored_neighbors) > top_k:
        combined_scores = np.array([c for _, _, c, _ in scored_neighbors])
        sorted_idx = np.argsort(-combined_scores)
        step = max(1, len(sorted_idx) // top_k)
        selected_idx = sorted_idx[::step][:top_k]
        scored_neighbors = [scored_neighbors[i] for i in selected_idx]

    # ===== 批量翻译 =====
    en_words = [w for w, _, _, _ in scored_neighbors]
    zh_words = batch_translate_en_zh(en_words)

    # ===== 过滤中文翻译 =====
    raw_results = []
    for i, (word, sim, combined, dir_score) in enumerate(scored_neighbors):
        zh = zh_words[i] if i < len(zh_words) else word
        if is_valid_medical_term(zh):
            raw_results.append({
                'bcv_word': word,
                'zh_word': zh,
                'similarity': float(sim),
                'combined_score': float(combined),
                'direction_score': float(dir_score)
            })

    if len(raw_results) < 3:
        raw_results = []
        for i, (word, sim, combined, dir_score) in enumerate(scored_neighbors[:10]):
            zh = zh_words[i] if i < len(zh_words) else word
            raw_results.append({
                'bcv_word': word,
                'zh_word': zh,
                'similarity': float(sim),
                'combined_score': float(combined),
                'direction_score': float(dir_score)
            })

    # ===== 指数机制采样 =====
    sampled = False
    if epsilon is not None and epsilon > 0 and len(raw_results) > 1:
        combined_scores = np.array([r['combined_score'] for r in raw_results])
        sensitivity = (float(np.max(combined_scores)) - float(np.min(combined_scores))) / 2 + 1e-8
        scores = np.exp(epsilon * combined_scores / (2 * sensitivity))
        probs = scores / np.sum(scores)
        idx = np.random.choice(len(raw_results), p=probs)
        chosen = raw_results.pop(idx)
        raw_results.insert(0, chosen)
        sampled = True

    elapsed = time.time() - t_start
    print(f"  [完成] 候选={len(raw_results)}, "
          f"chosen={raw_results[0]['zh_word']} "
          f"(sim={raw_results[0]['similarity']:.3f}, "
          f"dir={raw_results[0]['direction_score']:.3f}), "
          f"{elapsed:.1f}s", file=sys.stderr, flush=True)

    return {
        'mapping': {root_orig_zh: root_orig_en, root_pert_zh: root_pert_en, child_zh: child_en},
        'results': raw_results,
        'sampled': sampled,
        'epsilon': epsilon if sampled else None,
        'uniform_sample': uniform_sample,
        'alpha': alpha
    }


def recv_all(conn):
    data = b''
    while True:
        chunk = conn.recv(4096)
        if not chunk: break
        data += chunk
    return data.decode('utf-8')


server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server.bind(('127.0.0.1', 9996))
server.listen(5)
print("bcv_server 监听 9996...", flush=True)

while True:
    conn, addr = server.accept()
    try:
        data = recv_all(conn)
        req = json.loads(data)
        result = analogical_perturb(
            root_orig_zh=req['root_orig'],
            root_pert_zh=req['root_pert'],
            child_zh=req['child_orig'],
            top_k=req.get('top_k', 30),
            epsilon=req.get('epsilon', None),
            uniform_sample=req.get('uniform_sample', True),
            alpha=req.get('alpha', 0.6)
        )
        conn.sendall(json.dumps(result, ensure_ascii=False).encode('utf-8'))
    except Exception as e:
        conn.sendall(json.dumps({'error': str(e)}).encode('utf-8'))
    finally:
        conn.close()
