"""ST+FT MLDP 扰动服务（指数机制版本 + 分类词典 + ST类比推理 + 领域约束）"""
import os
os.environ['HF_HOME'] = '/root/.cache/huggingface_local'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

import json
import socket
import sys
import time
import numpy as np

print("加载ST...")
from sentence_transformers import SentenceTransformer
st = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')

print("加载FT...")
import fasttext
ft = fasttext.load_model('cc.zh.300.bin')

print("加载分类词典...")
DICT_DIR = 'dict'
typed_dicts = {}
for domain in ['disease', 'drug', 'symptom', 'medical']:
    wpath = f'{DICT_DIR}/{domain}_words.npy'
    vpath = f'{DICT_DIR}/{domain}_vecs.npy'
    if os.path.exists(wpath) and os.path.exists(vpath):
        words = np.load(wpath, allow_pickle=True)
        vecs = np.load(vpath)
        typed_dicts[domain] = (words, vecs)
        print(f"  {domain}: {len(words)}词")
    else:
        print(f"  ⚠️ {domain}: 文件不存在，跳过")

if not typed_dicts:
    print("❌ 没有可用词典，退出")
    sys.exit(1)

print("READY")


def recv_all(conn):
    data = b''
    while True:
        chunk = conn.recv(4096)
        if not chunk:
            break
        data += chunk
    return data.decode('utf-8')


def exponential_mechanism(word, epsilon, domain='medical', k=20,
                          analogy_orig=None, analogy_pert=None):
    """
    指数机制扰动
    如果提供 analogy_orig 和 analogy_pert，使用ST类比推理模式
    疾病领域自动加领域约束（与原词ST相似度>0.55）
    """
    t_total = time.time()

    if domain not in typed_dicts:
        domain = 'medical'

    words, vecs = typed_dicts[domain]

    # Step 1: 编码
    st_child = st.encode(word)

    # Step 2: 构建查询向量
    if analogy_orig and analogy_pert:
        # ===== ST类比推理模式 =====
        st_orig = st.encode(analogy_orig)
        st_pert = st.encode(analogy_pert)
        delta = st_child - st_orig
        query_vec = st_pert + delta
        print(f"  [ST类比] {word} - {analogy_orig} + {analogy_pert}",
              file=sys.stderr, flush=True)
    else:
        # ===== 直接相似度模式 =====
        query_vec = st_child

    # Step 3: 计算相似度
    sims = np.dot(vecs, query_vec) / (
        np.linalg.norm(vecs, axis=1) * np.linalg.norm(query_vec) + 1e-8
    )

    # Step 4: 排除原词
    word_list = list(words)
    if word in word_list:
        sims[word_list.index(word)] = -1.0

    # Step 5: 过滤 + 领域约束 + 均匀取样
    sorted_indices = np.argsort(-sims)
    valid_mask = sims[sorted_indices] > -0.5
    sorted_indices = sorted_indices[valid_mask]

    # 词典内相似度过滤
    min_sim = 0.5 if domain == 'disease' else 0.2
    sim_mask = sims[sorted_indices] > min_sim
    sorted_indices = sorted_indices[sim_mask]

    # ===== 疾病领域约束：防止跨领域跳跃 =====
    if domain == 'disease':
        sims_to_orig = np.dot(vecs[sorted_indices], st_child) / (
            np.linalg.norm(vecs[sorted_indices], axis=1) * np.linalg.norm(st_child) + 1e-8
        )
        orig_mask = sims_to_orig > 0.55
        sorted_indices = sorted_indices[orig_mask]
        print(f"  [领域约束] 保留{len(sorted_indices)}个同领域候选",
              file=sys.stderr, flush=True)

    n_valid = len(sorted_indices)

    # fallback：候选太少时放宽
    if n_valid < 5:
        sorted_indices = np.argsort(-sims)
        sorted_indices = sorted_indices[sims[sorted_indices] > -0.5]
        sorted_indices = sorted_indices[sims[sorted_indices] > 0.3]
        n_valid = len(sorted_indices)

    step = max(1, n_valid // k)
    selected_indices = sorted_indices[::step][:k]

    valid_sims = sims[selected_indices]
    valid_words = words[selected_indices]
    n_candidates = len(valid_words)

    print(f"  [候选池] domain={domain}, {n_candidates}词, "
          f"sim范围[{valid_sims[-1]:.3f}, {valid_sims[0]:.3f}]",
          file=sys.stderr, flush=True)

    # Step 6: 指数机制采样
    sensitivity = 0.5

    log_probs = epsilon * valid_sims / (2 * sensitivity)
    log_probs -= np.max(log_probs)
    probs = np.exp(log_probs)
    probs = probs / probs.sum()

    max_prob = float(np.max(probs))
    prob_entropy = float(-np.sum(probs * np.log(probs + 1e-10)))
    max_entropy = np.log(n_candidates) if n_candidates > 1 else 1.0
    norm_entropy = prob_entropy / max_entropy if max_entropy > 0 else 0.0

    chosen_local_idx = np.random.choice(n_candidates, p=probs)
    chosen_word = str(valid_words[chosen_local_idx])
    chosen_sim = float(valid_sims[chosen_local_idx])

    result = [{'new_word': chosen_word, 'st_sim': chosen_sim}]
    for idx in np.argsort(-valid_sims):
        w = str(valid_words[idx])
        if w != chosen_word:
            result.append({'new_word': w, 'st_sim': float(valid_sims[idx])})

    elapsed = time.time() - t_total
    mode = "ST类比" if analogy_orig else "直接"
    print(f"  [完成] {mode} ε={epsilon}, chosen={chosen_word} "
          f"(sim={chosen_sim:.3f}), 熵={norm_entropy:.3f}, "
          f"{elapsed:.2f}s", file=sys.stderr, flush=True)

    return result


# ===== 服务主循环 =====
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server.bind(('127.0.0.1', 9999))
server.listen(5)
print("model_server 监听 9999...", flush=True)

while True:
    conn, addr = server.accept()
    try:
        data = recv_all(conn)
        req = json.loads(data)
        word = req['word']
        epsilon = req.get('epsilon', 5.0)
        domain = req.get('domain', 'disease')
        analogy_orig = req.get('analogy_orig', None)
        analogy_pert = req.get('analogy_pert', None)

        print(f"\n查询: {word} (ε={epsilon}, domain={domain})", file=sys.stderr, flush=True)
        result = {
            'word': word,
            'epsilon': epsilon,
            'domain': domain,
            'candidates': exponential_mechanism(
                word, epsilon, domain=domain, k=20,
                analogy_orig=analogy_orig,
                analogy_pert=analogy_pert
            )
        }
        conn.sendall(json.dumps(result, ensure_ascii=False).encode('utf-8'))
    except Exception as e:
        conn.sendall(json.dumps({'error': str(e)}).encode('utf-8'))
    finally:
        conn.close()