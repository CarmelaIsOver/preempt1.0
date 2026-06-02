"""BCV 向量结构测试服务 - 监听模式"""
import os
import sys
import socket
import json
import numpy as np
from gensim.models import KeyedVectors

print("加载 BioConceptVec...")
bv = KeyedVectors.load_word2vec_format(
    os.path.expanduser('~/bioconceptvec_word2vec_skipgram.bin'),
    binary=True
)
print("READY")


def show_neighbors(word, topn=20):
    """展示某个词的最近邻"""
    if word not in bv:
        matches = [w for w in bv.key_to_index if word.lower() in w.lower()]
        if matches:
            return {'error': f'不在词表，近似: {matches[:10]}'}
        return {'error': f'不在词表'}

    v_word = bv[word]
    neighbors = bv.similar_by_vector(v_word, topn=topn)

    results = []
    for w, sim in neighbors:
        v_w = bv[w]
        angle = np.arccos(np.clip(sim, -1, 1)) * 180 / np.pi
        results.append({
            'rank': len(results) + 1,
            'word': w,
            'similarity': round(float(sim), 4),
            'norm': round(float(np.linalg.norm(v_w)), 4),
            'angle_deg': round(float(angle), 2)
        })

    return {
        'word': word,
        'norm': round(float(np.linalg.norm(v_word)), 4),
        'neighbors': results
    }


def show_delta(word1, word2):
    """分析两个词的关系向量"""
    if word1 not in bv or word2 not in bv:
        return {'error': '词不在词表'}

    v1 = bv[word1]
    v2 = bv[word2]
    delta = v2 - v1

    delta_neighbors = bv.similar_by_vector(delta, topn=10)

    return {
        'word1': word1, 'word2': word2,
        'cosine_sim': round(float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)), 4),
        'euclidean_dist': round(float(np.linalg.norm(delta)), 4),
        'delta_norm': round(float(np.linalg.norm(delta)), 4),
        'delta_neighbors': [(w, round(float(s), 4)) for w, s in delta_neighbors]
    }


def show_analogy(disease, drug, new_disease, topn=15):
    """模拟BCV类比推理"""
    if disease not in bv or drug not in bv or new_disease not in bv:
        return {'error': '有词不在词表'}

    v_disease = bv[disease]
    v_drug = bv[drug]
    v_new = bv[new_disease]

    delta = v_drug - v_disease
    v_result = v_new + delta
    delta_norm = delta / (np.linalg.norm(delta) + 1e-8)

    sim_ref = float(np.dot(v_drug, v_disease) /
                    (np.linalg.norm(v_drug) * np.linalg.norm(v_disease) + 1e-8))

    neighbors = bv.similar_by_vector(v_result, topn=topn)

    results = []
    for word, sim in neighbors:
        v_w = bv[word]
        v_offset = v_w - v_new
        offset_norm = v_offset / (np.linalg.norm(v_offset) + 1e-8)
        dir_score = float(np.dot(delta_norm, offset_norm))
        dist = float(np.linalg.norm(v_w - v_result))

        results.append({
            'rank': len(results) + 1,
            'word': word,
            'cosine_sim': round(float(sim), 4),
            'direction_score': round(dir_score, 4),
            'euclidean_dist': round(dist, 4)
        })

    return {
        'disease': disease, 'drug': drug, 'new_disease': new_disease,
        'ref_sim': round(sim_ref, 4),
        'delta_norm': round(float(np.linalg.norm(delta)), 4),
        'candidates': results
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
server.bind(('127.0.0.1', 9995))
server.listen(5)
print("struct_server 监听 9995...", flush=True)

while True:
    conn, addr = server.accept()
    try:
        data = recv_all(conn)
        req = json.loads(data)
        cmd = req.get('cmd', 'neighbors')

        if cmd == 'neighbors':
            result = show_neighbors(req['word'], req.get('topn', 20))
        elif cmd == 'delta':
            result = show_delta(req['word1'], req['word2'])
        elif cmd == 'analogy':
            result = show_analogy(req['disease'], req['drug'], req['new_disease'], req.get('topn', 15))
        else:
            result = {'error': f'未知命令: {cmd}'}

        conn.sendall(json.dumps(result, ensure_ascii=False).encode('utf-8'))
    except Exception as e:
        conn.sendall(json.dumps({'error': str(e)}).encode('utf-8'))
    finally:
        conn.close()