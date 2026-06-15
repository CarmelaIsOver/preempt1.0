"""
web_server.py — Anon Demo Web API

启动: python local_service/web_server.py --port 8000
"""

import sys
import os
import json
import argparse

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

from flask import Flask, request, jsonify, send_from_directory
from local_service.experiment.full_pipeline import PreemptPipeline

app = Flask(__name__)

_FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")


@app.route("/")
def index():
    return send_from_directory(_FRONTEND_DIR, "index.html")


@app.route("/library")
def library():
    return send_from_directory(_FRONTEND_DIR, "library.html")

_pipeline_cache = {}


def _get_pipeline(epsilon, use_local_ner):
    key = (epsilon, use_local_ner)
    if key not in _pipeline_cache:
        _pipeline_cache[key] = PreemptPipeline(
            epsilon=epsilon, use_local_ner=use_local_ner
        )
    return _pipeline_cache[key]


@app.after_request
def cors(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "POST, GET, OPTIONS"
    return response


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "Anon"})


@app.route("/api/sanitize", methods=["POST", "OPTIONS"])
def sanitize():
    if request.method == "OPTIONS":
        return jsonify({}), 200

    data = request.get_json(silent=True) or {}
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "text is required"}), 400

    epsilon = float(data.get("epsilon", 1.0))
    use_local_ner = data.get("use_local_ner", False)

    try:
        pipeline = _get_pipeline(epsilon, use_local_ner)
        result = pipeline.sanitize(text)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    vault_list = []
    t1_count = t2_count = t3_count = 0
    for token, info in result.get("vault", {}).items():
        vtype = info.get("type", "")
        tier = "T1" if vtype.startswith("t1") else ("T2" if vtype.startswith("t2") else "T3")
        if tier == "T1":
            t1_count += 1
        elif tier == "T2":
            t2_count += 1
        else:
            t3_count += 1
        vault_list.append({
            "tier": tier,
            "type": vtype,
            "original": token,
            "replacement": info.get("to", info.get("replacement", "")),
            "detail": info,
        })

    entities_out = []
    for ent in result.get("entities", []):
        if isinstance(ent, (list, tuple)):
            entities_out.append({"token": ent[0], "label": ent[1] if len(ent) > 1 else ""})
        else:
            entities_out.append(ent)

    return jsonify({
        "original": result["original"],
        "sanitized": result["sanitized"],
        "entities": entities_out,
        "expressions": result.get("expressions", []),
        "vault": vault_list,
        "stats": {
            "t1_count": t1_count,
            "t2_count": t2_count,
            "t3_count": t3_count,
            "entity_count": len(entities_out),
            "expression_count": len(result.get("expressions", [])),
        },
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Anon Web Server")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    args = parser.parse_args()

    print(f"Anon web server starting at http://{args.host}:{args.port}")
    print(f"API docs: http://{args.host}:{args.port}/api/health")
    app.run(host=args.host, port=args.port, debug=True)
