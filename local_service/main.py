import argparse
import json
from pathlib import Path

from local_sanitizer import LocalPreprocessor
from tee_contract import RelationEdgePayload, T2EntityPayload, TEEProcessRequestPayload


def main():
    parser = argparse.ArgumentParser(description="本地预处理服务：NER + t1 加密 + TEE 请求组装")
    parser.add_argument("--input", "-i", default="../data.txt", help="输入文本文件路径")
    parser.add_argument("--epsilon", "-e", type=float, default=1.0, help="隐私预算")
    parser.add_argument("--output", "-o", default="./local_output.json", help="输出请求 JSON 路径")
    parser.add_argument("--remote-response", default=None, help="远程侧返回的 DAG/edges JSON")
    parser.add_argument("--final-output", default="./final_output.json", help="本地 t2 加密后的最终输出路径")
    args = parser.parse_args()

    if args.remote_response:
        local_records = json.loads(Path(args.input).read_text(encoding="utf-8"))
        remote_records = json.loads(Path(args.remote_response).read_text(encoding="utf-8"))
        preprocessor = LocalPreprocessor(epsilon=args.epsilon)
        final_records = []
        for local_record, remote_record in zip(local_records, remote_records):
            tee_request = local_record["tee_request"]
            request = TEEProcessRequestPayload(
                fpe_sanitized_text=tee_request["fpe_sanitized_text"],
                t2_entities=[
                    T2EntityPayload(
                    token=entity["token"],
                    label=entity["label"],
                    value=entity["value"],
                    start_pos=entity["start_pos"],
                    end_pos=entity["end_pos"],
                    )
                    for entity in tee_request["t2_entities"]
                ],
                epsilon=tee_request["epsilon"],
            )
            relation_edges = [
                RelationEdgePayload(
                    from_entity_index=edge["from_entity_index"],
                    to_entity_index=edge["to_entity_index"],
                    relation_type=edge["relation_type"],
                    param=edge["param"],
                    has_temp_node=edge.get("has_temp_node", False),
                    temp_node_name=edge.get("temp_node_name", ""),
                )
                for edge in remote_record["tee_response"].get("edges", [])
            ]
            sanitized_text, dag_info = preprocessor.apply_remote_dag(request, relation_edges)
            final_records.append({
                "original_text": local_record["original_text"],
                "sanitized_text": sanitized_text,
                "dag_info": dag_info,
                "local_session": local_record.get("local_session", {}),
            })
        Path(args.final_output).write_text(
            json.dumps(final_records, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"本地 t2 加密完成，最终输出已写入: {args.final_output}")
        return

    input_path = Path(args.input)
    texts = [
        line.strip()
        for line in input_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    preprocessor = LocalPreprocessor(epsilon=args.epsilon)
    records = []
    for text in texts:
        request, session_info = preprocessor.prepare_remote_request(text)
        records.append({
            "original_text": text,
            "tee_request": request.to_dict(),
            "local_session": session_info,
        })

    output_path = Path(args.output)
    output_path.write_text(
        json.dumps(records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"本地预处理完成，TEE 请求已写入: {output_path}")


if __name__ == "__main__":
    main()
