import argparse
import json
from pathlib import Path

from remote_processor import process_tee_request


def main():
    parser = argparse.ArgumentParser(description="远程侧 DAG 构建服务")
    parser.add_argument("--input", "-i", required=True, help="本地侧输出的 local_output.json")
    parser.add_argument("--output", "-o", default="./remote_output.json", help="远程 DAG 输出路径")
    args = parser.parse_args()

    input_path = Path(args.input)
    records = json.loads(input_path.read_text(encoding="utf-8"))

    responses = []
    for record in records:
        response = process_tee_request(record["tee_request"])
        responses.append({
            "original_text": record.get("original_text", ""),
            "tee_response": response,
        })

    output_path = Path(args.output)
    output_path.write_text(
        json.dumps(responses, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"远程 DAG 构建完成，结果已写入: {output_path}")


if __name__ == "__main__":
    main()
