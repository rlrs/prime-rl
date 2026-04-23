import argparse
import json
from pathlib import Path

from openai import OpenAI


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--endpoint", required=True, help="OpenAI-compatible endpoint, with or without /v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--system-prompt", default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def normalize_base_url(endpoint: str) -> str:
    if endpoint.startswith("http://") or endpoint.startswith("https://"):
        return endpoint if endpoint.rstrip("/").endswith("/v1") else endpoint.rstrip("/") + "/v1"
    return f"http://{endpoint.rstrip('/')}/v1"


def load_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def build_messages(row: dict, system_prompt: str | None) -> list[dict]:
    if "messages" in row:
        messages = list(row["messages"])
    elif "prompt" in row:
        messages = [{"role": "user", "content": row["prompt"]}]
    else:
        raise ValueError("Each row must contain either 'messages' or 'prompt'.")

    if system_prompt is not None:
        return [{"role": "system", "content": system_prompt}, *messages]
    return messages


def generate_row(
    client: OpenAI,
    row: dict,
    model: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    system_prompt: str | None,
) -> dict:
    messages = build_messages(row, system_prompt)
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )
    content = response.choices[0].message.content
    return {
        **row,
        "model": model,
        "response": content,
        "usage": response.usage.model_dump() if response.usage is not None else None,
    }


def main() -> None:
    args = parse_args()
    if args.output_path.exists() and not args.overwrite:
        raise FileExistsError(args.output_path)

    rows = load_jsonl(args.input_path)
    client = OpenAI(api_key=args.api_key, base_url=normalize_base_url(args.endpoint))

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with args.output_path.open("w") as f:
        for row in rows:
            result = generate_row(
                client=client,
                row=row,
                model=args.model,
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_tokens,
                system_prompt=args.system_prompt,
            )
            f.write(json.dumps(result, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
