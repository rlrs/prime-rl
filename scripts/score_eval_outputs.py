import argparse
import json
import re
from pathlib import Path

from openai import OpenAI


DEFAULT_JUDGE_PROMPT_PATH = Path(__file__).with_name("danish_reward_judge_prompt.txt")
SCORE_PATTERN = re.compile(r"(?:Score\s*:\s*)?(10|[1-9])\s*/\s*10", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--endpoint", required=True, help="OpenAI-compatible endpoint, with or without /v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--parse-fail-score", type=float, default=3.0)
    parser.add_argument("--judge-prompt-path", type=Path, default=DEFAULT_JUDGE_PROMPT_PATH)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def normalize_base_url(endpoint: str) -> str:
    if endpoint.startswith("http://") or endpoint.startswith("https://"):
        return endpoint if endpoint.rstrip("/").endswith("/v1") else endpoint.rstrip("/") + "/v1"
    return f"http://{endpoint.rstrip('/')}/v1"


def load_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def load_judge_template(path: Path) -> str:
    return path.read_text()


def build_judge_prompt(template: str, row: dict) -> str:
    payload = {
        "conversation_history": [{"role": "user", "content": row["prompt"]}],
        "gold_response": row.get("gold_response") or row.get("golden response") or "",
        "ai_response": row["response"],
    }
    return template.replace("{{input}}", json.dumps(payload, ensure_ascii=False, indent=2))


def extract_score(raw_response: str, default_score: float) -> tuple[float, bool]:
    matches = list(SCORE_PATTERN.finditer(raw_response))
    if matches:
        return float(matches[-1].group(1)), False
    return float(default_score), True


def score_row(
    client: OpenAI,
    template: str,
    row: dict,
    model: str,
    temperature: float,
    max_tokens: int,
    parse_fail_score: float,
) -> dict:
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": build_judge_prompt(template, row)}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    raw_judge_response = response.choices[0].message.content or ""
    score, used_default = extract_score(raw_judge_response, default_score=parse_fail_score)
    return {
        **row,
        "judge_model": model,
        "judge_score": score,
        "judge_score_normalized": score / 10.0,
        "judge_used_default": used_default,
        "judge_raw_response": raw_judge_response,
        "judge_usage": response.usage.model_dump() if response.usage is not None else None,
    }


def main() -> None:
    args = parse_args()
    if args.output_path.exists() and not args.overwrite:
        raise FileExistsError(args.output_path)

    rows = load_jsonl(args.input_path)
    template = load_judge_template(args.judge_prompt_path)
    client = OpenAI(api_key=args.api_key, base_url=normalize_base_url(args.endpoint))

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with args.output_path.open("w") as f:
        for row in rows:
            scored_row = score_row(
                client=client,
                template=template,
                row=row,
                model=args.model,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                parse_fail_score=args.parse_fail_score,
            )
            f.write(json.dumps(scored_row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
