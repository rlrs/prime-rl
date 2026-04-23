import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-path", type=Path, required=True)
    parser.add_argument("--candidate-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--baseline-label", default="Baseline")
    parser.add_argument("--candidate-label", default="Candidate")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def index_rows(rows: list[dict]) -> dict[str, dict]:
    indexed = {}
    for row in rows:
        row_id = row.get("id")
        if row_id is None:
            raise ValueError(f"Missing 'id' in row from {row}")
        indexed[row_id] = row
    return indexed


def render_section(
    baseline_row: dict,
    candidate_row: dict,
    baseline_label: str,
    candidate_label: str,
) -> str:
    prompt = baseline_row.get("prompt")
    if prompt != candidate_row.get("prompt"):
        raise ValueError(f"Prompt mismatch for id={baseline_row['id']}")

    category = baseline_row.get("category", "uncategorized")
    baseline_score = render_judge_score(baseline_row)
    candidate_score = render_judge_score(candidate_row)
    return "\n".join(
        [
            f"## {baseline_row['id']} ({category})",
            "",
            "### Prompt",
            "",
            prompt,
            "",
            f"### {baseline_label}",
            "",
            baseline_score,
            "",
            baseline_row.get("response", "").strip(),
            "",
            f"### {candidate_label}",
            "",
            candidate_score,
            "",
            candidate_row.get("response", "").strip(),
            "",
            "### Notes",
            "",
            "- Better output:",
            "- Fluency:",
            "- Instruction following:",
            "- Factual/helpfulness:",
            "",
        ]
    )


def render_judge_score(row: dict) -> str:
    score = row.get("judge_score")
    if score is None:
        return ""
    normalized = row.get("judge_score_normalized")
    if normalized is not None:
        return f"_Judge score: {score:.1f}/10 ({normalized:.4f})_"
    return f"_Judge score: {score:.1f}/10_"


def render_comparison(
    *,
    baseline_path: Path,
    candidate_path: Path,
    output_path: Path,
    baseline_label: str,
    candidate_label: str,
) -> None:
    baseline = index_rows(load_jsonl(baseline_path))
    candidate = index_rows(load_jsonl(candidate_path))

    missing_in_candidate = sorted(set(baseline) - set(candidate))
    missing_in_baseline = sorted(set(candidate) - set(baseline))
    if missing_in_candidate or missing_in_baseline:
        raise ValueError(
            "Mismatched ids between files: "
            f"missing_in_candidate={missing_in_candidate}, "
            f"missing_in_baseline={missing_in_baseline}"
        )

    sections = [
        "# Manual Eval Comparison",
        "",
        f"- {baseline_label}: `{baseline_path}`",
        f"- {candidate_label}: `{candidate_path}`",
        "",
    ]

    for row_id in sorted(baseline):
        sections.append(
            render_section(
                baseline_row=baseline[row_id],
                candidate_row=candidate[row_id],
                baseline_label=baseline_label,
                candidate_label=candidate_label,
            )
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(sections) + "\n")


def main() -> None:
    args = parse_args()
    render_comparison(
        baseline_path=args.baseline_path,
        candidate_path=args.candidate_path,
        output_path=args.output_path,
        baseline_label=args.baseline_label,
        candidate_label=args.candidate_label,
    )


if __name__ == "__main__":
    main()
