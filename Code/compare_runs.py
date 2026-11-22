"""Compare two verdict runs on the same claims (read-only).

Example:
    python compare_runs.py --split train_val --candidate train_val_output_evidence_v2
"""
import argparse
import json
from collections import Counter
from math import comb
from pathlib import Path

TRUE_RATINGS = {"True", "Mostly True", "Correct Attribution"}
FALSE_RATINGS = {"False", "Miscaptioned", "Mixture", "Fake", "Mostly False"}


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def valid_answer(run_dir, video_id):
    try:
        verdict = load(run_dir / video_id / f"{video_id}_CV_result.json")["Final_Judgement"]
        answer = verdict["Answer"]
        reasons = verdict["Reasons"]
    except (OSError, ValueError, KeyError, TypeError):
        return None
    if answer not in ("True", "False") or not isinstance(reasons, str) or not reasons.strip():
        return None
    return answer


def metrics(gold, pred):
    n = len(gold)
    correct = sum(g == p for g, p in zip(gold, pred))
    result = {"n": n, "accuracy": correct / n if n else 0.0, "predictions": dict(Counter(pred))}
    f1s = []
    for cls in ("True", "False"):
        tp = sum(g == cls and p == cls for g, p in zip(gold, pred))
        fp = sum(g != cls and p == cls for g, p in zip(gold, pred))
        fn = sum(g == cls and p != cls for g, p in zip(gold, pred))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        result[f"{cls}_precision"], result[f"{cls}_recall"], result[f"{cls}_f1"] = precision, recall, f1
        f1s.append(f1)
    result["macro_f1"] = sum(f1s) / 2
    return result


def mcnemar_exact(baseline_only, candidate_only):
    n = baseline_only + candidate_only
    if n == 0:
        return 1.0
    tail = sum(comb(n, k) for k in range(min(baseline_only, candidate_only) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/datasets/ambatha/TRUE_Dataset")
    parser.add_argument("--split", default="train_val", choices=["test", "train_val"])
    parser.add_argument("--baseline", default=None, help="Defaults to <split>_output")
    parser.add_argument("--candidate", required=True, help="Directory name under --root")
    args = parser.parse_args()

    root = Path(args.root)
    baseline_dir = root / (args.baseline or f"{args.split}_output")
    candidate_dir = root / args.candidate
    candidate_ids = sorted(p.name for p in candidate_dir.iterdir() if p.is_dir())

    gold, base, cand = [], [], []
    missing = Counter()
    for video_id in candidate_ids:
        rating = load(root / args.split / f"{video_id}.json").get("rating")
        label = "True" if rating in TRUE_RATINGS else "False" if rating in FALSE_RATINGS else None
        b, c = valid_answer(baseline_dir, video_id), valid_answer(candidate_dir, video_id)
        if label is None:
            missing["unmapped rating"] += 1
        elif b is None or c is None:
            missing["baseline invalid" if b is None else "candidate invalid"] += 1
        else:
            gold.append(label)
            base.append(b)
            cand.append(c)

    if not gold:
        raise SystemExit("No claims have valid verdicts in both runs.")

    print(f"Candidate claims: {len(candidate_ids)} | compared on: {len(gold)} | excluded: {dict(missing)}")
    print(f"Gold labels: {dict(Counter(gold))}")
    results = {"baseline": metrics(gold, base), "candidate": metrics(gold, cand)}
    print(f"\n{'Metric':18} {'Baseline':>10} {'Candidate':>10} {'Change':>10}")
    for key in ("accuracy", "macro_f1", "True_precision", "True_recall", "False_precision", "False_recall"):
        b, c = results["baseline"][key], results["candidate"][key]
        print(f"{key:18} {b:>10.2%} {c:>10.2%} {c - b:>+10.2%}")
    print(f"Predictions        baseline {results['baseline']['predictions']} | candidate {results['candidate']['predictions']}")

    baseline_only = sum(b == g != c for g, b, c in zip(gold, base, cand))
    candidate_only = sum(c == g != b for g, b, c in zip(gold, base, cand))
    changed = sum(b != c for b, c in zip(base, cand))
    print(f"\nVerdicts changed: {changed}/{len(gold)}")
    print(f"Only baseline correct: {baseline_only} | only candidate correct: {candidate_only}")
    print(f"Exact McNemar p-value: {mcnemar_exact(baseline_only, candidate_only):.4f}")


if __name__ == "__main__":
    main()
