"""Read-only classification audit; does not repair or replace strict results."""
import argparse
import json
import math
from collections import Counter
from pathlib import Path


CONDITIONS = ("minimal_true_first", "minimal_false_first", "evidence_v2_true_first", "evidence_v2_false_first")
TRUE_RATINGS = {"True", "Mostly True", "Correct Attribution"}
FALSE_RATINGS = {"False", "Miscaptioned", "Mixture", "Fake", "Mostly False"}


def load(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def raw_verdict(path):
    text = path.read_text(encoding="utf-8-sig").strip()
    if text.startswith("```json\n") and text.endswith("```"):
        text = text[len("```json\n"):-3].strip()
    parsed = json.loads(text)
    verdict = parsed.get("Final_Judgement") if isinstance(parsed, dict) else None
    if not isinstance(verdict, dict):
        raise ValueError("Missing Final_Judgement object")
    if verdict.get("Answer") not in ("True", "False"):
        raise ValueError("Missing binary Answer")
    if not isinstance(verdict.get("Reasons"), str) or not verdict["Reasons"].strip():
        raise ValueError("Missing nonempty Reasons")
    return verdict


def saved_prediction(run, backend, condition, video_id):
    if backend == "minicpm":
        artifact = load(run / condition / video_id / f"{video_id}_CV_result.json")
        verdict = artifact.get("Final_Judgement") if isinstance(artifact, dict) else None
        if not isinstance(verdict, dict) or verdict.get("Answer") not in ("True", "False"):
            raise ValueError("Missing parsed MiniCPM verdict")
        if not isinstance(verdict.get("Reasons"), str) or not verdict["Reasons"].strip():
            raise ValueError("Missing parsed MiniCPM reasons")
        return verdict["Answer"]
    return raw_verdict(run / condition / video_id / "raw_response.txt")["Answer"]


def metrics(gold, predictions, ids):
    selected = list(ids)
    if not selected:
        return {"n": 0}
    recalls, scores = {}, []
    for label in ("True", "False"):
        true_positive = sum(gold[vid] == label and predictions[vid] == label for vid in selected)
        actual = sum(gold[vid] == label for vid in selected)
        predicted = sum(predictions[vid] == label for vid in selected)
        recalls[label] = true_positive / actual if actual else 0.0
        scores.append(2 * true_positive / (actual + predicted) if actual + predicted else 0.0)
    return {"n": len(selected), "accuracy": sum(gold[vid] == predictions[vid] for vid in selected) / len(selected),
            "macro_f1": sum(scores) / 2, "recall": recalls,
            "predictions": dict(Counter(predictions[vid] for vid in selected))}


def paired(gold, left, right):
    common = sorted(set(left) & set(right))
    left_only = sum(left[vid] == gold[vid] and right[vid] != gold[vid] for vid in common)
    right_only = sum(right[vid] == gold[vid] and left[vid] != gold[vid] for vid in common)
    discordant = left_only + right_only
    probability = min(1.0, 2 * sum(math.comb(discordant, index) for index in range(min(left_only, right_only) + 1)) / 2 ** discordant) if discordant else 1.0
    return {"n": len(common), "verdict_changes": sum(left[vid] != right[vid] for vid in common),
            "left_only_correct": left_only, "right_only_correct": right_only, "exact_mcnemar_p": probability}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--compare-run", type=Path, help="Optional second backend run using the same sample")
    parser.add_argument("--root", type=Path, default=Path("/datasets/ambatha/TRUE_Dataset"))
    args = parser.parse_args()
    manifest = load(args.run / "manifest.json")
    backend = manifest.get("backend", "qwen")
    ids = manifest["selected_ids"]
    if not ids or len(set(ids)) != len(ids) or any(not isinstance(vid, str) or not vid.isascii() or not vid.isdigit() for vid in ids):
        parser.error("Selected IDs must be unique ASCII-digit strings")
    gold = {}
    for vid in ids:
        rating = load(args.root / "train_val" / f"{vid}.json").get("rating")
        if rating not in TRUE_RATINGS | FALSE_RATINGS:
            raise ValueError(f"Unmapped rating for {vid}: {rating}")
        gold[vid] = "True" if rating in TRUE_RATINGS else "False"
    strict = {name: {} for name in CONDITIONS}
    attempts = load(args.run / "attempts.json")
    attempt_details = {name: {} for name in CONDITIONS}
    for attempt in attempts:
        if attempt.get("answer") in ("True", "False"):
            strict[attempt["condition"]][attempt["id"]] = attempt["answer"]
        attempt_details.setdefault(attempt.get("condition"), {})[attempt.get("id")] = attempt
    predictions = {name: {} for name in CONDITIONS}
    details = {name: [] for name in CONDITIONS}
    for name in CONDITIONS:
        for vid in ids:
            try:
                answer = saved_prediction(args.run, backend, name, vid)
                predictions[name][vid] = answer
                if vid not in strict[name] or attempt_details.get(name, {}).get(vid, {}).get("protocol_warnings"):
                    field = "The specific type of False Information is"
                    artifact_verdict = None
                    if backend == "minicpm":
                        artifact_verdict = load(args.run / name / vid / f"{vid}_CV_result.json").get("Final_Judgement", {})
                    details[name].append({"id": vid, "answer": answer, "gold": gold[vid],
                                          "strict_failure": vid not in strict[name],
                                          "protocol_warnings": attempt_details.get(name, {}).get(vid, {}).get("protocol_warnings", []),
                                          "empty_false_field_on_true": answer == "True" and isinstance(artifact_verdict, dict) and field in artifact_verdict and artifact_verdict[field] == ""})
            except (OSError, ValueError, TypeError) as error:
                details[name].append({"id": vid, "raw_error": str(error)})
    common = set(ids)
    for prediction in predictions.values():
        common &= set(prediction)
    result_source = "parsed MiniCPM protocol verdicts" if backend == "minicpm" else "raw JSON classifications"
    print(f"Backend: {backend}; selected: {len(ids)}; common {result_source}: {len(common)}")
    for name in CONDITIONS:
        prediction = predictions[name]
        print(f"\n{name}: strict-valid={len(strict[name])}/{len(ids)}; {result_source}={len(prediction)}/{len(ids)}")
        print("Own valid:", json.dumps(metrics(gold, prediction, prediction)))
        print("Common valid:", json.dumps(metrics(gold, prediction, common)))
        print("Accuracy counting unavailable raw answers as wrong:", sum(prediction[vid] == gold[vid] for vid in prediction) / len(ids))
        print("Strict failures / raw issues:", json.dumps(details[name]))
    for left, right in ((CONDITIONS[0], CONDITIONS[2]), (CONDITIONS[1], CONDITIONS[3]),
                        (CONDITIONS[0], CONDITIONS[1]), (CONDITIONS[2], CONDITIONS[3])):
        print(f"\n{left} vs {right}:", json.dumps(paired(gold, predictions[left], predictions[right])))
    print(f"\nClassification source: {result_source}; protocol warnings and strict failures are reported separately; explanations not assessed for factuality.")
    if args.compare_run:
        other_manifest = load(args.compare_run / "manifest.json")
        other_backend = other_manifest.get("backend", "qwen")
        other_ids = other_manifest.get("selected_ids", [])
        if set(ids) != set(other_ids):
            print(f"\nWARNING: selected ID sets differ: first={len(ids)}, second={len(other_ids)}")
        print(f"\nPaired backend comparison: {backend} ({args.run}) vs {other_backend} ({args.compare_run})")
        other_attempts = load(args.compare_run / "attempts.json")
        other_valid = {condition: {} for condition in CONDITIONS}
        for condition in CONDITIONS:
            for vid in other_ids:
                try:
                    other_valid[condition][vid] = saved_prediction(args.compare_run, other_backend, condition, vid)
                except (OSError, ValueError, TypeError):
                    pass
        for condition in CONDITIONS:
            first = predictions[condition]
            second = other_valid[condition]
            common_pair = sorted(set(first) & set(second))
            print(f"\n{condition}: first valid={len(first)}/{len(ids)}; second valid={len(second)}/{len(other_ids)}; paired={len(common_pair)}")
            print(f"  {backend}:", json.dumps(metrics(gold, first, common_pair)))
            print(f"  {other_backend}:", json.dumps(metrics(gold, second, common_pair)))
            print("  paired correctness:", json.dumps(paired(gold, first, second)))
        first_warnings = sum(len(attempt.get("protocol_warnings", [])) for attempt in attempts)
        second_warnings = sum(len(attempt.get("protocol_warnings", [])) for attempt in other_attempts)
        print(f"\nProtocol warning totals: {backend}={first_warnings}; {other_backend}={second_warnings}")


if __name__ == "__main__":
    main()