"""Read-only evidence diagnostics; no model inference, decoding or ASR."""
import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


def load(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def text(value):
    return value.strip() if isinstance(value, str) else ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/datasets/ambatha/TRUE_Dataset")
    parser.add_argument("--split", choices=("test", "train_val"), default="train_val")
    parser.add_argument("--run", help="Verdict directory name under root")
    parser.add_argument("--out", default="evidence_audit.csv")
    args = parser.parse_args()
    root = Path(args.root)
    ids = dict.fromkeys(line.strip() for line in (root / f"{args.split}_set.txt").read_text(
        encoding="utf-8-sig").splitlines() if line.strip())
    run = root / (args.run or f"{args.split}_output")
    groups = defaultdict(Counter)
    rows = []
    labels = {rating: "True" for rating in ("True", "Mostly True", "Correct Attribution")}
    labels.update({rating: "False" for rating in ("False", "Miscaptioned", "Mixture", "Fake", "Mostly False")})
    for video_id in ids:
        row = {"video_id": video_id}
        try:
            gold = load(root / args.split / f"{video_id}.json")
            source = gold.get("video_information") or gold.get("Video_information") or {}
            cached = load(root / f"{args.split}_output" / f"{video_id}.json")["Video_information"]
            transcript = text(cached.get("video_transcript"))
            description = text(cached.get("Video_descriptor"))
            row.update(
                transcript_words=len(transcript.split()), description_words=len(description.split()),
                transcript_changed=transcript != text(source.get("video_transcript")),
                transcript_status="missing" if not transcript else "short" if len(transcript.split()) < 20 else "present",
                description_status="present" if description else "missing",
            )
            try:
                verdict = load(run / video_id / f"{video_id}_CV_result.json")["Final_Judgement"]
                prediction = verdict.get("Answer")
                reasons = text(verdict.get("Reasons"))
                label = labels.get(gold.get("rating"))
                row.update(predicted=prediction, gold_label=label, reasons_words=len(reasons.split()))
                if prediction in ("True", "False") and reasons and label:
                    row["correct"] = prediction == label
                    for group in ("all", "transcript_" + row["transcript_status"],
                                  "description_" + row["description_status"]):
                        groups[group]["scored"] += 1
                        groups[group]["correct"] += row["correct"]
            except (OSError, ValueError, KeyError, TypeError) as error:
                row["verdict_error"] = str(error)
        except (OSError, ValueError, KeyError, TypeError) as error:
            row["evidence_error"] = str(error)
        rows.append(row)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(args.out).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Expected claims: {len(rows)}; evidence errors: {sum('evidence_error' in row for row in rows)}")
    print(f"Changed source/cached transcripts: {sum(row.get('transcript_changed', False) for row in rows)}")
    for group, counts in sorted(groups.items()):
        print(f"{group}: n={counts['scored']}, accuracy={counts['correct']/counts['scored']:.2%}")
    print(f"Details: {args.out}")
    print("Groups are observational, not causal. Transcript correctness requires audio/reference checking.")


if __name__ == "__main__":
    main()