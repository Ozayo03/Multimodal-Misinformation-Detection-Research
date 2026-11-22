"""Compare generated Reasons with TRUE gold rationales using BLEU, ROUGE, METEOR, BERTScore and BLEURT.

Read-only: never modifies dataset or pipeline outputs.

Usage:
    python explanation_report.py --metrics bleu rouge meteor   # CPU only, safe on the login node
    python explanation_report.py                               # all five metrics (run as a GPU job)
"""
import argparse
import csv
import json
import os
import re
from collections import Counter, defaultdict
from importlib import metadata
from pathlib import Path
from statistics import mean, stdev

TRUE_RATINGS = {"True", "Mostly True", "Correct Attribution"}
FALSE_RATINGS = {"False", "Miscaptioned", "Mixture", "Fake", "Mostly False"}
ALL_METRICS = ["bleu", "rouge", "meteor", "bertscore", "bleurt"]
REFERENCE_FIELDS = {"summary": "summary_rationales", "original": "original_rationales"}
# Roughly where BERTScore/BLEURT's 512-token input limit starts truncating text.
LONG_REFERENCE_WORDS = 400
_BLEURT_CACHE = {}


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def flatten(value):
    if isinstance(value, dict):
        return " ".join(flatten(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return " ".join(flatten(v) for v in value)
    return "" if value is None else str(value)


def tokenize(text):
    return re.findall(r"\w+|[^\w\s]", text.lower())


def score_bleu(hyps, refs, args):
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

    smooth = SmoothingFunction().method1
    scores = {"BLEU-1": [], "BLEU-4": []}
    for hyp, ref in zip(hyps, refs):
        hyp_tokens, ref_tokens = tokenize(hyp), [tokenize(ref)]
        scores["BLEU-1"].append(sentence_bleu(ref_tokens, hyp_tokens, weights=(1, 0, 0, 0), smoothing_function=smooth))
        scores["BLEU-4"].append(sentence_bleu(ref_tokens, hyp_tokens, weights=(0.25,) * 4, smoothing_function=smooth))
    return scores


def score_rouge(hyps, refs, args):
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = {"ROUGE-1": [], "ROUGE-2": [], "ROUGE-L": []}
    for hyp, ref in zip(hyps, refs):
        result = scorer.score(ref, hyp)
        scores["ROUGE-1"].append(result["rouge1"].fmeasure)
        scores["ROUGE-2"].append(result["rouge2"].fmeasure)
        scores["ROUGE-L"].append(result["rougeL"].fmeasure)
    return scores


def score_meteor(hyps, refs, args):
    from nltk.translate.meteor_score import meteor_score

    return {"METEOR": [meteor_score([tokenize(ref)], tokenize(hyp)) for hyp, ref in zip(hyps, refs)]}


def score_bertscore(hyps, refs, args):
    from bert_score import score

    (precision, recall, f1), config_hash = score(
        hyps, refs, model_type=args.bertscore_model, lang="en", rescale_with_baseline=True,
        batch_size=args.batch_size, device=args.device, return_hash=True,
    )
    args.config["bertscore_hash"] = config_hash
    return {"BERTScore-P": precision.tolist(), "BERTScore-R": recall.tolist(), "BERTScore-F1": f1.tolist()}


def score_bleurt(hyps, refs, args):
    import torch
    from bleurt_pytorch import BleurtForSequenceClassification, BleurtTokenizer
    from transformers.utils import logging as hf_logging

    # Truncation is counted explicitly below, so silence the per-batch overflow warnings.
    hf_logging.set_verbosity_error()
    if args.bleurt_model not in _BLEURT_CACHE:
        tokenizer = BleurtTokenizer.from_pretrained(args.bleurt_model)
        model = BleurtForSequenceClassification.from_pretrained(args.bleurt_model).to(args.device).eval()
        _BLEURT_CACHE[args.bleurt_model] = (tokenizer, model)
    tokenizer, model = _BLEURT_CACHE[args.bleurt_model]

    pair_lengths = [len(ids) for ids in tokenizer(refs, hyps, truncation=False)["input_ids"]]
    scores = []
    with torch.no_grad():
        for start in range(0, len(hyps), args.batch_size):
            inputs = tokenizer(
                refs[start:start + args.batch_size], hyps[start:start + args.batch_size],
                padding="longest", truncation=True, max_length=512, return_tensors="pt",
            ).to(args.device)
            scores.extend(model(**inputs).logits.flatten().tolist())
    return {"BLEURT": scores, "BLEURT-truncated": [float(n > 512) for n in pair_lengths]}


SCORERS = {
    "bleu": score_bleu,
    "rouge": score_rouge,
    "meteor": score_meteor,
    "bertscore": score_bertscore,
    "bleurt": score_bleurt,
}


def collect_rows(root, split):
    ids = dict.fromkeys(
        line.strip()
        for line in (root / f"{split}_set.txt").read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    )
    rows, skipped = [], defaultdict(int)

    for video_id in ids:
        try:
            gold = load(root / split / f"{video_id}.json")
            verdict = load(root / f"{split}_output" / video_id / f"{video_id}_CV_result.json")["Final_Judgement"]
            reasons = verdict["Reasons"]
            assert verdict["Answer"] in ("True", "False") and reasons.strip()
        except (OSError, ValueError, KeyError, TypeError, AssertionError):
            skipped["no valid verdict"] += 1
            continue

        summary = gold.get(REFERENCE_FIELDS["summary"])
        # summary_rationales also holds per-reason expansions; the synthesized paragraph is the summary itself.
        synthesized = summary.get("synthesized_rationale") if isinstance(summary, dict) else None
        references = {
            "summary": flatten(synthesized if synthesized else summary),
            "original": flatten(gold.get(REFERENCE_FIELDS["original"])),
        }
        if not any(text.strip() for text in references.values()):
            skipped["no reference rationale"] += 1
            continue

        rating = gold.get("rating")
        label = "True" if rating in TRUE_RATINGS else "False" if rating in FALSE_RATINGS else None
        row = {
            "split": split,
            "video_id": video_id,
            "rating": rating,
            "gold_label": label,
            "predicted": verdict["Answer"],
            "correct": None if label is None else label == verdict["Answer"],
            "reasons": reasons,
            "reasons_words": len(reasons.split()),
            "summary_source": "synthesized_rationale" if synthesized else "all_fields",
        }
        for name, text in references.items():
            row[f"_ref_{name}"] = text
            row[f"{name}_words"] = len(text.split())
        row["gold_summary_rationale"] = references["summary"]
        rows.append(row)

    return rows, skipped


def describe(values):
    values = [v for v in values if v is not None]
    if not values:
        return {"n": 0, "mean": None, "std": None}
    return {"n": len(values), "mean": mean(values), "std": stdev(values) if len(values) > 1 else 0.0}


def fmt(stats):
    if not stats["n"]:
        return "n/a"
    return f"{stats['mean']:.4f} +/- {stats['std']:.4f} (n={stats['n']})"


def package_versions():
    versions = {}
    for name in ("nltk", "rouge-score", "bert-score", "bleurt-pytorch", "torch", "transformers"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=os.getenv("DATASET_ROOT_DIR", "/datasets/ambatha/TRUE_Dataset"))
    parser.add_argument("--splits", nargs="+", default=["test", "train_val"])
    parser.add_argument("--metrics", nargs="+", choices=ALL_METRICS, default=ALL_METRICS)
    parser.add_argument("--references", nargs="+", choices=list(REFERENCE_FIELDS), default=list(REFERENCE_FIELDS))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--bertscore-model", default="roberta-large")
    parser.add_argument("--bleurt-model", default="lucadiliello/BLEURT-20")
    parser.add_argument("--device", default=None)
    parser.add_argument("--out-dir", default=".")
    args = parser.parse_args()

    if args.device is None and {"bertscore", "bleurt"} & set(args.metrics):
        import torch
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    args.config = {
        "metrics": args.metrics,
        "references": args.references,
        "tokenization": "lowercase, regex \\w+|[^\\w\\s] (BLEU, METEOR)",
        "bleu": "NLTK sentence_bleu, smoothing method1, averaged per claim",
        "rouge": "rouge-score F1, use_stemmer=True",
        "bertscore_model": args.bertscore_model,
        "bertscore_rescale_with_baseline": True,
        "bleurt_model": args.bleurt_model,
        "device": args.device,
        "versions": package_versions(),
    }
    print(f"Metrics: {args.metrics} | references: {args.references} | device: {args.device}")

    root, out_dir = Path(args.root), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {"config": args.config, "splits": {}}

    for split in args.splits:
        rows, skipped = collect_rows(root, split)
        if not rows:
            print(f"\n===== {split}: no claims could be scored | skipped: {dict(skipped)} =====")
            continue

        metric_columns = []
        for reference in args.references:
            scored = [r for r in rows if r[f"_ref_{reference}"].strip()]
            if not scored:
                continue
            hyps = [r["reasons"] for r in scored]
            refs = [r[f"_ref_{reference}"] for r in scored]
            for metric in args.metrics:
                for name, values in SCORERS[metric](hyps, refs, args).items():
                    if len(values) != len(scored):
                        raise RuntimeError(f"{name} returned {len(values)} scores for {len(scored)} claims")
                    column = f"{reference}_{name}"
                    metric_columns.append(column)
                    for row, value in zip(scored, values):
                        row[column] = value

        correct = [r for r in rows if r["correct"] is True]
        wrong = [r for r in rows if r["correct"] is False]
        split_report = {
            "scored": len(rows),
            "skipped": dict(skipped),
            "mean_words": {key: mean(r[f"{key}_words"] for r in rows) for key in ["reasons", *REFERENCE_FIELDS]},
            "long_references": {ref: sum(r[f"{ref}_words"] > LONG_REFERENCE_WORDS for r in rows) for ref in REFERENCE_FIELDS},
            "summary_source": dict(Counter(r["summary_source"] for r in rows)),
            "metrics": {},
        }

        print(f"\n===== {split}: {len(rows)} claims scored | skipped: {dict(skipped)} =====")
        print(f"Summary reference source: {split_report['summary_source']}")
        print("Mean words: " + ", ".join(f"{k} {v:.1f}" for k, v in split_report["mean_words"].items()))
        print(f"References over {LONG_REFERENCE_WORDS} words (may be truncated by BERTScore): "
              f"{split_report['long_references']}")
        print(f"\n{'Metric':26} {'All (mean +/- sd)':30} {'Correct verdicts':30} {'Wrong verdicts'}")
        for column in metric_columns:
            stats = {
                group: describe([r.get(column) for r in subset])
                for group, subset in (("all", rows), ("correct", correct), ("wrong", wrong))
            }
            split_report["metrics"][column] = stats
            print(f"{column:26} {fmt(stats['all']):30} {fmt(stats['correct']):30} {fmt(stats['wrong'])}")
        report["splits"][split] = split_report

        csv_path = out_dir / f"explanation_metrics_{split}.csv"
        fields = ["split", "video_id", "rating", "gold_label", "predicted", "correct", "summary_source", "reasons_words",
                  *(f"{ref}_words" for ref in REFERENCE_FIELDS), *metric_columns, "reasons", "gold_summary_rationale"]
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        print(f"Per-claim scores: {csv_path}")

    summary_path = out_dir / "explanation_metrics_summary.json"
    summary_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSummary and metric configuration: {summary_path}")


if __name__ == "__main__":
    main()
