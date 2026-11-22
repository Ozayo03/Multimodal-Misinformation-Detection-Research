"""Build a blinded, paired rationale-review packet from two development runs."""
import argparse
import csv
import json
import random
from pathlib import Path


CONDITIONS = ("minimal_true_first", "minimal_false_first", "evidence_v2_true_first", "evidence_v2_false_first")


def load(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_result(run, condition, video_id):
    path = run / condition / video_id / f"{video_id}_CV_result.json"
    result = load(path)
    verdict = result.get("Final_Judgement") if isinstance(result, dict) else None
    if not isinstance(verdict, dict) or verdict.get("Answer") not in ("True", "False"):
        raise ValueError(f"Invalid verdict in {path}")
    if not isinstance(verdict.get("Reasons"), str) or not verdict["Reasons"].strip():
        raise ValueError(f"Missing Reasons in {path}")
    return result


def warning_map(run):
    attempts = load(run / "attempts.json")
    return {(item.get("condition"), item.get("id")): item.get("protocol_warnings", [])
            for item in attempts if item.get("protocol_warnings")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-a", type=Path, required=True)
    parser.add_argument("--run-b", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("/datasets/ambatha/TRUE_Dataset"))
    parser.add_argument("--condition", choices=CONDITIONS, required=True)
    parser.add_argument("--sample-size", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20261109)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.sample_size < 1:
        parser.error("--sample-size must be positive")
    if args.out_dir.exists():
        parser.error("Output directory must not already exist")

    manifest_a, manifest_b = load(args.run_a / "manifest.json"), load(args.run_b / "manifest.json")
    ids_a, ids_b = set(manifest_a.get("selected_ids", [])), set(manifest_b.get("selected_ids", []))
    if not ids_a or ids_a != ids_b:
        parser.error("The two runs must have the same nonempty selected ID set")
    backend_a, backend_b = manifest_a.get("backend", "run_a"), manifest_b.get("backend", "run_b")
    model_a, model_b = backend_a, backend_b
    warnings_a, warnings_b = warning_map(args.run_a), warning_map(args.run_b)
    usable = []
    excluded = []
    inputs = args.data_root / "train_val_output_dataset_evidence100"
    for video_id in sorted(ids_a):
        try:
            result_a = read_result(args.run_a, args.condition, video_id)
            result_b = read_result(args.run_b, args.condition, video_id)
            source = load(inputs / video_id / f"{video_id}_CV_result.json")
            evidence = load(inputs / video_id / "dataset_evidence_input.json")
            claim = source["Claim"]
            video_info = source["Video_information"]
            if result_a.get("Claim") != claim or result_b.get("Claim") != claim:
                raise ValueError("Run claim differs from frozen shared claim")
            if result_a.get("Video_information") != video_info or result_b.get("Video_information") != video_info:
                raise ValueError("Run video information differs from frozen shared input")
            usable.append({"id": video_id, "claim": claim, "video_info": video_info, "evidence": evidence,
                           "a": result_a["Final_Judgement"], "b": result_b["Final_Judgement"],
                           "warnings_a": warnings_a.get((args.condition, video_id), []),
                           "warnings_b": warnings_b.get((args.condition, video_id), [])})
        except (OSError, ValueError, TypeError, KeyError) as error:
            excluded.append({"id": video_id, "error": str(error)})
    if len(usable) < args.sample_size:
        parser.error(f"Only {len(usable)} paired valid cases; requested {args.sample_size}")

    rng = random.Random(args.seed)
    disagreements = [item for item in usable if item["a"]["Answer"] != item["b"]["Answer"]]
    warning_cases = [item for item in usable if item["warnings_a"] or item["warnings_b"]]
    disagreement_ids = {item["id"] for item in disagreements}
    warning_only = [item for item in warning_cases if item["id"] not in disagreement_ids]
    clean_agreements = [item for item in usable if item["id"] not in disagreement_ids
                        and not item["warnings_a"] and not item["warnings_b"]]
    for group in (disagreements, warning_only, clean_agreements):
        rng.shuffle(group)
    selected = []
    strata = {}
    stratum_by_id = {}
    for label, group in (("verdict_disagreement", disagreements), ("protocol_warning", warning_only),
                         ("clean_agreement", clean_agreements)):
        take = min(args.sample_size - len(selected), len(group))
        selected.extend(group[:take])
        strata[label] = take
        stratum_by_id.update({item["id"]: label for item in group[:take]})
        if len(selected) == args.sample_size:
            break
    rng.shuffle(selected)
    assignments = {item["id"]: (("a", "b") if rng.random() < 0.5 else ("b", "a")) for item in selected}

    args.out_dir.mkdir(parents=True, exist_ok=False)
    packet_path = args.out_dir / "review_packet.csv"
    key_path = args.out_dir / "unblind_key.csv"
    columns = ["review_id", "claim", "video_information", "dataset_evidence", "answer_A", "reasons_A",
               "answer_B", "reasons_B", "claim_fidelity_A", "claim_fidelity_B", "evidence_support_A",
               "evidence_support_B", "verdict_reason_consistency_A", "verdict_reason_consistency_B",
               "unsupported_details_A", "unsupported_details_B", "preferred_explanation", "reviewer_notes"]
    with packet_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for index, item in enumerate(selected, 1):
            backend_for_a, backend_for_b = assignments[item["id"]]
            judgment_a, judgment_b = item[backend_for_a], item[backend_for_b]
            writer.writerow({"review_id": f"R{index:03d}", "claim": item["claim"],
                             "video_information": json.dumps(item["video_info"], ensure_ascii=False),
                             "dataset_evidence": json.dumps(item["evidence"], ensure_ascii=False),
                             "answer_A": judgment_a["Answer"], "reasons_A": judgment_a["Reasons"],
                             "answer_B": judgment_b["Answer"], "reasons_B": judgment_b["Reasons"]})
    with key_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["review_id", "video_id", "backend_A", "backend_B",
                                                    "warnings_A", "warnings_B", "selection_stratum"])
        writer.writeheader()
        for index, item in enumerate(selected, 1):
            source_for_a, source_for_b = assignments[item["id"]]
            warnings = {"a": item["warnings_a"], "b": item["warnings_b"]}
            writer.writerow({"review_id": f"R{index:03d}", "video_id": item["id"],
                             "backend_A": model_a if source_for_a == "a" else model_b,
                             "backend_B": model_a if source_for_b == "a" else model_b,
                             "warnings_A": json.dumps(warnings[source_for_a]), "warnings_B": json.dumps(warnings[source_for_b]),
                             "selection_stratum": stratum_by_id[item["id"]]})
    (args.out_dir / "selection_manifest.json").write_text(json.dumps({
        "condition": args.condition, "sample_size": len(selected), "seed": args.seed,
        "selection_strata": strata, "paired_valid_available": len(usable),
        "verdict_disagreements_available": len(disagreements), "warning_cases_available": len(warning_cases),
        "excluded_invalid_pairs": excluded, "review_packet_contains_gold_labels": False,
        "backend_a": model_a, "backend_b": model_b,
    }, indent=2), encoding="utf-8")
    print(f"Wrote {len(selected)} blinded paired cases to {packet_path}")
    print(f"Selection strata: {strata}; key at {key_path} (keep private until review is complete)")


if __name__ == "__main__":
    main()
