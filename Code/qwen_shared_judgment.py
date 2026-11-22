"""Compare local model judgments on saved MiniCPM descriptions and evidence."""
import argparse
import hashlib
import json
import logging
import os
import tempfile
from pathlib import Path

import torch

import Reasoner
from evidence_probe import build_probe_prompt, parse_probe, validate_probe


def load_json(path):
    with path.open(encoding="utf-8-sig") as handle:
        return json.load(handle)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/datasets/ambatha/TRUE_Dataset"))
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--backend", choices=("qwen", "minicpm"), default="qwen")
    parser.add_argument("--strategy", choices=("evidence_v2", "scope_v3", "evidence_probe_v1"), default="evidence_v2")
    parser.add_argument("--run-name", help="New output suffix, for example qwen_scope_control_v2")
    parser.add_argument("--input-mode", choices=("full", "evidence_only"), default="full")
    parser.add_argument("--split", choices=("train_val", "test"), default="train_val")
    parser.add_argument("--limit", type=int, default=0, help="First N prior Qwen pilot IDs; 0 means all")
    parser.add_argument("--ids", nargs="+", help="Specific prior Qwen pilot IDs")
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit must be nonnegative")
    if args.strategy in ("scope_v3", "evidence_probe_v1") and args.split != "train_val":
        parser.error("This strategy is a development experiment; use --split train_val")
    if args.input_mode == "evidence_only" and args.split != "train_val":
        parser.error("Evidence-only ablation is a development experiment; use --split train_val")
    if args.run_name and any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for char in args.run_name):
        parser.error("--run-name must contain only letters, digits, underscores or hyphens")
    if args.ids and (args.limit or any(not item.isascii() or not item.isdigit() for item in args.ids)):
        parser.error("--ids must contain ASCII-digit IDs and cannot be combined with --limit")
    if args.ids and len(set(args.ids)) != len(args.ids):
        parser.error("--ids cannot contain duplicates")
    if not args.model.is_dir():
        parser.error("--model must be an existing local directory")
    if not torch.cuda.is_available():
        raise SystemExit("No CUDA GPU is available; run on an allocated GPU compute node")

    mini = args.root / f"{args.split}_output_dataset_evidence100"
    prior = args.root / f"{args.split}_output_qwen_dataset_evidence_100"
    run_suffix = args.strategy
    run_name = args.run_name or f"{args.backend}_shared_minicpm_{run_suffix}"
    if not args.run_name and args.input_mode == "evidence_only":
        run_name += "_evidence_only"
    output = args.root / f"{args.split}_output_{run_name}"
    if not mini.is_dir() or not prior.is_dir():
        raise SystemExit("Missing MiniCPM evidence results or prior Qwen pilot directory")
    ids = sorted(path.name for path in prior.iterdir() if path.is_dir() and path.name.isascii() and path.name.isdigit())
    if args.ids:
        missing_ids = sorted(set(args.ids) - set(ids))
        if missing_ids:
            raise SystemExit(f"IDs absent from prior Qwen pilot: {', '.join(missing_ids)}")
        ids = args.ids
    if args.limit:
        ids = ids[:args.limit]
    if not ids:
        raise SystemExit("No prior Qwen pilot IDs found")
    output.mkdir(parents=True, exist_ok=True)

    major, _ = torch.cuda.get_device_capability(0)
    dtype = torch.bfloat16 if major >= 8 else torch.float16
    print(f"GPU: {torch.cuda.get_device_name(0)}; selected prior pilot IDs: {len(ids)}", flush=True)
    if args.backend == "qwen":
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        from qwen_pilot import generate

        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            str(args.model), torch_dtype=dtype, local_files_only=True, attn_implementation="sdpa"
        ).to("cuda").eval()
        processor = AutoProcessor.from_pretrained(str(args.model), local_files_only=True, use_fast=False)
        Reasoner.gpt4o_mini_analysis = lambda prompt: generate(
            model, processor, [{"type": "text", "text": "You are a helpful assistant.\n\n" + prompt}],
            max_new_tokens=512,
        )
    else:
        from transformers import AutoModel, AutoTokenizer

        model = AutoModel.from_pretrained(
            str(args.model), trust_remote_code=True, torch_dtype=dtype,
            local_files_only=True, attn_implementation="sdpa",
        ).to("cuda").eval()
        tokenizer = AutoTokenizer.from_pretrained(str(args.model), trust_remote_code=True, local_files_only=True)
        Reasoner.gpt4o_mini_analysis = lambda prompt: model.chat(
            image=None, msgs=[{"role": "user", "content": "You are a helpful assistant.\n\n" + prompt}],
            tokenizer=tokenizer, sampling=False, max_new_tokens=512,
        )
    code_hashes = {
        "diagnostic_sha256": digest(Path(__file__)),
        "reasoner_sha256": digest(Path(Reasoner.__file__)),
        "llm_call_sha256": digest(Path(__file__).with_name("LLM_Call.py")),
        "qwen_pilot_sha256": digest(Path(__file__).with_name("qwen_pilot.py")),
        "evidence_probe_sha256": digest(Path(__file__).with_name("evidence_probe.py")),
    }
    os.environ["USE_EVIDENCE_RETRIEVAL"] = "false"
    os.environ["JUDGMENT_STRATEGY"] = args.strategy
    os.environ["JUDGMENT_INPUT_MODE"] = "all"
    logging.basicConfig(level=logging.INFO)

    failures = 0
    for index, video_id in enumerate(ids, 1):
        print(f"Processing {index}/{len(ids)}: {video_id}", flush=True)
        original = mini / video_id / f"{video_id}_CV_result.json"
        evidence_path = mini / video_id / "dataset_evidence_input.json"
        folder = output / video_id
        result_path = folder / f"{video_id}_CV_result.json"
        if args.strategy == "evidence_probe_v1":
            result_path = folder / "evidence_probe.json"
        manifest_path = folder / "shared_judgment_manifest.json"
        temporary = None
        try:
            data = load_json(original)
            evidence = load_json(evidence_path)
            claim = data["Claim"]
            video_info = data["Video_information"]
            if not isinstance(claim, str) or not claim.strip() or not isinstance(video_info, dict):
                raise ValueError("Missing claim or video information")
            if not isinstance(evidence, dict) or not isinstance(evidence.get("Dataset_evidence"), list):
                raise ValueError("Missing dataset evidence manifest")
            if args.input_mode == "evidence_only":
                video_info = {}
            folder.mkdir(parents=True, exist_ok=True)
            manifest = {
                "condition": "shared_minicpm_description_and_dataset_evidence" if args.input_mode == "full" else "dataset_evidence_only",
                "input_mode": args.input_mode,
                "model": str(args.model), "gpu": torch.cuda.get_device_name(0), "dtype": str(dtype),
                "source_result": str(original), "source_result_sha256": digest(original),
                "source_evidence_manifest": str(evidence_path), "source_evidence_sha256": digest(evidence_path),
                "prompt_strategy": args.strategy, "max_new_tokens": 512,
                "backend": args.backend,
                "code_hashes": code_hashes,
            }
            if manifest_path.exists() and load_json(manifest_path) != manifest:
                raise ValueError("Existing manifest does not match current inputs or configuration")
            if result_path.exists():
                saved = load_json(result_path)
                if args.strategy == "evidence_probe_v1":
                    valid = bool(validate_probe(saved, claim, evidence))
                else:
                    valid = Reasoner.validate_json_structure(saved)
                if not manifest_path.exists() or not valid:
                    raise ValueError("Existing result is invalid or lacks matching provenance; not overwriting")
                print(f"Skipping matching valid result: {video_id}", flush=True)
                continue
            if not manifest_path.exists():
                manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            if args.strategy == "evidence_probe_v1":
                prompt_path = folder / "probe_prompt.json"
                raw_path = folder / "probe_raw_response.txt"
                if prompt_path.exists() or raw_path.exists():
                    raise ValueError("Existing probe attempt retained; use a new run name to retry")
                prompt = build_probe_prompt(claim, video_info, evidence)
                prompt_path.write_text(prompt, encoding="utf-8")
                response = Reasoner.gpt4o_mini_analysis(prompt)
                raw_path.write_text(response, encoding="utf-8")
                probe = parse_probe(response, claim, evidence)
                with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=folder, suffix=".tmp", delete=False) as handle:
                    temporary = Path(handle.name)
                    json.dump(probe, handle, ensure_ascii=False, indent=2)
                temporary.replace(result_path)
                temporary = None
                print(f"Completed probe {video_id}: {len(probe['observations'])} literal quotations; relations require review", flush=True)
                continue
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=folder, suffix=".tmp", delete=False) as handle:
                temporary = Path(handle.name)
                json.dump({"Claim": claim, "Video_information": video_info}, handle, ensure_ascii=False)
            verdict = Reasoner.process_claim_final(claim, video_info, evidence, str(temporary))
            if not Reasoner.validate_json_structure(verdict):
                raise ValueError("No valid final judgment")
            temporary.replace(result_path)
            temporary = None
            print(f"Completed {video_id}", flush=True)
        except (OSError, ValueError, TypeError, KeyError, RuntimeError) as error:
            failures += 1
            print(f"FAILED {video_id}: {type(error).__name__}: {error}", flush=True)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    print(f"Finished: {len(ids) - failures}/{len(ids)} completed or skipped; failures: {failures}", flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()