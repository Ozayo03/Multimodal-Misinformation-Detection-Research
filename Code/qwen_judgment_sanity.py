"""Synthetic inference diagnostics, not TRUE dataset evaluation."""
import argparse
import hashlib
import json
import platform
from pathlib import Path


CASES = (
    {"id": "direct_support", "claim": "The parcel weighs two kilograms.",
     "evidence": "The parcel weighs exactly two kilograms.", "expected": "True"},
    {"id": "direct_contradiction", "claim": "The parcel weighs two kilograms.",
     "evidence": "The parcel weighs exactly five kilograms, not two kilograms.", "expected": "False"},
    {"id": "reported_belief_support", "claim": "Staff believed the bell predicted arrivals.",
     "evidence": "Staff said they believed the bell predicted arrivals. No study tested the bell's predictive ability.",
     "expected": "True"},
    {"id": "reported_belief_contradiction", "claim": "Staff believed the bell predicted arrivals.",
     "evidence": "Every staff member explicitly denied believing the bell predicted arrivals.", "expected": "False"},
    {"id": "fiction_in_book_support", "claim": "The fictional story 'The Silver Tree' appears in a published book.",
     "evidence": "The published book 'Garden Tales' contains the fictional story 'The Silver Tree'.",
     "expected": "True"},
    {"id": "fiction_in_book_contradiction", "claim": "The fictional story 'The Silver Tree' appears in a published book.",
     "evidence": "The fictional story 'The Silver Tree' has never appeared in any published book.", "expected": "False"},
)


def build_prompt(case, order):
    return json.dumps({
        "Claim": case["claim"], "Evidence": case["evidence"],
        "Task": "Treat Evidence as given for this synthetic exercise. Does it support the exact Claim? "
                "Preserve attribution and qualifiers. Return only JSON with Answer and Reasons; Reasons at most 40 words.",
        "Allowed_answers": list(order),
    })


def parse_answer(raw):
    text = raw.strip()
    if text.startswith("```json\n") and text.endswith("```"):
        text = text[len("```json\n"):-3].strip()
    parsed = json.loads(text)
    if not isinstance(parsed, dict) or set(parsed) != {"Answer", "Reasons"}:
        raise ValueError("Expected only Answer and Reasons")
    if parsed["Answer"] not in ("True", "False"):
        raise ValueError("Answer must be True or False")
    if not isinstance(parsed["Reasons"], str) or not parsed["Reasons"].strip():
        raise ValueError("Reasons must be a nonempty string")
    return parsed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True, help="New directory; existing directories are refused")
    args = parser.parse_args()
    if not args.model.is_dir():
        parser.error("--model must be an existing local directory")
    if args.out_dir.exists():
        parser.error("--out-dir already exists; use a new name to preserve prior results")

    import torch
    import transformers
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from qwen_pilot import generate

    if not torch.cuda.is_available():
        raise SystemExit("No CUDA GPU available; submit to a GPU compute node")
    args.out_dir.mkdir(parents=True, exist_ok=False)
    dtype = torch.bfloat16 if torch.cuda.get_device_capability(0)[0] >= 8 else torch.float16
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(args.model), torch_dtype=dtype, local_files_only=True, attn_implementation="sdpa",
    ).to("cuda").eval()
    processor = AutoProcessor.from_pretrained(str(args.model), local_files_only=True, use_fast=False)
    manifest = {
        "purpose": "synthetic diagnostic only; no dataset accuracy claims",
        "model": str(args.model), "gpu": torch.cuda.get_device_name(0), "dtype": str(dtype),
        "python": platform.python_version(), "torch": torch.__version__,
        "transformers": transformers.__version__, "max_new_tokens": 512, "do_sample": False,
        "attention": "sdpa", "chat_template": processor.chat_template,
        "model_class": type(model).__name__, "processor_class": type(processor).__name__,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "generator_sha256": hashlib.sha256(Path(__file__).with_name("qwen_pilot.py").read_bytes()).hexdigest(),
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    results = []
    errors = 0
    for case in CASES:
        for order in (("True", "False"), ("False", "True")):
            name = f"{case['id']}_{order[0].lower()}_first"
            folder = args.out_dir / name
            folder.mkdir()
            prompt = build_prompt(case, order)
            content = [{"type": "text", "text": "You are a helpful assistant.\n\n" + prompt}]
            rendered = processor.apply_chat_template(
                [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True,
            )
            (folder / "prompt.json").write_text(prompt, encoding="utf-8")
            (folder / "rendered_prompt.txt").write_text(rendered, encoding="utf-8")
            record = {"id": case["id"], "answer_order": list(order), "expected": case["expected"],
                      "answer": None, "correct": False}
            try:
                raw = generate(model, processor, content, max_new_tokens=512)
                (folder / "raw_response.txt").write_text(raw, encoding="utf-8")
                parsed = parse_answer(raw)
                record.update(answer=parsed["Answer"], reasons=parsed["Reasons"],
                              correct=parsed["Answer"] == case["expected"])
            except (ValueError, TypeError, RuntimeError) as error:
                errors += 1
                record["error"] = f"{type(error).__name__}: {error}"
            (folder / "result.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
            results.append(record)
            print(f"{name}: expected={record['expected']} answer={record['answer']} correct={record['correct']}", flush=True)
    summary = {"attempted": len(results), "valid": len(results) - errors,
               "correct": sum(record["correct"] for record in results), "errors": errors,
               "order_changes": sum(results[index]["answer"] != results[index + 1]["answer"]
                                    for index in range(0, len(results), 2)
                                    if results[index]["answer"] is not None and results[index + 1]["answer"] is not None)}
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary), flush=True)
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()