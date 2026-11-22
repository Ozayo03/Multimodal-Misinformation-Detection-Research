"""Run an isolated Qwen pilot with MiniCPM's staged description and judgment prompts."""
import argparse
import json
import os
import random
from pathlib import Path

import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

import Reasoner
import VideoDescriptor
from dataset_evidence import select_dataset_evidence


def generate(model, processor, content, max_new_tokens=512):
    messages = [{"role": "user", "content": content}]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images, videos = process_vision_info(messages)
    inputs = processor(text=[prompt], images=images, videos=videos, padding=True, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    trimmed = [output[len(original):] for original, output in zip(inputs.input_ids, outputs)]
    answer = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    if not answer.strip():
        raise ValueError("Qwen returned an empty response")
    return answer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/datasets/ambatha/TRUE_Dataset"))
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--split", choices=("test", "train_val"), default="train_val")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--condition", choices=("none", "dataset"), required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--description-name", default="qwen_staged_v1")
    parser.add_argument("--video-id", help="Test one existing cached-description ID instead of sampling")
    parser.add_argument("--max-pixels", type=int, default=128 * 28 * 28)
    args = parser.parse_args()
    for name in (args.run_name, args.description_name):
        if not name or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for char in name):
            parser.error("Run and description names must contain only letters, digits, underscores or hyphens")
    if args.video_id and (not args.video_id.isascii() or not args.video_id.isdigit()):
        parser.error("--video-id must contain only ASCII digits")
    if args.limit < 1 or args.max_pixels < 1:
        parser.error("--limit and --max-pixels must be positive")
    if not args.model.is_dir():
        parser.error("--model must be an existing local directory")
    if not torch.cuda.is_available():
        raise SystemExit("No CUDA GPU available; submit this pilot to a GPU compute node")

    source = args.root / f"{args.split}_output"
    descriptions = args.root / f"{args.split}_qwen_descriptions_{args.description_name}"
    output = args.root / f"{args.split}_output_{args.run_name}"
    descriptions.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    files = sorted(source.glob("*.json"), reverse=True)
    random.Random(args.seed).shuffle(files)
    selected = [source / f"{args.video_id}.json"] if args.video_id else files[:args.limit]
    print(f"GPU: {torch.cuda.get_device_name(0)}; condition: {args.condition}; claims: {len(selected)}", flush=True)
    major, _ = torch.cuda.get_device_capability(0)
    dtype = torch.bfloat16 if major >= 8 else torch.float16
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(args.model), torch_dtype=dtype, local_files_only=True, attn_implementation="sdpa"
    ).to("cuda").eval()
    processor = AutoProcessor.from_pretrained(str(args.model), local_files_only=True, use_fast=False)

    def text_analysis(prompt):
        return generate(model, processor, [{"type": "text", "text": "You are a helpful assistant.\n\n" + prompt}])

    Reasoner.gpt4o_mini_analysis = text_analysis
    VideoDescriptor.gpt4o_mini_analysis = text_analysis
    os.environ["USE_EVIDENCE_RETRIEVAL"] = "false"
    os.environ["JUDGMENT_STRATEGY"] = "evidence_v2"
    os.environ["JUDGMENT_INPUT_MODE"] = "all"
    video_prompt = "Describe the key events in the video chronologically, including time, location, and participants. Focus on observable actions and processes, avoiding speculation. Provide a brief and accurate summary of the video content."
    image_prompt = "Accurately describe this image, including visible subjects, their actions, and the main scene or background information. Extract and describe any visible text. Only describe what can be directly observed in the image. Do not speculate on uncertain details, only describe the most certain elements."
    failed = 0
    for index, source_file in enumerate(selected, 1):
        video_id = source_file.stem
        directory = output / video_id
        result_path = directory / f"{video_id}_CV_result.json"
        print(f"Processing {index}/{len(selected)}: {video_id}", flush=True)
        try:
            if result_path.exists():
                try:
                    with result_path.open(encoding="utf-8") as handle:
                        if Reasoner.validate_json_structure(json.load(handle)):
                            print("Skipping valid verdict", flush=True)
                            continue
                except (ValueError, OSError):
                    pass
            data = json.loads(source_file.read_text(encoding="utf-8-sig"))
            video_dir = args.root / f"{args.split}_video"
            video_path = next((video_dir / f"{video_id}{ext}" for ext in (".mp4", ".mkv") if (video_dir / f"{video_id}{ext}").is_file()), None)
            if video_path is None:
                raise FileNotFoundError(f"Missing video for {video_id}")
            frame_paths = [video_dir / video_id / f"{number}.jpeg" for number in range(1, 8)]
            if not all(path.is_file() for path in frame_paths):
                raise FileNotFoundError(f"Missing cached keyframes for {video_id}")
            description_path = descriptions / f"{video_id}.json"
            if description_path.exists():
                descriptor = json.loads(description_path.read_text(encoding="utf-8"))["Video_descriptor"]
            else:
                frames = VideoDescriptor.encode_video(str(video_path))
                video_content = [{"type": "image", "image": frame, "max_pixels": args.max_pixels} for frame in frames]
                video_content.append({"type": "text", "text": video_prompt})
                video_summary = generate(model, processor, video_content)
                frame_summaries = []
                for path in frame_paths:
                    with Image.open(path) as image:
                        answer = generate(model, processor, [{"type": "image", "image": image.convert("RGB"), "max_pixels": args.max_pixels}, {"type": "text", "text": image_prompt}])
                    frame_summaries.append(f"{path.name}: {answer}")
                descriptor = VideoDescriptor.gpt_summary(video_summary, "; ".join(frame_summaries))
                temporary = description_path.with_suffix(".json.tmp")
                temporary.write_text(json.dumps({"Video_descriptor": descriptor}, ensure_ascii=False, indent=2), encoding="utf-8")
                temporary.replace(description_path)
            video_info = dict(data["Video_information"], Video_descriptor=descriptor)
            evidence = {}
            if args.condition == "dataset":
                annotation = json.loads((args.root / args.split / f"{video_id}.json").read_text(encoding="utf-8-sig"))
                evidence = select_dataset_evidence(annotation, 12000)
            directory.mkdir(parents=True, exist_ok=True)
            manifest = {"model": str(args.model), "condition": args.condition, "video": str(video_path), "keyframes": [str(path) for path in frame_paths], "video_max_frames": VideoDescriptor.MAX_NUM_FRAMES, "max_pixels": args.max_pixels, "seed": args.seed, "evidence": evidence}
            (directory / "qwen_input_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
            result_path.write_text(json.dumps({"Claim": data["claim"], "Video_information": video_info}, ensure_ascii=False), encoding="utf-8")
            verdict = Reasoner.process_claim_final(data["claim"], video_info, evidence, str(result_path))
            if not Reasoner.validate_json_structure(verdict):
                raise ValueError("No valid final judgment")
            print(f"Completed {video_id}", flush=True)
        except Exception as error:
            failed += 1
            print(f"FAILED {video_id}: {type(error).__name__}: {error}", flush=True)
    print(f"Finished: {len(selected) - failed}/{len(selected)} processed or skipped; failures: {failed}", flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()