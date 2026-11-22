"""Run Qwen2.5-VL text and video-frame inference on an allocated GPU."""
import argparse
import os
from pathlib import Path

import torch
from decord import VideoReader, cpu
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


def generate(model, processor, content):
    messages = [{"role": "user", "content": content}]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images, videos = process_vision_info(messages)
    inputs = processor(text=[prompt], images=images, videos=videos, padding=True, return_tensors="pt")
    inputs = inputs.to(model.device)
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=128, do_sample=False)
    generated = [result[len(original):] for original, result in zip(inputs.input_ids, output)]
    return processor.batch_decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=os.getenv("QWEN_MODEL_PATH"), required=not os.getenv("QWEN_MODEL_PATH"))
    parser.add_argument("--video", required=True, help="Path to one locally accessible dataset video")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("No CUDA GPU is available; run this test in a GPU compute allocation.")
    if not Path(args.model).is_dir():
        raise SystemExit(f"Model directory does not exist: {args.model}")
    if not Path(args.video).is_file():
        raise SystemExit(f"Video does not exist: {args.video}")

    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"PyTorch: {torch.__version__}; CUDA runtime: {torch.version.cuda}", flush=True)
    major, _ = torch.cuda.get_device_capability(0)
    dtype = torch.bfloat16 if major >= 8 else torch.float16
    print(f"Precision: {dtype}; model: {args.model}", flush=True)
    reader = VideoReader(args.video, ctx=cpu(0))
    if len(reader) == 0:
        raise ValueError(f"Video contains no decodable frames: {args.video}")
    frame = Image.fromarray(reader[len(reader) // 2].asnumpy()).convert("RGB")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=dtype, local_files_only=True, attn_implementation="sdpa"
    ).to("cuda").eval()
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)

    text_response = generate(model, processor, [{"type": "text", "text": "Reply with the word READY."}])
    if not text_response.strip():
        raise RuntimeError("The model returned an empty text response.")
    print(f"TEXT RESPONSE: {text_response}", flush=True)
    image_response = generate(model, processor, [
        {"type": "image", "image": frame},
        {"type": "text", "text": "Briefly describe what is visible in this video frame."},
    ])
    if not image_response.strip():
        raise RuntimeError("The model returned an empty image response.")
    print(f"VIDEO FRAME: {args.video}", flush=True)
    print(f"IMAGE RESPONSE: {image_response}", flush=True)
    print("QWEN SMOKE TEST PASSED", flush=True)


if __name__ == "__main__":
    main()