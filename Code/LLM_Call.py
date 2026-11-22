import json
import threading
import time

import regex
import requests
import torch
from PIL import Image
from decord import VideoReader, cpu
from transformers import AutoModel, AutoTokenizer

from Config import MODEL_CONFIG


_MINICPM_TEXT_MODEL = None
_MINICPM_TEXT_TOKENIZER = None
_MINICPM_TEXT_MODEL_PATH = None
_MINICPM_TEXT_LOCK = threading.Lock()

_MINICPM_VIDEO_MODEL = None
_MINICPM_VIDEO_TOKENIZER = None
_MINICPM_VIDEO_MODEL_PATH = None
_MINICPM_VIDEO_LOCK = threading.Lock()


def preferred_torch_dtype():
    if not torch.cuda.is_available():
        return torch.float32
    # bf16 is only hardware-accelerated on Ampere (sm_80) and newer.
    major, _ = torch.cuda.get_device_capability()
    return torch.bfloat16 if major >= 8 else torch.float16


def _resolve_llm_local_model_path():
    llm_cfg = MODEL_CONFIG.get("llm", {})
    llm_local_path = llm_cfg.get("local_model_path", "")
    if llm_local_path:
        return llm_local_path
    return MODEL_CONFIG.get("video_lmm", {}).get("local_model_path", "")


def _load_minicpm_text_model():
    global _MINICPM_TEXT_MODEL
    global _MINICPM_TEXT_TOKENIZER
    global _MINICPM_TEXT_MODEL_PATH

    model_path = _resolve_llm_local_model_path()
    if not model_path:
        raise ValueError(
            "MiniCPM text model path is empty. Set MODEL_CONFIG['llm']['local_model_path'] "
            "or MODEL_CONFIG['video_lmm']['local_model_path']."
        )

    if (
        _MINICPM_TEXT_MODEL is not None
        and _MINICPM_TEXT_TOKENIZER is not None
        and _MINICPM_TEXT_MODEL_PATH == model_path
    ):
        return _MINICPM_TEXT_MODEL, _MINICPM_TEXT_TOKENIZER

    torch_dtype = preferred_torch_dtype()
    model = AutoModel.from_pretrained(
        model_path,
        trust_remote_code=True,
        attn_implementation="sdpa",
        torch_dtype=torch_dtype,
    )
    model = model.eval().cuda() if torch.cuda.is_available() else model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    _MINICPM_TEXT_MODEL = model
    _MINICPM_TEXT_TOKENIZER = tokenizer
    _MINICPM_TEXT_MODEL_PATH = model_path
    return _MINICPM_TEXT_MODEL, _MINICPM_TEXT_TOKENIZER


def minicpm_text_analysis(prompt, system_prompt="You are a helpful assistant."):
    with _MINICPM_TEXT_LOCK:
        model, tokenizer = _load_minicpm_text_model()

    msgs = [
        {
            "role": "user",
            "content": f"{system_prompt}\n\n{prompt}",
        }
    ]

    return model.chat(
        image=None,
        msgs=msgs,
        tokenizer=tokenizer,
        sampling=False,
        max_new_tokens=MODEL_CONFIG.get("llm", {}).get("max_new_tokens", 512),
    )


def gpt4o_mini_analysis(prompt):
    llm_cfg = MODEL_CONFIG.get("llm", {})
    model_type = llm_cfg.get("model_type", "").lower()

    if model_type == "minicpm":
        return minicpm_text_analysis(prompt)

    url = llm_cfg.get("api_base", "https://cn2us02.opapi.win/v1/chat/completions")
    api_key = llm_cfg.get("model_key", "")
    model_name = llm_cfg.get("model_name", "gpt-4o-mini")

    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
    }

    headers = {
        "User-Agent": "Apifox/1.0.0 (https://apifox.com)",
        "Content-Type": "application/json",
    }
    if api_key:
        headers["Authorization"] = "Bearer " + api_key

    last_error = None
    for attempt in range(3):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=(10, 180))
            response.raise_for_status()
            res = response.json()
            return res["choices"][0]["message"]["content"]
        except (requests.RequestException, KeyError, IndexError, ValueError) as exc:
            last_error = exc
            if attempt < 2:
                time.sleep(1.5 * (attempt + 1))

    raise RuntimeError(f"LLM API request failed after retries: {last_error}")


def extract_complete_json(response_text):
    json_pattern = r"(\{(?:[^{}]|(?1))*\})"
    matches = regex.findall(json_pattern, response_text or "")
    for match in matches:
        try:
            return json.loads(match)
        except json.JSONDecodeError as e:
            print(f"JSON parse error: {e}")
        # Models often emit stray backslashes (e.g. \_ or \'), which are invalid JSON escapes.
        repaired = regex.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', match)
        try:
            return json.loads(repaired)
        except json.JSONDecodeError:
            continue
    return None


def analysis_video_minicpm(video_path, question):
    model_path = MODEL_CONFIG["video_lmm"].get("local_model_path", "")
    if not model_path:
        raise ValueError("MODEL_CONFIG['video_lmm']['local_model_path'] is empty.")

    def get_video_model_and_tokenizer(path):
        global _MINICPM_VIDEO_MODEL
        global _MINICPM_VIDEO_TOKENIZER
        global _MINICPM_VIDEO_MODEL_PATH

        if (
            _MINICPM_VIDEO_MODEL is not None
            and _MINICPM_VIDEO_TOKENIZER is not None
            and _MINICPM_VIDEO_MODEL_PATH == path
        ):
            return _MINICPM_VIDEO_MODEL, _MINICPM_VIDEO_TOKENIZER

        torch_dtype = preferred_torch_dtype()
        model = AutoModel.from_pretrained(
            path,
            trust_remote_code=True,
            attn_implementation="sdpa",
            torch_dtype=torch_dtype,
        )
        model = model.eval().cuda() if torch.cuda.is_available() else model.eval()
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)

        _MINICPM_VIDEO_MODEL = model
        _MINICPM_VIDEO_TOKENIZER = tokenizer
        _MINICPM_VIDEO_MODEL_PATH = path

        return _MINICPM_VIDEO_MODEL, _MINICPM_VIDEO_TOKENIZER

    def uniform_sample(items, n):
        gap = len(items) / n
        idxs = [int(i * gap + gap / 2) for i in range(n)]
        return [items[i] for i in idxs]

    def encode_video(path):
        max_num_frames = 64
        vr = VideoReader(path, ctx=cpu(0))
        sample_fps = max(1, round(vr.get_avg_fps()))
        if not len(vr):
            raise ValueError(f"Video contains no decodable frames: {path}")
        frame_idx = [i for i in range(0, len(vr), sample_fps)]
        if len(frame_idx) > max_num_frames:
            frame_idx = uniform_sample(frame_idx, max_num_frames)
        frames = vr.get_batch(frame_idx).asnumpy()
        frames = [Image.fromarray(v.astype("uint8")) for v in frames]
        print(f"Video: {path}, num frames: {len(frames)}")
        return frames

    try:
        with _MINICPM_VIDEO_LOCK:
            model, tokenizer = get_video_model_and_tokenizer(model_path)

        frames = encode_video(video_path)
        msgs = [{"role": "user", "content": frames + [question]}]
        params = {
            "use_image_id": False,
            "max_slice_nums": 1,
        }

        answer = model.chat(
            image=None,
            msgs=msgs,
            tokenizer=tokenizer,
            **params,
        )
        return answer

    except Exception as e:
        print(f"Error while processing video: {str(e)}")
        return None
