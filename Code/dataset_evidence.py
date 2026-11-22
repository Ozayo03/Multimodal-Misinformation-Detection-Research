"""Select source excerpts without consuming gold verdicts or rationale annotations."""
import re


def select_dataset_evidence(annotation, max_chars=12000):
    if max_chars <= 0:
        raise ValueError("DATASET_EVIDENCE_MAX_CHARS must be positive")
    evidence = annotation.get("evidences", {})
    if not isinstance(evidence, dict):
        raise ValueError("Expected an evidences dictionary")
    selected = []
    remaining = max_chars
    keys = sorted((key for key in evidence if re.fullmatch(r"evidence\d+", key)),
                  key=lambda key: int(key[len("evidence"):]))
    for key in keys:
        entry = evidence[key]
        if not isinstance(entry, list) or not entry or not isinstance(entry[0], str):
            continue
        excerpt = entry[0].strip()
        if not excerpt or not remaining:
            continue
        urls = entry[1] if len(entry) > 1 and isinstance(entry[1], list) else []
        selected.append({
            "id": key,
            "text": excerpt[:remaining],
            "urls": [url for url in urls if isinstance(url, str) and url.startswith(("https://", "http://"))],
            "truncated": len(excerpt) > remaining,
        })
        remaining -= len(selected[-1]["text"])
    return {
        "Dataset_evidence": selected,
        "Evidence_selection": {
            "policy": "numbered excerpts in numeric order; no gold annotations",
            "max_text_chars": max_chars,
            "used_text_chars": max_chars - remaining,
            "available_numbered_entries": len(keys),
        },
    }