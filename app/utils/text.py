"""Text processing utilities: truncation, stringify, normalization."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List


def truncate_text(text: Any, max_chars: int = 2000) -> str:
    if text is None:
        return ""
    if not isinstance(text, str):
        try:
            text = json.dumps(text, ensure_ascii=False)
        except Exception:
            text = str(text)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + " ..."


def stringify_message_content(msg: Any) -> str:
    if msg is None:
        return ""
    if isinstance(msg, str):
        return msg
    if isinstance(msg, list):
        parts = []
        for x in msg:
            if isinstance(x, dict) and "text" in x:
                parts.append(str(x.get("text", "")))
            else:
                parts.append(stringify_message_content(x))
        return " ".join([p for p in parts if p])
    if isinstance(msg, dict):
        if "content" in msg and isinstance(msg["content"], str):
            return msg["content"]
        try:
            return json.dumps(msg, ensure_ascii=False)
        except Exception:
            return str(msg)
    return str(msg)


def normalize_chat_or_text(raw: Any) -> List[Dict[str, Any]]:
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict):
        if "role" in raw and "content" in raw:
            return [raw]
    if isinstance(raw, str):
        raw_strip = raw.strip()
        if raw_strip.startswith("[") or raw_strip.startswith("{"):
            try:
                parsed = json.loads(raw_strip)
                if isinstance(parsed, list):
                    return parsed
                if isinstance(parsed, dict) and "role" in parsed and "content" in parsed:
                    return [parsed]
            except Exception:
                pass
        return [{"role": "user", "content": raw_strip}]
    return [{"role": "user", "content": str(raw)}]


def normalize_text(text: Any) -> str:
    q = "" if text is None else str(text)
    q = q.replace("\r\n", "\n").replace("\r", "\n")
    q = re.sub(r"^\[[^\]]+\]\s*", "", q.strip())
    q = " ".join(q.split()).strip()
    return q


def join_nonempty(parts: List[str], sep: str = "\n") -> str:
    values = [normalize_text(x) for x in parts if normalize_text(x)]
    return sep.join(values)


def extract_tools_from_text(text: str, max_items: int = 20) -> str:
    text = text or ""
    names: List[str] = []
    for m in re.finditer(r"toolCall\(name=['\"]([^'\"]+)['\"]", text):
        names.append(m.group(1))
    heuristics = ["read", "write", "exec", "search", "browser", "python", "shell", "web_search", "web_fetch"]
    low = text.lower()
    for h in heuristics:
        if re.search(rf"\b{re.escape(h)}\b", low):
            names.append(h)
    uniq: List[str] = []
    seen: set[str] = set()
    for n in names:
        if n not in seen:
            seen.add(n)
            uniq.append(n)
    if uniq:
        return "\n".join([f"- {x}" for x in uniq[:max_items]])
    return ""
