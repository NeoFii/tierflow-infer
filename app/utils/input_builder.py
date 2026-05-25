"""Route input builders: shared_record and full LLM input from chat messages.

v3: no proxy format support; enhanced tool_calls / tool role handling.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple, Union

from app.utils.text import (
    join_nonempty,
    normalize_chat_or_text,
    normalize_text,
    stringify_message_content,
    truncate_text,
    extract_tools_from_text,
)


# ---------------------------------------------------------------------------
# Canonical text builders (for tool route)
# ---------------------------------------------------------------------------
def canonicalize_shared_record(rec: Dict[str, Any]) -> str:
    parts = []
    parts.append(f"Instruction: {rec.get('instruction', '') or 'N/A'}")
    parts.append(f"Input: Available actions/tools: {rec.get('action_space', '') or 'N/A'}")
    parts.append(f"User: {rec.get('task', '') or 'N/A'}")
    parts.append(f"Previous context: {rec.get('context', '') or 'N/A'}")
    parts.append(f"Latest observation: {rec.get('observation', '') or 'N/A'}")
    parts.append(f"State: hasLastStep={str(bool(rec.get('has_lastStep', False))).lower()}")
    return "\n".join(parts).strip()


def build_tool_canonical_text_from_text(user_text: Any) -> str:
    text = normalize_text(user_text)
    return f"User: {text or 'N/A'}"


# ---------------------------------------------------------------------------
# shared_record_from_chat_messages — v3 enhanced
# ---------------------------------------------------------------------------
def shared_record_from_chat_messages(
    messages: List[Dict[str, Any]],
    request_id: str = "chat",
) -> Dict[str, Any]:
    normalized = normalize_chat_or_text(messages)
    system_msgs: List[str] = []
    user_msgs: List[str] = []
    assistant_msgs: List[str] = []
    tool_msgs: List[str] = []
    tool_names: List[str] = []

    for msg in normalized:
        role = str(msg.get("role", "")).lower()
        content = truncate_text(stringify_message_content(msg.get("content", "")), 1800)

        if role == "system":
            if content:
                system_msgs.append(content)
        elif role == "user":
            if content:
                user_msgs.append(content)
        elif role == "assistant":
            # v3: handle tool_calls in assistant messages
            tc = msg.get("tool_calls")
            if tc and isinstance(tc, list):
                names = [
                    t.get("function", {}).get("name", "")
                    for t in tc if isinstance(t, dict)
                ]
                names = [n for n in names if n]
                tool_names.extend(names)
                assistant_msgs.append(f"[tool_calls: {' '.join(names)}]")
            elif content:
                assistant_msgs.append(content)
        elif role == "tool":
            # v3: handle tool role messages
            if content:
                tool_msgs.append(truncate_text(content, 800))

    instruction = "\n".join(system_msgs).strip() or "Chat completion routing for the current task."
    task = user_msgs[-1] if user_msgs else "N/A"

    recent_parts: List[str] = []
    for msg in normalized[-6:]:
        role = str(msg.get("role", "user")).lower()
        content = truncate_text(stringify_message_content(msg.get("content", "")), 600)
        if role == "assistant" and not content:
            tc = msg.get("tool_calls")
            if tc and isinstance(tc, list):
                names = [t.get("function", {}).get("name", "") for t in tc if isinstance(t, dict)]
                content = f"[tool_calls: {' '.join(n for n in names if n)}]"
        if content:
            recent_parts.append(f"{role.title()}: {content}")

    context = "\n".join(recent_parts[:-1]).strip() if len(recent_parts) > 1 else ""
    observation = assistant_msgs[-1] if assistant_msgs else ""

    # action_space: combine extracted tools from text + explicit tool_calls names
    action_space = extract_tools_from_text("\n".join(system_msgs + user_msgs + assistant_msgs))
    if tool_names:
        existing = set(action_space.split("\n")) if action_space else set()
        for name in tool_names:
            entry = f"- {name}"
            if entry not in existing:
                action_space = (action_space + "\n" + entry).strip() if action_space else entry

    shared = {
        "id": request_id,
        "source": "chat_messages",
        "instruction": instruction,
        "action_space": truncate_text(action_space, 2500),
        "task": truncate_text(task, 1200),
        "context": truncate_text(context, 1800),
        "observation": truncate_text(observation, 1800),
        "has_lastStep": bool(assistant_msgs),
        "time_text": "",
    }
    shared["canonical_text"] = build_tool_canonical_text_from_text(task)
    return shared


# ---------------------------------------------------------------------------
# build_full_llm_input_for_chat_messages — v3 enhanced
# ---------------------------------------------------------------------------
def build_full_llm_input_for_chat_messages(
    messages: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, str]], str]:
    chat: List[Dict[str, str]] = []
    for msg in normalize_chat_or_text(messages):
        role = str(msg.get("role", "user")).lower().strip()
        content = stringify_message_content(msg.get("content", ""))

        # v3: assistant with tool_calls but no content
        if role == "assistant" and not content:
            tc = msg.get("tool_calls")
            if tc and isinstance(tc, list):
                tc_names = [
                    t.get("function", {}).get("name", "")
                    for t in tc if isinstance(t, dict)
                ]
                content = f"[Calling tools: {', '.join(n for n in tc_names if n)}]"

        # v3: tool role → merge as context
        if role == "tool":
            content = f"[Tool result] {truncate_text(content, 1500)}"
            role = "assistant"

        content = truncate_text(content, 2500)
        if not content:
            continue
        if role not in ("system", "user", "assistant"):
            role = "user"
        chat.append({"role": role, "content": content})

    if not chat:
        chat = [{"role": "user", "content": "N/A"}]

    debug_text = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in chat])
    return chat, debug_text


# ---------------------------------------------------------------------------
# Proto semantic text builder
# ---------------------------------------------------------------------------
def build_proto_semantic_text(raw_input: Union[str, Dict[str, Any], List[Any]]) -> str:
    if isinstance(raw_input, list):
        if all(isinstance(x, dict) and "role" in x and "content" in x for x in raw_input):
            user_msgs = [
                normalize_text(stringify_message_content(x.get("content", "")))
                for x in raw_input
                if str(x.get("role", "")).lower() == "user"
            ]
            if user_msgs:
                return user_msgs[-1]
        return normalize_text(str(raw_input))

    if isinstance(raw_input, str):
        return normalize_text(raw_input)

    return normalize_text(str(raw_input))
