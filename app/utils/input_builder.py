"""Route input builders: shared_record and full LLM input from chat messages.

v3: no proxy format support; enhanced tool_calls / tool role handling.
v4: unified routing_input — Original Task + Previous Input/Output/ToolCalls/ToolResults.
"""

from __future__ import annotations

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
# v4: Unified routing input builder
# ---------------------------------------------------------------------------

def _serialize_tool_calls(tool_calls: list) -> str:
    parts = []
    for tc in (tool_calls or []):
        if not isinstance(tc, dict):
            continue
        func = tc.get("function", {})
        name = func.get("name", "unknown")
        args = truncate_text(func.get("arguments", ""), 500)
        parts.append(f"- {name}({args})")
    return "\n".join(parts) if parts else "N/A"


def _serialize_tool_results(tool_msgs: List[Dict[str, Any]]) -> str:
    parts = []
    for msg in tool_msgs:
        name = msg.get("name", "")
        content = truncate_text(stringify_message_content(msg.get("content", "")), 800)
        label = f"({name}) " if name else ""
        parts.append(f"- {label}→ {content}")
    return "\n".join(parts) if parts else "N/A"


def _serialize_assistant_block(assistant_msg: Dict[str, Any], following_tools: List[Dict[str, Any]]) -> str:
    parts = []
    content = truncate_text(stringify_message_content(assistant_msg.get("content", "")), 1500)
    if content:
        parts.append(content)
    tc = assistant_msg.get("tool_calls")
    if tc:
        parts.append("[Tool Calls]\n" + _serialize_tool_calls(tc))
    if following_tools:
        parts.append("[Tool Results]\n" + _serialize_tool_results(following_tools))
    return "\n".join(parts) if parts else "N/A"


def build_routing_input(messages: List[Dict[str, Any]]) -> str:
    """Extract structured routing input from OpenAI messages.

    Returns a text block with sections:
      [Original Task]      — first user message
      [Previous Input]     — what the previous LLM call saw = the turn before last
      [Previous Output]    — last assistant output + tool calls
      [Previous Tool Calls]  — tool_calls from last assistant
      [Previous Tool Results] — tool role messages after last assistant
    """
    normalized = normalize_chat_or_text(messages)

    non_system = [m for m in normalized if str(m.get("role", "")).lower() != "system"]

    original_task = "N/A"
    for m in non_system:
        if str(m.get("role", "")).lower() == "user":
            original_task = truncate_text(stringify_message_content(m.get("content", "")), 1500)
            break

    assistant_indices = [i for i, m in enumerate(non_system) if str(m.get("role", "")).lower() == "assistant"]

    if not assistant_indices:
        return f"[Original Task]\n{original_task}"

    def _tools_after(start_idx: int, end_idx: int) -> List[Dict[str, Any]]:
        tools = []
        for i in range(start_idx + 1, end_idx):
            if str(non_system[i].get("role", "")).lower() == "tool":
                tools.append(non_system[i])
        return tools

    last_ast_idx = assistant_indices[-1]
    last_ast_msg = non_system[last_ast_idx]
    last_tools = _tools_after(last_ast_idx, len(non_system))

    prev_output = truncate_text(stringify_message_content(last_ast_msg.get("content", "")), 1500) or "N/A"
    prev_tool_calls = _serialize_tool_calls(last_ast_msg.get("tool_calls"))
    prev_tool_results = _serialize_tool_results(last_tools)

    prev_input = "N/A"
    if len(assistant_indices) >= 2:
        prev_ast_idx = assistant_indices[-2]
        prev_ast_msg = non_system[prev_ast_idx]
        boundary = last_ast_idx
        prev_ast_tools = _tools_after(prev_ast_idx, boundary)
        prev_input = _serialize_assistant_block(prev_ast_msg, prev_ast_tools)

    sections = [
        f"[Original Task]\n{original_task}",
        f"[Previous Input]\n{prev_input}",
        f"[Previous Output]\n{prev_output}",
        f"[Previous Tool Calls]\n{prev_tool_calls}",
        f"[Previous Tool Results]\n{prev_tool_results}",
    ]
    return "\n\n".join(sections)


# ---------------------------------------------------------------------------
# Legacy canonical text builders (kept for tests)
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
# shared_record_from_chat_messages — v4: uses build_routing_input
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
    shared["canonical_text"] = build_routing_input(messages)
    return shared


# ---------------------------------------------------------------------------
# build_full_llm_input_for_chat_messages — v4: wraps build_routing_input
# ---------------------------------------------------------------------------
def build_full_llm_input_for_chat_messages(
    messages: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, str]], str]:
    routing_text = build_routing_input(messages)
    chat = [{"role": "user", "content": routing_text}]
    return chat, routing_text


# ---------------------------------------------------------------------------
# Proto semantic text builder
# ---------------------------------------------------------------------------
def build_proto_semantic_text(raw_input: Union[str, Dict[str, Any], List[Any]]) -> str:
    if isinstance(raw_input, list):
        if all(isinstance(x, dict) and "role" in x and "content" in x for x in raw_input):
            return build_routing_input(raw_input)
        return normalize_text(str(raw_input))

    if isinstance(raw_input, str):
        return normalize_text(raw_input)

    return normalize_text(str(raw_input))
