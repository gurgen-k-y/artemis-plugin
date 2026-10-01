"""Shared message, image, and structured tool-call handling for local clients."""

from __future__ import annotations

import base64
from collections.abc import Callable, Sequence
from io import BytesIO
import json
from pathlib import Path
import tempfile
from typing import Any
from urllib.parse import unquote_to_bytes
from uuid import uuid4

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from PIL import Image, ImageOps, UnidentifiedImageError


MAX_IMAGE_EDGE = 1600
MAX_IMAGE_BYTES = 768 * 1024


def format_tools(
    tools: Sequence[dict[str, Any] | type | Callable[..., Any] | BaseTool],
) -> list[dict[str, Any]]:
    return [convert_to_openai_tool(tool) for tool in tools]


def response_contract(tools: list[dict[str, Any]], tool_choice: Any) -> tuple[dict[str, Any], str]:
    if not tools:
        return (
            {
                "type": "object",
                "properties": {"content": {"type": "string"}},
                "required": ["content"],
                "additionalProperties": False,
            },
            "Return a JSON object with one string field named content.",
        )
    required, forced_name = _tool_choice(tool_choice)
    available = [tool["function"]["name"] for tool in tools]
    allowed = [forced_name] if forced_name else available
    if forced_name and forced_name not in available:
        raise ValueError(f"Unknown forced tool {forced_name!r}; available tools: {available}")
    schema = {
        "type": "object",
        "properties": {
            "kind": {
                "type": "string",
                "enum": ["tool_call"] if required else ["tool_call", "final"],
            },
            "content": {"type": "string"},
            "tool_name": {"type": "string", "enum": allowed},
            "tool_arguments_json": {"type": "string"},
        },
        "required": ["kind", "content", "tool_name", "tool_arguments_json"],
        "additionalProperties": False,
    }
    prompt = (
        "Artemis executes tools. Select from the supplied schemas and return only the required "
        "JSON object. For kind=tool_call, encode the arguments object in tool_arguments_json. "
        "For kind=final, return the answer in content and use the first tool name with '{}'."
    )
    return schema, prompt


def parse_response(text: str) -> dict[str, Any]:
    value = text.strip()
    if value.startswith("```"):
        lines = value.splitlines()[1:]
        if lines and lines[-1].strip() == "```":
            lines.pop()
        value = "\n".join(lines)
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {"content": text}
    return parsed if isinstance(parsed, dict) else {"content": str(parsed)}


def parse_tool_call(parsed: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    name = str(parsed.get("tool_name") or "")
    raw_args = parsed.get("tool_arguments_json") or "{}"
    if isinstance(raw_args, str):
        try:
            args = json.loads(raw_args)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON arguments for tool {name!r}: {raw_args}") from exc
    else:
        args = raw_args
    if not isinstance(args, dict):
        raise ValueError(f"Tool arguments for {name!r} must be an object")
    return name, args


def system_instructions(messages: list[BaseMessage]) -> str:
    parts: list[str] = []
    for message in messages:
        if isinstance(message, SystemMessage):
            text, _images, _temp = content_parts(message.content, materialize_images=False)
            parts.extend(text)
    return (
        "\n\n".join(parts) or "Follow the user instructions and return only the requested result."
    )


def message_transcript(
    messages: list[BaseMessage],
) -> tuple[str, list[dict[str, Any]], list[Path]]:
    sections: list[str] = []
    images: list[dict[str, Any]] = []
    temp_paths: list[Path] = []
    for message in messages:
        if isinstance(message, SystemMessage):
            continue
        texts, message_images, paths = content_parts(message.content)
        images.extend(message_images)
        temp_paths.extend(paths)
        if isinstance(message, HumanMessage):
            role = "USER"
        elif isinstance(message, ToolMessage):
            role = f"TOOL RESULT ({message.name or message.tool_call_id})"
        elif isinstance(message, AIMessage):
            role = "ASSISTANT"
            if message.tool_calls:
                texts.append("Tool calls: " + json.dumps(message.tool_calls, default=str))
        else:
            role = message.type.upper()
        sections.append(f"[{role}]\n" + "\n".join(texts))
    return "\n\n".join(sections), images, temp_paths


def content_parts(
    content: Any, *, materialize_images: bool = True
) -> tuple[list[str], list[dict[str, Any]], list[Path]]:
    if isinstance(content, str):
        return ([content] if content else []), [], []
    if not isinstance(content, list):
        return [str(content)], [], []
    texts: list[str] = []
    images: list[dict[str, Any]] = []
    paths: list[Path] = []
    for block in content:
        if isinstance(block, str):
            texts.append(block)
            continue
        if not isinstance(block, dict):
            texts.append(str(block))
            continue
        if block.get("type") in {"text", "input_text"}:
            texts.append(str(block.get("text") or ""))
            continue
        image_url = _image_url(block)
        if image_url and materialize_images:
            image, temporary = prepare_image(image_url)
            paths.extend([image] if temporary else [])
            images.append({"type": "localImage", "path": str(image)})
            continue
        texts.append(json.dumps(block, default=str))
    return texts, images, paths


def prepare_image(value: str) -> tuple[Path, bool]:
    if value.lower().startswith("data:image/"):
        header, separator, payload = value.partition(",")
        if not separator:
            raise ValueError("Invalid image data URL")
        raw = (
            base64.b64decode(payload) if ";base64" in header.lower() else unquote_to_bytes(payload)
        )
        suffix = {"image/png": ".png", "image/jpeg": ".jpg", "image/webp": ".webp"}.get(
            header[5:].split(";", 1)[0].lower(), ".img"
        )
        prepared, suffix = _compress_image(raw, suffix)
        return _write_temp_image(prepared, suffix), True
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"Local client image must be a data URL or existing file: {value}")
    raw = path.read_bytes()
    prepared, suffix = _compress_image(raw, path.suffix or ".img")
    if prepared == raw:
        return path, False
    return _write_temp_image(prepared, suffix), True


def _compress_image(raw: bytes, suffix: str) -> tuple[bytes, str]:
    try:
        with Image.open(BytesIO(raw)) as source:
            image = ImageOps.exif_transpose(source).copy()
    except (OSError, UnidentifiedImageError, ValueError):
        return raw, suffix
    if max(image.size) <= MAX_IMAGE_EDGE and len(raw) <= MAX_IMAGE_BYTES:
        return raw, suffix
    if image.mode in {"RGBA", "LA"} or (image.mode == "P" and "transparency" in image.info):
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, "white")
        image = Image.alpha_composite(background, rgba).convert("RGB")
    else:
        image = image.convert("RGB")
    image.thumbnail((MAX_IMAGE_EDGE, MAX_IMAGE_EDGE), Image.Resampling.LANCZOS)
    quality = 82
    while True:
        output = BytesIO()
        image.save(output, format="JPEG", quality=quality, optimize=True)
        encoded = output.getvalue()
        if len(encoded) <= MAX_IMAGE_BYTES or quality <= 50:
            return encoded, ".jpg"
        quality -= 8


def _write_temp_image(raw: bytes, suffix: str) -> Path:
    directory = Path(tempfile.gettempdir()) / "artemis-client-images"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{uuid4().hex}{suffix}"
    path.write_bytes(raw)
    return path


def _image_url(block: dict[str, Any]) -> str | None:
    if block.get("type") not in {"image_url", "input_image", "image"}:
        return None
    value = block.get("image_url") or block.get("url")
    if isinstance(value, dict):
        value = value.get("url")
    source = block.get("source")
    if not value and isinstance(source, dict) and source.get("data"):
        value = f"data:{source.get('media_type') or 'image/png'};base64,{source['data']}"
    return str(value) if value else None


def _tool_choice(value: Any) -> tuple[bool, str | None]:
    if value is True or value in {"any", "required"}:
        return True, None
    if isinstance(value, str) and value not in {"auto", "none"}:
        return True, value
    if isinstance(value, dict):
        function = value.get("function") or {}
        name = function.get("name") or value.get("name")
        return bool(name), name
    return False, None
