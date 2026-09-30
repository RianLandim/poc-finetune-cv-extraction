"""The single source of every prompt in this project (ADR 0011).

Dataset building, training, evaluation and serving all import from here. If you find
yourself building a prompt string or a chat message list anywhere else, that is the bug.

Two modalities share the system prompt and the target; only the user turn differs:
- text:   the PDF's extracted text, inline.
- vision: one image placeholder per page, filled in by the processor.
"""

from __future__ import annotations

import json

from cvx.schema import SCHEMA_VERSION, Curriculo, to_target

PROMPT_VERSION = "1.0.0"

SYSTEM_PROMPT = (
    "Você extrai informações de currículos brasileiros. Responda somente com um objeto "
    "JSON no formato indicado, sem texto antes ou depois. Use apenas informações "
    "presentes no currículo; quando um campo não aparecer, use null ou lista vazia. "
    "Datas no formato AAAA-MM (ou AAAA quando só houver o ano). Não extraia CPF, RG "
    "nem data de nascimento."
)

INSTRUCTION = "Extraia o currículo para JSON."

# What the Qwen3.5 / Qwen3-VL processors render for one {"type": "image"} item. The HF
# processor later expands <|image_pad|> into one token per vision patch group.
IMAGE_BLOCK = "<|vision_start|><|image_pad|><|vision_end|>"


def _messages_text(cv_text: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"{INSTRUCTION}\n\n<curriculo>\n{cv_text.strip()}\n</curriculo>"},
    ]


def _messages_vision(n_pages: int) -> list[dict]:
    if n_pages < 1:
        raise ValueError("a resume has at least one page")
    content = [{"type": "image"} for _ in range(n_pages)]
    content.append({"type": "text", "text": INSTRUCTION})
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": content},
    ]


def messages(*, cv_text: str | None = None, n_pages: int | None = None) -> list[dict]:
    """Chat messages for exactly one modality: pass cv_text OR n_pages."""
    if (cv_text is None) == (n_pages is None):
        raise ValueError("pass exactly one of cv_text (text) or n_pages (vision)")
    return _messages_text(cv_text) if cv_text is not None else _messages_vision(n_pages)


def render_prompt(msgs: list[dict], template_owner) -> str:
    """Inference-time string. Thinking mode is disabled unconditionally.

    ``template_owner`` is a tokenizer (text) or a processor (vision); both expose
    ``apply_chat_template``.
    """
    return template_owner.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )


def to_server_prompt(prompt: str, media_marker: str) -> str:
    """The same prompt, in the form llama-server's raw /completion expects (ADR 0011).

    llama.cpp's mtmd inserts <|vision_start|>/<|vision_end|> around each image itself, so
    the whole HF image block becomes one media marker. The marker is randomised per
    server run; read it from GET /props ("media_marker"). Phase 0 measured the result
    token-identical to the HF processor's input (docs/spike/2026-09-30-phase0.md).
    """
    n_images = prompt.count(IMAGE_BLOCK)
    if n_images == 0 and "<|image_pad|>" in prompt:
        raise ValueError("prompt has an image pad outside the expected image block")
    return prompt.replace(IMAGE_BLOCK, media_marker)


def render_training_text(msgs: list[dict], cv: Curriculo, template_owner) -> str:
    """Full train-time string: prompt plus the target assistant turn.

    Invariant enforced by tests/test_prompt_parity.py: this string always starts with
    exactly what render_prompt() produces for the same messages.
    """
    vision = isinstance(msgs[0]["content"], list)
    target = to_target(cv)
    assistant = [{"type": "text", "text": target}] if vision else target
    full = msgs + [{"role": "assistant", "content": assistant}]
    return template_owner.apply_chat_template(
        full, tokenize=False, add_generation_prompt=False, enable_thinking=False
    )


def parse_generation(text: str) -> tuple[Curriculo | None, str | None]:
    """(parsed, error). Any ``<think>`` is a format failure, never stripped silently."""
    if "<think>" in text:
        return None, "think_block"
    stripped = text.strip()
    if stripped.startswith("```"):
        return None, "code_fence"
    try:
        raw = json.loads(stripped)
    except json.JSONDecodeError:
        return None, "invalid_json"
    try:
        return Curriculo.model_validate(raw), None
    except ValueError:
        return None, "schema_violation"


def manifest() -> dict:
    """Written next to every dataset; eval refuses a model trained on another version."""
    return {"prompt_version": PROMPT_VERSION, "schema_version": SCHEMA_VERSION}
