# ADR 0011 — One prompt renderer; raw /completion (vision pending)

- **Status:** Proposed — vision endpoint pending phase 0
- **Date:** 2026-09-30

## Context

In the persona repo, evaluating through `/v1/chat/completions` let llama-server re-apply
its own Jinja template, breaking train/inference parity. Evaluation there goes through
raw `/completion` with a prompt rendered by our own code.

## Decision

- `src/cvx/prompt.py` is the only place a prompt or message list is built.
  `tests/test_prompt_parity.py` asserts the training text starts with the inference
  prompt, for both the tokenizer (text) and the processor (vision).
- `enable_thinking=False` always; any `<think>` in a generation is a format failure.
- Text: raw `/completion` only.
- **Vision: open.** The spike must confirm the pinned llama-server accepts images on
  `/completion` (multimodal data plus media markers in the raw prompt). If it does not,
  a follow-up ADR accepts the chat endpoint for vision only, with parity checked by
  comparing the server's rendered prompt against `render_prompt`.
