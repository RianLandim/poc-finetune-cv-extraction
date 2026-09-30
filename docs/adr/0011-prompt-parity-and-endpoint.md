# ADR 0011 — One prompt renderer; raw /completion for both modalities

- **Status:** Accepted — vision path verified in the phase 0 spike
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
- **Vision: raw `/completion` too.** The body is
  `{"prompt": {"prompt_string": …, "multimodal_data": [base64, …]}}`. The prompt string is
  `render_prompt(...)` passed through `to_server_prompt(prompt, marker)`, which replaces
  each HF image block with the server's media marker; mtmd re-inserts the vision
  start/end tokens.
- The marker is **randomised per server run**; read it from `GET /props`
  (`media_marker`) on every eval run. Never hardcode `<__media__>`.
- Phase 0 measured parity: 1,465 input tokens on both sides for a 1000×1400 page.
  Re-check this count on any llama.cpp or processor upgrade.
