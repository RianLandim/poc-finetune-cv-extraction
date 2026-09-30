import pytest

from cvx.prompt import (
    IMAGE_BLOCK,
    PROMPT_VERSION,
    SYSTEM_PROMPT,
    manifest,
    messages,
    parse_generation,
    render_prompt,
    render_training_text,
    to_server_prompt,
)
from cvx.schema import SCHEMA_VERSION, to_target

CV_TEXT = "Joana Ribeiro Lopes\njoana.lopes@example.com\nExperiência\n..."


def test_versions_are_in_the_manifest():
    assert manifest() == {"prompt_version": PROMPT_VERSION, "schema_version": SCHEMA_VERSION}


def test_exactly_one_modality():
    with pytest.raises(ValueError):
        messages()
    with pytest.raises(ValueError):
        messages(cv_text="x", n_pages=1)


def test_text_and_vision_share_the_system_prompt():
    text = messages(cv_text=CV_TEXT)
    vision = messages(n_pages=2)
    assert text[0]["content"] == SYSTEM_PROMPT
    assert vision[0]["content"][0]["text"] == SYSTEM_PROMPT
    assert [c["type"] for c in vision[1]["content"]] == ["image", "image", "text"]


def test_parse_accepts_the_target(cv):
    parsed, error = parse_generation(to_target(cv))
    assert error is None and parsed == cv


@pytest.mark.parametrize("text, error", [
    ("<think>\n\n</think>{}", "think_block"),
    ('```json\n{"nome": "x"}\n```', "code_fence"),
    ('{"nome": ', "invalid_json"),
    ('{"nome": "x", "cpf": "1"}', "schema_violation"),
])
def test_parse_failures_are_classified(text, error):
    assert parse_generation(text) == (None, error)


def test_server_prompt_swaps_each_image_block_for_the_marker():
    prompt = f"<|im_start|>user\n{IMAGE_BLOCK}{IMAGE_BLOCK}Extraia<|im_end|>"
    out = to_server_prompt(prompt, "<__media_X__>")
    assert out == "<|im_start|>user\n<__media_X__><__media_X__>Extraia<|im_end|>"


# Qwen3.5-4B is natively multimodal: its processor serves both modalities.
@pytest.fixture(scope="module", params=["unsloth/Qwen3.5-4B", "unsloth/Qwen3-VL-4B-Instruct"])
def processor(request):
    transformers = pytest.importorskip("transformers")
    return transformers.AutoProcessor.from_pretrained(request.param)


@pytest.fixture(scope="module")
def tokenizer(processor):
    return processor.tokenizer


@pytest.mark.slow
def test_text_training_text_starts_with_inference_prompt(tokenizer, cv):
    msgs = messages(cv_text=CV_TEXT)
    assert render_training_text(msgs, cv, tokenizer).startswith(render_prompt(msgs, tokenizer))


@pytest.mark.slow
def test_vision_training_text_starts_with_inference_prompt(processor, cv):
    msgs = messages(n_pages=2)
    assert render_training_text(msgs, cv, processor).startswith(render_prompt(msgs, processor))


@pytest.mark.slow
def test_processor_renders_the_expected_image_block(processor):
    assert render_prompt(messages(n_pages=2), processor).count(IMAGE_BLOCK) == 2


@pytest.mark.slow
def test_no_open_think_block(tokenizer):
    prompt = render_prompt(messages(cv_text=CV_TEXT), tokenizer)
    assert "<think>" not in prompt or "<think>\n\n</think>" in prompt
