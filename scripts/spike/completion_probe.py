#!/usr/bin/env python
"""Phase 0 spike -- can raw /completion take our prompt, for text AND images?

Renders a small synthetic resume (known values) as text and as a page image, builds the
prompt with cvx.prompt + the HF processor, swaps the processor's image block for
llama.cpp's media marker, and sends both through /completion. Prints what came back,
whether it parsed, and whether the server tokenised the prompt the way HF did.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import httpx  # noqa: E402

from cvx.prompt import messages, parse_generation, render_prompt  # noqa: E402

CV_TEXT = """Joana Ribeiro Lopes
joana.lopes@example.com · (48) 99876-5432 · Florianópolis, SC

EXPERIÊNCIA PROFISSIONAL
Técnica de enfermagem — Hospital Vale Verde          03/2021 – atual
Atendimento em UTI adulto.
Auxiliar de enfermagem — Clínica Santa Luzia          02/2018 – 01/2021

FORMAÇÃO
Técnico em Enfermagem — Escola Técnica Horizonte      2017

HABILIDADES
Administração de medicamentos, Curativos, Pacote Office

IDIOMAS
Espanhol (intermediário)"""



def page_png(text: str) -> bytes:
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("RGB", (1000, 1400), "white")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 22)
    except OSError:
        font = ImageFont.load_default(size=22)
    draw.multiline_text((60, 60), text, fill="black", font=font, spacing=10)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def vram() -> int:
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, check=True)
    return int(out.stdout.strip())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processor", required=True)
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--image-block", default="<|vision_start|><|image_pad|><|vision_end|>")
    args = ap.parse_args()

    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(args.processor)
    client = httpx.Client(base_url=args.url, timeout=600)
    decode = {"temperature": 0, "n_predict": 1024, "cache_prompt": False}

    # text
    prompt = render_prompt(messages(cv_text=CV_TEXT), processor)
    r = client.post("/completion", json={"prompt": prompt, **decode}).json()
    parsed, err = parse_generation(r["content"])
    hf_ids = processor.tokenizer(prompt, add_special_tokens=False).input_ids
    srv_ids = client.post("/tokenize", json={"content": prompt, "parse_special": True}).json()["tokens"]
    print("TEXT tokens_hf=%d tokens_srv=%d same=%s prompt_n=%s vram=%d err=%s" % (
        len(hf_ids), len(srv_ids), hf_ids == srv_ids, r["timings"]["prompt_n"], vram(), err))
    print("TEXT content:", r["content"][:600].replace("\n", " "))

    # vision
    prompt = render_prompt(messages(n_pages=1), processor)
    assert prompt.count(args.image_block) == 1, prompt
    # The server randomises its media marker per run (anti prompt-injection); ask for it.
    marker = client.get("/props").json()["media_marker"]
    raw = prompt.replace(args.image_block, marker)
    b64 = base64.b64encode(page_png(CV_TEXT)).decode()
    r = client.post("/completion", json={"prompt": {"prompt_string": raw, "multimodal_data": [b64]}, **decode}).json()
    if "content" not in r:
        print("VISION error:", json.dumps(r)[:600])
        return
    parsed, err = parse_generation(r["content"])
    print("VISION prompt_n=%s vram=%d err=%s" % (r["timings"]["prompt_n"], vram(), err))
    print("VISION content:", r["content"][:600].replace("\n", " "))
    print("VISION prompt head:", repr(prompt[:400]))


if __name__ == "__main__":
    main()
