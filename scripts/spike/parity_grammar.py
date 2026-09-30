import base64, io, sys, json
sys.path.insert(0, "src"); sys.path.insert(0, "scripts/spike")
import httpx
from PIL import Image
from transformers import AutoProcessor
from cvx.prompt import messages, render_prompt, parse_generation
from cvx.schema import json_schema
from completion_probe import CV_TEXT, page_png

proc = AutoProcessor.from_pretrained("unsloth/Qwen3.5-4B")
c = httpx.Client(base_url="http://127.0.0.1:8080", timeout=600)
png = page_png(CV_TEXT)
prompt = render_prompt(messages(n_pages=1), proc)
hf = proc(text=[prompt], images=[[Image.open(io.BytesIO(png))]], return_tensors="pt")
print("HF image grid_thw:", hf["image_grid_thw"].tolist(), "total ids:", hf["input_ids"].shape[1])
marker = c.get("/props").json()["media_marker"]
raw = prompt.replace("<|vision_start|><|image_pad|><|vision_end|>", marker)
body = {"prompt": {"prompt_string": raw, "multimodal_data": [base64.b64encode(png).decode()]},
        "temperature": 0, "n_predict": 1024, "cache_prompt": False, "json_schema": json_schema()}
r = c.post("/completion", json=body).json()
print("server prompt_n:", r["timings"]["prompt_n"])
cv, err = parse_generation(r["content"])
print("GRAMMAR err:", err)
print(json.dumps(cv.model_dump() if cv else r["content"][:500], ensure_ascii=False)[:900])
