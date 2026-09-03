import torch
import os
import json
from PIL import Image
from diffusers import LongCatImageEditPipeline

out_dir = "custom_dataset"
os.makedirs(os.path.join(out_dir, "remove"), exist_ok=True)

# 1. Load the pipeline
pipe = LongCatImageEditPipeline.from_pretrained(
    "./ckpt/LongCat-Image-Edit",
    torch_dtype=torch.bfloat16,
)
pipe.to("cuda")

with open(os.path.join(out_dir, "manifest.jsonl"), "r", encoding="utf-8") as f:
    items = [json.loads(line) for line in f]

for item in items:
    instruction = item["remove_instruction"]
    if not instruction or instruction.strip() == "None":
        continue
    scene_name = item["scene"]
    out_path = os.path.join(out_dir, "remove", scene_name)
    if os.path.exists(out_path):
        continue
    try:
        scene_image = Image.open(os.path.join(out_dir, "fuse", scene_name)).convert("RGB")
        image = pipe(
            scene_image,
            instruction,
            num_inference_steps=20,
            guidance_scale=4.5,
            generator=torch.Generator("cpu").manual_seed(43),
        ).images[0]
        image.save(out_path)
    except:
        with open(os.path.join(out_dir, "errorlongcat.txt"), "a+") as f:
            f.write(scene_name + "\n")
