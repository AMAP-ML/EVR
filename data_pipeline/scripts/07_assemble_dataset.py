import json
import os

out_dir = "custom_dataset"

pairs = []

with open(os.path.join(out_dir, "rep.txt"), "r", encoding="utf-8") as f:
    for line in f:
        parts = line.rstrip("\n").split(";", 2)
        if len(parts) != 3:
            continue
        ref, scene, instruction = parts
        pairs.append({
            "task": "replace",
            "prompt": instruction,
            "ref_image": f"{out_dir}/ref/" + ref,
            "target_image": f"{out_dir}/fuse/" + scene,
            "requirement": instruction,
        })

with open(os.path.join(out_dir, "addp.txt"), "r", encoding="utf-8") as f:
    for line in f:
        parts = line.rstrip("\n").split(";", 2)
        if len(parts) != 3:
            continue
        ref, remove, instruction = parts
        pairs.append({
            "task": "add",
            "prompt": instruction,
            "ref_image": f"{out_dir}/ref/" + ref,
            "target_image": f"{out_dir}/remove/" + remove,
            "requirement": instruction,
        })

with open(os.path.join(out_dir, "metadata.jsonl"), "w", encoding="utf-8") as f:
    for p in pairs:
        f.write(json.dumps(p, ensure_ascii=False) + "\n")

n_replace = sum(1 for p in pairs if p["task"] == "replace")
print(f"replace pairs: {n_replace}")
print(f"add pairs: {len(pairs) - n_replace}")
print(f"total: {len(pairs)} -> {out_dir}/metadata.jsonl")
