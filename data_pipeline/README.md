# EVR Data Pipeline

Construction code for the **EVR** multi-reference image-editing dataset. Product reference
images and matching scene images are generated with diffusion models, the product is removed
from each scene with an image-editing model, and the images are paired and annotated with edit
instructions by a VLM.

Tasks produced:

- **replace** — ref ↔ a *different* scene of the same category: replace the object in the
  scene with the reference product.
- **add** — ref ↔ *its own* object-free scene: add the reference product into the scene.

## Pipeline

```
category list (in scripts/01)
        │
        ▼
[01] Qwen3-VL: product-image prompts per category
     ZImage: generate product image ──────────────► ref/
     Qwen3-VL: quality filter + recognize actual category
               + write scene prompt                (bad images deleted)
     ZImage: generate scene image ────────────────► fuse/   (same numeric ID as its ref)
     Qwen3-VL: write a removal instruction
     everything recorded per item ────────────────► manifest.jsonl
        │
        ▼
[02] LongCat-Image-Edit: remove the object from its scene ──► remove/   (same filename)
        │
        ├──────────────────────────────────────────────┐
        ▼                                              ▼
[03] same-category pairing                          [05] same-ID pairing
     (ref ↔ different-ID scene,                      (ref ↔ its own remove image)
      category from manifest.jsonl) ──► replace.txt        ──► add.txt
        │                                                  │
        ▼                                                  ▼
[04] Qwen3-VL: "Replace ..." ──► rep.txt          [06] Qwen3-VL: "Add ..." ──► addp.txt
        │                                                  │
        └──────────────────────┬───────────────────────────┘
                               ▼
                 [07] assemble ──► metadata.jsonl
```

## Quick Start

Checkpoints:

```
ckpt/
├── Zimage/
├── LongCat-Image-Edit/
└── Qwen3-VL-8B-Thinking/     # thinking version required
```

Serve the VLM (using reward server env):

```bash
vllm serve ckpt/Qwen3-VL-8B-Thinking --port 8000 --served-model-name Qwen3-VL-8B-Thinking
```

Run the pipeline from the project root (using training env):

```bash
python data_pipeline/scripts/01_generate_ref_and_scene.py
python data_pipeline/scripts/02_remove_objects.py
python data_pipeline/scripts/03_make_replace_pairs.py
python data_pipeline/scripts/04_generate_replace_instructions.py
python data_pipeline/scripts/05_make_add_pairs.py
python data_pipeline/scripts/06_generate_add_instructions.py
python data_pipeline/scripts/07_assemble_dataset.py
```

All outputs are written to `custom_dataset/` under the project root (directories are created
automatically): `ref/`, `fuse/`, `remove/`, `manifest.jsonl`, the intermediate `*.txt` files,
and `metadata.jsonl`.

Output: `custom_dataset/metadata.jsonl` with one entry per pair —
`{"task", "prompt", "ref_image", "target_image", "requirement"}` (image paths relative to the
project root).
