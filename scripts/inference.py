import torch
from PIL import Image
from diffusers import QwenImageEditPlusPipeline
from peft import PeftModel

pipeline = QwenImageEditPlusPipeline.from_pretrained("./ckpt/Qwen-Image-Edit", torch_dtype=torch.bfloat16)
print("pipeline loaded")
# pipeline.transformer = PeftModel.from_pretrained( pipeline.transformer, './ckpt/checkpoint-40/lora')
pipeline.to('cuda')
pipeline.set_progress_bar_config(disable=None)

imagepath1 = './demo/686_Acer_Nitro_5_gaming_laptop_closed_fron.png'
imagepath2 = './demo/3652_A_young_professional_working_on_a_slee.png'
prompt = 'Replace the laptop in Picture 2 with the laptop in Picture 1, ensuring that all other elements in Picture 2 remain unchanged.'
image1 = Image.open(imagepath1)
image2 = Image.open(imagepath2)
inputs = {
    "image": [image1, image2],
    "prompt": prompt,
    "generator": torch.manual_seed(18),
    "true_cfg_scale": 4.0,
    "negative_prompt": '',
    "num_inference_steps": 20,
    "guidance_scale": 1.0,
    "num_images_per_prompt": 1,
}
with torch.inference_mode():
    output = pipeline(**inputs)
    output_image = output.images[0]
    # output_image.save(f"./output/demo_finetuned.png")
    output_image.save(f"./output/demo_basemodel.png")

