import torch
from diffusers import ZImagePipeline
from openai import OpenAI
import os
import json
import base64
import mimetypes

out_dir = "custom_dataset"
os.makedirs(os.path.join(out_dir, "ref"), exist_ok=True)
os.makedirs(os.path.join(out_dir, "fuse"), exist_ok=True)

# Size limits for quick tests. Set both to None to run the full pipeline.
# Example: a small smoke test = 2 categories x 5 product prompts each.
max_categories = 2
max_prompts_per_category = 2

world_index = 0
if os.path.exists(os.path.join(out_dir, "manifest.jsonl")):
    with open(os.path.join(out_dir, "manifest.jsonl"), "r", encoding="utf-8") as f:
        for line in f:
            world_index = max(world_index, json.loads(line)["id"] + 1)

def image_to_data_url(image_path):
    mime_type, _ = mimetypes.guess_type(image_path)
    if mime_type is None:
        mime_type = 'image/png'  # default fallback

    with open(image_path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode('utf-8')
    return f"data:{mime_type};base64,{encoded}"

# Set OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"
client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)
# 1. Load the pipeline
# Use bfloat16 for optimal performance on supported GPUs
pipe = ZImagePipeline.from_pretrained(
    "./ckpt/Zimage",
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=False,
)
pipe.to("cuda")

# Categories to generate. Each entry drives one VLM prompt-list request followed by
# image generation. Extend this list to add new categories.
list_class = [
    # Apparel
    "Car",
    "Smart watch",
    "Camera", "Game console","TV", "Keyboard","Mouse","Laptop",

    "Accessory_Scarf", "Accessory_Gloves", "Accessory_Belt", "Accessory_Necktie",

    # Shoes & Bags
    "Sneakers", "Leather shoes", "Sandals", "Slippers", "Boots",
    "Backpack", "Handbag", "Wallet", "Luggage", "Waist bag",

    # Beauty & Personal Care
    "Skincare_Face cream", "Skincare_Lotion", "Skincare_Essence", "Skincare_Facial mask",
    "Makeup_Lipstick", "Makeup_Eyeshadow", "Makeup_Foundation", "Makeup_Blush",
    "Perfume", "Shampoo and hair care products", "Oral care products", "Beauty tools",

    # Consumer Electronics


    # Home & Daily Use
    "Bedding_Quilt", "Bedding_Pillow", "Bedding_Bed sheet",
    "Storage products", "Ornaments", "Lamps", "Cleaning tools",

    # Food & Beverages
    "Snacks", "Beverages", "Condiments", "Instant food", "Fresh food", "Alcohol",

    # Maternal & Baby Products
    "Milk powder", "Diapers", "Toys", "Baby clothing", "Baby bottles", "Safety seats",

    # Sports & Outdoors
    "Fitness equipment", "Sportswear", "Balls", "Camping gear", "Cycling gear",

    # Books & Media
    "Paper books",

    # Car Accessories
    "Car accessories",

    # Pet Supplies
    "Pet food", "Pet toys", "Pet beds", "Leashes", "Pet grooming products", "Pet accessories",

    # Office & Stationery
    "Pens", "Notebooks", "Folders", "Printing consumables", "Desk organizers",

    # Jewelry
    "Necklaces", "Rings", "Earrings", "Watches",

    # Health & Wellness
    "Vitamins", "Health supplements", "Massagers", "Thermometers", "Home medical devices"
]

# Candidate categories the VLM may pick from when classifying the *actual* main object
# of each generated image (drift correction). Same list as list_class: a drifted
# generation is simply filed under another category within the list.
product_subcategories = list_class

for class_name in list_class[:max_categories]:
    try:
        chat_response = client.chat.completions.create(
            model="Qwen3-VL-8B-Thinking",
            messages=[
                {"role": "user", "content": f"I am building a multi-image fusion editing dataset, and I need you to help me generate a list of prompts for realistic item images with clean backgrounds based on the item category. These items are generally product/item display images. Please include as many types, brands, forms, angles, and styles as possible, but try to keep them real. Please output English prompts directly as a list, avoid using quotation marks and special characters such as / in the prompts, at most {max_prompts_per_category or 50} entries, in the format ['xxx','xxx',...]. Item: {class_name}"},
            ],
            max_tokens=32768,
            temperature=0.6,
            top_p=0.95,
            extra_body={
                "top_k": 20,
            },
        )
        refprompts = eval(chat_response.choices[0].message.content.split('</think>\n\n')[1])
        refprompts = refprompts[:max_prompts_per_category]

        # [Optional] Attention Backend
        # Diffusers uses SDPA by default. Switch to Flash Attention for better efficiency if supported:
        # pipe.transformer.set_attention_backend("flash")    # Enable Flash-Attention-2
        # pipe.transformer.set_attention_backend("_flash_3") # Enable Flash-Attention-3

        # [Optional] Model Compilation
        # Compiling the DiT model accelerates inference, but the first run will take longer to compile.
        # pipe.transformer.compile()

        scenep= ""
        for prompt in refprompts:
            try:
                image = pipe(
                    prompt=prompt,
                    height=1024,
                    width=1024,
                    num_inference_steps=9,  # This actually results in 8 DiT forwards
                    guidance_scale=0.0,     # Guidance should be 0 for the Turbo models
                    generator=torch.Generator("cuda").manual_seed(42),
                ).images[0]
                if len(prompt) > 38:
                    prompt = prompt[0:38]
                ref_name = f'{world_index}_{prompt.replace(" ", "_")}.png'
                ref_image_url = f'{out_dir}/ref/{ref_name}'
                image.save(ref_image_url)
                data_url = image_to_data_url(ref_image_url)
                system_message={
                    "role": "system",
                    "content": '''You are a professional visual quality assessor and prompt engineer, focused on creating high-quality display images for product showcasing. Your task is:
                    Accept an item image, along with the scene image prompts previously generated for this category of products, and try to ensure diverse scene image prompts.
                    1. **Evaluate the quality of the given item image**:
                    - Check whether any of the following problems exist: mixed semantics, chaotic presentation, unclear structure, violation of real-world physics or logic, etc.
                    - If any of the above problems exist, output `None` directly.

                    2. **If the item image has no quality problems**, you need to complete the following two tasks:
                    a. **Recognize the actual category of the item image**: select the category that the main object in the image actually belongs to from the candidate category list, and output the category name directly.
                    b. **Generate a suitable usage scene image description (prompt) in English for the item**, so that the item image can later be fused into it to form the final product display image. This description should:
                    - Accurately reflect the typical scene where the item might be used (e.g.: sports watch -> photo of a mountain climber).
                    - Scene images should be as human-centered as possible, such as holding or wearing; the scene image should ensure holding or wearing actions as much as possible so that the item can be reasonably fused; for household items, home appliances, animals, vegetables and other things without an obvious relation to people, use natural scenes.
                    - The description should be detailed and specific, including details of environment, lighting, viewing angle, etc., to facilitate generating realistic background images.
                    - Use natural and fluent language, suitable for text-to-image models (such as Stable Diffusion); avoid using quotation marks in the prompt.
                    - The scene should be as complete as possible; for example, the person should be shown relatively completely, and should not be a close-up of the product just for product display; the product can occupy a frame proportion that keeps it recognizable.

                    Based on the above guidelines, evaluate the given item image, and output only `None` or a Python list ["category", "English scene description prompt"]. Do not provide extra explanation or text.
                    '''
                }
                message = [
                    system_message,
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": data_url}
                            },
                            {
                                "type": "text",
                                "text": "Candidate category list: " + str(product_subcategories) + "\nPreviously generated scene image prompts: " + scenep + "\nPlease determine whether the image has quality problems; if not, output in the format [\"category\", \"English prompt for the scene image\"]"
                            }

                        ]
                    }
                ]
                chat_response = client.chat.completions.create(
                    model="Qwen3-VL-8B-Thinking",
                    messages=message,
                    max_tokens=8192,
                    temperature=0.6,
                    top_p=0.95,
                    extra_body={
                        "top_k": 20,
                    },
                )
                try:
                    output = chat_response.choices[0].message.content.split('</think>\n\n')[1].strip()
                    category, scene_prompt = eval(output)
                except:
                    scene_prompt = "None"
                if scene_prompt == "None":
                    os.remove(ref_image_url)
                    continue
                scenep = scenep + ",\'" + scene_prompt+ "\'"
                image = pipe(
                    prompt=scene_prompt,
                    height=1024,
                    width=1024,
                    num_inference_steps=9,  # This actually results in 8 DiT forwards
                    guidance_scale=0.0,     # Guidance should be 0 for the Turbo models
                    generator=torch.Generator("cuda").manual_seed(42),
                ).images[0]
                if len(scene_prompt) > 38:
                    scene_prompt = scene_prompt[0:38]
                scene_name = f'{world_index}_{scene_prompt.replace(" ", "_")}.png'
                scene_image_url = f'{out_dir}/fuse/{scene_name}'
                image.save(scene_image_url)

                try:
                    messages = [{
                        "role": "user", "content": [
                            {"type": "image_url", "image_url": {"url": image_to_data_url(ref_image_url)}},
                            {"type": "image_url", "image_url": {"url": image_to_data_url(scene_image_url)}},
                            {"type": "text", "text": '''
                            You have the following task:
                            Determine whether the object of image 2 can be removed.
                                - Firstly, determine the object of image 1
                                - Scecondly,determine whether objects in image 2 that are similar or identical to the object of image 1 can be removed. (Situations that cannot be removed: Removing them will result in inappropriate and legal content such as pornography and NSFW images.)
                                If it can be removed, please output an remove instruction that can be recognized by the image editing model, output the instruction directly without any other characters. If it cannot be removed, please output "None" without any other characters.
                                example 1:
                                Image 1: Necklace
                                Image 2: A woman wearing a necklace
                                Output: Remove the necklace worn by the woman's neck.
                                example 2:
                                Image 1: Mobile phone
                                Image 2: A person holding a mobile phone
                                Output: Remove the mobile phone held by the person, and make the person's hand naturally stretched.
                                example 3:
                                Image 1: A coat
                                Image 2: A person wearing a coat
                                Output: None
                            '''}
                        ]
                    }]
                    chat_response = client.chat.completions.create(
                        model="Qwen3-VL-8B-Thinking",
                        messages=messages,
                        max_tokens=8192,
                        temperature=0.6,
                        top_p=0.95,
                        extra_body={
                            "top_k": 20,
                        },
                    )
                    remove_instruction = chat_response.choices[0].message.content.split('</think>\n\n')[1].strip()
                except:
                    remove_instruction = None

                with open(os.path.join(out_dir, "manifest.jsonl"), "a+", encoding="utf-8") as f:
                    f.write(json.dumps({
                        "id": world_index,
                        "category": category,
                        "ref": ref_name,
                        "product_prompt": prompt,
                        "scene": scene_name,
                        "scene_prompt": scene_prompt,
                        "remove_instruction": remove_instruction,
                    }, ensure_ascii=False) + '\n')
                world_index += 1
            except:
                with open(os.path.join(out_dir, 'error.txt'), 'a') as f:
                    f.write(class_name + ':' + prompt + '\n')
                continue
    except:
        with open(os.path.join(out_dir, 'error.txt'), 'a') as f:
            f.write(class_name + '\n')
        continue
