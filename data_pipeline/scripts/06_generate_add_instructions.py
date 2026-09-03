from openai import OpenAI
import os
import base64
import mimetypes

out_dir = "custom_dataset"

# Set OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"
client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

def image_to_data_url(image_path):
    mime_type, _ = mimetypes.guess_type(image_path)
    if mime_type is None:
        mime_type = 'image/png'  # default fallback

    with open(image_path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode('utf-8')
    return f"data:{mime_type};base64,{encoded}"

with open(os.path.join(out_dir, 'add.txt'), 'r') as f:
    img = f.readlines()

for p in img:
    p1, p2 = p[:-1].split(';')[0], p[:-1].split(';')[1]
    try:
        messages = [{
            "role": "user", "content": [
                {"type": "image_url", "image_url": {"url": image_to_data_url(os.path.join(out_dir, "ref", p1))}},
                {"type": "image_url", "image_url": {"url": image_to_data_url(os.path.join(out_dir, "remove", p2))}},
                {"type": "text", "text": '''
                    I want to perform the task of object insertion. You need to analyze the two input images and add the object in image 1 to a proper location in image 2. Please help me output an instruction prompt for item insertion that can be recognized by the image editing model
                    example:
                    Add/Place xx in Picture 1 to <location> in Picture 2, ensuring that all other elements in Picture 2 remain unchanged.
                    Make the man/woman in Picture 2 wear/hold..... xxx from Picture 1, ensuring that all other elements in Picture 2 remain unchanged.
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
        model_output = chat_response.choices[0].message.content.split('</think>\n\n')[1]

        with open(os.path.join(out_dir, "addp.txt"), "a+", encoding="utf-8") as f:
            f.write(p1 + ';' + p2 + ';' + model_output + '\n')
    except:
        with open(os.path.join(out_dir, 'erroradd.txt'), 'a+') as f:
            f.write(p1 + ';' + p2 + '\n')
        continue
