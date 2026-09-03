import os
import asyncio
from typing import List
from PIL import Image
from io import BytesIO
import base64
import pickle
import traceback
from flask import Flask, request
from openai import AsyncOpenAI, APIConnectionError
import prompt_template as prompt_template
import re
import ast
import random
import string

def generate_random_string(length):
    characters = string.ascii_letters + string.digits
    return ''.join(random.choices(characters, k=length))

# 控制最大并发API请求数，避免连接被拒绝
MAX_CONCURRENT_API_CALLS = 15

app = Flask(__name__)

# API配置从环境变量读取
VLM_URL = os.getenv("VLM_URL")
VLM_MODEL = os.getenv("VLM_MODEL")
VLM_API_KEY = os.getenv("VLM_API_KEY")


def get_base64(image):
    image_data = BytesIO()
    image.save(image_data, format="JPEG")
    image_data_bytes = image_data.getvalue()
    encoded_image = base64.b64encode(image_data_bytes).decode("utf-8")
    return encoded_image


def image_to_api_content(image):
    """将PIL Image转为API所需的image_url格式"""
    b64 = get_base64(image)
    return {
        "type": "image_url",
        "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
    }


def build_api_messages(conversation_item):
    """将内部消息格式转为OpenAI兼容的messages格式"""
    messages = []
    content_list = []
    for part in conversation_item["content"]:
        if part["type"] == "image_pil":
            content_list.append(image_to_api_content(part["image_pil"]))
        elif part["type"] == "text":
            content_list.append({"type": "text", "text": part["text"]})
    messages.append({"role": "user", "content": content_list})
    return messages


async def api_call(client, messages, semaphore, max_tokens=1024, temperature=0.6, top_p=0.95, max_retries=5):
    """单次异步API调用，带并发限制和重试机制"""
    for attempt in range(max_retries):
        try:
            async with semaphore:
                response = await client.chat.completions.create(
                    model=VLM_MODEL,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    extra_body={"enable_thinking": False},
                )
                return response.choices[0].message.content
        except APIConnectionError as e:
            if attempt < max_retries - 1:
                wait_time = 3 ** (attempt + 1)
                print(f"API connection error (attempt {attempt+1}/{max_retries}), retrying in {wait_time}s: {e}")
                await asyncio.sleep(wait_time)
            else:
                print(f"API connection error after {max_retries} attempts, raising")
                raise


async def evaluate_image_async(client, image_bytes, prompt, ref_image_bytes=None, scene_image_bytes=None, requirement="", semaphore=None):
    """异步评估单张图片，保持原有25+5两轮流程"""
    # 解码图片
    image = Image.open(BytesIO(image_bytes), formats=["jpeg"])
    ref_image = Image.open(BytesIO(ref_image_bytes), formats=["jpeg"])
    scene_image = Image.open(BytesIO(scene_image_bytes), formats=["jpeg"])
    # name = generate_random_string(10)
    # 构建5个维度的评估消息
    Quality = {
        "role": "user",
        "content": [
            {"type": "image_pil", "image_pil": image},
            {"type": "text", "text": prompt_template.QUA_SCORE},
        ],
    }
    Consistency = {
        "role": "user",
        "content": [
            {"type": "image_pil", "image_pil": ref_image},
            {"type": "image_pil", "image_pil": image},
            {"type": "text", "text": prompt_template.CON_SCORE.format(
                prompt=prompt.replace('Picture 1', 'Reference Image').replace('Picture 2', "Scene Image"))},
        ],
    }
    Scene = {
        "role": "user",
        "content": [
            {"type": "image_pil", "image_pil": scene_image},
            {"type": "image_pil", "image_pil": image},
            {"type": "text", "text": prompt_template.SCE_SCORE.format(
                prompt=prompt.replace('Picture 1', 'Reference Image').replace('Picture 2', "Scene Image"))},
        ],
    }
    Harmony = {
        "role": "user",
        "content": [
            {"type": "image_pil", "image_pil": ref_image},
            {"type": "image_pil", "image_pil": image},
            {"type": "text", "text": prompt_template.HAR_SCORE},
        ],
    }
    Instruction = {
        "role": "user",
        "content": [
            {"type": "image_pil", "image_pil": ref_image},
            {"type": "image_pil", "image_pil": scene_image},
            {"type": "image_pil", "image_pil": image},
            {"type": "text", "text": prompt_template.PRO_SCORE.format(prompt=prompt)},
        ],
    }

    conversation = [Quality, Consistency, Scene, Harmony, Instruction]

    # ========== 第一轮：25次评估（5维度 x 5次）异步并发 ==========
    eval_tasks = []
    for item in conversation:
        messages = build_api_messages(item)
        for _ in range(5):
            eval_tasks.append(api_call(client, messages, semaphore, max_tokens=1024, temperature=0.6, top_p=0.95))

    eval_results = await asyncio.gather(*eval_tasks)

    # 解析25次评估结果 -> score_total[5][5], output_total[5][5]
    score_total, output_total = [], []
    for i in range(5):  # 5个维度
        score_tmp, output_tmp = [], []
        for j in range(5):  # 每维度5次
            output_text = eval_results[i * 5 + j]
            print(output_text)
            try:
                Score = re.search(r'<Score>(\d+)</Score>', output_text)
                score_val = int(Score.group(1))
                score_tmp.append(score_val)
                output_tmp.append(output_text)
            except Exception as e:
                print(f"Error in evaluate: {e}")
                reason = '<Reason>This answer is unavaliable</Reason><Score>1</Score>'
                score_tmp.append(1.0)
                output_tmp.append(reason)
        score_total.append(score_tmp)
        output_total.append(output_tmp)

    # ========== 第二轮：5次验证异步并发 ==========
    QALIST = {
        "role": "user",
        "content": [
            {"type": "image_pil", "image_pil": image},
            {"type": "text", "text": prompt_template.AGG_SCORE.format(
                task=prompt_template.task_1, prompt='', evaluations=str(output_total[0]))},
        ],
    }
    CONLIST = {
        "role": "user",
        "content": [
            {"type": "image_pil", "image_pil": ref_image},
            {"type": "image_pil", "image_pil": image},
            {"type": "text", "text": prompt_template.AGG_SCORE.format(
                task=prompt_template.task_2, prompt="", evaluations=str(output_total[1]))},
        ],
    }
    SCELIST = {
        "role": "user",
        "content": [
            {"type": "image_pil", "image_pil": scene_image},
            {"type": "image_pil", "image_pil": image},
            {"type": "text", "text": prompt_template.AGG_SCORE.format(
                task=prompt_template.task_3, prompt="", evaluations=str(output_total[2]))},
        ],
    }
    HARLIST = {
        "role": "user",
        "content": [
            {"type": "image_pil", "image_pil": ref_image},
            {"type": "image_pil", "image_pil": image},
            {"type": "text", "text": prompt_template.AGG_SCORE.format(
                task=prompt_template.task_4, prompt="", evaluations=str(output_total[3]))},
        ],
    }
    INSLIST = {
        "role": "user",
        "content": [
            {"type": "image_pil", "image_pil": ref_image},
            {"type": "image_pil", "image_pil": scene_image},
            {"type": "image_pil", "image_pil": image},
            {"type": "text", "text": prompt_template.AGG_SCORE.format(
                task=prompt_template.task_5, prompt='Instruction: ' + prompt, evaluations=str(output_total[4]))},
        ],
    }

    verify_conversation = [QALIST, CONLIST, SCELIST, HARLIST, INSLIST]
    verify_tasks = []
    for item in verify_conversation:
        messages = build_api_messages(item)
        verify_tasks.append(api_call(client, messages, semaphore, max_tokens=4096, temperature=0.6, top_p=0.95))

    verify_results = await asyncio.gather(*verify_tasks)

    # 解析验证结果 -> mask[5][5]
    mask = []
    for output_text in verify_results:
        print(output_text)
        try:
            ValidList = re.search(r'<Answer>(.*?)</Answer>', output_text, re.DOTALL)
            ValidList = ValidList.group(1).strip()
            ValidList = ast.literal_eval(ValidList)
        except:
            ValidList = [1, 1, 1, 1, 1]
        mask.append(ValidList)

    # ========== 计算最终reward ==========
    final_score = []
    for i in range(len(score_total)):
        try:
            valid_item = [d for m, d in zip(mask[i], score_total[i]) if m == 1]
            if valid_item == []:
                final_score.append(sum(score_total[i]) / len(score_total[i]))
            else:
                final_score.append(sum(valid_item) / len(valid_item))
        except:
            if score_total[i] == []:
                final_score.append(1.0)

    reward = pow(final_score[0] * final_score[1] * final_score[2] * final_score[3] * final_score[4], 1 / 5)
    reward = (reward - 1) / 4
    return reward


async def evaluate_images_batch_async(image_bytes_list, prompts, ref_image_bytes_list=None, sce_image_bytes_list=None, requirements=[]):
    """批量异步评估多张图片"""
    if not requirements:
        requirements = [""] * len(prompts)
    if ref_image_bytes_list is None:
        ref_image_bytes_list = [None] * len(prompts)
    if sce_image_bytes_list is None:
        sce_image_bytes_list = [None] * len(prompts)

    # 每个请求自己的事件循环创建自己的 client，避免连接池跨循环复用
    client = AsyncOpenAI(api_key=VLM_API_KEY, base_url=VLM_URL)

    # 创建信号量限制全局并发数
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_API_CALLS)

    tasks = []
    for image_bytes, prompt, ref_image_bytes, sce_image_bytes, requirement in zip(
        image_bytes_list, prompts, ref_image_bytes_list, sce_image_bytes_list, requirements
    ):
        tasks.append(evaluate_image_async(client, image_bytes, prompt, ref_image_bytes, sce_image_bytes, requirement, semaphore))

    scores = await asyncio.gather(*tasks)
    return list(scores)


def evaluate_images(image_bytes_list, prompts, ref_image_bytes_list=None, sce_image_bytes_list=None, requirements=[]):
    """同步入口，供Flask路由调用"""
    scores = asyncio.run(
        evaluate_images_batch_async(image_bytes_list, prompts, ref_image_bytes_list, sce_image_bytes_list, requirements)
    )
    return scores


@app.route("/mode/<mode>", methods=["POST"])
def inference_mode(mode):
    data = request.get_data()

    assert mode in ["logits_non_cot"], "Invalid mode"

    data = pickle.loads(data)
    image_bytes_list = data["images"]
    ref_image_bytes_list = data.get("ref_images", None)
    sce_image_bytes_list = data.get("sce_images", None)
    prompts = data["prompts"]
    metadatas = data.get("metadatas", [])
    requirements = []
    for metadata in metadatas:
        requirements.append(metadata.get("requirement", ""))

    scores = evaluate_images(
        image_bytes_list, prompts, ref_image_bytes_list, sce_image_bytes_list, requirements
    )

    response = {"scores": scores}
    response = pickle.dumps(response)
    returncode = 200

    return response, returncode


if __name__ == "__main__":
    print(f"Starting Flask server with external VLM API...")
    print(f"VLM_URL: {VLM_URL}")
    print(f"VLM_MODEL: {VLM_MODEL}")
    app.run(host="0.0.0.0", port=12343, debug=False)
