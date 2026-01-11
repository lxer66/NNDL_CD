import os
import json
import torch
import random
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# 第一次运行前请在终端设置环境变量，执行：export HF_ENDPOINT=https://hf-mirror.com

IMAGE_FOLDER = "deepfashion_data/aug_images"
ORIGINAL_JSON_PATH = "deepfashion_data/json/captions.json"
OUTPUT_JSON_PATH = "deepfashion_data/json/captions_aug.json"
MODEL_PATH = "Qwen/Qwen2-VL-7B-Instruct"

# 关键加速参数
# 建议按显存修改
BATCH_SIZE = 24
# CPU读取线程数
NUM_WORKERS = 12

def load_model():
    print("正在加载 Qwen2-VL-7B 模型...")
    # 使用 bfloat16 + flash_attention_2 达到最快推理速度
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    return model, processor

class ImageCaptionDataset(Dataset):
    """
    自定义数据集，用于批量读取图片和文本
    """
    def __init__(self, image_folder, caption_map, processed_files):
        self.image_folder = image_folder
        self.caption_map = caption_map
        # 过滤掉已经处理过的和JSON里没有的
        self.file_list = [
            f for f in os.listdir(image_folder) 
            if f in caption_map and f not in processed_files and f.lower().endswith(('.jpg', '.png'))
        ]
        print(f"待处理任务队列: {len(self.file_list)} 张图片")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        filename = self.file_list[idx]
        img_path = os.path.join(self.image_folder, filename)
        
        # 尝试预读取图片，如果坏了返回 None
        try:
            image = Image.open(img_path).convert("RGB")
            return {
                "image": image,
                "filename": filename
            }
        except Exception as e:
            print(f"读取错误 {filename}: {e}")
            return None

def custom_collate_fn(batch):
    # 过滤掉读取失败的 None 数据
    batch = [item for item in batch if item is not None]
    return batch

def main():
    # 1. 读取历史进度
    if os.path.exists(OUTPUT_JSON_PATH):
        with open(OUTPUT_JSON_PATH, 'r', encoding='utf-8') as f:
            augmented_data = json.load(f)
        print(f"已加载历史进度: {len(augmented_data)} 条")
    else:
        augmented_data = {}

    # 2. 读取原始 JSON (扁平结构)
    with open(ORIGINAL_JSON_PATH, 'r', encoding='utf-8') as f:
        caption_map = json.load(f) # 直接读取字典

    # 3. 加载模型
    model, processor = load_model()

    # 4. 构建 DataLoader
    dataset = ImageCaptionDataset(IMAGE_FOLDER, caption_map, set(augmented_data.keys()))
    if len(dataset) == 0:
        print("所有图片均已处理完成！")
        return

    dataloader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=NUM_WORKERS,
        collate_fn=custom_collate_fn # 处理坏图
    )

    # 5. 批量推理循环
    print(f"开始批量处理，Batch Size = {BATCH_SIZE}...")
    
    # 计数器用于定期保存
    save_counter = 0
    
    verbs = ["wears", "is dressed in", "sports", "features", "has", "models"]

    for batch in tqdm(dataloader):
        if not batch: continue # 跳过空批次

        images = [item['image'] for item in batch]
        filenames = [item['filename'] for item in batch]

        # 构造批量 Messages
        messages_list = []
        
        for _ in range(len(images)):

            chosen_verb = random.choice(verbs)

            prompt_text = (
                "Describe the image in a detailed but simple paragraph (strictly 40-60 words). "
                "Strictly follow these rules:\n"
                "1. **VERB VARIETY:** Do NOT repeat the same verb (e.g., 'sports') for both the top and bottom clothing. Mix them up (e.g., use 'wears' for top, 'has on' for bottom).\n"
               f"2. **ATTENTION:** Start the sentence STRICTLY with 'A man/guy/woman/lady {chosen_verb}...'."
                "3. **SENTENCE VARIETY:** For the second sentence (bottom clothing), you MUST vary the structure. Do NOT just use 'He has on'.\n"
                "   - Option A: 'His [pants/jeans...] are...'\n"
                "   - Option B: 'He wears...'\n"
                "   - Option C: 'He has on...'\n"
                "   ... and so on"
                "4. **DIRECTNESS:** Do NOT fucking use phrases like 'He pairs this with' or 'He completes the look'. Just describe the items directly. Avoid flowery transitions.\n"
                "5. **CONTENT:** Focus 80% on clothing details (**fit, pattern, neckline, material**) and 20% on background.\n"
                "6. **FORBIDDEN:** DO NOT describe the pose (e.g., standing, sitting, hands in pockets), lighting, or use the word 'confidently'.\n\n"

                "Here are three examples of the VARIED style I want:\n"
                "- 'A young man wears a fitted heather grey short-sleeve t-shirt with a classic crew neckline. His black slim-fit jeans feature distinct white stitching and he wears black sneakers. The background shows a blurred city street with a red bus.'\n"
                "- 'A guy is dressed in a maroon sleeveless tank top made of smooth cotton fabric with a deep round neckline. He wears black trousers with a subtle vertical side stripe and a relaxed fit. On his feet are clean white leather sneakers. Behind him is a textured light brown stone wall.'\n"
                "- 'A male model sports a blue and white vertical striped long-sleeve shirt with a pointed collar and buttoned cuffs. He has on beige chino pants with a straight-leg fit and a smooth texture. The setting is a room filled with wooden tables.'\n\n"

                "Now, describe the provided image following this style:"
            )

            messages_list.append([
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": "placeholder"}, 
                        {"type": "text", "text": prompt_text},
                    ],
                }
            ])

        # 批量预处理
        texts = [
            processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            for msg in messages_list
        ]
        
        # Qwen2-VL 的 Processor 支持批量 image 处理
        inputs = processor(
            text=texts,
            images=images,
            padding=True,
            return_tensors="pt"
        )
        inputs = inputs.to("cuda")

        # 批量生成
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=128, temperature=0.8, top_p=0.9, do_sample=True)

        # 批量解码
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_texts = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        # 存入结果
        for fname, new_cap in zip(filenames, output_texts):
            augmented_data[fname] = new_cap
        
        # 定期保存 (每处理 10 个 Batch 存一次)
        save_counter += 1
        if save_counter % 10 == 0:
            with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:
                json.dump(augmented_data, f, ensure_ascii=False, indent=2)

    # 最终保存
    with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(augmented_data, f, ensure_ascii=False, indent=2)
    print("全部完成！")

if __name__ == '__main__':
    main()