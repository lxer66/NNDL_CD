import json
import os
import random
from collections import Counter

# 配置
# 输入路径
captions_json_path = 'deepfashion_data/json/captions.json'
image_folder = 'deepfashion_data/images'

# 输出路径
output_json_path = 'deepfashion_data/dataset_deepfashion.json'

# 参数设置
max_len = 100       # 截断超过100个词的描述
min_word_freq = 5   # 词频阈值，出现少于5次的词会被标记为 <unk>

def main():
    print("正在读取原始 JSON 文件...")
    with open(captions_json_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    print(f"原始数据共包含 {len(raw_data)} 张图片。")

    dataset = {
        "images": [],
        "dataset": "deepfashion"
    }

    # 用于划分数据集 (85% 训练, 5% 验证, 10% 测试)
    # 设置随机种子保证每次运行结果一致
    random.seed(42)
    
    print("正在处理文本并构建数据集结构...")
    count = 0
    
    # 遍历字典 items
    for filename, caption in raw_data.items():
        # 1. 检查图片文件是否存在 ，防止报错
        full_img_path = os.path.join(image_folder, filename)
        if not os.path.exists(full_img_path):
            print(f"警告: 找不到图片 {filename}，已跳过。")
            continue

        # 2. 处理文本 (Tokenization)
        # 转小写，替换掉句号和逗号，按空格切分
        tokens = caption.lower().replace('.', ' .').replace(',', ' ,').split()
        # 移除空字符
        tokens = [t for t in tokens if t]
        
        # 截断过长文本
        if len(tokens) > max_len:
            tokens = tokens[:max_len]

        # 3. 随机划分 Train/Val/Test
        rand = random.random()
        if rand < 0.85:
            split = 'train'
        elif rand < 0.90:
            split = 'val'
        else:
            split = 'test'

        # 4. 构建单个条目
        img_entry = {
            "filepath": "",
            "filename": filename,
            "imgid": count,
            "split": split,
            "sentences": [
                {
                    "tokens": tokens,
                    "raw": caption
                }
            ]
        }
        
        dataset["images"].append(img_entry)
        count += 1

    # 保存新的 JSON
    print(f"正在保存转换后的数据到 {output_json_path} ...")
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f)

    print(f"完成！成功处理 {count} 张图片。")

if __name__ == '__main__':
    main()