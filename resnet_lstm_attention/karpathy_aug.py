import json
import os

# 配置
# 1. 旧的 Karpathy JSON (为了获取 train/val/test 的划分信息)
OLD_KARPATHY_JSON = 'deepfashion_data/dataset_deepfashion.json'

# 2. 大模型新生成的字典 JSON
AUGMENTED_JSON = 'deepfashion_data/json/captions_aug.json'

# 3. 输出的最终 JSON (给 create_input_files.py 用)
OUTPUT_KARPATHY_JSON = 'deepfashion_data/dataset_deepfashion_aug.json'

def main():
    print(f"正在读取旧划分信息: {OLD_KARPATHY_JSON}")
    with open(OLD_KARPATHY_JSON, 'r', encoding='utf-8') as f:
        old_data = json.load(f)

    print(f"正在读取新生成的描述: {AUGMENTED_JSON}")
    with open(AUGMENTED_JSON, 'r', encoding='utf-8') as f:
        aug_data = json.load(f) # 字典 {filename: text}

    new_dataset = {
        "dataset": "deepfashion_aug", # 改个新名字
        "images": []
    }

    count = 0
    skipped = 0

    print("正在合并数据...")
    # 遍历旧数据，保持原有的 split 划分不变
    for img in old_data['images']:
        filename = img['filename']
        
        # 检查该图是否生成新的描述
        if filename in aug_data:
            new_caption = aug_data[filename]
            
            # 简单的分词 (Tokenization)，去标点
            tokens = new_caption.lower().replace('.', ' .').replace(',', ' ,').split()
            tokens = [t for t in tokens if t]

            # 构建新的条目
            new_entry = {
                "filepath": "",
                "filename": filename,
                "imgid": img['imgid'],
                "split": img['split'], # 继承旧的划分
                "sentences": [
                    {
                        "tokens": tokens,
                        "raw": new_caption
                    }
                ]
            }
            new_dataset['images'].append(new_entry)
            count += 1
        else:
            skipped += 1

    print(f"合并完成！共包含 {count} 张图片。")
    print(f"跳过(未找到新描述) {skipped} 张图片。")

    with open(OUTPUT_KARPATHY_JSON, 'w', encoding='utf-8') as f:
        json.dump(new_dataset, f)
    
    print(f"新数据集已保存至: {OUTPUT_KARPATHY_JSON}")

if __name__ == '__main__':
    main()