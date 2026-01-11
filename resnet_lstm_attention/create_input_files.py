from utils import create_input_files

if __name__ == '__main__':
    # 创建输入文件
    create_input_files(
        dataset='deepfashion_aug',
        karpathy_json_path='deepfashion_data/dataset_deepfashion_aug.json', # 上一步生成的json
        image_folder='deepfashion_data/aug_images',
        captions_per_image=1,       # DeepFashion每张图只有1个描述
        min_word_freq=5,            # 过滤生僻词
        output_folder='deepfashion_data/', # 输出位置
        max_len=100                 # 最大文本长度
    )