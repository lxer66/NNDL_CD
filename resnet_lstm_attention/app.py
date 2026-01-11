import torch
import torch.nn.functional as F
import numpy as np
import json
import torchvision.transforms as transforms
from PIL import Image
import gradio as gr
from models import Encoder, DecoderWithAttention

# 模型和词表路径
MODEL_PATH = '13_checkpoint_deepfashion_aug_1_cap_per_img_5_min_word_freq.pth.tar'
WORDMAP_PATH = 'WORDMAP_deepfashion_aug_1_cap_per_img_5_min_word_freq.json'

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载资源
print("正在加载词表...")
with open(WORDMAP_PATH, 'r') as j:
    word_map = json.load(j)
rev_word_map = {v: k for k, v in word_map.items()}
vocab_size = len(word_map)

print(f"正在加载模型: {MODEL_PATH}")
checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)

encoder = checkpoint['encoder'].to(device)
decoder = checkpoint['decoder'].to(device)
encoder.eval()
decoder.eval()

# 预处理 (与训练时一致：518x518)
transform = transforms.Compose([
    transforms.Resize((518, 518)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# 推理核心逻辑
def generate_caption(image_pil, beam_size=1):
    """
    输入: PIL图片对象
    输出: 生成的文本字符串
    """
    # 1. 图片预处理
    if image_pil is None:
        return "请先上传一张图片！"
    
    image = image_pil.convert('RGB')
    image = transform(image)
    image = image.unsqueeze(0).to(device) # (1, 3, 518, 518)

    # 2. 编码
    encoder_out = encoder(image)  # (1, 14, 14, 2048)
    enc_image_size = encoder_out.size(1)
    encoder_dim = encoder_out.size(3)
    
    # Flatten
    encoder_out = encoder_out.view(1, -1, encoder_dim)  # (1, num_pixels, encoder_dim)
    num_pixels = encoder_out.size(1)
    
    # Expand for beam search
    k = beam_size
    encoder_out = encoder_out.expand(k, num_pixels, encoder_dim)

    # 3. 解码准备
    k_prev_words = torch.LongTensor([[word_map['<start>']]] * k).to(device)
    seqs = k_prev_words
    top_k_scores = torch.zeros(k, 1).to(device)
    
    step = 1
    h, c = decoder.init_hidden_state(encoder_out)

    complete_seqs = list()
    complete_seqs_scores = list()

    # 4. 逐步解码
    while True:
        embeddings = decoder.embedding(k_prev_words).squeeze(1)
        awe, _ = decoder.attention(encoder_out, h)
        gate = decoder.sigmoid(decoder.f_beta(h))
        awe = gate * awe

        h, c = decoder.decode_step(torch.cat([embeddings, awe], dim=1), (h, c))
        scores = decoder.fc(h)
        scores = F.log_softmax(scores, dim=1)

        # 屏蔽词 (Blacklist)
        poison_words = ['<unk>'] 
        for pw in poison_words:
            if pw in word_map:
                scores[:, word_map[pw]] = -float('inf')

        # 防复读
        if step > 1:
            for i in range(k):
                current_seq_indices = seqs[i].tolist()
                for word_idx in current_seq_indices:
                    if word_idx not in [word_map['<start>'], word_map['<unk>'], word_map['<pad>']]:
                        scores[i, word_idx] -= 5.0 # 惩罚已出现的词

        scores = top_k_scores.expand_as(scores) + scores

        if step == 1:
            top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)
        else:
            top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)

        prev_word_inds = top_k_words // vocab_size
        next_word_inds = top_k_words % vocab_size

        seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)
        
        incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if next_word != word_map['<end>']]
        complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

        if len(complete_inds) > 0:
            complete_seqs.extend(seqs[complete_inds].tolist())
            complete_seqs_scores.extend(top_k_scores[complete_inds])
        k -= len(complete_inds)

        if k == 0: break
        
        seqs = seqs[incomplete_inds]
        h = h[prev_word_inds[incomplete_inds]]
        c = c[prev_word_inds[incomplete_inds]]
        encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
        top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
        k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

        if step > 50: break
        step += 1

    # 5. 选择最佳结果
    if len(complete_seqs_scores) > 0:
        i = complete_seqs_scores.index(max(complete_seqs_scores))
        seq = complete_seqs[i]
    else:
        seq = seqs[0].tolist()

    # 6. 转换回文本
    # 过滤掉 <start>, <end>, <pad>
    result_words = [
        rev_word_map[ind] 
        for ind in seq 
        if rev_word_map[ind] not in {'<start>', '<end>', '<pad>'}
    ]
    
    caption = " ".join(result_words)
    # 首字母大写，加句号
    caption = caption.capitalize()
    if not caption.endswith('.'):
        caption += '.'
        
    return caption

# 构建 UI 界面
def main():
    # Gradio 界面定义
    demo = gr.Interface(
        fn=generate_caption, # 绑定的处理函数
        inputs=gr.Image(type="pil", label="上传服饰图片"), # 输入组件
        outputs=gr.Textbox(label="生成描述"), # 输出组件
        title="DeepFashion 图像描述生成器",
        description="基于 ResNet-101 + LSTM + Attention 的服饰图像描述模型",
        examples=[
            ["MEN-Denim-id_00000080-01_7_additional.jpg"], 
            ["MEN-Shirts_Polos-id_00003706-01_1_front.jpg"]
        ] # 示例图片路径，方便演示。需做修改
    )
    
    # 启动服务
    demo.launch(share=True) # share=True

if __name__ == '__main__':
    main()