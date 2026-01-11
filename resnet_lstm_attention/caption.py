import torch
import torch.nn.functional as F
import numpy as np
import json
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import skimage.transform
import argparse
from PIL import Image
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def caption_image_beam_search(encoder, decoder, image_path, word_map, beam_size):
    """
    Reads an image and captions it with beam search.
    """

    k = beam_size
    vocab_size = len(word_map)

    # 1. 读取图片
    img = Image.open(image_path).convert('RGB')
    
    # 2. 定义和训练时一模一样的预处理
    transform = transforms.Compose([
        transforms.Resize((518, 518)), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    # 3. 应用预处理
    image = transform(img)  # (3, 518, 518)

    # Encode
    image = image.unsqueeze(0).to(device)
    encoder_out = encoder(image)
    
    # 获取特征图长宽
    enc_image_h = encoder_out.size(1) 
    enc_image_w = encoder_out.size(2) 
    encoder_dim = encoder_out.size(3) 

    # Flatten encoding
    encoder_out = encoder_out.view(1, -1, encoder_dim)  # (1, num_pixels, encoder_dim)
    num_pixels = encoder_out.size(1)

    # Expand to batch size k
    encoder_out = encoder_out.expand(k, num_pixels, encoder_dim)

    # Tensor to store top k previous words at each step
    k_prev_words = torch.LongTensor([[word_map['<start>']]] * k).to(device) # (k, 1)
    seqs = k_prev_words 
    top_k_scores = torch.zeros(k, 1).to(device) 
    seqs_alpha = torch.ones(k, 1, enc_image_h, enc_image_w).to(device)

    complete_seqs = list()
    complete_seqs_alpha = list()
    complete_seqs_scores = list()

    step = 1
    h, c = decoder.init_hidden_state(encoder_out)

    while True:
        embeddings = decoder.embedding(k_prev_words).squeeze(1)
        awe, alpha = decoder.attention(encoder_out, h)
        
        # Reshape alpha
        alpha = alpha.view(-1, enc_image_h, enc_image_w)

        gate = decoder.sigmoid(decoder.f_beta(h))
        awe = gate * awe

        h, c = decoder.decode_step(torch.cat([embeddings, awe], dim=1), (h, c))

        scores = decoder.fc(h)
        scores = F.log_softmax(scores, dim=1)

        # 1. 屏蔽词列表 (Blacklist)
        # 标签平滑 Label Smoothing 的副作用：模型有时会倾向于生成 <unk> 词，将分数设置为负无穷屏蔽。
        poison_words = ['<unk>'] 
        for pw in poison_words:
            if pw in word_map:
                scores[:, word_map[pw]] = -float('inf')

        # 2. 强力防复读补丁
        # 逻辑：检测当前生成的句子里，哪些词已经出现过了。
        # 如果出现过，就把得分扣掉一大截，强迫模型换个词说。
        if step > 1:
            for i in range(k): # 遍历 beam (k=1时就是当前序列)
                current_seq_indices = seqs[i].tolist()
                for word_idx in current_seq_indices:
                    # 跳过 <start>, <unk> 等特殊词
                    if word_idx not in [word_map['<start>'], word_map['<unk>'], word_map['<pad>']]:
                        # 惩罚系数：5.0
                        scores[i, word_idx] -= 5.0 

        scores = top_k_scores.expand_as(scores) + scores

        if step == 1:
            top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)
        else:
            top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)

        prev_word_inds = top_k_words // vocab_size
        next_word_inds = top_k_words % vocab_size

        seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)
        seqs_alpha = torch.cat([seqs_alpha[prev_word_inds], alpha[prev_word_inds].unsqueeze(1)], dim=1)

        incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if next_word != word_map['<end>']]
        complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

        if len(complete_inds) > 0:
            complete_seqs.extend(seqs[complete_inds].tolist())
            complete_seqs_alpha.extend(seqs_alpha[complete_inds].tolist())
            complete_seqs_scores.extend(top_k_scores[complete_inds])
        k -= len(complete_inds)

        if k == 0:
            break
        
        seqs = seqs[incomplete_inds]
        seqs_alpha = seqs_alpha[incomplete_inds]
        h = h[prev_word_inds[incomplete_inds]]
        c = c[prev_word_inds[incomplete_inds]]
        encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
        top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
        k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

        # 防止死循环：超过50个词强制停止
        if step > 50:
            break
        step += 1

    if len(complete_seqs_scores) > 0:
        i = complete_seqs_scores.index(max(complete_seqs_scores))
        seq = complete_seqs[i]
        alphas = complete_seqs_alpha[i]
    else:
        seq = seqs[0].tolist()
        alphas = seqs_alpha[0].tolist()

    return seq, alphas


def visualize_att(image_path, seq, alphas, rev_word_map, smooth=True):
    image = Image.open(image_path)
    # 这里的 resize 只为了可视化显示，不影响模型推理
    image = image.resize([518, 518], Image.LANCZOS) 

    words = [rev_word_map[ind] for ind in seq]
    
    # 打印最终结果（去掉 start 和 end）
    result_caption = []
    for w in words:
        if w == '<start>': continue
        if w == '<end>': break
        result_caption.append(w)
    
    print("\nGenerated Caption: ", " ".join(result_caption))

    # 如果生成的词太多，只画前 50 个
    limit = min(len(words), 50)
    
    # 计算子图布局
    n_cols = 5
    n_rows = int(np.ceil(limit / float(n_cols)))
    
    plt.figure(figsize=(15, 3 * n_rows)) # 调整图片大小

    for t in range(limit):
        plt.subplot(n_rows, n_cols, t + 1)
        plt.text(0, 1, '%s' % (words[t]), color='black', backgroundcolor='white', fontsize=12)
        plt.imshow(image)
        
        current_alpha = alphas[t, :]
        if smooth:
            alpha = skimage.transform.pyramid_expand(current_alpha.numpy(), upscale=32, sigma=8)
        else:
            alpha = skimage.transform.resize(current_alpha.numpy(), [518, 518])
            
        if t == 0:
            plt.imshow(alpha, alpha=0)
        else:
            plt.imshow(alpha, alpha=0.8)
            
        plt.set_cmap(cm.Greys_r)
        plt.axis('off')
    
    # 保存结果而不是直接 show，服务器上看不见弹窗
    # 做前端时需修改
    save_name = "result_" + os.path.basename(image_path)
    plt.savefig(save_name)
    print(f"可视化结果已保存为: {save_name}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DeepFashion Captioning')
    parser.add_argument('--img', '-i', help='path to image')
    parser.add_argument('--model', '-m', help='path to model')
    parser.add_argument('--word_map', '-wm', help='path to word map JSON')
    parser.add_argument('--beam_size', '-b', default=1, type=int, help='beam size for beam search')
    parser.add_argument('--dont_smooth', dest='smooth', action='store_false', help='do not smooth alpha overlay')

    args = parser.parse_args()
    
    # 默认路径
    # 根据文件路径调整
    img_path = args.img if args.img else 'deepfashion_data/aug_images/MEN-Shirts_Polos-id_00003706-01_1_front.jpg'
    model_path = args.model if args.model else '13_checkpoint_deepfashion_aug_1_cap_per_img_5_min_word_freq.pth.tar'
    word_map_path = args.word_map if args.word_map else 'deepfashion_data/WORDMAP_deepfashion_aug_1_cap_per_img_5_min_word_freq.json'

    # Load model
    checkpoint = torch.load(model_path, map_location=str(device), weights_only=False)
    
    decoder = checkpoint['decoder']
    decoder = decoder.to(device)
    decoder.eval()
    encoder = checkpoint['encoder']
    encoder = encoder.to(device)
    encoder.eval()

    # Load word map
    with open(word_map_path, 'r') as j:
        word_map = json.load(j)
    rev_word_map = {v: k for k, v in word_map.items()}

    seq, alphas = caption_image_beam_search(encoder, decoder, img_path, word_map, args.beam_size)
    alphas = torch.FloatTensor(alphas)

    visualize_att(img_path, seq, alphas, rev_word_map, args.smooth)