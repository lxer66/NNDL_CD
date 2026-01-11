import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from datasets import *
from utils import *
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
import torch.nn.functional as F
from tqdm import tqdm
import json
import os
import numpy as np

# 配置
data_folder = './deepfashion_data'
data_name = 'deepfashion_aug_1_cap_per_img_5_min_word_freq'
checkpoint_file = '13_checkpoint_deepfashion_aug_1_cap_per_img_5_min_word_freq.pth.tar'  # 选择模型文件
word_map_file = os.path.join(data_folder, 'WORDMAP_' + data_name + '.json')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True

def evaluate(beam_size):
    """
    Evaluation
    :param beam_size: beam size at which to generate captions for evaluation
    :return: BLEU-4 score
    """
    # Load model
    print(f"正在加载模型: {checkpoint_file}")
    checkpoint = torch.load(checkpoint_file, map_location=device, weights_only=False)
    decoder = checkpoint['decoder']
    decoder = decoder.to(device)
    decoder.eval()
    encoder = checkpoint['encoder']
    encoder = encoder.to(device)
    encoder.eval()

    # Load word map
    with open(word_map_file, 'r') as j:
        word_map = json.load(j)
    rev_word_map = {v: k for k, v in word_map.items()}
    vocab_size = len(word_map)

    # Normalization transform
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    eval_transform = transforms.Compose([
        transforms.Resize((518, 518)), 
        transforms.ToTensor(),
        normalize
    ])

    # DataLoader
    loader = torch.utils.data.DataLoader(
        CaptionDataset(data_folder, data_name, 'TEST', transform=eval_transform),
        batch_size=1, shuffle=True, num_workers=8, pin_memory=True)

    # Lists to store references (true captions), and hypothesis (prediction) for each image
    references = list()
    hypotheses = list()

    print("开始在测试集上生成描述 ...")

    # For each image
    for i, (image, caps, caplens, allcaps) in enumerate(
            tqdm(loader, desc="EVALUATING AT BEAM SIZE " + str(beam_size))):

        k = beam_size
        image = image.to(device)

        # Encode
        encoder_out = encoder(image)
        enc_image_h = encoder_out.size(1)
        enc_image_w = encoder_out.size(2)
        encoder_dim = encoder_out.size(3)

        encoder_out = encoder_out.view(1, -1, encoder_dim)
        num_pixels = encoder_out.size(1)
        encoder_out = encoder_out.expand(k, num_pixels, encoder_dim)

        k_prev_words = torch.LongTensor([[word_map['<start>']]] * k).to(device)
        seqs = k_prev_words
        top_k_scores = torch.zeros(k, 1).to(device)

        complete_seqs = list()
        complete_seqs_scores = list()

        step = 1
        h, c = decoder.init_hidden_state(encoder_out)

        while True:
            embeddings = decoder.embedding(k_prev_words).squeeze(1)
            awe, _ = decoder.attention(encoder_out, h)
            gate = decoder.sigmoid(decoder.f_beta(h))
            awe = gate * awe

            h, c = decoder.decode_step(torch.cat([embeddings, awe], dim=1), (h, c))
            scores = decoder.fc(h)
            scores = F.log_softmax(scores, dim=1)
            
            # 加入推理干预 ，防止测试时复读
            if step > 1:
                for ix in range(k):
                    current_seq_indices = seqs[ix].tolist()
                    for word_idx in current_seq_indices:
                        if word_idx not in [word_map['<start>'], word_map['<unk>'], word_map['<pad>']]:
                            scores[ix, word_idx] -= 5.0 # 惩罚重复词

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

        if len(complete_seqs_scores) > 0:
            i = complete_seqs_scores.index(max(complete_seqs_scores))
            seq = complete_seqs[i]
        else:
            seq = seqs[0].tolist()

        # References
        img_caps = allcaps[0].tolist()
        img_captions = list(
            map(lambda c: [w for w in c if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}],
                img_caps))
        references.append(img_captions)

        # Hypotheses
        hypotheses.append([w for w in seq if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}])

        assert len(references) == len(hypotheses)

    # 计算指标
    print("\n正在计算指标 (BLEU-4, METEOR, ROUGE-L) ...")
    
    # BLEU-4
    bleu4 = corpus_bleu(references, hypotheses)

    # METEOR 和 ROUGE
    meteor_scores = []
    rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rouge_l_scores = []

    for i in range(len(hypotheses)):
        # 转换 ID 为 单词列表
        hyp_words = [rev_word_map[w] for w in hypotheses[i]]
        ref_words_list = [[rev_word_map[w] for w in ref] for ref in references[i]]
        
        # 还原成句子字符串
        hyp_str = " ".join(hyp_words)
        # 取第一个参考句子作为 ROUGE 的参考
        ref_str = " ".join(ref_words_list[0]) 

        # METEOR (NLTK)
        # NLTK 的 meteor_score 需要 word token lists
        meteor = meteor_score(ref_words_list, hyp_words)
        meteor_scores.append(meteor)

        # ROUGE-L
        scores = rouge.score(ref_str, hyp_str)
        rouge_l_scores.append(scores['rougeL'].fmeasure)

    avg_meteor = np.mean(meteor_scores)
    avg_rouge_l = np.mean(rouge_l_scores)

    return bleu4, avg_meteor, avg_rouge_l


if __name__ == '__main__':
    # 使用 beam_size = 1，和推理时的最佳实践保持一致
    beam_size = 1
    
    bleu4, meteor, rouge_l = evaluate(beam_size)
    
    print("\n-------------------------------------------")
    print(f"Test Set Results @ Beam Size {beam_size}")
    print("-------------------------------------------")
    print(f"BLEU-4:   {bleu4:.4f}")
    print(f"METEOR:   {meteor:.4f}")
    print(f"ROUGE-L:  {rouge_l:.4f}")
    print("-------------------------------------------")