import json
import re
from collections import Counter
from pathlib import Path
from statistics import mean

import nltk
from nltk.corpus import stopwords

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CAPTIONS_PATH = PROJECT_ROOT / "data" / "captions_aug.json"


def load_captions() -> dict:
    with open(CAPTIONS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def get_stopwords() -> set:
    try:
        return set(stopwords.words("english"))
    except LookupError:
        nltk.download("stopwords", quiet=True)
        return set(stopwords.words("english"))


def tokenize(text: str) -> list:
    return re.findall(r"[A-Za-z']+", text.lower())


def analyze():
    captions = load_captions()
    stopword_set = get_stopwords()

    all_tokens = []
    tokens_no_stop = []
    per_caption_lengths = []
    per_caption_lengths_no_stop = []

    for caption in captions.values():
        words = tokenize(caption)
        filtered = [w for w in words if w not in stopword_set]

        all_tokens.extend(words)
        tokens_no_stop.extend(filtered)
        per_caption_lengths.append(len(words))
        per_caption_lengths_no_stop.append(len(filtered))

    freq_all = Counter(all_tokens)
    freq_no_stop = Counter(tokens_no_stop)

    print(f"样本数: {len(captions)}")
    print(f"总词数: {len(all_tokens)}，不含停用词: {len(tokens_no_stop)}")
    print(f"不同词数: {len(freq_all)}，不含停用词: {len(freq_no_stop)}")
    print(f"平均每条词数: {mean(per_caption_lengths):.2f}，不含停用词: {mean(per_caption_lengths_no_stop):.2f}")
    print(f"最大词数: {max(per_caption_lengths)}，不含停用词: {max(per_caption_lengths_no_stop)}")

    print("\n最常见的 20 个词（含停用词）:")
    for word, count in freq_all.most_common(20):
        print(f"  {word}: {count}")

    print("\n最常见的 20 个非停用词:")
    for word, count in freq_no_stop.most_common(20):
        print(f"  {word}: {count}")


if __name__ == "__main__":
    analyze()