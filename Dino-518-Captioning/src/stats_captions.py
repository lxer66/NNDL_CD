import json
from pathlib import Path
from statistics import mean
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CAPTIONS_PATH = PROJECT_ROOT / "data" / "captions_aug.json"
TOKENIZER_PATH = PROJECT_ROOT / "models" / "flan-t5-base"

def main():
    with open(CAPTIONS_PATH, "r", encoding="utf-8") as f:
        caps = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)

    word_lens = []
    token_lens = []

    for _, cap in caps.items():
        cap_str = cap.strip()
        word_lens.append(len(cap_str.split()))
        token_lens.append(len(tokenizer(cap_str, add_special_tokens=False).input_ids))

    print(f"Samples: {len(word_lens)}")
    print(f"Words  -> avg: {mean(word_lens):.2f}, max: {max(word_lens)}")
    print(f"Tokens -> avg: {mean(token_lens):.2f}, max: {max(token_lens)}")

if __name__ == "__main__":
    main()