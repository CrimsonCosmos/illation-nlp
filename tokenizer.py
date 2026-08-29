"""Train / load a small byte-level BPE tokenizer for the illation model."""
from pathlib import Path

from tokenizers import ByteLevelBPETokenizer

HERE = Path(__file__).parent
TOK_DIR = HERE / "tokenizer_files"
VOCAB_SIZE = 8000

SPECIAL_TOKENS = ["<|pad|>", "<|bos|>", "<|eos|>", "<|thought_start|>", "<|thought_end|>", "<|lookup|>"]


def train(corpus_path: Path = HERE / "data" / "corpus.txt"):
    TOK_DIR.mkdir(exist_ok=True)
    tok = ByteLevelBPETokenizer()
    tok.train(
        files=[str(corpus_path)],
        vocab_size=VOCAB_SIZE,
        min_frequency=2,
        special_tokens=SPECIAL_TOKENS,
    )
    tok.save_model(str(TOK_DIR))
    print(f"tokenizer trained, vocab={tok.get_vocab_size()}, saved to {TOK_DIR}")
    return tok


def load() -> ByteLevelBPETokenizer:
    return ByteLevelBPETokenizer(
        str(TOK_DIR / "vocab.json"),
        str(TOK_DIR / "merges.txt"),
    )


if __name__ == "__main__":
    train()
    tok = load()
    ids = tok.encode("The Count stood at the window, watching the mist rise from the churchyard.").ids
    print(ids)
    print(tok.decode(ids))
