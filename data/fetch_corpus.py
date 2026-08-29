"""Download a small public-domain corpus from Project Gutenberg and strip boilerplate."""
import re
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).parent

# A handful of classic public-domain novels -> a few million tokens of coherent prose.
BOOKS = {
    "dracula": 345,
    "frankenstein": 84,
    "sherlock_holmes": 1661,
    "pride_and_prejudice": 1342,
    "moby_dick": 2701,
    "war_of_the_worlds": 36,
    "picture_of_dorian_gray": 174,
    "jekyll_and_hyde": 43,
}

START_RE = re.compile(r"\*\*\*\s*START OF (THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", re.IGNORECASE | re.DOTALL)
END_RE = re.compile(r"\*\*\*\s*END OF (THE|THIS) PROJECT GUTENBERG EBOOK.*", re.IGNORECASE | re.DOTALL)


def strip_boilerplate(text: str) -> str:
    m = START_RE.search(text)
    if m:
        text = text[m.end():]
    m = END_RE.search(text)
    if m:
        text = text[: m.start()]
    return text.strip()


def main():
    out_dir = HERE / "books"
    out_dir.mkdir(exist_ok=True)
    corpus_parts = []
    for name, book_id in BOOKS.items():
        dest = out_dir / f"{name}.txt"
        if not dest.exists():
            url = f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"
            print(f"downloading {name} from {url}")
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=30) as resp:
                    raw = resp.read().decode("utf-8", errors="ignore")
                dest.write_text(raw, encoding="utf-8")
                time.sleep(0.5)
            except Exception as e:
                print(f"  FAILED: {e}")
                continue
        else:
            raw = dest.read_text(encoding="utf-8", errors="ignore")
        cleaned = strip_boilerplate(raw)
        corpus_parts.append(f"\n\n<|book:{name}|>\n\n" + cleaned)

    corpus = "".join(corpus_parts)
    corpus_path = HERE / "corpus.txt"
    corpus_path.write_text(corpus, encoding="utf-8")
    print(f"corpus written: {corpus_path} ({len(corpus):,} chars, {len(corpus.split()):,} words)")


if __name__ == "__main__":
    main()
