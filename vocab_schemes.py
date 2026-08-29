"""Candidate vocab schemes for the Internal model, compared empirically (see plan's
"Vocab sweep"). Each scheme builds a base character alphabet; the "capped_combinatorial"
scheme additionally learns a fixed table of frequent adjacent-character pairs from the
actual training corpus (a single BPE-style merge pass, capped at n_combo entries) so that
every combinatorial token is guaranteed frequent by construction, not just plausible.
"""
import unicodedata
from collections import Counter

ASCII_LO, ASCII_HI = 0x20, 0x7E  # printable ASCII


def ascii_alphabet():
    return [chr(c) for c in range(ASCII_LO, ASCII_HI + 1)]


def curated_unicode_alphabet(target_size=3000):
    """Latin extended + common technical/math/programming symbols, capped at target_size."""
    ranges = [
        (0x0020, 0x007E),  # basic latin
        (0x00A0, 0x024F),  # latin-1 supplement + latin extended A/B
        (0x2010, 0x2027),  # general punctuation (dashes, quotes)
        (0x2030, 0x205E),
        (0x2070, 0x209C),  # super/subscripts
        (0x2100, 0x214F),  # letterlike symbols
        (0x2190, 0x21FF),  # arrows
        (0x2200, 0x22FF),  # mathematical operators
        (0x2300, 0x23FF),  # misc technical
        (0x25A0, 0x25FF),  # geometric shapes
        (0x2600, 0x26FF),  # misc symbols
    ]
    chars = []
    seen = set()
    for lo, hi in ranges:
        for cp in range(lo, hi + 1):
            ch = chr(cp)
            if ch in seen:
                continue
            if unicodedata.category(ch) == "Cn":  # unassigned
                continue
            seen.add(ch)
            chars.append(ch)
            if len(chars) >= target_size:
                return chars
    return chars


def full_unicode_alphabet(max_size=20000):
    """All assigned Unicode code points, up to a practical enumeration cap. This is the
    base alphabet for the 'capped_combinatorial' scheme -- large but bounded, since
    scanning/holding the entire ~150K-code-point space buys nothing extra once combined
    with the corpus-frequency-capped combinatorial layer on top."""
    chars = []
    cp = 0x20
    while len(chars) < max_size and cp <= 0x2FFFF:
        ch = chr(cp)
        if ch.isprintable() and unicodedata.category(ch) != "Cn":
            chars.append(ch)
        cp += 1
    return chars


def build_combo_table(corpus_text, base_alphabet, n_combo=2048):
    """Single BPE-style merge pass: count adjacent-pair frequencies restricted to the base
    alphabet, keep the top n_combo pairs. Ties every combinatorial token to actual corpus
    statistics, so combos are frequent by construction rather than by hope."""
    base_set = set(base_alphabet)
    counts = Counter()
    prev = None
    for ch in corpus_text:
        if ch not in base_set:
            prev = None
            continue
        if prev is not None:
            counts[(prev, ch)] += 1
        prev = ch
    return [pair for pair, _ in counts.most_common(n_combo)]


class InternalVocab:
    """Maps between text and token ids for a given scheme. Token id layout:
    [0 .. len(base_alphabet)-1] = base characters
    [len(base_alphabet) .. +n_combo-1] = combinatorial tokens (scheme 'capped_combinatorial' only)
    Encoding uses greedy longest-match (checks 2-char combos before falling back to single
    base chars) -- this achieves the same effect as an explicit escape-token protocol
    without needing one, since every combo already has its own vocab id the model can pick
    directly via its own softmax, same as any base character.
    """

    UNK = "�"

    def __init__(self, scheme, corpus_text=None, curated_size=3000, full_size=20000, n_combo=2048):
        self.scheme = scheme
        if scheme == "ascii":
            self.base_alphabet = ascii_alphabet()
            self.combo_table = []
        elif scheme == "curated_unicode":
            self.base_alphabet = curated_unicode_alphabet(curated_size)
            self.combo_table = []
        elif scheme == "capped_combinatorial":
            self.base_alphabet = full_unicode_alphabet(full_size)
            assert corpus_text is not None, "capped_combinatorial needs corpus_text to build the combo table"
            self.combo_table = build_combo_table(corpus_text, self.base_alphabet, n_combo)
        else:
            raise ValueError(f"unknown vocab scheme: {scheme}")

        if self.UNK not in self.base_alphabet:
            self.base_alphabet = self.base_alphabet + [self.UNK]
        self.unk_id = self.base_alphabet.index(self.UNK)

        self.id_to_tok = list(self.base_alphabet) + [a + b for a, b in self.combo_table]
        self.tok_to_id = {t: i for i, t in enumerate(self.id_to_tok)}
        self.combo_start = len(self.base_alphabet)
        self.combo_pairs = {(a, b): self.combo_start + i for i, (a, b) in enumerate(self.combo_table)}

    @property
    def vocab_size(self):
        return len(self.id_to_tok)

    def encode(self, text):
        ids = []
        i, n = 0, len(text)
        while i < n:
            if i + 1 < n and (text[i], text[i + 1]) in self.combo_pairs:
                ids.append(self.combo_pairs[(text[i], text[i + 1])])
                i += 2
            else:
                ids.append(self.tok_to_id.get(text[i], self.unk_id))
                i += 1
        return ids

    def decode(self, ids):
        return "".join(self.id_to_tok[i] for i in ids)
