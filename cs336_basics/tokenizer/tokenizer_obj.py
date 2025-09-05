from typing import Iterable, Iterator
import regex as re

class Tokenizer:
    def __init__(self,
                 vocab: dict[int, bytes],
                 merges: list[tuple[bytes, bytes]],
                 special_tokens: list[str] | None = None):
        """Construct a tokenizer from a given vocabulary,
        list of merges, and (optionally) a list of special tokens"""
        self.vocab = vocab
        self.reversed_vocab = {v: k for k, v in self.vocab.items()}
        self.merges = merges
        self.special_tokens = special_tokens
        special_tokens = sorted(self.special_tokens, key=len, reverse=True)
        protected = "|".join(re.escape(t) for t in special_tokens)

        # split_pattern = "(" + "|".join([re.escape(token) for token in self.special_tokens]) + ")"
        """Encode an input text into a sequence of token IDs"""
        self.PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        # 3) One-pass matcher: special tokens OR core pieces
        self.RX = re.compile(rf"(?P<tok>{protected})|(?P<core>{self.PAT})", re.V1)

        # Merge ranks for O(1) pair lookup
        # merges is a list/iterable of tuples of bytes: [(b'a', b'b'), ...]
        self.merge_ranks = {pair: i for i, pair in enumerate(merges)}



    def encode(self, text: str) -> list[int]:

        tokens = []
        for match  in self.RX.finditer(text):
            special_word = match .group("tok")
            if special_word is not None:
                tokens.extend([self.reversed_vocab[special_word.encode("utf8")]])
            else:
                # Core piece → run your usual pre-tokenization logic
                piece = match.group("core")
                # if you still want to re-apply PAT inside each piece:
                for m in re.finditer(self.PAT, piece, re.V1):
                    sub = m.group(0)
                    word_bytes = sub.encode('utf-8')
                    word_bytes = [word_bytes[i:i+1] for i in range(len(word_bytes))]
                    for merge in self.merges:
                        merged_word = []
                        i = 0
                        while i < len(word_bytes):
                            if i < len(word_bytes) - 1 and merge[0] == word_bytes[i] and merge[1] == word_bytes[i + 1]:
                                merged_word.append(word_bytes[i] + word_bytes[i + 1])
                                i += 2
                            else:
                                merged_word.append(word_bytes[i])
                                i += 1
                        word_bytes = merged_word
                        if len(word_bytes) == 1:
                            break
                    tokens.extend([self.reversed_vocab[k] for k in word_bytes])
        return tokens



    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """Given an iterable of strings (e.g., a Python file handle), return a generator that lazily yields token IDs. This is
        required for memory-efficient tokenization of large files that we cannot directly load into memory"""

        for text in iterable: # iterate over each line in the file
            for tid in self.encode(text): # encode each line into token ids
                yield tid # yield each token id one by one

    def decode(self, ids: list[int]) -> str:
        """Decode a sequence of token IDs into text"""
        raise NotImplementedError()

    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens=None):
        """method that constructs and return a Tokenizer from a serialized vocabulary and list of merges
        (in the same format that your BPE training code output) and (optionally) a list of special tokens."""
        raise NotImplementedError()

#
if __name__ == "__main__":
    tokenizer = Tokenizer(vocab={0: b' ', 1: b'a', 2: b'c', 3: b'e', 4: b'h', 5: b't', 6: b'th', 7: b' c', 8: b' a', 9: b'the', 10: b' at'},
                          merges=[(b't', b'h'), (b' ', b'c'), (b' ', b'a'), (b'th', b'e'), (b' a', b't')])
    tokenizer.encode('the cat ate')

