from typing import Iterable, Iterator, List
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
        # protected = r"(?!x)x"  # matches nothing

        # if special_tokens is not None:
        #     special_tokens = sorted(special_tokens, key=len, reverse=True)
        #     protected = "|".join(re.escape(t) for t in special_tokens)

        # split_pattern = "(" + "|".join([re.escape(token) for token in self.special_tokens]) + ")"
        """Encode an input text into a sequence of token IDs"""
        self.PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        # 3) One-pass matcher: special tokens OR core pieces
        # self.RX = re.compile(rf"(?P<tok>{protected})|(?P<core>{self.PAT})", re.V1)

        # Merge ranks for O(1) pair lookup
        # merges is a list/iterable of tuples of bytes: [(b'a', b'b'), ...]
        self.merge_ranks = {pair: i for i, pair in enumerate(merges)}


    def get_new_candidates(self, word, merged_pair, start_index) -> List[dict]:
        new_candidates = {}
        if start_index > 0:
            new_candidate = (word[start_index - 1], merged_pair)
            if new_candidate in self.merge_ranks:
                new_candidates.update({new_candidate: self.merge_ranks[new_candidate]})
        if start_index + 2 < len(word):
            new_candidate = (merged_pair, word[start_index + 2])
            if new_candidate in self.merge_ranks:
                new_candidates.update({new_candidate: self.merge_ranks[new_candidate]})

        return new_candidates

    def merge_pair_word(self, word, pair_to_merge):
        new_candidates = {}
        new_word = []
        i = 0
        merged_pair = pair_to_merge[0] + pair_to_merge[1]
        while i < len(word) - 1:
            if word[i] == pair_to_merge[0] and word[i + 1] == pair_to_merge[1]:
                new_word.append(merged_pair)
                new_candidates.update(self.get_new_candidates(word=word, merged_pair=merged_pair, start_index=i))
                i += 2
            else:
                new_word.extend(word[i: i+1])
                i += 1
        if i == len(word) - 1:
            new_word.extend(word[i: i+1])

        return new_word, new_candidates


    def get_initial_candidates(self, word):
        candidates = {}
        for i in range(len(word) - 1):
            candidate = (word[i: i+1], word[i+1: i+2])
            if candidate in self.merge_ranks:
                candidates.update({candidate: self.merge_ranks[candidate]})
        return candidates

    def encode(self, text: str) -> list[int]:
        # split the text on special tokens
        if self.special_tokens is not None:
            sorted_tokens = sorted(self.special_tokens, key=len, reverse=True)
            split_pattern = f"({'|'.join([re.escape(token) for token in sorted_tokens])})"
            splits = re.split(split_pattern, text)
        else:
            splits = [text]
        tokens = []

        for split in splits:
            if self.special_tokens is not None and split in self.special_tokens:
                tokens.append(self.reversed_vocab[split.encode("utf8")])
            else:
                for match in self.PAT.finditer(split):
                    sub = match.group(0)
                    word_bytes = sub.encode('utf-8')
                    candidates = self.get_initial_candidates(word=word_bytes)
                    word_bytes = [word_bytes[i:i+1] for i in range(len(word_bytes))]
                    while len(candidates) > 0:
                        earliest_pair = min(candidates, key=candidates.get)
                        candidates.pop(earliest_pair)
                        word_bytes, new_candidates = self.merge_pair_word(word=word_bytes, pair_to_merge=earliest_pair)
                        candidates.update(new_candidates)
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
        bytes_seq = b''.join([self.vocab[i] for i in ids])
        return bytes_seq.decode("utf8", errors='replace')

    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens=None):
        """method that constructs and return a Tokenizer from a serialized vocabulary and list of merges
        (in the same format that your BPE training code output) and (optionally) a list of special tokens."""
        raise NotImplementedError()

#
if __name__ == "__main__":
    tokenizer = Tokenizer(vocab={0: b' ', 1: b'a', 2: b'c', 3: b'e', 4: b'h', 5: b't', 6: b'th', 7: b' c', 8: b' a', 9: b'the', 10: b' at'},
                          merges=[(b't', b'h'), (b' ', b'c'), (b' ', b'a'), (b'th', b'e'), (b' a', b't')])
    tokenizer.encode('the cat ate')

