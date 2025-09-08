import datetime
import multiprocessing
import regex as re
from collections import Counter, defaultdict
import pickle
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


from functools import lru_cache

@lru_cache(maxsize=None)
def _inv_pad(b: bytes) -> bytes:
    """Invert bytes and pad with 0xFF to fix prefix ordering."""
    return bytes(255 - x for x in b) + b'\xff'

def build_heap(pairs_dict):
    """
    Heap item: (-count, inv_pad(a), inv_pad(b), original_pair)
    → pop gives: max count, then pair DESC (tuple-lex on (a, b)).
    """
    heap = [(-s['c'], _inv_pad(p[0]), _inv_pad(p[1]), p)
            for p, s in pairs_dict.items() if s['c'] > 0]
    heapq.heapify(heap)
    return heap

def push_to_heap(heap, count, pair):
    heapq.heappush(heap, (-count, _inv_pad(pair[0]), _inv_pad(pair[1]), pair))

def pop_from_heap(heap):
    negc, _ka, _kb, pair = heapq.heappop(heap)
    return -negc, pair

def init_vocab():
    vocab = {256: "<|endoftext|>".encode('utf-8')}
    byte_map = {i: bytes([i]) for i in range(256)}
    vocab.update(byte_map)
    return vocab


def split_docs_on_special_characters(input_path: str,
                                     special_tokens: list,
                                     num_sample_docs: int = None) -> list[str]:

    with open(input_path, "r", encoding="utf-8") as f:
        data = f.read()
    split_pattern = "|".join([re.escape(token) for token in special_tokens])
    stories = re.split(split_pattern, data)
    if num_sample_docs is None:
        num_sample_docs = len(stories)
    stories = [s for s in stories if s.strip()][:num_sample_docs]
    return stories

def pre_tokenizing(text: str) -> tuple[dict[int, int], list[tuple[bytes]]]:
    # Count word frequencies
    word_counter = Counter(match.group(0) for match in re.finditer(PAT, text))
    # Map index to frequency
    # frequency_table = {tuple(c.encode('utf-8') for c in word): freq for idx, (word, freq) in enumerate(word_counter.items())}
    frequency_table =  {
        tuple(bytes([b]) for b in w.encode("utf-8")): freq
        for w, freq in word_counter.items()
    }

    return frequency_table

def get_corpus_word_freq(doc_word_freq_dicts):
    corpus_word_freq = defaultdict(int)
    for doc_word_freq in doc_word_freq_dicts:
        for word, freq in doc_word_freq.items():
            corpus_word_freq[word] += freq

    return corpus_word_freq


def merge(corpus_word_freq: dict[int, int], word_list: list[tuple[bytes]], n_iterations: int) ->  list[tuple[bytes, bytes]]:
    """
    Iteratively merges the most frequent consecutive byte pairs in the word list.
    Returns a list of merged byte pairs.
    """
    pairs_dict = merge_consecutive_bytes_pairs(corpus_word_freq, word_list)
    # Convert pairs_dict to a heap
    heap = build_heap(pairs_dict)


    frequent_merges = []
    for i in range(n_iterations):
        while heap:
            count, most_freq_pair = pop_from_heap(heap)
            # Check if the count is up-to-date
            if most_freq_pair in pairs_dict and pairs_dict[most_freq_pair]['c'] == count:
                break

        if not heap:
            break

        # Find the most frequent pair
        # most_freq_pair, stats = max(pairs_dict.items(), key=sort_by_count_and_lex)
        frequent_merges.append(most_freq_pair)
        most_freq_pair_word_indicies = pairs_dict[most_freq_pair]['w']  # contains indices of words where most_freq_pair appears
        pairs_dict.pop(most_freq_pair)



        for idx in most_freq_pair_word_indicies: # iterate over all the words containing the most frequent pair
            word_tuple = word_list[idx]
            word_freq = corpus_word_freq[idx]
            # merge the most frequent pair into a single bytes object in the word tuple, update pairs_dict accordingly
            merged_word = merge_and_update_pairs_dict(word_tuple=word_tuple,
                                                      most_freq_pair=most_freq_pair,
                                                      pairs_dict=pairs_dict,
                                                      word_freq=word_freq,
                                                      word_idx=idx,
                                                      heap=heap)
            word_list[idx] = merged_word

    return frequent_merges



# Sort by count descending, then tuple descending (for lexicographical order)
def sort_by_count_and_lex(item):
    count = item[1]['c']
    pair = item[0]
    return count, pair

import heapq
def merge_consecutive_bytes_pairs(frequency_table: dict[int, int],
                                  word_list: list[tuple[bytes]]) -> dict:
    """
    Counts consecutive byte pairs in words from the word list, summing their frequencies.
    """

    pairs_dict = defaultdict(lambda: {'c': 0, 'w': set()})
    for idx, freq in frequency_table.items():
        byte_tuple = word_list[idx]
        for i in range(len(byte_tuple) - 1):
            pair = (byte_tuple[i], byte_tuple[i + 1])
            pairs_dict[pair]['c'] += freq
            pairs_dict[pair]['w'].add(idx)




    return pairs_dict



def merge_and_update_pairs_dict(word_tuple: tuple[bytes],
                                most_freq_pair: tuple[bytes],
                                pairs_dict: dict,
                                word_freq:int,
                                word_idx: int,
                                heap) -> tuple[bytes]:
    """
       Merges the most frequent byte pair in a word tuple into a single element, updates the pairs dictionary,
    and adjusts counts for affected pairs.

    This function replaces consecutive occurrences of the most frequent byte pair in the word tuple with a
    single bytes object. It updates the pairs dictionary by:
    - Adding new pairs formed by the merged element and its neighbors.
    - Removing or decrementing counts for pairs that are no longer valid due to the merge.
    """
    merged_pair = b''.join(most_freq_pair)
    merged_word = []
    i = 0
    updated_pairs = set()
    while i < len(word_tuple):
        if i < len(word_tuple) - 1 and (word_tuple[i], word_tuple[i + 1]) == most_freq_pair:
            # append as a pair
            merged_word.append(merged_pair)
            new_right_pair = (merged_pair, word_tuple[i + 2])  if i + 2 < len(word_tuple) else None
            if new_right_pair:
                pairs_dict[new_right_pair]['w'].add(word_idx)
                pairs_dict[new_right_pair]['c'] += word_freq
                updated_pairs.add(new_right_pair)
            new_left_pair = (word_tuple[i - 1], merged_pair) if i > 0 else None
            if new_left_pair:
                pairs_dict[new_left_pair]['w'].add(word_idx)
                pairs_dict[new_left_pair]['c'] += word_freq
                updated_pairs.add(new_left_pair)

            # update merged_pairs_dict to remove pairs that are no longer valid
            left_pair = (word_tuple[i - 1], word_tuple[i]) if i > 0 else None
            if left_pair:
                pairs_dict[left_pair]['c'] -= word_freq
                updated_pairs.add(left_pair)
            right_pair = (word_tuple[i + 1], word_tuple[i + 2]) if i + 2 < len(word_tuple) else None
            if right_pair:
                pairs_dict[right_pair]['c'] -= word_freq
                updated_pairs.add(right_pair)
            # skip the next byte as it's merged, move to the byte after next
            i += 2
        else:
            merged_word.append(word_tuple[i])
            i += 1

    for pair in updated_pairs:
        push_to_heap(heap, pairs_dict[pair]['c'], pair)
    return tuple(merged_word)


def bpe_tokenizing(input_path: str,
                   vocab_size: int,
                   special_tokens: list[str],
                   num_sample_docs: int = None,
                   save_to_disk: bool = False) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:

    vocab = init_vocab()
    docs = split_docs_on_special_characters(input_path=input_path,
                                            special_tokens=special_tokens,
                                            num_sample_docs=num_sample_docs)
    num_docs = len(docs)
    print(f'number of documents: {num_docs}')
    s = datetime.datetime.now()
    # pre-tokenizing each document to get word frequencies
    doc_word_freq_dicts = []
    with multiprocessing.Pool(processes=8) as pool:
        doc_word_freq_dicts = pool.map(pre_tokenizing, docs)
    print('pre-tokenizing took:', datetime.datetime.now() - s)

    s = datetime.datetime.now()
    # Get the corpus word frequencies
    corpus_word_freq = get_corpus_word_freq(doc_word_freq_dicts)
    word_list = list(corpus_word_freq.keys()) # keep words in a list
    # replace byte_tuples keys in corpus_word_freq with their index in the word_list (optimized memory usage)
    word_to_idx = {w: i for i, w in enumerate(word_list)}
    corpus_word_freq = {word_to_idx[w]: f for w, f in corpus_word_freq.items()}
    print('getting corpus word frequencies took:', datetime.datetime.now() - s)

    s = datetime.datetime.now()
    # apply merges on the corpus
    n_iterations = vocab_size - len(vocab)
    merges = merge(corpus_word_freq=corpus_word_freq,
                   word_list=word_list,
                   n_iterations=n_iterations)
    print('merging took:', datetime.datetime.now() - s)

    s = datetime.datetime.now()
    # update the vocabulary with the merges
    max_token = max(vocab.keys())
    for bytes_tuple in merges:
        token_bytes = b''.join(bytes_tuple)
        max_token+= 1
        vocab[max_token] = token_bytes
    print('after merge took:', datetime.datetime.now() - s)

    if save_to_disk:
        # Save vocab and merges to disk
        with open(f"open_web_vocab_vocab_size_{vocab_size}_num_docs_{num_docs}.pkl", "wb") as vocab_file:
            pickle.dump(vocab, vocab_file)

        with open(f"open_web_merges_vocab_size_{vocab_size}_num_docs_{num_docs}.pkl", "wb") as merges_file:
            pickle.dump(merges, merges_file)

    return vocab, merges
#
if __name__ == "__main__":
    # Example usage
    input_path = "../../data/TinyStoriesV2-GPT4-train.txt"
    special_tokens = ["<|endoftext|>"]
    s = datetime.datetime.now()
    vocab, merges = bpe_tokenizing(input_path=input_path,
                                   special_tokens=special_tokens,
                                   vocab_size=10_000,
                                   num_sample_docs=100000,
                                   save_to_disk=True)
    print('total time taken:', datetime.datetime.now() - s)
    # a = 1
    # pairs_dict = {
    #     (b' a', b'nd'): {'c': 609, 'w': {0, 1, 2}},
    #     (b' ', b'd'): {'c': 609, 'w': {0, 1, 2}},
    #     (b't', b'h'): {'c': 400, 'w': {0, 1, 2}},
    #     (b' c', b'om'): {'c': 400, 'w': {0, 1, 2}},
    #     (b' a', b'b'): {'c': 300, 'w': {2}},
    # }
    # heap = build_heap(pairs_dict)
    # while heap:
    #     a,b = pop_from_heap(heap)
    #     print(a,b)

    # p1 = ((609, (b' a', b'nd'))),  (609, (b' ', b'd'))

