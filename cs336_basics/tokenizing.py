import multiprocessing
import regex as re
from collections import Counter, defaultdict

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def init_vocab():
    vocab = {257: "<|endoftext|>".encode('utf-8')}
    byte_map = {i: bytes([i]) for i in range(256)}
    vocab.update(byte_map)
    return vocab


def split_docs_on_special_characters(input_path: str,
                                     special_tokens: list):
    with open(input_path, "r", encoding="utf-8") as f:
        data = f.read()
    split_pattern = "|".join([re.escape(token) for token in special_tokens])
    stories = re.split(split_pattern, data)
    return stories

def pre_tokenizing(story: str) -> tuple[dict[int, int], list[tuple[bytes]]]:
    # Count word frequencies
    word_counter = Counter(match.group(0) for match in re.finditer(PAT, story))
    # Map index to frequency
    frequency_table = {tuple(c.encode('utf-8') for c in word): freq for idx, (word, freq) in enumerate(word_counter.items())}

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
    merged_pairs_dict = merge_consecutive_bytes_pairs(corpus_word_freq, word_list)

    frequent_merges = []
    for i in range(n_iterations):
        # Find the most frequent pair
        most_freq_pair, stats = max(merged_pairs_dict.items(), key=sort_by_count_and_lex)
        frequent_merges.append(most_freq_pair)
        merged_pairs_dict.pop(most_freq_pair)
        print(most_freq_pair, stats['c'])

        most_freq_pair_word_indicies = stats['w'] # stats['w'] contains indices of words where most_freq_pair appears
        # Update the word list at affected words, to new word structure (merged on the msot frequent pair)
        for idx in most_freq_pair_word_indicies:
            merged_word = merge_word(word_tuple=word_list[idx], merged_pair=most_freq_pair)
            word_list[idx] = merged_word

        # work only with the frequencies of the affected words (runtime optimization)
        merged_words_freq = {idx: corpus_word_freq[idx] for idx in most_freq_pair_word_indicies}

        # Update pairs_dict for next iteration
        new_merged_pairs = merge_consecutive_bytes_pairs(frequency_table=merged_words_freq,
                                                          word_list=word_list,
                                                          merged_pair=most_freq_pair)
        merged_pairs_dict.update(new_merged_pairs)

    return frequent_merges

def bpe_tokenizing(input_path: str,
                   vocab_size: int,
                   special_tokens: list[str]):

    vocab = init_vocab()
    docs = split_docs_on_special_characters(input_path=input_path,
                                            special_tokens=special_tokens)

    # pre-tokenizing each document to get word frequencies
    doc_word_freq_dicts = []
    for doc in docs[:3]:# TODO: multi-processing on all stories
        doc_word_freq_dicts.append(pre_tokenizing(doc))

    # Get the corpus word frequencies
    corpus_word_freq = get_corpus_word_freq(doc_word_freq_dicts)
    word_list = list(corpus_word_freq.keys()) # keep words in a list
    # replace byte_tuples keys in corpus_word_freq with their index in the word_list (optimized memory usage)
    corpus_word_freq = {word_list.index(word): freq for word, freq in corpus_word_freq.items()}

    # apply merges on the corpus
    n_iterations = vocab_size - len(vocab)
    merges = merge(corpus_word_freq=corpus_word_freq,
                   word_list=word_list,
                   n_iterations=n_iterations)
    # update the vocabulary with the merges
    max_token = max(vocab.keys())
    for bytes_tuple in merges:
        token_bytes = b''.join(bytes_tuple)
        max_token+= 1
        vocab[max_token] = token_bytes

    return merges, vocab





# Sort by count descending, then tuple descending (for lexicographical order)
def sort_by_count_and_lex(item):
    count = item[1]['c']
    pair = item[0]
    return count, pair

def merge_consecutive_bytes_pairs(frequency_table: dict[int, int],
                                  word_list: list[tuple[bytes]],
                                  merged_pair: tuple[bytes, bytes]=None) -> dict:
    """
    Counts consecutive byte pairs in words from the word list, summing their frequencies.
    If `merged_pair` is provided, only pairs matching this tuple are counted.
    Returns a dictionary mapping pairs to their count and set of word indices.
    """

    merged_pair_bytes = b''.join(merged_pair) if merged_pair else None
    pairs_dict = defaultdict(lambda: {'c': 0, 'w': set()})
    for idx, freq in frequency_table.items():
        byte_tuple = word_list[idx]
        for i in range(len(byte_tuple) - 1):
            pair = (byte_tuple[i], byte_tuple[i + 1])
            if merged_pair_bytes is None:
                pairs_dict[pair]['c'] += freq
                pairs_dict[pair]['w'].add(idx)
            else:
                if merged_pair_bytes in pair:
                    pairs_dict[pair]['c'] += freq
                    pairs_dict[pair]['w'].add(idx)

    return pairs_dict



def merge_word(word_tuple: tuple[bytes], merged_pair: tuple[bytes]) -> tuple[bytes]:
    """
    Merges consecutive occurrences of `seek_bytes` in `word_tuple` into a single bytes object.
    """
    merged = []
    i = 0
    while i < len(word_tuple):
        if i < len(word_tuple) - 1 and (word_tuple[i], word_tuple[i + 1]) == merged_pair:
            merged.append(word_tuple[i] + word_tuple[i + 1])
            i += 2
        else:
            merged.append(word_tuple[i])
            i += 1
    return tuple(merged)




if __name__ == "__main__":
    # Example usage
    input_path = "../data/TinyStoriesV2-GPT4-train.txt"
    special_tokens = ["<|endoftext|>"]
    bpe_tokenizing(input_path=input_path,
                   special_tokens=special_tokens,
                   vocab_size=300)

