import datetime

from cs336_basics.tokenizer.tokenizer_obj import Tokenizer
from cs336_basics.tokenizer import train_utils

if __name__ == '__main__':
    # # Example usage
    input_path = "../../data/TinyStoriesV2-GPT4-train.txt"
    special_tokens = ["<|endoftext|>"]
    docs = train_utils.split_docs_on_special_characters(input_path=input_path,
                                                        special_tokens=special_tokens,
                                                        num_sample_docs=10)
    vocab, merges = train_utils.bpe_tokenizing(input_path=input_path,
                                               special_tokens=special_tokens,
                                               vocab_size=10_000,
                                               num_sample_docs=10)
    tokenizer = Tokenizer(vocab=vocab, merges=merges)

    bytes_lengths = []
    encoding_times = []
    for doc in docs:
        start_time = datetime.datetime.now()
        tokenizer.encode(doc)
        end_time = datetime.datetime.now()
        total_encoding_time = end_time - start_time
        bytes_len = len(doc.encode('utf-8'))
        bytes_lengths.append(bytes_len)
        encoding_times.append(total_encoding_time.total_seconds())
        print('throughput (bytes/sec):', bytes_len / total_encoding_time.total_seconds())

    print('avg bytes length:', sum(bytes_lengths) / len(bytes_lengths))
    print('avg encoding time:', sum(encoding_times) / len(encoding_times))
    print('avg throughput (bytes/sec):', sum(bytes_lengths) / sum(encoding_times))

    #     avg_compression_ratio.append(compression_ratio)
    #     print(compression_ratio)


