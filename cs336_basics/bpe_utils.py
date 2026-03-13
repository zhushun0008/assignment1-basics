import regex as re
from copy import deepcopy
import pickle
import os 
from typing import BinaryIO
from multiprocessing import Pool
import time
def init_vocab(special_tokens):
    # 1. init vocab with bytes
    # 1.1. add special token
    # special_tokens = ['<|endoftext|>)']
    vocab = dict()
    for s in special_tokens:
        s_bts = s.encode("utf-8")
        vocab[len(vocab)] = s_bts
        # print(s_bts)
    # add the 256 byte value
    for i in range(256):
        # vocab[len(vocab)] = chr(i).encode("utf-8")
        vocab[len(vocab)] = bytes([i])
    
    return vocab


def str_to_bts_tuple(input_str: str, encoding: str = "utf-8"):
    return tuple(bytes([b]) for b in input_str.encode(encoding))

def pre_tokenize(input_path, special_tokens, **kwargs):
    # wc_dict = {}
    wc_bytes_dict = {}
    
    # input_path = None
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    # PAT = r"""\S+"""

    # input_text = None

    with open(input_path, 'r', encoding='utf-8') as f:
        input_text = f.read()

    # input_text = "low low low low low lower lower widest widest widest newest newest newest newest newest newest"
    doc_delimit = "|".join([re.escape(s) for s in special_tokens])
    doc_list = re.split(doc_delimit, input_text)
    
    for tmp_doc in doc_list:
        # raw_data = input_text.split()
        # raw_data = re.findall(PAT, tmp_doc)
        raw_data = re.finditer(PAT, tmp_doc)
        for m in raw_data:
            # w = m
            w = m.group()
            w_bts = str_to_bts_tuple(w)
            # w_bts = tuple(w.encode('utf-8'))
            # wc_dict[tuple(w)] = wc_dict.get(tuple(w), 0) + 1
            # print(w_bts, wc_bytes_dict.get(w_bts, 0))
            wc_bytes_dict[w_bts] = wc_bytes_dict.get(w_bts, 0) + 1
    
    return wc_bytes_dict


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def pre_tokenize_per_chunk(chunk, special_tokens_pat):
    
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    # PAT = r"""\S+"""
    sub_chunk_data_list = re.split(special_tokens_pat, chunk)
    wc_bytes_dict = {}
    
    for tmp_doc in sub_chunk_data_list:
        raw_data = re.finditer(PAT, tmp_doc)
        for m in raw_data:
            w = m.group()
            w_bts = str_to_bts_tuple(w)
            wc_bytes_dict[w_bts] = wc_bytes_dict.get(w_bts, 0) + 1
    
    return wc_bytes_dict


def parallel_pre_tokenize(input_path, special_tokens, **kwargs):
  
    num_workers = max(1, min(os.cpu_count() - 1, 8))
    special_tokens_pat = "|".join([re.escape(s) for s in special_tokens])
            
    with open(input_path, "rb") as f:
        t0 = time.time()
        boundaries = find_chunk_boundaries(f, num_workers, b"<|endoftext|>")
        print(f"find_chunk_boundaries: {time.time() - t0:.2f} s")
        
        async_results = []
        t1 = time.time()
        with Pool(num_workers) as pool:
            # The following is a serial implementation, but you can parallelize this
            # by sending each start/end pair to a set of processes.
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                f.seek(start)
                chunk = f.read(end - start).decode("utf-8", errors="ignore")
                async_results.append(pool.apply_async(pre_tokenize_per_chunk, args = (chunk, special_tokens_pat)))
            
            print(f"submit tasks: {time.time() - t1:.2f}s")

            t2 = time.time()
            
            results = [ar.get() for ar in async_results]
            print(f"get results: {time.time() - t2:.2f}s")
        
    
    merge_dict = {}
    t3 = time.time()
    for d in results:
        for k,v in d.items():
            merge_dict[k] = merge_dict.get(k, 0) + v
    
    print(f"merge: {time.time() - t3:.2f}s")

    return merge_dict

def bpe_merge(vocab, wc_bytes_dict, vocab_size, max_merge_runs=None):
    merges = []
    max_merge_runs = None
    cur_run = 0
    # vocab_size = 500
    while len(vocab) < vocab_size and (max_merge_runs is None or cur_run < max_merge_runs):

        pair_cnt_dict = {}
        max_freq = 0
        # max_pair = None
        # count the frequency of each pair
        # find the most frequent pair
        for w_bts, cnt in wc_bytes_dict.items():

            # print(w_bts)
            # w = w_bts.decode('utf-8')
            # print(w)
            for i in range(len(w_bts) - 1):
                pair = w_bts[i : i + 2]
                pair_cnt_dict[pair] = pair_cnt_dict.get(pair, 0) + cnt
                if pair_cnt_dict[pair] > max_freq:
                    max_freq = pair_cnt_dict[pair]
                    # max_pair = pair
        # take the lexicographically greater pair
        max_freq_pair_list = []
        for pair, cnt in pair_cnt_dict.items():
            if cnt == max_freq:
                max_freq_pair_list.append(pair)

        max_pair = max(max_freq_pair_list)
        merges.append(max_pair)
        vocab[len(vocab)] = b"".join(max_pair)

        new_wc_bytes_dict = dict()
        for w_bts, cnt in wc_bytes_dict.items():
            i = 0
            merged_w_bts_list = []
            # handle one pre-token
            while i < len(w_bts):
                # last char
                if i == len(w_bts) - 1:
                    merged_w_bts_list.append(w_bts[i])
                    break

                if w_bts[i : i + 2] == max_pair:
                    merged_w_bts_list.append(b"".join(w_bts[i : i + 2]))
                    i += 2
                else:
                    merged_w_bts_list.append(w_bts[i])
                    i += 1
            new_wc_bytes_dict[tuple(merged_w_bts_list)] = cnt

        wc_bytes_dict = deepcopy(new_wc_bytes_dict)
        cur_run += 1
        # print(
        #     f"run:{cur_run}, max_pair:{max_pair}, max_freq: {max_freq}, wc_bytes_dict: {wc_bytes_dict}, vocab:{vocab}"
        # )
    return (vocab, merges)


def update_pair_cnt_and_get_max_cnt(wc_bytes_dict, pair_cnt_dict):

    max_freq = 0
    for w_bts, cnt in wc_bytes_dict.items():
      # print(w_bts)
      # w = w_bts.decode('utf-8')
      # print(w)
      for i in range(len(w_bts) - 1):
          pair = w_bts[i : i + 2]
          pair_cnt_dict[pair] = pair_cnt_dict.get(pair, 0) + cnt
          if pair_cnt_dict[pair] > max_freq:
              max_freq = pair_cnt_dict[pair]

    return max_freq

def serialing_data(file_name, data):
    with open(file_name, 'wb') as f:
        pickle.dump(data, f)
        
def deserialing_data(file_name):
    obj = None
    with open(file_name, 'rb') as f:
        obj = pickle.load(f)
    return obj

def opt_bpe_merge(vocab, wc_bytes_dict, vocab_size, max_merge_runs=None):
    merges = []
    max_merge_runs = None
    cur_run = 0
    # vocab_size = 500
    while len(vocab) < vocab_size and (max_merge_runs is None or cur_run < max_merge_runs):

        pair_cnt_dict = {}
        max_freq = 0
        # max_pair = None
        # count the frequency of each pair
        # find the most frequent pair
        for w_bts, cnt in wc_bytes_dict.items():

            # print(w_bts)
            # w = w_bts.decode('utf-8')
            # print(w)
            for i in range(len(w_bts) - 1):
                pair = w_bts[i : i + 2]
                pair_cnt_dict[pair] = pair_cnt_dict.get(pair, 0) + cnt
                if pair_cnt_dict[pair] > max_freq:
                    max_freq = pair_cnt_dict[pair]
                    # max_pair = pair
        # take the lexicographically greater pair
        
        
        max_freq_pair_list = []
        for pair, cnt in pair_cnt_dict.items():
            if cnt == max_freq:
                max_freq_pair_list.append(pair)

        max_pair = max(max_freq_pair_list)
        merges.append(max_pair)
        vocab[len(vocab)] = b"".join(max_pair)

        new_wc_bytes_dict = dict()
        for w_bts, cnt in wc_bytes_dict.items():
            i = 0
            merged_w_bts_list = []
            # handle one pre-token
            while i < len(w_bts):
                # last char
                if i == len(w_bts) - 1:
                    merged_w_bts_list.append(w_bts[i])
                    break

                if w_bts[i : i + 2] == max_pair:
                    merged_w_bts_list.append(b"".join(w_bts[i : i + 2]))
                    i += 2
                else:
                    merged_w_bts_list.append(w_bts[i])
                    i += 1
            new_wc_bytes_dict[tuple(merged_w_bts_list)] = cnt

        wc_bytes_dict = new_wc_bytes_dict
        cur_run += 1
        # print(
        #     f"run:{cur_run}, max_pair:{max_pair}, max_freq: {max_freq}, wc_bytes_dict: {wc_bytes_dict}, vocab:{vocab}"
        # )
    return (vocab, merges)

