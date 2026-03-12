import regex as re


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

