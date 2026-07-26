import re
from itertools import pairwise

class BPETokenizer:
    def __init__(self):
        self.stoi = {}
        self.itos = {}
        self.merges = {}
        self.pattern = r"['’‘]s|['’‘]t|['’‘]re|['’‘]re|['’‘]ve|['’‘]m|['’‘]ll|['’‘]d | ?[^\W\d_]+| ?\d+| ?[^\w\s\d]+|\s+(?!\S)|\s+"

    def train(self, text, target_vocab_size):
        vocab = BPETokenizer._list_of_unique_characters(text) # not updated anymore, only initial
        self.stoi = {token: integer_id for integer_id, token in enumerate(vocab)}
        self.itos = {integer_id: token for integer_id, token in enumerate(vocab)}
        special_tokens = ["<|eos|>"]

        assert target_vocab_size >= len(vocab) + len(special_tokens)
        number_merges_needed = target_vocab_size - len(vocab) - len(special_tokens)
        # token_ids = [self.stoi[char] for char in text]

        preprocessed_text = re.findall(self.pattern, text)
        token_ids = [([self.stoi[char] for char in chunk]) for chunk in preprocessed_text]

        for i in range(number_merges_needed):
            pair_counts = {}
            for chunk in token_ids:
                for i_chunk in range(len(chunk)-1):
                    pair = (chunk[i_chunk],chunk[i_chunk+1])
                    if not pair in pair_counts:
                        pair_counts[pair] = 1
                    else:
                        pair_counts[pair] += 1
            if not pair_counts:
                break
            best_pair = max(pair_counts, key=pair_counts.get)
            new_token_id = len(self.stoi)
            self.stoi[f"{self.itos[best_pair[0]]}{self.itos[best_pair[1]]}"] = new_token_id
            self.itos[new_token_id] = f"{self.itos[best_pair[0]]}{self.itos[best_pair[1]]}"
            self.merges[best_pair] = new_token_id
        
            new_token_ids = []
            for chunk in token_ids:
                new_token_ids.append(BPETokenizer._merge_pair(chunk,best_pair,new_token_id))
            token_ids = new_token_ids

        for st in special_tokens:
            new_id = len(self.stoi)
            self.stoi[st] = new_id
            self.itos[new_id] = st

    def encode(self, text):
        preprocessed_text = re.findall(self.pattern, text)
        token_ids = [([self.stoi[char] for char in chunk]) for chunk in preprocessed_text]

        tokenized_text = []
        for i in range(len(token_ids)):
            chunk = token_ids[i]

            while len(chunk) >= 2:
                # iterate over all adjacent pairs
                best_merge = min(pairwise(chunk), key=lambda p: self.merges.get(p, float("inf")))
                if best_merge not in self.merges:
                    break
                new_chunk = BPETokenizer._merge_pair(chunk,best_merge, self.merges[best_merge])
                chunk = new_chunk
            tokenized_text.extend(chunk)

        return tokenized_text
        

    def decode(self, token_ids):
        text = "".join([self.itos[tok] for tok in token_ids])
        return text

    def _merge_pair(token_ids, pair, new_token_id):
        j=0
        new_chunk = []
        while j < len(token_ids):
            if j < len(token_ids)-1 and (token_ids[j],token_ids[j+1]) == pair:
                new_chunk.append(new_token_id)
                j+=2
            else:
                new_chunk.append(token_ids[j])
                j+=1
        return new_chunk

    def _list_of_unique_characters(text):
        res = set()
        for c in text:
            res.add(c)
        return sorted(list(res))
