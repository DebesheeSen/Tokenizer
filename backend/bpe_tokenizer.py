import pickle 
from tqdm.auto import tqdm

class BPETokenizer:
    def __init__(self):
        self.merges = {}
        self.vocab = {}

    def get_counts(self, ids):
        counts = {}
        for pair in zip(ids, ids[1:]):
            counts[pair] = counts.get(pair, 0) + 1
        return counts

    def merge(self, ids, pair, idx): # ids : list of ids, pair: pair to be merged, idx : replace with this
        new_ids = []
        i=0
        while i < len(ids):
            if (i < len(ids)-1 and ids[i] == pair[0] and ids[i+1] == pair[1]):
                new_ids.append(idx)
                i += 2
            else:
                new_ids.append(ids[i])
                i += 1
                
        return new_ids
    
    def decode(self, ids):
        tokens = b"".join(self.vocab[idx] for idx in ids)
        text = tokens.decode("utf-8", errors="replace")
        return text

    def encode(self, text):
        tokens = list(text.encode('utf-8'))
        while len(tokens) >= 2:
            stats = self.get_counts(tokens)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            idx = self.merges[pair]
            tokens = self.merge(tokens, pair, idx)
        return tokens
    
    def train(self, text, vocab_size):
        ids = list(text.encode("utf-8"))
        self.vocab = {i: bytes([i]) for i in range(256)}
        num_merges = vocab_size - 256
        self.history = []

        for i in tqdm(range(num_merges), desc="Training BPE Tokenizer", unit="merge"):
            stats = self.get_counts(ids)
            if not stats:
                break

            pair = max(stats, key=stats.get)
            idx = 256 + i
            ids = self.merge(ids, pair, idx)
            self.merges[pair] = idx
            self.vocab[idx] = self.vocab[pair[0]] + self.vocab[pair[1]]
            self.history.append({
                    "pair": pair,
                    "frequency": stats[pair],
                    "id": idx,
                })

    def save(self, filename):
        with open(filename, "wb") as f:
            pickle.dump(
                {
                    "merges": self.merges,
                    "vocab": self.vocab,
                },
                f,
            )

    def load(self, filename):
        with open(filename, "rb") as f:
            data = pickle.load(f)

        self.merges = data["merges"]
        self.vocab = data["vocab"]


    def show_vocab(self, n=20):
        for idx in sorted(self.vocab)[:n]:
            print(idx, self.vocab[idx])


    def compression_ratio(self, text):
        chars = len(text)
        tokens = len(self.encode(text))
        return chars / tokens

    
    def token_frequency(self, text):
        ids = self.encode(text)
        counts = {}

        for i in ids:
            counts[i] = counts.get(i, 0) + 1
        return counts