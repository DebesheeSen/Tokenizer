import os
import functools
from concurrent.futures import ProcessPoolExecutor

import regex as re
from tqdm.auto import tqdm
from bpe_tokenizer import BPETokenizer

GPT2PAT = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
GPT4PAT = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""


# ---------------------------------------------------------------------------
# Module-level helper functions.
# These MUST live at module scope (not as methods or lambdas) so that
# ProcessPoolExecutor workers can pickle and import them.
# ---------------------------------------------------------------------------

def get_pair_counts(ids):
    """Count adjacent-pair frequencies in a single chunk's id list."""
    counts = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def merge_ids(ids, pair, idx):
    """Replace every occurrence of `pair` in `ids` with the new id `idx`."""
    new_ids = []
    i = 0
    n = len(ids)
    while i < n:
        if i < n - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            new_ids.append(idx)
            i += 2
        else:
            new_ids.append(ids[i])
            i += 1
    return new_ids


def merge_ids_worker(args):
    """Unpack-and-call wrapper so executor.map can pass a single tuple arg."""
    ids, pair, idx = args
    return merge_ids(ids, pair, idx)


def _merge_priority(pair, merges):
    """Priority function used instead of a lambda for min(..., key=...)."""
    return merges.get(pair, float("inf"))


def encode_chunk_worker(args):
    """Encode a single text chunk against a fixed `merges` dict."""
    chunk_str, merges = args
    ids = list(chunk_str.encode("utf-8"))

    priority_fn = functools.partial(_merge_priority, merges=merges)

    while len(ids) >= 2:
        stats = get_pair_counts(ids)
        pair = min(stats, key=priority_fn)

        if pair not in merges:
            break

        ids = merge_ids(ids, pair, merges[pair])

    return ids


# ---------------------------------------------------------------------------
# Original single-process pretokenizing BPE tokenizer.
# ---------------------------------------------------------------------------

class PreTokenizer(BPETokenizer):
    def __init__(self, pattern=GPT4PAT):
        super().__init__()
        self.pattern = re.compile(pattern)

    def _pre_tokenize(self, text):
        return self.pattern.findall(text)

    def train(self, text, vocab_size):
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.history = []
        num_merges = vocab_size - 256
        chunks = self._pre_tokenize(text)
        ids = [list(chunk.encode("utf-8")) for chunk in chunks]

        for i in tqdm(range(num_merges), desc="Training Regex+BPE", unit="merge"):

            pair_counter = {}

            for chunk in ids:
                stats = self.get_counts(chunk)
                for pair, count in stats.items():
                    pair_counter[pair] = pair_counter.get(pair, 0) + count

            if not pair_counter:
                break

            pair = max(pair_counter, key=pair_counter.get)
            idx = 256 + i
            ids = [self.merge(chunk, pair, idx) for chunk in ids]
            self.merges[pair] = idx
            self.vocab[idx] = self.vocab[pair[0]] + self.vocab[pair[1]]

            self.history.append(
                {
                    "pair": pair,
                    "frequency": pair_counter[pair],
                    "id": idx,
                }
            )

    def encode(self, text):
        chunks = self._pre_tokenize(text)
        output = []
        for chunk in chunks:
            ids = list(chunk.encode("utf-8"))
            while len(ids) >= 2:
                stats = self.get_counts(ids)
                pair = min(stats, key=functools.partial(_merge_priority, merges=self.merges))

                if pair not in self.merges:
                    break

                ids = self.merge(ids, pair, self.merges[pair])
            output.extend(ids)
        return output


# ---------------------------------------------------------------------------
# Multi-process version. Parallelizes:
#   - per-chunk pair counting and per-chunk merging during train()
#   - per-chunk encoding during encode()
# across a persistent process pool.
# ---------------------------------------------------------------------------

class ProcessPoolTokenizer(PreTokenizer):
    def __init__(self, pattern=GPT4PAT, num_workers=None, chunksize=1):
        super().__init__(pattern=pattern)
        self.num_workers = num_workers or os.cpu_count()
        self.chunksize = chunksize

    def train(self, text, vocab_size):
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.history = []
        num_merges = vocab_size - 256
        chunks = self._pre_tokenize(text)
        ids = [list(chunk.encode("utf-8")) for chunk in chunks]

        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            for i in tqdm(range(num_merges), desc="Training Regex+BPE (ProcessPool)", unit="merge"):

                pair_counter = {}
                for stats in executor.map(get_pair_counts, ids, chunksize=self.chunksize):
                    for pair, count in stats.items():
                        pair_counter[pair] = pair_counter.get(pair, 0) + count

                if not pair_counter:
                    break

                pair = max(pair_counter, key=pair_counter.get)
                idx = 256 + i

                merge_args = [(chunk, pair, idx) for chunk in ids]
                ids = list(executor.map(merge_ids_worker, merge_args, chunksize=self.chunksize))

                self.merges[pair] = idx
                self.vocab[idx] = self.vocab[pair[0]] + self.vocab[pair[1]]

                self.history.append(
                    {
                        "pair": pair,
                        "frequency": pair_counter[pair],
                        "id": idx,
                    }
                )

    def encode(self, text):
        chunks = self._pre_tokenize(text)
        args = [(chunk, self.merges) for chunk in chunks]

        output = []
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            for ids in executor.map(encode_chunk_worker, args, chunksize=self.chunksize):
                output.extend(ids)

        return output