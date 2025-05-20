import torch
from torch.utils.data import Dataset
from collections import defaultdict
import pandas as pd

class TransliterationDataset(Dataset):
    def __init__(self, pairs, input_vocab, output_vocab):
        self.pairs = pairs
        self.input_vocab = input_vocab
        self.output_vocab = output_vocab
        self.sos = output_vocab['<sos>']
        self.eos = output_vocab['<eos>']

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        source, target = self.pairs[idx]
        input_ids = [self.input_vocab[c] for c in source]
        target_ids = [self.sos] + [self.output_vocab[c] for c in target] + [self.eos]
        return torch.tensor(input_ids), torch.tensor(target_ids)

def build_vocab(pairs):
    input_chars = set()
    output_chars = set()
    for source, target in pairs:
        if isinstance(source, str) and isinstance(target, str):
            input_chars.update(source)
            output_chars.update(target)

    input_vocab = {c: i + 1 for i, c in enumerate(sorted(input_chars))}
    input_vocab['<pad>'] = 0
    output_vocab = {c: i + 3 for i, c in enumerate(sorted(output_chars))}
    output_vocab.update({'<pad>': 0, '<sos>': 1, '<eos>': 2})

    return input_vocab, output_vocab

def collate_fn(batch):
    inputs, targets = zip(*batch)
    input_lens = [len(seq) for seq in inputs]
    target_lens = [len(seq) for seq in targets]

    inputs_padded = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=0)
    targets_padded = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=0)

    return inputs_padded, targets_padded, input_lens, target_lens

def load_pairs(path):
    df = pd.read_csv(path, sep="\t", header=None, names=["target", "source", "count"], dtype=str)
    df.dropna(subset=["source", "target"], inplace=True)
    return list(zip(df["source"], df["target"]))
