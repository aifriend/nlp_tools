import torch
from torch.utils.data import Dataset


class DocContent(Dataset):
    MAX_LENGTH = 1024

    def __init__(self, df, tokenizer):
        self.lyrics = []
        for row in df['Lyric']:
            self.lyrics.append(torch.tensor(
                tokenizer.encode(f"<|ns_date|>{row[:self.MAX_LENGTH]}<|endoftext|>")
            ))
        self.lyrics_count = len(self.lyrics)

    def __len__(self):
        return self.lyrics_count

    def __getitem__(self, item):
        return self.lyrics[item]
