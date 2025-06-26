import torch
from torch.utils.data import Dataset
import json
import pickle
import numpy as np
from utils.vocab import Vocabulary

class CaptionDataset(Dataset):
    def __init__(self, captions_file, features_file, vocab_file, transform=None):
        with open(captions_file, 'r') as f:
            self.captions_data = json.load(f)

        with open(features_file, 'r') as f:
            self.image_features = json.load(f)

        with open(vocab_file, 'rb') as f:
            self.vocab = pickle.load(f)

        self.transform = transform

        self.entries = []
        for img_id, captions in self.captions_data.items():
            if img_id in self.image_features:
                for cap in captions:
                    self.entries.append((img_id, cap))

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        img_id, caption = self.entries[index]
        feature = torch.tensor(self.image_features[img_id], dtype=torch.float32)

        # Numericalize caption with <SOS> and <EOS>
        numericalized = [self.vocab.stoi["<SOS>"]]
        numericalized += self.vocab.numericalize(caption)
        numericalized += [self.vocab.stoi["<EOS>"]]

        caption_tensor = torch.tensor(numericalized, dtype=torch.long)

        return feature, caption_tensor

# Custom collate function to pad captions
def caption_collate_fn(batch):
    features, captions = zip(*batch)
    features = torch.stack(features)

    lengths = [len(cap) for cap in captions]
    padded_captions = torch.nn.utils.rnn.pad_sequence(
        captions, batch_first=True, padding_value=0
    )

    return features, padded_captions, lengths
