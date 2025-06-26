import torch
import pickle
from models.model import CaptionGenerator
from utils.dataset import CaptionDataset, caption_collate_fn
from torch.utils.data import DataLoader

# === Load the vocabulary size ===
with open('data/vocab.pkl', 'rb') as f:
    vocab = pickle.load(f)
vocab_size = len(vocab)

# === Hyperparameters ===
embed_size = 256
hidden_size = 512
num_layers = 1

# === Initialize model ===
model = CaptionGenerator(embed_size, hidden_size, vocab_size, num_layers)
print("✅ Model initialized.")

# === Load a batch ===
dataset = CaptionDataset(
    captions_file='data/cleaned_captions.json',
    features_file='data/image_features.json',
    vocab_file='data/vocab.pkl'
)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=caption_collate_fn)

features, captions, lengths = next(iter(dataloader))  # one batch
print("✅ Batch shapes:", features.shape, captions.shape)

# === Forward pass ===
outputs = model(features, captions[:, :-1])  # remove last token
print("✅ Output shape:", outputs.shape)  # should be (B, T, vocab_size)
