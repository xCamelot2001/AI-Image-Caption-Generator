from torch.utils.data import DataLoader
from utils.dataset import CaptionDataset, caption_collate_fn
from utils.vocab import Vocabulary
from models.model import CaptionGenerator
from utils.dataset import CaptionDataset, caption_collate_fn
import torch
from torch.utils.data import DataLoader
import pickle

# Load vocab size
with open('data/vocab.pkl', 'rb') as f:
    vocab = pickle.load(f)
vocab_size = len(vocab)

# Hyperparameters
embed_size = 256
hidden_size = 512
num_layers = 1

# Model
model = CaptionGenerator(embed_size, hidden_size, vocab_size, num_layers)
print(f"✅ Model initialized with vocab size: {vocab_size}")


dataset = CaptionDataset(
    captions_file='data/cleaned_captions.json',
    features_file='data/image_features.json',
    vocab_file='data/vocab.pkl'
)

dataloader = DataLoader(
    dataset, batch_size=32, shuffle=True, collate_fn=caption_collate_fn
)

# Test batch
for batch in dataloader:
    features, captions, lengths = batch
    print("✅ Feature shape:", features.shape)    # (32, 2048)
    print("✅ Captions shape:", captions.shape)   # (32, max_seq_len)
    print("✅ First caption indices:", captions[0])
    break
