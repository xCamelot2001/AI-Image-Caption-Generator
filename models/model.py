import torch
import torch.nn as nn

class CaptionGenerator(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(CaptionGenerator, self).__init__()

        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)

        # To transform the 2048-dim CNN vector to match the LSTM hidden size
        self.init_hidden = nn.Linear(2048, hidden_size)

    def forward(self, image_features, captions):
        """
        image_features: (B, 2048)
        captions: (B, max_len) — already numericalized
        """
        embeddings = self.embed(captions)                    # (B, max_len, embed_size)
        h0 = torch.tanh(self.init_hidden(image_features))    # (B, hidden_size)
        h0 = h0.unsqueeze(0)                                  # (1, B, hidden_size)
        c0 = torch.zeros_like(h0)                            # Initial cell state

        outputs, _ = self.lstm(embeddings, (h0, c0))          # (B, max_len, hidden_size)
        logits = self.linear(outputs)                         # (B, max_len, vocab_size)

        return logits
