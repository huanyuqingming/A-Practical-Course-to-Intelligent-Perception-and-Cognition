import torch.nn as nn
from transformers import GPT2Config, GPT2LMHeadModel

class RNN(nn.Module):
    def __init__(self, vocab_size, emb_size=256, hidden_size=512, num_layers=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, emb_size)
        self.rnn = nn.RNN(emb_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        x = self.embed(x)
        out, hidden = self.rnn(x, hidden)
        logits = self.fc(out)
        return logits, hidden

class LSTM(nn.Module):
    def __init__(self, vocab_size, emb_size=256, hidden_size=512, num_layers=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, emb_size)
        self.lstm = nn.LSTM(emb_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        x = self.embed(x)
        out, hidden = self.lstm(x, hidden)
        logits = self.fc(out)
        return logits, hidden

class Transformer(nn.Module):
    def __init__(self, vocab_size, emb_size=256, num_heads=4, num_layers=4):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, emb_size)
        encoder_layer = nn.TransformerEncoderLayer(emb_size, num_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc = nn.Linear(emb_size, vocab_size)

    def forward(self, x, _=None):
        x = self.embed(x).permute(1, 0, 2)  # seq_len, batch, embed
        out = self.transformer(x)
        logits = self.fc(out).permute(1, 0, 2)
        return logits, None


def get_gpt_model(vocab_size):
    config = GPT2Config(
        vocab_size=vocab_size,
        n_positions=512,
        n_embd=384,
        n_layer=4,
        n_head=4
    )
    model = GPT2LMHeadModel(config)
    return model
