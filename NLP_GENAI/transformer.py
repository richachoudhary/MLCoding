from collections import Counter
import re





class SimpleTokenizer:
    def __init__(self, min_freq=1):
        self.min_freq = min_freq
        self.word2idx = {'<pad>': 0, '<unk>': 1}
        self.idx2word = {0: '<pad>', 1: '<unk>'}

    def build_vocab(self, sentences):
        word_freq = Counter()
        for sentence in sentences:
            tokens = self._tokenize(sentence)
            word_freq.update(tokens)
        
        for word, freq in word_freq.items():
            if freq >= self.min_freq:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word

    def _tokenize(self, sentence):
        # Basic whitespace and punctuation split
        sentence = sentence.lower()
        sentence = re.sub(r"[^a-zA-Z0-9]+", " ", sentence)
        return sentence.strip().split()

    def encode(self, sentence, max_len=None):
        tokens = self._tokenize(sentence)
        ids = [self.word2idx.get(tok, self.word2idx['<unk>']) for tok in tokens]
        if max_len:
            ids = ids[:max_len] + [self.word2idx['<pad>']] * max(0, max_len - len(ids))
        return ids

    def decode(self, ids):
        return ' '.join([self.idx2word.get(i, '<unk>') for i in ids])
    
# example usage

texts = [
    "I am a student",
    "You are a teacher"
]

# Example Usage

sentences = [
    "Hello world!",
    "Transformer models are amazing.",
    "Hello machine learning."
]

tokenizer = SimpleTokenizer(min_freq=1)
tokenizer.build_vocab(sentences)

for s in sentences:
    print(f"Original: {s}")
    print(f"Encoded : {tokenizer.encode(s, max_len=6)}")
    print()
    print(f"Decode sentence : {tokenizer.decode([4, 9, 2, 3])}")
    print()


import torch
import math

class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        """
        d_model: embedding dimension
        max_len: maximum length of input sequence
        """
        super().__init__()

        # Step 1: Create a matrix of shape (max_len, d_model)
        pe = torch.zeros(max_len, d_model)

        # Step 2: Create a vector of positions [0, 1, 2, ..., max_len-1]
        position = torch.arange(0, max_len).unsqueeze(1)  # Shape: (max_len, 1)

        # Step 3: Compute the denominator term of the PE formula
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        # Shape: (d_model/2,)

        # Step 4: Apply sin to even indices
        pe[:, 0::2] = torch.sin(position * div_term)

        # Step 5: Apply cos to odd indices
        pe[:, 1::2] = torch.cos(position * div_term)

        # Step 6: Add batch dimension (1, max_len, d_model)
        pe = pe.unsqueeze(0)

        # Step 7: Register as a buffer (not a parameter)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: (batch_size, seq_len, d_model)
        returns: input + positional encoding (trimmed to seq_len)
        """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len]
        return x
    
# ✅ Example: Add Positional Encoding to Toy Inputs


# Example embedding input: (batch=1, seq_len=4, d_model=32)
dummy_input = torch.zeros(1, 4, 32)

# Create the PositionalEncoding module
pe = PositionalEncoding(d_model=32, max_len=10)

# Add positional info to the dummy input
output = pe(dummy_input)

print(output[0])  # Print the PE vectors for each position

import matplotlib.pyplot as plt

pe_matrix = pe.pe[0, :100].detach().numpy()  # (100, d_model)

# plt.figure(figsize=(14, 6))
# plt.plot(pe_matrix[:, 0], label="dim 0")
# plt.plot(pe_matrix[:, 1], label="dim 1")
# plt.plot(pe_matrix[:, 2], label="dim 2")
# plt.plot(pe_matrix[:, 3], label="dim 3")
# plt.legend()
# plt.title("Positional Encoding Patterns (First 100 positions)")
# plt.xlabel("Position")
# plt.ylabel("Encoding Value")
# plt.grid(True)
# plt.show()


import torch
import torch.nn.functional as F

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Q: (batch_size, seq_len_q, d_k)
    K: (batch_size, seq_len_k, d_k)
    V: (batch_size, seq_len_k, d_v)
    mask: (batch_size, seq_len_q, seq_len_k), optional
    """
    d_k = Q.size(-1)  # key dimension

    # Step 1: Compute raw scores (dot product)
    scores = torch.matmul(Q, K.transpose(-2, -1))  # (batch, seq_q, seq_k)

    # Step 2: Scale the scores
    scores = scores / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))

    # Step 3: Apply mask (optional) - to be used in decoder
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))

    # Step 4: Normalize scores with softmax
    attn_weights = F.softmax(scores, dim=-1)

    # Step 5: Multiply by values
    output = torch.matmul(attn_weights, V)  # (batch, seq_q, d_v)

    return output, attn_weights
# Toy example


# Toy Q, K, V (batch_size=1, seq_len=3, dim=4)
Q = torch.tensor([[[1.0, 0.0, 1.0, 0.0]]])  # shape (1, 1, 4)
K = torch.tensor([[[1.0, 0.0, 1.0, 0.0],
                   [0.0, 1.0, 0.0, 1.0],
                   [1.0, 1.0, 0.0, 0.0]]])  # shape (1, 3, 4)
V = torch.tensor([[[10.0, 0.0],
                   [0.0, 10.0],
                   [5.0, 5.0]]])  # shape (1, 3, 2)

output, attn_weights = scaled_dot_product_attention(Q, K, V)

print("Attention Weights:\n", attn_weights)
print("Output:\n", output)