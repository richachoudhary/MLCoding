import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np
import pickle 

def set_seed(seed=42):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

set_seed(42)

def generate_cbow_data(sentences, window_size=2):
    data = []
    for sent in sentences:
        words = sent.split()
        for i in range(window_size, len(words) - window_size):
            context = words[i - window_size:i] + words[i+1:i+window_size+1]
            target = words[i]
            data.append((context, target))
    return data

class CBOW(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super(CBOW, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embed_size)
        self.linear = nn.Linear(embed_size, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs).mean(dim=0)
        out = self.linear(embeds)
        return out

sentences = ["the cat sat on the mat", "the dog sat on the log", "the bank is closed today money money", "the river bank is wide"]
tokens = list(set(" ".join(sentences).split()))
word2idx = {w: i for i, w in enumerate(tokens)}
idx2word = {i: w for w, i in word2idx.items()}

data = generate_cbow_data(sentences)   
model = CBOW(len(tokens), 10)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

for epoch in range(100):
    total_loss = 0
    for context, target in data:
        context_idxs = torch.tensor([word2idx[w] for w in context], dtype=torch.long)
        target_idx = torch.tensor([word2idx[target]], dtype=torch.long)

        model.zero_grad()
        output = model(context_idxs)
        loss = loss_fn(output.unsqueeze(0), target_idx)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    # if epoch % 10 == 0:
        # print(f"Epoch {epoch}, Loss: {total_loss:.4f}")
    
def create_embeddings(text):
    context_idxs= torch.tensor([word2idx[w] for w in context], dtype= torch.long)
    output = model(context_idxs)
    print(output)



embedding_matrix = model.embeddings.weight.data

bank_idx = word2idx['bank']
print("Embedding for 'bank':")
print(embedding_matrix[bank_idx])

# Compare with other word embeddings (e.g., 'river', 'money')


def cosine_similarity(vec1, vec2):
    return F.cosine_similarity(vec1.unsqueeze(0), vec2.unsqueeze(0)).item()

money_vec = embedding_matrix[word2idx['money']]
river_vec = embedding_matrix[word2idx['river']]
bank_vec = embedding_matrix[bank_idx]
cat_vec = embedding_matrix[word2idx['cat']]
mat_vec = embedding_matrix[word2idx['mat']]
log_vec = embedding_matrix[word2idx['log']]

print("Similarity between 'bank' and 'money':", cosine_similarity(bank_vec, money_vec))
print("Similarity between 'bank' and 'river':", cosine_similarity(bank_vec, river_vec))
print("Similarity between 'cat' and 'mat':", cosine_similarity(cat_vec, mat_vec))
print("Similarity between 'cat' and 'log':", cosine_similarity(cat_vec, log_vec))

# Visualization using PCA of the embeddings created using trained CBOW model



# Get embeddings from the model (usually the input embedding layer)
embeddings = model.embeddings.weight.detach().numpy()  # shape: (vocab_size, embedding_dim)
x

embeddings_dict = {
    word: model.embeddings.weight[idx].detach().numpy()
    for word, idx in word2idx.items()
}

with open("embeddings.pkl", "wb") as f:
    pickle.dump(embeddings_dict, f)


