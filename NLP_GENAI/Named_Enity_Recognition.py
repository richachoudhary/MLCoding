import torch 
import torch.nn as nn 

class BiLSTMTagger(nn.Module):
    def __init__(self, vocab_size, tagset_size, embedding_dim=10, hidden_dim=16):
        super(BiLSTMTagger, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, num_layers=1,
                            bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim, tagset_size)

    def forward(self, x):
        embeds = self.embedding(x)
        lstm_out, _ = self.lstm(embeds)
        tag_scores = self.fc(lstm_out)
        return tag_scores

X = [["John", "lives", "in", "New", "York"]]
y = [["B-PER", "O", "O", "B-LOC", "I-LOC"]]

word2idx = {"<PAD>": 0, "John": 1, "lives": 2, "in": 3, "New": 4, "York": 5}
tag2idx = {"O": 0, "B-PER": 1, "B-LOC": 2, "I-LOC": 3}
X_idx = [[word2idx[token] for token in sent] for sent in X]
y_idx = [[tag2idx[label] for label in tags] for tags in y]
print("X_idx",X_idx)

model = BiLSTMTagger(vocab_size=len(word2idx), tagset_size=len(tag2idx))
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

X_tensor = torch.tensor(X_idx, dtype=torch.long)
y_tensor = torch.tensor(y_idx, dtype=torch.long)

for epoch in range(100):
    model.zero_grad()
    output = model(X_tensor)
    loss = loss_fn(output.view(-1, len(tag2idx)), y_tensor.view(-1))
    if epoch%10 ==0 : 
        print(f"Loss = {loss}")
    loss.backward()
    optimizer.step()
