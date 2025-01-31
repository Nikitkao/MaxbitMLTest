import torch.nn as nn
import torch

class TreeHealthModel(nn.Module):
    def __init__(self, num_numeric, cat_cardinalities, embedding_dim=10, hidden_dim=128):
        super(TreeHealthModel, self).__init__()
        
        # Эмбеддинги для категориальных признаков
        self.embeddings = nn.ModuleList([nn.Embedding(cat_size, embedding_dim) for cat_size in cat_cardinalities])
        
        # Размер входных данных
        input_size = num_numeric + len(cat_cardinalities) * embedding_dim
        
        # Полносвязные слои
        self.fc1 = nn.Linear(input_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 3)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x_num, x_cat):
        cat_embeddings = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        cat_embeddings = torch.cat(cat_embeddings, dim=1)
        x = torch.cat([x_num, cat_embeddings], dim=1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x