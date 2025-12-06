import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import time
import os

# --- CONFIGURATION ---
MODEL_MODE = 'grf'  # 'grf', 'baseline', 'mp'
DATA_PATH = "./data/"
BATCH_SIZE = 16
LR = 1e-3 # Learning Rate
EPOCHS = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
K_NEIGHBORS = 6
VAL_SPLIT = 0.2

# --- DATASET ---
class RobotArmDataset(Dataset):
    def __init__(self, points_path, knn_path):
        if not os.path.exists(points_path):
            raise FileNotFoundError(f"File {points_path} not found.")
        self.points = np.load(points_path).astype(np.float32)
        self.knn = np.load(knn_path).astype(np.int64)
        
        # Normalization stats
        self.mean = np.mean(self.points, axis=(0, 1))
        self.std = np.std(self.points, axis=(0, 1))
        self.points = (self.points - self.mean) / (self.std + 1e-6)

    def __len__(self):
        return len(self.points) - 1

    def __getitem__(self, idx):
        return torch.from_numpy(self.points[idx]), \
               torch.from_numpy(self.knn[idx]), \
               torch.from_numpy(self.points[idx + 1])

# --- MODELS ---
class LinearAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Linear(dim, dim)
        
    def forward(self, x):
        B, N, D = x.shape
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k = torch.relu(q) + 1e-6, torch.relu(k) + 1e-6
        kv = torch.einsum("bnd,bne->bde", k, v)
        z = 1 / (torch.einsum("bnd,bd->bn", q, k.sum(dim=1)) + 1e-6)
        attn = torch.einsum("bnd,bde,bn->bne", q, kv, z)
        return self.to_out(attn)

class TopologicalGRFLayer(nn.Module):
    def __init__(self, dim, k_neighbors, hops=3):
        super().__init__()
        self.k = k_neighbors
        self.hops = hops
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Linear(dim, dim)

    def forward(self, x, knn_idx):
        B, N, D = x.shape
        # Sparse Matrix Construction
        src = torch.arange(N, device=x.device).view(1, N, 1).expand(B, N, self.k)
        batch_off = torch.arange(B, device=x.device).view(B, 1, 1) * N
        indices = torch.stack([(knn_idx + batch_off).view(-1), (src + batch_off).view(-1)])
        values = torch.ones(indices.shape[1], device=x.device)
        adj = torch.sparse_coo_tensor(indices, values, (B*N, B*N))
        
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        v_f, k_f = v.view(B*N, D), k.view(B*N, D)
        
        # Random Walk Diffusion
        for _ in range(self.hops):
            v_f = torch.sparse.mm(adj, v_f) / (self.k + 1e-6)
            k_f = torch.sparse.mm(adj, k_f) / (self.k + 1e-6)
            
        attn = (q * k_f.view(B, N, D)).sum(dim=-1, keepdim=True)
        return self.to_out(attn * v_f.view(B, N, D))

class SimpleMessagePassing(nn.Module):
    def __init__(self, dim, k_neighbors):
        super().__init__()
        self.k = k_neighbors
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, knn_idx):
        B, N, D = x.shape
        flat_idx = knn_idx.view(B, N * self.k).unsqueeze(-1).expand(-1, -1, D)
        neighbors = torch.gather(x, 1, flat_idx.view(B, N * self.k, D).long()).view(B, N, self.k, D)
        return self.proj(neighbors.mean(dim=2))

class UnifiedInterlacer(nn.Module):
    def __init__(self, mode='grf', input_dim=3, embed_dim=64):
        super().__init__()
        self.mode = mode
        self.embedding = nn.Linear(input_dim, embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        
        if mode == 'grf':
            self.l1 = TopologicalGRFLayer(embed_dim, K_NEIGHBORS)
            self.l3 = TopologicalGRFLayer(embed_dim, K_NEIGHBORS)
        elif mode == 'mp':
            self.l1 = SimpleMessagePassing(embed_dim, K_NEIGHBORS)
            self.l3 = SimpleMessagePassing(embed_dim, K_NEIGHBORS)
        else:
            self.l1 = nn.Identity()
            self.l3 = nn.Identity()
            
        self.l2 = LinearAttention(embed_dim)
        self.head = nn.Linear(embed_dim, 3)

    def forward(self, x, knn):
        h = self.embedding(x)
        h = h + (self.l1(self.norm1(h), knn) if self.mode != 'baseline' else self.norm1(h))
        h = h + self.l2(self.norm2(h))
        h = h + (self.l3(self.norm3(h), knn) if self.mode != 'baseline' else self.norm3(h))
        return self.head(h)

# --- MAIN ---
def main():
    dataset = RobotArmDataset("points.npy", "knn_indices.npy")
    train_size = int(len(dataset) * (1 - VAL_SPLIT))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    
    model = UnifiedInterlacer(mode=MODEL_MODE).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    criterion = nn.MSELoss()
    
    print(f"Start Training: {MODEL_MODE.upper()}")
    for ep in range(EPOCHS):
        model.train()
        losses = []
        for x, knn, y in train_loader:
            x, knn, y = x.to(DEVICE), knn.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            pred = model(x, knn)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        print(f"Ep {ep+1}: {np.mean(losses):.6f}") # the mean because of batches

    # SAVE MODEL WEIGHTS
    torch.save({
        'model_state_dict': model.state_dict(),
        'mean': dataset.mean,
        'std': dataset.std,
        'mode': MODEL_MODE
    }, f"model_{MODEL_MODE}.pth")
    print(f"Model saved to model_{MODEL_MODE}.pth")

if __name__ == "__main__":
    os.chdir(DATA_PATH)

    MODEL_MODE = 'baseline'
    main()
    MODEL_MODE = 'mp'
    main()
    MODEL_MODE = 'grf'
    main()