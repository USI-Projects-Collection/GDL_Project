import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
import time
import os

# --- CONFIGURATION ---
MODEL_MODE = 'grf'  # 'grf', 'baseline', 'mp'
DATA_PATH = "./data/"
BATCH_SIZE = 16
LR = 1e-3 # Learning Rate
EPOCHS = 500
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
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
        flat_idx = knn_idx.reshape(B, N * self.k).unsqueeze(-1).expand(-1, -1, D)
        neighbors = torch.gather(x, 1, flat_idx.reshape(B, N * self.k, D).long()).reshape(B, N, self.k, D)
        return self.proj(neighbors.mean(dim=2))

class UnifiedInterlacer(nn.Module):
    def __init__(self, mode='grf', input_dim=3, embed_dim=128, num_layers=5):
        super().__init__()
        self.mode = mode
        self.num_layers = num_layers
        self.embedding = nn.Linear(input_dim, embed_dim)
        
        # Create layer norms for each layer (2 per block: graph + attention)
        self.norms = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(num_layers * 2)])
        
        # Create graph layers (GRF, MP, or Identity based on mode)
        if mode == 'grf':
            self.graph_layers = nn.ModuleList([TopologicalGRFLayer(embed_dim, K_NEIGHBORS) for _ in range(num_layers)])
        elif mode == 'mp':
            self.graph_layers = nn.ModuleList([SimpleMessagePassing(embed_dim, K_NEIGHBORS) for _ in range(num_layers)])
        else:
            self.graph_layers = nn.ModuleList([nn.Identity() for _ in range(num_layers)])
        
        # Create attention layers
        self.attn_layers = nn.ModuleList([LinearAttention(embed_dim) for _ in range(num_layers)])
        
        self.head = nn.Linear(embed_dim, 3)

    def forward(self, x, knn):
        h = self.embedding(x)
        
        for i in range(self.num_layers):
            # Graph layer
            if self.mode != 'baseline':
                h = h + self.graph_layers[i](self.norms[i * 2](h), knn)
            else:
                h = h + self.norms[i * 2](h)
            
            # Attention layer
            h = h + self.attn_layers[i](self.norms[i * 2 + 1](h))
        
        return self.head(h)

# --- MAIN ---
def main():
    dataset = RobotArmDataset("points.npy", "knn_indices.npy")
    train_size = int(len(dataset) * (1 - VAL_SPLIT))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = UnifiedInterlacer(mode=MODEL_MODE).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    # Track losses
    train_losses = []
    val_losses = []

    print(f"Start Training: {MODEL_MODE.upper()}")
    for ep in range(EPOCHS):
        # Training
        model.train()
        epoch_train_losses = []
        for x, knn, y in train_loader:
            x, knn, y = x.to(DEVICE), knn.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            pred = model(x, knn)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            epoch_train_losses.append(loss.item())

        # Validation
        model.eval()
        epoch_val_losses = []
        with torch.no_grad():
            for x, knn, y in val_loader:
                x, knn, y = x.to(DEVICE), knn.to(DEVICE), y.to(DEVICE)
                pred = model(x, knn)
                loss = criterion(pred, y)
                epoch_val_losses.append(loss.item())

        train_loss = np.mean(epoch_train_losses)
        val_loss = np.mean(epoch_val_losses)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Ep {ep+1}: Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

    # Plot and save loss curves
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, EPOCHS + 1), train_losses, label='Training Loss', marker='o')
    plt.plot(range(1, EPOCHS + 1), val_losses, label='Validation Loss', marker='s')
    plt.title(f'Training and Validation Loss - {MODEL_MODE.upper()}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(f'loss_plot_{MODEL_MODE}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Loss plot saved to loss_plot_{MODEL_MODE}.png")

    # Save losses to numpy file
    np.savez(f'losses_{MODEL_MODE}.npz', 
             train_losses=np.array(train_losses), 
             val_losses=np.array(val_losses))
    print(f"Losses saved to losses_{MODEL_MODE}.npz")
    print()

    # SAVE MODEL WEIGHTS (include losses in checkpoint)
    torch.save({
        'model_state_dict': model.state_dict(),
        'mean': dataset.mean,
        'std': dataset.std,
        'mode': MODEL_MODE,
        'train_losses': train_losses,
        'val_losses': val_losses
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
