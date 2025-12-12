import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import gc  # Add garbage collector

# --- CONFIGURATION ---
MODEL_MODE = 'grf'  # 'grf', 'baseline', 'mp'
DATA_PATH = "/kaggle/input/4096-5/data/"
OUTPUT_PATH = '/kaggle/working/'
BATCH_SIZE = 16
LR = 1e-3 # Learning Rate
EPOCHS = 100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

points_path = os.path.join(DATA_PATH, "points.npy")
knn_path = os.path.join(DATA_PATH, "knn_indices.npy")


K_NEIGHBORS = 6
VAL_SPLIT = 0.1
TRAIN_ROLLOUT_STEPS = 3  # Number of autoregressive steps during training

# --- DYNAMIC KNN FUNCTION ---
def compute_knn_torch(points, k):
    """Compute KNN dynamically on GPU - works with batched tensors"""
    B, N, D = points.shape
    # Compute pairwise distances
    dist_mat = torch.cdist(points, points, p=2)
    # Get k+1 nearest neighbors (includes self), then exclude self
    _, knn_indices = torch.topk(dist_mat, k=k+1, dim=-1, largest=False)
    return knn_indices[:, :, 1:]  # Shape: (B, N, k)

# --- DATASET ---
class RobotArmDataset(Dataset):
    def __init__(self, points_path, knn_path, rollout_steps=1):
        if not os.path.exists(points_path):
            raise FileNotFoundError(f"File {points_path} not found.")
        self.points = np.load(points_path).astype(np.float32)
        self.knn = np.load(knn_path).astype(np.int64)
        self.rollout_steps = rollout_steps

        # Normalization stats
        self.mean = np.mean(self.points, axis=(0, 1))
        self.std = np.std(self.points, axis=(0, 1))
        self.points = (self.points - self.mean) / (self.std + 1e-6)

    def __len__(self):
        # Account for needing rollout_steps future frames
        return len(self.points) - self.rollout_steps

    def __getitem__(self, idx):
        # Return input, knn, and sequence of rollout_steps targets
        input_frame = torch.from_numpy(self.points[idx])
        input_knn = torch.from_numpy(self.knn[idx])
        # Stack future targets: [t+1, t+2, ..., t+rollout_steps]
        targets = torch.stack([
            torch.from_numpy(self.points[idx + i + 1])
            for i in range(self.rollout_steps)
        ])  # Shape: (rollout_steps, N, 3)
        return input_frame, input_knn, targets

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
    def __init__(self, dim, k_neighbors, hops=5):
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
    def __init__(self, mode='grf', input_dim=3, embed_dim=64, num_layers=4):
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
    dataset = RobotArmDataset(points_path, knn_path, rollout_steps=TRAIN_ROLLOUT_STEPS)
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

    print(f"Start Training: {MODEL_MODE.upper()} with {TRAIN_ROLLOUT_STEPS}-step rollout")
    for ep in range(EPOCHS):
        # Training with multi-step rollout
        model.train()
        epoch_train_losses = []
        for x, knn, targets in train_loader:
            # x: (B, N, 3), knn: (B, N, K), targets: (B, ROLLOUT_STEPS, N, 3)
            x, knn, targets = x.to(DEVICE), knn.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()

            # Multi-step rollout (BPTT - no detach)
            current_input = x
            current_knn = knn
            step_losses = []

            for step in range(TRAIN_ROLLOUT_STEPS):
                # Predict next frame
                pred = model(current_input, current_knn)

                # Loss against ground truth target at this step
                gt_target = targets[:, step]  # (B, N, 3)
                step_loss = criterion(pred, gt_target)
                step_losses.append(step_loss)

                # CLOSED LOOP: prediction becomes next input (NO detach for BPTT)
                current_input = pred

                # DYNAMIC KNN: recompute graph on predicted coordinates
                current_knn = compute_knn_torch(current_input, K_NEIGHBORS)

            # Total loss = average across all steps
            total_loss = torch.stack(step_losses).mean()
            total_loss.backward()
            optimizer.step()
            epoch_train_losses.append(total_loss.item())

        # Validation with same rollout
        model.eval()
        epoch_val_losses = []
        with torch.no_grad():
            for x, knn, targets in val_loader:
                x, knn, targets = x.to(DEVICE), knn.to(DEVICE), targets.to(DEVICE)

                current_input = x
                current_knn = knn
                step_losses = []

                for step in range(TRAIN_ROLLOUT_STEPS):
                    pred = model(current_input, current_knn)
                    gt_target = targets[:, step]
                    step_loss = criterion(pred, gt_target)
                    step_losses.append(step_loss)

                    current_input = pred
                    current_knn = compute_knn_torch(current_input, K_NEIGHBORS)

                total_loss = torch.stack(step_losses).mean()
                epoch_val_losses.append(total_loss.item())

        train_loss = np.mean(epoch_train_losses)
        val_loss = np.mean(epoch_val_losses)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Ep {ep+1}: Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

    # Plot and save loss curves
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, EPOCHS + 1), train_losses, label='Training Loss')
    plt.plot(range(1, EPOCHS + 1), val_losses, label='Validation Loss')
    plt.title(f'Training and Validation Loss - {MODEL_MODE.upper()} ({TRAIN_ROLLOUT_STEPS}-step rollout)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(f'/kaggle/working/loss_plot_{MODEL_MODE}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Loss plot saved to loss_plot_{MODEL_MODE}.png")

    # Save losses to numpy file
    np.savez(f'losses_{MODEL_MODE}.npz',
             train_losses=np.array(train_losses),
             val_losses=np.array(val_losses))
    print(f"Losses saved to losses_{MODEL_MODE}.npz")
    print()

    # SAVE MODEL WEIGHTS
    torch.save({
        'model_state_dict': model.state_dict(),
        'mean': dataset.mean,
        'std': dataset.std,
        'mode': MODEL_MODE,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'rollout_steps': TRAIN_ROLLOUT_STEPS
    }, f"{OUTPUT_PATH}model_{MODEL_MODE}.pth")
    print(f"Model saved to model_{MODEL_MODE}.pth")

if __name__ == "__main__":
    # os.chdir(DATA_PATH)

    for mode in ['grf', 'mp', 'baseline']:
        MODEL_MODE = mode
        print(f"\n{'='*50}")
        print(f"Starting training for: {MODEL_MODE.upper()}")
        print(f"{'='*50}\n")

        main()

        # --- AGGRESSIVE CLEANUP ---
        # Clear all cached memory on GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        # Force Python garbage collection
        gc.collect()

        print(f"\n[Cleanup] GPU cache cleared after {MODEL_MODE.upper()} training.\n")
