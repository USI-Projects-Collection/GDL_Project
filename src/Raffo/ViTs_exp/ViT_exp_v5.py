import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import networkx as nx
import time
import os

# --- 1. CONFIGURATION ---
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
EPOCHS = 15
IMAGE_SIZE = 32
PATCH_SIZE = 4
DIM = 64
DEPTH = 2
NUM_HEADS = 4
MLP_DIM = 128
DROPOUT = 0.1

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

print(f"Using Device: {DEVICE}")

# --- 2. DATASET ---
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# --- 3. ATTENTION MODULES ---

class SoftmaxAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.reshape(B, N, self.num_heads, -1).transpose(1, 2), qkv)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.to_out(out)

class LinearAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Linear(dim, dim)
        self.eps = 1e-6

    def feature_map(self, x):
        return torch.nn.functional.relu(x) + self.eps

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.reshape(B, N, self.num_heads, -1).transpose(1, 2), qkv)
        q = self.feature_map(q)
        k = self.feature_map(k)
        kv = k.transpose(-2, -1) @ v
        z = 1 / (q @ k.sum(dim=-2, keepdim=True).transpose(-2, -1) + self.eps)
        out = (q @ kv) * z
        out = out.transpose(1, 2).reshape(B, N, C)
        return self.to_out(out)

class GRFExactAttention(nn.Module):
    def __init__(self, dim, num_heads, num_patches, n_walks, p_halt):
        super().__init__()
        self.num_heads = num_heads
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Linear(dim, dim)
        self.eps = 1e-6
        self.register_buffer('mask', self._generate_grf_mask(num_patches, n_walks, p_halt))

    def _generate_grf_mask(self, N, n_walks, p_halt):
        side = int(np.sqrt(N))
        G = nx.grid_2d_graph(side, side)
        mapping = {node: i for i, node in enumerate(sorted(list(G.nodes())))}
        G = nx.relabel_nodes(G, mapping)
        mask = torch.zeros(N, N)
        for start_node in range(N):
            for _ in range(n_walks):
                curr = start_node
                while True:
                    mask[start_node, curr] += 1.0
                    if np.random.rand() < p_halt: break
                    neighbors = sorted(list(G.neighbors(curr)))
                    if not neighbors: break
                    curr = np.random.choice(neighbors)
            mask[start_node] /= max(n_walks, 1)
        return mask.to(DEVICE)

    def feature_map(self, x):
        return torch.nn.functional.relu(x) + self.eps

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.reshape(B, N, self.num_heads, -1).transpose(1, 2), qkv)
        q = self.feature_map(q)
        k = self.feature_map(k)
        
        q_graph = (q.transpose(-2, -1) @ self.mask).transpose(-2, -1)
        k_graph = (k.transpose(-2, -1) @ self.mask).transpose(-2, -1)
        q = q + 0.1 * q_graph
        k = k + 0.1 * k_graph 

        linear_kernel = q @ k.transpose(-2, -1)
        masked_kernel = linear_kernel * self.mask.unsqueeze(0).unsqueeze(0)
        z = 1 / (masked_kernel.sum(dim=-1, keepdim=True) + self.eps)
        out = (masked_kernel @ v) * z
        out = out.transpose(1, 2).reshape(B, N, C)
        return self.to_out(out)

class MAlphaAttention(nn.Module):
    """
    Exact M_alpha(G) Masked Linear Attention.
    Calculates the exact matrix power series instead of using random walks.
    M = Sum(alpha_k * W^k)
    """
    def __init__(self, dim, num_heads, num_patches, order=5, decay=0.5):
        super().__init__()
        self.num_heads = num_heads
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Linear(dim, dim)
        self.eps = 1e-6
        self.register_buffer('mask', self._generate_exact_mask(num_patches, order, decay))

    def _generate_exact_mask(self, N, order, decay):
        # 1. Adjacency Matrix
        side = int(np.sqrt(N))
        G = nx.grid_2d_graph(side, side)
        mapping = {node: i for i, node in enumerate(sorted(list(G.nodes())))}
        G = nx.relabel_nodes(G, mapping)
        A = nx.to_numpy_array(G)
        
        # Normalize (Random Walk Normalization)
        D_inv = np.diag(1.0 / np.maximum(A.sum(axis=1), 1))
        W = D_inv @ A
        
        # 2. Power Series: M = I + aW + a^2W^2 ...
        M = np.eye(N)
        W_k = np.eye(N)
        coeff = 1.0
        
        # Approximate series up to `order` terms
        for _ in range(order):
            W_k = W_k @ W
            coeff *= decay
            M += coeff * W_k
            
        # Normalize mask to keep attention scale roughly 1
        M = M / M.sum(axis=1, keepdims=True)
        return torch.tensor(M, dtype=torch.float32).to(DEVICE)

    def feature_map(self, x):
        return torch.nn.functional.relu(x) + self.eps

    def forward(self, x):
        # Use exact mask same way as GRF
        B, N, C = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.reshape(B, N, self.num_heads, -1).transpose(1, 2), qkv)
        
        q = self.feature_map(q)
        k = self.feature_map(k)
        
        q_graph = (q.transpose(-2, -1) @ self.mask).transpose(-2, -1)
        k_graph = (k.transpose(-2, -1) @ self.mask).transpose(-2, -1)
        q = q + 0.1 * q_graph
        k = k + 0.1 * k_graph 

        linear_kernel = q @ k.transpose(-2, -1)
        masked_kernel = linear_kernel * self.mask.unsqueeze(0).unsqueeze(0)
        z = 1 / (masked_kernel.sum(dim=-1, keepdim=True) + self.eps)
        out = (masked_kernel @ v) * z
        out = out.transpose(1, 2).reshape(B, N, C)
        return self.to_out(out)

class ToeplitzAttention(nn.Module):
    """
    Simulated Toeplitz Masking.
    Mask M_ij depends only on spatial distance on the grid.
    """
    def __init__(self, dim, num_heads, num_patches, decay=0.8):
        super().__init__()
        self.num_heads = num_heads
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Linear(dim, dim)
        self.eps = 1e-6
        self.register_buffer('mask', self._generate_toeplitz_mask(num_patches, decay))

    def _generate_toeplitz_mask(self, N, decay):
        side = int(np.sqrt(N))
        mask = np.zeros((N, N))
        
        for i in range(N):
            for j in range(N):
                # Convert flat index to (x, y)
                xi, yi = i // side, i % side
                xj, yj = j // side, j % side
                
                # Manhattan distance
                dist = abs(xi - xj) + abs(yi - yj)
                
                # Exponential decay mask (Toeplitz-like structure)
                mask[i, j] = decay ** dist
                
        # Normalize
        mask = mask / mask.sum(axis=1, keepdims=True)
        return torch.tensor(mask, dtype=torch.float32).to(DEVICE)

    def feature_map(self, x):
        return torch.nn.functional.relu(x) + self.eps

    def forward(self, x):
        # Same symmetric injection logic for fair comparison
        B, N, C = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.reshape(B, N, self.num_heads, -1).transpose(1, 2), qkv)
        
        q = self.feature_map(q)
        k = self.feature_map(k)
        
        q_graph = (q.transpose(-2, -1) @ self.mask).transpose(-2, -1)
        k_graph = (k.transpose(-2, -1) @ self.mask).transpose(-2, -1)
        q = q + 0.1 * q_graph
        k = k + 0.1 * k_graph 

        linear_kernel = q @ k.transpose(-2, -1)
        masked_kernel = linear_kernel * self.mask.unsqueeze(0).unsqueeze(0)
        z = 1 / (masked_kernel.sum(dim=-1, keepdim=True) + self.eps)
        out = (masked_kernel @ v) * z
        out = out.transpose(1, 2).reshape(B, N, C)
        return self.to_out(out)

# --- 4. MODEL ---
class ViT(nn.Module):
    def __init__(self, attention_type='softmax', n_walks=50, p_halt=0.1):
        super().__init__()
        self.patch_size = PATCH_SIZE
        num_patches = (IMAGE_SIZE // PATCH_SIZE) ** 2
        patch_dim = 3 * PATCH_SIZE ** 2
        self.patch_embed = nn.Linear(patch_dim, DIM)
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, DIM))
        self.layers = nn.ModuleList([])
        
        for _ in range(DEPTH):
            if attention_type == 'softmax':
                attn = SoftmaxAttention(DIM, NUM_HEADS)
            elif attention_type == 'linear':
                attn = LinearAttention(DIM, NUM_HEADS)
            elif attention_type == 'grf':
                attn = GRFExactAttention(DIM, NUM_HEADS, num_patches, n_walks, p_halt)
            elif attention_type == 'm_alpha':
                attn = MAlphaAttention(DIM, NUM_HEADS, num_patches)
            elif attention_type == 'toeplitz':
                attn = ToeplitzAttention(DIM, NUM_HEADS, num_patches)
            
            self.layers.append(nn.ModuleList([
                nn.LayerNorm(DIM),
                attn,
                nn.LayerNorm(DIM),
                nn.Sequential(
                    nn.Linear(DIM, MLP_DIM), nn.GELU(), nn.Dropout(DROPOUT),
                    nn.Linear(MLP_DIM, DIM), nn.Dropout(DROPOUT)
                )
            ]))
        self.mlp_head = nn.Sequential(nn.LayerNorm(DIM), nn.Linear(DIM, 10))

    def forward(self, img):
        p = self.patch_size
        x = img.unfold(2, p, p).unfold(3, p, p).reshape(img.shape[0], -1, 3 * p * p)
        x = self.patch_embed(x)
        B, N, _ = x.shape
        x += self.pos_embed[:, :N]
        for norm1, attn, norm2, mlp in self.layers:
            x = x + attn(norm1(x))
            x = x + mlp(norm2(x))
        return self.mlp_head(x.mean(dim=1))

# --- 5. TRAINING UTILS ---
def train_and_evaluate(model_type, n_walks=50, p_halt=0.1):
    print(f"\n--- Training {model_type.upper()} ---")
    
    torch.manual_seed(42)
    np.random.seed(42)
    
    model = ViT(attention_type=model_type, n_walks=n_walks, p_halt=p_halt).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    
    start_time = time.time()
    for epoch in range(EPOCHS):
        model.train()
        for images, labels in trainloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    
    train_time = time.time() - start_time
    
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    acc = 100 * correct / total
    print(f"   -> Result: Acc = {acc:.2f}%")
    return acc

# --- 6. REPLICATION RUNNERS ---

def replicate_table_1_complete():
    print("\n" + "="*70)
    print("📜 REPLICATING TABLE 1 (COMPLETE): All 5 Methods")
    print("   Goal: Softmax > (GRF ~= M_alpha ~= Toeplitz) > Linear")
    print("="*70)
    
    acc_softmax = train_and_evaluate('softmax')
    acc_toeplitz = train_and_evaluate('toeplitz')
    acc_m_alpha = train_and_evaluate('m_alpha')
    acc_grf = train_and_evaluate('grf', n_walks=50, p_halt=0.1)
    acc_linear = train_and_evaluate('linear')
    
    print("\n[TABLE 1 COMPLETE RESULT]")
    print(f"{'Method':<25} {'Accuracy':<10} {'Type'}")
    print("-" * 50)
    print(f"{'Unmasked Softmax':<25} {acc_softmax:<10.2f} {'Exact Dense'}")
    print(f"{'Toeplitz-masked Linear':<25} {acc_toeplitz:<10.2f} {'Structure Bias'}")
    print(f"{'M_alpha(G)-masked':<25} {acc_m_alpha:<10.2f} {'Exact Topo'}")
    print(f"{'GRF-masked Linear':<25} {acc_grf:<10.2f} {'Stochastic Topo'}")
    print(f"{'Unmasked Linear':<25} {acc_linear:<10.2f} {'Baseline'}")
    print("="*70)

if __name__ == "__main__":
    replicate_table_1_complete()