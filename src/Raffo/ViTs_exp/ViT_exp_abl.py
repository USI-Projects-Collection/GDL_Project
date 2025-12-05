import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import networkx as nx
import time
import matplotlib.pyplot as plt
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
# Auto-detect device
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

# Default Graph Parameters (Overwritten in specific runs)
N_WALKS = 50          
P_HALT = 0.1          
MAX_WALK_LEN = 10

# --- 2. DATASET ---
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
# Set num_workers=0 to avoid multiprocessing issues in interactive environments
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
        # Pre-compute the Mask
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
            mask[start_node] /= max(n_walks, 1) # Avoid div by zero
        return mask.to(DEVICE)

    def feature_map(self, x):
        return torch.nn.functional.relu(x) + self.eps

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.reshape(B, N, self.num_heads, -1).transpose(1, 2), qkv)
        q = self.feature_map(q)
        k = self.feature_map(k)
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
                # Pass specific graph params here
                attn = GRFExactAttention(DIM, NUM_HEADS, num_patches, n_walks, p_halt)
            
            self.layers.append(nn.ModuleList([
                nn.LayerNorm(DIM),
                attn,
                nn.LayerNorm(DIM),
                nn.Sequential(
                    nn.Linear(DIM, MLP_DIM),
                    nn.GELU(),
                    nn.Dropout(DROPOUT),
                    nn.Linear(MLP_DIM, DIM),
                    nn.Dropout(DROPOUT)
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
    print(f"\n--- Training {model_type.upper()} (n={n_walks}, p={p_halt}) ---")
    
    # Seeding for fairness
    torch.manual_seed(42)
    np.random.seed(42)
    
    model = ViT(attention_type=model_type, n_walks=n_walks, p_halt=p_halt).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    
    start_time = time.time()
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for images, labels in trainloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {running_loss/len(trainloader):.4f}")

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
    print(f"Result: Acc = {acc:.2f}%, Time = {train_time:.2f}s")
    return acc, train_time

# --- 6. EXPERIMENT RUNNERS ---

def run_main_comparison():
    print("\n" + "="*60)
    print("🔬 EXPERIMENT 1: COMPARATIVE PERFORMANCE (Main Result)")
    print("="*60)
    
    results = {}
    results['softmax'] = train_and_evaluate('softmax')
    results['linear'] = train_and_evaluate('linear')
    # Best config: p=0.1, n=50
    results['grf'] = train_and_evaluate('grf', n_walks=50, p_halt=0.1)
    
    print("\n" + "="*50)
    print(f"{'Method':<15} {'Accuracy':<15} {'Time (s)':<15}")
    print("-" * 50)
    for name, (acc, t) in results.items():
        print(f"{name:<15} {acc:<15.2f} {t:<15.2f}")
    print("="*50)

def run_ablation_study():
    print("\n" + "="*60)
    print("🔬 EXPERIMENT 2: ABLATION STUDY (Replicating Fig. 6)")
    print("   Setting p_halt = 0.5 to study convergence")
    print("="*60)
    
    walker_counts = [1, 10, 100, 1000]
    accuracies = []
    
    for n in walker_counts:
        # Use p_halt=0.5 as per Section C.2 of the paper
        acc, _ = train_and_evaluate('grf', n_walks=n, p_halt=0.5)
        accuracies.append(acc)
        
    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(walker_counts, accuracies, 'o-', color='#1f77b4', linewidth=2)
    plt.xscale('log')
    plt.xlabel('Number of Walkers ($n$)')
    plt.ylabel('Accuracy (%)')
    plt.title('Ablation: Impact of Walker Count ($p_{halt}=0.5$)')
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.savefig('combined_ablation_plot.png', dpi=300)
    print("\n✅ Ablation plot saved to 'combined_ablation_plot.png'")

if __name__ == "__main__":
    print(f"Using Device: {DEVICE}")
    
    # RUN EVERYTHING
    run_main_comparison()
    run_ablation_study()