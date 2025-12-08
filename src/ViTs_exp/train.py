import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import networkx as nx
import time
from models import ViT

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

# --- 5. TRAINING UTILS ---
def train_and_evaluate(model_type, n_walks=50, p_halt=0.1):
    print(f"\n--- Training {model_type.upper()} ---")
    
    torch.manual_seed(42)
    np.random.seed(42)
    model = ViT(PATCH_SIZE, IMAGE_SIZE, DIM, DEPTH, NUM_HEADS, DROPOUT, MLP_DIM, DEVICE, attention_type=model_type, n_walks=n_walks, p_halt=p_halt)
    model = model.to(DEVICE)
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