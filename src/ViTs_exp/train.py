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
from models import ViT

def get_dataloaders(dataset_name, batch_size, resize_image=32, manipulate_images=False):
    transform_compose_list = [
            transforms.Resize((resize_image, resize_image)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]

    if manipulate_images:
        transform_compose_list = [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(resize_image, padding=4),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)
        ] + transform_compose_list
        
    if 'mnist' in dataset_name.lower():
        # Resize to 32x32 to match patch logic easily
        transform_compose_list = transform_compose_list[:-1] + [transforms.Normalize((0.5,), (0.5,))]

    transform = transforms.Compose(transform_compose_list)
        
    if dataset_name.lower() == 'cifar10':
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        num_classes = 10
        channels = 3
    elif dataset_name.lower() == 'cifar100':
        trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
        testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
        num_classes = 100
        channels = 3
    elif dataset_name.lower() == 'mnist':
        trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        num_classes = 10
        channels = 1
    elif dataset_name.lower() == 'fashionmnist':
        trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
        testset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
        num_classes = 10
        channels = 1
    else:
        raise ValueError("Unknown dataset")

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    return trainloader, testloader, num_classes, channels


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

def train_and_evaluate(model_type, dataset_name='cifar10', n_walks=50, p_halt=0.1, manipulate_images=False):
    print(f"\n--- Training {model_type.upper()} on {dataset_name.upper()} ---")

    trainloader, testloader, num_classes, channels = get_dataloaders(dataset_name, manipulate_images)

    model = ViT(
        patch_size=PATCH_SIZE,
        image_size=IMAGE_SIZE,
        dim=DIM,
        depth=DEPTH,
        num_heads=NUM_HEADS,
        dropout=DROPOUT,
        mlp_dim=MLP_DIM,
        device=DEVICE,
        channels=channels,
        attention_type=model_type,
        n_walks=n_walks,
        p_halt=p_halt,
        num_classes=num_classes
    )

    torch.manual_seed(42)
    np.random.seed(42)


    model = model.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    start_time = time.time()

    # --- TRAINING LOOP WITH PER-EPOCH LOGGING ---
    final_acc = 0.0

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

        avg_train_loss = running_loss / len(trainloader)

        # Evaluate after every epoch
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

        epoch_acc = 100 * correct / total
        final_acc = epoch_acc # Store last accuracy

        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_train_loss:.4f} | Test Acc: {epoch_acc:.2f}%")

    train_time = time.time() - start_time
    print(f"   -> Final Result: Acc = {final_acc:.2f}%")
    return final_acc

def replicate_table_1_complete(dataset_name, manipulate_images=False):
    print("\n" + "="*70)
    print(f"5.2 Visual transformer training on {dataset_name}")
    print("="*70)

    acc_softmax = train_and_evaluate('softmax', dataset_name=dataset_name, manipulate_images=manipulate_images)
    acc_toeplitz = train_and_evaluate('toeplitz', dataset_name=dataset_name, manipulate_images=manipulate_images)
    acc_m_alpha = train_and_evaluate('m_alpha', dataset_name=dataset_name, manipulate_images=manipulate_images)
    acc_grf = train_and_evaluate('grf', dataset_name=dataset_name, n_walks=50, p_halt=0.1, manipulate_images=manipulate_images)
    acc_linear = train_and_evaluate('linear', dataset_name=dataset_name, manipulate_images=manipulate_images)

    print(f"\nCOMPLETE RESULT - {dataset_name}]")
    print(f"{'Method':<25} {'Accuracy':<10}")
    print("-" * 50)
    print(f"{'Unmasked Softmax':<25} {acc_softmax:<10.2f} ")
    print(f"{'Toeplitz-masked Linear':<25} {acc_toeplitz:<10.2f}")
    print(f"{'M_alpha(G)-masked':<25} {acc_m_alpha:<10.2f} ")
    print("-" * 50)
    print(f"{'GRF-masked Linear':<25} {acc_grf:<10.2f}")
    print(f"{'Unmasked Linear':<25} {acc_linear:<10.2f}")
    print("="*70)
    
if __name__ == '__main__':
    datasets = ['cifar10', 'cifar100', 'fashionMNIST', 'MNIST']
    for dataset_name in datasets:
        replicate_table_1_complete(dataset_name)