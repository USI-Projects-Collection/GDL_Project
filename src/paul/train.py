import torch
import torch.optim as optim
from torch import nn
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import DataLoader, random_split
from model import ViT
import time
import copy
import matplotlib.pyplot as plt
import os

BATCH_SIZE = 128
LEARNING_RATE = 1e-3  # Lowered for stability
EPOCHS = 15      

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

def get_dataloaders(dataset_name):
    if dataset_name.lower() == 'cifar10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        full_trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        img_size = 32
        channels = 3
    elif dataset_name.lower() == 'mnist':
        transform = transforms.Compose([
            transforms.Resize((32, 32)), # Resize to 32x32 to match patch logic easily
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        full_trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        img_size = 32
        channels = 1
    else:
        raise ValueError("Unknown dataset")

    # Split training into train and validation (80/20)
    train_size = int(0.8 * len(full_trainset))
    val_size = len(full_trainset) - train_size
    trainset, valset = random_split(full_trainset, [train_size, val_size])

    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    valloader = DataLoader(valset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    return trainloader, valloader, testloader, img_size, channels

def train_and_evaluate(model_type, dataset_name='cifar10'):
    print(f"\n--- Training ViT on {dataset_name.upper()} with {model_type.upper()} Attention ---")
    
    trainloader, valloader, testloader, img_size, channels = get_dataloaders(dataset_name)
    
    model = ViT(attention_type=model_type, image_size=img_size, channels=channels).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    
    best_val_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    
    history = {'loss': [], 'val_acc': []}
    
    start_time = time.time()
    for epoch in range(EPOCHS):
        # --- Training Phase ---
        model.train()
        running_loss = 0.0
        for i, (images, labels) in enumerate(trainloader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        # --- Validation Phase ---
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in valloader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_acc = 100 * val_correct / val_total
        avg_loss = running_loss / len(trainloader)
        
        history['loss'].append(avg_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())

    train_time = time.time() - start_time
    
    # Load best model weights
    model.load_state_dict(best_model_wts)
    print(f"Best Val Acc: {best_val_acc:.2f}%")

    # --- Test Phase ---
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
    return acc, train_time, history

def plot_results(all_histories, all_results, dataset_name):
    # Create plots directory if it doesn't exist
    if not os.path.exists('plots'):
        os.makedirs('plots')

    # 1. Loss and Accuracy Curves
    plt.figure(figsize=(14, 6))
    
    # Loss Plot
    plt.subplot(1, 2, 1)
    for name, hist in all_histories.items():
        plt.plot(hist['loss'], label=name)
    plt.title(f'{dataset_name.upper()} Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Accuracy Plot
    plt.subplot(1, 2, 2)
    for name, hist in all_histories.items():
        plt.plot(hist['val_acc'], label=name)
    plt.title(f'{dataset_name.upper()} Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'plots/{dataset_name}_training_curves.png')
    print(f"Saved training curves to plots/{dataset_name}_training_curves.png")
    
    # 2. Time vs Accuracy Scatter Plot
    plt.figure(figsize=(10, 6))
    for name, (acc, t) in all_results.items():
        plt.scatter(t, acc, label=name, s=100)
        plt.text(t, acc, f' {name}', fontsize=9)
        
    plt.title(f'{dataset_name.upper()} Efficiency: Time vs Accuracy')
    plt.xlabel('Training Time (s)')
    plt.ylabel('Test Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    plt.savefig(f'plots/{dataset_name}_efficiency.png')
    print(f"Saved efficiency plot to plots/{dataset_name}_efficiency.png")

if __name__ == "__main__":
    # print device being used
    print(f"Using device: {DEVICE}")

    datasets_to_test = ['cifar10', 'mnist']
    models_to_test = ['softmax', 'linear', 'grf', 'toeplitz', 'm_alpha']
    
    for dataset in datasets_to_test:
        print(f"\n\n{'='*20} DATASET: {dataset.upper()} {'='*20}")
        results = {}
        histories = {}
        try:
            for model_name in models_to_test:
                acc, t, hist = train_and_evaluate(model_name, dataset_name=dataset)
                results[model_name] = (acc, t)
                histories[model_name] = hist
            
            # Generate plots
            plot_results(histories, results, dataset)
            
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
        
        print("\n" + "="*50)
        print(f"Results for {dataset.upper()}")
        print(f"{'Method':<15} {'Test Acc':<15} {'Time (s)':<15}")
        print("-" * 50)
        for name, (acc, t) in results.items():
            print(f"{name:<15} {acc:<15.2f} {t:<15.2f}")
        print("="*50)