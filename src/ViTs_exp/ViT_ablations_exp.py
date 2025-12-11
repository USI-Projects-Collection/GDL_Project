from train import train_and_evaluate
import matplotlib as plt
import torch

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

def run_ablation_study():
    print("\n" + "="*60)
    print("5.2 ABLATION STUDY (Replicating Fig. 6)")
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
    plt.savefig('combined_ablation_plot-2.png', dpi=300)
    print("\nAblation plot saved to 'combined_ablation_plot-2.png'")

if __name__ == "__main__":
    print(f"Using Device: {DEVICE}")
    run_ablation_study()