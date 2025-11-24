import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# Experiment Parameters (from Section 4.1)
N_VALUES = [2**i for i in range(6, 13)]  # 64, 128, ..., 4096
D_MODEL = 8     # d
M_FEATURE = 8   # m
N_WALKS = 4     # n (walkers per node)
P_HALT = 0.5    # Termination probability
TRIALS = 10     # Average over 10 seeds for GRF stability

def count_flops_softmax(N, d):
    """
    Theoretical FLOPs for Vanilla Softmax Attention.
    1. Q @ K.T: (N, d) x (d, N) -> (N, N) => 2 * N^2 * d
    2. A @ V:   (N, N) x (N, d) -> (N, d) => 2 * N^2 * d
    Total: ~4 * N^2 * d
    """
    return 4 * (N**2) * d

def count_flops_linear(N, m, d):
    """
    Theoretical FLOPs for Linear Attention.
    1. K.T @ V: (m, N) x (N, d) -> (m, d) => 2 * N * m * d
    2. Q @ KV:  (N, m) x (m, d) -> (N, d) => 2 * N * m * d
    Total: ~4 * N * m * d
    """
    return 4 * N * m * d

def simulate_unique_visits(N, n_walks, p_halt):
    """
    Simulates random walks on a 1D grid graph to count unique visited nodes.
    This is necessary because GRF FLOPs depend on the sparsity (NNZ).
    """
    # 1D Grid Graph: Neighbors of i are i-1 and i+1 (if they exist)
    total_unique_visits = 0
    
    for start_node in range(N):
        visited_in_all_walks = set()
        
        for _ in range(n_walks):
            current = start_node
            # Always visit start node
            visited_in_all_walks.add(current)
            
            while True:
                # Geometric termination
                if np.random.random() < p_halt:
                    break
                
                # Get neighbors (1D grid)
                neighbors = []
                if current > 0: neighbors.append(current - 1)
                if current < N - 1: neighbors.append(current + 1)
                
                if not neighbors:
                    break
                    
                # Pick neighbor uniformly
                current = np.random.choice(neighbors)
                visited_in_all_walks.add(current)
        
        # The number of non-zeros for this node's GRF feature
        total_unique_visits += len(visited_in_all_walks)
        
    return total_unique_visits

def count_flops_grf(N, m, d, n_walks, p_halt, trials):
    """
    Theoretical FLOPs for GRF-Masked Linear Attention.
    
    1. Calculate Sparsity:
       NNZ_total = Sum over nodes (unique_nodes_visited * m)
    
    2. Sparse Matrix Multiplication:
       Att = Phi_Q @ (Phi_K.T @ V)
       Cost = 2 * NNZ * d  (for K part)
            + 2 * NNZ * d  (for Q part)
       Total = 4 * NNZ * d
    """
    avg_unique_visits = 0
    for _ in range(trials):
        avg_unique_visits += simulate_unique_visits(N, n_walks, p_halt)
    avg_unique_visits /= trials
    
    # Total non-zero entries in the sparse feature matrix Phi
    # Each unique visited node contributes 'm' non-zeros due to outer product
    total_nnz = avg_unique_visits * m
    
    return 4 * total_nnz * d

def run_experiment():
    print(f"{'N':<6} {'Softmax':<12} {'Linear':<12} {'GRF (Ours)':<12}")
    print("-" * 45)

    softmax_flops = []
    linear_flops = []
    grf_flops = []

    for N in N_VALUES:
        # 1. Softmax
        s_ops = count_flops_softmax(N, D_MODEL)
        softmax_flops.append(s_ops)
        
        # 2. Linear
        l_ops = count_flops_linear(N, M_FEATURE, D_MODEL)
        linear_flops.append(l_ops)
        
        # 3. GRF
        g_ops = count_flops_grf(N, M_FEATURE, D_MODEL, N_WALKS, P_HALT, TRIALS)
        grf_flops.append(g_ops)
        
        print(f"{N:<6} {s_ops:<12.2e} {l_ops:<12.2e} {g_ops:<12.2e}")

    # --- Plotting to match Figure 3 ---
    plt.figure(figsize=(8, 6))
    
    # Log-Log Plot
    plt.loglog(N_VALUES, softmax_flops, 's-', color='#d62728', label='Softmax', linewidth=2, markersize=8)
    plt.loglog(N_VALUES, linear_flops, 'o-', color='black', label='Linear', linewidth=2, markersize=8)
    plt.loglog(N_VALUES, grf_flops, 'x-', color='#1f77b4', label='Linear + GRFs', linewidth=2, markersize=8)

    # Styling
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.xlabel('Number of graph nodes, $N$', fontsize=12)
    plt.ylabel('FLOPs', fontsize=12)
    plt.title('Time Complexity Scaling (Replication of Fig. 3)', fontsize=14)
    plt.legend(fontsize=12)
    
    # Adjust axes to match paper aesthetic
    plt.xticks(N_VALUES, labels=[str(n) for n in N_VALUES])
    plt.minorticks_off()
    
    plt.tight_layout()
    plt.savefig('replication_figure_3.png', dpi=300)
    print("\nPlot saved to 'replication_figure_3.png'")

if __name__ == "__main__":
    run_experiment()