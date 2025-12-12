import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# --- 1. PARAMETERS TO MATCH PAPER EXACTLY ---
# Range: 2^0 (1) to 2^12 (4096)
N_VALUES = [2**i for i in range(0, 13)]  
D_MODEL = 8     
M_FEATURE = 8   
N_WALKS = 4     
P_HALT = 0.5    
TRIALS = 10     

# --- SCALING FACTOR ---
SCALE_FACTOR = 1e6 

def count_flops_softmax(N, d):
    # Total: ~4 * N^2 * d
    raw_flops = 4 * (N**2) * d
    return raw_flops / SCALE_FACTOR

def count_flops_linear(N, m, d):
    # Total: ~4 * N * m * d
    raw_flops = 4 * N * m * d
    return raw_flops / SCALE_FACTOR

def simulate_unique_visits(N, n_walks, p_halt):
    total_unique_visits = 0
    for start_node in range(N):
        visited_in_all_walks = set()
        visited_in_all_walks.add(start_node) # Always visit self
        
        for _ in range(n_walks):
            current = start_node
            while True:
                if np.random.random() < p_halt:
                    break
                # 1D Grid neighbors
                neighbors = []
                if current > 0: neighbors.append(current - 1)
                if current < N - 1: neighbors.append(current + 1)
                if not neighbors: break
                current = np.random.choice(neighbors)
                visited_in_all_walks.add(current)
        
        total_unique_visits += len(visited_in_all_walks)
    return total_unique_visits

def count_flops_grf(N, m, d, n_walks, p_halt, trials):
    avg_unique_visits = 0
    for _ in range(trials):
        avg_unique_visits += simulate_unique_visits(N, n_walks, p_halt)
    avg_unique_visits /= trials
    
    total_nnz = avg_unique_visits * m
    raw_flops = 4 * total_nnz * d
    return raw_flops / SCALE_FACTOR

def run_experiment():
    print(f"{'N':<6} {'Softmax':<12} {'Linear':<12} {'GRF (Ours)':<12} (All / 10^6)")
    print("-" * 55)

    softmax_flops = []
    linear_flops = []
    grf_flops = []

    for N in N_VALUES:
        s_ops = count_flops_softmax(N, D_MODEL)
        l_ops = count_flops_linear(N, M_FEATURE, D_MODEL)
        g_ops = count_flops_grf(N, M_FEATURE, D_MODEL, N_WALKS, P_HALT, TRIALS)
        
        softmax_flops.append(s_ops)
        linear_flops.append(l_ops)
        grf_flops.append(g_ops)
        
        print(f"{N:<6} {s_ops:<12.2e} {l_ops:<12.2e} {g_ops:<12.2e}")

    # --- PLOTTING ---
    plt.figure(figsize=(8, 6))
    
    plt.loglog(N_VALUES, softmax_flops, 's-', color='#d62728', label='Softmax', linewidth=2, markersize=8)
    plt.loglog(N_VALUES, linear_flops, 'o-', color='black', label='Linear', linewidth=2, markersize=8)
    plt.loglog(N_VALUES, grf_flops, 'x-', color='#1f77b4', label='Linear + GRFs', linewidth=2, markersize=8)

    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.xlabel('Number of graph nodes, $N$', fontsize=12)
    plt.ylabel('FLOPs ($/10^6$)', fontsize=12)
    plt.title('Time Complexity Scaling', fontsize=14)
    plt.legend(fontsize=12)
    
    # Set X-Axis to 2^0 to 2^12
    plt.xlim(1, 4096) 
    
    # Set Y-Axis to 10^-5 to 10^3
    plt.ylim(1e-5, 1e3)
    
    # Optional: Set ticks explicitly to match the paper's powers of 2
    plt.xticks(N_VALUES[::2], labels=[str(n) for n in N_VALUES[::2]]) # Show every other power of 2
    
    plt.tight_layout()
    plt.savefig('replication_figure_3_exact.png', dpi=300)
    print("\nPlot saved to 'eplication_figure_3_exact.png'")

if __name__ == "__main__":
    run_experiment()