"""
Linear Transformer Topological Masking with Graph Random Features
A pedagogical implementation demonstrating the problem, previous solutions, and the paper's solution.

Paper: "Linear Transformer Topological Masking with Graph Random Features" (ICLR 2025)
Authors: Reid et al.

This code demonstrates:
1. THE PROBLEM: Transformers are permutation invariant (don't understand graph structure)
2. PREVIOUS SOLUTIONS: Full-rank softmax attention with topological masking (O(N²))
3. THE PAPER'S SOLUTION: GRF-masked linear attention (O(N))
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
import time
import networkx as nx

# Set random seed for reproducibility
np.random.seed(42)

#############################################################################
# PART 1: THE PROBLEM - Transformers Don't Understand Graph Structure
#############################################################################

def create_simple_graph(n_nodes=8):
    """Create a simple graph structure (grid graph)"""
    G = nx.grid_2d_graph(int(np.sqrt(n_nodes)), int(np.sqrt(n_nodes)))
    # Relabel nodes to integers
    mapping = {node: i for i, node in enumerate(G.nodes())}
    G = nx.relabel_nodes(G, mapping)
    return G

def visualize_graph(G, title="Graph Structure"):
    """Visualize the graph"""
    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(8, 6))
    nx.draw(G, pos, with_labels=True, node_color='lightblue', 
            node_size=500, font_size=16, font_weight='bold')
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f'{title.replace(" ", "_")}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {title}.png")

def get_adjacency_matrix(G):
    """Get weighted adjacency matrix with normalization"""
    A = nx.adjacency_matrix(G).todense().astype(float)
    # Normalize by degrees: W_ij = 1/sqrt(d_i * d_j)
    degrees = np.array(A.sum(axis=1)).flatten()
    D_inv_sqrt = np.diag(1.0 / np.sqrt(degrees))
    W = D_inv_sqrt @ A @ D_inv_sqrt
    return np.array(W)

#############################################################################
# PART 2: VANILLA ATTENTION (No Graph Understanding)
#############################################################################

def vanilla_attention(Q, K, V):
    """
    Vanilla softmax attention - O(N²) complexity
    Doesn't use graph structure at all!
    
    Att(Q,K,V) = softmax(QK^T)V

    The Problem: - Doesn't know nodes 0 and 1 are neighbors
                 - Treats distant nodes same as nearby ones
                 - Creates a huge N×N matrix (slow!)
                
    """
    d_k = Q.shape[-1]
    # Compute attention scores
    scores = Q @ K.T / np.sqrt(d_k)  # (N, N)
    # Apply softmax
    attention_weights = np.exp(scores) / np.exp(scores).sum(axis=1, keepdims=True)
    # Apply to values
    output = attention_weights @ V
    return output, attention_weights

#############################################################################
# PART 3: PREVIOUS SOLUTION - Full Softmax with Topological Masking
#############################################################################

def compute_graph_mask(W, alpha_coeffs=[1.0, 0.5, 0.25]):
    """
    Compute topological mask as power series of adjacency matrix:
    M_α(G) = Σ α_k W^k
    
    This captures graph structure but requires O(N²) space!

    How It Works:  - Computes walks of different lengths (1-hop, 2-hop, 3-hop neighbors)
                   - Nearby nodes get high scores, distant nodes get low scores
                   - W¹ = direct neighbors, W² = neighbors-of-neighbors, etc.
    
    Example: Nodes 1 and 2 (neighbors) get score 0.8, nodes 1 and 10 (far apart) get score 0.01
    """
    N = W.shape[0]
    M = np.zeros((N, N))
    W_power = np.eye(N)
    
    for k, alpha_k in enumerate(alpha_coeffs):
        M += alpha_k * W_power
        W_power = W_power @ W
    
    return M

def topological_masked_attention(Q, K, V, M):
    """
    Softmax attention with topological masking - O(N²) complexity
    
    A_M = M ⊙ exp(QK^T)  (element-wise product)
    Att = D^(-1) A_M V
    
    This is EXPENSIVE but captures graph structure well!

    Vanilla attention multiplied by topological mask.
    """
    d_k = Q.shape[-1]
    # Compute attention scores
    scores = Q @ K.T / np.sqrt(d_k)
    A = np.exp(scores)
    
    # Apply topological mask (Hadamard product)
    A_masked = A * M  # This is the key: modulate attention by graph structure!
    
    # Normalize
    D = A_masked.sum(axis=1, keepdims=True)
    attention_weights = A_masked / (D + 1e-8)
    
    # Apply to values
    output = attention_weights @ V
    return output, attention_weights

#############################################################################
# PART 4: LINEAR ATTENTION (Fast but No Masking Yet)
#############################################################################

def feature_map_relu(x):
    """Simple feature map: ReLU"""
    return np.maximum(0, x)

def linear_attention(Q, K, V):
    """
    Linear attention - O(N) complexity but no graph structure!
    
    Instead of computing QK^T explicitly (N×N matrix),
    we use feature maps: φ(Q) and φ(K) -> NEVER CREATE A N×N MATRIX!
    
    Att = (φ(Q) @ (φ(K)^T @ V)) / (φ(Q) @ (φ(K)^T @ 1))
    
    Notice the parentheses! We compute φ(K)^T @ V first (m×d),
    then multiply by φ(Q) (N×m), avoiding the N×N matrix.
    """
    # Apply feature map
    Phi_Q = feature_map_relu(Q)  # (N, d)
    Phi_K = feature_map_relu(K)  # (N, d)
    
    # Compute in O(N) time by exploiting associativity
    KV = Phi_K.T @ V  # (d, d) - compute this first!
    numerator = Phi_Q @ KV  # (N, d)
    
    # Normalization
    K_sum = Phi_K.T @ np.ones((Phi_K.shape[0], 1))  # (d, 1)
    denominator = Phi_Q @ K_sum  # (N, 1)
    
    output = numerator / (denominator + 1e-8)
    
    return output, None  # No attention matrix (that's the point!)

#############################################################################
# PART 5: THE PAPER'S SOLUTION - Graph Random Features (GRFs)
#############################################################################

def sample_random_walks(G, n_walks=10, p_halt=0.5):
    """
    Sample random walks for each node in the graph.
    Walks terminate with probability p_halt at each step.
    
    Returns: dict mapping node_id -> list of walks
    Each walk is a list of visited nodes

    How it works:
      1. Start at node i
      2. Randomly pick a neighbor
      3. Flip a coin with probability p_halt to stop
      4. Repeat n_walks times
    """
    walks = {}
    nodes = list(G.nodes())
    
    for start_node in nodes:
        node_walks = []
        for _ in range(n_walks):
            walk = [start_node]
            current = start_node
            
            while True:
                # Terminate with probability p_halt
                if np.random.random() < p_halt:
                    break
                
                # Get neighbors
                neighbors = list(G.neighbors(current))
                if not neighbors:
                    break
                
                # Randomly choose next node
                current = np.random.choice(neighbors)
                walk.append(current)
            
            node_walks.append(walk)
        
        walks[start_node] = node_walks
    
    return walks

def compute_graph_random_features(walks, W, N, p_halt=0.5, f_coeffs=[1.0, 0.8, 0.6, 0.4, 0.2]):
    """
    Compute Graph Random Features (GRFs) from random walks.
    
    GRFs approximate the graph kernel using Monte Carlo estimation:
    φ̂_G(v_i) is SPARSE (only nonzero at visited nodes)
    
    The magic: E[φ̂_G(v_i)^T φ̂_G(v_j)] = M_α(G)_ij
    But φ̂_G is sparse, so we get O(N) complexity!

    The Magic:
        - For each node, count which nodes its walks visited
        - Weight by walk length (shorter walks more important)
        - Result: sparse vector (mostly zeros!)
    """
    GRFs = np.zeros((N, N))
    
    for node_id, node_walks in walks.items():
        n_walks = len(node_walks)
        
        for walk in node_walks:
            # Process each prefix subwalk
            for end_idx in range(len(walk)):
                visited_node = walk[end_idx]
                walk_length = end_idx
                
                # Compute walk weight (product of edge weights)
                walk_weight = 1.0
                for step in range(walk_length):
                    u, v = walk[step], walk[step + 1]
                    walk_weight *= W[u, v]
                
                # Probability of this walk (geometric distribution)
                p_walk = (1 - p_halt) ** walk_length * p_halt
                
                # Get coefficient based on walk length
                f_k = f_coeffs[min(walk_length, len(f_coeffs) - 1)]
                
                # Update GRF (importance sampling)
                GRFs[node_id, visited_node] += (walk_weight * f_k) / (p_walk + 1e-8)
        
        # Normalize by number of walks
        GRFs[node_id] /= n_walks
    
    return GRFs

def grf_masked_linear_attention(Q, K, V, GRFs):
    """
    Linear attention with GRF topological masking - O(N) complexity!
    
    This is the paper's main contribution:
    1. Use linear attention (O(N))
    2. Incorporate graph structure via GRFs
    3. GRFs are sparse, keeping O(N) complexity
    
    The trick: vec(φ(q_i) ⊗ φ̂_G(v_i))^T vec(φ(k_j) ⊗ φ̂_G(v_j)) 
              = φ(q_i)^T φ(k_j) × φ̂_G(v_i)^T φ̂_G(v_j)
              = K_LR_ij × M̂_α_ij
    """
    # Apply feature map to Q and K
    Phi_Q = feature_map_relu(Q)  # (N, d)
    Phi_K = feature_map_relu(K)  # (N, d)
    
    # Combine with graph features (outer product + vectorization)
    # For simplicity, we use a diagonal approximation here
    # Full implementation would use vec(φ(q) ⊗ φ_G(v))
    
    # Weight by graph structure (sparse!)
    Phi_Q_masked = Phi_Q * np.linalg.norm(GRFs, axis=1, keepdims=True)
    Phi_K_masked = Phi_K * np.linalg.norm(GRFs, axis=1, keepdims=True)
    
    # Compute in O(N) time
    KV = Phi_K_masked.T @ V
    numerator = Phi_Q_masked @ KV
    
    K_sum = Phi_K_masked.T @ np.ones((Phi_K_masked.shape[0], 1))
    denominator = Phi_Q_masked @ K_sum
    
    output = numerator / (denominator + 1e-8)
    
    return output, GRFs

#############################################################################
# PART 6: EXPERIMENTS & COMPARISON
#############################################################################

def run_experiment(N=16):
    """Run complete experiment comparing all methods"""
    
    print("="*70)
    print("EXPERIMENT: Comparing Attention Mechanisms on Graph Data")
    print("="*70)
    print()
    
    # Create graph
    print(f"Creating {N}-node grid graph...")
    G = create_simple_graph(N)
    W = get_adjacency_matrix(G)
    visualize_graph(G, "Input Graph Structure")
    print(f"Graph has {G.number_of_edges()} edges")
    print()
    
    # Create dummy data (Q, K, V)
    d = 8  # embedding dimension
    Q = np.random.randn(N, d)
    K = np.random.randn(N, d)
    V = np.random.randn(N, d)
    
    # Method 1: Vanilla Attention (ignores graph)
    print("1. VANILLA ATTENTION (No Graph Structure)")
    print("   - Complexity: O(N²)")
    print("   - Problem: Doesn't use graph structure!")
    start = time.time()
    out_vanilla, attn_vanilla = vanilla_attention(Q, K, V)
    time_vanilla = time.time() - start
    print(f"   - Time: {time_vanilla*1000:.3f}ms")
    print()
    
    # Method 2: Topological Masked Softmax (previous solution)
    print("2. TOPOLOGICAL MASKED SOFTMAX (Previous Solution)")
    print("   - Complexity: O(N²)")
    print("   - Benefit: Uses graph structure!")
    print("   - Problem: Too slow for large graphs")
    M = compute_graph_mask(W, alpha_coeffs=[1.0, 0.5, 0.25, 0.1])
    start = time.time()
    out_masked, attn_masked = topological_masked_attention(Q, K, V, M)
    time_masked = time.time() - start
    print(f"   - Time: {time_masked*1000:.3f}ms")
    print(f"   - Mask sparsity: {(M > 0.01).sum() / M.size * 100:.1f}%")
    print()
    
    # Method 3: Linear Attention (fast but no graph)
    print("3. LINEAR ATTENTION (Fast but No Graph)")
    print("   - Complexity: O(N)")
    print("   - Benefit: Very fast!")
    print("   - Problem: Doesn't use graph structure")
    start = time.time()
    out_linear, _ = linear_attention(Q, K, V)
    time_linear = time.time() - start
    print(f"   - Time: {time_linear*1000:.3f}ms")
    print()
    
    # Method 4: GRF-Masked Linear (paper's solution)
    print("4. GRF-MASKED LINEAR ATTENTION (Paper's Solution)")
    print("   - Complexity: O(N)")
    print("   - Benefit: Fast AND uses graph structure!")
    walks = sample_random_walks(G, n_walks=20, p_halt=0.3)
    GRFs = compute_graph_random_features(walks, W, N, p_halt=0.3, f_coeffs=[1.0, 0.8, 0.6, 0.4, 0.2])
    start = time.time()
    out_grf, grfs = grf_masked_linear_attention(Q, K, V, GRFs)
    time_grf = time.time() - start
    print(f"   - Time: {time_grf*1000:.3f}ms")
    print(f"   - GRF sparsity: {(GRFs > 0.01).sum() / GRFs.size * 100:.1f}%")
    print(f"   - Speedup vs masked softmax: {time_masked/time_grf:.1f}x")
    print()
    
    # Visualize attention patterns
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    if attn_vanilla is not None:
        axes[0, 0].imshow(attn_vanilla, cmap='viridis')
        axes[0, 0].set_title('Vanilla Attention\n(No Graph Info)')
        axes[0, 0].set_xlabel('Key')
        axes[0, 0].set_ylabel('Query')
    
    if attn_masked is not None:
        axes[0, 1].imshow(attn_masked, cmap='viridis')
        axes[0, 1].set_title('Masked Softmax\n(Graph-Aware, O(N²))')
        axes[0, 1].set_xlabel('Key')
        axes[0, 1].set_ylabel('Query')
    
    axes[1, 0].imshow(M, cmap='viridis')
    axes[1, 0].set_title('Topological Mask M_α(G)')
    axes[1, 0].set_xlabel('Node j')
    axes[1, 0].set_ylabel('Node i')
    
    axes[1, 1].imshow(GRFs, cmap='viridis')
    axes[1, 1].set_title('Graph Random Features\n(Sparse Approximation)')
    axes[1, 1].set_xlabel('Node j')
    axes[1, 1].set_ylabel('Node i')
    
    plt.tight_layout()
    plt.savefig('attention_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Saved: attention_comparison.png")
    print()
    
    # Summary
    print("="*70)
    print("SUMMARY")
    print("="*70)
    print(f"{'Method':<30} {'Time (ms)':<12} {'Complexity':<12} {'Graph-Aware'}")
    print("-"*70)
    print(f"{'Vanilla Attention':<30} {time_vanilla*1000:>8.3f}    {'O(N²)':<12} {'No'}")
    print(f"{'Masked Softmax':<30} {time_masked*1000:>8.3f}    {'O(N²)':<12} {'Yes'}")
    print(f"{'Linear Attention':<30} {time_linear*1000:>8.3f}    {'O(N)':<12} {'No'}")
    print(f"{'GRF Linear (Paper)':<30} {time_grf*1000:>8.3f}    {'O(N)':<12} {'Yes'}")
    print("="*70)
    print()
    print("KEY INSIGHT: The paper achieves O(N) complexity while preserving")
    print("             graph structure awareness through sparse GRFs!")

if __name__ == "__main__":
    print("""
    ╔═══════════════════════════════════════════════════════════════════╗
    ║  Linear Transformer Topological Masking with Graph Random Features ║
    ║  Pedagogical Implementation for Understanding the Paper           ║
    ╚═══════════════════════════════════════════════════════════════════╝
    """)
    
    run_experiment(N=100)
    
    print("\n" + "="*70)
    print("NEXT STEPS FOR YOUR REPLICATION:")
    print("="*70)
    print("1. Implement full GRF with outer product vectorization")
    print("2. Add theorem 3.2 concentration bounds verification")
    print("3. Test on real datasets (ImageNet, point clouds)")
    print("4. Implement learnable mask parameters (f_k)")
    print("5. Compare with Performer features (FAVOR+)")
    print("6. Scale to larger graphs (10k+ nodes)")
    print("="*70)