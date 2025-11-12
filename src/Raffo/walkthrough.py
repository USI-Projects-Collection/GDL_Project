"""
DYNAMIC GRAPH EXAMPLE: Understanding Attention Mechanisms

This script demonstrates 4 attention mechanisms on a line graph
of configurable size (default n_nodes = 4).

The graph is simple:
    0 --- 1 --- 2 --- ... --- (n-1)

Each method shows how attention behaves on this graph.
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

np.random.seed(42)

# ============================================================================
# CONFIGURATION
# ============================================================================
n_nodes = 4  # 👈 Change this to use any graph size (e.g. 4, 6, 10)
print("="*80)
print(f"GRAPH WALKTHROUGH: {n_nodes}-Node Line Graph — 4 Attention Mechanisms")
print("="*80)
print()

# ============================================================================
# STEP 1: CREATE A LINE GRAPH
# ============================================================================
print(f"STEP 1: Creating a {n_nodes}-node line graph")
print("-" * 80)

G = nx.path_graph(n_nodes)
print("Graph structure:")
print("  " + " --- ".join(map(str, range(n_nodes))))
print()
print("Edges:", list(G.edges()))
for i in range(n_nodes):
    print(f"Node {i} neighbors:", list(G.neighbors(i)))
print()

A = nx.adjacency_matrix(G).todense().astype(float)
degrees = np.array(A.sum(axis=1)).flatten()
D_inv_sqrt = np.diag(1.0 / np.sqrt(degrees))
W = D_inv_sqrt @ A @ D_inv_sqrt
print("Normalized adjacency W:")
print(np.round(W, 3))
print()

# ============================================================================
# STEP 2: CREATE DUMMY Q, K, V MATRICES
# ============================================================================
print("\n" + "="*80)
print("STEP 2: Creating Query, Key, Value matrices")
print("-" * 80)

x = np.linspace(1.0, 0.4, n_nodes)
Q = np.stack([x, 1 - x], axis=1)
K = np.stack([x * 0.9 + 0.1, 1 - x * 0.9 - 0.1], axis=1)
V = np.stack([np.linspace(1.0, 0.5, n_nodes),
              np.linspace(0.0, 1.0, n_nodes)], axis=1)

print("Q (queries):")
print(Q)
print("\nK (keys):")
print(K)
print("\nV (values):")
print(V)
print()

# ============================================================================
# METHOD 1: VANILLA ATTENTION (NO GRAPH STRUCTURE)
# ============================================================================
print("\n" + "="*80)
print("METHOD 1: VANILLA ATTENTION (Ignores Graph Structure)")
print("="*80)

scores_vanilla = Q @ K.T
exp_scores = np.exp(scores_vanilla)
attn_vanilla = exp_scores / exp_scores.sum(axis=1, keepdims=True)
output_vanilla = attn_vanilla @ V

print("Attention weights (Vanilla):")
print(np.round(attn_vanilla, 3))
print()

# ============================================================================
# METHOD 2: MASKED SOFTMAX (GRAPH-AWARE)
# ============================================================================
print("\n" + "="*80)
print("METHOD 2: MASKED SOFTMAX ATTENTION (Graph-Aware, O(N²))")
print("="*80)

I = np.eye(n_nodes)
W2 = W @ W
M = 1.0 * I + 0.5 * W + 0.2 * W2
scores_masked = Q @ K.T
A_masked = np.exp(scores_masked) * M
attn_masked = A_masked / A_masked.sum(axis=1, keepdims=True)
output_masked = attn_masked @ V

print("Topological mask M:")
print(np.round(M, 3))
print("\nAttention weights (Masked):")
print(np.round(attn_masked, 3))
print()

# ============================================================================
# METHOD 3: LINEAR ATTENTION (FAST BUT NO GRAPH)
# ============================================================================
print("\n" + "="*80)
print("METHOD 3: LINEAR ATTENTION (Fast O(N) but No Graph)")
print("="*80)

Phi_Q = np.maximum(0, Q)
Phi_K = np.maximum(0, K)
KV = Phi_K.T @ V
numerator = Phi_Q @ KV
K_sum = Phi_K.T @ np.ones((n_nodes, 1))
denominator = Phi_Q @ K_sum
output_linear = numerator / denominator

print("Output (Linear Attention):")
print(np.round(output_linear, 3))
print()

# ============================================================================
# METHOD 4: GRF-MASKED LINEAR (FAST + GRAPH-AWARE)
# ============================================================================
print("\n" + "="*80)
print("METHOD 4: GRF-MASKED LINEAR ATTENTION (Fast O(N) + Graph-Aware)")
print("="*80)

n_walks = 20
p_halt = 0.3
f_coeffs = [1.0, 0.8, 0.6, 0.4]

walks = {}
for start_node in range(n_nodes):
    walks[start_node] = []
    for _ in range(n_walks):
        walk = [start_node]
        current = start_node
        while True:
            if np.random.random() < p_halt:
                break
            neighbors = list(G.neighbors(current))
            if not neighbors:
                break
            current = np.random.choice(neighbors)
            walk.append(current)
        walks[start_node].append(walk)

GRFs = np.zeros((n_nodes, n_nodes))
for node_id, node_walks in walks.items():
    for walk in node_walks:
        for end_idx in range(len(walk)):
            visited_node = walk[end_idx]
            walk_length = end_idx
            walk_weight = 1.0
            for step in range(walk_length):
                u, v = walk[step], walk[step + 1]
                walk_weight *= W[u, v]
            p_walk = (1 - p_halt) ** walk_length * p_halt
            f_k = f_coeffs[min(walk_length, len(f_coeffs) - 1)]
            GRFs[node_id, visited_node] += (walk_weight * f_k) / (p_walk + 1e-8)
    GRFs[node_id] /= n_walks

grf_weights = np.linalg.norm(GRFs, axis=1, keepdims=True)
Phi_Q_masked = Phi_Q * grf_weights
Phi_K_masked = Phi_K * grf_weights
KV_grf = Phi_K_masked.T @ V
numerator_grf = Phi_Q_masked @ KV_grf
K_sum_grf = Phi_K_masked.T @ np.ones((n_nodes, 1))
denominator_grf = Phi_Q_masked @ K_sum_grf
output_grf = numerator_grf / denominator_grf

print("Graph Random Features (GRFs):")
print(np.round(GRFs, 3))
print("\nOutput with GRF masking:")
print(np.round(output_grf, 3))
print()

# ============================================================================
# VISUALIZATION
# ============================================================================
print("="*80)
print("FINAL COMPARISON: All 4 Methods")
print("="*80)
print()

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
pos = {i: (i, 0) for i in range(n_nodes)}

# Plot 1: Graph
nx.draw(G, pos, ax=axes[0, 0], with_labels=True, node_color='lightblue',
        node_size=800, font_size=14, font_weight='bold')
axes[0, 0].set_title(f'Graph Structure\n({n_nodes}-node line)', fontsize=14, fontweight='bold')

# Plot 2: Vanilla
im = axes[0, 1].imshow(attn_vanilla, cmap='YlOrRd', vmin=0, vmax=np.max(attn_vanilla))
axes[0, 1].set_title('Vanilla Attention\n(No Graph Info)', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('Key')
axes[0, 1].set_ylabel('Query')
plt.colorbar(im, ax=axes[0, 1])

# Plot 3: Masked
im = axes[0, 2].imshow(attn_masked, cmap='YlOrRd', vmin=0, vmax=np.max(attn_masked))
axes[0, 2].set_title('Masked Softmax\n(Graph-Aware)', fontsize=14, fontweight='bold')
axes[0, 2].set_xlabel('Key')
axes[0, 2].set_ylabel('Query')
plt.colorbar(im, ax=axes[0, 2])

# Plot 4: Mask M
im = axes[1, 0].imshow(M, cmap='YlOrRd', vmin=0, vmax=np.max(M))
axes[1, 0].set_title('Topological Mask M', fontsize=14, fontweight='bold')
axes[1, 0].set_xlabel('Node j')
axes[1, 0].set_ylabel('Node i')
plt.colorbar(im, ax=axes[1, 0])

# Plot 5: GRFs
im = axes[1, 1].imshow(GRFs, cmap='YlOrRd', vmin=0, vmax=np.max(GRFs))
axes[1, 1].set_title('Graph Random Features', fontsize=14, fontweight='bold')
axes[1, 1].set_xlabel('Node j')
axes[1, 1].set_ylabel('Node i')
plt.colorbar(im, ax=axes[1, 1])

# Plot 6: Output comparison
methods = ['Vanilla', 'Masked', 'Linear', 'GRF']
outputs = np.array([
    output_vanilla[0],
    output_masked[0],
    output_linear[0],
    output_grf[0]
])
x = np.arange(len(methods))
width = 0.35
axes[1, 2].bar(x - width/2, outputs[:, 0], width, label='Dim 1')
axes[1, 2].bar(x + width/2, outputs[:, 1], width, label='Dim 2')
axes[1, 2].set_xticks(x)
axes[1, 2].set_xticklabels(methods)
axes[1, 2].set_ylabel('Output Value')
axes[1, 2].legend()
axes[1, 2].set_title('Node 0 Output Comparison', fontsize=14, fontweight='bold')
axes[1, 2].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('dynamic_graph_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Saved: dynamic_graph_comparison.png")
print()

# ============================================================================
# SUMMARY
# ============================================================================
print("="*80)
print(f"SUMMARY TABLE (Node 0’s Attention — first {min(5, n_nodes)} columns)")
print("="*80)
header = "Method".ljust(20)
for j in range(min(5, n_nodes)):
    header += f"Node {j:<8}"
print(header)
print("-"*80)
print("Vanilla".ljust(20) + " ".join(f"{attn_vanilla[0, j]:>8.2%}" for j in range(min(5, n_nodes))))
print("Masked Softmax".ljust(20) + " ".join(f"{attn_masked[0, j]:>8.2%}" for j in range(min(5, n_nodes))))
print("="*80)
print()