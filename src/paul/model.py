import torch
import torch.nn as nn
import numpy as np
import networkx as nx

# --- 1. CONFIGURATION ---
BATCH_SIZE = 128
LEARNING_RATE = 1e-3  # Lowered for stability
EPOCHS = 15           # Increased to allow convergence comparison
IMAGE_SIZE = 32
PATCH_SIZE = 4
DIM = 64
DEPTH = 2
NUM_HEADS = 4
MLP_DIM = 128
DROPOUT = 0.1
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

# Graph Parameters
N_WALKS = 50          # More walks = cleaner mask estimate
P_HALT = 0.1          # Higher p_halt = more local mask (paper uses 0.5 often)
MAX_WALK_LEN = 10

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
        # ... (Same generation logic as before) ...
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

        # --- IMPROVEMENT: Symmetric Topological Injection ---
        # Apply mask to BOTH Query and Key to match Eq. 6 (Symmetric Kernel)
        # q' = q + 0.1 * (Mask @ q)
        # k' = k + 0.1 * (Mask @ k)
        
        q_graph = (q.transpose(-2, -1) @ self.mask).transpose(-2, -1)
        k_graph = (k.transpose(-2, -1) @ self.mask).transpose(-2, -1)
        
        # Inject signal
        q = q + 0.1 * q_graph
        k = k + 0.1 * k_graph 

        # --- End Improvement ---

        # Linear Attention Calculation
        linear_kernel = q @ k.transpose(-2, -1)
        
        # We STILL multiply by the mask explicitly here for the "Exact" version
        # to guarantee the topology is enforced.
        masked_kernel = linear_kernel * self.mask.unsqueeze(0).unsqueeze(0)
        
        z = 1 / (masked_kernel.sum(dim=-1, keepdim=True) + self.eps)
        out = (masked_kernel @ v) * z
        out = out.transpose(1, 2).reshape(B, N, C)
        return self.to_out(out)
    
class MAlphaExactAttention(nn.Module):
    """
    M_alpha(G)-masked Linear Attention (Exact O(N^2) implementation).
    Computes M = sum(alpha_k * W^k) explicitly and applies it to QK^T.
    """
    def __init__(self, dim, num_heads, num_patches, max_k=5):
        super().__init__()
        self.num_heads = num_heads
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Linear(dim, dim)
        self.eps = 1e-6
        
        self.max_k = max_k
        # Learnable coefficients for powers of W (k=0 to max_k)
        self.alphas = nn.Parameter(torch.ones(max_k + 1) / (max_k + 1))
        
        # Precompute powers of normalized adjacency matrix
        self.register_buffer('power_matrices', self._generate_power_matrices(num_patches, max_k))

    def _generate_power_matrices(self, N, max_k):
        side = int(np.sqrt(N))
        G = nx.grid_2d_graph(side, side)
        # Ensure consistent ordering
        mapping = {node: i for i, node in enumerate(sorted(list(G.nodes())))}
        G = nx.relabel_nodes(G, mapping)
        
        W = nx.adjacency_matrix(G, nodelist=range(N)).toarray()
        W = torch.tensor(W, dtype=torch.float32)
        
        # Row-normalize (Random Walk Normalization)
        d = W.sum(dim=1, keepdim=True)
        W = W / (d + 1e-6)
        
        powers = []
        curr = torch.eye(N)
        for _ in range(max_k + 1):
            powers.append(curr)
            curr = curr @ W
            
        return torch.stack(powers).to(DEVICE) # Shape (K+1, N, N)

    def feature_map(self, x):
        return torch.nn.functional.relu(x) + self.eps

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.reshape(B, N, self.num_heads, -1).transpose(1, 2), qkv)
        
        q = self.feature_map(q)
        k = self.feature_map(k)
        
        # Compute Mask M = sum(alpha_k * W^k)
        # einsum: k, knm -> nm
        mask = torch.einsum('k, knm -> nm', self.alphas, self.power_matrices)
        
        # Linear Kernel: Q K^T
        linear_kernel = q @ k.transpose(-2, -1)
        
        # Apply Mask (Element-wise)
        masked_kernel = linear_kernel * mask.unsqueeze(0).unsqueeze(0)
        
        # Normalize
        z = 1 / (masked_kernel.sum(dim=-1, keepdim=True) + self.eps)
        out = (masked_kernel @ v) * z
        
        out = out.transpose(1, 2).reshape(B, N, C)
        return self.to_out(out)
    
class ToeplitzLinearAttention(nn.Module):
    """
    O(N log N) Linear Attention with a learnable Toeplitz mask.
    Uses FFT for efficient convolution instead of N^2 matrix multiplication.
    """
    def __init__(self, dim, num_heads, num_patches):
        super().__init__()
        self.num_heads = num_heads
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Linear(dim, dim)
        self.eps = 1e-6
        
        # Learnable Toeplitz filter parameters
        # We need 2*N - 1 parameters to represent a full Toeplitz matrix (diagonals)
        # indices: -(N-1) to (N-1)
        self.n = num_patches
        self.toeplitz_w = nn.Parameter(torch.randn(2 * num_patches - 1))

    def feature_map(self, x):
        return torch.nn.functional.relu(x) + self.eps

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.reshape(B, N, self.num_heads, -1).transpose(1, 2), qkv)
        
        # q, k, v shape: (B, Heads, N, Head_Dim)
        q = self.feature_map(q)
        k = self.feature_map(k)
        
        # We need to compute:
        # Numerator:   Sum_j T_{i-j} * (k_j * v_j)
        # Denominator: Sum_j T_{i-j} * k_j
        
        # 1. Prepare signals for convolution
        # K shape: (B, H, N, D)
        # V shape: (B, H, N, D) -> We treat V as scalar channels here? No.
        # The linear attention formula is: Sum_d Q_id * Sum_j T_ij * K_jd * V_jc
        # We need to convolve (K_d * V_c) with T.
        # This is expensive if we do it for all D*D pairs.
        # Optimization:
        # Let's rewrite: Out_c = Sum_d Q_d * (T * (K_d * V_c))
        
        # Construct signal: (B, H, N, D, D) -> Outer product K_d * V_c
        # This might be memory intensive if Head_Dim is large. 
        # Here Head_Dim = 64/4 = 16. 16*16 = 256 channels. Feasible.
        
        # Signal for Numerator: K unsqueeze(-1) * V unsqueeze(-2) -> (B, H, N, D, D)
        kv_signal = k.unsqueeze(-1) * v.unsqueeze(-2) 
        
        # Signal for Denominator: K (we can treat it as K * 1) -> (B, H, N, D)
        k_signal = k
        
        # Flatten batch and heads for FFT
        # kv_signal: (B*H, N, D*D)
        kv_flat = kv_signal.reshape(B * self.num_heads, N, -1)
        # k_flat: (B*H, N, D)
        k_flat = k_signal.reshape(B * self.num_heads, N, -1)
        
        # Concatenate to do one big FFT
        # combined: (B*H, N, D*D + D)
        combined_signal = torch.cat([kv_flat, k_flat], dim=-1)
        
        # 2. Perform FFT Convolution
        # Pad to 2N to avoid circular convolution artifacts
        fft_len = 2 * N
        
        # FFT of Signal
        # (B*H, 2N, Channels)
        sig_f = torch.fft.rfft(combined_signal, n=fft_len, dim=1)
        
        # FFT of Filter
        # Filter shape: (2N-1). Pad to 2N.
        # We need to be careful with alignment. 
        # Standard convolution T_{i-j} x_j corresponds to filter centered correctly.
        # We'll use the standard trick: Pad filter to 2N, roll if necessary, or just use standard padding.
        # Let's assume 'same' convolution semantics.
        w_padded = torch.zeros(fft_len, device=x.device)
        w_padded[:2*N-1] = self.toeplitz_w
        # Roll to align zero-lag to index 0 for FFT
        w_rolled = torch.roll(w_padded, shifts=-(N-1), dims=0)
        w_f = torch.fft.rfft(w_rolled, n=fft_len)
        
        # Multiply
        out_f = sig_f * w_f.unsqueeze(0).unsqueeze(-1)
        
        # Inverse FFT
        out_time = torch.fft.irfft(out_f, n=fft_len, dim=1)
        
        # Crop to original length N (keep the valid part)
        # With our padding/rolling, the valid result starts at 0
        out_cropped = out_time[:, :N, :]
        
        # 3. Split back into Numerator and Denominator
        dim_sq = k.shape[-1] ** 2
        kv_conv = out_cropped[:, :, :dim_sq].reshape(B, self.num_heads, N, k.shape[-1], k.shape[-1])
        k_conv = out_cropped[:, :, dim_sq:].reshape(B, self.num_heads, N, k.shape[-1])
        
        # 4. Contract with Q
        # Numerator: (Q_d) * (KV_conv_dc) -> Sum over d -> Out_c
        # q: (B, H, N, D)
        # kv_conv: (B, H, N, D, D) (indices: d, c)
        # einsum: bhnd, bhndc -> bhnc
        num = torch.einsum('bhnd,bhndc->bhnc', q, kv_conv)
        
        # Denominator: (Q_d) * (K_conv_d) -> Sum over d
        # k_conv: (B, H, N, D)
        den = torch.einsum('bhnd,bhnd->bhn', q, k_conv)
        den = den.unsqueeze(-1) + self.eps
        
        out = num / den
        
        out = out.transpose(1, 2).reshape(B, N, C)
        return self.to_out(out)
    
    
# --- 4. MODEL & TRAIN for ViT Experiment ---
class ViT(nn.Module):
    def __init__(self, attention_type='softmax', image_size=32, patch_size=4, channels=3, num_classes=10):
        super().__init__()
        self.patch_size = patch_size
        self.channels = channels
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2

        self.patch_embed = nn.Linear(patch_dim, DIM)
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, DIM))
        
        self.layers = nn.ModuleList([])
        for _ in range(DEPTH):
            if attention_type == 'softmax':
                attn = SoftmaxAttention(DIM, NUM_HEADS)
            elif attention_type == 'linear':
                attn = LinearAttention(DIM, NUM_HEADS)
            elif attention_type == 'grf':
                # Use the EXACT simulation module
                attn = GRFExactAttention(DIM, NUM_HEADS, num_patches, N_WALKS, P_HALT)
            elif attention_type == 'toeplitz':
                attn = ToeplitzLinearAttention(DIM, NUM_HEADS, num_patches)
            elif attention_type == 'm_alpha':
                attn = MAlphaExactAttention(DIM, NUM_HEADS, num_patches)
            
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

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(DIM),
            nn.Linear(DIM, num_classes)
        )

    def forward(self, img):
        p = self.patch_size
        # img shape: (B, C, H, W)
        # unfold height: (B, C, H/p, W, p)
        # unfold width: (B, C, H/p, W/p, p, p)
        # reshape: (B, H/p * W/p, C*p*p)
        x = img.unfold(2, p, p).unfold(3, p, p).permute(0, 2, 3, 1, 4, 5).reshape(img.shape[0], -1, self.channels * p * p)
        
        x = self.patch_embed(x)
        B, N, _ = x.shape
        x += self.pos_embed[:, :N]
        for norm1, attn, norm2, mlp in self.layers:
            x = x + attn(norm1(x))
            x = x + mlp(norm2(x))
        return self.mlp_head(x.mean(dim=1))