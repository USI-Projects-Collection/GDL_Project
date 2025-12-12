import torch
import numpy as np
import os
import sys
from train import UnifiedInterlacer

# --- CONFIG ---
DATA_PATH = "./data/"
MODEL_MODE = 'grf'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
K_NEIGHBORS = 6

def compute_knn_torch(points, k):
    """
    Calcola KNN brute-force su GPU. 
    Input: (1, N, 3) -> Output: (1, N, K)
    """
    # points: (B, N, 3)
    B, N, D = points.shape
    # Distanza Euclidea al quadrato: (x-y)^2 = x^2 + y^2 - 2xy
    # x^2: (B, N, 1)
    x_sq = (points ** 2).sum(dim=-1, keepdim=True)
    # y^2 (transpose): (B, 1, N)
    y_sq = x_sq.transpose(1, 2)
    # 2xy: (B, N, N)
    xy = torch.bmm(points, points.transpose(1, 2))
    
    dist_mat = x_sq + y_sq - 2 * xy
    
    # topk returns values and indices. We want indices of smallest distances.
    # largest=False prende i più piccoli.
    # k+1 perché il più vicino è se stesso (distanza 0)
    _, knn_indices = torch.topk(dist_mat, k=k+1, dim=-1, largest=False)
    
    # Rimuoviamo il primo (se stesso)
    return knn_indices[:, :, 1:]

def main():
    print(f"Loading model_{MODEL_MODE}.pth ...")
    
    # FIX: weights_only=False permette di caricare numpy arrays (mean/std) salvati nel dizionario
    checkpoint = torch.load(f"model_{MODEL_MODE}.pth", map_location=DEVICE, weights_only=False)
    
    # 1. Setup Modello
    model = UnifiedInterlacer(mode=checkpoint['mode']).to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Statistiche di normalizzazione salvate
    mean = torch.tensor(checkpoint['mean']).to(DEVICE)
    std = torch.tensor(checkpoint['std']).to(DEVICE)
    
    # 2. Carichiamo la Ground Truth (Solo per il primo frame e confronto)
    gt_points_np = np.load("points.npy")
    num_frames = gt_points_np.shape[0]
    
    print(f"Generating rollout for {num_frames} frames in CLOSED LOOP...")
    
    # 3. Closed Loop Inference
    # Partiamo dal Frame 0 Reale
    current_frame = torch.from_numpy(gt_points_np[0]).unsqueeze(0).to(DEVICE) # (1, N, 3)
    # Normalizziamo subito
    current_frame = (current_frame - mean) / (std + 1e-6)
    
    predictions = []
    
    # Aggiungiamo il frame 0 denormalizzato alla lista (è l'inizio)
    start_frame_denorm = (current_frame * (std + 1e-6) + mean).cpu().numpy()
    predictions.append(start_frame_denorm.squeeze(0))
    
    with torch.no_grad():
        for t in range(num_frames - 1):
            # A. Calcoliamo il KNN sui punti ATTUALI (che potrebbero essere predetti)
            # Questo è cruciale: il grafo evolve con la predizione fisica
            knn_indices = compute_knn_torch(current_frame, K_NEIGHBORS)
            
            # B. Predizione
            next_frame_norm = model(current_frame, knn_indices)
            
            # C. Aggiornamento stato (Closed Loop: l'output diventa l'input)
            current_frame = next_frame_norm
            
            # D. Salvataggio (Denormalizziamo per salvare)
            frame_denorm = (next_frame_norm * (std + 1e-6) + mean).cpu().numpy()
            predictions.append(frame_denorm.squeeze(0))
            
            if t % 50 == 0:
                print(f"Generated frame {t}/{num_frames}")

    # 4. Salvataggio finale
    pred_array = np.array(predictions) # (F, N, 3)
    save_name = f"rollout_{MODEL_MODE}.npy"
    np.save(save_name, pred_array)
    
    print(f"Done! Saved {save_name} with shape {pred_array.shape}")
    print("Now upload this file to the visualizer as 'Prediction'.")

if __name__ == "__main__":
    os.chdir(DATA_PATH)        
    main()