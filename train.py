# train.py

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast
import os 
import torch.nn.functional as F 
import torch.nn.utils as torch_utils 

from model import TextEncoder, ObjectVAE, LATENT_DIM
from train_dataset import TrainDataset, collate_fn 
from losses import VAELoss, InfoNCELoss, BatchHardTripletLoss
from data_utils import OBJECT_FEATURE_DIM, get_or_compute_stats 


SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
LOCAL_SBERT_PATH = os.path.join(SCRIPT_DIR, "downloaded-models", "all-MiniLM-L6-v2")
TRAIN_DIR = os.path.join(SCRIPT_DIR, "train")
TEXT_MODEL_SAVE_PATH = os.path.join(SCRIPT_DIR, "text_encoder.pth")
OBJECT_MODEL_SAVE_PATH = os.path.join(SCRIPT_DIR, "object_vae.pth")

# --- (超参数：微调模式) ---
EPOCHS = 50
BATCH_SIZE = 2048
BASE_LEARNING_RATE = 1e-5 
SBERT_LEARNING_RATE = 1e-6 

KL_WEIGHT = 0.05          
VAE_WEIGHT = 0.1 
ALIGNMENT_WEIGHT = 1.0
TRIPLET_WEIGHT = 0.5      

FIXED_LOGIT_SCALE = 60.0  

NUM_WORKERS = 0 
CLIP_GRAD_NORM = 1.0


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print("Initializing feature statistics...")
    mean, std = get_or_compute_stats(TRAIN_DIR)
    mean = mean.to(device); std = std.to(device)
    
    print("Initializing Dataset...")
    dataset = TrainDataset(TRAIN_DIR, mean=mean.cpu(), std=std.cpu()) 
    loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True, 
        collate_fn=collate_fn, num_workers=NUM_WORKERS,
        pin_memory=True, drop_last=True 
    )
    
    print("Initializing Models...")
    text_encoder = TextEncoder(latent_dim=LATENT_DIM, model_name=LOCAL_SBERT_PATH).to(device)
    object_vae = ObjectVAE(input_dim=OBJECT_FEATURE_DIM, latent_dim=LATENT_DIM).to(device)

   
    if os.path.exists(TEXT_MODEL_SAVE_PATH):
        print(f"Loading existing weights from {TEXT_MODEL_SAVE_PATH}")
        try: text_encoder.load_state_dict(torch.load(TEXT_MODEL_SAVE_PATH, map_location=device))
        except: pass
    if os.path.exists(OBJECT_MODEL_SAVE_PATH):
        print(f"Loading existing weights from {OBJECT_MODEL_SAVE_PATH}")
        try: object_vae.load_state_dict(torch.load(OBJECT_MODEL_SAVE_PATH, map_location=device))
        except: pass
   

   
    vae_loss_fn = VAELoss(kl_weight=KL_WEIGHT).to(device)
  
    align_loss_fn = InfoNCELoss(static_logit_scale=FIXED_LOGIT_SCALE).to(device)
   
    triplet_loss_fn = BatchHardTripletLoss(margin=0.2).to(device)

    
    optimizer = optim.AdamW([
        {'params': object_vae.parameters(), 'lr': BASE_LEARNING_RATE},
        {'params': text_encoder.parameters(), 'lr': SBERT_LEARNING_RATE}
    ])
    
    scaler = GradScaler(enabled=(device.type == 'cuda'))
    
    print(f"--- Starting Fine-tuning (Fixed Logit={FIXED_LOGIT_SCALE} + Triplet Loss) ---")
    
    for epoch in range(EPOCHS):
        text_encoder.train()
        object_vae.train()
        
        total_epoch_loss = 0
        total_vae_loss = 0
        total_align_loss = 0
        total_triplet_loss = 0
        
        epoch_avg_max_sim = 0.0
        epoch_avg_mean_sim = 0.0
        epoch_avg_neg_sim = 0.0 
        epoch_avg_margin = 0.0 
        
        progress_bar = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for (texts, obj_tensors) in progress_bar:
            B, D_obj = obj_tensors.shape
            obj_tensors = obj_tensors.to(device)
            
            optimizer.zero_grad()
            
            with autocast(enabled=(device.type == 'cuda')):
                
                recon_x, mu_vae, logvar_vae = object_vae(obj_tensors)
                
                loss_vae, recon_loss, kld_loss = vae_loss_fn(recon_x, obj_tensors, mu_vae, logvar_vae)
                
                text_embeddings = text_encoder(texts) 
                obj_embeddings = mu_vae
                
                # 计算损失
                loss_align = align_loss_fn(text_embeddings, obj_embeddings)
                loss_triplet = triplet_loss_fn(text_embeddings, obj_embeddings)
                
                total_loss = (VAE_WEIGHT * loss_vae) + \
                             (ALIGNMENT_WEIGHT * loss_align) + \
                             (TRIPLET_WEIGHT * loss_triplet)
            
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch_utils.clip_grad_norm_(object_vae.parameters(), CLIP_GRAD_NORM)
            torch_utils.clip_grad_norm_(text_encoder.parameters(), CLIP_GRAD_NORM)
            scaler.step(optimizer)
            scaler.update()
            
          
            with torch.no_grad():
                text_norm = F.normalize(text_embeddings.detach(), p=2, dim=1)
                obj_norm = F.normalize(obj_embeddings.detach(), p=2, dim=1)
                
                sim_matrix = text_norm @ obj_norm.T
                pos_scores = sim_matrix.diag()
                
                eye_mask = torch.eye(B, device=device).bool()
                sim_matrix_neg = sim_matrix.clone()
                sim_matrix_neg.masked_fill_(eye_mask, -float('inf'))
                hard_neg_scores = sim_matrix_neg.max(dim=1)[0]
                
                margins = pos_scores - hard_neg_scores
                
                epoch_avg_max_sim += pos_scores.max().item()
                epoch_avg_mean_sim += pos_scores.mean().item()
                epoch_avg_neg_sim += hard_neg_scores.mean().item()
                epoch_avg_margin += margins.mean().item()

            total_epoch_loss += total_loss.item()
            total_vae_loss += loss_vae.item() 
            total_align_loss += loss_align.item()
            total_triplet_loss += loss_triplet.item()

            num_batches_so_far = progress_bar.n + 1
            progress_bar.set_postfix(
                L=f"{total_loss.item():.2f}",
                Ali=f"{total_align_loss / num_batches_so_far:.3f}", 
                Trip=f"{total_triplet_loss / num_batches_so_far:.3f}",
                Max_S=f"{epoch_avg_max_sim / num_batches_so_far:.4f}", 
                Avg_S=f"{epoch_avg_mean_sim / num_batches_so_far:.4f}",
                Neg_S=f"{epoch_avg_neg_sim / num_batches_so_far:.4f}",
                Marg=f"{epoch_avg_margin / num_batches_so_far:.4f}",
                Logit=f"{FIXED_LOGIT_SCALE:.1f}" 
            )
            
        avg_pos = epoch_avg_mean_sim / len(loader)
        avg_neg = epoch_avg_neg_sim / len(loader)
        avg_marg = epoch_avg_margin / len(loader)
        print(f"Epoch {epoch+1} Done. Loss: {total_epoch_loss/len(loader):.4f}")
        print(f"--> Sim Stats: Pos_Avg={avg_pos:.4f}, Neg_Avg={avg_neg:.4f}, Margin={avg_marg:.4f}")

    print("--- Training Complete ---")
    print(f"Saving models to {TEXT_MODEL_SAVE_PATH} and {OBJECT_MODEL_SAVE_PATH}")
    torch.save(text_encoder.state_dict(), TEXT_MODEL_SAVE_PATH)
    torch.save(object_vae.state_dict(), OBJECT_MODEL_SAVE_PATH)
    print("Models saved successfully.")


if __name__ == "__main__":
    main()
