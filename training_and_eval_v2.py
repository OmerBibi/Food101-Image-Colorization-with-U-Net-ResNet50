import os, json, math, random, time, csv
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import Food101
from torchvision.models import resnet50, ResNet50_Weights

from PIL import Image
import matplotlib.pyplot as plt
from skimage.color import rgb2lab, lab2rgb
from sklearn.neighbors import NearestNeighbors

# -----------------------------------------------------------------------------
# 1. CONFIGURATION & PATHS
# -----------------------------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Artifacts from Notebook A
ART_DIR = Path("artifacts/food101_step10_sigma5_T042")
CENTERS_NPY = ART_DIR / "ab_centers_k259.npy"
WEIGHTS_NPY = ART_DIR / "ab_weights_k259.npy"

# Load Centers & Weights
centers = np.load(CENTERS_NPY).astype(np.float32) 
ab_weights = np.load(WEIGHTS_NPY).astype(np.float32)
K = centers.shape[0]

# Training Params
AB_MIN, AB_MAX = -110.0, 110.0
SOFT_KNN = 5
SIGMA_SOFT = 5.0
ANNEAL_T = 0.42

DATA_ROOT = Path("./data")
BATCH_SIZE = 64
EPOCHS = 45
LR_DECODER = 1e-3
LR_ENCODER = 1e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
FREEZE_EPOCHS = 2  # Warm up decoder for 2 epochs before unfreezing encoder

# Output Folders
OUT_DIR = ART_DIR / "train_runs" / "long_run_45"
CKPT_DIR = OUT_DIR / "checkpoints"
VIZ_DIR  = OUT_DIR / "viz"
STRIP_DIR = OUT_DIR / "strips"
[D.mkdir(parents=True, exist_ok=True) for D in [CKPT_DIR, VIZ_DIR, STRIP_DIR]]

# -----------------------------------------------------------------------------
# 2. LOGGING & MODEL MANAGEMENT
# -----------------------------------------------------------------------------
class TrainingLogger:
    def __init__(self, out_dir):
        self.csv_path = out_dir / "progress.csv"
        self.plot_path = out_dir / "training_curves.png"
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'train_loss', 'val_loss', 'lr', 'time_sec'])

    def log(self, epoch, train_loss, val_loss, lr, time_sec):
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, f"{train_loss:.6f}", f"{val_loss:.6f}", f"{lr:.8f}", f"{time_sec:.2f}"])
        self._plot()

    def _plot(self):
        try:
            data = np.genfromtxt(self.csv_path, delimiter=',', names=True)
            if data.size < 2: return
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
            ax1.plot(data['epoch'], data['train_loss'], label='Train Loss', marker='o')
            ax1.plot(data['epoch'], data['val_loss'], label='Val Loss', marker='s')
            ax1.set_title("Loss vs. Epoch"); ax1.legend(); ax1.grid(True)
            ax2.plot(data['epoch'], data['lr'], color='orange', label='LR')
            ax2.set_title("Learning Rate vs. Epoch"); ax2.legend(); ax2.grid(True)
            plt.tight_layout()
            plt.savefig(self.plot_path); plt.close()
        except Exception: pass

class TopModelManager:
    def __init__(self, ckpt_dir, max_keep=5):
        self.ckpt_dir = ckpt_dir
        self.max_keep = max_keep
        self.best_models = [] 

    def save_if_best(self, val_loss, state_dict, epoch):
        path = self.ckpt_dir / f"best_ep{epoch:03d}_loss{val_loss:.4f}.pt"
        self.best_models.append((val_loss, path))
        torch.save(state_dict, path)
        self.best_models.sort(key=lambda x: x[0])
        if len(self.best_models) > self.max_keep:
            _, worst_path = self.best_models.pop(-1)
            if worst_path.exists(): worst_path.unlink()

# -----------------------------------------------------------------------------
# 3. DATA & UTILS
# -----------------------------------------------------------------------------
def pil_to_rgb01(img): return (np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0)
def rgb01_to_lab(rgb01): return rgb2lab(rgb01).astype(np.float32)

def clamp_ab(lab, ab_min=AB_MIN, ab_max=AB_MAX):
    lab = lab.copy()
    lab[..., 1] = np.clip(lab[..., 1], ab_min, ab_max)
    lab[..., 2] = np.clip(lab[..., 2], ab_min, ab_max)
    return lab

class ResizeShortSide:
    def __init__(self, short_side: int, interpolation=Image.BICUBIC):
        self.short_side = short_side
        self.interpolation = interpolation
    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        scale = self.short_side / float(min(w, h))
        return img.resize((int(round(w * scale)), int(round(h * scale))), self.interpolation)

class ColorizationFood101(torch.utils.data.Dataset):
    def __init__(self, base_ds, indices, tf, centers):
        self.base, self.indices, self.tf = base_ds, list(indices), tf
        self.centers = centers.astype(np.float32)
        self.nn5 = NearestNeighbors(n_neighbors=SOFT_KNN).fit(self.centers)

    def __len__(self): return len(self.indices)

    def __getitem__(self, k):
        i = self.indices[k]
        img, _ = self.base[i]
        img = self.tf(img)
        lab = clamp_ab(rgb01_to_lab(pil_to_rgb01(img)))
        L, ab = lab[..., 0:1], lab[..., 1:3]
        L01 = (L / 100.0).astype(np.float32)
        H, W, _ = ab.shape
        dists, idx5 = self.nn5.kneighbors(ab.reshape(-1, 2), return_distance=True)
        w5 = np.exp(-(dists**2) / (2.0 * SIGMA_SOFT**2)).astype(np.float32)
        w5 /= (w5.sum(axis=1, keepdims=True) + 1e-12)
        qstar = idx5[:, 0].reshape(H, W)
        return {
            "L": torch.from_numpy(L01).permute(2,0,1),
            "idx5": torch.from_numpy(idx5.reshape(H,W,5)),
            "w5": torch.from_numpy(w5.reshape(H,W,5)),
            "qstar": torch.from_numpy(qstar.astype(np.int64))
        }

# -----------------------------------------------------------------------------
# 4. MODEL ARCHITECTURE
# -----------------------------------------------------------------------------
def adapt_resnet50_first_conv_to_1ch(m: nn.Module):
    conv1 = m.conv1
    new_conv = nn.Conv2d(1, conv1.out_channels, kernel_size=conv1.kernel_size,
                         stride=conv1.stride, padding=conv1.padding, bias=False)
    with torch.no_grad():
        new_conv.weight.copy_(conv1.weight.mean(dim=1, keepdim=True))
    m.conv1 = new_conv
    return m

class ConvGNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, groups=32):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, 1, 1, bias=False)
        self.gn = nn.GroupNorm(min(groups, out_ch), out_ch)
        self.act = nn.ReLU(inplace=True)
    def forward(self, x): return self.act(self.gn(self.conv(x)))

class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.c1 = ConvGNReLU(in_ch + skip_ch, out_ch)
        self.c2 = ConvGNReLU(out_ch, out_ch)
    def forward(self, x, skip):
        x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
        return self.c2(self.c1(torch.cat([x, skip], dim=1)))

class UNetResNet50(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        enc = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.enc = adapt_resnet50_first_conv_to_1ch(enc)
        self.b1, self.b2 = ConvGNReLU(2048, 1024), ConvGNReLU(1024, 1024)
        self.up4 = UpBlock(1024, 1024, 512)
        self.up3 = UpBlock(512, 512, 256)
        self.up2 = UpBlock(256, 256, 128)
        self.up1 = UpBlock(128, 64, 64)
        self.final_up = ConvGNReLU(64, 64)
        self.head = nn.Conv2d(64, num_classes, 1)

    def forward(self, L):
        x = self.enc.conv1(L); x = self.enc.bn1(x); x0 = self.enc.relu(x)
        x = self.enc.maxpool(x0); x1 = self.enc.layer1(x)
        x2 = self.enc.layer2(x1); x3 = self.enc.layer3(x2); x4 = self.enc.layer4(x3)
        b = self.b2(self.b1(x4))
        d3 = self.up4(b, x3); d2 = self.up3(d3, x2); d1 = self.up2(d2, x1); d0 = self.up1(d1, x0)
        u = F.interpolate(d0, scale_factor=2.0, mode="bilinear", align_corners=False)
        return self.head(self.final_up(u))

# -----------------------------------------------------------------------------
# 5. CORE TRAINING FUNCTIONS
# -----------------------------------------------------------------------------
def weighted_soft_ce_loss(logits, idx5, w5, qstar, ab_weights_t):
    logp = F.log_softmax(logits, dim=1)
    idx = idx5.permute(0, 3, 1, 2).contiguous()
    w = w5.permute(0, 3, 1, 2).contiguous()
    gathered = torch.gather(logp, dim=1, index=idx)
    per_pix = -(w * gathered).sum(dim=1)
    return (ab_weights_t[qstar] * per_pix).mean()

@torch.no_grad()
def get_visuals(logits, L01, centers_np, T=ANNEAL_T):
    # RGB prediction (Annealed Mean)
    centers_t = torch.from_numpy(centers_np).to(logits.device).to(logits.dtype)
    p = F.softmax(logits / T, dim=1)
    ab = torch.einsum("bkhw,kc->bchw", p, centers_t)
    L = (L01 * 100.0).cpu().numpy()
    ab_np = ab.cpu().numpy()
    B, _, H, W = L.shape
    rgbs = []
    for i in range(B):
        lab = np.zeros((H,W,3), dtype=np.float32)
        lab[...,0], lab[...,1:] = L[i,0], ab_np[i].transpose(1,2,0)
        rgbs.append(np.clip(lab2rgb(lab), 0, 1))
    
    # Entropy Map
    p_full = F.softmax(logits, dim=1)
    entropy = -torch.sum(p_full * torch.log(p_full + 1e-10), dim=1).cpu().numpy()
    return rgbs, entropy

# -----------------------------------------------------------------------------
# 6. MAIN EXECUTION
# -----------------------------------------------------------------------------
def main():
    print(f"Starting 45-epoch run on {DEVICE}")
    train_base = Food101(root=str(DATA_ROOT), split="train", download=True)
    n_total = len(train_base)
    idx = np.arange(n_total); np.random.default_rng(SEED).shuffle(idx)
    n_val = int(round(0.1 * n_total))
    trn_idx, val_idx = idx[n_val:].tolist(), idx[:n_val].tolist()

    train_ds = ColorizationFood101(train_base, trn_idx, transforms.Compose([ResizeShortSide(256), transforms.RandomCrop(224), transforms.RandomHorizontalFlip()]), centers)
    val_ds = ColorizationFood101(train_base, val_idx, transforms.Compose([ResizeShortSide(256), transforms.CenterCrop(224)]), centers)
    
    # Static images for the 5-image Color-Consistency Strip
    strip_indices = [3, 79, 51, 0, 82] 
    strip_batch = torch.stack([val_ds[i]['L'] for i in strip_indices]).to(DEVICE)
    strip_master = []

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True)

    model = UNetResNet50(num_classes=K).to(DEVICE)
    ab_weights_t = torch.from_numpy(ab_weights).to(DEVICE)
    
    # Initial: Frozen encoder
    for p in model.enc.parameters(): p.requires_grad = False
    
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR_DECODER, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-7)
    scaler = torch.amp.GradScaler(enabled=(DEVICE=="cuda"))
    
    logger = TrainingLogger(OUT_DIR)
    mm = TopModelManager(CKPT_DIR)

    total_start_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()
        
        # Unfreeze Logic
        if epoch == FREEZE_EPOCHS + 1:
            print(">>> Unfreezing Encoder")
            for p in model.enc.parameters(): p.requires_grad = True
            optimizer = torch.optim.AdamW([
                {"params": [p for n,p in model.named_parameters() if not n.startswith("enc.")], "lr": LR_DECODER},
                {"params": model.enc.parameters(), "lr": LR_ENCODER}
            ], weight_decay=WEIGHT_DECAY)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS-FREEZE_EPOCHS, eta_min=1e-7)

        # TRAIN
        model.train()
        train_loss_total = 0
        for batch in train_loader:
            L, idx5, w5, qstar = [batch[k].to(DEVICE, non_blocking=True) for k in ["L", "idx5", "w5", "qstar"]]
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(enabled=(DEVICE=="cuda"), device_type='cuda'):
                loss = weighted_soft_ce_loss(model(L), idx5, w5, qstar, ab_weights_t)
            scaler.scale(loss).backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer); scaler.update()
            train_loss_total += loss.item()
        
        avg_train = train_loss_total / len(train_loader)

        # VALIDATE
        model.eval()
        val_loss_total = 0
        with torch.no_grad():
            for batch in val_loader:
                L, idx5, w5, qstar = [batch[k].to(DEVICE, non_blocking=True) for k in ["L", "idx5", "w5", "qstar"]]
                val_loss_total += weighted_soft_ce_loss(model(L), idx5, w5, qstar, ab_weights_t).item()
        
        avg_val = val_loss_total / len(val_loader)
        curr_lr = optimizer.param_groups[0]['lr']
        
        epoch_duration = time.time() - epoch_start
        logger.log(epoch, avg_train, avg_val, curr_lr, epoch_duration)
        mm.save_if_best(avg_val, model.state_dict(), epoch)

        # VISUALIZE
        with torch.no_grad():
            rgbs, entropies = get_visuals(model(strip_batch), strip_batch, centers)
            strip_master.append(np.hstack(rgbs))
            Image.fromarray((np.vstack(strip_master) * 255).astype(np.uint8)).save(STRIP_DIR / "consistency_filmstrip.png")
            plt.imsave(VIZ_DIR / f"entropy_ep{epoch:03d}.png", entropies[-1], cmap='jet')

        elapsed = time.time() - total_start_time
        eta = (elapsed / epoch) * (EPOCHS - epoch)
        print(f"E[{epoch}/{EPOCHS}] Train: {avg_train:.4f} | Val: {avg_val:.4f} | Time: {epoch_duration:.1f}s | ETA: {eta/60:.1f}m")
        scheduler.step()

if __name__ == "__main__":
    main()