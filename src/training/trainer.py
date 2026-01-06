"""Training orchestrator for colorization model."""

import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from .logger import TrainingLogger
from .checkpoint import TopModelManager
from .metrics import PerceptualMetrics
from ..models.losses import weighted_soft_ce_loss
from ..utils.visualization import get_visuals, decode_annealed_mean
from ..utils.color_utils import lab_to_rgb


class Trainer:
    """Training orchestrator for colorization model.

    Args:
        model: The U-Net colorization model
        train_loader: Training data loader
        val_loader: Validation data loader
        config: Configuration dictionary
        centers: Color space centers array
        ab_weights: Per-bin rebalancing weights
        val_dataset: Validation dataset (for strip visualization)
        output_dirs: Tuple of (checkpoint_dir, viz_dir, strip_dir)
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: dict,
        centers: np.ndarray,
        ab_weights: np.ndarray,
        val_dataset,
        output_dirs: tuple
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.centers = centers
        self.device = config['device']

        ckpt_dir, viz_dir, strip_dir = output_dirs
        self.viz_dir = viz_dir
        self.strip_dir = strip_dir

        # Convert weights to tensor
        self.ab_weights_t = torch.from_numpy(ab_weights).to(self.device)

        # Setup perceptual metrics (if enabled)
        val_config = config.get('validation', {})
        self.compute_perceptual_metrics = val_config.get('compute_perceptual_metrics', False)

        if self.compute_perceptual_metrics:
            metrics_config = val_config.get('metrics', {})
            self.metrics_computer = PerceptualMetrics(
                device=self.device,
                use_lpips=metrics_config.get('lpips', {}).get('enabled', True),
                use_ssim=metrics_config.get('ssim', {}).get('enabled', True),
                use_psnr=metrics_config.get('psnr', {}).get('enabled', True),
                lpips_net=metrics_config.get('lpips', {}).get('net', 'alex')
            )
            track_metrics = ['loss', 'lpips', 'ssim', 'psnr']
        else:
            self.metrics_computer = None
            track_metrics = ['loss']

        # Setup logging
        self.logger = TrainingLogger(Path(ckpt_dir).parent, track_metrics=track_metrics)

        # Setup checkpoint managers (one per metric)
        ckpt_config = config.get('checkpointing', {})
        save_top_k = ckpt_config.get('save_top_k', 3)
        metrics_to_track = ckpt_config.get('metrics_to_track', ['loss'])

        self.checkpoint_managers = {}
        for metric_name in metrics_to_track:
            metric_ckpt_dir = ckpt_dir / metric_name
            metric_ckpt_dir.mkdir(parents=True, exist_ok=True)
            self.checkpoint_managers[metric_name] = TopModelManager(
                metric_ckpt_dir,
                max_keep=save_top_k,
                metric_name=metric_name
            )

        # Setup optimizer (initially frozen encoder)
        self._setup_optimizer(frozen_encoder=True)

        # Setup scheduler and scaler
        epochs = config['training']['epochs']
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epochs, eta_min=1e-7
        )
        self.scaler = torch.amp.GradScaler(
            enabled=(self.device == "cuda")
        )

        # Strip visualization setup
        self.strip_indices = [3, 79, 51, 0, 82]
        self.strip_batch = torch.stack([
            val_dataset[i]['L'] for i in self.strip_indices
        ]).to(self.device)
        self.strip_master = []

    def _setup_optimizer(self, frozen_encoder: bool = False):
        """Setup optimizer with different learning rates for encoder/decoder."""
        if frozen_encoder:
            for p in self.model.enc.parameters():
                p.requires_grad = False
            self.optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=self.config['training']['lr_decoder'],
                weight_decay=self.config['training']['weight_decay']
            )
        else:
            for p in self.model.enc.parameters():
                p.requires_grad = True
            self.optimizer = torch.optim.AdamW([
                {
                    "params": [
                        p for n, p in self.model.named_parameters()
                        if not n.startswith("enc.")
                    ],
                    "lr": self.config['training']['lr_decoder']
                },
                {
                    "params": self.model.enc.parameters(),
                    "lr": self.config['training']['lr_encoder']
                }
            ], weight_decay=self.config['training']['weight_decay'])

    def train_epoch(self) -> float:
        """Run one training epoch."""
        self.model.train()
        train_loss_total = 0

        for batch in self.train_loader:
            L = batch["L"].to(self.device, non_blocking=True)
            idx5 = batch["idx5"].to(self.device, non_blocking=True)
            w5 = batch["w5"].to(self.device, non_blocking=True)
            qstar = batch["qstar"].to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(
                enabled=(self.device == "cuda"),
                device_type='cuda'
            ):
                loss = weighted_soft_ce_loss(
                    self.model(L), idx5, w5, qstar, self.ab_weights_t
                )

            self.scaler.scale(loss).backward()
            nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config['training']['grad_clip']
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()

            train_loss_total += loss.item()

        return train_loss_total / len(self.train_loader)

    @torch.no_grad()
    def validate(self) -> dict:
        """Run validation with multiple metrics.

        Returns:
            Dictionary with metric names as keys and average values as floats
        """
        self.model.eval()

        # Initialize metrics accumulator
        metrics_accumulator = {'loss': 0.0}
        if self.compute_perceptual_metrics:
            metrics_accumulator.update({'lpips': 0.0, 'ssim': 0.0, 'psnr': 0.0})

        for batch in self.val_loader:
            L = batch["L"].to(self.device, non_blocking=True)
            idx5 = batch["idx5"].to(self.device, non_blocking=True)
            w5 = batch["w5"].to(self.device, non_blocking=True)
            qstar = batch["qstar"].to(self.device, non_blocking=True)

            # Compute loss
            logits = self.model(L)
            loss = weighted_soft_ce_loss(logits, idx5, w5, qstar, self.ab_weights_t)
            metrics_accumulator['loss'] += loss.item()

            # Compute perceptual metrics
            if self.compute_perceptual_metrics:
                # Generate RGB predictions
                pred_rgb = self._logits_to_rgb(logits, L)

                # Get ground truth RGB
                target_rgb = batch["rgb_gt"].to(self.device, non_blocking=True)

                # Compute metrics
                batch_metrics = self.metrics_computer.compute_batch_metrics(
                    pred_rgb, target_rgb
                )
                for k, v in batch_metrics.items():
                    metrics_accumulator[k] += v

        # Average over batches
        num_batches = len(self.val_loader)
        return {k: v / num_batches for k, v in metrics_accumulator.items()}

    def _logits_to_rgb(self, logits: torch.Tensor, L01: torch.Tensor) -> torch.Tensor:
        """Convert model logits to RGB images.

        Args:
            logits: Model output logits (B, K, H, W)
            L01: Luminance channel (B, 1, H, W) in [0, 1]

        Returns:
            RGB images (B, 3, H, W) in [0, 1]
        """
        # Decode to ab channels
        ab = decode_annealed_mean(
            logits,
            self.centers,
            T=self.config['soft_encoding']['anneal_t']
        )

        # Scale L back to [0, 100]
        L = L01 * 100.0

        # Construct LAB
        lab = torch.cat([L, ab], dim=1)  # (B, 3, H, W)

        # Convert to RGB
        B, _, H, W = lab.shape
        rgb_batch = []
        for i in range(B):
            lab_np = lab[i].permute(1, 2, 0).cpu().numpy()  # (H, W, 3)
            rgb = lab_to_rgb(lab_np)  # Returns [0, 1]
            rgb_batch.append(torch.from_numpy(rgb).permute(2, 0, 1))

        return torch.stack(rgb_batch).to(self.device)

    @torch.no_grad()
    def visualize_strip(self, epoch: int):
        """Generate consistency filmstrip visualization."""
        self.model.eval()
        rgbs, entropies = get_visuals(
            self.model(self.strip_batch),
            self.strip_batch,
            self.centers,
            T=self.config['soft_encoding']['anneal_t']
        )

        self.strip_master.append(np.hstack(rgbs))
        Image.fromarray(
            (np.vstack(self.strip_master) * 255).astype(np.uint8)
        ).save(self.strip_dir / "consistency_filmstrip.png")

        plt.imsave(
            self.viz_dir / f"entropy_ep{epoch:03d}.png",
            entropies[-1],
            cmap='jet'
        )

    def train(self):
        """Main training loop."""
        epochs = self.config['training']['epochs']
        freeze_epochs = self.config['training']['freeze_epochs']
        total_start_time = time.time()

        for epoch in range(1, epochs + 1):
            epoch_start = time.time()

            # Unfreeze encoder after warmup
            if epoch == freeze_epochs + 1:
                print(">>> Unfreezing Encoder")
                self._setup_optimizer(frozen_encoder=False)
                self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer,
                    T_max=epochs - freeze_epochs,
                    eta_min=1e-7
                )

            # Train and validate
            avg_train = self.train_epoch()
            val_metrics = self.validate()

            curr_lr = self.optimizer.param_groups[0]['lr']
            epoch_duration = time.time() - epoch_start

            # Log progress
            self.logger.log(epoch, avg_train, val_metrics, curr_lr, epoch_duration)

            # Save checkpoints for all tracked metrics
            for metric_name, manager in self.checkpoint_managers.items():
                metric_value = val_metrics.get(metric_name)
                if metric_value is not None:
                    # For SSIM and PSNR (higher is better), negate for comparison
                    if metric_name in ['ssim', 'psnr']:
                        save_value = -metric_value
                    else:
                        save_value = metric_value

                    manager.save_if_best(
                        save_value,
                        self.model.state_dict(),
                        epoch
                    )

            # Visualize
            self.visualize_strip(epoch)

            # Print progress
            elapsed = time.time() - total_start_time
            eta = (elapsed / epoch) * (epochs - epoch)

            # Build metrics string
            metrics_str = f"Train: {avg_train:.4f} | Val Loss: {val_metrics['loss']:.4f}"
            if self.compute_perceptual_metrics:
                metrics_str += f" | LPIPS: {val_metrics.get('lpips', 0):.4f}"
                metrics_str += f" | SSIM: {val_metrics.get('ssim', 0):.4f}"
                metrics_str += f" | PSNR: {val_metrics.get('psnr', 0):.2f}"

            print(
                f"E[{epoch}/{epochs}] {metrics_str} | "
                f"Time: {epoch_duration:.1f}s | ETA: {eta/60:.1f}m"
            )

            self.scheduler.step()
