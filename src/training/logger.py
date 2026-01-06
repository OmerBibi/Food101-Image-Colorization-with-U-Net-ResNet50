"""Training progress logger with CSV and plot generation.

Logs training metrics including loss and perceptual quality metrics (LPIPS, SSIM, PSNR).
"""

import csv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List


class TrainingLogger:
    """Training progress logger with CSV and plot generation.

    Supports logging multiple validation metrics and generates comprehensive
    training curves with separate subplots for each metric type.

    Args:
        out_dir: Output directory for logs and plots
        track_metrics: List of validation metrics to track (e.g., ['loss', 'lpips', 'ssim', 'psnr'])
    """

    def __init__(self, out_dir: Path, track_metrics: List[str] = None):
        self.csv_path = out_dir / "progress.csv"
        self.plot_path = out_dir / "training_curves.png"
        self.track_metrics = track_metrics or ['loss']

        # Create CSV with header
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            header = ['epoch', 'train_loss']
            for metric in self.track_metrics:
                header.append(f'val_{metric}')
            header.extend(['lr', 'time_sec'])
            writer.writerow(header)

    def log(self, epoch: int, train_loss: float, val_metrics: Dict[str, float],
            lr: float, time_sec: float):
        """Log metrics for an epoch.

        Args:
            epoch: Current epoch number
            train_loss: Training loss value
            val_metrics: Dictionary of validation metrics
            lr: Current learning rate
            time_sec: Epoch duration in seconds
        """
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            row = [epoch, f"{train_loss:.6f}"]
            for metric in self.track_metrics:
                value = val_metrics.get(metric, 0.0)
                row.append(f"{value:.6f}")
            row.extend([f"{lr:.8f}", f"{time_sec:.2f}"])
            writer.writerow(row)
        self._plot()

    def _plot(self):
        """Generate training curves plot with multiple metrics."""
        try:
            data = np.genfromtxt(self.csv_path, delimiter=',', names=True)
            if data.size < 2:
                return

            # Determine number of subplots needed
            num_plots = 2  # Always have Loss and LR
            has_lpips = 'val_lpips' in data.dtype.names
            has_ssim = 'val_ssim' in data.dtype.names
            has_psnr = 'val_psnr' in data.dtype.names

            if has_lpips:
                num_plots += 1
            if has_ssim or has_psnr:
                num_plots += 1

            # Create figure with subplots
            fig, axes = plt.subplots(num_plots, 1, figsize=(10, 4 * num_plots))
            if num_plots == 1:
                axes = [axes]

            plot_idx = 0

            # Subplot 1: Loss curves
            axes[plot_idx].plot(data['epoch'], data['train_loss'],
                               label='Train Loss', marker='o')
            axes[plot_idx].plot(data['epoch'], data['val_loss'],
                               label='Val Loss', marker='s')
            axes[plot_idx].set_title("Loss vs. Epoch")
            axes[plot_idx].set_xlabel("Epoch")
            axes[plot_idx].set_ylabel("Loss")
            axes[plot_idx].legend()
            axes[plot_idx].grid(True)
            plot_idx += 1

            # Subplot 2: LPIPS curve (if available)
            if has_lpips:
                axes[plot_idx].plot(data['epoch'], data['val_lpips'],
                                   label='Val LPIPS', marker='o', color='purple')
                axes[plot_idx].set_title("LPIPS vs. Epoch (Lower is Better)")
                axes[plot_idx].set_xlabel("Epoch")
                axes[plot_idx].set_ylabel("LPIPS")
                axes[plot_idx].legend()
                axes[plot_idx].grid(True)
                plot_idx += 1

            # Subplot 3: SSIM and PSNR curves (if available)
            if has_ssim or has_psnr:
                if has_ssim:
                    ax_ssim = axes[plot_idx]
                    ax_ssim.plot(data['epoch'], data['val_ssim'],
                                label='Val SSIM', marker='o', color='green')
                    ax_ssim.set_xlabel("Epoch")
                    ax_ssim.set_ylabel('SSIM', color='green')
                    ax_ssim.tick_params(axis='y', labelcolor='green')
                    ax_ssim.legend(loc='upper left')
                    ax_ssim.grid(True)

                if has_psnr:
                    if has_ssim:
                        ax_psnr = ax_ssim.twinx()
                    else:
                        ax_psnr = axes[plot_idx]
                    ax_psnr.plot(data['epoch'], data['val_psnr'],
                                label='Val PSNR', marker='s', color='blue')
                    if not has_ssim:
                        ax_psnr.set_xlabel("Epoch")
                    ax_psnr.set_ylabel('PSNR (dB)', color='blue')
                    ax_psnr.tick_params(axis='y', labelcolor='blue')
                    ax_psnr.legend(loc='upper right')
                    if not has_ssim:
                        ax_psnr.grid(True)

                if has_ssim and has_psnr:
                    axes[plot_idx].set_title("SSIM & PSNR vs. Epoch (Higher is Better)")
                elif has_ssim:
                    axes[plot_idx].set_title("SSIM vs. Epoch (Higher is Better)")
                else:
                    axes[plot_idx].set_title("PSNR vs. Epoch (Higher is Better)")
                plot_idx += 1

            # Last Subplot: Learning Rate
            axes[plot_idx].plot(data['epoch'], data['lr'],
                               color='orange', label='LR')
            axes[plot_idx].set_title("Learning Rate vs. Epoch")
            axes[plot_idx].set_xlabel("Epoch")
            axes[plot_idx].set_ylabel("Learning Rate")
            axes[plot_idx].legend()
            axes[plot_idx].grid(True)

            plt.tight_layout()
            plt.savefig(self.plot_path, dpi=100)
            plt.close()
        except Exception as e:
            # Silently fail to avoid interrupting training
            pass
