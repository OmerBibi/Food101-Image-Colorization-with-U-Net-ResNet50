"""Unified inference manager for image colorization.

Provides a simple interface for colorizing images and video frames.
"""

from pathlib import Path
from typing import Union, Dict
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F

from .color_utils import pil_to_rgb01, rgb01_to_lab, clamp_ab, lab_to_rgb
from .visualization import decode_annealed_mean


class ColorizationInference:
    """Unified inference manager for colorization tasks."""

    def __init__(
        self,
        model_path: Path,
        centers_path: Path,
        config: Dict,
        device: str = "cuda"
    ):
        """Initialize inference manager.

        Args:
            model_path: Path to model checkpoint
            centers_path: Path to color bin centers .npy file
            config: Configuration dictionary
            device: Device to run inference on ("cuda" or "cpu")
        """
        self.config = config
        self.device = device if device == "cpu" else ("cuda" if torch.cuda.is_available() else "cpu")

        # Load color bin centers
        self.centers = np.load(centers_path).astype(np.float32)
        self.K = self.centers.shape[0]

        # Load model
        self.model = self._load_model(model_path)
        self.model.eval()

        # Setup preprocessing transform
        from ..data.transforms import get_inference_transforms
        use_full_size = config.get('inference', {}).get('use_full_size', True)
        self.preprocess_tf = get_inference_transforms(config, use_full_size=use_full_size)

        print(f"Inference manager initialized:")
        print(f"  Device: {self.device}")
        print(f"  Color bins: {self.K}")
        print(f"  Model: {model_path.name}")
        print(f"  Full-size inference: {use_full_size}")

    def _load_model(self, model_path: Path):
        """Load model from checkpoint.

        Args:
            model_path: Path to checkpoint file

        Returns:
            Loaded model in eval mode
        """
        from ..models.unet_resnet50 import UNetResNet50

        # Create model architecture
        model = UNetResNet50(
            num_classes=self.K,
            model_config=self.config['model']
        ).to(self.device)

        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

        # Handle different checkpoint formats
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint

        model.load_state_dict(state_dict)
        model.eval()

        return model

    @torch.no_grad()
    def colorize_image(
        self,
        image: Union[Path, str, Image.Image, np.ndarray],
        temperature: float = None,
        return_entropy: bool = False,
        use_full_size: bool = None
    ) -> Dict[str, np.ndarray]:
        """Colorize a single image.

        Args:
            image: Input image (Path, PIL.Image, or numpy array)
            temperature: Annealing temperature (None = use config default)
            return_entropy: Whether to include entropy map
            use_full_size: Override config setting for full-size inference

        Returns:
            Dictionary with keys:
                - 'rgb': (H, W, 3) colorized RGB image in [0, 1]
                - 'L': (H, W) grayscale L channel in [0, 1]
                - 'ab': (H, W, 2) predicted ab channels
                - 'entropy': (H, W) entropy map (if return_entropy=True)
        """
        # Load and preprocess image
        img_pil = self._load_image(image)

        # Apply transform (use override if provided)
        if use_full_size is not None:
            # Create temporary transform with override
            from ..data.transforms import get_inference_transforms
            img_tf = get_inference_transforms(self.config, use_full_size=use_full_size)(img_pil)
        else:
            # Use default transform
            img_tf = self.preprocess_tf(img_pil)

        # Convert to LAB
        rgb01 = pil_to_rgb01(img_tf)
        lab = clamp_ab(
            rgb01_to_lab(rgb01),
            self.config['color']['ab_min'],
            self.config['color']['ab_max']
        )
        L = lab[..., 0:1]  # (H, W, 1)
        L01 = (L / 100.0).astype(np.float32)

        # Prepare input tensor
        L_tensor = torch.from_numpy(L01).permute(2, 0, 1).unsqueeze(0).to(self.device)

        # Run inference
        logits = self.model(L_tensor)

        # Decode ab channels
        T = temperature if temperature is not None else self.config['soft_encoding']['anneal_t']
        ab_pred = decode_annealed_mean(logits, self.centers, T)

        # Convert to RGB
        H, W = L.shape[:2]
        lab_pred = np.zeros((H, W, 3), dtype=np.float32)
        lab_pred[..., 0] = L.squeeze()
        lab_pred[..., 1:] = ab_pred[0].cpu().numpy().transpose(1, 2, 0)
        rgb = lab_to_rgb(lab_pred)

        # Prepare results
        result = {
            'rgb': rgb,
            'L': L.squeeze() / 100.0,
            'ab': ab_pred[0].cpu().numpy().transpose(1, 2, 0)
        }

        # Optional entropy map
        if return_entropy:
            p = F.softmax(logits, dim=1)
            entropy = -torch.sum(p * torch.log(p + 1e-10), dim=1)
            result['entropy'] = entropy[0].cpu().numpy()

        return result

    def _load_image(
        self,
        image: Union[Path, str, Image.Image, np.ndarray]
    ) -> Image.Image:
        """Load image from various input formats.

        Args:
            image: Input image in various formats

        Returns:
            PIL Image in RGB mode
        """
        if isinstance(image, (Path, str)):
            return Image.open(image).convert('RGB')
        elif isinstance(image, Image.Image):
            return image.convert('RGB')
        elif isinstance(image, np.ndarray):
            # Assume numpy array is in [0, 1] range
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            return Image.fromarray(image)
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")

    @torch.no_grad()
    def colorize_batch(
        self,
        images: list,
        temperature: float = None,
        batch_size: int = None,
        use_full_size: bool = None
    ) -> list:
        """Colorize multiple images efficiently.

        Note: When use_full_size=True, images may have different sizes,
        so they are processed individually (batch_size is ignored).

        Args:
            images: List of images (paths, PIL Images, or numpy arrays)
            temperature: Annealing temperature (None = use config default)
            batch_size: Batch size for processing (None = use config default)
            use_full_size: Override config setting for full-size inference

        Returns:
            List of result dictionaries (same format as colorize_image)
        """
        if batch_size is None:
            batch_size = self.config.get('inference', {}).get('batch_size', 16)

        # Determine full-size mode
        full_size = use_full_size if use_full_size is not None else \
                    self.config.get('inference', {}).get('use_full_size', True)

        # When using full-size inference, images have different dimensions
        if full_size:
            print("Full-size mode: Processing images individually")

        results = []
        for i in range(0, len(images), 1 if full_size else batch_size):
            batch = images[i:i + (1 if full_size else batch_size)]
            for img in batch:
                result = self.colorize_image(
                    img,
                    temperature=temperature,
                    use_full_size=use_full_size
                )
                results.append(result)

        return results
