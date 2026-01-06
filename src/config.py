"""Configuration management for Food101 Colorization.

Loads and validates configuration from YAML files.
"""

from pathlib import Path
from typing import Dict, Tuple
import yaml
import torch


class Config:
    """Central configuration manager.

    Args:
        config_path: Path to YAML configuration file
    """

    def __init__(self, config_path: str = "configs/default.yaml"):
        self.config = self._load_yaml(config_path)
        self._validate()
        self._setup_derived_values()

    def _load_yaml(self, path: str) -> Dict:
        """Load YAML configuration file."""
        with open(path, 'r') as f:
            config = yaml.safe_load(f)
        return config

    def _validate(self):
        """Validate required fields exist."""
        required_sections = [
            'seed', 'paths', 'color', 'soft_encoding',
            'data', 'model', 'training', 'checkpointing', 'output'
        ]
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required config section: {section}")

        # Validate model architecture
        model_config = self.config['model']
        if 'encoder' not in model_config:
            raise ValueError("Missing 'encoder' in model config")
        if 'decoder_channels' not in model_config:
            raise ValueError("Missing 'decoder_channels' in model config")
        if len(model_config['decoder_channels']) != 5:
            raise ValueError(
                f"decoder_channels must have 5 elements, "
                f"got {len(model_config['decoder_channels'])}"
            )

    def _setup_derived_values(self):
        """Setup derived paths and device."""
        # Auto-detect CUDA if device is set to "cuda"
        if self.config['device'] == "cuda":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = self.config['device']

        # Update config with actual device
        self.config['device'] = self.device

        # Convert string paths to Path objects
        self.data_root = Path(self.config['paths']['data_root'])
        self.artifact_dir = Path(self.config['paths']['artifact_dir'])
        self.centers_path = self.artifact_dir / self.config['paths']['centers_file']
        self.weights_path = self.artifact_dir / self.config['paths']['weights_file']

        # Setup output directories
        output_base = self.artifact_dir / "train_runs" / self.config['output']['run_name']
        self.checkpoint_dir = output_base / self.config['output']['checkpoint_dir']
        self.viz_dir = output_base / self.config['output']['viz_dir']
        self.strip_dir = output_base / self.config['output']['strip_dir']

    def get_output_dirs(self) -> Tuple[Path, Path, Path]:
        """Returns (checkpoint_dir, viz_dir, strip_dir)."""
        return self.checkpoint_dir, self.viz_dir, self.strip_dir

    def create_output_dirs(self):
        """Create all output directories if they don't exist."""
        for d in [self.checkpoint_dir, self.viz_dir, self.strip_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def __repr__(self):
        return f"Config(device={self.device}, encoder={self.config['model']['encoder']})"
