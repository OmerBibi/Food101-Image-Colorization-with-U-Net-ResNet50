from setuptools import setup, find_packages

setup(
    name="food101-colorization",
    version="0.1.0",
    description="Food101 Image Colorization using U-Net + ResNet50",
    author="Omer",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.24.0",
        "scikit-image>=0.20.0",
        "scikit-learn>=1.2.0",
        "Pillow>=9.5.0",
        "matplotlib>=3.7.0",
        "PyYAML>=6.0",
        "tqdm>=4.65.0",
    ],
    python_requires=">=3.8",
)
