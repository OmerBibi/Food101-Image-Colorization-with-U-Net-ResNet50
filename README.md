# 🍔 Food Colorization using Deep Networks: Grayscale to Vibrant Color
### *Advanced Deep Learning Colorization using U-Net & ResNet-50*

<p align="center"> 
<img src="https://github.com/user-attachments/assets/a82e6132-8949-4641-8513-6786c188b3e2" alt="Colorized Food Commercial" width="600" /> </p>

<p align="center"> 

<p align="center"> 
Final project for the Technion's EE Deep Learning course (046217)
<p align="center"> 

---
## 📑 Table of Contents

- [📖 Introduction](#-introduction)
- [🛠️ Model Architecture & Methodology](#️-model-architecture--methodology)
- [🖼️ Results Gallery](#️-results-gallery)
- [📂 Repository Contents](#-repository-contents)
- [💻 Setup & Usage](#-setup--usage)

---

## 📖 Introduction
This project implements a colorization pipeline using a **U-Net** architecture with a **ResNet-50** backbone, trained on the **Food-101** dataset. By leveraging the CIELAB color space, the model learns to predict the $a$ and $b$ (color) channels given the $L$ (luminance) channel.

### 🎯 Project Highlights
* **High-Fidelity Colorization**: Uses annealed mean decoding ($T=0.42$) for vibrant results.
* **Video Capability**: Successfully processes video sequences frame-by-frame.
* **Smart Architecture**: Combines pre-trained ResNet-50 features with U-Net skip connections.

---

## 🛠️ Model Architecture & Methodology

The core of this project is a **Quantized Color Class Prediction** model. Instead of predicting a single color value (which leads to "muddy" gray results), the model predicts a probability distribution over 259 discrete color bins.

| Stage | Description |
| :--- | :--- |
| **Preprocessing** | Lab color space conversion, color quantization, and rare-class rebalancing. |
| **Encoder** | Modified **ResNet-50** (accepting 1-channel grayscale input). |
| **Decoder** | Symmetric upsampling path with **Skip Connections** to preserve edges. |
| **Inference** | Annealed Mean decoding to balance color saturation and realism. |
<p align="center">
<img width="444" height="608" alt="graphviz (1)" src="https://github.com/user-attachments/assets/a8545a93-3179-4076-b949-c9a55e5f22d0" />
<p align="center">
  
---

## 🖼️ Results Gallery

### Side-by-Side Test Comparisons
Below are several examples from the test set showing the input grayscale, our model's prediction, and the ground truth.

<p align="center">
  <table>
    <tr>
      <td><img src="https://github.com/user-attachments/assets/7afb2432-cf7f-4c82-8b96-8d093ed2bf5d" width="450" alt="Evaluation 1"></td>
      <td><img src="https://github.com/user-attachments/assets/54b3ece4-7d31-47ef-916f-5ad68c6bd6dc" width="450" alt="Evaluation 2"></td>
    </tr>
  </table>
</p>

### Training Evolution
We monitored the model's progress throughout training using a "consistency filmstrip":
<p align="center"> 
<img width="606.5" height="1171" alt="consistency_filmstrip_10_epoch" src="https://github.com/user-attachments/assets/a5841314-6860-44b3-8307-56f339085c7d" />
<p align="center"> 

---

## 📂 Repository Contents
* `data_and_preprocess.ipynb`: Data preparation, color binning, and weight calculation.
* `training_and_eval_v2.py`: The main training script with validation logic.
* `visualization.ipynb`: Inference tools for images, videos, and entropy visualization.

---

## 💻 Setup & Usage
### 📌 Overview
This repository implements image colorization on **Food-101** using a **U-Net decoder with a ResNet50 encoder**, trained in LAB color space with **soft-encoded ab bins**.
📂 Main files:
- `data_and_preprocess.ipynb` – dataset download and preprocessing utilities  
- `training_and_eval_v2.py` – training, validation, checkpointing  
- `visualization.ipynb` – inference and visualization
- 
### 📥 1) Clone the repository
```bash
git clone https://github.com/OmerBibi/Food101-Image-Colorization-with-U-Net-ResNet50
cd Food101-Image-Colorization-with-U-Net-ResNet50
```
### 🐍 2) Environment setup (recommended)
```bash
conda create -n foodcolor python=3.10 -y
conda activate foodcolor
```
### 📦 3) **Install dependencies:**
   ```bash
  pip install torch torchvision numpy scikit-image matplotlib scikit-learn pillow tqdm
   ```
⚠️ Notes:
* For GPU training, install a CUDA-enabled PyTorch build matching your CUDA version.
### 🗂️ 4) **Folder structure (important):**
The code expects the following layout:
   ```csharp
Food101-Image-Colorization-with-U-Net-ResNet50/
├─ artifacts/
│  └─ food101_step10_sigma5_T042/
│     ├─ ab_centers_k259.npy
│     ├─ ab_weights_k259.npy
│     └─ train_runs/
│        └─ long_run_45/
│           └─ checkpoints/
│             ├─ best_ep009_loss1.4589.pt
│           ├─ viz/
│           └─ strips/
├─ data/
├─ data_and_preprocess.ipynb
├─ training_and_eval_v2.py
└─ visualization.ipynb
```
✅ Make sure the `artifacts/food101_step10_sigma5_T042/` folder exists and contains the required .npy files and checkpoints.
### 🍔 5) **Download Food-101 dataset**:
Run the first cells of: `data_and_preprocess.ipynb`
⬇️ This will download Food-101 into the data/ directory.
💡 Alternative: running the training script will also trigger the download automatically if the dataset is missing.
### 🚀 6) **Training**:
To train or retrain the model
 ```bash
python training_and_eval_v2.py
 ```
What happens:
* Food-101 is split into train / validation
* RGB images are converted to LAB
* ab channels are soft-encoded using KNN
* Encoder is frozen for warmup, then unfrozen
* Best checkpoints are saved automatically
* Visual diagnostics are written to:
  `train_runs/.../viz/` , `train_runs/.../strips/consistency_filmstrip.png`

### 🎨 7) **Inference and visualization**:
Use `visualization.ipynb`
Typical steps inside the notebook:

1. Load ab_centers_k259.npy

2. Build the model with num_classes = 259

3. Load a checkpoint (best_epXXX_*.pt)

4. Run inference on grayscale images

5. Decode logits using annealed mean

6. Convert LAB → RGB and visualize results

🎛️Important inference parameter:

`ANNEAL_T = 0.42`
Lower values give sharper colors but may introduce artifacts.

### 🛠️ 8) **Customization**:
You can modify:

- ⚙️ Training hyperparameters in training_and_eval_v2.py

- 🧪 Preprocessing logic in data_and_preprocess.ipynb

- 🔢 Number of ab bins (K)
- - ⚠️ Changing K requires regenerating centers and weights and retraining.

### ❗ 9) **Common issues**:
- 📁 Missing artifacts - Check folder name: food101_step10_sigma5_T042

- 🧯 CUDA out-of-memory - Reduce batch size

- 🎭 Desaturated or unstable colors - Tune ANNEAL_T during inference
  
---
*Inspired by the "Colorful Image Colorization" paper (Zhang et al.) and built for the Food-101 Challenge.*
