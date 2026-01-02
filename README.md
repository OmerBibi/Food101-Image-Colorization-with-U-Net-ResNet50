# 🍔 Food Colorization using Deep Networks: Grayscale to Vibrant Color
### *Advanced Deep Learning Colorization using U-Net & ResNet-50*

<p align="center"> <img src="https://github.com/user-attachments/assets/a82e6132-8949-4641-8513-6786c188b3e2" alt="Colorized Food Commercial" width="600" /> </p>

<p align="center"> 

---

## 📖 Introduction
This project implements a sophisticated colorization pipeline using a **U-Net** architecture with a **ResNet-50** backbone, trained on the **Food-101** dataset. By leveraging the CIELAB color space, the model learns to predict the $a$ and $b$ (color) channels given the $L$ (luminance) channel.

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
<img width="560" height="1120" alt="consistency_filmstrip_10_epoch" src="https://github.com/user-attachments/assets/0bb880d1-4ab6-44ee-a2a2-efe2584f6a4c" />

---

## 📂 Repository Contents
* `data_and_preprocess.ipynb`: Data preparation, color binning, and weight calculation.
* `training_and_eval_v2.py`: The main training script with validation logic.
* `visualization.ipynb`: Inference tools for images, videos, and entropy visualization.

---

## 💻 Setup & Usage

1. **Clone the repo:**
   ```bash
   git clone https://github.com/OmerBibi/Food101-Image-Colorization-with-U-Net-ResNet50`
    ```
2. **Install requirements:**
   ```bash
   pip install torch torchvision numpy scikit-image matplotlib scikit-learn
   ```
3. **Weights:** Ensure `ab_weights_k259.npy`, `ab_centers_k259.npy` and `best_ep009_loss1.4589.pt` are in the `artifacts/` folder.
4.  Run the first cell of `data_and_preprocess.ipynb` to download the dataset
5.  Use `visualization.ipynb` for visualization and inference
6.  If you want to train/change pre-processing, make sure to check `data_and_preprocess.ipynb` and `training_and_eval_v2.py`

---
*Inspired by the "Colorful Image Colorization" paper (Zhang et al.) and built for the Food-101 Challenge.*
