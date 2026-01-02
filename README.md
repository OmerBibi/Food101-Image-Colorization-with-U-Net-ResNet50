

# 🎨 Deep Image Colorization with U-Net & ResNet-50

Transform black-and-white memories into vibrant colors using deep learning. This project implements a sophisticated colorization pipeline using a **U-Net** architecture with a **ResNet-50** encoder, trained on the Food-101 dataset.

---

## 🚀 Key Features

* **U-Net Architecture**: Features skip connections to preserve spatial details from the input grayscale image.
* **ResNet-50 Backbone**: Leverages ImageNet pre-trained weights for robust feature extraction.
* **CIELAB Color Space**: Predicted in the  space, where the model learns to predict the  (green-red) and  (blue-yellow) channels given the  (lightness) channel.
* **Probabilistic Distribution**: Instead of simple regression, the model predicts a probability distribution over quantized color bins (K=259) to handle multimodal color uncertainty.
* **Annealed Mean Decoding**: Uses a temperature-parameterized softmax () to produce vibrant yet natural color outputs.

---

## 🛠️ Architecture Detail

The model consists of an encoder (downsampling) and a decoder (upsampling):

| Component | Description |
| --- | --- |
| **Encoder** | Modified ResNet-50. The first layer is adapted to accept 1-channel grayscale input. |
| **Bridge** | Two `ConvGNReLU` layers (GroupNorm + ReLU) with 1024/2048 channels. |
| **Decoder** | Symmetric up-blocks using bilinear interpolation and skip connections from the encoder. |
| **Head** |  Convolution mapping 64 features to 259 color bin classes. |

---

## 📊 Preprocessing & Training

The pipeline involves sophisticated data preparation:

1. **Quantization**: Color space is divided into 259 discrete bins based on the distribution of the Food-101 dataset.
2. **Class Rebalancing**: Uses rare-color weighting to prevent the model from always predicting desaturated "safe" colors (like gray or brown).
3. **Soft Labels**: Training targets are generated using Gaussian smoothing () over the 5 nearest color neighbors.

---

## 💻 Usage

### 1. Prerequisites

```bash
pip install torch torchvision numpy scikit-image matplotlib pillow scikit-learn

```
### 2. Training

Run the data_and_preproces notebook which Handles dataset loading, color bin quantization, and class weight calculation:
```bash
data_and_preprocess.ipynb

```

### 3. Training

Run the training script which includes validation and real-time visualization:

```bash
python training_and_eval_v2.py

```

### 4. Inference & Visualization

Use the `visualization.ipynb` notebook to load a checkpoint and colorize custom images.

```python
# Model initialization
model = UNetResNet50(num_classes=259)
model.load_state_dict(torch.load('best_model.pt'))
model.eval()

```

---

## 🖼️ Results

Below is a "consistency filmstrip" showing the model's evolution during training:
<img width="560" height="1120" alt="consistency_filmstrip_10_epoch" src="https://github.com/user-attachments/assets/0bb880d1-4ab6-44ee-a2a2-efe2584f6a4c" />

And few more examples from the test set:

<img width="450" height="1797" alt="eval_predictions_1" src="https://github.com/user-attachments/assets/7afb2432-cf7f-4c82-8b96-8d093ed2bf5d" />
<img width="450" height="1797" alt="eval_predictions_2" src="https://github.com/user-attachments/assets/54b3ece4-7d31-47ef-916f-5ad68c6bd6dc" />


---

## 📂 Project Structure

* `data_and_preprocess.ipynb`: Handles dataset loading, color bin quantization, and class weight calculation.
* `training_and_eval_v2.py`: Core training loop featuring weighted loss and validation tracking.
* `visualization.ipynb`: Tools for model evaluation, ranking validation results, and generating side-by-side comparisons.

---

## 📜 References

* Dataset: [Food-101](https://www.google.com/search?q=https://www.vision.ee.ethz.ch/datasets_extra/food-101/).
* Inspired by: *Colorful Image Colorization* (Zhang et al.).

---
