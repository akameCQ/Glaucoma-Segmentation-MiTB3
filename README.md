# 👁️ Glaucoma Detection & Segmentation: An Experimental Deep Learning Journey
[![Colab Environment](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/) 
![Python](https://img.shields.io/badge/Python-3.8%2B-blue) 
![PyTorch](https://img.shields.io/badge/PyTorch-Red) 
![YOLOv8](https://img.shields.io/badge/YOLO-v8-yellow)
This repository contains a comprehensive **Google Colab workspace** dedicated to the automated detection and segmentation of Glaucoma from retinal fundus images. 
Unlike standard pipelines, this project documents an **extensive experimental journey**—testing different loss functions, data augmentation strategies, YOLO cropping margins, and fine-tuning techniques to achieve the most robust clinical results.
## 🚀 The Pipeline Overview
1. **Stage 1 (Detection):** YOLOv8 is used to intelligently detect and isolate the Optic Disc.
2. **Stage 2 (Preprocessing):** Green-channel extraction, CLAHE filtering, and adaptive padding.
3. **Stage 3 (Segmentation):** A `Mix Vision Transformer (MiT-B3)` based U-Net segments the Optic Disc and Cup.
4. **Stage 4 (Interpretability):** Monte Carlo (MC) Dropout is applied to generate clinical Uncertainty Heatmaps and Confidence Scores.
---
## 🧪 The Training Journey & Experiments
This project was built iteratively on a **Google Colab A100 GPU**. The notebook acts as a log of the key experiments and iterations conducted during development:
### 1. Loss Function Optimization
Finding the right balance between pixel-wise accuracy and overall shape preservation was critical. I experimented with multiple loss combinations:
*   **Attempt 1:** `0.5 CrossEntropy + 0.5 Dice Loss` (Baseline).
*   **Attempt 2:** `0.2 CrossEntropy + 0.8 Dice Loss` (Improved boundary detection).
*   **Final "Cila" (Fine-Tuning) Attempt:** Implemented a custom **Hybrid Loss (`Focal Loss + Dice Loss`)** to force the model to focus on the hardest-to-predict boundary pixels.
### 2. YOLO Cropping Strategies (Zoom-In vs. Zoom-Out)
*   Initially tested a tight **15% padding** around the YOLO bounding box.
*   Discovered that some masks were bleeding out of the frame. Transitioned to a **40% wide-angle padding** strategy (Zoom-out) to ensure the entire biological structure fits into the 512x512 canvas without stretching.
### 3. Data Augmentation Strategies (Albumentations)
*   **Heavy Augmentation (Main Training):** Applied extreme biological distortions (`ElasticTransform`, `GridDistortion`) to simulate varied retinal topologies, alongside heavy noise and color jittering to prevent overfitting.
*   **Light Augmentation (Fine-Tuning):** Created a separate "Cila" (Polishing) dataloader that removes geometric distortions and only applies subtle brightness/sharpening changes to solidify the model's confidence in the final epochs.
### 4. Hardware & VRAM Optimization (A100 Turbo Mode)
*   Transitioned the model from 3-channel RGB to **1-channel Grayscale** processing to optimize VRAM and focus purely on structure.
*   Implemented `torch.cuda.amp.autocast()` and `GradScaler` for Mixed Precision Training.
*   Maximized Colab's A100 GPU utilizing `batch_size=32`, `num_workers=8`, and `pin_memory=True`.
---
## 🖼️ Rich Visual Outputs & Analysis
The included `.ipynb` notebook is heavily focused on visual evaluation. Instead of just printing numbers, the code generates comprehensive visual panels:
*   **Monte Carlo (MC) Dropout Analysis:** The notebook runs 20 forward passes with active dropout layers to generate an **Uncertainty Heatmap** (showing high-variance pixels) and a **Confidence Score**.
*   **Side-by-Side Comparisons:** Ground Truth (Doctor annotations) vs. Model Predictions are plotted together.
*   **Categorical Evaluation:** The evaluation loop automatically groups test images by their Dice Score (e.g., *Flawless 95-100%*, *Challenging 60-80%*) and plots 3 samples from each category to visually debug where the model struggles and where it excels.
*   **Bounding Box Debugging:** Visualizations comparing the tight bounding boxes vs. wide-angle padding strategies directly on the raw fundus images.
