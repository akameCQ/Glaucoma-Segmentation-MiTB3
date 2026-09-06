# 👁️ Glaucoma Detection & Segmentation: An End-to-End Clinical Deep Learning Journey
[![Colab Environment](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/) 
![Python](https://img.shields.io/badge/Python-3.8%2B-blue) 
![PyTorch](https://img.shields.io/badge/PyTorch-Red) 
![YOLOv8](https://img.shields.io/badge/YOLO-v8-yellow)
![Transformers](https://img.shields.io/badge/Architecture-MiT--B3-green)
This repository contains a comprehensive **Research & Engineering Workspace** dedicated to the automated detection and segmentation of Glaucoma from retinal fundus images (REFUGE2 and DRISHTI-GS datasets). 
Instead of presenting just a finalized model, this project documents a highly iterative **Machine Learning Engineering journey**. It details every technical challenge encountered—from bounding box clipping to extreme VRAM consumption—and the algorithmic solutions implemented to overcome them.
---
## 🚀 1. The Two-Stage Architecture
The system operates in a precision-focused, two-stage pipeline:
1. **Localization:** A YOLOv8 object detection model scans the raw, high-resolution fundus image and draws a bounding box precisely around the Optic Disc.
2. **Segmentation:** A `Mix Vision Transformer (MiT-B3)` based U-Net takes the cropped Optic Disc and performs pixel-perfect segmentation of the Disc and Cup structures.
---
## 🛠️ 2. Clinical Image Processing (The "Green Channel" Strategy)
Standard RGB images carry noise that confuses segmentation boundaries. To fix this, I engineered a clinical-grade preprocessing pipeline:
* **Green-Channel Extraction:** Extracted only the Green channel (G) from the RGB matrix, as it provides the highest biological contrast for retinal blood vessels and optic structures.
* **CLAHE Enhancement:** Applied Contrast Limited Adaptive Histogram Equalization locally to sharpen the boundary between the Optic Disc and Cup.
* **No-Stretch Padding:** Resizing medical images stretches and deforms anatomical shapes. I developed an aspect-ratio-preserving padding function that centers the image on a `512x512` canvas using black bars, ensuring 0% biological distortion.
---
## 🔬 3. YOLOv8 Cropping Strategies & Diagnostics
During development, I noticed the segmentation model struggled with edge pixels. I iterated through multiple cropping strategies:
* **Attempt 1 (15% Tight Crop):** Zoomed in closely on the YOLO bounding box. *Result:* Mask boundaries occasionally bled out of the frame.
* **Attempt 2 (40% Wide FOV Crop):** Expanded the YOLO margin to 40%. *Result:* Ensured full structural containment, giving the MiT-B3 model enough background context to map the disc accurately.
* **Fallback Protocol:** If YOLO fails to detect a disc (due to a severely corrupted image), the system automatically triggers a mathematically centered static crop to prevent pipeline crashes.
---
## 🧪 4. Model Architecture & Loss Engineering
Transitioning from standard CNNs to a **Mix Vision Transformer (MiT-B3)** required careful parameter tuning:
* **Input Channel Optimization:** Shifted the MiT-B3 architecture from a 3-channel RGB input to a highly optimized 1-channel Grayscale setup, reducing computational overhead while maximizing structural focus.
* **Loss Function Evolution:**
  1. *Baseline:* `0.5 CrossEntropy + 0.5 Dice Loss`
  2. *Boundary-Focused:* `0.2 CrossEntropy + 0.8 Dice Loss`
  3. *Final Fine-Tuning:* Developed a custom **Hybrid `Focal + Dice` Loss**. The Focal Loss heavily penalizes the model for hard-to-classify pixels, while the Dice Loss maintains the overall topological shape of the optic cup.
---
## 🧬 5. SOTA Data Augmentation (Albumentations)
To prevent memorization and simulate biological variance, two distinct transformation pipelines were built:
* **Heavy Training Pipeline:** Applied `ElasticTransform` and `GridDistortion` to simulate retinal warping. Added `RandomBrightnessContrast` and `GaussNoise` to build immunity against various hospital camera sensors.
* **Light Fine-Tuning Pipeline ("Polishing"):** In the final epochs, geometric distortions were disabled. Only subtle sharpening and lighting variations were kept to finalize the model's confidence.
---
## ⚙️ 6. Hardware & Memory Optimization (A100 Turbo)
Processing high-resolution medical data quickly exhausts VRAM. I engineered the `DataLoader` and training loop to maximize Google Colab's A100 GPU:
* Integrated `torch.cuda.amp.autocast()` and `GradScaler()` for **Mixed Precision Training**, effectively halving VRAM usage.
* Maximized throughput using `batch_size=32`, `num_workers=8`, `pin_memory=True`, and `prefetch_factor=2`.
---
## 🧠 7. Advanced Interpretability: Monte Carlo (MC) Dropout
In medical AI, predicting a mask is not enough; the doctor needs to know *how confident* the AI is. Instead of standard inference, I implemented an **MC-Dropout** module:
* The model runs **20 stochastic forward passes** per image with active dropout layers.
* **Outputs Generated:** 
  1. A Mean Prediction Mask.
  2. A **Confidence Score** (%).
  3. An **Uncertainty Heatmap** (Variance Map) highlighting the exact pixels where the model hesitated (usually the blurry border between the disc and the cup).
---
## 📊 8. Automated Performance Auditing
The notebook includes an intelligent evaluation block that categorizes predictions based on their Dice Score (e.g., *Flawless 95-100%*, *Excellent 90-95%*, *Challenging 60-80%*). It automatically generates Matplotlib panels plotting the Raw Image, Ground Truth, Model Prediction, and Uncertainty Heatmap side-by-side for rapid visual debugging.
---
*Note: To ensure reproducibility, a **Consolidated End-to-End Pipeline** compiling all the optimal strategies discovered in this research is provided at the very end of the notebook.*
