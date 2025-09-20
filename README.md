# Visual Decoding from EEG Signals

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

A novel, end-to-end deep learning pipeline that reconstructs high-fidelity images from raw EEG brain signals, bridging the gap between neural activity and generative AI.

---

## Overall Architecture

![Project Architecture]
<img width="2279" height="996" alt="image" src="https://github.com/user-attachments/assets/698fc373-fb90-445f-b1ff-191022c0e621" />


## 📖 Introduction

This project explores the frontier of brain-computer interfaces by creating a system to understand and recreate visual stimuli directly from brain activity. By leveraging a suite of advanced deep learning models, we translate the complex, high-dimensional data of EEG signals into coherent text and depth information, which in turn guide a state-of-the-art generative model to reconstruct the original visual scene. The ultimate vision is to enable a true "thought-to-image" pipeline.

## ✨ Key Features

- **End-to-End Pipeline:** A complete, multi-stage framework from raw EEG signal processing to final image reconstruction.
- **Cross-Modal Alignment:** Pioneered a technique using **MaskedCLIP** to successfully align the latent spaces of EEG data and text embeddings, allowing the model to "understand" the semantic content of brainwaves.
- **Dual-Guidance Generation:** The final image reconstruction in **Stable Diffusion** is guided by both the decoded text prompt and a predicted **Depth Map**, leading to superior structural coherence in the output.
- **Rigorous Evaluation:** The model's architecture was validated through a comprehensive ablation study, proving the positive contribution of each component and achieving a peak performance of **24.14% SSIM**.

## ⚙️ How It Works

The pipeline processes the data in several key stages:

1.  **EEG Feature Extraction:** A **Variational Autoencoder (VAE)** and **LabRaM** model process the raw EEG signals to extract compact, meaningful vector embeddings.
2.  **Image Captioning:** A **BLIP-2** model generates descriptive text captions from the images in the training dataset.
3.  **Cross-Modal Alignment:** The core of the project. A **MaskedCLIP** model is trained to map the EEG embeddings to their corresponding text caption embeddings, learning the relationship between thoughts and language.
4.  **Text & Depth Decoding:**
    - The aligned "thought" embedding is fed into a **GPT-2** model to decode it into a full-sentence text prompt.
    - In parallel, a **GCNN** model predicts a 3D depth map of the scene directly from the EEG signal.
5.  **Image Reconstruction:** A **Stable Diffusion** model takes both the text prompt and the depth map as input to generate the final, high-fidelity reconstructed image.

## 🛠️ Technologies Used

- **Python**
- **PyTorch** & **TensorFlow**
- **Transformers** (CLIP, GPT-2, LabRaM)
- **Stable Diffusion**
- **Variational Autoencoders (VAE)**
- **Graph Convolutional Neural Networks (GCNN)**
- **Scikit-Learn**, **Pandas**, **Numpy**

## 📊 Results

The end-to-end pipeline successfully reconstructs recognizable images that are semantically aligned with the original visual stimuli. The final, fully-equipped model achieved a peak **Structural Similarity Index Measure (SSIM) of 14.32%**.
<img width="2277" height="932" alt="image" src="https://github.com/user-attachments/assets/e14d6e7b-ac82-4b1a-96fb-b8769df09c74" />



## 🚀 Setup and Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/mududevudu/Visual-Decoding-from-EEG-Signals
    cd Visual-Decoding-from-EEG-Signals
   
## 📄 License

This project is licensed under the **MIT License**.
