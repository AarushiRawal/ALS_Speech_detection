# Dysarthric Speech Detection using HuBERT + LoRA

## Overview
This project implements a deep learning model to classify speech as **dysarthric** or **normal (control)** using a pretrained **HuBERT** model with **LoRA (Low-Rank Adaptation)** for efficient fine-tuning.

---

## Features
- End-to-end speech classification pipeline  
- Transfer learning using pretrained HuBERT  
- Efficient fine-tuning with LoRA  
- Handles class imbalance using weighted loss  
- Uses raw audio (no handcrafted features like MFCC)  

---

## Tech Stack
- Python  
- PyTorch  
- HuggingFace Transformers  
- NumPy, Pandas  
- Librosa / Torchaudio  


---

## Methodology

### Data Preprocessing
- Audio resampling and normalization  
- Padding/truncation to fixed length  
- Optional augmentation  

### Feature Extraction
- Raw waveform input  
- HuBERT extracts contextual speech representations  

### Model Architecture
- Pretrained HuBERT backbone  
- LoRA layers for parameter-efficient fine-tuning  
- Classification head for binary output  

### Training
- Loss: `BCEWithLogitsLoss` (with class weights)  
- Optimizer: AdamW  
- Learning rate scheduling  
- Mixed precision training (AMP + GradScaler)  

---

## Evaluation Metrics
- Accuracy  
- Precision  
- Recall  
- F1 Score  

---

## Results
| Metric    | Value |
|----------|------|
| Accuracy | 81%  |
| Precision| 84%  |
| Recall   | 82%  |
| F1 Score | 82%  |
