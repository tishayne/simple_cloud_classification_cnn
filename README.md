# Cloud Cover Classification using Xception (Transfer Learning + Fine-Tuning)

This project aims to classify sky images into **Clear**, **Cloudy**, and **Overcast** using a Convolutional Neural Network (CNN) with **Xception** as the base model.  
The model was trained using transfer learning and further fine-tuned by unfreezing:
- the top **1/3 of the layers**
- the top **2/3 of the layers**
- **all layers**

### ✨ Key Result  
Fine-tuning **44 layers** produced the strongest performance:  
➡ **93.86% Test Accuracy**

---

## Project Overview

The workflow includes:
- Removing color-shifted images
- Removing raindrop-affected images using a lightweight **MobileNet + SVM detector**
- Manually labelling the cleaned images into 3 cloud cover types
- Training and fine-tuning an Xception CNN
- Saving the best models automatically for evaluation

---

## 📁 Folder Structure

data/
│
├── raw/ # Original unprocessed images
│
├── raindrop_clean_split/ # Manual labelling of clean and raindrop images
│ ├── clean/ # Manually labelled images without raindrops
│ └── raindrop/ # Manually labelled images with raindrops
│
├── dataset_split/ # Manual labelling of cloud types
│ ├── clear/
│ ├── cloudy/
│ └── overcast/
│
└── dataset/ # FINAL dataset created automatically
├── train/
├── val/
└── test/


data/
├─ raw/
├─ raindrop_clean_split/
│  ├─ clean/
│  └─ raindrop/
├─ dataset_split/
│  ├─ clear/
│  ├─ cloudy/
│  └─ overcast/
└─ dataset/
   ├─ train/
   ├─ val/
   └─ test/


---

## 🧠 Two Manual Labelling Steps (Important!)

| Folder | Label What? | Purpose |
|--------|-------------|---------|
| `raindrop_clean_split/` | clean / raindrop | Train SVM raindrop detector |
| `dataset_split/` | clear / cloudy / overcast | Train CNN cloud classifier |

> **Do not manually modify anything inside `/dataset/`** — this folder is created automatically and overwritten when running the split notebook.

---

##  With Retraining From Scratch use this order:

| Step | Notebook | Purpose |
|------|----------|---------|
| 1 | `01_train_raindrop_svm.ipynb` | Train MobileNet + SVM model using `raindrop_clean_split/` |
| 2 | `02_preprocess_dataset_with_svm.ipynb` | Use SVM to filter `data/raw/` and remove raindrop images |
| 3 | `03_train_val_test_split.ipynb` | Create the final `dataset/train/`, `dataset/val/`, `dataset/test/` |
| 4 | `04_train_xception_and_finetune.ipynb` | Train & fine-tune Xception with different frozen layers |
| 5 | `05_test_and_visualise_model.ipynb` | Evaluate and visualise results |

---

## 💾 Model Saving

The system automatically saves the strongest model for each experiment to:

📁 `models/best_models/`

This avoids retraining and allows consistent comparison across fine-tuning strategies.

