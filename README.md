---

# Pediatric-Sleep-Apnea-SpO₂-Analysis

This repository provides tools for preprocessing SpO₂ signals, organizing data for cross-validation, and evaluating deep learning models for assessing pediatric sleep apnea severity. The **SpO₂_Preprocessing_CHAT** notebook automates data downloads and structuring, while **ResNet_CNNBiLSTMAttention.ipynb** facilitates model training and evaluation. Below are setup instructions, usage details, and file structure guidelines.

---

## Table of Contents

1. [Introduction](#introduction)
2. [File and Directory Structure](#file-and-directory-structure)
3. [Setup and Installation](#setup-and-installation)
4. [Usage](#usage)
   - [Data Preprocessing](#data-preprocessing)
   - [Cross-Validation Preparation](#cross-validation-preparation)
   - [Model Training and Evaluation](#model-training-and-evaluation)
5. [Additional Notes](#additional-notes)

---

### Introduction

This repository supports the research publication ["Deep learning-enabled analysis of overnight peripheral oxygen saturation signals for pediatric obstructive sleep apnea severity assessment"](https://doi.org/10.1038/s41598-024-67729-9) (DOI: 10.1038/s41598-024-67729-9). It enables deep learning model evaluation for assessing pediatric sleep apnea severity using SpO₂ signals. The **SpO₂_Preprocessing_CHAT** notebook handles data extraction, event detection, and segmentation, while **ResNet_CNNBiLSTMAttention** evaluates ResNet and CNN-BiGRU-Attention models.

---

### File and Directory Structure

**Required files and directories before running the notebooks**:

1. **Demographic and Polysomnographic Information Files**:
   - `chat-baseline-dataset-0.12.0.csv`
   - `chat-followup-dataset-0.12.0.csv`

2. **XML Folder** (in `./CHAT_dataset/XML`):
   - Stores XML annotation files for apnea and oxygen desaturation events:
   ```plaintext
   XML/
   ├── baseline/    # Baseline group XML files
   └── followup/    # Follow-up group XML files
   ```

3. **CSV Folder** (created in Google Drive or `./CHAT_dataset/CSV`):
   - Saves apnea and desaturation event details in CSV format:
   ```plaintext
   CSV/
   ├── ap_ends/     # Apnea event end times
   ├── od_nadirs/   # Oxygen desaturation nadir times
   └── od_starts/   # Oxygen desaturation start times
   ```

4. **Processed Data Folder**:
   - Stores numpy arrays with preprocessed data:
   ```plaintext
   processed_data/
   ├── AHIs.npy      # AHI values per signal
   ├── IDs.npy       # Subject IDs
   └── data.npy      # 20-minute SpO₂ segments with indexing
   ```

5. **Folds Folder** (for 3-fold cross-validation):
   - Each fold contains training, validation, and test data arrays for evaluation:
   ```plaintext
   Folds/
   ├── fold_1/
   │   ├── OCM_testAHI1.npy 
   │   ├── OCM_testArray1.npy 
   │   ├── OCM_testIDs1.npy 
   │   ├── OCM_trainAHI1.npy 
   │   ├── OCM_trainArray1.npy 
   │   ├── OCM_trainIDs1.npy 
   │   ├── OCM_valAHI1.npy 
   │   ├── OCM_valArray1.npy 
   │   └── OCM_valIDs1.npy
   ├── fold_2/ (similar structure)
   └── fold_3/ (similar structure)
   ```

6. **Models Folder**:
   - Contains pretrained weights for each fold:
   ```plaintext
   Models/
   ├── fold1_weights.h5
   ├── fold2_weights.h5
   ├── fold3_weights.h5
   └── attention_weights.pth    # PyTorch weights for attention extraction
   ```

---

### Setup and Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/username/repo_name.git
   ```
2. Install required libraries
   
3. Ensure directory paths in notebooks are correctly updated to reference all files.

---

### Usage

#### 1. Data Preprocessing

**SpO₂_Preprocessing_CHAT Notebook**:

- Open `SpO₂_Preprocessing_CHAT.ipynb` in Google Colab.
- Set the **webpage number** (1-5) and **dataset group** (baseline or followup).
- Execute cells to download `.edf` files, detect apnea/desaturation events, and save processed data in `.npy` format in Google Drive:
  - **AHI arrays**: Apnea-Hypopnea Index values per signal.
  - **ID arrays**: Subject IDs.
  - **Data arrays**: 20-minute segmented SpO₂ signals.

#### 2. Cross-Validation Preparation

- After preprocessing all pages for both groups, upload processed data for cross-validation.
- Merge `.npy` arrays into structured files for n-fold cross-validation in **Folds**.

#### 3. Model Training and Evaluation

The `ResNet_CNNBiLSTMAttention.ipynb` notebook includes:

- **ResNet** for feature extraction (TensorFlow).
- **CNN-BiGRU-Attention** for interpretability (TensorFlow and PyTorch).

Steps:

1. **Load Pretrained Weights**:
   - Upload pretrained weights from `./Models` for each fold.
   
2. **Evaluate Model Performance**:
   - Run evaluations on cross-validation data with metrics like accuracy, precision, and recall.
   
3. **Attention Score Visualization**:
   - Load `.pth` weights for CNN-BiGRU-Attention in PyTorch.
   - Extract and plot attention scores to observe model focus on SpO₂ signal features.

---

### Additional Notes

- **Data Validation**: Preprocessing validates apnea and desaturation event quality.
- **Interpretability**: Attention visualizations offer insights into model focus on SpO₂ signal features.

This repository is a complete toolkit for processing SpO₂ data and evaluating deep learning models for pediatric sleep apnea research with interpretable model output.

---

**Repository Description**:  
Repository for ["Deep learning-enabled analysis of overnight peripheral oxygen saturation signals for pediatric obstructive sleep apnea severity assessment"](https://doi.org/10.1038/s41598-024-67729-9). It includes data preprocessing, segmentation, and deep learning model evaluation pipelines for SpO₂-based pediatric OSA severity assessment.

---

