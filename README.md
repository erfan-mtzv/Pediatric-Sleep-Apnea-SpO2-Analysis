# Pediatric-Sleep-Apnea-SpO2-Analysis

This repository provides notebooks for preprocessing SpO₂ signals, organizing data for k-fold cross-validation, and evaluating deep learning models for pediatric sleep apnea severity assessment. The preprocessing notebook automates downloading and organizing data, while the model evaluation notebook enables the training and analysis of ResNet and CNN-BiGRU-Attention models. Detailed setup instructions, usage steps, and file structure guidance are below.

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

This repository accompanies the research publication, ["Deep learning-enabled analysis of overnight peripheral oxygen saturation signals for pediatric obstructive sleep apnea severity assessment"](https://doi.org/10.1038/s41598-024-67729-9) (DOI: 10.1038/s41598-024-67729-9). It supports the analysis and evaluation of pediatric sleep apnea severity by applying deep learning models to segmented SpO₂ signals. The preprocessing workflow includes signal extraction, apnea and desaturation event detection, and segment preparation, while the model evaluation notebook provides interpretability through the CNN-BiGRU-Attention network’s attention layer.

---

### File and Directory Structure

**Files and directories required before running the notebooks**:

1. **Demographic and Polysomnographic Information Files**:
   - `chat-baseline-dataset-0.12.0.csv`
   - `chat-followup-dataset-0.12.0.csv`

2. **XML Folder** (in `./CHAT_dataset/XML`):
   - Contains XML annotation files for apnea and oxygen desaturation events:
   ```plaintext
   XML/
   ├── baseline/    # Baseline group annotation XML files
   └── followup/    # Follow-up group annotation XML files
   ```

3. **CSV Folder** (created in your Google Drive or `./CHAT_dataset/CSV`):
   - Stores CSV files with apnea and desaturation event details:
   ```plaintext
   CSV/
   ├── ap_ends/     # Apnea event end times for each signal
   ├── od_nadirs/   # Candidate oxygen desaturation nadir times for each signal
   └── od_starts/   # Candidate oxygen desaturation start times for each signal
   ```

4. **Processed Data Folder**:
   - Contains numpy arrays with preprocessed data:
   ```plaintext
   processed_data/
   ├── AHIs.npy      # Array of AHI values per signal
   ├── IDs.npy       # Array of subject IDs for signals
   └── data.npy      # Array of 20-minute segmented SpO₂ signals with segment indexing
   ```

5. **Folds Folder** (used for 3-fold cross-validation):
   - Each fold contains training, validation, and test data arrays for model evaluation:
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
   
2. Install necessary libraries
   
3. Update directory paths in notebooks as needed to ensure all files are correctly referenced.

---

### Usage

#### 1. Data Preprocessing

**SpO2_preprocessing_CHAT Notebook**:

- Open `SpO2_preprocessing_CHAT.ipynb` in Google Colab.
- Set the **webpage number** (1-5) and **dataset group name** (baseline or followup) for each run.
- Execute cells to download `.edf` files, identify apnea/desaturation events, and save processed data as `.npy` arrays in Google Drive:
  - **AHI arrays**: Apnea-Hypopnea Index values for each signal
  - **ID arrays**: Subject IDs for each signal
  - **Data arrays**: 20-minute segmented SpO₂ signals with segment indexing to avoid mismatches

#### 2. Cross-Validation Preparation

- After preprocessing all pages for both groups upload processed data and apply n-fold cross validation.
The last cells of the SpO2_preprocessing_CHAT notebook can be used for merging `.npy` arrays into organized for applying n-fold cross-validation, which are then saved in the **Folds folder**.

#### 3. Model Training and Evaluation

The `ResNet_CNNBiLSTMAttention.ipynb` notebook implements:

- **ResNet** for feature extraction (TensorFlow)
- **CNN-BiGRU-Attention** network for interpretability (TensorFlow and PyTorch)

Steps:

1. **Load Pretrained Weights**:
   - Upload pretrained weights from `./Models` for each fold (e.g., `attention_model_fold_1_weights.h5`).
   
2. **Evaluate Model Performance**:
   - Assess performance on k-fold cross-validation data with metrics like accuracy, precision, and recall.
   
3. **Attention Score Visualization** (CNN-BiGRU-Attention in PyTorch):
   - Load `.pth` weights for the CNN-BiGRU-Attention model in PyTorch.
   - Extract and plot attention scores to observe model focus on SpO₂ signal features.

---

### Additional Notes

- **Directory Management**: The notebook deletes intermediate `.edf` files post-processing to conserve space.
- **Data Validation**: Preprocessing notebook validates apnea and desaturation event data quality.
- **Attention Layer Insight**: Visualizations provide interpretability by highlighting areas of focus in SpO₂ signals during prediction.

This repository is a complete toolkit for preparing SpO₂ data and evaluating deep learning models for pediatric sleep apnea research, with structured and interpretable model output.

**Repository Description**:   
Repository for the published paper titled ["Deep learning-enabled analysis of overnight peripheral oxygen saturation signals for pediatric obstructive sleep apnea severity assessment"](https://doi.org/10.1038/s41598-024-67729-9) (DOI: 10.1038/s41598-024-67729-9). This repository includes data preprocessing and model evaluation pipelines, designed for SpO₂ signal processing, segmentation, and deep learning model applications for pediatric OSA severity assessment. 
 
