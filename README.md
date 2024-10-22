
# Kaggle RSNA Breast Cancer Detection - Silver Medal Solution

This project focuses on detecting breast cancer based on CT imaging data, formulated as a binary classification problem. The competition evaluates solutions using the pF1 score.

## Steps for the Solution

### 1. Data Extraction
The CT images cannot be directly used in standard formats, so we extract them using the **NVIDIA DALI** framework for fast processing.
- **Run**: `step1_extract_raw_data.ipynb`
- This step mirrors the code from the [open-source solution](https://www.kaggle.com/code/lucasrr/rsna-generate-1024x1024-data).

### 2. Cropping the Images
To remove unnecessary blank areas from the CT scans, edge detection is employed to identify and crop out irrelevant parts, focusing on the breast region.
- **Run**: `step2_crop_data.ipynb`

### 3. Convert to TFRecord
Transform the cropped image data into TFRecord format for efficient loading during model training.
- **Run**: `step3_tfrecord.ipynb`

### 4. Model Training
We train the model using **ConvNextV2**, employing data augmentation techniques (rotation, brightness adjustments, flipping, and cropping). The training is optimized using the **AdamW** optimizer with **Stochastic Weight Averaging (SWA)** across five cross-validation folds.
- **Run**: `step4_train.ipynb`

### 5. Model Inference
Using the five trained models, we generate predictions. The results are averaged, and the top 2.1% quantile of probability values is used as the threshold for binary classification.
- **Run**: `step5_infer.ipynb`

## Final Result
The ensemble model achieves a final ranking of **top xx%,** placing in the top xxx4% of the competition.

