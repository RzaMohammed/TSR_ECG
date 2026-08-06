# TSRNet: Testing Results and Areas for Improvement

This document outlines the results of our local pipeline testing on the `TSRNet` architecture, as well as several architectural and programmatic areas for improvement identified during the execution process.

## 1. Testing Setup & Outputs

We executed the full lifecycle of the project using synthetic dummy data to verify the functionality of the dataset loading, training loop, and inference evaluation.

### A. Data Generation
- **Process**: Since the PTB-XL dataset was not provided, we generated a synthetic dataset with $N=4$ samples (`train.npy`, `test.npy`, `label.npy`). 
- **Fix Applied**: We applied Gaussian noise to the samples so that they were not entirely identical. (Without noise, the anomaly scores were exactly the same, causing a division-by-zero error during score normalization).

### B. Model Training (`train.py`)
- **Command**: `python train.py --epochs 2 --batch_size 2 --save_model 1`
- **Output**: 
  - The model successfully initialized and calculated **2.17 M total trainable parameters**.
  - The `Total_Loss` decreased rapidly over the epochs (e.g., `1.21` -> `0.99` -> `0.95`).
  - The model hit a perfect `1.0 AUC` on the dummy testing data immediately at Epoch 0. 
  - **Checkpoint saved**: Because the AUC hit `1.0` and didn't exceed that threshold in subsequent epochs, the system only saved the first checkpoint at `ckpt/TSRNet-0.pt`.

### C. Inference Testing (`test.py`)
- **Command**: `python test.py --load_path ckpt/TSRNet-0.pt`
- **Output**: 
  - Loaded the saved model successfully.
  - Successfully iterated over the masked testing sequence.
  - Outputted a final **Detection AUC: 1.0**.

---

## 2. Room for Improvement

During the execution and code review, several flaws and areas for optimization were identified in the codebase:

### Architectural & Code Fixes
1. **Divide-by-Zero Vulnerability in AUC Normalization**:
   - In both `train.py` and `test.py`, the code normalizes scores using:
     ```python
     scores = (scores - min_anomaly_score) / (max_anomaly_score - min_anomaly_score)
     ```
   - **Improvement**: If the model predicts the exact same score for all samples (as we saw initially), this causes a `ValueError: Input contains NaN`. A safety check (`if max == min:`) should be added to prevent this crash.

2. **Rigid Model Checkpoint Logic**:
   - The script only saves a model if `auc_result > old_auc_result`. 
   - **Improvement**: If the model reaches a perfect score early on (or hits a plateau), it stops saving entirely, even if the training loss is still decreasing and the model is becoming more confident. It should save the "best" model based on AUC *and* save the final epoch's model regardless.

3. **Missing Import Bug**:
   - The original `train.py` script forgot to import `TSRNet_time`, causing a `NameError` crash when run in the default time-domain mode. 
   - **Improvement**: We fixed this locally, but robust unit testing or a simple linter (like `flake8` or `mypy`) would have caught this instantly.

### Machine Learning Optimizations
4. **Hardcoded Hyperparameters**:
   - Many critical variables, like the sliding window patch length (`patch_length = time_length // 100`) and the STFT window parameters (`fs=500`, `nperseg=125`), are hardcoded deep inside the training loops and dataloader. 
   - **Improvement**: These should be extracted to the `argparse` configuration so researchers can tune the sliding windows and frequency bands without modifying the core files.

5. **Lack of Validation Split**:
   - The `train.py` script evaluates its "best" model directly on the `TestSet`. 
   - **Improvement**: In rigorous ML pipelines, it should evaluate against a separate `ValidationSet` during training, and keep the `TestSet` completely unseen until final inference to prevent data leakage and overfitting.
