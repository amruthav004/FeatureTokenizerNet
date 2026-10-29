# Supplementary files for PPD_Final.ipynb

This package documents the supplied notebook for classification of the **Suicide attempt** response in `postnatal_data.csv`. The workflow uses a gated residual neural network named `FeatureTokenizerNet`, training-set SMOTE, SHAP explanations, and five-fold comparisons with Logistic Regression, Random Forest and XGBoost. The model name is retained from the code; its implementation is a feedforward network without an attention layer.

The target has three recorded categories: `No`, `Not interested to say`, and `Yes`. The code does not define a separate postpartum-depression diagnosis label. Interpret results as classification of these recorded responses. `Not interested to say` is a response category, not an intermediate severity level.

## Package contents

| File | Purpose |
| --- | --- |
| `README.md` | Setup, execution order, outputs and reproducibility notes |
| `config.yaml` | Readable reference configuration transcribed from the source |
| `config.json` | The same configuration in JSON format |
| `data_dictionary.json` | Original field names, counts, feature order and encoding mappings |
| `metadata.json` | Source hash, notebook sections and verification status |
| `requirements.txt` | Dependencies identified in imports and installation cells |

Place these files beside the original `PPD_Final.ipynb`. The notebook, dataset and checkpoint are not included in this supplementary archive. All cell numbers below refer to one-based cell positions, including the empty final cell.

## Data and preprocessing

Saved notebook output reports 1,503 rows and 11 original columns. `Timestamp` is removed, leaving nine predictors and the target. Raw target counts are 709 `No`, 335 `Not interested to say`, and 459 `Yes`. These values come from saved outputs; the CSV was not supplied for independent checking.

The notebook fills missing values with each column's mode. For object columns whose entire vocabulary belongs to `No`/`Sometimes`/`Yes` and their lowercase equivalents, it applies codes 0/1/2. Other object columns use `LabelEncoder`. In particular, `Feeling anxious` is encoded as 0 and 2, while the target uses 0=`No`, 1=`Not interested to say`, 2=`Yes`. Exact field names, raw category counts and the expected feature order are provided in `data_dictionary.json`. No feature scaling is applied.

The holdout split is stratified with `test_size=0.2` and `random_state=42`. Saved output gives 1,202 training rows and 301 held-out rows. SMOTE changes the training class counts from [567, 268, 367] to [567, 567, 567], producing 1,701 training rows. SMOTE is fitted on training data only, including within each cross-validation fold.

## Setup and execution

### Google Colab

1. Open the original `PPD_Final.ipynb` in Google Colab.
2. Upload the supplementary files to the runtime working directory if needed. Run `%pip install -r requirements.txt` in a notebook cell to install the listed dependencies. Restart the runtime if the installation requests it.
3. Place your CSV at `/content/gdrive/MyDrive/dataset/postnatal_data.csv`, or edit `DATASET_PATH` in notebook cell 3. Authorize the notebook's Drive mount.
4. Run cells 1–23 for data preparation, holdout training and evaluation. Run cell 24 for SHAP. Cells 25–26 perform the standalone five-fold evaluation; cell 27 runs the separate baseline comparison and writes its CSV files.
5. Download generated outputs before ending the runtime. They are written to the current working directory, not automatically to Drive.

The original installation cells do not list every later dependency; `requirements.txt` includes XGBoost and SciPy used in cell 27. Its entries are unpinned because the supplied notebook does not record package versions. It is a dependency inventory rather than a tested environment lockfile.

### Local Jupyter

Install with `python -m pip install -r requirements.txt`, then launch `jupyter lab PPD_Final.ipynb`. Remove the `from google.colab import drive` import and the `drive.mount(...)` call, and set `DATASET_PATH` to your local CSV. These are edits to make in your own notebook copy; this package does not modify the uploaded notebook.

Cell 27 explicitly passes `multi_class='auto'` to `LogisticRegression`. If the installed library rejects that argument, review compatibility with the source environment or update the call before rerunning; record any change and the actual installed versions. After a successful run, capture them with `python -m pip freeze > requirements-lock.txt`.

## Configuration use

`config.yaml` and `config.json` contain equivalent reference settings. **The supplied notebook uses hard-coded values and does not automatically load either file.** Editing these files alone does not change an experiment. Apply settings to the corresponding notebook cells or explicitly connect them during a later refactoring.

For example, JSON can be read without an additional dependency:

```python
import json
from pathlib import Path

config = json.loads(Path("config.json").read_text())
DATASET_PATH = config["data"]["csv_path"]
target_column = config["data"]["target_column"]
```

Place these assignments before CSV loading and replace the existing hard-coded assignments. Other configuration fields still require explicit wiring to their corresponding code.

The instantiated network uses embedding width 128, five residual blocks, dropout 0.25, and a sigmoid gate. The head has a 64-unit hidden layer. Training uses batch size 32, up to 50 epochs, cross-entropy loss and AdamW with learning rate 0.0008 and weight decay 0.0001. The learning-rate scheduler monitors validation accuracy with patience 3 and factor 0.5; early stopping uses patience 7. Gradient norms are clipped at 2.0. These are the instantiated settings, including five blocks even though one earlier class definition defaults to four.

## Generated outputs

| Output | Source cell | Content |
| --- | --- | --- |
| `best_feature_tokenizer_model.pth` | 16 | Holdout model state dictionary selected by the highest observed held-out accuracy |
| `shap_feature_importance_ppd.csv` | 24 | `Feature` and `Mean_Abs_SHAP`, sorted by decreasing importance |
| `baseline_comparison_per_fold.csv` | 27 | Fold, model and eight evaluation metrics |
| `baseline_comparison_summary.csv` | 27 | Model-wise metric strings formatted as mean ± sample standard deviation |
| `baseline_ttest_accuracy.csv` | 27 | Paired accuracy comparisons against each baseline |
| `baseline_ttest_f1.csv` | 27 | Paired weighted-F1 comparisons against each baseline |

Plots are displayed inline rather than saved as image files. Standalone cross-validation tables in cell 25 are also displayed without explicit CSV export. Cell 26 plots the last fold's learning curves. The checkpoint contains model weights; it does not include fitted preprocessing objects, feature order, optimizer state or a complete inference pipeline.

SHAP uses up to 50 background rows from SMOTE-resampled training data and up to 20 held-out samples, with sampling seed 42 and `nsamples=50`. The global importance table averages absolute SHAP values over samples and classes. The aggregated summary plot therefore uses unsigned magnitudes; class-wise plots retain the original class-specific SHAP values.

## Reproducibility and evaluation notes

- Mode imputation and label encoding are fitted on the full dataset before the holdout split and cross-validation. A revised evaluation should fit preprocessing on each training partition and transform its validation/test partition with those fitted objects.
- The holdout `test_loader` is used during training for scheduling, early stopping and checkpoint selection. Each cross-validation validation fold is likewise used for epoch selection and scoring. These scores reflect the implemented selection procedure and are not results from an untouched final test set or nested cross-validation.
- The holdout neural-network training is not globally seeded beforehand. Later cross-validation sections seed NumPy and PyTorch with 42 once before the fold loop; they do not reset seeds per fold or enable deterministic algorithms. Exact rerun equality is not established.
- Cross-validation code assigns `best_state = model.state_dict()` without a deep copy, so the stored state can change during subsequent updates. Fold metrics are computed from predictions captured at the best observed epoch, but the restored in-memory weights are not guaranteed to represent that epoch. A future correction should clone the state before continuing training.
- Later cells redefine classes and overwrite variables such as `model` and training tensors. Run the holdout analysis and SHAP before cross-validation; rerunning earlier evaluation cells after later sections requires restoring the intended model and data.
- ROC, precision–recall and calibration plotting loops assume the observed three-class setting. A change to two classes requires adapting the binarized-label shape handling.
- Paired t-tests use five fold scores for accuracy and weighted F1, with a 0.05 threshold and no multiple-comparison adjustment. Fold dependence and model-selection reuse limit interpretation of these tests.
- Raw categorical predictors are numerically encoded and passed to ordinary SMOTE, which can produce fractional category codes. This describes the supplied implementation; categorical-aware resampling would be a separate methodological change.

The saved holdout output reports accuracy 0.9668 and weighted F1 0.9669. The baseline-comparison output reports FeatureTokenizerNet five-fold accuracy 0.958078 ± 0.024079 (sample standard deviation). These are recorded notebook outputs, not new runs performed when preparing this package, and should be interpreted with the evaluation procedure above.

## Provenance and availability

The notebook source and saved outputs were inspected to prepare these files. Training was not rerun because the dataset and original environment were not provided. The source notebook's SHA-256 hash is recorded in `metadata.json`. Dataset provenance, author information, reuse license and exact software versions were not established from the supplied material and have not been invented. Add verified publication and licensing information before public release.
