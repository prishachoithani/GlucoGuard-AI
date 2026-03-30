# GlucoGuard AI
### Predicting who needs a blood test — before they know they need one

**BYOP Project · Fundamentals of AI & ML · B.Tech First Year**

**Author:** Prisha Choithani | **GitHub:** [@prishachoithani](https://github.com/prishachoithani)

## The Real Problem — Why Not Just Get a Blood Test?

**"If diabetes can be confirmed with a blood test, why build an ML model at all?"**

This is the right question to ask — and the answer is what makes this project meaningful.

A blood test **confirms** diabetes. But confirmation is only useful if you already suspect the disease. The real crisis is the silent majority — people who have no idea they are at risk and will never walk into a lab until it is too late.

Consider the ground reality in India:

- Over **57 million people** live with **undiagnosed** Type 2 diabetes
- Type 2 diabetes is symptom-free for years — you feel completely normal while the disease progresses
- A blood test requires a lab, a doctor's prescription, and awareness that you need one
- In rural and semi-urban areas, all three of these are hard to access simultaneously

**GlucoGuard AI answers a fundamentally different question:** Given a patient's basic biometric data — weight, blood pressure, age, glucose reading — should this person be urgently flagged for a clinical test?

This is a **triage tool**, not a diagnostic tool. The difference is everything.

The real power is in the **pre-diabetic window**. Type 2 diabetes develops over years through a reversible pre-diabetic stage. Caught in this window, the condition can be reversed entirely through lifestyle changes. Caught after symptoms appear, it cannot. A machine learning model can flag high-risk individuals years before a blood test would ever be ordered. That is what makes this project matter.

## Results and Outputs

### Terminal Output — Model Accuracy Reports

<img width="665" height="638" alt="terminal1" src="https://github.com/user-attachments/assets/13762303-7d3e-41c6-806a-e1ed790ae4c1" />

The classification report shows results for all three models. Decision Tree leads with **78.57% accuracy**, followed by Logistic Regression at **76.62%** and KNN at **75.32%**. Recall on the diabetic class (class 1) is the more important metric in a healthcare screening context.

<img width="880" height="132" alt="terminal2" src="https://github.com/user-attachments/assets/4a71f94f-ccff-42df-86d3-23ec916997a6" />

Final prediction output confirms the model correctly identifies diabetic risk from a sample input, with all outputs saved.

### Correlation Heatmap

<img width="640" height="480" alt="heatmap" src="https://github.com/user-attachments/assets/56c2e01a-be94-47d2-89b5-95ccfa7b1e2d" />

**Glucose (0.49)** and **BMI (0.31)** are the strongest predictors of diabetes. BloodPressure and SkinThickness are the weakest, confirmed later by feature importance.

### Model Comparison

<img width="640" height="480" alt="model_comparison" src="https://github.com/user-attachments/assets/1f26d76b-fc7b-479f-8e93-795403bb3388" />

Decision Tree achieves the highest accuracy at ~0.79, showing that non-linear decision boundaries suit this dataset better than the linear boundary of Logistic Regression.

### KNN Hyperparameter Tuning — Error vs K

<img width="640" height="480" alt="knn_plot" src="https://github.com/user-attachments/assets/95c4d1f0-33cc-48c5-a475-2842c8cc3d99" />

At low k (k=1), the model overfits — error is high on unseen data. As k increases, error decreases and stabilises. This plot was used to select the optimal k value.

### Decision Tree Overfitting Analysis

<img width="640" height="480" alt="overfitting" src="https://github.com/user-attachments/assets/6fb8106e-8311-4c97-a203-33498973cc23" />

Train accuracy (blue) climbs toward 95% as depth increases, while test accuracy (orange) plateaus around 77-79% and becomes unstable — the bias-variance tradeoff in practice.

### Feature Importance

<img width="640" height="480" alt="feature_importance" src="https://github.com/user-attachments/assets/700f4ed2-8e3b-4166-b29f-4af4cbed6d82" />

**Glucose dominates at ~0.51** — nearly 3x higher than the next feature. BMI (~0.20) and Age (~0.16) follow. This matches clinical knowledge — blood glucose is the primary biomarker of diabetes.

### PCA Visualisation

<img width="640" height="480" alt="pca" src="https://github.com/user-attachments/assets/ff611b01-30fd-4f2d-9e89-12606e4b0819" />

8 features compressed to 2 principal components. Purple = No Diabetes, Yellow = Diabetes. The partial separation confirms the features contain learnable signal even in 2D.

## Project Structure

```
GlucoGuard-AI/
│
├── main.py                        ← Entry point — run this
│
├── src/
│   ├── data_preprocessing.py      ← Load, clean, scale, split
│   ├── train_models.py            ← Train all 3 classifiers
│   ├── evaluation.py              ← Metrics, accuracy, reports
│   ├── visualization.py           ← All 6 plot functions
│   └── predict.py                 ← Risk prediction function
│
├── data/
│   └── diabetes.csv               ← PIMA Indians dataset
│
├── images/                        ← All output plots live here
│
├── results.txt                    ← Model metrics written here
├── requirements.txt
└── README.md
```

## Dataset

**PIMA Indians Diabetes Dataset** — collected by the National Institute of Diabetes and Digestive and Kidney Diseases (1988).

- **768** patient records · female · age 21 and above
- **8** non-invasive input features
- **1** binary target: `Outcome` — 0 = No Diabetes, 1 = Diabetes
- Class split: ~65% non-diabetic · ~35% diabetic

| Feature | Description | Clinical Relevance |
|---|---|---|
| `Pregnancies` | Number of pregnancies | Gestational diabetes is a known risk factor |
| `Glucose` | Plasma glucose (2-hr OGTT) | The single strongest predictor |
| `BloodPressure` | Diastolic BP (mm Hg) | Hypertension frequently co-occurs with diabetes |
| `SkinThickness` | Triceps skinfold (mm) | Proxy for body fat distribution |
| `Insulin` | 2-hr serum insulin (µU/ml) | Measures insulin resistance directly |
| `BMI` | Body mass index (kg/m²) | Second strongest predictor after glucose |
| `DiabetesPedigreeFunction` | Genetic/family history score | Quantifies hereditary risk |
| `Age` | Age in years | Risk rises sharply after age 45 |

## Setup and Running the Project

Requirements: Python 3.8+

```bash
# 1. Clone the repo
git clone https://github.com/prishachoithani/GlucoGuard-AI.git
cd GlucoGuard-AI

# 2. Create a virtual environment (recommended)
python -m venv venv

# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the project
python main.py
```

Expected output:

```
Data loaded and preprocessed
Models trained
Evaluation complete

Sample Prediction: Diabetic Risk

DONE! Check 'images' folder and results.txt
```

## Models and Why Each Was Chosen

| Model | Core Idea | Why Included |
|---|---|---|
| **Logistic Regression** | Estimates probability using the sigmoid function | Interpretable, outputs a risk probability not just a label |
| **Decision Tree** | Recursive if-else splits on feature values | Rules readable by a clinician; clearly demonstrates overfitting |
| **K-Nearest Neighbours** | Classifies by majority vote of k nearest points | Perfect for demonstrating bias-variance tradeoff via k-sweep |

## Course Concepts Applied

| Concept from Syllabus | Where It Appears in This Project |
|---|---|
| Supervised Learning | All 3 models trained on labelled patient records |
| Binary Classification | Diabetic vs Non-Diabetic prediction |
| Bayesian Statistics | Probabilistic reasoning in prediction output |
| Probability Theory | predict_proba() outputs, precision/recall metrics |
| Overfitting and Underfitting | Decision tree depth analysis — train vs test curve |
| Bias-Variance Tradeoff | KNN error vs k plot |
| Hyperparameter Tuning | k=1 to 19 sweep for KNN |
| Cross-Validation | Train/test split with stratification |
| Feature Scaling | StandardScaler — fit on train only, no data leakage |
| Dimensionality Reduction | PCA for 2D class-separation visualisation |
| Feature Importance | Bar chart showing relative predictor strength |
| Curse of Dimensionality | KNN sensitivity to irrelevant features |
| Statistical Decision Theory | Threshold-based classification at 0.5 |

## Limitations

- This is a screening tool, not a medical diagnosis. Every high-risk output must be followed up with a clinical blood test.
- Dataset covers a specific demographic — Pima Indian women aged 21+. Generalisation to broader populations requires retraining.
- Class imbalance (65:35) means recall on the diabetic class matters more than overall accuracy.
- No ML model should be deployed in healthcare without clinical validation and regulatory review.

## References

1. Smith, J.W. et al. (1988). Using the ADAP Learning Algorithm to Forecast the Onset of Diabetes Mellitus. Annual Symposium on Computer Application in Medical Care.
2. UCI Machine Learning Repository — PIMA Indians Diabetes Dataset. https://archive.ics.uci.edu/ml/datasets/diabetes
3. Scikit-learn Documentation. https://scikit-learn.org
4. Géron, A. (2022). Hands-On Machine Learning with Scikit-Learn, Keras and TensorFlow. O'Reilly Media.

---

Built for a B.Tech AIML course · Designed around a real problem · [prishachoithani/GlucoGuard-AI](https://github.com/prishachoithani/GlucoGuard-AI)
