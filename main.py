import sys
import os maaaammmm

# Fix import path maaaammm
sys.path.append(os.path.join(os.path.dirname(__file__), "src")) maaaam

# Imports
from src.data_preprocessing import load_and_preprocess
from src.train_models import train_models
from src.evaluation import evaluate_models 

from src.visualization import (
    plot_heatmap,
    plot_model_comparison, 
    knn_tuning,
    decision_tree_overfit,
    feature_importance,
    pca_visualization
)

from src.predict import predict_diabetes

# ------------------------------
# Fix dataset path automatically
# ------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(BASE_DIR, "data", "diabetes.csv")

# Check if file exists
if not os.path.exists(data_path):
    print("❌ ERROR: diabetes.csv not found!")
    print("👉 Put the file inside: data/diabetes.csv")
    exit()

# ------------------------------
# Load Data
# ------------------------------
df, X, y, X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, scaler = load_and_preprocess(data_path)

# ------------------------------
# Visualizations
# ------------------------------
plot_heatmap(df)

# ------------------------------
# Train Models
# ------------------------------
models = train_models(X_train, X_train_scaled, y_train)

# ------------------------------
# Evaluate
# ------------------------------
results = evaluate_models(models, X_test, X_test_scaled, y_test)

# ------------------------------
# Additional Analysis
# ------------------------------
plot_model_comparison(results)
knn_tuning(X_train_scaled, X_test_scaled, y_train, y_test)
decision_tree_overfit(X_train, X_test, y_train, y_test)
feature_importance(X, X_train, y_train)
pca_visualization(X, y)

# ------------------------------
# Prediction Example
# ------------------------------
sample = X.iloc[0].values
prediction = predict_diabetes(models["Logistic Regression"], scaler, sample)

print("\nSample Prediction:", prediction)
print("\n✅ DONE! Check 'images' folder and results.txt")
