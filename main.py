from src.data_preprocessing import load_and_preprocess
from src.train_models import train_models
from src.evaluation import evaluate_models
from src.visualization import *
from src.predict import predict_diabetes

# Load data
df, X, y, X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, scaler = load_and_preprocess("data/diabetes.csv")

# Visualizations
plot_heatmap(df)

# Train models
models = train_models(X_train, X_train_scaled, y_train)

# Evaluate
results = evaluate_models(models, X_test, X_test_scaled, y_test)

# More analysis
plot_model_comparison(results)
knn_tuning(X_train_scaled, X_test_scaled, y_train, y_test)
decision_tree_overfit(X_train, X_test, y_train, y_test)
feature_importance(X, X_train, y_train)
pca_visualization(X, y)

# Prediction example
sample = X.iloc[0].values
print("\nPrediction:", predict_diabetes(models["Logistic Regression"], scaler, sample))
