import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

os.makedirs("images", exist_ok=True)

def plot_heatmap(df):
    plt.figure()
    sns.heatmap(df.corr(), annot=True)
    plt.title("Correlation Heatmap")
    plt.savefig("images/heatmap.png")
    plt.close()

def plot_model_comparison(results):
    plt.figure()
    plt.bar(results.keys(), results.values())
    plt.title("Model Comparison")
    plt.ylabel("Accuracy")
    plt.savefig("images/model_comparison.png")
    plt.close()

def knn_tuning(X_train_scaled, X_test_scaled, y_train, y_test):
    error_rates = []

    for k in range(1, 20):
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train_scaled, y_train)
        pred = knn.predict(X_test_scaled)
        error_rates.append(np.mean(pred != y_test))

    plt.figure()
    plt.plot(range(1, 20), error_rates)
    plt.title("KNN Error vs K")
    plt.savefig("images/knn_plot.png")
    plt.close()

def decision_tree_overfit(X_train, X_test, y_train, y_test):
    train_acc, test_acc = [], []
    depths = range(1, 10)

    for d in depths:
        model = DecisionTreeClassifier(max_depth=d)
        model.fit(X_train, y_train)

        train_acc.append(model.score(X_train, y_train))
        test_acc.append(model.score(X_test, y_test))

    plt.figure()
    plt.plot(depths, train_acc, label="Train")
    plt.plot(depths, test_acc, label="Test")
    plt.legend()
    plt.title("Overfitting Analysis")
    plt.savefig("images/overfitting.png")
    plt.close()

def feature_importance(X, X_train, y_train):
    dt = DecisionTreeClassifier(max_depth=5)
    dt.fit(X_train, y_train)

    plt.figure()
    plt.bar(X.columns, dt.feature_importances_)
    plt.xticks(rotation=45)
    plt.title("Feature Importance")
    plt.savefig("images/feature_importance.png")
    plt.close()

def pca_visualization(X, y):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    plt.figure()
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y)
    plt.title("PCA Visualization")
    plt.savefig("images/pca.png")
    plt.close()
