from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

def train_models(X_train, X_train_scaled, y_train):
    models = {} 

    lr = LogisticRegression()
    lr.fit(X_train_scaled, y_train)
    models["Logistic Regression"] = lr

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train_scaled, y_train)
    models["KNN"] = knn

    dt = DecisionTreeClassifier(max_depth=5)
    dt.fit(X_train, y_train)
    models["Decision Tree"] = dt

    return models
