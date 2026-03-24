from sklearn.metrics import accuracy_score, classification_report

def evaluate_models(models, X_test, X_test_scaled, y_test):
    results = {}

    for name, model in models.items():
        if name == "Decision Tree":
            y_pred = model.predict(X_test)
        else:
            y_pred = model.predict(X_test_scaled)

        acc = accuracy_score(y_test, y_pred)
        results[name] = acc

        print(f"\n{name}")
        print("Accuracy:", acc)
        print(classification_report(y_test, y_pred))

    return results
