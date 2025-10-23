# pip install scikit-learn
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

# 1) Data
X, y = load_iris(return_X_y=True)  # 150 samples, 4 features, 3 classes
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# 2) Model (1 hidden layer with 16 neurons)
clf = MLPClassifier(hidden_layer_sizes=(16,), max_iter=800, random_state=42)

# 3) Train
clf.fit(X_train, y_train)

# 4) Evaluate
pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, pred))
print(classification_report(y_test, pred))

# 5) Run a few tests
samples = X_test[:5]
print("Sample predictions:", clf.predict(samples))
print("True labels:       ", y_test[:5])
