from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

data = load_breast_cancer()
X, y = data.data, data.target

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0)
clf = DecisionTreeClassifier(random_state=0)
clf.fit(Xtr, ytr)

print("Accuracy:", clf.score(Xte, yte))

plt.figure(figsize=(14,8))
plot_tree(clf, feature_names=data.feature_names, class_names=data.target_names, filled=True)
plt.show()

new = Xte[0].reshape(1,-1)
pred = clf.predict(new)
print("New sample prediction:", data.target_names[pred][0])
