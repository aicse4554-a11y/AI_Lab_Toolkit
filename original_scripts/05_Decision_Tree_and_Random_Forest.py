from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Training data
X = [[1], [2], [3], [4], [5], [6]]
Y = [0, 0, 0, 1, 1, 1]

# Decision Tree
dt = DecisionTreeClassifier()
dt.fit(X, Y)

# Random Forest
rf = RandomForestClassifier(n_estimators=10)
rf.fit(X, Y)

# Prediction
test = [[5]]

print("Decision Tree Prediction:", dt.predict(test)[0])
print("Random Forest Prediction:", rf.predict(test)[0])
