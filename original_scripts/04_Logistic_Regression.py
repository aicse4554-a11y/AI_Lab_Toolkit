from sklearn.linear_model import LogisticRegression

# Training data
X = [[1], [2], [3], [4], [5], [6]]
Y = [0, 0, 0, 1, 1, 1]

# Create model
model = LogisticRegression()

# Train the model
model.fit(X, Y)

# Predict
prediction = model.predict([[5]])

print("Predicted Class:", prediction[0])
