import pandas as pd
from sklearn.linear_model import LinearRegression

# Create dummy dataset
data = pd.DataFrame({
    'sqft': [800, 1000, 1200, 1500, 1800],
    'bedrooms': [2, 2, 3, 3, 4],
    'bathrooms': [1, 1, 2, 2, 3],
    'price': [200000, 250000, 320000, 400000, 520000]
})

X = data[['sqft', 'bedrooms', 'bathrooms']]
y = data['price']

model = LinearRegression()
model.fit(X, y)

# Predict new house
prediction = model.predict([[1400, 3, 2]])
print("Predicted Price:", prediction[0])
