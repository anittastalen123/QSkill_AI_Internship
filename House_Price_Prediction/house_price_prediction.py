from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load dataset
housing = fetch_california_housing()

# Input features
X = housing.data

# Output (house price)
y = housing.target

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create Linear Regression model
model = LinearRegression()

# Train model
model.fit(X_train, y_train)

# Predict house prices
y_pred = model.predict(X_test)

# Check error
mse = mean_squared_error(y_test, y_pred)

print("Mean Squared Error:", mse)