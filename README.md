# Predicting-house-Price-
run in python 
Code 
# Machine Learning Project: Predicting House Prices

# Importing necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Step 1: Load the dataset
def load_data():
    # For simplicity, we create a synthetic dataset.
    data = {
        'Size (sq ft)': [750, 800, 850, 900, 950, 1000, 1100, 1200, 1300, 1400],
        'Number of Rooms': [2, 2, 2, 3, 3, 3, 4, 4, 5, 5],
        'Price ($)': [150000, 160000, 170000, 180000, 190000, 200000, 220000, 240000, 260000, 280000]
    }
    return pd.DataFrame(data)

# Step 2: Preprocess the data
def preprocess_data(df):
    X = df[['Size (sq ft)', 'Number of Rooms']]
    y = df['Price ($)']
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Train the model
def train_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

# Step 4: Evaluate the model
def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    return mse, predictions

# Main function
def main():
    # Load and preprocess data
    data = load_data()
    X_train, X_test, y_train, y_test = preprocess_data(data)

    # Train the model
    model = train_model(X_train, y_train)

    # Evaluate the model
    mse, predictions = evaluate_model(model, X_test, y_test)

    # Print results
    print("Mean Squared Error:", mse)
    print("Predicted vs Actual Prices:")
    comparison = pd.DataFrame({
        'Actual': y_test.values,
        'Predicted': predictions
    })
    print(comparison)

# Run the project
if __name__ == "__main__":
    main()
