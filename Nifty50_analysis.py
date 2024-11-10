import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt

# Load the dataset using the absolute path, skipping the first 2 rows of metadata
data = pd.read_csv('/Users/kaustubh/Desktop/VSCode_Proj/Nifty50_Prediction/NIFTY50_data.csv', skiprows=2)

# Print the first few rows and the column names for debugging
print(data.head())
print(data.columns)

# Rename columns for easier access (adjusting to 7 columns)
data.columns = ['Date', 'Price', 'Adj Close', 'Close', 'High', 'Low', 'Open']  # Removed 'Volume'

# Convert 'Date' to datetime format
data['Date'] = pd.to_datetime(data['Date'])

# Extract month and year from the Date
data['Month'] = data['Date'].dt.month
data['Year'] = data['Date'].dt.year

# Add lag features (previous day's closing price as an example)
data['Prev Close'] = data['Close'].shift(1)

# Drop rows with missing values (due to the lag feature)
data = data.dropna()

# Select features (Month, Year, and Prev Close) and target variable (Close)
features = data[['Month', 'Year', 'Prev Close']]
target = data['Close']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Create a polynomial regression model (degree 3 to capture non-linear trends)
degree = 3  # You can experiment with the degree of the polynomial
poly_model = make_pipeline(PolynomialFeatures(degree), LinearRegression())

# Train the polynomial regression model
poly_model.fit(X_train, y_train)

# Evaluate the model on the test set
test_score = poly_model.score(X_test, y_test)
print(f"Model R² on test data: {test_score:.2f}")

# Function to predict Nifty50 index using the polynomial model
def predict_nifty_poly(month, year, prev_close):
    input_data = pd.DataFrame([[month, year, prev_close]], columns=['Month', 'Year', 'Prev Close'])
    prediction = poly_model.predict(input_data)
    return prediction[0]

# User input for month, year, and previous closing price
input_month = int(input("Enter the month (1-12): "))
input_year = int(input("Enter the year (e.g., 2024): "))
input_prev_close = float(input("Enter the previous day's closing price: "))

# Make the prediction
predicted_value = predict_nifty_poly(input_month, input_year, input_prev_close)
print(f"The predicted Nifty50 index for {input_month}/{input_year} is: {predicted_value:.2f}")

# Optional: Visualize the results
plt.scatter(data['Date'], data['Close'], color='blue', label='Actual Prices')
plt.scatter(pd.to_datetime(f"{input_year}-{input_month}-01"), predicted_value, color='red', label='Predicted Price')
plt.title('Nifty50 Index Prediction')
plt.xlabel('Date')
plt.ylabel('Nifty50 Closing Price')
plt.legend()
plt.show()
