import pandas as pd
import statsmodels.api as sm # type: ignore
import matplotlib.pyplot as plt

# Load and clean the CSV file
file_path = 'bitcoin_data.csv'  # Replace with your file path
data = pd.read_csv(file_path)

# Remove the first row if it contains non-numeric placeholders
data = data.drop(0)

# Convert columns to numeric, ignoring non-numeric values
numeric_columns = ['adj close', 'close', 'high', 'low', 'open', 'volume']
for col in numeric_columns:
    data[col] = pd.to_numeric(data[col], errors='coerce')

# Drop rows with missing values in numeric columns
data = data.dropna(subset=numeric_columns)

# Function to train OLS regression and create plots
def train_and_plot_ols(data, target_column):
    # Define features and target
    X = data.drop(columns=[target_column, 'date', 'timestamp'])
    y = data[target_column]
    
    # Add constant for the intercept
    X = sm.add_constant(X)
    
    # Fit the model
    model = sm.OLS(y, X).fit()
    predictions = model.predict(X)
    
    # Plot Actual vs Predicted values
    plt.figure(figsize=(10, 6))
    plt.scatter(y, predictions, alpha=0.5)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')  # 45-degree reference line
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(f'Actual vs Predicted for {target_column}')
    plt.show()
    
    # Plot Residuals
    residuals = y - predictions
    plt.figure(figsize=(10, 6))
    plt.scatter(predictions, residuals, alpha=0.5)
    plt.axhline(0, color='r', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title(f'Residuals for {target_column}')
    plt.show()

# Run regression and plot for each target column
for target in numeric_columns:
    train_and_plot_ols(data, target)
