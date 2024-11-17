# Blockchain Price Predictor

Welcome to the **Blockchain Price Predictor** project! This tool aims to forecast cryptocurrency prices using machine learning techniques, helping users gain insights into potential market trends.

## Project Overview

Our project leverages historical cryptocurrency data to make price predictions using regression models. We aim to build a tool that can provide predictive insights for various cryptocurrencies. At this stage, the project includes data collection, preprocessing, and an initial regression model for price forecasting.

## Current Project Status

The project is currently in its early development phase, with the following milestones completed:

1. **Data Collection**: Successfully gathered historical cryptocurrency data using modules like `yfinance` and `statsmodels`.
2. **Data Preprocessing**: Cleaned and formatted data to ensure accuracy and consistency, including handling missing values and outliers.
3. **Initial Model Implementation**: Developed a ordinary least square(OLS) regression model to predict cryptocurrency prices based on historical data.

## Tech Stack

- **Python**: Main programming language.
- **Libraries**:
  - **NumPy & Pandas**: For data manipulation and preprocessing.
  - **statsmodels**: For building the regression model.
  - **Matplotlib**: For data visualization and analysis.
  - **yfinance**: For collecting historical crypto price data.

## Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/zenblocklabs/bpp.git
   ```
2. **Navigate to the project directory**:
   ```bash
   cd bpp
   ```
3. **Install required packages**:
   ```bash
   pip install -r requirements.txt
   ```
4. **Run the data collection script**:
   ```bash
   main.py
   ```
5. **Run the indicators script**:
   ```bash
   indicators.py
   ```
6. **Run the model training script**:
   ```bash
   regression.py
   ```

## Project Structure

- **data/**: Contains scripts for data collection and preprocessing.
- **models/**: Holds the model scripts and initial regression model implementation.
- **notebooks/**: Jupyter notebooks for data analysis and visualization.
- **README.md**: Project overview and setup instructions (this file).

## Roadmap

### Completed Tasks
- [x] Data collection and initial data preprocessing.
- [x] Linear regression model implementation for basic price prediction.

### Upcoming Milestones
- [ ] Evaluate model accuracy and refine feature selection.
- [ ] Experiment with advanced machine learning models (e.g., Random Forest, LSTM).
- [ ] Integrate additional features like social media sentiment analysis.

## Contributing

Contributions to the project are welcome! Please feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License.

---
