# Financial Analysis using CVXPY

This project demonstrates financial analysis using the CVXPY library in Python. The analysis includes loading and preprocessing stock data, calculating daily returns, and performing portfolio optimization.

## Table of Contents
1. [Introduction](#introduction)
2. [Data Preprocessing](#data-preprocessing)
3. [Daily Returns Calculation](#daily-returns-calculation)
4. [Portfolio Optimization](#portfolio-optimization)
5. [Results](#results)

---

## Introduction

This project focuses on analyzing financial data using the CVXPY library. The goal is to preprocess stock data, calculate daily returns, and perform portfolio optimization to maximize returns while minimizing risk.

---

## Data Preprocessing

The dataset used in this project is `all_stocks_5yr.csv`, which contains stock data for various companies over five years. The preprocessing steps include:

1. **Loading the dataset**:
   ```python
   import pandas as pd
   data = pd.read_csv('/content/all_stocks_5yr.csv')
2. **Checking for missing values**:
   ```python
   data.isnull().sum()
   data.dropna(inplace=True)
3. **Filtering data for selected stocks (AAPL, GOOGL, MSFT)**:
   ```python
   selected_tickers = ['AAPL', 'GOOGL', 'MSFT']
   filtered_data = data[data['Name'].isin(selected_tickers)]
5. **Converting dates to datetime format and sorting the data**:
   ```python
   filtered_data['date'] = pd.to_datetime(filtered_data['date'])
   filtered_data.sort_values(by=['Name', 'date'], inplace=True)
7. **Daily Returns Calculation**:  
   - **Daily returns are calculated for each stock using the formula**:
     <p>
       Daily Return = (Close Price<sub>t</sub> - Close Price<sub>t-1</sub>) / Close Price<sub>t-1</sub>
    </p>

   - **The returns are then pivoted to create a returns matrix**:
   ```python
   filtered_data['daily_return'] = filtered_data.groupby('Name')['close'].pct_change()
   returns = filtered_data.pivot(index='date', columns='Name', values='daily_return').dropna()

8. **Portfolio optimization is performed using CVXPY to maximize returns while minimizing risk. The optimization problem is defined as**:

   $$
   \text{Minimize}: \frac{1}{2} w^T \Sigma w - \mu^T w
   $$

   **Subject to**:

   $$
   1^T w = 1, \quad w \geq 0
   $$

   Where:

   - \( w \) is the vector of portfolio weights.
   - \( \Sigma \) is the covariance matrix of returns.
   - \( \mu \) is the vector of mean returns.
9. **Code Implementation**:
   ```python
   import cvxpy as cp

   # Mean returns and covariance matrix
   mean_returns = returns.mean().values
   cov_matrix = returns.cov().values

   # Portfolio weights
   weights = cp.Variable(len(mean_returns))

   # Objective function (minimize risk)
   portfolio_risk = cp.quad_form(weights, cov_matrix)
   objective = cp.Minimize(portfolio_risk)

   # Constraints
   constraints = [cp.sum(weights) == 1, weights >= 0]

   # Solve the problem
   problem = cp.Problem(objective, constraints)
   problem.solve()

   # Optimal weights
   optimal_weights = weights.value

## Results

The results of the portfolio optimization are visualized using heatmaps and plots to show the optimal portfolio weights and the risk-return trade-off.

### Optimal Portfolio Weights:

| Stock  | Weight |
|--------|--------|
| AAPL   | 0.45   |
| GOOGL  | 0.35   |
| MSFT   | 0.20   |
