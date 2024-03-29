# Mean-Variance Portfolio Optimization

## Description
This Python script aims to optimize a portfolio of stocks using mean-variance optimization techniques. It fetches historical stock prices from Yahoo Finance, calculates daily returns, and computes mean returns and the covariance matrix. The optimization process aims to maximize the Sharpe ratio, which measures the risk-adjusted return of the portfolio. The script displays selected stocks and their weights in the optimized portfolio, along with portfolio return, volatility, and Sharpe ratio metrics.

## Features
- Fetches historical stock prices for a given list of symbols from Yahoo Finance.
- Calculates daily returns and statistics including mean returns and the covariance matrix.
- Optimizes the portfolio based on the Sharpe ratio using mean-variance optimization techniques.
- Displays selected stocks and their weights in the optimized portfolio.
- Provides portfolio metrics including return, volatility, and Sharpe ratio.

## Usage
1. Install necessary dependencies such as yfinance, pandas, numpy, and scipy.
2. Update the list of symbols with desired stocks for portfolio optimization.
3. Set the start and end dates for fetching historical data.
4. Define risk aversion and maximum individual weight constraint parameters.
5. Run the script to perform portfolio optimization using mean-variance techniques.
6. Review the selected stocks and their weights, as well as portfolio metrics.

## Notes
- This script serves as a basic example of mean-variance portfolio optimization and can be extended or modified as needed.
- Deep learning techniques can be incorporated for more advanced optimization strategies.

## Author
[Hishok]