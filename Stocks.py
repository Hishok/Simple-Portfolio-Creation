import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

# Function to fetch historical stock prices
def get_historical_prices(symbols, start, end):
    prices = yf.download(symbols, start=start, end=end)['Adj Close']
    return prices

# Function to calculate daily returns
def calculate_daily_returns(prices):
    returns = prices.pct_change().dropna()
    return returns

# Function to calculate mean returns and covariance matrix
def calculate_statistics(returns):
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    return mean_returns, cov_matrix

# Function to get S&P 500 index
def get_sp500_index(start, end):
    sp500 = yf.download('^GSPC', start=start, end=end)['Adj Close']
    return sp500.pct_change().dropna()

# Function to get UK risk-free rate
def get_uk_risk_free_rate():
    # Provided risk-free rate
    risk_free_rate = 0.0524  # 5.24%
    return risk_free_rate

# Function for portfolio optimization with tactical asset allocation and downside risk protection
def optimize_portfolio(mean_returns, cov_matrix, risk_aversion, max_weight, stop_loss, risk_free_rate=0.0):
    num_assets = len(mean_returns)
    returns = np.random.multivariate_normal(mean_returns, cov_matrix, 10000)  # Monte Carlo simulation for VaR
    
    def objective_function(weights):
        portfolio_return = np.sum(mean_returns * weights)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
        m_squared = portfolio_return - 0.5 * portfolio_volatility**2
        treynor_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
        jensens_alpha = portfolio_return - (risk_free_rate + treynor_ratio * (portfolio_return - risk_free_rate))
        
        # VaR calculation
        var = portfolio_return - portfolio_volatility * np.percentile(np.dot(returns, weights), 5)
        
        # CVaR calculation
        cvar = (1 / (1 - 0.05)) * np.mean(np.clip(np.dot(returns, weights) - var, a_min=0, a_max=None))
        
        # Combine all metrics into a single objective function
        objective = -(sharpe_ratio + m_squared + treynor_ratio + jensens_alpha - var - cvar)
        return objective

    constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1},
                   {'type': 'ineq', 'fun': lambda weights: weights - 0.01},  # Minimum weight constraint
                   {'type': 'ineq', 'fun': lambda weights: 0.99 - weights},  # Maximum weight constraint
                   {'type': 'ineq', 'fun': lambda weights: max_weight - np.max(weights)})  # Maximum individual weight constraint

    bounds = tuple((0, None) for asset in range(num_assets))  # Relaxing bounds on asset weights
    initial_weights = [1. / num_assets] * num_assets

    result = minimize(objective_function, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)

    weights = result.x
    portfolio_return = np.sum(mean_returns * weights)
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    
    # Adjust risk-free rate to match portfolio currency and investment horizon
    adjusted_risk_free_rate = risk_free_rate  # Adjust as needed
    
    sharpe_ratio = (portfolio_return - adjusted_risk_free_rate) / portfolio_volatility
    m_squared = portfolio_return - 0.5 * portfolio_volatility**2
    treynor_ratio = (portfolio_return - adjusted_risk_free_rate) / portfolio_volatility
    jensens_alpha = portfolio_return - (adjusted_risk_free_rate + treynor_ratio * (portfolio_return - adjusted_risk_free_rate))
    var = portfolio_return - portfolio_volatility * np.percentile(np.dot(returns, weights), 5)
    cvar = (1 / (1 - 0.05)) * np.mean(np.clip(np.dot(returns, weights) - var, a_min=0, a_max=None))
    
    # Apply stop-loss to protect against downside risk
    if np.min(weights) < stop_loss:
        weights = np.maximum(weights, stop_loss * np.ones(num_assets))
        weights /= np.sum(weights)
    
    return weights, (portfolio_return, portfolio_volatility, result.fun,
                     sharpe_ratio, m_squared, treynor_ratio, jensens_alpha, var, cvar)

# Example parameters
start_date = '2000-01-01'
end_date = pd.Timestamp.now().strftime('%Y-%m-%d')  # Current date
risk_aversion = 1.0
max_weight = 0.2  # Maximum individual weight constraint
stop_loss = 0.05  # Stop-loss threshold

# List of symbols from different markets
symbols = ['AAPL', 'VUSA.L', 'VWRL.L', 'LLOY.L', 'SHOP', 'NVDA', 'SMT.L',
           'BP.L', 'TSLA', 'DIS', 'KO', 'GOOGL', 'MCD', 'AXP', 'AMD',
           'SBUX', 'T', 'AV.L', 'PEP', 'BARC.L', 'PYPL', 'NFLX',
           'AMZN', 'COIN', 'MSFT', 'ORCL', 'META', 'CRM', 'IBM']

# Fetch historical stock prices for all symbols
prices = get_historical_prices(symbols, start_date, end_date)

# Calculate daily returns and statistics
returns = calculate_daily_returns(prices)
mean_returns, _ = calculate_statistics(returns)

# Use Ledoit-Wolf shrinkage estimator to estimate the covariance matrix
lw = LedoitWolf()
lw.fit(returns)
cov_matrix = lw.covariance_

# Get UK risk-free rate
risk_free_rate = get_uk_risk_free_rate()

# Optimize portfolio with maximum weight constraint and stop-loss
weights, portfolio_metrics = optimize_portfolio(mean_returns, cov_matrix + np.eye(len(cov_matrix)) * 0.001, risk_aversion, max_weight, stop_loss, risk_free_rate)

# Display selected stocks and their weights
selected_stocks = [symbols[i] for i, weight in enumerate(weights) if weight > 0]
selected_weights = [weight for weight in weights if weight > 0]
print("Selected Stocks and Weights in Portfolio:")
for stock, weight in zip(selected_stocks, selected_weights):
    print(f"{stock}: {weight:.2%}")

# Print portfolio metrics
portfolio_return, portfolio_volatility, neg_obj_val, sharpe_ratio, m_squared, treynor_ratio, jensens_alpha, var, cvar = portfolio_metrics
print("\nPortfolio Metrics:")
print("Portfolio Return: This is the expected return of the portfolio.")
print(f"{portfolio_return:.2%}")
print("Portfolio Volatility: This measures the risk or dispersion of returns of the portfolio.")
print(f"{portfolio_volatility:.2%}")
print("Sharpe Ratio: This measures the risk-adjusted return of the portfolio.")
print(f"{sharpe_ratio:.4f}")
print("M-squared: This measures the portfolio's risk-adjusted return relative to a benchmark.")
print(f"{m_squared:.2%}")
print("Treynor Ratio: This measures the excess return per unit of systematic risk.")
print(f"{treynor_ratio:.4f}")
print("Jensen's Alpha: This measures the portfolio's excess return relative to its expected return.")
print(f"{jensens_alpha:.2%}")
print("Value at Risk (VaR): This represents the maximum potential loss (in percentage terms) with a confidence level of 95%.")
print(f"{var:.2%}")
print("Conditional Value at Risk (CVaR): This represents the expected loss beyond the VaR, given that the loss exceeds the VaR.")
print(f"{cvar:.2%}")

# Compare portfolio performance against S&P 500
sp500_returns = get_sp500_index(start_date, end_date)
portfolio_returns = returns.dot(weights)
portfolio_cumulative_returns = (1 + portfolio_returns).cumprod()
sp500_cumulative_returns = (1 + sp500_returns).cumprod()

# Plot portfolio and S&P 500 cumulative returns
plt.figure(figsize=(10, 6))
plt.plot(portfolio_cumulative_returns.index, portfolio_cumulative_returns, label='Portfolio')
plt.plot(sp500_cumulative_returns.index, sp500_cumulative_returns, label='S&P 500')
plt.title('Portfolio vs. S&P 500 Cumulative Returns')
plt.xlabel('Date')
plt.ylabel('Cumulative Return')
plt.legend()
plt.grid(True)
plt.show()
