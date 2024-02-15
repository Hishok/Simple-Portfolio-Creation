import yfinance as yf
import pandas as pd
import numpy as np
from scipy.optimize import minimize
bond_data = yf.Ticker('^FTSE')
bond_yield = bond_data.history(period="1d")['Close'].iloc[-1] / 100  # Convert percentage to decimal
print(bond_yield)