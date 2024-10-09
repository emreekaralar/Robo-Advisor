import yfinance as yf
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import datetime

sp500_tickers = [
    'A', 'AAL', 'AAP', 'AAPL', 'ABBV', 'ABC', 'ABMD', 'ABT', 'ACN', 'ADBE',
    'ADI', 'ADM', 'ADP', 'ADSK', 'AEE', 'AEP', 'AES', 'AFL', 'AIG', 'AIZ',
    'AJG', 'AKAM', 'ALB', 'ALGN', 'ALK', 'ALL', 'ALLE', 'ALXN', 'AMAT', 'AMD',
    'AME', 'AMGN', 'AMP', 'AMT', 'AMZN', 'ANET', 'ANSS', 'ANTM', 'AON', 'AOS',
    'APA', 'APD', 'APH', 'APTV', 'ARE', 'ATO', 'ATVI', 'AVB', 'AVGO', 'AVY',
    'AWK', 'AXP', 'AZO', 'BA', 'BAC', 'BAX', 'BBY', 'BDX', 'BEN', 'BF.B',
    'BIIB', 'BK', 'BKNG', 'BLK', 'BLL', 'BMY', 'BR', 'BRK.B', 'BSX', 'BWA',
    'BXP', 'C', 'CAG', 'CAH', 'CAT', 'CB', 'CBOE', 'CBRE', 'CCI', 'CCL',
    'CDNS', 'CE', 'CERN', 'CF', 'CFG', 'CHD', 'CHRW', 'CHTR', 'CI', 'CINF',
    'CL', 'CLX', 'CMA', 'CMCSA', 'CME', 'CMG', 'CMI', 'CMS', 'CNC', 'CNP',
    'COF', 'COG', 'COO', 'COP', 'COST', 'CPB', 'CPRT', 'CRM', 'CSCO', 'CSX',
    'CTAS', 'CTSH', 'CTVA', 'CTXS', 'CVS', 'CVX', 'D', 'DAL', 'DD', 'DE',
    'DFS', 'DG', 'DGX', 'DHI', 'DHR', 'DIS', 'DISCA', 'DISCK', 'DISH', 'DLR',
    'DLTR', 'DOV', 'DRE', 'DRI', 'DTE', 'DUK', 'DVA', 'DVN', 'DXC', 'EA',
    'EBAY', 'ECL', 'ED', 'EFX', 'EIX', 'EL', 'EMN', 'EMR', 'EOG', 'EQIX',
    'EQR', 'ES', 'ESS', 'ETN', 'ETR', 'EVRG', 'EW', 'EXC', 'EXPD', 'EXPE',
    'EXR', 'F', 'FAST', 'FB', 'FBHS', 'FCX', 'FDX', 'FE', 'FFIV', 'FIS',
    'FISV', 'FITB', 'FLIR', 'FLR', 'FLS', 'FMC', 'FOX', 'FOXA', 'FRC', 'FRT',
    'FTI', 'FTNT', 'FTV', 'GD', 'GE', 'GILD', 'GIS', 'GL', 'GLW', 'GM',
    'GOOG', 'GOOGL', 'GPC', 'GPN', 'GPS', 'GRMN', 'GS', 'GWW', 'HAL', 'HAS',
    'HBAN', 'HBI', 'HCA', 'HD', 'HES', 'HFC', 'HIG', 'HII', 'HLT', 'HOLX',
    'HON', 'HP', 'HPE', 'HPQ', 'HRL', 'HSIC', 'HST', 'HSY', 'HUM', 'IBM',
    'ICE', 'IDXX', 'IEX', 'IFF', 'ILMN', 'INCY', 'INTC', 'INTU', 'IP', 'IPG',
    'IPGP', 'IQV', 'IR', 'IRM', 'ISRG', 'IT', 'ITW', 'IVZ', 'JBHT', 'JCI',
    'JKHY', 'JNJ', 'JNPR', 'JPM', 'K', 'KEY', 'KEYS', 'KHC', 'KIM', 'KLAC',
    'KMB', 'KMI', 'KMX', 'KO', 'KR', 'KSS', 'KSU', 'L', 'LB', 'LDOS', 'LEG',
    'LEN', 'LH', 'LHX', 'LIN', 'LKQ', 'LLY', 'LMT', 'LNC', 'LNT', 'LOW',
    'LRCX', 'LUV', 'LVS', 'LW', 'LYB', 'M', 'MA', 'MAA', 'MAR', 'MAS', 'MCD',
    'MCHP', 'MCK', 'MCO', 'MDLZ', 'MDT', 'MET', 'MGM', 'MHK', 'MKC', 'MKTX',
    'MLM', 'MMC', 'MMM', 'MNST', 'MO', 'MOS', 'MPC', 'MRK', 'MRO', 'MS',
    'MSCI', 'MSFT', 'MSI', 'MTB', 'MTD', 'MU', 'MXIM', 'NCLH', 'NDAQ', 'NEE',
    'NEM', 'NFLX', 'NI', 'NKE', 'NLOK', 'NLSN', 'NOC', 'NOV', 'NRG', 'NSC',
    'NTAP', 'NTRS', 'NUE', 'NVDA', 'NWL', 'NWS', 'NWSA', 'O', 'ODFL', 'OKE',
    'OMC', 'ORCL', 'ORLY', 'OXY', 'PAYC', 'PAYX', 'PBCT', 'PCAR', 'PEAK',
    'PEG', 'PEP', 'PFE', 'PFG', 'PG', 'PGR', 'PH', 'PHM', 'PKG', 'PKI', 'PLD',
    'PM', 'PNC', 'PNR', 'PNW', 'PPG', 'PPL', 'PRGO', 'PRU', 'PSA', 'PSX',
    'PTC', 'PVH', 'PWR', 'PXD', 'PYPL', 'QCOM', 'QRVO', 'RCL', 'RE', 'REG',
    'REGN', 'RF', 'RHI', 'RJF', 'RL', 'RMD', 'ROK', 'ROL', 'ROP', 'ROST',
    'RSG', 'RTX', 'SBAC', 'SBUX', 'SCHW', 'SEE', 'SHW', 'SIVB', 'SJM', 'SLB',
    'SLG', 'SNA', 'SNPS', 'SO', 'SPG', 'SPGI', 'SRE', 'STE', 'STT', 'STX',
    'STZ', 'SWK', 'SWKS', 'SYF', 'SYK', 'SYY', 'T', 'TAP', 'TDG', 'TDY',
    'TEL', 'TER', 'TFC', 'TFX', 'TGT', 'TIF', 'TJX', 'TMO', 'TMUS', 'TPR',
    'TRMB', 'TROW', 'TRV', 'TSCO', 'TSN', 'TTWO', 'TWTR', 'TXN', 'TXT', 'TYL',
    'UA', 'UAA', 'UAL', 'UDR', 'UHS', 'ULTA', 'UNH', 'UNM', 'UNP', 'UPS',
    'URI', 'USB', 'V', 'VAR', 'VFC', 'VIAC', 'VLO', 'VMC', 'VNO', 'VRSK',
    'VRSN', 'VRTX', 'VTR', 'VTRS', 'VZ', 'WAB', 'WAT', 'WBA', 'WDC', 'WEC',
    'WELL', 'WFC', 'WHR', 'WLTW', 'WM', 'WMB', 'WMT', 'WRB', 'WRK', 'WU',
    'WY', 'WYNN', 'XEL', 'XLNX', 'XOM', 'XRAY', 'XYL', 'YUM', 'ZBH', 'ZBRA',
    'ZION', 'ZTS'
]


def fetch_historical_data(tickers, period="1y"):
    data = yf.download(tickers, period=period)['Adj Close']
    return data

def calculate_returns(prices):
    returns = prices.pct_change().dropna()
    return returns

def calculate_var(returns, confidence_level=0.95):
    z_score = np.percentile(returns, 100 * (1 - confidence_level))
    var = -z_score * returns.std()
    return var

def calculate_max_drawdown(cumulative_returns):
    drawdown = cumulative_returns / cumulative_returns.cummax() - 1
    max_drawdown = drawdown.min()
    return max_drawdown

def optimize_portfolio(returns):
    num_assets = len(returns.columns)
    mean_returns = returns.mean()
    cov_matrix = returns.cov()

    weights = np.ones(num_assets) / num_assets

    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}

    bounds = [(0, 1)] * num_assets

    def portfolio_risk(weights):
        return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

    result = minimize(portfolio_risk, weights, bounds=bounds, constraints=constraints)

    return result.x

def display_portfolio_allocation(tickers, weights):
    allocation = pd.DataFrame({
        'Ticker': tickers,
        'Weight': weights
    })
    print("\nOptimal Portfolio Allocation:")
    print(allocation)

def calculate_sharpe_ratio(returns, risk_free_rate=0.01):
    excess_return = returns.mean() - risk_free_rate
    portfolio_volatility = returns.std()
    sharpe_ratio = excess_return / portfolio_volatility
    return sharpe_ratio.mean()

def run_robo_advisor(tickers):
    # Fetch historical price data
    price_data = fetch_historical_data(tickers)

    price_data = price_data.dropna(axis=1, how='all')

    if price_data.empty:
        print("No price data available for the provided tickers.")
        return None, None

    returns = calculate_returns(price_data)

    optimal_weights = optimize_portfolio(returns)

    display_portfolio_allocation(returns.columns.tolist(), optimal_weights)

    cumulative_returns = (returns + 1).cumprod()
    var = calculate_var(returns)
    max_drawdown = calculate_max_drawdown(cumulative_returns)
    sharpe_ratio = calculate_sharpe_ratio(returns)

    print(f"\nRisk Metrics:")
    print(f"Value at Risk (VaR): {var}")
    print(f"Max Drawdown: {max_drawdown}")
    print(f"Sharpe Ratio: {sharpe_ratio}")

    return returns.columns.tolist(), optimal_weights

screened_stocks, optimal_weights = run_robo_advisor(sp500_tickers)

if screened_stocks:
    screened_df = pd.DataFrame({'Ticker': screened_stocks, 'Weight': optimal_weights})
    file_path = r'C:\Users\karal\PycharmProjects\Robo-Advisor\risk_metrics.xlsx'
    screened_df.to_excel(file_path, index=False)
    print("\nResults have been written to 'risk_metrics.xlsx'")
else:
    print("No data to write to Excel.")
