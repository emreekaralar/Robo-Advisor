from statistics import LinearRegression

import yfinance as yf
import pandas as pd
import numpy as np
import newsapi as ns
from newsapi import NewsApiClient
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from scipy.optimize import minimize
import datetime

from yfinance import tickers

newsapi = NewsApiClient(api_key='460c36f80cd1425b9f0ef852937b69cc')
analyzer = SentimentIntensityAnalyzer()

sp500_tickers = [
    'MMM', 'AOS', 'ABT', 'ABBV', 'ACN', 'ADBE', 'AMD', 'AES', 'AFL', 'A', 'APD', 'ABNB', 'AKAM', 'ALB', 'ARE', 'ALGN',
    'ALLE', 'LNT', 'ALL', 'GOOGL', 'GOOG', 'MO', 'AMZN', 'AMCR', 'AMTM', 'AEE', 'AEP', 'AXP', 'AIG', 'AMT', 'AWK',
    'AMP', 'AME', 'AMGN', 'APH', 'ADI', 'ANSS', 'AON', 'APA', 'AAPL', 'AMAT', 'APTV', 'ACGL', 'ADM', 'ANET', 'AJG',
    'AIZ', 'T', 'ATO', 'ADSK', 'ADP', 'AZO', 'AVB', 'AVY', 'AXON', 'BKR', 'BALL', 'BAC', 'BAX', 'BDX', 'BRK.B', 'BBY',
    'TECH', 'BIIB', 'BLK', 'BX', 'BK', 'BA', 'BKNG', 'BWA', 'BSX', 'BMY', 'AVGO', 'BR', 'BRO', 'BF.B', 'BLDR', 'BG',
    'BXP', 'CHRW', 'CDNS', 'CZR', 'CPT', 'CPB', 'COF', 'CAH', 'KMX', 'CCL', 'CARR', 'CTLT', 'CAT', 'CBOE', 'CBRE',
    'CDW', 'CE', 'COR', 'CNC', 'CNP', 'CF', 'CRL', 'SCHW', 'CHTR', 'CVX', 'CMG', 'CB', 'CHD', 'CI', 'CINF', 'CTAS',
    'CSCO', 'C', 'CFG', 'CLX', 'CME', 'CMS', 'KO', 'CTSH', 'CL', 'CMCSA', 'CAG', 'COP', 'ED', 'STZ', 'CEG', 'COO',
    'CPRT', 'GLW', 'CPAY', 'CTVA', 'CSGP', 'COST', 'CTRA', 'CRWD', 'CCI', 'CSX', 'CMI', 'CVS', 'DHR', 'DRI', 'DVA',
    'DAY', 'DECK', 'DE', 'DELL', 'DAL', 'DVN', 'DXCM', 'FANG', 'DLR', 'DFS', 'DG', 'DLTR', 'D', 'DPZ', 'DOV', 'DOW',
    'DHI', 'DTE', 'DUK', 'DD', 'EMN', 'ETN', 'EBAY', 'ECL', 'EIX', 'EW', 'EA', 'ELV', 'EMR', 'ENPH', 'ETR', 'EOG',
    'EPAM', 'EQT', 'EFX', 'EQIX', 'EQR', 'ERIE', 'ESS', 'EL', 'EG', 'EVRG', 'ES', 'EXC', 'EXPE', 'EXPD', 'EXR', 'XOM',
    'FFIV', 'FDS', 'FICO', 'FAST', 'FRT', 'FDX', 'FIS', 'FITB', 'FSLR', 'FE', 'FI', 'FMC', 'F', 'FTNT', 'FTV', 'FOXA',
    'FOX', 'BEN', 'FCX', 'GRMN', 'IT', 'GE', 'GEHC', 'GEV', 'GEN', 'GNRC', 'GD', 'GIS', 'GM', 'GPC', 'GILD', 'GPN',
    'GL', 'GDDY', 'GS', 'HAL', 'HIG', 'HAS', 'HCA', 'DOC', 'HSIC', 'HSY', 'HES', 'HPE', 'HLT', 'HOLX', 'HD', 'HON',
    'HRL', 'HST', 'HWM', 'HPQ', 'HUBB', 'HUM', 'HBAN', 'HII', 'IBM', 'IEX', 'IDXX', 'ITW', 'INCY', 'IR', 'PODD',
    'INTC', 'ICE', 'IFF', 'IP', 'IPG', 'INTU', 'ISRG', 'IVZ', 'INVH', 'IQV', 'IRM', 'JBHT', 'JBL', 'JKHY', 'J', 'JNJ',
    'JCI', 'JPM', 'JNPR', 'K', 'KVUE', 'KDP', 'KEY', 'KEYS', 'KMB', 'KIM', 'KMI', 'KKR', 'KLAC', 'KHC', 'KR', 'LHX',
    'LH', 'LRCX', 'LW', 'LVS', 'LDOS', 'LEN', 'LLY', 'LIN', 'LYV', 'LKQ', 'LMT', 'L', 'LOW', 'LULU', 'LYB', 'MTB',
    'MRO', 'MPC', 'MKTX', 'MAR', 'MMC', 'MLM', 'MAS', 'MA', 'MTCH', 'MKC', 'MCD', 'MCK', 'MDT', 'MRK', 'META', 'MET',
    'MTD', 'MGM', 'MCHP', 'MU', 'MSFT', 'MAA', 'MRNA', 'MHK', 'MOH', 'TAP', 'MDLZ', 'MPWR', 'MNST', 'MCO', 'MS', 'MOS',
    'MSI', 'MSCI', 'NDAQ', 'NTAP', 'NFLX', 'NEM', 'NWSA', 'NWS', 'NEE', 'NKE', 'NI', 'NDSN', 'NSC', 'NTRS', 'NOC',
    'NCLH', 'NRG', 'NUE', 'NVDA', 'NVR', 'NXPI', 'ORLY', 'OXY', 'ODFL', 'OMC', 'ON', 'OKE', 'ORCL', 'OTIS', 'PCAR',
    'PKG', 'PLTR', 'PANW', 'PARA', 'PH', 'PAYX', 'PAYC', 'PYPL', 'PNR', 'PEP', 'PFE', 'PCG', 'PM', 'PSX', 'PNW', 'PNC',
    'POOL', 'PPG', 'PPL', 'PFG', 'PG', 'PGR', 'PLD', 'PRU', 'PEG', 'PTC', 'PSA', 'PHM', 'QRVO', 'PWR', 'QCOM', 'DGX',
    'RL', 'RJF', 'RTX', 'O', 'REG', 'REGN', 'RF', 'RSG', 'RMD', 'RVTY', 'ROK', 'ROL', 'ROP', 'ROST', 'RCL', 'SPGI',
    'CRM', 'SBAC', 'SLB', 'STX', 'SRE', 'NOW', 'SHW', 'SPG', 'SWKS', 'SJM', 'SW', 'SNA', 'SOLV', 'SO', 'LUV', 'SWK',
    'SBUX', 'STT', 'STLD', 'STE', 'SYK', 'SMCI', 'SYF', 'SNPS', 'SYY', 'TMUS', 'TROW', 'TTWO', 'TPR', 'TRGP', 'TGT',
    'TEL', 'TDY', 'TFX', 'TER', 'TSLA', 'TXN', 'TXT', 'TMO', 'TJX', 'TSCO', 'TT', 'TDG', 'TRV', 'TRMB', 'TFC', 'TYL',
    'TSN', 'USB', 'UBER', 'UDR', 'ULTA', 'UNP', 'UAL', 'UPS', 'URI', 'UNH', 'UHS', 'VLO', 'VTR', 'VLTO', 'VRSN',
    'VRSK', 'VZ', 'VRTX', 'VTRS', 'VICI', 'V', 'VST', 'VMC', 'WRB', 'GWW', 'WAB', 'WBA', 'WMT', 'DIS', 'WBD', 'WM',
    'WAT', 'WEC', 'WFC', 'WELL', 'WST', 'WDC', 'WY', 'WMB', 'WTW', 'WYNN', 'XEL', 'XYL', 'YUM', 'ZBRA', 'ZBH', 'ZTS'
]


def fetch_stock_news_sentiment(ticker):
    today = datetime.date.today()
    from_date = today - datetime.timedelta(days=7)
    articles = newsapi.get_everything(q=ticker, from_param=str(from_date), to=str(today), language='en', sort_by='relevancy')

    sentiment_scores = []

    for article in articles['articles']:
        content = article['description'] or article['title']
        if content:
            sentiment = analyzer.polarity_scores(content)['compound']
            sentiment_scores.append(sentiment)

    if sentiment_scores:
        return np.mean(sentiment_scores)
    else:
        return 0

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
    return excess_return / portfolio_volatility

def run_robo_advisor(tickers):
    screened_stocks = []

    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            data = stock.info

            pe_ratio = data.get('trailingPE', None)
            peg_ratio = data.get('pegRatio', None)
            dividend_yield = data.get('dividendYield', None)
            debt_to_equity = data.get('debtToEquity', None)
            roe = data.get('returnOnEquity', None)
            fcf = data.get('freeCashflow', None)
            market_cap = data.get('marketCap', None)

            sentiment_score = fetch_stock_news_sentiment(ticker)

            if pe_ratio and pe_ratio < 20 and peg_ratio and peg_ratio < 1.5:
                if dividend_yield and dividend_yield > 0.02:
                    if debt_to_equity and debt_to_equity < 1.0:
                        if fcf and fcf > 0 and sentiment_score > 0:
                            screened_stocks.append(ticker)
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")

    if not screened_stocks:
        print("No stocks passed the screening criteria.")
        return None, None

    price_data = fetch_historical_data(screened_stocks)

    returns = calculate_returns(price_data)

    optimal_weights = optimize_portfolio(returns)

    display_portfolio_allocation(screened_stocks, optimal_weights)

    cumulative_returns = (returns + 1).cumprod()
    var = calculate_var(returns)
    max_drawdown = calculate_max_drawdown(cumulative_returns)
    sharpe_ratio = calculate_sharpe_ratio(returns)

    print(f"Value at Risk (VaR): {var}")
    print(f"Max Drawdown: {max_drawdown}")
    print(f"Sharpe Ratio: {sharpe_ratio}")

    return screened_stocks, optimal_weights


screened_stocks, optimal_weights = run_robo_advisor(sp500_tickers)

if screened_stocks:
    screened_df = pd.DataFrame({'Ticker': screened_stocks})
    file_path = r'C:\Users\karal\PycharmProjects\Robo-Advisor\screened_stocks1.xlsx'
    screened_df.to_excel(file_path, index=False)
    print("Results have been written to 'screened_stocks1.xlsx'")
else:
    print("No data to write to Excel.")

