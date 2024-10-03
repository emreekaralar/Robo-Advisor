import yfinance as yf
import pandas as pd
import numpy as np
import newsapi as ns
from newsapi import NewsApiClient
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from scipy.optimize import minimize
import datetime

newsapi = NewsApiClient(api_key='460c36f80cd1425b9f0ef852937b69cc')
analyzer = SentimentIntensityAnalyzer()

sp500_top_50 = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'FB', 'BRK.B', 'JNJ', 'JPM', 'V', 'PG', 'NVDA', 'TSLA', 'HD',
                'DIS', 'MA', 'UNH', 'VZ', 'ADBE', 'PYPL', 'NFLX', 'INTC', 'CMCSA', 'CSCO', 'PEP', 'PFE',
                'KO', 'MRK', 'NKE', 'ABT', 'CRM', 'T', 'XOM', 'WMT', 'BAC', 'MCD', 'ORCL', 'CVX', 'COST',
                'WFC', 'DHR', 'MDT', 'TMO', 'LLY', 'ACN', 'AVGO', 'QCOM', 'HON', 'TXN', 'NEE']

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

def screen_stocks_with_sentiment(tickers):
    screened_stocks = []

    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            data = stock.info

            pe_ratio = data.get('trailingPE', None)
            peg_ratio = data.get('pegRatio', None)
            dividend_yield = data.get('dividendYield', None)
            debt_to_equity = data.get('debtToEquity', None)
            roe = data.get('returnOnEquity', None)  # Corrected key for ROE
            fcf = data.get('freeCashflow', None)    # Corrected key for FCF
            market_cap = data.get('marketCap', None)

            sentiment_score = fetch_stock_news_sentiment(ticker)

            # Define your screening criteria here
            # Example criteria:
            # - PE Ratio less than 25
            # - Positive sentiment score
            # - ROE greater than 0.15
            if (pe_ratio is not None and pe_ratio < 25 and
                sentiment_score > 0 and
                roe is not None and roe > 0.15):

                stock_data = {
                    'Ticker': ticker,
                    'PE Ratio': pe_ratio,
                    'PEG Ratio': peg_ratio,
                    'Dividend Yield': dividend_yield,
                    'Debt to Equity': debt_to_equity,
                    'ROE': roe,
                    'Free Cash Flow': fcf,
                    'Market Cap': market_cap,
                    'Sentiment Score': sentiment_score
                }
                screened_stocks.append(stock_data)

        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")

    return pd.DataFrame(screened_stocks)

screened_df = screen_stocks_with_sentiment(sp500_top_50)

file_path = r'C:\Users\karal\PycharmProjects\Robo-Advisor\screened_stocks.xlsx'
screened_df.to_excel(file_path, index=False)
print("Results have been written to 'screened_stocks.xlsx'")