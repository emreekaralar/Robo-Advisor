import yfinance as yf
import pandas as pd
import numpy as np
from newsapi import NewsApiClient
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import datetime

# Initialize the News API client and sentiment analyzer
newsapi = NewsApiClient(api_key='460c36f80cd1425b9f0ef852937b69cc')  # Replace with your News API key
analyzer = SentimentIntensityAnalyzer()

# List of S&P 500 top 50 companies
sp500_top_50 = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'FB', 'BRK.B', 'JNJ', 'JPM', 'V', 'PG', 'NVDA', 'TSLA', 'HD',
                'DIS', 'MA', 'UNH', 'VZ', 'ADBE', 'PYPL', 'NFLX', 'INTC', 'CMCSA', 'CSCO', 'PEP', 'PFE',
                'KO', 'MRK', 'NKE', 'ABT', 'CRM', 'T', 'XOM', 'WMT', 'BAC', 'MCD', 'ORCL', 'CVX', 'COST',
                'WFC', 'DHR', 'MDT', 'TMO', 'LLY', 'ACN', 'AVGO', 'QCOM', 'HON', 'TXN', 'NEE']

def fetch_stock_news_sentiment(ticker):
    today = datetime.date.today()
    from_date = today - datetime.timedelta(days=7)
    try:
        articles = newsapi.get_everything(
            q=ticker,
            from_param=str(from_date),
            to=str(today),
            language='en',
            sort_by='relevancy',
            page_size=100  # Maximum number of articles per request
        )
    except Exception as e:
        print(f"Error fetching news for {ticker}: {e}")
        return 0

    sentiment_scores = []

    for article in articles['articles']:
        content = article.get('description') or article.get('title')
        if content:
            sentiment = analyzer.polarity_scores(content)['compound']
            sentiment_scores.append(sentiment)

    if sentiment_scores:
        return np.mean(sentiment_scores)
    else:
        return 0

def collect_stocks_data(tickers):
    stocks_data = []

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
            stocks_data.append(stock_data)

        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")

    return pd.DataFrame(stocks_data)


stocks_df = collect_stocks_data(sp500_top_50)
stocks_df.to_excel('all_stocks_scores.xlsx', index=False)

print("All stocks data has been written to 'all_stocks_scores.xlsx'")
