import yfinance as yf

sp500_data = yf.download("^GSPC", start="2022-01-01", end="2024-12-01")
sp500_data['price_change'] = sp500_data['Close'].pct_change()
sp500_data = sp500_data[['price_change']]
sp500_data.reset_index(inplace=True)
sp500_data.columns = ['date', 'price_change']

sp500_data.to_csv("data/sp500_data.csv")
