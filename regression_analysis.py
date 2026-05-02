import pandas as pd
import matplotlib.pyplot as plt

sp500_data = pd.read_csv("data/sp500_data.csv")
sentiment_data = pd.read_csv("data/sentiment_data.csv")

sp500_data["date"] = pd.to_datetime(sp500_data["date"])
sentiment_data["date"] = pd.to_datetime(sentiment_data["date"])
press_release_dates = sentiment_data["date"]

sp500_data = sp500_data.resample('2M', on='date').mean().reset_index()
sentiment_data = sentiment_data.resample('2M', on='date').mean().reset_index()

fig, ax1 = plt.subplots(figsize=(12, 6))

ax1.plot(sp500_data["date"], sp500_data["price_change"], color='red', label='Price Change', marker='o')
ax1.set_ylabel('Price Change', color='red')
ax1.tick_params(axis='y', labelcolor='red')

ax2 = ax1.twinx()
ax2.plot(sentiment_data["date"], sentiment_data["sentiment score"], color='blue', label='Sentiment Score', marker='o')
ax2.set_ylabel('Sentiment Score', color='blue')
ax2.tick_params(axis='y', labelcolor='blue')

plt.title('Price Change vs Sentiment Over Time')
ax1.set_xlabel('Date')
fig.tight_layout()

ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

plt.draw()
plt.savefig("regression_graph.png")