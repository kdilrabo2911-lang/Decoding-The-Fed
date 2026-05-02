import requests
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.nn.functional import softmax
import torch
import pandas as pd
import re
from datetime import datetime

tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")

def download_and_extract_text(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    text_content = ' '.join([p.get_text() for p in soup.find_all('p')])
    
    return text_content

def analyze_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
    outputs = model(**inputs)
    probs = softmax(outputs.logits, dim=-1)
    sentiment = ["positive", "negative", "neutral"]
    max_prob_index = torch.argmax(probs, dim=1).item()
    sentiment_label = sentiment[max_prob_index]
    
    return {
        "sentiment": sentiment_label,
        "positive_prob": probs[0][0].item(),
        "negative_prob": probs[0][1].item(),
        "neutral_prob": probs[0][2].item()
    }
    
web_files = [
    #2024 files
    "https://www.federalreserve.gov/newsevents/pressreleases/monetary20240131a.htm",
    "https://www.federalreserve.gov/newsevents/pressreleases/monetary20240320a.htm",
    "https://www.federalreserve.gov/newsevents/pressreleases/monetary20240501a.htm",
    "https://www.federalreserve.gov/newsevents/pressreleases/monetary20240612a.htm",
    "https://www.federalreserve.gov/newsevents/pressreleases/monetary20240731a.htm",
    "https://www.federalreserve.gov/newsevents/pressreleases/monetary20240918a.htm",
    "https://www.federalreserve.gov/newsevents/pressreleases/monetary20241107a.htm"
    
    #2023
    "https://www.federalreserve.gov/newsevents/pressreleases/monetary20230201a.htm",
    "https://www.federalreserve.gov/newsevents/pressreleases/monetary20230322a.htm",
    "https://www.federalreserve.gov/newsevents/pressreleases/monetary20230503a.htm",
    "https://www.federalreserve.gov/newsevents/pressreleases/monetary20230614a.htm",
    "https://www.federalreserve.gov/newsevents/pressreleases/monetary20230726a.htm",
    "https://www.federalreserve.gov/newsevents/pressreleases/monetary20230920a.htm",
    "https://www.federalreserve.gov/newsevents/pressreleases/monetary20231101a.htm",
    "https://www.federalreserve.gov/newsevents/pressreleases/monetary20231213a.htm"
    
    #2022
    "https://www.federalreserve.gov/newsevents/pressreleases/monetary20220126a.htm",
    "https://www.federalreserve.gov/newsevents/pressreleases/monetary20220316a.htm",
    "https://www.federalreserve.gov/newsevents/pressreleases/monetary20220504a.htm",
    "https://www.federalreserve.gov/newsevents/pressreleases/monetary20220615a.htm",
    "https://www.federalreserve.gov/newsevents/pressreleases/monetary20220727a.htm",
    "https://www.federalreserve.gov/newsevents/pressreleases/monetary20220921a.htm",
    "https://www.federalreserve.gov/newsevents/pressreleases/monetary20221102a.htm",
    "https://www.federalreserve.gov/newsevents/pressreleases/monetary20221214a.htm"
]


results = []
for url in web_files:
    text = download_and_extract_text(url)

    sentences = text.split(". ")
    
    doc_sentiments = []
    for sentence in sentences:
        sentiment_result = analyze_sentiment(sentence)
        sentiment_result["text"] = sentence
        doc_sentiments.append(sentiment_result)
    results.append({
        "url": url,
        "sentiments": doc_sentiments
    })


def extract_dates(web_files):
    dates = []
    for url in web_files:
        match = re.search(r'(\d{4})(\d{2})(\d{2})', url)

        if match:
            year, month, day = match.groups()
            try:
                date = datetime.strptime(f"{year}-{month}-{day}", "%Y-%m-%d")
                dates.append(date.strftime("%Y/%m/%d"))
            except ValueError:
                dates.append(None)

        else:
            dates.append(None)
    return dates

sentiments = []
for result in results:
    df = pd.DataFrame(result["sentiments"])
    sentiments.append(df["positive_prob"].mean())
    
df.to_csv("data/sentiment_table.csv")

sentiment_data = pd.DataFrame(columns = ["date", "sentiment score"])
sentiment_data["date"] = extract_dates(web_files)
sentiment_data["sentiment score"] = sentiments

sentiment_data.to_csv("data/sentiment_data.csv")