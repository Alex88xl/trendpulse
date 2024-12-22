from urllib.parse import urljoin

import matplotlib.pyplot as plt
import nltk
import pandas as pd
import requests
from bs4 import BeautifulSoup
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from pytrends.request import TrendReq

nltk.download('vader_lexicon') 

# 1. Google Trends - Daten sammeln
def fetch_google_trends(keywords, timeframe="now 7-d", geo="DE"):
    pytrends = TrendReq()
    pytrends.build_payload(keywords, timeframe=timeframe, geo=geo)
    trends = pytrends.interest_over_time()
    if not trends.empty:
        trends = trends.drop(columns=['isPartial'])
    return trends

# fetch google trends based on region, return top 5 keywords without keywords
def fetch_google_trends_keywords(region="germany"):
    pytrends = TrendReq()
        # Get top trending searches for today in the specified region
    trends = pytrends.trending_searches(pn=region)

    return trends

# 2. Twitter - Daten sammeln
def fetch_twitter_data(apikey, query, limit=20):
    url = "https://twitter-x.p.rapidapi.com/search/"
    headers = {
        "x-rapidapi-host": "twitter-x.p.rapidapi.com",
        "x-rapidapi-key": apikey,
    }
    params = {"query": query, "section": "top", "limit": limit}
    try:
        response = requests.get(url, headers=headers, params=params)

        if response.status_code == 200:
            return response.json()  # Parse JSON response
        else:
            # st.error(f"Error fetching tweets: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print("Error fetching tweets:", e)
        return None
    
def parse_twitter_response(response):
    tweets = []
    try:
        entries = response.get("data", {}).get("search_by_raw_query", {}).get("search_timeline", {}).get("timeline", {}).get("instructions", [])[0].get("entries", [])

        for entry in entries:
            content = entry.get("content", {}).get("itemContent", {}).get("tweet_results", {}).get("result", {})
            if content:
                tweet_data = {
                    "id": content.get("rest_id"),
                    "text": content.get("legacy", {}).get("full_text"),
                    "created_at": content.get("legacy", {}).get("created_at"),
                    "user_id": content.get("legacy", {}).get("user_id_str"),
                    "favorite_count": content.get("legacy", {}).get("favorite_count", 0),
                    "retweet_count": content.get("legacy", {}).get("retweet_count", 0),
                    "lang": content.get("legacy", {}).get("lang")
                }
                tweets.append(tweet_data)
    except Exception as e:
        print("Error parsing Twitter response:", e)
    return tweets

# 3. Web-Scraping von Nachrichtenartikeln
def fetch_news_articles(url, max_days=7):
    """
    Fetch news articles from a given URL without requiring CSS selectors.

    Parameters:
    - url (str): The website URL to scrape articles from.
    - max_days (int): Maximum age of articles in days.

    Returns:
    - List[Dict]: List of dictionaries containing title, link, and publication date (if available).
    """
    try:
        # Fetch the webpage content
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # Raise HTTPError for bad responses
        soup = BeautifulSoup(response.content, 'html.parser')

        # Attempt to detect article containers
        article_data = []
        for article in soup.find_all(['article', 'div', 'li']):
            title = None
            link = None
            pub_date = None

            # Try to extract the title
            title_element = article.find(['h1', 'h2', 'h3', 'a'])
            if title_element:
                title = title_element.get_text(strip=True)

            # Try to extract the link
            link_element = article.find('a')
            if link_element:
                link = link_element.get('href')
                if link and not link.startswith('http'):
                    # Handle relative URLs
                    link = urljoin(url, link)

            # Try to extract the publication date
            date_element = article.find(['time', 'span'], {'class': lambda x: x and 'date' in x.lower()})
            if date_element:
                try:
                    pub_date = pd.to_datetime(date_element.get_text(strip=True))
                except Exception:
                    pass  # Skip unparsable dates

            # Filter articles based on max_days
            if pub_date and (pd.Timestamp.now() - pub_date).days > max_days:
                continue

            # Add to article data if title and link exist
            if title and link:
                article_data.append({
                    "title": title,
                    "link": link,
                    "pub_date": pub_date
                })

        return article_data

    except Exception as e:
        print(f"Error fetching articles from {url}: {e}")
        return []

# 4. Sentiment-Analyse
def sentiment_analysis(tweets):
    try:
        sid = SentimentIntensityAnalyzer()
        for tweet in tweets:
            text = tweet.get("text", "")
            sentiment = sid.polarity_scores(text)
            tweet["sentiment"] = sentiment["compound"]
        return tweets
    except Exception as e:
        print("Error performing sentiment analysis:", e)
        return tweets

def fetch_reddit_posts(apikey, query, sort="RELEVANCE", time="day", nsfw=1):
    url = "https://reddit-scraper2.p.rapidapi.com/search_posts"
    headers = {
        "x-rapidapi-host": "reddit-scraper2.p.rapidapi.com",
        "x-rapidapi-key": apikey,
    }
    params = {
        "query": query,
        "sort": sort,
        "time": time,
        "nsfw": nsfw,
    }

    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        return response.json().get("data", [])
    except Exception as e:
        print("Error fetching Reddit posts:", e)
        return []

def fetch_top_headlines(apikey, count=5, country="DE", lang="de", topic="NATIONAL"):
    url = "https://real-time-news-data.p.rapidapi.com/topic-headlines"
    headers = {
        "X-RapidAPI-Host": "real-time-news-data.p.rapidapi.com",
        "X-RapidAPI-Key": apikey,
    }
    params = {
        "limit": count,
        "country": country,
        "lang": lang,
        "topic": topic,
    }
    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        news_data = response.json()
        articles = news_data.get("data", [])
        return articles
    except Exception as e:
        print("Error fetching top headlines:", e)
        return []

# 5. Visualisierung
def plot_trends(trends, title="Google Trends"):
    trends.plot(figsize=(10, 6))
    plt.title(title)
    plt.xlabel("Datum")
    plt.ylabel("Interesse")
    plt.grid()
    plt.show()

