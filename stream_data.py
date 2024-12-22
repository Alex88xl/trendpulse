from datetime import datetime

import pandas as pd
import streamlit as st

from data_pipeline import (fetch_google_trends, fetch_google_trends_keywords,
                           fetch_reddit_posts, fetch_top_headlines,
                           fetch_twitter_data, parse_twitter_response,
                           sentiment_analysis)


def main():
    st.title("Interactive Data Pipeline")

    with st.expander("About"):
        st.write(
            """
            This is an interactive data pipeline that fetches data from various sources like Google Trends, Twitter, Reddit, and News APIs. 
            You can configure the data sources and parameters using the checkboxes and input fields below. 
            Click the 'Run Data Pipeline' button to fetch and display the data.

            You need an rapidapi.com API key to fetch data from the APIs.
            You need to subscribe to the following APIs:
            - twitter-x
            - reddit-scraper2
            - real-time-news-data
            """
        )

    st.write("You need to setup an api token here: https://www.rapidapi.com/")
    api_key = st.text_input("Enter your API Key:")

    # User Input for Google Trends
    col1, col2 = st.columns([4, 1])
    with col1:
        st.header("Step 1: Configure Google Trends")
    with col2:
        show_google_trends = st.checkbox("Show Google", value=True)
    
    if show_google_trends:   
        google_trend_terms = st.text_input("Enter keywords (comma-separated):", "Migration, Inflation")
        google_trend_regions = st.selectbox("Select region for Google Trends:", ["DE", "US", "GB", "FR", "IT"])
        google_timeframe = st.selectbox("Select timeframe for Google Trends:", ["now 7-d", "today 1-m", "today 3-m", "today 12-m"])

        st.write("You can also fetch top keywords for a region:")
        google_trend_regions_keywords = st.text_input("Enter region for top keywords:", "germany")

    # User Input for Twitter
    col1, col2 = st.columns([4, 1])
    with col1:
        st.header("Step 2: Configure Twitter Data")
    with col2:
        show_twitter = st.checkbox("Show Twitter", value=True)
    
    if show_twitter:   
        st.write("We use the following API: https://twitter-x.p.rapidapi.com/search/")
        twitter_query = st.text_input("Enter your search query:", "Elon Musk OR Tesla")
        twitter_limit = st.slider("Number of tweets to fetch:", min_value=1, max_value=50, value=20)

    # User Input for News Articles
    col1, col2 = st.columns([4, 1])
    with col1:
        st.header("Step 3: Fetch News Articles")
    with col2:
        show_news_articles = st.checkbox("Show News", value=True)
    
    if show_news_articles:  
        st.write("We use the following API: https://real-time-news-data.p.rapidapi.com/topic-headlines")
        news_limit = st.slider("Number of news articles to fetch:", min_value=1, max_value=10, value=5)
        news_country = st.selectbox("Select country for news:", ["DE", "US", "GB", "FR", "IT"])
        news_lang = st.selectbox("Select language for news:", ["de", "en", "fr", "it"])
        news_topic = st.selectbox("Select topic for news:", ["WORLD", "NATIONAL", "BUSINESS", "TECHNOLOGY", "ENTERTAINMENT", "SPORTS", "SCIENCE", "HEALTH"])

    # User Input for Reddit
    col1, col2 = st.columns([4, 1])
    with col1:
        st.header("Step 4: Reddit Analysis")
    with col2:
        show_reddit = st.checkbox("Show Reddit", value=True)

    if show_reddit:
        st.write("We use the following API: https://reddit-scraper2.p.rapidapi.com/search_posts")
        r_query = st.text_input("Enter Query (e.g., 'germany christmas'):", value="germany christmas")
        r_sort = st.selectbox("Sort by:", ["RELEVANCE", "HOT", "NEW", "TOP", "COMMENTS"], index=0)
        r_time = st.selectbox("Time range:", ["hour", "day", "week", "month", "year", "all"], index=1)
        r_nsfw = st.checkbox("Include NSFW posts?", value=False)
        r_limit = st.slider("Number of Posts to Fetch:", min_value=1, max_value=50, value=10)

    # Run Data Pipeline Button
    if st.button("Run Data Pipeline"):
        st.title("Data Pipeline Results")

        # Google Trends
        if show_google_trends:
            st.subheader("Google Trends Results")
            google_keywords = [kw.strip() for kw in google_trend_terms.split(",")]
            google_trends = fetch_google_trends(google_keywords, timeframe=google_timeframe, geo=google_trend_regions)
            st.line_chart(google_trends)

            # Fetch top keywords for a region
            if google_trend_regions_keywords:
                st.subheader("Top Keywords for a Region")
                top_keywords = fetch_google_trends_keywords(google_trend_regions_keywords)
                st.write(top_keywords)

        # Twitter Data
        if show_twitter:
            st.subheader("Twitter Data")
            response = fetch_twitter_data(api_key, twitter_query, twitter_limit)

            if response:
                # Parse tweets
                st.info("Parsing tweets...")
                tweets = parse_twitter_response(response)

                # Perform sentiment analysis
                st.info("Performing sentiment analysis...")
                tweets = sentiment_analysis(tweets)

                # Prepare data for display
                df = pd.DataFrame(tweets)
                if not df.empty:
                    df["created_at"] = pd.to_datetime(df["created_at"])  # Format date
                    st.subheader("Tweet Sentiment Analysis")
                    # sentiment is a compound score between -1 and 1 and should be mapped to a categorical value
                    df["sentiment_mapped"] = df["sentiment"].apply(lambda x: "positive" if x > 0 else "negative" if x < 0 else "neutral")
                    st.dataframe(df[["created_at", "text", "favorite_count", "retweet_count", "sentiment", "sentiment_mapped"]])

                    # Display sentiment distribution
                    st.subheader("Sentiment Distribution")
                    st.bar_chart(df["sentiment_mapped"].value_counts())
                else:
                    st.warning("No tweets found for the query.")

        # Reddit Posts
        if show_reddit:
            st.subheader("Reddit Posts")
            posts = fetch_reddit_posts(api_key, r_query, r_sort, r_time, int(r_nsfw))
            st.info(f"Fetching posts for query '{r_query}'...")
            
            if posts:
                posts = posts[:r_limit]
                st.success(f"Fetched {len(posts)} posts!")
                with st.expander("Reddit Posts", expanded=False):
                    for post in posts:
                        creation_date = datetime.strptime(post['creationDate'], "%Y-%m-%dT%H:%M:%S.%f%z")
                        st.markdown(f"### {post['title']}")
                        st.write(f"Author: {post['author']['name']}")
                        st.write(f"Subreddit: {post['subreddit']['url']}")
                        st.write(f"Comments: {post['comments']} | Score: {post['score']} | Upvote Ratio: {post['upvoteRatio']:.2f}")
                        st.write(f"Created: {creation_date.strftime('%Y-%m-%d %H:%M:%S')}")
                        st.markdown(f"[Read More]({post['url']})")
                        st.write("---")
            else:
                st.warning("No posts found. Try a different query or adjust the parameters.")

        # News Articles
        if show_news_articles:
            st.subheader("News Articles")
            news_articles = fetch_top_headlines(api_key, news_limit, country=news_country, lang=news_lang, topic=news_topic)
            
            if news_articles:
                st.success("Here are the top headlines:")
                with st.expander("Top Headlines", expanded=False):
                    for article in news_articles:
                        st.markdown(f"### {article['title']}")
                        st.write(article['snippet'])
                        st.markdown(f"[Read more]({article['link']})")
                        st.write("---")
            else:
                st.warning("No articles found. Please check your API key or try again later.")

# Run Streamlit App
if __name__ == "__main__":  
    main()
