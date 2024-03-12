import pandas as pd
from datetime import datetime

# For positive and negative reviews
df_sent = pd.read_csv("meta_reviews_sentiment.csv")
# For sentiment score
df_proc = pd.read_csv("Processed_data.csv")

# Sorting reviews to get positive reviews
def sortPositiveReviews(product_id):
    df_pos = df_sent[df_sent["asin"] == product_id].sort_values(
        "sentiment_score",
        ascending=False,
    )
    return df_pos


# Sorting reviews to get negative reviews
def sortNegativeReviews(product_id):
    df_neg = df_sent[df_sent["asin"] == product_id].sort_values(
        "sentiment_score",
        ascending=True,
    )
    return df_neg


# Get positive review of a product
def getPositiveReview(df_pos, i):
    reviewer_name = df_pos["reviewerName_x"][i - 1 : i].values[0]
    review_time = df_pos["reviewTime_x"][i - 1 : i].values[0]
    review_time = datetime.strptime(review_time, "%m %d, %Y")
    review_text = df_pos["reviewText_x"][i - 1 : i].values[0]
    overall_score = df_pos["overall_x"][i - 1 : i].values[0]
    summary = df_pos["summary_x"][i - 1 : i].values[0]
    sentiment_score = df_pos["sentiment_score"][i - 1 : i].values[0]
    positive_review = {
        "reviewer_name": reviewer_name,
        "review_time": review_time,
        "review_text": review_text,
        "overall_score": int(overall_score),
        "sentiment_score": sentiment_score,
        "summary": summary,
    }
    return positive_review


# Get positive review of a product
def getNegativeReview(df_neg, i):
    reviewer_name = df_neg["reviewerName_x"][i - 1 : i].values[0]
    review_time = df_neg["reviewTime_x"][i - 1 : i].values[0]
    review_time = datetime.strptime(review_time, "%m %d, %Y")
    review_text = df_neg["reviewText_x"][i - 1 : i].values[0]
    overall_score = df_neg["overall_x"][i - 1 : i].values[0]
    summary = df_neg["summary_x"][i - 1 : i].values[0]
    sentiment_score = df_neg["sentiment_score"][i - 1 : i].values[0]
    negative_review = {
        "reviewer_name": reviewer_name,
        "review_time": review_time,
        "review_text": review_text,
        "overall_score": int(overall_score),
        "sentiment_score": sentiment_score,
        "summary": summary,
    }
    return negative_review


# Get top overall sentiment score of a product
def getOverallSentiment(product_id):
    sentiment = df_proc[df_proc["asin"] == product_id]["sentiment_score"].values[0]
    return float(sentiment)


# Get top overall rating of a product
def getOverallRating(product_id):
    rating = df_proc[df_proc["asin"] == product_id]["overall"].values[0]
    return float(rating)
