
from nltk.stem import PorterStemmer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import DataFrame
import nltk
import json
import pickle
import gzip

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression
from sklearn import neighbors
from scipy.spatial.distance import cosine
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn import preprocessing

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from textblob import TextBlob


import re
import string
from wordcloud import WordCloud, STOPWORDS
from sklearn.metrics import mean_squared_error
from sklearn.utils import shuffle

import pandas as pd
import gzip


def parse(path):
    g = gzip.open(path, 'rb')
    for l in g:
        yield json.loads(l)


def getDF(path):
    i = 0
    df = {}
    for d in parse(path):
        df[i] = d
        i += 1
        if(i >= 900000):
            break
    return pd.DataFrame.from_dict(df, orient='index')


df_elec_review = getDF(
    '/content/drive/My Drive/BE_proj/Data/reviews_Electronics_5.json.gz')

df_elec_review.head()

# electronics_reviews.dropna()

df_elec_review.shape

# Combining Summary and ReviewText to form a single column 'summary'

df_elec_review['summary'] = df_elec_review['summary'] + \
    ' ' + df_elec_review['reviewText']

# # Separate out upvotes and downvotes from 'helpful'
# votes = list(zip(*list(df_elec_review['helpful'].values)))
# df_elec_review['upvotes'] = np.array(votes[0])
# df_elec_review['downvotes'] = np.array(votes[1])

# Dropping columns which are not useful anymore
unnecessary_columns = ['reviewTime', 'unixReviewTime', 'helpful', 'reviewText']
df_elec_review.drop(columns=unnecessary_columns, inplace=True)

df_elec_review.head(1)


# Dimension of the new data frame
df_elec_review.shape


# frames = [df1, df2, df3, df4]
# df = pd.concat(frames)
df = df_elec_review.copy()
df.shape

# df.to_csv('all.csv')

"""### Product based CF"""

count = df.groupby("asin", as_index=False).count()
mean = df.groupby("asin", as_index=False).mean()

dfMerged = pd.merge(df, count, how='right', on=['asin'])
dfMerged.head()


df.shape

# rename column
dfMerged["totalReviewers"] = dfMerged["reviewerID_y"]
dfMerged["overallScore"] = dfMerged["overall_x"]
dfMerged["summaryReview"] = dfMerged["summary_x"]

dfNew = dfMerged[['asin', 'summaryReview', 'overallScore', "totalReviewers"]]

"""Selecting products which have more than 5 reviews


"""

dfMerged = dfMerged.sort_values(by='totalReviewers', ascending=False)
dfCount = dfMerged[dfMerged.totalReviewers >= 5]
dfCount = dfCount.reset_index()
dfCount.drop('index', axis=1, inplace=True)
dfCount

dfCount = dfCount.groupby('asin').filter(lambda x: len(x) >= 100)
dfCount.reset_index(inplace=True)
dfCount.drop('index', axis=1, inplace=True)
dfCount

g = dfCount.groupby('asin', group_keys=False)
g = g.apply(lambda x: x.sample(100).reset_index(drop=True))
dfCount = g.copy()
dfCount.reset_index(drop=True, inplace=True)

dfCount

counts = dfCount['asin'].value_counts().to_dict()
print(counts)

"""### Grouping all the summary Reviews by product ID"""

dfProductReview = df.groupby("asin", as_index=False).mean()
ProductReviewSummary = dfCount.groupby("asin")["summaryReview"].apply(list)
ProductReviewSummary = pd.DataFrame(ProductReviewSummary)
ProductReviewSummary.to_csv("ProductReviewSummary.csv")

dfProductReview.head(2)

ProductReviewSummary.head(2)

"""### create dataframe with certain columns"""

df3 = pd.read_csv("ProductReviewSummary.csv")
df3 = pd.merge(df3, dfProductReview, on="asin", how='inner')

df3 = df3[['asin', 'summaryReview', 'overall']]

# df3 = dfCount[['asin','summaryReview','overallScore']]

df3


"""### Text Cleaning - Summary column"""

# function for tokenizing summary
st = PorterStemmer()

regEx = re.compile('[^a-z]+')


def cleanReviews(reviewText):
    reviewText = reviewText.lower()
    reviewText = regEx.sub(' ', reviewText).strip()

    reviewText = (" ".join([st.stem(i) for i in reviewText.split()]))

    return reviewText


# clean summary
df3["summaryClean"] = df3["summaryReview"].apply(cleanReviews)

df3


df4 = df3[['asin']]
df4

df4.loc[0]


reviews = df3["summaryClean"]
countVector = CountVectorizer(
    max_features=5000, stop_words='english', ngram_range=(1, 2))
transformedReviews = countVector.fit_transform(reviews)


dfReviews = DataFrame(transformedReviews.A,
                      columns=countVector.get_feature_names())
dfReviews = dfReviews.astype(int)

# save
# dfReviews.to_csv("dfReviews.csv")

dfReviews


"""### PCA

#### Standard scaler
"""

# Standardizing the features
x_scaled_review = StandardScaler().fit_transform(dfReviews)

pca = PCA(n_components=50)
principalComponents = pca.fit_transform(x_scaled_review)
principalDf = pd.DataFrame(data=principalComponents)

principalDf.shape


finalDf = pd.concat([df4, df3[["overall"]], principalDf], axis=1)

finalDf.head(1)

finalDf = finalDf.sample(frac=1).reset_index(drop=True)

finalDf.head()


finalDf.to_csv("Processed_data.csv")

"""### Label encoding the target column"""

le = preprocessing.LabelEncoder()

finalDf['asin'] = le.fit_transform(finalDf['asin'])

le.classes_

finalDf.head(1)

le.inverse_transform(finalDf['asin'])


"""### KNN"""

# Create feature and target arrays
X = finalDf.drop('asin', axis=1)
y = finalDf.asin

# Split into training and test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=5)

knn = KNeighborsClassifier(n_neighbors=7)

knn.fit(X, y)

# Predict on dataset which model has not seen before
# print(knn.predict(X_test))

# print(knn.score(X_test, y_test))

distances, indices = knn.kneighbors(X.iloc[[1]],  n_neighbors=10)
print(distances, indices)
og = y.iloc[1]
# print(og)
print("og:", le.inverse_transform([og]))

for i in range(6):
    if(i == 0):
        continue
    a = y.iloc[indices[0][i]]
    print(le.inverse_transform([a]))


"""### Exporting the model pkl file"""

# Its important to use binary mode
knnPickle = open('knnpickle_file', 'wb')

# source, destination
pickle.dump(knn, knnPickle)


# load the model from disk
# loaded_model = pickle.load(open('knnpickle_file', 'rb'))
# result = loaded_model.predict(X_test)
