import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import math

# loading ML model
with open("finalizedmode.sav", "rb") as f:
    knn_model = pickle.load(f)

# loading dataset
df = pd.read_csv("Processed_data.csv")
df_meta = pd.read_csv("meta_elec.csv")


# using label encoder
le = LabelEncoder()
df["asin_labelled"] = le.fit_transform(df["asin"])

X = df.drop(["Unnamed: 0", "asin", "asin_labelled"], axis=1)
y = df.asin_labelled


def getRecommendedProductId(selectedId, i):
    ind = df[df["asin"] == selectedId].index.values

    # generating recommendation product IDs
    query = X.iloc[ind]
    r = []
    distances, indices = knn_model.kneighbors(query, n_neighbors=50)
    a = y.iloc[indices[0][i * 4]]
    r.append(le.inverse_transform([a]))
    prod_id = r[0][0]
    return prod_id


def getRandomProductId():
    r = df.sample(1)
    prod_id = r["asin"].tolist()[0]
    return prod_id


def getProductDetails(prod_id):
    asin = prod_id
    details = df_meta.loc[df_meta["asin"] == asin]
    prod_asin = asin
    prod_title = details.iloc[0]["title"]
    prod_desc = details.iloc[0]["description"]
    prod_img_url = details.iloc[0]["imUrl"]
    prod_price = details.iloc[0]["price"]
    if math.isnan(prod_price):
        prod_price = 299
    product = {
        "id": prod_asin,
        "title": prod_title,
        "description": prod_desc,
        "image": prod_img_url,
        "price": prod_price,
    }
    return product
