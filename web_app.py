from model import *
from sentiment import *
from flask import Flask, render_template, request, session, redirect
from functools import wraps
import pymongo
import random as random

app = Flask(__name__)
app.secret_key = b"[|\xc6z\xae\xa8\xfa\xee\xa3\xd1\x0e\x7f0\xb4\xd9\x10"

import user.routes


# Database
client = pymongo.MongoClient("localhost", 27017)
db = client.shoppify

# Decorators
def login_required(f):
    @wraps(f)
    def wrap(*arg, **kwargs):
        if "logged_in" in session:
            return f(*arg, **kwargs)
        else:
            return redirect("/")

    return wrap


@app.route("/index/", methods=["POST", "GET"])
@login_required
def index():
    randomProducts1 = []
    for _ in range(4):
        prod_id = getRandomProductId()
        product = getProductDetails(prod_id)
        randomProducts1.append(product)
    randomProducts2 = []
    for _ in range(4):
        prod_id = getRandomProductId()
        product = getProductDetails(prod_id)
        randomProducts2.append(product)
    return render_template(
        "index.html",
        initialProducts1=randomProducts1,
        initialProducts2=randomProducts2,
    )


@app.route("/product/")
@login_required
def product():
    profilePhotos = [
        "/static/images/profile_photo.jpg",
        "/static/images/profile_photo1.jpg",
        "/static/images/profile_photo2.jpg",
        "/static/images/profile_photo3.jpg",
        "/static/images/profile_photo4.jpg",
        "/static/images/profile_photo5.jpg",
        "/static/images/profile_photo6.jpg",
        "/static/images/profile_photo7.jpg",
        "/static/images/profile_photo8.jpg",
        "/static/images/profile_photo9.jpg",
    ]
    random.shuffle(profilePhotos)
    selectedId = request.args.get("id")
    selectedProduct = getProductDetails(selectedId)
    recommendedProducts = []
    for i in range(1, 5):
        prod_id = getRecommendedProductId(selectedId, i)
        product = getProductDetails(prod_id)
        recommendedProducts.append(product)
    randomProducts2 = []
    for _ in range(4):
        prod_id = getRandomProductId()
        product = getProductDetails(prod_id)
        randomProducts2.append(product)
    positive_reviews = []
    df_positive_reviews = sortPositiveReviews(selectedId)
    df_negative_reviews = sortNegativeReviews(selectedId)
    for i in range(1, 6):
        positive_review = getPositiveReview(df_positive_reviews, i)
        positive_reviews.append(positive_review)
    negative_reviews = []
    for i in range(1, 6):
        negative_review = getNegativeReview(df_negative_reviews, i)
        negative_reviews.append(negative_review)
    reviews = positive_reviews + negative_reviews
    reviews.sort(key=lambda r: r["review_time"], reverse=True)
    for review in reviews:
        review["review_time"] = review["review_time"].strftime("%d-%m-%Y")
    return render_template(
        "product.html",
        selectedProduct=selectedProduct,
        recommendatedProducts=recommendedProducts,
        randomProducts=randomProducts2,
        reviews=reviews,
        overallSentimentScore=getOverallSentiment(selectedId),
        overallRatingScore=getOverallRating(selectedId),
        profilePhotos=profilePhotos,
    )


@app.route("/contact/")
def contact():
    return render_template("contact.html")


@app.route("/checkout/")
@login_required
def checkout():
    selectedId = request.args.get("id")
    selectedProduct = getProductDetails(selectedId)
    return render_template("checkout.html", selectedProduct=selectedProduct)


@app.route("/cart/")
@login_required
def cart():
    selectedId = request.args.get("id")
    selectedProduct = getProductDetails(selectedId)
    return render_template("cart.html", selectedProduct=selectedProduct)


@app.route("/categories/")
@login_required
def categories():
    return render_template("categories.html")


@app.route("/")
@app.route("/login/")
def login():
    return render_template("login-page.html")


@app.route("/signup/")
def signup():
    return render_template("signup-page.html")
