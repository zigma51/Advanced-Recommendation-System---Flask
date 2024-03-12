from flask import Flask
from app import app
from user.models import User


@app.route("/signup/user/", methods=["POST"])
def user_signup():
    user = User()
    return user.signup()


@app.route("/signout/user/")
def user_signout():
    user = User()
    return user.signout()


@app.route("/login/user/", methods=["POST"])
def user_login():
    user = User()
    return user.login()