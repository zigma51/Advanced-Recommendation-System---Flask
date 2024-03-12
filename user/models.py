from flask import Flask, jsonify, request, session, redirect
from werkzeug.utils import redirect
from passlib.hash import pbkdf2_sha256
import uuid
import web_app as webapp


class User:
    def start_session(self, user):
        del user["password"]
        session["logged_in"] = True
        session["user"] = user
        return jsonify(user), 200

    def signup(self):
        user = {
            "_id": uuid.uuid4().hex,
            "name": request.form.get("name"),
            "email": request.form.get("email"),
            "password": request.form.get("password"),
            "interests": request.form.get("interests"),
            "checkbox": request.form.get("checkbox"),
        }

        # Encrpyt password
        user["password"] = pbkdf2_sha256.encrypt(user["password"])

        # check of existing user
        if webapp.db.users.find_one({"email": user["email"]}):
            return jsonify({"error": "Email address already in use"}), 400

        if webapp.db.users.insert_one(user):
            return self.start_session(user)

        return jsonify({"error": "Sign Up failed"}), 400

    def signout(self):
        session.clear()
        return redirect("/")

    def login(self):
        user = webapp.db.users.find_one({"email": request.form.get("email")})

        if user and pbkdf2_sha256.verify(
            request.form.get("password"), user["password"]
        ):
            return self.start_session(user)

        return jsonify({"error": "Invalid Login credentials"}), 401