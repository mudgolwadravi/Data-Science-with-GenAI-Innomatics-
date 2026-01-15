from flask import Flask, request, render_template, redirect
from flask_sqlalchemy import SQLAlchemy
import re
import random
import string
import requests

app = Flask(__name__)

# ------------------ Database Configuration ------------------
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///urls.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)

# ------------------ Database Model ------------------
class URL(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    original_url = db.Column(db.String(500), nullable=False)
    short_code = db.Column(db.String(10), unique=True, nullable=False)

# ------------------ Create Database ------------------
with app.app_context():
    db.create_all()

# ------------------ Helper Functions ------------------
def generate_short_code(length=6):
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

def is_valid_url(url):
    # Step 1: Format validation
    pattern = r'^https?://[^\s/$.?#].[^\s]*$'
    if not re.match(pattern, url):
        return False

    # Step 2: Reachability validation
    try:
        response = requests.head(url, timeout=5, allow_redirects=True)
        return response.status_code < 400
    except requests.RequestException:
        return False


@app.route("/", methods=["GET", "POST"])
def home():
    short_url = None
    error = None

    if request.method == "POST":
        original_url = request.form.get("url", "").strip()

        if not original_url:
            error = "No URL submitted"
        elif not is_valid_url(original_url):
            error = "Invalid or unreachable URL"
        else:
            short_code = generate_short_code()

            # Ensure short code uniqueness
            while URL.query.filter_by(short_code=short_code).first():
                short_code = generate_short_code()

            new_url = URL(
                original_url=original_url,
                short_code=short_code
            )
            db.session.add(new_url)
            db.session.commit()

            short_url = request.host_url + short_code

    return render_template("home.html", short_url=short_url, error=error)

@app.route("/<short_code>")
def redirect_url(short_code):
    url_entry = URL.query.filter_by(short_code=short_code).first()
    if url_entry:
        return redirect(url_entry.original_url)
    return "Invalid or expired URL"

@app.route("/history")
def history():
    urls = URL.query.all()
    return render_template("history.html", urls=urls)


if __name__ == "__main__":
    app.run(debug=True)
