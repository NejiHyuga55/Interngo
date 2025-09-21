from flask import Flask, redirect, url_for, render_template, request, session, flash
from datetime import timedelta
from flask_sqlalchemy import SQLAlchemy
import joblib
import os
from dotenv import load_dotenv
import helper  # Import helper.py
import json

load_dotenv()  # Load environment variables from .env file

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "fallback_secret_key")
# Use Render's writable directory for SQLite
DB_PATH = os.environ.get('DATABASE_URL', 'sqlite:///instance/users.sqlite3')
app.config['SQLALCHEMY_DATABASE_URI'] = DB_PATH
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.permanent_session_lifetime = timedelta(hours=1)

db = SQLAlchemy(app)

# Load AI Recommendation Model
try:
    recommender_data = joblib.load('internship_recommender.pkl')
    recommender = recommender_data['recommender']
    print("✓ AI Recommendation model loaded successfully!")
except Exception as e:
    print(f"Error loading recommendation model: {e}")
    recommender = None

class users(db.Model):
    _id = db.Column("id", db.Integer, primary_key=True)
    name = db.Column(db.String(100))
    email = db.Column(db.String(100), unique=True)
    password = db.Column(db.String(100))
    skills = db.Column(db.String(500))

    def __init__(self, name, email, password, skills="[]"):
        self.name = name
        self.email = email
        self.password = password
        self.skills = skills

# Home page
@app.route("/")
def home():
    user = session.get("user")
    
    # Get recommendations for logged-in users
    recommendations = []
    if user and recommender:
        user_obj = users.query.filter_by(email=session.get("email")).first()
        if user_obj and user_obj.skills:
            try:
                user_skills = json.loads(user_obj.skills)
                if user_skills:
                    recommendations = recommender.recommend_by_skills(user_skills, top_n=3)
            except:
                pass
    
    return render_template("first.html", Username=user, recommendations=recommendations)

# Register
@app.route("/register/", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        fullname = request.form["fullname"]
        email = request.form["email"]
        password = request.form["password"]
        confirm_password = request.form["confirm_password"]
        role = request.form["role"]
        skills = request.form.getlist("skills")
        
        # Check if passwords match
        if password != confirm_password:
            flash("Passwords do not match!")
            return redirect(url_for("register"))
        
        # Check if email already exists
        existing_user = users.query.filter_by(email=email).first()
        if existing_user:
            flash("Email already registered! Please login instead.")
            return redirect(url_for("login"))
        
        # Create new user with skills
        new_user = users(name=fullname, email=email, password=password, skills=json.dumps(skills))
        db.session.add(new_user)
        db.session.commit()
        
        # Set session variables
        session["user"] = fullname
        session["email"] = email
        
        flash(f"Registration successful! Welcome {fullname}!")
        return redirect(url_for("home"))
    
    return render_template("register.html")

# Login
@app.route("/login/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        session.permanent = True
        email = request.form["nm"]
        password = request.form["password"]
        remember = request.form.get("remember") == "on"
        
        # Look up user in DB
        found_user = users.query.filter_by(email=email, password=password).first()
        
        if found_user:
            session["user"] = found_user.name
            session["email"] = found_user.email
            
            # Set session permanence based on remember me
            session.permanent = remember
            
            flash(f"Welcome back, {found_user.name}!", "success")
            return redirect(url_for("home"))
        else:
            flash("Invalid credentials! Please try again or register.", "error")
            return redirect(url_for("login"))
    
    return render_template("login.html")

# Input Page
@app.route("/input", methods=["GET"])
def input_page():
    if "user" not in session:
        flash("Please login first to get recommendations", "info")
        return redirect(url_for("login"))
    return render_template("input.html")

# Process Input
@app.route("/process-input", methods=["POST"])
def process_input():
    if "user" not in session:
        return redirect(url_for("login"))
    
    try:
        # Get form data
        education = request.form.get("education", "").strip()
        field_of_study = request.form.get("field_of_study", "").strip()
        skills = request.form.get("skills", "").strip()
        experience = request.form.get("experience", "").strip()
        sector = request.form.get("sector", "").strip()
        location = request.form.get("location", "").strip()
        
        # Validate required fields
        if not all([education, field_of_study, skills, sector, location]):
            flash("Please fill in all required fields", "error")
            return redirect(url_for("input_page"))
        
        # Process skills into a list
        skills_list = [skill.strip() for skill in skills.split(",") if skill.strip()]
        
        # Store in session for recommendations
        session["user_preferences"] = {
            "education": education,
            "field_of_study": field_of_study,
            "skills": skills_list,
            "experience": experience,
            "sector": sector,
            "location": location
        }
        
        # Update user skills in database
        user = users.query.filter_by(email=session["email"]).first()
        if user:
            user.skills = json.dumps(skills_list)
            db.session.commit()
        
        # Redirect to output page
        return redirect(url_for("output"))
        
    except Exception as e:
        flash("An error occurred while processing your request", "error")
        return redirect(url_for("input_page"))

@app.route("/process-input", methods=["GET"])
def process_input_redirect():
    flash("Please complete the input form first", "info")
    return redirect(url_for("input_page"))

# About
@app.route("/about")
def about():
    return render_template("about.html", user=session.get("user"))

# Output
@app.route("/output")
def output():
    if "user" not in session:
        return redirect(url_for("login"))
    
    if "user_preferences" not in session:
        flash("Please complete the input form first", "info")
        return redirect(url_for("input_page"))
    
    # Get recommendations using helper.py
    recommendations = []
    if hasattr(helper, "LightweightRecommender"):
        skills = session["user_preferences"].get("skills", [])
        recommender = helper.LightweightRecommender()
        recommendations = recommender.recommend_by_skills(skills, top_n=5)
    elif recommender:
        skills = session["user_preferences"].get("skills", [])
        recommendations = recommender.recommend_by_skills(skills, top_n=5)
    
    return render_template("output.html", 
                         recommendations=recommendations,
                         preferences=session["user_preferences"],
                         user=session.get("user"))

# API endpoint for recommendations
@app.route("/api/recommend", methods=["GET"])
def api_recommend():
    if "user" not in session:
        return {"error": "Login required"}, 401
    
    if "user_preferences" not in session:
        return {"error": "Complete input form first"}, 400
    
    skills = session["user_preferences"].get("skills", [])
    recommendations = []
    
    if recommender:
        recommendations = recommender.recommend_by_skills(skills, top_n=10)
    
    return {"recommendations": recommendations}

# Logout
@app.route("/logout/")
def logout():
    flash("You have been logged out!", "info")
    session.pop("user", None)
    session.pop("email", None)
    session.pop("user_preferences", None)
    return redirect(url_for("login"))

if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    # For local development, run Flask server
    if os.environ.get("RENDER") != "true":
        app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
    # For Render, Gunicorn will serve the app using Procfile
