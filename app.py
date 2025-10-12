# app.py
import re
import pyotp, hashlib
from flask import Flask, render_template, request, redirect, session, flash, url_for, jsonify
from flask_sqlalchemy import SQLAlchemy
from sentence_transformers import SentenceTransformer, util
import json, torch
import faiss, os, docx, PyPDF2
from transformers import pipeline
from datetime import datetime

app = Flask(__name__)
app.secret_key = "supersecretkey"
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///database.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db = SQLAlchemy(app)

# ----------------------
# User model
# ----------------------
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100))
    email = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)   # hashed password
    secret_key = db.Column(db.String(32), nullable=False)

# ----------------------
# ContactQuery model
# ----------------------
class ContactQuery(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), nullable=False)
    subject = db.Column(db.String(150), nullable=False)
    message = db.Column(db.Text, nullable=False)
    date_submitted = db.Column(db.DateTime, default=datetime.utcnow)

# ----------------------
# Load FAQ JSON
# ----------------------
FAQ_PATH = "faq.json"
faq_list = []
if os.path.exists(FAQ_PATH):
    with open(FAQ_PATH, "r", encoding="utf-8") as f:
        raw_faq = json.load(f)
        for item in raw_faq:
            if "question" in item and "answer" in item:
                faq_list.append({"question": item["question"].strip(), "answer": item["answer"].strip()})
            elif "questions" in item and "answer" in item:
                for q in item["questions"]:
                    faq_list.append({"question": q.strip(), "answer": item["answer"].strip()})
print(f"Loaded {len(faq_list)} FAQ entries.")

# ----------------------
# Upload folder
# ----------------------
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# ----------------------
# SentenceTransformer model
# ----------------------
model = SentenceTransformer("all-MiniLM-L6-v2")
questions = [item["question"] for item in faq_list]
answers = [item["answer"] for item in faq_list]
question_embeddings = model.encode(questions, convert_to_tensor=True) if questions else None

def get_answer(user_input, top_k=1):
    if not questions or question_embeddings is None:
        return None
    user_embedding = model.encode(user_input, convert_to_tensor=True)
    scores = util.pytorch_cos_sim(user_embedding, question_embeddings)[0]
    topk = torch.topk(scores, k=min(top_k, len(scores))).indices.tolist()
    best_idx = topk[0] if isinstance(topk, list) else int(topk)
    return answers[best_idx]

# QA & Summarizer pipelines
qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# FAISS index
dimension = 384
index = faiss.IndexFlatL2(dimension)
documents = []

# ----------------------
# Helper functions
# ----------------------
def extract_text(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    text = ""
    try:
        if ext == ".txt":
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
        elif ext == ".pdf":
            with open(file_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    if page.extract_text():
                        text += page.extract_text() + "\n"
        elif ext == ".docx":
            doc = docx.Document(file_path)
            for para in doc.paragraphs:
                if para.text:
                    text += para.text + "\n"
    except Exception as e:
        print("Error extracting text:", e)
    return text.strip()

def add_document_to_index(text, source):
    global documents
    if not text:
        return
    CHUNK_SIZE = 300
    chunks = [text[i:i+CHUNK_SIZE] for i in range(0, len(text), CHUNK_SIZE) if text[i:i+CHUNK_SIZE].strip()]
    if not chunks:
        return
    import numpy as np
    embeddings = model.encode(chunks)
    if isinstance(embeddings, np.ndarray):
        index.add(embeddings.astype("float32"))
    else:
        index.add(np.array(embeddings).astype("float32"))
    for chunk in chunks:
        documents.append({"text": chunk, "source": source})

def summarize_text(text, chunk_size=800):
    if not text:
        return ""
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size) if text[i:i+chunk_size].strip()]
    summaries = []
    for chunk in chunks:
        try:
            summary = summarizer(chunk, max_length=150, min_length=50, do_sample=False)[0].get("summary_text", "")
            summaries.append(summary)
        except:
            summaries.append("⚠️ Could not summarize this chunk.")
    return " ".join(summaries)

# returns (answer, score) for FAQ/semantic match
def get_answer_with_score(user_input):
    """
    Returns (best_answer, similarity_score) using the precomputed question_embeddings.
    If FAQ embeddings are not available, returns (None, 0.0).
    """
    try:
        if not questions or question_embeddings is None:
            return (None, 0.0)
        user_emb = model.encode(user_input, convert_to_tensor=True)
        sims = util.pytorch_cos_sim(user_emb, question_embeddings)[0]
        max_val, max_idx = torch.max(sims, dim=0)
        score = float(max_val)
        return (answers[int(max_idx)], score)
    except Exception as e:
        print("get_answer_with_score error:", e)
        return (None, 0.0)


def is_gibberish(text):
    """
    Heuristic gibberish detector:
     - require at least 3 alphabetic characters
     - require letters to be at least 40% of the string
     - reject repeated single-character strings (e.g., "aaaaa")
    """
    if not text or len(text.strip()) < 2:
        return True
    s = text.strip()
    letters = re.findall(r"[A-Za-z]", s)
    if len(letters) < 3:
        return True
    letter_ratio = len(letters) / max(1, len(s))
    if letter_ratio < 0.4:
        return True
    if len(set(s.lower())) == 1:
        return True
    return False

# ----------------------
# Routes
# ----------------------
@app.route("/")
def home():
    return render_template("home.html")

@app.route("/about")
def about():
    return render_template("about.html")

# Contact page (GET + POST)
@app.route("/contact", methods=["GET", "POST"])
def contact():
    if request.method == "POST":
        name = request.form.get("name")
        email = request.form.get("email")
        subject = request.form.get("subject")
        message = request.form.get("message")
        if not all([name, email, subject, message]):
            flash("Please fill all fields", "error")
            return redirect("/contact")
        new_query = ContactQuery(name=name, email=email, subject=subject, message=message)
        db.session.add(new_query)
        db.session.commit()
        flash("Your query has been submitted successfully!", "success")
        return redirect("/contact")
    return render_template("contact.html")


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        name = request.form["name"]
        email = request.form["email"]
        password = hashlib.sha256(request.form["password"].encode()).hexdigest()
        secret_key = pyotp.random_base32()

        if User.query.filter_by(email=email).first():
            flash("Email already registered!", "danger")
            return redirect("/signup")

        user = User(name=name, email=email, password=password, secret_key=secret_key)
        db.session.add(user)
        db.session.commit()
        flash("Account created! Please login.", "success")
        return redirect("/login")
    return render_template("signup.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["email"]
        password = hashlib.sha256(request.form["password"].encode()).hexdigest()

        user = User.query.filter_by(email=email, password=password).first()
        if user:
            session["email"] = email

            # Generate OTP
            totp = pyotp.TOTP(user.secret_key)
            otp = totp.now()

            # For dev: print OTP in terminal
            print(f"🔐 OTP for {email} is: {otp}")

            flash("OTP has been generated! Check your terminal.", "info")
            return redirect("/verify-otp")
        else:
            flash("Invalid credentials!", "danger")
    return render_template("login.html")

@app.route("/verify-otp", methods=["GET", "POST"])
def verify_otp():
    if request.method == "POST":
        otp = request.form["otp"]
        email = session.get("email")

        user = User.query.filter_by(email=email).first()
        if not user:
            flash("Session expired. Please login again.", "danger")
            return redirect("/login")

        totp = pyotp.TOTP(user.secret_key)
        if totp.verify(otp, valid_window=1):
            flash("OTP Verified! 🎉", "success")
            return redirect("/dashboard")
        else:
            flash("Invalid OTP!", "danger")
    return render_template("verify_otp.html")

@app.route("/dashboard")
def dashboard():
    return render_template("view.html")   # your main UI

@app.route("/business")
def business():
    return render_template("business.html")

def is_gibberish(text):
    """Detect gibberish input (nonsense or random letters)."""
    import re

    # ✅ Treat common greetings as valid input
    greetings = ["hi", "hello", "hey", "good morning", "good afternoon", "good evening"]
    if text.lower().strip() in greetings:
        return False

    # Empty or very short text (not greeting)
    if len(text.strip()) < 2:
        return True

    # Check for too many consonants or random sequences
    if re.fullmatch(r"[a-zA-Z]{2,}", text):
        vowels = len(re.findall(r"[aeiouAEIOU]", text))
        if vowels / max(len(text), 1) < 0.2:  # too few vowels → likely gibberish
            return True

    # Check for weird symbols or numbers dominating
    if re.fullmatch(r"[^a-zA-Z0-9\s]+", text):
        return True

    return False


@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json.get("message", "").strip()
    if not user_message:
        return jsonify({"reply": "Please type a question."})

    # Helpline message to show when nothing confident found
    HELPLINE_MSG = (
        "Sorry, I couldn't find a confident answer.\n"
        "helpdesk@org.in | ph: 5555 4444 3333\n"
    )

    # Thresholds (tune these if you need stricter/looser behavior)
    FAQ_THRESHOLD = 0.60       # cosine similarity threshold for FAQ match
    QA_SCORE_THRESHOLD = 0.25  # min score returned by QA pipeline to accept answer

    # 0) quick gibberish filter
    if is_gibberish(user_message):
        return jsonify({"reply": HELPLINE_MSG})

    # 1) Try FAQ semantic match (fast) but accept only if similarity >= FAQ_THRESHOLD
    try:
        faq_ans, faq_score = get_answer_with_score(user_message)
        if faq_ans and faq_score >= FAQ_THRESHOLD:
            return jsonify({"reply": faq_ans})
        else:
            # debug logging (optional)
            print(f"FAQ low score: {faq_score:.3f} for query: {user_message}")
    except Exception as e:
        print("FAQ lookup error:", e)

    # 2) Try FAISS + QA on uploaded docs (if any)
    try:
        if index.ntotal > 0 and documents:
            query_vec = model.encode([user_message])
            import numpy as _np
            qv = _np.array(query_vec).astype("float32")
            k = min(3, index.ntotal)
            D, I = index.search(qv, k=k)  # D distances, I indices
            idxs = [int(i) for i in I[0] if i != -1]
            context = " ".join([documents[i]["text"] for i in idxs if 0 <= i < len(documents)])
            if context.strip():
                try:
                    answer = qa_pipeline(question=user_message, context=context)
                    ans_text = answer.get("answer", "").strip()
                    ans_score = float(answer.get("score", 0.0))
                    if ans_text and ans_score >= QA_SCORE_THRESHOLD:
                        # optional: shorten answer for readability
                        try:
                            short = summarizer(ans_text, max_length=60, min_length=10, do_sample=False)[0].get("summary_text","")
                            if short and len(short) < len(ans_text):
                                return jsonify({"reply": short})
                        except Exception:
                            pass
                        return jsonify({"reply": ans_text})
                    else:
                        print(f"Low QA confidence: score={ans_score:.3f} for query: {user_message}")
                except Exception as e:
                    print("QA pipeline error:", e)
    except Exception as e:
        print("FAISS/QA error:", e)

    # 3) final fallback: try a relaxed FAQ match (optional). If you don't want this, skip to helpline.
    try:
        # relaxed threshold (e.g., FAQ_THRESHOLD - 0.15)
        fallback_ans, fallback_score = get_answer_with_score(user_message)
        if fallback_ans and fallback_score >= (FAQ_THRESHOLD - 0.15):
            return jsonify({"reply": fallback_ans})
    except Exception:
        pass

    # 4) nothing confident — return helpline
    return jsonify({"reply": HELPLINE_MSG})

@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"reply": "❌ No file uploaded"})
    
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"reply": "❌ No file selected"})
    
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(file_path)

    text = extract_text(file_path)
    if not text:
        return jsonify({"reply": "❌ Could not extract text"})

    # Summarize the document (chunked)
    summary = summarize_text(text)

    # Add document chunks to FAISS
    add_document_to_index(text, file.filename)

    return jsonify({"reply": f"✅ {file.filename} added.\nSummary: {summary}"})




@app.route("/logout")
def logout():
    session.clear()
    flash("You have been logged out!", "info")
    return redirect(url_for("home"))

@app.route("/users")
def users():
    all_users = User.query.all()
    return "<br>".join([f"{u.id} - {u.name} - {u.email}" for u in all_users])

@app.route("/admin/queries")
def admin_queries():
    queries = ContactQuery.query.order_by(ContactQuery.date_submitted.desc()).all()
    return render_template("admin_queries.html", queries=queries)
    
if __name__ == "__main__":
    with app.app_context():
        db.create_all()  # Create all tables
    app.run(debug=True)
