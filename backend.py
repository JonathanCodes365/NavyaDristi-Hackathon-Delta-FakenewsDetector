from flask import Flask, request, jsonify, render_template
import pandas as pd
import re
import string
import joblib

app = Flask(__name__)

# Load model and vectorizer
model = joblib.load("lr.model.jb")
vectorizer = joblib.load("vectorizer.jb")

# Load smart news CSV
NEWS = pd.read_csv("smart_news.csv")  # Columns: title, description, url, thumbnail, source, label

# ---------------- Text cleaning ----------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', ' ', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text.strip()

# ---------------- Find matching articles from CSV ----------------
def get_related_articles(claim):
    claim_words = clean_text(claim).split()
    
    def contains_words(text):
        text_clean = clean_text(str(text))
        return any(word in text_clean for word in claim_words)
    
    matched = NEWS[
        NEWS['title'].apply(contains_words) | NEWS['description'].apply(contains_words)
    ]
    
    articles_list = []
    for _, row in matched.iterrows():
        articles_list.append({
            "title": row['title'],
            "description": row['description'],
            "url": row['url'],
            "thumbnail": row['thumbnail'],
            "source": row['source']
        })
    
    return articles_list

# ---------------- Get all news for homepage ----------------
@app.route("/api/news", methods=["GET"])
def get_news():
    limit = request.args.get('limit', 20, type=int)
    news_data = []
    
    for _, row in NEWS.head(limit).iterrows():
        news_data.append({
            "title": row['title'],
            "description": row['description'],
            "url": row['url'],
            "thumbnail": row['thumbnail'],
            "source": row['source']
        })
    
    return jsonify({"news": news_data})

# ---------------- Search news ----------------
@app.route("/api/search", methods=["GET"])
def search_news():
    query = request.args.get('q', '')
    
    if not query:
        return jsonify({"news": []})
    
    query_words = clean_text(query).split()
    
    def search_match(text):
        text_clean = clean_text(str(text))
        return any(word in text_clean for word in query_words)
    
    results = NEWS[
        NEWS['title'].apply(search_match) | 
        NEWS['description'].apply(search_match)
    ]
    
    news_data = []
    for _, row in results.iterrows():
        news_data.append({
            "title": row['title'],
            "description": row['description'],
            "url": row['url'],
            "thumbnail": row['thumbnail'],
            "source": row['source']
        })
    
    return jsonify({"news": news_data})

# ---------------- Verify endpoint ----------------
@app.route("/verify", methods=["POST"])
def verify():
    data = request.get_json()
    claim = data.get("claim", "")
    additional_info = data.get("additional_info", "")
    
    if not claim:
        return jsonify({"error": "No claim provided"})
    
    # ML model prediction
    text = f"{claim} {additional_info}"
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    pred = model.predict(vec)[0]
    proba = model.predict_proba(vec)[0]
    
    result = "Real" if pred == 1 else "Fake"
    fake_probability = int(proba[0] * 100)  # Probability of being fake
    
    # Fetch matching articles
    articles = get_related_articles(claim)
    
    # Separate articles based on label if available
    supporting = []
    refuting = []
    neutral = []
    
    for article in articles:
        # Check if article is in CSV with label
        matched_row = NEWS[
            (NEWS['title'] == article['title']) & 
            (NEWS['url'] == article['url'])
        ]
        
        if not matched_row.empty:
            label = matched_row.iloc[0].get('label', 'neutral')
            article_data = {
                "source": article['source'],
                "snippet": article['description'],
                "url": article['url'],
                "text": article['title']
            }
            
            if label == 'real' or label == 1:
                supporting.append(article_data)
            elif label == 'fake' or label == 0:
                refuting.append(article_data)
            else:
                neutral.append(article_data)
        else:
            neutral.append({
                "source": article['source'],
                "snippet": article['description'],
                "url": article['url'],
                "text": article['title']
            })
    
    # Determine verdict color
    if fake_probability >= 70:
        color = "red"  # Likely Fake
    elif fake_probability <= 30:
        color = "green"  # Likely Accurate
    elif 30 < fake_probability < 70:
        color = "orange"  # Uncertain
    else:
        color = "grey"  # Inconclusive
    
    return jsonify({
        "result": result,
        "percentage": fake_probability,
        "color": color,
        "supporting": supporting,
        "refuting": refuting,
        "neutral": neutral,
        "articles": articles
    })

# ---------------- Landing page with news feed ----------------
@app.route("/")
def home():
    return render_template("index.html")

# ---------------- Verification page ----------------
@app.route("/verify-page")
def verify_page():
    return render_template("verify.html")

if __name__ == "__main__":
    app.run(debug=True)
