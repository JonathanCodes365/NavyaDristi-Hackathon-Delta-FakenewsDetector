# app.py - FIXED VERSION WITH NEPAL SOURCES & LIVE NEWS WIDGET

from flask import Flask, request, jsonify, render_template
import pandas as pd
import os
import re
import requests
import string
import joblib
from difflib import SequenceMatcher

# ---------------- Verification Module ----------------
try:
    from verify_article1 import verify_article
    VERIFY_MODULE_AVAILABLE = True
    print("✓ Using verify_article_improved.py with Nepal sources support")
except ImportError:
    VERIFY_MODULE_AVAILABLE = False
    print("⚠ Verification module not found")

app = Flask(__name__)

# ---------------- Configuration ----------------
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY", "2a545d0a809240aca808f97e53cd68fd")

# ---------------- Load ML Model ----------------
try:
    model = joblib.load("lr.model.jb")
    vectorizer = joblib.load("vectorizer.jb")
    ML_MODEL_AVAILABLE = True
    print("✓ ML model loaded")
except Exception as e:
    ML_MODEL_AVAILABLE = False
    print("⚠ ML model not loaded:", e)

# ---------------- Load News CSV ----------------
try:
    NEWS = pd.read_csv("smart_news.csv")
    for col in ['title', 'description', 'url', 'thumbnail', 'source']:
        if col in NEWS.columns:
            NEWS[col] = NEWS[col].astype(str).str.strip()
    print(f"✓ Loaded {len(NEWS)} articles from CSV")
except Exception as e:
    NEWS = pd.DataFrame()
    print("⚠ Could not load smart_news.csv:", e)

# ---------------- Text Cleaning ----------------
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

# ---------------- CSV Matching ----------------
def get_article_by_claim(claim_text, threshold=0.5):
    if NEWS.empty:
        return []
    
    cleaned_claim = clean_text(claim_text)
    matched_articles = []
    
    for _, row in NEWS.iterrows():
        cleaned_title = clean_text(row['title'])
        similarity = SequenceMatcher(None, cleaned_claim, cleaned_title).ratio()
        if similarity >= threshold:
            matched_articles.append({
                "title": row['title'],
                "description": row['description'],
                "url": row['url'],
                "thumbnail": row['thumbnail'],
                "source": row['source'],
                "similarity": similarity
            })
    
    matched_articles.sort(key=lambda x: x['similarity'], reverse=True)
    return matched_articles

# ---------------- Verification Function ----------------
def verify_claim_comprehensive(claim_text):
    results = {
        "verdict": "Uncertain",
        "confidence": 0.0,
        "supporting": [],
        "refuting": [],
        "neutral": [],
        "percentage": 50,
        "method": "unknown"
    }

    # PRIMARY: Advanced verification with Nepal sources
    if VERIFY_MODULE_AVAILABLE:
        try:
            nepal_keywords = ["nepal", "kathmandu", "pokhara", "nepali", "nepalese", 
                              "himalayan", "everest", "नेपाल"]
            is_nepal_related = any(kw in claim_text.lower() for kw in nepal_keywords)
            
            result = verify_article(
                title=claim_text,
                text="",
                max_pages_per_query=8,
                debug=True
            )
            
            verdict = result.get("verdict", "inconclusive")
            confidence = result.get("confidence", 0.0)
            evidence = result.get("evidence", [])
            
            nepali_sources = sum(1 for e in evidence if e.get("is_nepal", False))
            
            if verdict == "refuted":
                results["verdict"] = "Likely Fake"
                results["percentage"] = int(confidence * 100)
            elif verdict == "supported":
                results["verdict"] = "Likely Accurate"
                results["percentage"] = 100 - int(confidence * 100)
            else:
                results["verdict"] = "Uncertain" if confidence <= 0.5 else "Uncertain (Leaning Fake)"
                results["percentage"] = 50 if confidence <= 0.5 else 60
            
            results["confidence"] = confidence
            results["method"] = "verify_with_nepal_sources"
            
            for e in evidence:
                evidence_item = {
                    "source": e.get("source", "Unknown"),
                    "snippet": e.get("snippet", ""),
                    "url": e.get("url", ""),
                    "text": e.get("title", ""),
                    "is_nepali": e.get("is_nepal", False)
                }
                stance = e.get("stance", "inconclusive")
                if stance == "refuted":
                    results["refuting"].append(evidence_item)
                elif stance == "supported":
                    results["supporting"].append(evidence_item)
                else:
                    results["neutral"].append(evidence_item)
            
            if is_nepal_related and nepali_sources >= 2:
                if verdict == "refuted" and results["percentage"] < 70:
                    results["percentage"] = min(results["percentage"] + 10, 85)
                elif verdict == "supported" and results["percentage"] > 30:
                    results["percentage"] = max(results["percentage"] - 10, 15)
            
            return results
        
        except Exception as e:
            print("❌ Error in verification:", e)
            results["method"] = "verify_error"

    # FALLBACK: ML model
    if results["method"] in ["unknown", "verify_error"] and ML_MODEL_AVAILABLE:
        try:
            cleaned = clean_text(claim_text)
            vec = vectorizer.transform([cleaned])
            pred = model.predict(vec)[0]
            proba = model.predict_proba(vec)[0]
            ml_fake_prob = int(proba[0] * 100)
            
            results["percentage"] = ml_fake_prob
            results["confidence"] = max(proba)
            
            if ml_fake_prob >= 65:
                results["verdict"] = "Likely Fake"
            elif ml_fake_prob <= 35:
                results["verdict"] = "Likely Accurate"
            else:
                results["verdict"] = "Uncertain"
            
            results["method"] = "ml_model"
            
        except Exception as e:
            print("ML model error:", e)
            results["method"] = "ml_error"
    
    # Add CSV matches
    csv_results = get_article_by_claim(claim_text, threshold=0.4)
    for csv_art in csv_results[:3]:
        results["supporting"].append({
            "source": f"{csv_art['source']} (Database {csv_art['similarity']:.0%})",
            "snippet": csv_art["description"],
            "url": csv_art["url"],
            "text": csv_art["title"],
            "is_nepali": False
        })
    
    return results

# ---------------- API Routes ----------------
@app.route("/api/news", methods=["GET"])
def get_news():
    if NEWS.empty:
        return jsonify({"news": []})
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

@app.route("/api/search", methods=["GET"])
def search_news():
    query = request.args.get('q', '').strip()
    if not query:
        return jsonify({"news": []})
    
    try:
        url = f"https://newsapi.org/v2/everything?q={requests.utils.quote(query)}&language=en&pageSize=20&apiKey={NEWSAPI_KEY}"
        resp = requests.get(url, timeout=10)
        data = resp.json()
        news_data = []
        for article in data.get("articles", []):
            news_data.append({
                "title": article.get("title"),
                "description": article.get("description") or "",
                "url": article.get("url"),
                "thumbnail": article.get("urlToImage") or "https://via.placeholder.com/120x90?text=No+Image",
                "source": article.get("source", {}).get("name","Unknown")
            })
        return jsonify({"news": news_data})
    except Exception as e:
        print("Error fetching search news:", e)
        return jsonify({"news": []})

@app.route("/api/live-news-widget", methods=["GET"])
def live_news_widget():
    """
    Fetch live business/finance and tech news for the widget using NewsAPI only.
    Ensures every article has a thumbnail.
    """
    news_data = []
    try:
        # 1️⃣ Fetch finance/business news
        url_finance = f"https://newsapi.org/v2/everything?q=finance OR business OR economy&language=en&pageSize=5&apiKey={NEWSAPI_KEY}"
        resp1 = requests.get(url_finance, timeout=10).json()

        # 2️⃣ Fetch technology news
        url_tech = f"https://newsapi.org/v2/everything?q=technology OR AI OR gadgets&language=en&pageSize=5&apiKey={NEWSAPI_KEY}"
        resp2 = requests.get(url_tech, timeout=10).json()

        # Combine results
        articles = resp1.get("articles", []) + resp2.get("articles", [])

        # Placeholder thumbnail
        placeholder_thumb = "https://via.placeholder.com/120x90?text=No+Image"

        # 3️⃣ Format for frontend
        for article in articles:
            thumbnail = article.get("urlToImage") or placeholder_thumb
            news_data.append({
                "title": article.get("title") or "No title",
                "url": article.get("url") or "#",
                "source": article.get("source", {}).get("name","Unknown") if isinstance(article.get("source"), dict) else article.get("source","Unknown"),
                "description": article.get("description") or "No description available",
                "thumbnail": thumbnail
            })

        # 4️⃣ If no articles found, return a default placeholder article
        if not news_data:
            news_data.append({
                "title": "No live business/tech news available",
                "url": "#",
                "source": "NewsLens",
                "description": "Try again later.",
                "thumbnail": placeholder_thumb
            })

    except Exception as e:
        print("Error fetching live news:", e)
        news_data.append({
            "title": "Error fetching live news",
            "url": "#",
            "source": "NewsLens",
            "description": "Please try again later.",
            "thumbnail": "https://via.placeholder.com/120x90?text=No+Image"
        })

    return jsonify({"news": news_data})



@app.route("/verify", methods=["POST"])
def verify():
    data = request.get_json()
    claim_text = data.get("claim_text", "")
    if not claim_text:
        return jsonify({"error": "No claim text provided"}), 400

    result = verify_claim_comprehensive(claim_text)
    
    verdict_color_map = {
        "Likely Accurate": "green",
        "Likely Fake": "red",
        "Uncertain": "orange",
        "Uncertain (Leaning Fake)": "orange",
        "No Reliable Sources Found": "grey"
    }
    color = verdict_color_map.get(result["verdict"], "grey")
    
    csv_articles = get_article_by_claim(claim_text, threshold=0.4)
    
    response = {
        "result": result["verdict"],
        "percentage": result["percentage"],
        "color": color,
        "supporting": result["supporting"][:7],
        "refuting": result["refuting"][:7],
        "neutral": result.get("neutral", [])[:5],
        "articles": [
            {
                "title": a["title"],
                "description": a["description"],
                "url": a["url"],
                "thumbnail": a["thumbnail"],
                "source": a["source"]
            }
            for a in csv_articles[:3]
        ],
        "confidence": result.get("confidence", 0),
        "method": result.get("method", "unknown")
    }
    
    nepali_count = sum(1 for item in (response["supporting"] + response["refuting"]) 
                      if item.get("is_nepali", False))
    
    print(f"\n📊 FINAL API RESPONSE:")
    print(f"   Verdict: {response['result']}")
    print(f"   Percentage: {response['percentage']}%")
    print(f"   Color: {response['color']}")
    print(f"   Method: {response['method']}")
    print(f"   Supporting: {len(response['supporting'])}")
    print(f"   Refuting: {len(response['refuting'])}")
    if nepali_count > 0:
        print(f"   🇳🇵 Nepali sources: {nepali_count}")
    print()
    
    return jsonify(response)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/verify-page")
def verify_page():
    return render_template("verify.html")

@app.route("/health")
def health():
    status = {
        "status": "ok",
        "verify_module": VERIFY_MODULE_AVAILABLE,
        "ml_model": ML_MODEL_AVAILABLE,
        "csv_loaded": not NEWS.empty,
        "csv_articles": len(NEWS) if not NEWS.empty else 0,
        "nepal_sources_enabled": VERIFY_MODULE_AVAILABLE
    }
    return jsonify(status)

if __name__ == "__main__":
    print("\n" + "="*70)
    print("FAKE NEWS DETECTOR - WITH NEPAL SOURCES")
    print("="*70)
    print(f"✓ Flask app initialized")
    print(f"✓ Verify module: {'Available' if VERIFY_MODULE_AVAILABLE else 'Not available'}")
    print(f"✓ ML model: {'Available' if ML_MODEL_AVAILABLE else 'Not available'}")
    print(f"✓ CSV articles: {len(NEWS) if not NEWS.empty else 0}")
    print(f"🇳🇵 Nepal sources: Kathmandu Post, Himalayan Times, My Republica, etc.")
    print("="*70 + "\n")
    
    app.run(debug=True, port=5000)


