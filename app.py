
from flask import Flask, request, jsonify, render_template
import pandas as pd
import os
import re
import requests
import string
import joblib
from difflib import SequenceMatcher

# FIXED IMPORT - Use the correct module name
try:
    from verify_article1 import verify_article
    VERIFY_MODULE_AVAILABLE = True
    print("✓ Using verify_article_improved.py with Nepal sources support")
except ImportError:
    VERIFY_MODULE_AVAILABLE = False
    print("⚠ Verification module not found")

app = Flask(__name__)

# ---------------- Configuration ----------------
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY", "947f0e92eefd464caab081103c41bbc7")

# Load model and vectorizer (optional)
try:
    model = joblib.load("lr.model.jb")
    vectorizer = joblib.load("vectorizer.jb")
    ML_MODEL_AVAILABLE = True
    print("✓ ML model loaded")
except Exception as e:
    ML_MODEL_AVAILABLE = False
    print("⚠ ML model not loaded:", e)

# Load smart news CSV
try:
    NEWS = pd.read_csv("smart_news.csv")
    for col in ['title', 'description', 'url', 'thumbnail', 'source']:
        if col in NEWS.columns:
            NEWS[col] = NEWS[col].astype(str).str.strip()
    print(f"✓ Loaded {len(NEWS)} articles from CSV")
except Exception as e:
    NEWS = pd.DataFrame()
    print("⚠ Could not load smart_news.csv:", e)

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

# ---------------- CSV Matching ----------------
def get_article_by_claim(claim_text, threshold=0.5):
    """Fuzzy match against local CSV news"""
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
    
    # Sort by similarity
    matched_articles.sort(key=lambda x: x['similarity'], reverse=True)
    return matched_articles

# ---------------- Main Verification Function ----------------
def verify_claim_comprehensive(claim_text):
    """
    Comprehensive verification with Nepal sources
    """
    
    results = {
        "verdict": "Uncertain",
        "confidence": 0.0,
        "supporting": [],
        "refuting": [],
        "neutral": [],
        "percentage": 50,
        "method": "unknown"
    }
    
    # PRIMARY: Try advanced verification module
    if VERIFY_MODULE_AVAILABLE:
        try:
            print(f"\n{'='*70}")
            print(f"VERIFYING CLAIM: {claim_text}")
            
            # Check if Nepal-related
            nepal_keywords = ["nepal", "kathmandu", "pokhara", "nepali", "nepalese", 
                            "himalayan", "everest", "नेपाल"]
            is_nepal_related = any(kw in claim_text.lower() for kw in nepal_keywords)
            
            if is_nepal_related:
                print(f"🇳🇵 NEPAL-RELATED CLAIM DETECTED - Will search Nepali sources")
            
            print(f"{'='*70}")
            
            # Call the verification with Nepal support
            result = verify_article(
                title=claim_text,
                text="",
                max_pages_per_query=8,
                debug=True
            )
            
            verdict = result.get("verdict", "inconclusive")
            confidence = result.get("confidence", 0.0)
            evidence = result.get("evidence", [])
            
            # Count Nepali sources in evidence
            nepali_sources = sum(1 for e in evidence if e.get("is_nepal", False))
            
            print(f"\nVerification Result:")
            print(f"  Verdict: {verdict}")
            print(f"  Confidence: {confidence:.1%}")
            print(f"  Evidence count: {len(evidence)}")
            if nepali_sources > 0:
                print(f"  🇳🇵 Nepali sources: {nepali_sources}/{len(evidence)}")
            
            # Convert verdict to our format
            if verdict == "refuted":
                results["verdict"] = "Likely Fake"
                results["percentage"] = int(confidence * 100)
            elif verdict == "supported":
                results["verdict"] = "Likely Accurate"
                # For "Likely Accurate", show the inverse percentage
                # If confidence is 0.8 (80% confident it's accurate), show as 20% fake probability
                results["percentage"] = 100 - int(confidence * 100)
            else:
                if confidence > 0.5:
                    results["verdict"] = "Uncertain (Leaning Fake)"
                    results["percentage"] = 60
                else:
                    results["verdict"] = "Uncertain"
                    results["percentage"] = 50
            
            results["confidence"] = confidence
            results["method"] = "verify_with_nepal_sources"
            
            # Categorize evidence and highlight Nepali sources
            for e in evidence:
                stance = e.get("stance", "inconclusive")
                source = e.get("source", "Unknown")
                
                evidence_item = {
                    "source": source,
                    "snippet": e.get("snippet", ""),
                    "url": e.get("url", ""),
                    "text": e.get("title", ""),
                    "is_nepali": e.get("is_nepal", False)
                }
                
                if stance == "refuted":
                    results["refuting"].append(evidence_item)
                elif stance == "supported":
                    results["supporting"].append(evidence_item)
                else:
                    results["neutral"].append(evidence_item)
            
            # Boost confidence if we have Nepali sources on Nepal topics
            if is_nepal_related and nepali_sources >= 2:
                if verdict == "refuted" and results["percentage"] < 70:
                    results["percentage"] = min(results["percentage"] + 10, 85)
                elif verdict == "supported" and results["percentage"] > 30:
                    results["percentage"] = max(results["percentage"] - 10, 15)
                print(f"  🎯 Nepal topic boost applied")
            
            print(f"\nFinal Results:")
            print(f"  Verdict: {results['verdict']}")
            print(f"  Percentage: {results['percentage']}%")
            print(f"  Supporting: {len(results['supporting'])}")
            print(f"  Refuting: {len(results['refuting'])}")
            print(f"{'='*70}\n")
            
            return results
            
        except Exception as e:
            print(f"❌ Error in verification: {e}")
            import traceback
            traceback.print_exc()
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
            print(f"Using ML model: {results['verdict']} ({ml_fake_prob}%)")
            
        except Exception as e:
            print(f"ML model error: {e}")
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
        articles = data.get("articles", [])

        news_data = []
        for article in articles:
            news_data.append({
                "title": article.get("title"),
                "description": article.get("description") or "",
                "url": article.get("url"),
                "thumbnail": article.get("urlToImage") or "https://via.placeholder.com/120x90?text=No+Image",
                "source": article.get("source", {}).get("name","Unknown")
            })
        
        return jsonify({"news": news_data})
    except Exception as e:
        print("Error fetching news:", e)
        return jsonify({"news": []})

@app.route("/api/live-news-widget", methods=["GET"])
def live_news_widget():
    news_data = []
    try:
        url_business = f"https://newsapi.org/v2/top-headlines?category=business&language=en&pageSize=5&apiKey={NEWSAPI_KEY}"
        resp1 = requests.get(url_business, timeout=10).json()
        url_tech = f"https://newsapi.org/v2/top-headlines?category=technology&language=en&pageSize=5&apiKey={NEWSAPI_KEY}"
        resp2 = requests.get(url_tech, timeout=10).json()

        all_articles = resp1.get("articles", []) + resp2.get("articles", [])
        for article in all_articles:
            news_data.append({
                "title": article.get("title"),
                "url": article.get("url"),
                "source": article.get("source", {}).get("name","Unknown"),
                "description": article.get("description") or "",
                "thumbnail": article.get("urlToImage") or "https://via.placeholder.com/120x90?text=No+Image"
            })
    except Exception as e:
        print("Error fetching live news:", e)

    return jsonify({"news": news_data})

@app.route("/verify", methods=["POST"])
def verify():
    """Main verification endpoint with Nepal sources"""
    data = request.get_json()
    claim_text = data.get("claim_text", "")
    
    if not claim_text:
        return jsonify({"error": "No claim text provided"}), 400

    # Use comprehensive verification
    result = verify_claim_comprehensive(claim_text)
    
    # Map verdict to color
    verdict_color_map = {
        "Likely Accurate": "green",
        "Likely Fake": "red",
        "Uncertain": "orange",
        "Uncertain (Leaning Fake)": "orange",
        "No Reliable Sources Found": "grey"
    }
    color = verdict_color_map.get(result["verdict"], "grey")
    
    # Get matching articles from CSV
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
    
    # Count Nepali sources in response
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
    """Health check endpoint"""
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
