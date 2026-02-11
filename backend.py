from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
import requests, time

app = Flask(__name__)
CORS(app)

NEWS_API_KEY = "947f0e92eefd464caab081103c41bbc7"  # replace with your key
cached_articles = []
last_fetched = 0

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get-news")
def get_news():
    global cached_articles, last_fetched
    if time.time() - last_fetched > 300:  # cache for 5 mins
        url = f"https://newsapi.org/v2/top-headlines?country=us&apiKey={NEWS_API_KEY}"
        response = requests.get(url)
        data = response.json()
        cached_articles = [
            {"title": a.get("title"), "url": a.get("url"), "urlToImage": a.get("urlToImage")}
            for a in data.get("articles", [])
        ]
        last_fetched = time.time()
    return jsonify({"articles": cached_articles})
@app.route("/search-news")
def search_news():
    raw_query = request.args.get("query", "").strip()
    if not raw_query:
        return jsonify({"articles": []})

    stopwords = {"is","a","the","and","or","of","in","to","for","was"}
    keywords = [w for w in raw_query.split() if w.lower() not in stopwords]
    if not keywords:
        keywords = [raw_query]

    query_param = " AND ".join([f'"{w}"' for w in keywords])

    url = (
        f"https://newsapi.org/v2/everything?"
        f"qInTitle={query_param}&language=en&sortBy=publishedAt&pageSize=10&apiKey={NEWS_API_KEY}"
    )
    response = requests.get(url)
    data = response.json()

    articles = [
        {"title": a.get("title"), "url": a.get("url"), "urlToImage": a.get("urlToImage")}
        for a in data.get("articles", [])
    ]
    return jsonify({"articles": articles})


@app.route("/verify", methods=["POST"])
def verify_news():
    content = request.json
    text = content.get("text", "")
    result = {
        "ai_label": "Real",
        "confidence": 0.92,
        "cross_result": "Verified",
        "cross_sources": ["https://example.com/news1", "https://example.com/news2"]
    }
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)
