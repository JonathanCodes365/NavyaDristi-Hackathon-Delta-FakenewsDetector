# verify.py
import os
import re
import requests
import feedparser
from urllib.parse import quote
from typing import List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ------------------ Config ------------------
# Read GNews API key from environment (export GNEWS_API_KEY="...")
GNEWS_API_KEY = os.getenv("GNEWS_API_KEY", "")

# Toggle filters and sources
POP_CULTURE_FILTER = True  # Skip obvious pop-culture pages
INCLUDE_RSS = True         # Add RSS evidence
GNEWS_ENABLED = True  # static config only (do NOT mutate)

# ------------------ Static terms ------------------
NEGATIVE_TERMS = {
    "false", "fake", "hoax", "debunked", "incorrect", "misinformation",
    "pseudoscience", "conspiracy theory", "discredited", "fabricated",
    "no evidence", "refute", "refuted", "not true", "myth", "disproved"
}

POP_CULTURE_TERMS = ["video game", "films", "songs", "television", "novel", "series", "movie", "fiction"]
SCIENCE_TERMS = ["extraterrestrial", "ufo", "nasa", "seti", "astronomy", "planet", "life", "astrobiology"]

# ------------------ Utilities ------------------
def first_sentence(text: str) -> str:
    if not text:
        return ""
    parts = re.split(r"[.!?]\s+", text.strip())
    return parts[0] if parts else text.strip()

def build_queries(title: str, text: str, max_queries: int = 3) -> List[str]:
    q = [title.strip()] if title else []
    fs = first_sentence(text)
    if fs and (not title or fs.lower() != title.lower()) and len(fs.split()) >= 4:
        q.append(fs)
    tokens = re.findall(r"[A-Za-z0-9]+", (title or fs).lower())
    keywords = [t for t in tokens if len(t) > 3][:4]
    if keywords and " ".join(keywords) not in q:
        q.append(" ".join(keywords))
    return q[:max_queries]

def cosine_sim(a: str, b: str) -> float:
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1,2), max_features=5000)
    try:
        X = vec.fit_transform([a, b])
        sim = cosine_similarity(X[0], X[1])[0][0]
        return float(sim)
    except ValueError:
        return 0.0

# ------------------ Wikipedia ------------------
def wiki_search(query: str, limit: int = 5, debug: bool = False) -> List[str]:
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "srlimit": str(limit),
        "format": "json"
    }
    headers = {"User-Agent": "FakeNewsDetector/1.0"}

    try:
        r = requests.get(url, params=params, headers=headers, timeout=10)
        r.raise_for_status()
        data = r.json()
        return [item["title"] for item in data.get("query", {}).get("search", [])]
    except Exception as e:
        if debug:
            print("wiki_search error:", e)
        return []

def wiki_summary(title: str, debug: bool = False):
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{quote(title)}"
    headers = {"User-Agent": "FakeNewsDetector/1.0"}
    try:
        r = requests.get(url, headers=headers, timeout=10)
        if r.status_code != 200:
            return None
        data = r.json()
        return {
            "source": "wikipedia",
            "title": data.get("title", title),
            "url": data.get("content_urls", {}).get("desktop", {}).get("page",
                   f"https://en.wikipedia.org/wiki/{quote(title)}"),
            "snippet": data.get("extract", "")
        }
    
    except Exception as e:
        if debug:
            print("wiki_summary error:", e)
        return None

def wiki_categories(title: str) -> List[str]:
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "prop": "categories",
        "titles": title,
        "format": "json",
        "cllimit": "max"
    }
    headers = {"User-Agent": "FakeNewsDetector/1.0"}
    try:
        r = requests.get(url, params=params, headers=headers, timeout=10)
        data = r.json()
    except Exception:
        return []
    pages = data.get("query", {}).get("pages", {})
    categories = []
    for p in pages.values():
        for c in p.get("categories", []):
            categories.append(c.get("title", "").lower())
    return categories

def is_pop_culture(categories: List[str], snippet_lower: str) -> bool:
    return any(any(term in c for term in POP_CULTURE_TERMS) for c in categories) or \
           any(term in snippet_lower for term in POP_CULTURE_TERMS)

def gnews_search(query: str, max_results: int = 5, lang: str = "en",
                 country: str = "us", debug: bool = False):

    # Check for missing API key
    if not GNEWS_API_KEY:
        if debug:
            print("WARNING: GNews API key missing. GNews evidence will be skipped.")
        return []

    url = "https://gnews.io/api/v4/search"
    params = {
        "q": query,
        "lang": lang,
        "country": country,
        "max": max_results,
        "apikey": GNEWS_API_KEY
    }

    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()

        if "errors" in data:
            if debug:
                print("GNews API error:", data["errors"])
            return []

        articles = []

        for art in data.get("articles", []):
            snippet = (
                art.get("description")
                or art.get("content")
                or art.get("title", "")
            )

            articles.append({
                "source": art.get("source", {}).get("name", ""),
                "title": art.get("title", ""),
                "url": art.get("url", ""),
                "snippet": snippet
            })

        return articles

    except requests.exceptions.RequestException as e:
        if debug:
            print("GNews HTTP request failed:", e)
        return []

# ------------------ RSS ------------------
TRUSTED_RSS_FEEDS = [
    "http://rss.cnn.com/rss/cnn_topstories.rss",
    "http://feeds.bbci.co.uk/news/rss.xml",
    "http://feeds.reuters.com/reuters/worldNews",
    "http://feeds.abcnews.com/abcnews/usheadlines",
    "http://feeds.nbcnews.com/feeds/worldnews"
]

def rss_search(feed_urls: List[str], limit: int = 5, debug: bool = False):
    articles = []

    for url in feed_urls:
        try:
            feed = feedparser.parse(url)
            count = 0

            for entry in feed.entries:
                if count >= limit:
                    break

                snippet = (
                    entry.get("summary")
                    or entry.get("description", "")
                    or entry.get("title", "")
                )

                articles.append({
                    "source": feed.feed.get("title", "rss"),
                    "title": entry.get("title", ""),
                    "url": entry.get("link", ""),
                    "snippet": snippet
                })

                count += 1

        except Exception as e:
            if debug:
                print("RSS fetch error for", url, ":", e)

    return articles

# ------------------ Stance and verdict ------------------
def simple_stance(claim: str, snippet: str, sim: float, sim_thresh: float = 0.15) -> Tuple[str, float]:
    text = (snippet or "").lower()
    has_neg = any(term in text for term in NEGATIVE_TERMS)

    if has_neg and sim >= 0.12:
        return "refuted", min(0.9, 0.5 + sim)

    if sim >= sim_thresh:
        return "supported", min(0.85, 0.4 + sim)

    return "inconclusive", max(0.35, sim)

def compute_verdict(evidence: List[dict]) -> Tuple[str, float, List[dict]]:
    evidence_sorted = sorted(
        evidence,
        key=lambda e: (e.get("score", 0.0), e.get("similarity", 0.0)),
        reverse=True
    )
    top = evidence_sorted[:5]

    support = sum(e.get("score", 0.0) for e in top if e.get("stance") == "supported")
    refute  = sum(e.get("score", 0.0) for e in top if e.get("stance") == "refuted")
    total   = max(1e-6, support + refute)

    if support > refute * 1.3 and support > 0.5:
        verdict = "supported"
    elif refute > support * 1.3 and refute > 0.5:
        verdict = "refuted"
    else:
        verdict = "inconclusive"

    confidence = min(0.95, max(0.5, abs(support - refute) / (total + 1e-6)))
    top_for_output = [
        {k: v for k, v in e.items() if k in ("source", "title", "url", "snippet", "stance", "score")}
        for e in top
    ]
    return verdict, float(confidence), top_for_output

# ------------------ Main verify ------------------
def verify_article(title: str, text: str, max_pages_per_query: int = 3, debug: bool = False):

    claim = title.strip() if title else first_sentence(text)
    if not claim:
        return {
            "verdict": "inconclusive",
            "confidence": 0.0,
            "evidence": [],
            "debug": {"queries": []}
        }

    queries = build_queries(title or "", text or "")
    seen_titles = set()
    evidence = []

    if debug:
        print("Claim:", claim)
        print("Queries:", queries)

    # For each query
    for q in queries:

        # --- Wikipedia ---
        wiki_titles = wiki_search(q, limit=max_pages_per_query, debug=debug)

        for t in wiki_titles:
            if t in seen_titles:
                continue
            seen_titles.add(t)

            categories = wiki_categories(t)
            summ = wiki_summary(t, debug=debug)

            if not summ or not summ.get("snippet"):
                continue

            snippet_lower = summ["snippet"].lower()

            if POP_CULTURE_FILTER and is_pop_culture(categories, snippet_lower):
                continue

            sim = cosine_sim(claim, summ["snippet"])

            if any(sc in snippet_lower for sc in SCIENCE_TERMS):
                sim = min(1.0, sim + 0.03)

            stance, score = simple_stance(claim, summ["snippet"], sim)

            evidence.append({
                **summ,
                "stance": stance,
                "score": float(score),
                "similarity": float(sim)
            })

        # --- GNews ---
        if GNEWS_ENABLED and GNEWS_API_KEY:
            news_articles = gnews_search(
                q,
                max_results=max_pages_per_query,
                debug=debug
            )

            for art in news_articles:
                sim = cosine_sim(claim, art["snippet"])
                stance, score = simple_stance(claim, art["snippet"], sim)

                evidence.append({
                    **art,
                    "stance": stance,
                    "score": float(score),
                    "similarity": float(sim)
                })

        # --- RSS ---
        if INCLUDE_RSS:
            rss_articles = rss_search(TRUSTED_RSS_FEEDS, limit=max_pages_per_query)

            for art in rss_articles:
                key = (art.get("source", ""), art.get("title", ""))
                if key in seen_titles:
                    continue
                seen_titles.add(key)

                sim = cosine_sim(claim, art["snippet"])
                stance, score = simple_stance(claim, art["snippet"], sim)

                evidence.append({
                    **art,
                    "stance": stance,
                    "score": float(score),
                    "similarity": float(sim)
                })

    # --- Fallback Wikipedia pass ---
    if not any(e.get("source") == "wikipedia" for e in evidence):

        if debug:
            print("No Wikipedia evidence found. Running fallback pass.")

        for q in queries:
            wiki_titles = wiki_search(q, limit=max_pages_per_query, debug=debug)

            for t in wiki_titles:
                summ = wiki_summary(t, debug=debug)

                if not summ or not summ.get("snippet"):
                    continue

                sim = cosine_sim(claim, summ["snippet"])
                stance, score = simple_stance(claim, summ["snippet"], sim)

                evidence.append({
                    **summ,
                    "stance": stance,
                    "score": float(score),
                    "similarity": float(sim)
                })

    if not evidence:
        return {
            "verdict": "inconclusive",
            "confidence": 0.0,
            "evidence": [],
            "debug": {"queries": queries}
        }

    verdict, confidence, top_evidence = compute_verdict(evidence)

    return {
        "verdict": verdict,
        "confidence": confidence,
        "evidence": top_evidence,
        "debug": {"queries": queries}
    }
