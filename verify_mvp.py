import requests
import re
from urllib.parse import quote
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

NEGATIVE_TERMS = {
    "false", "fake", "hoax", "debunked", "incorrect", "misinformation",
    "pseudoscience", "conspiracy theory", "discredited", "fabricated",
    "no evidence", "refute", "refuted", "not true", "myth", "disproved"
}

def first_sentence(text: str) -> str:
    if not text:
        return ""
    parts = re.split(r"[.!?]\s+", text.strip())
    return parts[0] if parts else text.strip()

def build_queries(title: str, text: str, max_queries: int = 3):
    q = [title.strip()]
    fs = first_sentence(text)
    if fs and fs.lower() != title.lower() and len(fs.split()) >= 4:
        q.append(fs)
    # Add a shorter keywordy version of the title
    tokens = re.findall(r"[A-Za-z0-9]+", title.lower())
    keywords = [t for t in tokens if len(t) > 3][:4]
    if keywords and " ".join(keywords) not in q:
        q.append(" ".join(keywords))
    return q[:max_queries]

def wiki_search(query: str, limit: int = 5):
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "srlimit": str(limit),
        "format": "json"
    }
    headers = {
        "User-Agent": "FakeNewsDetector/1.0 (ngawangt.sherpa777@gmail.com)"
    }
    r = requests.get(url, params=params, headers=headers, timeout=10)
    r.raise_for_status()
    data = r.json()
    return [item["title"] for item in data.get("query", {}).get("search", [])]

def wiki_summary(title: str):
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{quote(title)}"
    headers = {
        "User-Agent": "FakeNewsDetector/1.0 (ngawangt.sherpa777@gmail.com)"
    }
    r = requests.get(url, headers=headers, timeout=10)
    if r.status_code != 200:
        return None
    data = r.json()
    return {
        "source": "wikipedia",
        "title": data.get("title", title),
        "url": data.get("content_urls", {}).get("desktop", {}).get("page", f"https://en.wikipedia.org/wiki/{quote(title)}"),
        "snippet": data.get("extract", "")
    }

def cosine_sim(a: str, b: str) -> float:
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1,2), max_features=5000)
    try:
        X = vec.fit_transform([a, b])
        sim = cosine_similarity(X[0], X[1])[0][0]
        return float(sim)
    except ValueError:
        return 0.0

def simple_stance(claim: str, snippet: str, sim: float, sim_thresh: float = 0.22):
    text = (snippet or "").lower()
    has_neg = any(term in text for term in NEGATIVE_TERMS)
    if has_neg and sim >= 0.15:
        return "refuted", min(0.9, 0.5 + sim)
    if sim >= sim_thresh:
        return "supported", min(0.85, 0.4 + sim)
    return "inconclusive", max(0.4, sim)

def verify_article(title: str, text: str, max_pages_per_query: int = 3):
    claim = title.strip() if title else first_sentence(text)
    if not claim:
        return {"verdict": "inconclusive", "confidence": 0.0, "evidence": [], "debug": {"queries": []}}

    queries = build_queries(title, text)
    seen_titles = set()
    evidence = []

    for q in queries:
        titles = wiki_search(q, limit=max_pages_per_query)
        for t in titles:
            if t in seen_titles:
                continue
            seen_titles.add(t)
            summ = wiki_summary(t)
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
        return {"verdict": "inconclusive", "confidence": 0.0, "evidence": [], "debug": {"queries": queries}}

    evidence.sort(key=lambda e: (e["score"], e["similarity"]), reverse=True)
    top = evidence[:5]

    support = sum(e["score"] for e in top if e["stance"] == "supported")
    refute = sum(e["score"] for e in top if e["stance"] == "refuted")
    total = max(1e-6, support + refute)

    if support > refute * 1.8 and support > 0.6:
        verdict = "supported"
    elif refute > support * 1.5 and refute > 0.5:
        verdict = "refuted"
    else:
        verdict = "inconclusive"

    confidence = min(0.95, max(0.5, abs(support - refute) / (total + 1e-6)))

    return {
        "verdict": verdict,
        "confidence": float(confidence),
        "evidence": [
            {k: v for k, v in e.items() if k in ("source", "title", "url", "snippet", "stance", "score")}
            for e in top
        ],
        "debug": {"queries": queries}
    }

if __name__ == "__main__":
    # Quick manual test
    examples = [
        {"title": "The Earth is flat", "text": ""},
        {"title": "COVID-19 vaccines cause microchips", "text": ""},
        {"title": "Barack Obama was born in Hawaii", "text": ""},
    ]
    for ex in examples:
        print("\n=== Claim:", ex["title"])
        result = verify_article(ex["title"], ex["text"])
        print("Verdict:", result["verdict"], "| Confidence:", round(result["confidence"], 3))
        for ev in result["evidence"][:3]:
            print("-", ev["stance"].upper(), "|", ev["title"], "|", ev["url"])