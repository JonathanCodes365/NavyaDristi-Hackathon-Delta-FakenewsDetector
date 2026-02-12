# verify_article_improved.py - More Decisive Version
import os
import re
import requests
import datetime
from urllib.parse import quote
from typing import List, Tuple

# ------------------ Config ------------------
GNEWS_API_KEY = os.getenv("GNEWS_API_KEY", "7c5375c69233f6a731714ab09584a219")
GOOGLE_CSE_API_KEY = os.getenv("GOOGLE_CSE_API_KEY", "AIzaSyBypIQi0kJ4JsVwAdM15-ELWr7kmNVPyMA")
GOOGLE_CSE_ENGINE_ID = os.getenv("GOOGLE_CSE_ENGINE_ID", "04ac3616a037847f6")

DEBUG = False

# ------------------ ENHANCED Detection Terms ------------------
# CRITICAL: These indicate the claim is FAKE/DEBUNKED
STRONG_NEGATIVE_TERMS = {
    "false", "fake", "hoax", "debunked", "incorrect", "misinformation",
    "disinformation", "fabricated", "no evidence", "refuted", "myth", 
    "disproved", "fact check", "fact-check", "misleading", "unproven", 
    "unverified", "baseless", "unfounded", "scam", "fraud", "untrue",
    "falsely claims", "no proof", "discredited"
}

# MODERATE: Suggest caution
MODERATE_NEGATIVE_TERMS = {
    "conspiracy theory", "pseudoscience", "rumor", "unconfirmed",
    "alleged", "claims without", "not supported"
}

# POSITIVE: Indicate claim is TRUE/VERIFIED
POSITIVE_TERMS = {
    "confirmed", "verified", "proven", "established", "validated",
    "documented", "evidence shows", "studies show", "research confirms",
    "scientists confirm", "experts confirm", "officially", "peer-reviewed"
}

# Fiction filters
FICTION_TERMS = {
    "video game", "films", "movie", "television", "novel", "series", "fiction",
    "sci-fi", "fantasy", "superhero", "marvel", "dc comics", "anime", "manga",
    "netflix", "streaming", "blockbuster", "box office", "trailer"
}

# Trusted source domains
TRUSTED_DOMAINS = {
    "wikipedia.org", "britannica.com", "reuters.com", "apnews.com",
    "bbc.com", "nature.com", "science.org", "nih.gov", "cdc.gov",
    "who.int", "nasa.gov", "edu", "gov"
}

# ------------------ Content Classification ------------------
def is_fiction_content(text: str, title: str = "") -> bool:
    """Check if content is fiction/entertainment"""
    combined = f"{title} {text}".lower()
    fiction_count = sum(1 for term in FICTION_TERMS if term in combined)
    return fiction_count >= 2

def is_trusted_source(url: str) -> bool:
    """Check if URL is from a trusted domain"""
    url_lower = url.lower()
    return any(domain in url_lower for domain in TRUSTED_DOMAINS)

# ------------------ Google CSE Search ------------------
def google_cse_search(query: str, max_results: int = 8):
    """Search using Google Custom Search Engine"""
    if not GOOGLE_CSE_API_KEY or not GOOGLE_CSE_ENGINE_ID:
        if DEBUG:
            print("⚠ Google CSE API key or Engine ID missing")
        return []
    
    url = "https://customsearch.googleapis.com/customsearch/v1"
    params = {
        "key": GOOGLE_CSE_API_KEY,
        "cx": GOOGLE_CSE_ENGINE_ID,
        "q": query,
        "num": min(max_results, 10)
    }
    
    try:
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
        
        articles = []
        for item in data.get("items", [])[:max_results]:
            title = item.get("title", "")
            snippet = item.get("snippet", "")
            url_link = item.get("link", "")
            
            # Filter fiction
            if is_fiction_content(snippet, title):
                if DEBUG:
                    print(f"  ✗ Filtered (fiction): {title[:50]}...")
                continue
            
            articles.append({
                "source": item.get("displayLink", "Google Search"),
                "title": title,
                "url": url_link,
                "snippet": snippet,
                "is_trusted": is_trusted_source(url_link)
            })
        
        return articles
        
    except Exception as e:
        if DEBUG:
            print(f"Google CSE error: {e}")
        return []

# ------------------ Wikipedia Search ------------------
def wiki_search(query: str, limit: int = 3):
    """Search Wikipedia"""
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
        
        results = []
        for item in data.get("query", {}).get("search", []):
            title = item["title"]
            # Get summary
            summary_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{quote(title)}"
            summary_resp = requests.get(summary_url, headers=headers, timeout=10)
            
            if summary_resp.status_code == 200:
                summary_data = summary_resp.json()
                snippet = summary_data.get("extract", "")
                
                # Filter fiction
                if is_fiction_content(snippet, title):
                    continue
                
                results.append({
                    "source": "Wikipedia",
                    "title": title,
                    "url": summary_data.get("content_urls", {}).get("desktop", {}).get("page", ""),
                    "snippet": snippet,
                    "is_trusted": True
                })
        
        return results
    except Exception as e:
        if DEBUG:
            print(f"Wikipedia error: {e}")
        return []

# ------------------ IMPROVED Stance Detection ------------------
def analyze_evidence(claim: str, snippet: str, source: str, is_trusted: bool) -> dict:
    """
    Analyze evidence and determine stance with improved scoring
    """
    text_lower = snippet.lower()
    claim_lower = claim.lower()
    
    # Count different types of terms
    strong_negative = sum(1 for term in STRONG_NEGATIVE_TERMS if term in text_lower)
    moderate_negative = sum(1 for term in MODERATE_NEGATIVE_TERMS if term in text_lower)
    positive = sum(1 for term in POSITIVE_TERMS if term in text_lower)
    
    # Calculate relevance (simple keyword matching)
    claim_words = set(re.findall(r'\b\w+\b', claim_lower))
    snippet_words = set(re.findall(r'\b\w+\b', text_lower))
    common_words = claim_words & snippet_words
    relevance = len(common_words) / max(len(claim_words), 1)
    
    # Boost for trusted sources
    trust_multiplier = 1.3 if is_trusted else 1.0
    
    # Determine stance and score
    base_score = relevance * 0.5
    
    # REFUTED (Fake)
    if strong_negative >= 2 or (strong_negative >= 1 and relevance > 0.3):
        stance = "refuted"
        score = (0.7 + (strong_negative * 0.1)) * trust_multiplier
        
    # REFUTED (Moderate)
    elif strong_negative >= 1 or moderate_negative >= 2:
        stance = "refuted"
        score = (0.5 + (strong_negative * 0.1)) * trust_multiplier
        
    # SUPPORTED (Strong)
    elif positive >= 2 and strong_negative == 0:
        stance = "supported"
        score = (0.7 + (positive * 0.05)) * trust_multiplier
        
    # SUPPORTED (Moderate) - trusted source, no negatives, decent relevance
    elif is_trusted and strong_negative == 0 and relevance > 0.4:
        stance = "supported"
        score = (0.5 + relevance) * trust_multiplier
        
    # SUPPORTED (Weak) - some relevance, no strong negatives
    elif relevance > 0.5 and strong_negative == 0:
        stance = "supported"
        score = (0.3 + relevance) * trust_multiplier
        
    # INCONCLUSIVE
    else:
        stance = "inconclusive"
        score = base_score
    
    if DEBUG:
        print(f"    Analysis: stance={stance}, score={score:.3f}, relevance={relevance:.3f}")
        print(f"      Strong neg={strong_negative}, Mod neg={moderate_negative}, Pos={positive}")
    
    return {
        "stance": stance,
        "score": min(1.0, score),  # Cap at 1.0
        "relevance": relevance,
        "strong_negative": strong_negative,
        "positive": positive
    }

# ------------------ Build Queries ------------------
def build_queries(title: str, text: str) -> List[str]:
    """Build search queries from claim"""
    queries = []
    
    # Main query: the claim itself
    if title:
        queries.append(title.strip())
    
    # Add fact-check queries
    if title:
        queries.append(f"{title.strip()} fact check")
        queries.append(f"is {title.strip()} true")
    
    return queries[:3]  # Max 3 queries

# ------------------ IMPROVED Verdict Computation ------------------
def compute_verdict(evidence: List[dict]) -> Tuple[str, float, List[dict]]:
    """
    Compute final verdict with more decisive thresholds
    """
    if not evidence:
        return "inconclusive", 0.0, []
    
    # Sort by score
    evidence_sorted = sorted(
        evidence,
        key=lambda e: e.get("score", 0.0),
        reverse=True
    )
    
    # Consider top 10 pieces of evidence
    top = evidence_sorted[:10]
    
    # Calculate totals
    support_score = sum(e.get("score", 0.0) for e in top if e.get("stance") == "supported")
    refute_score = sum(e.get("score", 0.0) for e in top if e.get("stance") == "refuted")
    total_score = support_score + refute_score
    
    # Count evidence pieces
    support_count = sum(1 for e in top if e.get("stance") == "supported")
    refute_count = sum(1 for e in top if e.get("stance") == "refuted")
    
    if DEBUG:
        print(f"\n  Verdict Calculation:")
        print(f"    Support score: {support_score:.3f} (count: {support_count})")
        print(f"    Refute score: {refute_score:.3f} (count: {refute_count})")
        print(f"    Total score: {total_score:.3f}")
    
    # IMPROVED DECISION LOGIC
    
    # No meaningful evidence
    if total_score < 0.5:
        return "inconclusive", 0.5, top
    
    # Calculate ratios
    if total_score > 0:
        refute_ratio = refute_score / total_score
        support_ratio = support_score / total_score
    else:
        refute_ratio = 0
        support_ratio = 0
    
    # REFUTED (Fake) - More decisive
    # If we have ANY strong refutation evidence, lean toward fake
    if refute_score > 0.8:  # Strong refutation
        verdict = "refuted"
        confidence = min(0.95, 0.7 + refute_score * 0.2)
        
    elif refute_ratio > 0.4 and refute_score > 0.3:  # Moderate refutation
        verdict = "refuted"
        confidence = min(0.85, 0.6 + refute_ratio * 0.2)
        
    # SUPPORTED (Accurate) - More decisive
    # If we have good support and little/no refutation, mark as accurate
    elif support_score > 1.2:  # Strong support
        verdict = "supported"
        confidence = min(0.95, 0.7 + support_score * 0.1)
        
    elif support_ratio > 0.6 and support_score > 0.5:  # Moderate support
        verdict = "supported"
        confidence = min(0.85, 0.6 + support_ratio * 0.2)
        
    # MIXED SIGNALS - Be careful
    elif abs(support_score - refute_score) < 0.3:
        verdict = "inconclusive"
        confidence = 0.5
        
    # DEFAULT: Go with the stronger side
    elif support_score > refute_score:
        verdict = "supported"
        confidence = 0.6 + (support_ratio * 0.2)
    else:
        verdict = "refuted"
        confidence = 0.6 + (refute_ratio * 0.2)
    
    if DEBUG:
        print(f"    Final: {verdict} (confidence: {confidence:.3f})")
    
    return verdict, float(confidence), top

# ------------------ MAIN VERIFICATION ------------------
def verify_article(title: str, text: str, max_pages_per_query: int = 5, debug: bool = False):
    """
    Main verification function with improved decision making
    """
    global DEBUG
    DEBUG = debug
    
    claim = title.strip() if title else ""
    if not claim:
        return {
            "verdict": "inconclusive",
            "confidence": 0.0,
            "evidence": [],
            "debug": {"queries": [], "reason": "No claim provided"}
        }
    
    if DEBUG:
        print("=" * 70)
        print("FAKE NEWS DETECTOR - IMPROVED DECISIVE VERSION")
        print("=" * 70)
        print(f"Claim: {claim}")
        print("=" * 70)
    
    # Build queries
    queries = build_queries(claim, text)
    
    all_evidence = []
    seen_urls = set()
    
    for q_idx, q in enumerate(queries, 1):
        if DEBUG:
            print(f"\n[Query {q_idx}/{len(queries)}]: {q}")
        
        # 1. Wikipedia Search
        wiki_results = wiki_search(q, limit=3)
        for result in wiki_results:
            url = result.get("url", "")
            if url in seen_urls:
                continue
            seen_urls.add(url)
            
            analysis = analyze_evidence(
                claim=claim,
                snippet=result["snippet"],
                source=result["source"],
                is_trusted=True
            )
            
            all_evidence.append({
                **result,
                **analysis
            })
            
            if DEBUG:
                print(f"  ✓ Wiki: {result['title'][:50]}...")
        
        # 2. Google CSE Search
        cse_results = google_cse_search(q, max_results=max_pages_per_query)
        for result in cse_results:
            url = result.get("url", "")
            if url in seen_urls:
                continue
            seen_urls.add(url)
            
            analysis = analyze_evidence(
                claim=claim,
                snippet=result["snippet"],
                source=result["source"],
                is_trusted=result.get("is_trusted", False)
            )
            
            all_evidence.append({
                **result,
                **analysis
            })
            
            if DEBUG:
                print(f"  ✓ CSE: {result['title'][:50]}...")
    
    # Compute verdict
    if not all_evidence:
        if DEBUG:
            print("\n❌ No evidence found!")
        return {
            "verdict": "inconclusive",
            "confidence": 0.0,
            "evidence": [],
            "debug": {
                "queries": queries,
                "reason": "No evidence found"
            }
        }
    
    verdict, confidence, top_evidence = compute_verdict(all_evidence)
    
    # Format evidence for output
    formatted_evidence = []
    for e in top_evidence:
        formatted_evidence.append({
            "source": e.get("source", "Unknown"),
            "title": e.get("title", ""),
            "url": e.get("url", ""),
            "snippet": e.get("snippet", "")[:200],  # Truncate long snippets
            "stance": e.get("stance", "inconclusive"),
            "score": round(e.get("score", 0), 3),
            "relevance": round(e.get("relevance", 0), 3)
        })
    
    result = {
        "verdict": verdict,
        "confidence": confidence,
        "evidence": formatted_evidence,
        "debug": {
            "queries": queries,
            "total_evidence": len(all_evidence),
            "top_evidence_count": len(formatted_evidence)
        }
    }
    
    if DEBUG:
        print("\n" + "=" * 70)
        print("FINAL RESULT")
        print("=" * 70)
        print(f"Verdict: {verdict.upper()}")
        print(f"Confidence: {confidence:.1%}")
        print(f"Total evidence: {len(all_evidence)}")
        print(f"\nTop 5 Evidence:")
        for i, e in enumerate(formatted_evidence[:5], 1):
            print(f"  {i}. [{e['stance'].upper()}] {e['source']}")
            print(f"     Score: {e['score']:.3f} | {e['title'][:60]}...")
        print("=" * 70)
    
    return result

# ------------------ Main ------------------
if __name__ == "__main__":
    import sys
    
    # Test cases
    test_claims = [
        ("Vaccines cause autism", False),
        ("Earth orbits the Sun", True),
        ("COVID-19 vaccines contain microchips", False),
        ("Climate change is real", True),
        ("5G causes coronavirus", False),
    ]
    
    print("\n" + "=" * 70)
    print("TESTING VERIFICATION SYSTEM")
    print("=" * 70)
    
    for claim, expected_true in test_claims:
        print(f"\n\nTesting: {claim}")
        print(f"Expected: {'ACCURATE' if expected_true else 'FAKE'}")
        print("-" * 70)
        
        result = verify_article(claim, "", debug=False)
        
        verdict = result["verdict"]
        confidence = result["confidence"]
        
        print(f"Result: {verdict.upper()} ({confidence:.1%} confidence)")
        
        # Check if correct
        is_correct = (expected_true and verdict == "supported") or \
                     (not expected_true and verdict == "refuted")
        
        print(f"Status: {'✓ CORRECT' if is_correct else '✗ INCORRECT'}")
