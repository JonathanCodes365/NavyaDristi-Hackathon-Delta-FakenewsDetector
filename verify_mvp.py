
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

# ------------------ NEPAL-SPECIFIC DOMAINS ------------------
NEPAL_DOMAINS = {
    "kathmandupost.com",
    "thehimalayantimes.com",
    "myrepublica.nagariknetwork.com",
    "ekantipur.com",
    "setopati.com",
    "ratopati.com",
    "onlinekhabar.com",
    "nepalisansar.com",
    "gorkhapatra.org.np",
    "risingnepaldaily.com",
    "nepalitimes.com",
    "newsofnepal.com"
}

# ------------------ ENHANCED Detection Terms ------------------
STRONG_NEGATIVE_TERMS = {
    "false", "fake", "hoax", "debunked", "incorrect", "misinformation", 
    "disinformation", "fabricated", "no evidence", "refuted", "myth", 
    "disproved", "fact check", "fact-check", "misleading", "unproven", 
    "unverified", "baseless", "unfounded", "scam", "fraud", "untrue", 
    "falsely claims", "no proof", "discredited"
}

MODERATE_NEGATIVE_TERMS = {
    "conspiracy theory", "pseudoscience", "rumor", "unconfirmed", 
    "alleged", "claims without", "not supported"
}

POSITIVE_TERMS = {
    "confirmed", "verified", "proven", "established", "validated", 
    "documented", "evidence shows", "studies show", "research confirms", 
    "scientists confirm", "experts confirm", "officially", "peer-reviewed",
    "according to", "reports confirm", "announced"
}

FICTION_TERMS = {
    "video game", "films", "movie", "television", "novel", "series", 
    "fiction", "sci-fi", "fantasy", "superhero", "marvel", "dc comics", 
    "anime", "manga", "netflix", "streaming", "blockbuster", "box office", 
    "trailer"
}

TRUSTED_DOMAINS = {
    "wikipedia.org", "britannica.com", "reuters.com", "apnews.com", 
    "bbc.com", "nature.com", "science.org", "nih.gov", "cdc.gov", 
    "who.int", "nasa.gov", "edu", "gov"
}

# Combine Nepal domains with trusted domains
ALL_TRUSTED_DOMAINS = TRUSTED_DOMAINS | NEPAL_DOMAINS

# ------------------ Content Classification ------------------
def is_fiction_content(text: str, title: str = "") -> bool:
    """Check if content is fiction/entertainment"""
    combined = f"{title} {text}".lower()
    fiction_count = sum(1 for term in FICTION_TERMS if term in combined)
    return fiction_count >= 2

def is_trusted_source(url: str) -> bool:
    """Check if URL is from a trusted domain (including Nepal)"""
    url_lower = url.lower()
    return any(domain in url_lower for domain in ALL_TRUSTED_DOMAINS)

def is_nepal_source(url: str) -> bool:
    """Check if URL is from a Nepal source"""
    url_lower = url.lower()
    return any(domain in url_lower for domain in NEPAL_DOMAINS)

# ------------------ NEPAL-SPECIFIC GNews Search ------------------
def gnews_search_nepal(query: str, max_results: int = 10):
    """Search GNews API specifically for Nepal sources"""
    if not GNEWS_API_KEY or GNEWS_API_KEY == "your_gnews_key_here":
        if DEBUG:
            print("⚠ GNews API key missing")
        return []
    
    url = "https://gnews.io/api/v4/search"
    params = {
        "q": query,
        "token": GNEWS_API_KEY,
        "lang": "en",
        "country": "np",  # Nepal country code
        "max": max_results
    }
    
    try:
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
        
        articles = []
        for item in data.get("articles", [])[:max_results]:
            title = item.get("title", "")
            description = item.get("description", "")
            url_link = item.get("url", "")
            source_name = item.get("source", {}).get("name", "Nepal News")
            
            # Filter fiction
            if is_fiction_content(description, title):
                if DEBUG:
                    print(f" 🇳🇵 ✗ Filtered (fiction): {title[:50]}...")
                continue
            
            articles.append({
                "source": f"🇳🇵 {source_name}",
                "title": title,
                "url": url_link,
                "snippet": description,
                "is_trusted": True,  # Nepal sources trusted for Nepal news
                "is_nepal": True
            })
            
            if DEBUG:
                print(f" 🇳🇵 ✓ Nepal source: {source_name} - {title[:50]}...")
        
        return articles
        
    except Exception as e:
        if DEBUG:
            print(f"🇳🇵 GNews Nepal search error: {e}")
        return []

# ------------------ Google CSE Search WITH NEPAL BIAS ------------------
def google_cse_search(query: str, max_results: int = 8, prefer_nepal: bool = False):
    """Search using Google Custom Search Engine with optional Nepal bias"""
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
    
    # Add Nepal regional bias if requested
    if prefer_nepal:
        params["gl"] = "np"  # Geolocation: Nepal
        params["cr"] = "countryNP"  # Country restrict: Nepal
    
    try:
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
        
        articles = []
        for item in data.get("items", [])[:max_results]:
            title = item.get("title", "")
            snippet = item.get("snippet", "")
            url_link = item.get("link", "")
            source = item.get("displayLink", "Google Search")
            
            # Filter fiction
            if is_fiction_content(snippet, title):
                if DEBUG:
                    print(f" ✗ Filtered (fiction): {title[:50]}...")
                continue
            
            # Check if Nepal source
            is_nepal = is_nepal_source(url_link)
            if is_nepal:
                source = f"🇳🇵 {source}"
            
            articles.append({
                "source": source,
                "title": title,
                "url": url_link,
                "snippet": snippet,
                "is_trusted": is_trusted_source(url_link),
                "is_nepal": is_nepal
            })
            
            if DEBUG and is_nepal:
                print(f" 🇳🇵 ✓ Nepal source via CSE: {source}")
        
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
                    "is_trusted": True,
                    "is_nepal": False
                })
        
        return results
        
    except Exception as e:
        if DEBUG:
            print(f"Wikipedia error: {e}")
        return []

# ------------------ IMPROVED Stance Detection ------------------
def analyze_evidence(claim: str, snippet: str, source: str, is_trusted: bool, is_nepal: bool = False) -> dict:
    """Analyze evidence with Nepal source boost and BALANCED matching"""
    text_lower = snippet.lower()
    claim_lower = claim.lower()
    
    # Count different types of terms
    strong_negative = sum(1 for term in STRONG_NEGATIVE_TERMS if term in text_lower)
    moderate_negative = sum(1 for term in MODERATE_NEGATIVE_TERMS if term in text_lower)
    positive = sum(1 for term in POSITIVE_TERMS if term in text_lower)
    
    # BALANCED RELEVANCE CALCULATION
    # Remove only common stopwords, keep important ones
    stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                 'of', 'with', 'by', 'from', 'this', 'that', 'these', 'those'}
    
    # Get all words first (3+ chars to avoid junk)
    claim_words_all = set(re.findall(r'\b\w{3,}\b', claim_lower))
    snippet_words_all = set(re.findall(r'\b\w{3,}\b', text_lower))
    
    # Remove stopwords
    claim_words = claim_words_all - stopwords
    snippet_words = snippet_words_all - stopwords
    
    if not claim_words or len(claim_words) < 2:
        # Claim too short or no meaningful words
        return {
            "stance": "inconclusive",
            "score": 0.0,
            "relevance": 0.0,
            "strong_negative": 0,
            "positive": 0
        }
    
    # Calculate common meaningful words
    common_words = claim_words & snippet_words
    
    # BALANCED: Need at least 2 matching words OR 30% overlap (more lenient than before)
    min_matches = min(2, max(1, int(len(claim_words) * 0.3)))
    
    if len(common_words) < min_matches:
        # Not relevant enough - mark as inconclusive with low score
        return {
            "stance": "inconclusive",
            "score": 0.05,
            "relevance": 0.0,
            "strong_negative": strong_negative,
            "positive": positive
        }
    
    # Calculate relevance based on meaningful word overlap
    relevance = len(common_words) / len(claim_words)
    
    # BOOST relevance if we have many matches
    if len(common_words) >= 3:
        relevance = min(1.0, relevance * 1.2)  # 20% boost for 3+ word matches
    
    # Check for named entities (optional boost, not requirement)
    claim_entities = set(re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', claim))
    snippet_entities = set(re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', snippet))
    
    entity_match_count = len(claim_entities & snippet_entities) if claim_entities else 0
    
    # Boost relevance if entities match
    if entity_match_count > 0:
        relevance = min(1.0, relevance + 0.1 * entity_match_count)
    
    # Trust multiplier
    trust_multiplier = 1.0
    if is_trusted:
        trust_multiplier = 1.2
    if is_nepal:
        trust_multiplier = 1.4
    
    # Determine stance and score - BALANCED THRESHOLDS
    base_score = relevance * 0.4
    
    # REFUTED (Fake) - Strong negative signals
    if strong_negative >= 3:
        stance = "refuted"
        score = (0.85 + (strong_negative * 0.05)) * trust_multiplier
    elif strong_negative >= 2 and relevance > 0.4:
        stance = "refuted"
        score = (0.7 + (strong_negative * 0.05)) * trust_multiplier
    elif strong_negative >= 1 and relevance > 0.5:
        stance = "refuted"
        score = (0.55 + (strong_negative * 0.1)) * trust_multiplier
    elif moderate_negative >= 2 and relevance > 0.4:
        stance = "refuted"
        score = 0.5 * trust_multiplier
    
    # SUPPORTED (Accurate) - Positive signals with good relevance
    elif positive >= 2 and strong_negative == 0 and relevance > 0.4:
        stance = "supported"
        score = (0.65 + (positive * 0.05)) * trust_multiplier
    
    # Trusted source with decent relevance and no negatives
    elif is_trusted and strong_negative == 0 and relevance > 0.45:
        stance = "supported"
        score = (0.5 + relevance * 0.4) * trust_multiplier
    
    # High relevance with some positive indicators
    elif relevance > 0.6 and positive >= 1 and strong_negative == 0:
        stance = "supported"
        score = (0.45 + relevance * 0.3) * trust_multiplier
    
    # Very high relevance alone can indicate support
    elif relevance > 0.7 and strong_negative == 0 and moderate_negative == 0:
        stance = "supported"
        score = (0.4 + relevance * 0.25) * trust_multiplier
    
    # INCONCLUSIVE - Default for unclear cases
    else:
        stance = "inconclusive"
        score = base_score
    
    if DEBUG:
        print(f" Analysis: stance={stance}, score={score:.3f}, relevance={relevance:.3f}")
        print(f" Common words ({len(common_words)}): {', '.join(list(common_words)[:5])}...")
        print(f" Strong neg={strong_negative}, Mod neg={moderate_negative}, Pos={positive}")
        if is_nepal:
            print(f" 🇳🇵 Nepal source boost applied (x{trust_multiplier})")
    
    return {
        "stance": stance,
        "score": min(1.0, score),
        "relevance": relevance,
        "strong_negative": strong_negative,
        "positive": positive
    }

# ------------------ Build Queries WITH NEPAL DETECTION ------------------
def build_queries(title: str, text: str) -> List[str]:
    """Build search queries with Nepal-specific additions"""
    queries = []
    
    # Check if Nepal-related
    nepal_keywords = ["nepal", "kathmandu", "pokhara", "nepali", "nepalese", 
                     "himalayan", "everest", "नेपाल"]
    is_nepal_related = any(kw in title.lower() for kw in nepal_keywords)
    
    # Main query
    if title:
        queries.append(title.strip())
    
    # Add fact-check queries
    if title:
        queries.append(f"{title.strip()} fact check")
        
        # Add Nepal-specific query if relevant
        if is_nepal_related:
            queries.append(f"{title.strip()} Nepal news")
    
    return queries[:3]

# ------------------ IMPROVED Verdict Computation ------------------
def compute_verdict(evidence: List[dict]) -> Tuple[str, float, List[dict]]:
    """Compute final verdict with BALANCED thresholds"""
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
    inconclusive_count = sum(1 for e in top if e.get("stance") == "inconclusive")
    
    # Count Nepal sources
    nepal_count = sum(1 for e in top if e.get("is_nepal", False))
    
    # Calculate average relevance for each side
    support_relevance = [e.get("relevance", 0.0) for e in top if e.get("stance") == "supported"]
    refute_relevance = [e.get("relevance", 0.0) for e in top if e.get("stance") == "refuted"]
    
    avg_support_rel = sum(support_relevance) / len(support_relevance) if support_relevance else 0
    avg_refute_rel = sum(refute_relevance) / len(refute_relevance) if refute_relevance else 0
    
    if DEBUG:
        print(f"\n Verdict Calculation:")
        print(f" Support score: {support_score:.3f} (count: {support_count}, avg_rel: {avg_support_rel:.3f})")
        print(f" Refute score: {refute_score:.3f} (count: {refute_count}, avg_rel: {avg_refute_rel:.3f})")
        print(f" Inconclusive count: {inconclusive_count}")
        print(f" Total score: {total_score:.3f}")
        if nepal_count > 0:
            print(f" 🇳🇵 Nepal sources: {nepal_count}/{len(top)}")
    
    # BALANCED threshold for inconclusive
    if total_score < 0.4:
        return "inconclusive", 0.5, top
    
    # Calculate ratios
    if total_score > 0:
        refute_ratio = refute_score / total_score
        support_ratio = support_score / total_score
    else:
        refute_ratio = 0
        support_ratio = 0
    
    # BALANCED VERDICT LOGIC
    
    # REFUTED (Fake) - Strong evidence of fake news
    if refute_score > 1.2:  # Very strong refutation
        verdict = "refuted"
        confidence = min(0.92, 0.75 + refute_score * 0.12)
    
    elif refute_score > 0.7:  # Strong refutation
        verdict = "refuted"
        confidence = min(0.85, 0.65 + refute_score * 0.15)
    
    elif refute_ratio > 0.6 and refute_count >= 2:  # Clear majority refuting
        verdict = "refuted"
        confidence = min(0.80, 0.60 + refute_ratio * 0.2)
    
    elif refute_score > 0.4 and support_score < 0.5:  # Moderate refutation, weak support
        verdict = "refuted"
        confidence = 0.65 + refute_score * 0.1
    
    # SUPPORTED (Accurate) - Strong evidence of accuracy
    elif support_score > 1.5 and refute_score < 0.3:  # Very strong support, minimal refutation
        verdict = "supported"
        confidence = min(0.92, 0.70 + support_score * 0.1)
    
    elif support_score > 1.0 and refute_score < 0.4:  # Strong support, weak refutation
        verdict = "supported"
        confidence = min(0.85, 0.65 + support_score * 0.12)
    
    elif support_ratio > 0.7 and support_count >= 3:  # Clear majority supporting
        verdict = "supported"
        confidence = min(0.80, 0.60 + support_ratio * 0.18)
    
    elif support_score > 0.6 and refute_score < 0.2:  # Decent support, very weak refutation
        verdict = "supported"
        confidence = 0.65 + support_score * 0.1
    
    # MIXED or BALANCED - Close scores or conflicting evidence
    elif abs(support_score - refute_score) < 0.3:
        # Too close to call
        verdict = "inconclusive"
        confidence = 0.5
    
    elif support_count > 0 and refute_count > 0:
        # Both sides have evidence - go with stronger side but lower confidence
        if support_score > refute_score * 1.5:  # Support significantly stronger
            verdict = "supported"
            confidence = 0.60
        elif refute_score > support_score * 1.5:  # Refute significantly stronger
            verdict = "refuted"
            confidence = 0.60
        else:
            verdict = "inconclusive"  # Too mixed
            confidence = 0.5
    
    # DEFAULT: Go with stronger side
    elif support_score > refute_score:
        verdict = "supported"
        confidence = 0.58 + (support_ratio * 0.15)
    else:
        verdict = "refuted"
        confidence = 0.58 + (refute_ratio * 0.15)
    
    # QUALITY CHECK: Downgrade if average relevance is too low
    if verdict == "supported" and avg_support_rel < 0.4:
        verdict = "inconclusive"
        confidence = 0.5
        if DEBUG:
            print(f" ⚠ Downgraded to inconclusive: low support relevance ({avg_support_rel:.3f})")
    
    if verdict == "refuted" and avg_refute_rel < 0.4:
        verdict = "inconclusive"
        confidence = 0.5
        if DEBUG:
            print(f" ⚠ Downgraded to inconclusive: low refute relevance ({avg_refute_rel:.3f})")
    
    if DEBUG:
        print(f" Final: {verdict} (confidence: {confidence:.3f})")
    
    return verdict, float(confidence), top

# ------------------ MAIN VERIFICATION WITH NEPAL SUPPORT ------------------
def verify_article(title: str, text: str, max_pages_per_query: int = 8, debug: bool = False):
    """Main verification with Nepal sources integration"""
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
    
    # Detect if Nepal-related
    nepal_keywords = ["nepal", "kathmandu", "pokhara", "nepali", "nepalese", 
                     "himalayan", "everest", "नेपाल"]
    is_nepal_related = any(kw in claim.lower() for kw in nepal_keywords)
    
    if DEBUG:
        print("=" * 70)
        print("FAKE NEWS DETECTOR - WITH NEPAL SOURCES")
        print("=" * 70)
        print(f"Claim: {claim}")
        if is_nepal_related:
            print("🇳🇵 NEPAL-RELATED CLAIM DETECTED")
        print("=" * 70)
    
    # Build queries
    queries = build_queries(claim, text)
    all_evidence = []
    seen_urls = set()
    
    for q_idx, q in enumerate(queries, 1):
        if DEBUG:
            print(f"\n[Query {q_idx}/{len(queries)}]: {q}")
        
        # 1. NEPAL-SPECIFIC: Search GNews for Nepal sources
        if is_nepal_related:
            nepal_results = gnews_search_nepal(q, max_results=5)
            for result in nepal_results:
                url = result.get("url", "")
                if url in seen_urls:
                    continue
                seen_urls.add(url)
                
                analysis = analyze_evidence(
                    claim=claim,
                    snippet=result["snippet"],
                    source=result["source"],
                    is_trusted=True,
                    is_nepal=True
                )
                
                all_evidence.append({
                    **result,
                    **analysis
                })
        
        # 2. Wikipedia Search
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
                is_trusted=True,
                is_nepal=False
            )
            
            all_evidence.append({
                **result,
                **analysis
            })
            
            if DEBUG:
                print(f" ✓ Wiki: {result['title'][:50]}...")
        
        # 3. Google CSE Search (with Nepal bias if relevant)
        cse_results = google_cse_search(
            q, 
            max_results=max_pages_per_query,
            prefer_nepal=is_nepal_related
        )
        for result in cse_results:
            url = result.get("url", "")
            if url in seen_urls:
                continue
            seen_urls.add(url)
            
            analysis = analyze_evidence(
                claim=claim,
                snippet=result["snippet"],
                source=result["source"],
                is_trusted=result.get("is_trusted", False),
                is_nepal=result.get("is_nepal", False)
            )
            
            all_evidence.append({
                **result,
                **analysis
            })
            
            if DEBUG:
                print(f" ✓ CSE: {result['title'][:50]}...")
    
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
            "snippet": e.get("snippet", "")[:200],
            "stance": e.get("stance", "inconclusive"),
            "score": round(e.get("score", 0), 3),
            "relevance": round(e.get("relevance", 0), 3),
            "is_nepal": e.get("is_nepal", False)
        })
    
    result = {
        "verdict": verdict,
        "confidence": confidence,
        "evidence": formatted_evidence,
        "debug": {
            "queries": queries,
            "total_evidence": len(all_evidence),
            "top_evidence_count": len(formatted_evidence),
            "nepal_sources": sum(1 for e in formatted_evidence if e.get("is_nepal", False))
        }
    }
    
    if DEBUG:
        print("\n" + "=" * 70)
        print("FINAL RESULT")
        print("=" * 70)
        print(f"Verdict: {verdict.upper()}")
        print(f"Confidence: {confidence:.1%}")
        print(f"Total evidence: {len(all_evidence)}")
        if result["debug"]["nepal_sources"] > 0:
            print(f"🇳🇵 Nepal sources: {result['debug']['nepal_sources']}")
        print(f"\nTop 5 Evidence:")
        for i, e in enumerate(formatted_evidence[:5], 1):
            nepal_flag = "🇳🇵 " if e.get("is_nepal") else ""
            print(f" {i}. {nepal_flag}[{e['stance'].upper()}] {e['source']}")
            print(f"    Score: {e['score']:.3f} | {e['title'][:60]}...")
        print("=" * 70)
    
    return result


# ------------------ Main ------------------
if __name__ == "__main__":
    # Test with Nepal claim
    test_claim = "Nepal earthquake 2015 killed thousands"
    print("\n" + "=" * 70)
    print("TESTING WITH NEPAL CLAIM")
    print("=" * 70)
    
    result = verify_article(test_claim, "", debug=True)
    
    print(f"\n\nFinal Result:")
    print(f"Verdict: {result['verdict']}")
    print(f"Confidence: {result['confidence']:.1%}")
    print(f"Nepal sources found: {result['debug']['nepal_sources']}")
