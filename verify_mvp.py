# verify_article1.py - DECISIVE VERSION with Nepal Sources Support
import os
import re
import requests
import datetime
from urllib.parse import quote
from typing import List, Tuple, Dict

# ------------------ Config ------------------
GNEWS_API_KEY = os.getenv("GNEWS_API_KEY", "7c5375c69233f6a731714ab09584a219")
GOOGLE_CSE_API_KEY = os.getenv("GOOGLE_CSE_API_KEY", "AIzaSyBypIQi0kJ4JsVwAdM15-ELWr7kmNVPyMA")
GOOGLE_CSE_ENGINE_ID = os.getenv("GOOGLE_CSE_ENGINE_ID", "04ac3616a037847f6")

DEBUG = False

# ------------------ NEPAL-SPECIFIC SOURCES ------------------
NEPAL_NEWS_SOURCES = {
    "kathmandupost.com": {"name": "Kathmandu Post", "trust": 0.95},
    "thehimalayantimes.com": {"name": "Himalayan Times", "trust": 0.95},
    "myrepublica.com": {"name": "My Republica", "trust": 0.90},
    "ekantipur.com": {"name": "Ekantipur", "trust": 0.90},
    "setopati.com": {"name": "Setopati", "trust": 0.85},
    "ratopati.com": {"name": "Ratopati", "trust": 0.85},
    "onlinekhabar.com": {"name": "Online Khabar", "trust": 0.85},
    "gorkhapatra.org.np": {"name": "Gorkhapatra", "trust": 0.90},
    "nepalnews.com": {"name": "Nepal News", "trust": 0.80},
    "nepalitimes.com": {"name": "Nepali Times", "trust": 0.90},
}

NEPAL_KEYWORDS = [
    "nepal", "kathmandu", "pokhara", "nepali", "nepalese", "himalayan",
    "everest", "नेपाल", "काठमाडौं", "lalitpur", "bhaktapur", "biratnagar",
    "chitwan", "terai", "madhesh", "sherpa", "gurkha", "rupee", "npr"
]

# ------------------ ENHANCED Detection Terms ------------------
STRONG_NEGATIVE_TERMS = {
    "false", "fake", "hoax", "debunked", "incorrect", "misinformation",
    "disinformation", "fabricated", "no evidence", "refuted", "myth", 
    "disproved", "fact check", "fact-check", "misleading", "unproven", 
    "unverified", "baseless", "unfounded", "scam", "fraud", "untrue",
    "falsely claims", "no proof", "discredited", "debunk", "not true",
    "never happened", "did not happen", "झूटो", "गलत"  # Nepali: false, wrong
}

MODERATE_NEGATIVE_TERMS = {
    "conspiracy theory", "pseudoscience", "rumor", "unconfirmed",
    "alleged", "claims without", "not supported", "questioned", 
    "disputed", "controversial", "doubtful"
}

POSITIVE_TERMS = {
    "confirmed", "verified", "proven", "established", "validated",
    "documented", "evidence shows", "studies show", "research confirms",
    "scientists confirm", "experts confirm", "officially", "peer-reviewed",
    "announced", "reported by", "according to", "statement from",
    "official statement", "press release", "government confirms"
}

# Context modifiers that strengthen claims
CONTEXT_BOOSTERS = {
    "government", "ministry", "official", "spokesperson", "announced",
    "parliament", "supreme court", "police", "investigation", "arrest",
    "सरकार", "मन्त्रालय"  # Nepali: government, ministry
}

FICTION_TERMS = {
    "video game", "films", "movie", "television", "novel", "series", "fiction",
    "sci-fi", "fantasy", "superhero", "marvel", "dc comics", "anime", "manga",
    "netflix", "streaming", "blockbuster", "box office", "trailer", "character"
}

TRUSTED_DOMAINS = {
    "wikipedia.org", "britannica.com", "reuters.com", "apnews.com",
    "bbc.com", "nature.com", "science.org", "nih.gov", "cdc.gov",
    "who.int", "nasa.gov", ".edu", ".gov", "un.org", "worldbank.org"
}

# ------------------ Helper Functions ------------------
def is_nepal_related(text: str) -> bool:
    """Check if content is Nepal-related"""
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in NEPAL_KEYWORDS)

def get_nepal_source_info(url: str) -> Dict:
    """Get Nepal source information from URL"""
    url_lower = url.lower()
    for domain, info in NEPAL_NEWS_SOURCES.items():
        if domain in url_lower:
            return {"is_nepal_source": True, **info}
    return {"is_nepal_source": False, "name": None, "trust": 0.0}

def is_fiction_content(text: str, title: str = "") -> bool:
    """Check if content is fiction/entertainment"""
    combined = f"{title} {text}".lower()
    fiction_count = sum(1 for term in FICTION_TERMS if term in combined)
    return fiction_count >= 2

def is_trusted_source(url: str) -> bool:
    """Check if URL is from a trusted domain"""
    url_lower = url.lower()
    # Check Nepal sources first
    nepal_info = get_nepal_source_info(url)
    if nepal_info["is_nepal_source"]:
        return True
    # Check other trusted domains
    return any(domain in url_lower for domain in TRUSTED_DOMAINS)

def extract_key_entities(text: str) -> set:
    """Extract key entities from text (simple implementation)"""
    # Remove common words
    stopwords = {"the", "a", "an", "is", "are", "was", "were", "in", "on", "at", 
                 "to", "for", "of", "with", "by", "from", "about", "has", "have"}
    words = re.findall(r'\b[A-Z][a-z]+\b|\b\w{4,}\b', text)
    return {w.lower() for w in words if w.lower() not in stopwords}

# ------------------ Google CSE Search with Nepal Priority ------------------
def google_cse_search(query: str, max_results: int = 10, prefer_nepal: bool = False):
    """Search using Google Custom Search Engine with Nepal source priority"""
    if not GOOGLE_CSE_API_KEY or not GOOGLE_CSE_ENGINE_ID:
        if DEBUG:
            print("⚠ Google CSE API key or Engine ID missing")
        return []
    
    url = "https://customsearch.googleapis.com/customsearch/v1"
    
    # If Nepal-related, add Nepal sources to query
    search_query = query
    if prefer_nepal:
        nepal_domains = " OR ".join([f"site:{domain}" for domain in list(NEPAL_NEWS_SOURCES.keys())[:3]])
        search_query = f"{query} ({nepal_domains})"
        if DEBUG:
            print(f"  🇳🇵 Nepal-prioritized query: {search_query[:100]}...")
    
    params = {
        "key": GOOGLE_CSE_API_KEY,
        "cx": GOOGLE_CSE_ENGINE_ID,
        "q": search_query,
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
            
            # Check Nepal source
            nepal_info = get_nepal_source_info(url_link)
            
            articles.append({
                "source": nepal_info.get("name") or item.get("displayLink", "Google Search"),
                "title": title,
                "url": url_link,
                "snippet": snippet,
                "is_trusted": is_trusted_source(url_link),
                "is_nepal_source": nepal_info["is_nepal_source"],
                "trust_score": nepal_info.get("trust", 0.5 if is_trusted_source(url_link) else 0.3)
            })
        
        # Sort: Nepal sources first, then by trust
        articles.sort(key=lambda x: (x["is_nepal_source"], x["trust_score"]), reverse=True)
        
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
            summary_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{quote(title)}"
            summary_resp = requests.get(summary_url, headers=headers, timeout=10)
            
            if summary_resp.status_code == 200:
                summary_data = summary_resp.json()
                snippet = summary_data.get("extract", "")
                
                if is_fiction_content(snippet, title):
                    continue
                
                results.append({
                    "source": "Wikipedia",
                    "title": title,
                    "url": summary_data.get("content_urls", {}).get("desktop", {}).get("page", ""),
                    "snippet": snippet,
                    "is_trusted": True,
                    "is_nepal_source": False,
                    "trust_score": 0.85
                })
        
        return results
    except Exception as e:
        if DEBUG:
            print(f"Wikipedia error: {e}")
        return []

# ------------------ IMPROVED Stance Detection ------------------
def analyze_evidence(claim: str, snippet: str, source: str, trust_score: float, 
                     is_nepal_source: bool, is_nepal_claim: bool) -> dict:
    """
    Analyze evidence with improved scoring to reduce inconclusive results
    """
    text_lower = snippet.lower()
    claim_lower = claim.lower()
    
    # Count different types of terms
    strong_negative = sum(1 for term in STRONG_NEGATIVE_TERMS if term in text_lower)
    moderate_negative = sum(1 for term in MODERATE_NEGATIVE_TERMS if term in text_lower)
    positive = sum(1 for term in POSITIVE_TERMS if term in text_lower)
    context_boost = sum(1 for term in CONTEXT_BOOSTERS if term in text_lower)
    
    # Calculate relevance using entity matching
    claim_entities = extract_key_entities(claim)
    snippet_entities = extract_key_entities(snippet)
    common_entities = claim_entities & snippet_entities
    
    # Enhanced relevance calculation
    if len(claim_entities) > 0:
        entity_match = len(common_entities) / len(claim_entities)
    else:
        entity_match = 0.0
    
    # Word-level relevance
    claim_words = set(re.findall(r'\b\w{3,}\b', claim_lower))
    snippet_words = set(re.findall(r'\b\w{3,}\b', text_lower))
    common_words = claim_words & snippet_words
    word_match = len(common_words) / max(len(claim_words), 1)
    
    relevance = max(entity_match, word_match * 0.8)
    
    # NEPAL BOOST: If it's a Nepal claim from a Nepal source, boost relevance
    if is_nepal_claim and is_nepal_source:
        relevance = min(1.0, relevance * 1.3)
        if DEBUG:
            print(f"      🇳🇵 Nepal source boost applied: {relevance:.3f}")
    
    # Trust multiplier
    trust_multiplier = 0.8 + (trust_score * 0.5)  # Range: 0.8 to 1.3
    
    # Determine stance and score with LOWER thresholds to reduce inconclusive
    base_score = relevance * 0.4
    
    # REFUTED (Strong) - More aggressive detection
    if strong_negative >= 2:
        stance = "refuted"
        score = (0.8 + (strong_negative * 0.05)) * trust_multiplier
        
    # REFUTED (Moderate)
    elif strong_negative >= 1 and relevance > 0.25:  # LOWERED from 0.3
        stance = "refuted"
        score = (0.65 + (strong_negative * 0.1)) * trust_multiplier
        
    # REFUTED (Weak but counts)
    elif strong_negative >= 1 or moderate_negative >= 2:
        stance = "refuted"
        score = (0.5 + (strong_negative * 0.1)) * trust_multiplier
        
    # SUPPORTED (Strong)
    elif positive >= 2 and strong_negative == 0 and relevance > 0.3:
        stance = "supported"
        score = (0.75 + (positive * 0.03)) * trust_multiplier
        
    # SUPPORTED (Moderate) - More lenient
    elif positive >= 1 and strong_negative == 0 and relevance > 0.35:  # LOWERED from 0.4
        stance = "supported"
        score = (0.6 + (positive * 0.05) + (context_boost * 0.05)) * trust_multiplier
        
    # SUPPORTED (Weak) - NEW: Accept weaker signals
    elif trust_score > 0.7 and strong_negative == 0 and relevance > 0.3:
        stance = "supported"
        score = (0.45 + relevance) * trust_multiplier
        
    # SUPPORTED (Context-based) - NEW: For official announcements
    elif context_boost >= 2 and strong_negative == 0 and relevance > 0.25:
        stance = "supported"
        score = (0.5 + (context_boost * 0.08)) * trust_multiplier
        
    # Still inconclusive BUT with some lean
    elif relevance > 0.4:
        # Lean toward support if no negatives and decent relevance
        if strong_negative == 0 and moderate_negative == 0:
            stance = "supported"
            score = (0.35 + relevance * 0.3) * trust_multiplier
        else:
            stance = "inconclusive"
            score = base_score
    else:
        stance = "inconclusive"
        score = base_score
    
    if DEBUG:
        print(f"    Analysis: stance={stance}, score={score:.3f}, relevance={relevance:.3f}")
        print(f"      Strong neg={strong_negative}, Mod neg={moderate_negative}, Pos={positive}, Context={context_boost}")
        print(f"      Trust={trust_score:.2f}, Nepal={is_nepal_source}")
    
    return {
        "stance": stance,
        "score": min(1.0, score),
        "relevance": relevance,
        "strong_negative": strong_negative,
        "positive": positive,
        "context_boost": context_boost,
        "trust_score": trust_score
    }

# ------------------ Build Queries ------------------
def build_queries(title: str, text: str, is_nepal_related_claim: bool) -> List[str]:
    """Build search queries from claim with Nepal optimization"""
    queries = []
    
    # Main query
    if title:
        queries.append(title.strip())
    
    # Fact-check queries
    if title:
        queries.append(f"{title.strip()} fact check")
        
        # Nepal-specific query variations
        if is_nepal_related_claim:
            queries.append(f"{title.strip()} Nepal news")
            queries.append(f"{title.strip()} Kathmandu")
    
    return queries[:4]  # Max 4 queries

# ------------------ IMPROVED Verdict Computation ------------------
def compute_verdict(evidence: List[dict], is_nepal_claim: bool) -> Tuple[str, float, List[dict]]:
    """
    Compute final verdict with REDUCED inconclusive results
    """
    if not evidence:
        return "inconclusive", 0.3, []
    
    # Sort by score
    evidence_sorted = sorted(
        evidence,
        key=lambda e: e.get("score", 0.0),
        reverse=True
    )
    
    # Consider top 12 pieces (increased from 10)
    top = evidence_sorted[:12]
    
    # Calculate totals with weighted scoring
    support_score = 0.0
    refute_score = 0.0
    inconclusive_score = 0.0
    
    support_count = 0
    refute_count = 0
    nepal_support = 0
    nepal_refute = 0
    
    for e in top:
        stance = e.get("stance", "inconclusive")
        score = e.get("score", 0.0)
        is_nepal = e.get("trust_score", 0.5) > 0.8  # High trust = likely Nepal source
        
        if stance == "supported":
            support_score += score
            support_count += 1
            if is_nepal:
                nepal_support += 1
        elif stance == "refuted":
            refute_score += score
            refute_count += 1
            if is_nepal:
                nepal_refute += 1
        else:
            inconclusive_score += score * 0.3  # Reduce impact of inconclusive
    
    total_score = support_score + refute_score + inconclusive_score
    
    if DEBUG:
        print(f"\n  Verdict Calculation:")
        print(f"    Support: score={support_score:.3f}, count={support_count}")
        print(f"    Refute: score={refute_score:.3f}, count={refute_count}")
        print(f"    Inconclusive: score={inconclusive_score:.3f}")
        print(f"    Total score: {total_score:.3f}")
        if is_nepal_claim:
            print(f"    🇳🇵 Nepal support={nepal_support}, refute={nepal_refute}")
    
    # VERY LOW threshold - make a decision with minimal evidence
    if total_score < 0.3:
        # Even with low evidence, try to make a call
        if support_count > refute_count:
            return "supported", 0.45, top
        elif refute_count > support_count:
            return "refuted", 0.45, top
        else:
            return "inconclusive", 0.35, top
    
    # Calculate ratios
    if total_score > 0:
        refute_ratio = refute_score / total_score
        support_ratio = support_score / total_score
    else:
        refute_ratio = 0
        support_ratio = 0
    
    # DECISIVE LOGIC - Lower thresholds
    
    # REFUTED (Strong)
    if refute_score > 0.6:  # LOWERED from 0.8
        verdict = "refuted"
        confidence = min(0.92, 0.65 + refute_score * 0.2)
        
    # REFUTED (Moderate)
    elif refute_ratio > 0.35 and refute_score > 0.25:  # LOWERED from 0.4 and 0.3
        verdict = "refuted"
        confidence = min(0.85, 0.55 + refute_ratio * 0.25)
        
    # REFUTED (Weak but decisive)
    elif refute_count >= 2 and refute_score > support_score:
        verdict = "refuted"
        confidence = 0.6
        
    # SUPPORTED (Strong)
    elif support_score > 0.9:  # LOWERED from 1.2
        verdict = "supported"
        confidence = min(0.92, 0.65 + support_score * 0.15)
        
    # SUPPORTED (Moderate)
    elif support_ratio > 0.5 and support_score > 0.4:  # LOWERED from 0.6 and 0.5
        verdict = "supported"
        confidence = min(0.85, 0.55 + support_ratio * 0.25)
        
    # SUPPORTED (Weak but decisive)
    elif support_count >= 2 and support_score > refute_score:
        verdict = "supported"
        confidence = 0.6
        
    # NEPAL BOOST - If Nepal claim with Nepal sources, be more decisive
    elif is_nepal_claim and (nepal_support > 0 or nepal_refute > 0):
        if nepal_support > nepal_refute:
            verdict = "supported"
            confidence = 0.65
        elif nepal_refute > nepal_support:
            verdict = "refuted"
            confidence = 0.65
        elif support_score > refute_score:
            verdict = "supported"
            confidence = 0.55
        else:
            verdict = "refuted"
            confidence = 0.55
        if DEBUG:
            print(f"    🇳🇵 Nepal boost applied: {verdict}")
    
    # MIXED SIGNALS - Still make a call
    elif abs(support_score - refute_score) < 0.2:
        # Even with mixed signals, lean toward the stronger side
        if support_count > refute_count:
            verdict = "supported"
            confidence = 0.52
        elif refute_count > support_count:
            verdict = "refuted"
            confidence = 0.52
        else:
            verdict = "inconclusive"
            confidence = 0.45
    
    # DEFAULT: Go with the stronger side with lower confidence
    elif support_score > refute_score:
        verdict = "supported"
        confidence = 0.55 + (support_ratio * 0.15)
    else:
        verdict = "refuted"
        confidence = 0.55 + (refute_ratio * 0.15)
    
    if DEBUG:
        print(f"    Final: {verdict} (confidence: {confidence:.3f})")
    
    return verdict, float(confidence), top

# ------------------ MAIN VERIFICATION ------------------
def verify_article(title: str, text: str, max_pages_per_query: int = 8, debug: bool = False):
    """
    Main verification function with Nepal support and reduced inconclusive results
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
    
    # Check if Nepal-related
    is_nepal_claim = is_nepal_related(claim)
    
    if DEBUG:
        print("=" * 70)
        print("FAKE NEWS DETECTOR - DECISIVE VERSION WITH NEPAL SOURCES")
        print("=" * 70)
        print(f"Claim: {claim}")
        if is_nepal_claim:
            print("🇳🇵 NEPAL-RELATED CLAIM DETECTED")
        print("=" * 70)
    
    # Build queries
    queries = build_queries(claim, text, is_nepal_claim)
    
    all_evidence = []
    seen_urls = set()
    
    for q_idx, q in enumerate(queries, 1):
        if DEBUG:
            print(f"\n[Query {q_idx}/{len(queries)}]: {q}")
        
        # 1. Wikipedia Search (skip for Nepal-specific queries)
        if q_idx <= 2:  # Only first 2 queries
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
                    trust_score=result["trust_score"],
                    is_nepal_source=False,
                    is_nepal_claim=is_nepal_claim
                )
                
                all_evidence.append({**result, **analysis})
                
                if DEBUG:
                    print(f"  ✓ Wiki: {result['title'][:50]}... [{analysis['stance']}]")
        
        # 2. Google CSE Search (with Nepal priority)
        cse_results = google_cse_search(q, max_results=max_pages_per_query, 
                                        prefer_nepal=is_nepal_claim)
        for result in cse_results:
            url = result.get("url", "")
            if url in seen_urls:
                continue
            seen_urls.add(url)
            
            analysis = analyze_evidence(
                claim=claim,
                snippet=result["snippet"],
                source=result["source"],
                trust_score=result.get("trust_score", 0.5),
                is_nepal_source=result.get("is_nepal_source", False),
                is_nepal_claim=is_nepal_claim
            )
            
            all_evidence.append({**result, **analysis})
            
            nepal_flag = "🇳🇵 " if result.get("is_nepal_source") else ""
            if DEBUG:
                print(f"  ✓ {nepal_flag}CSE: {result['title'][:50]}... [{analysis['stance']}]")
    
    # Compute verdict
    if not all_evidence:
        if DEBUG:
            print("\n❌ No evidence found!")
        return {
            "verdict": "supported",  # DEFAULT to supported instead of inconclusive
            "confidence": 0.4,
            "evidence": [],
            "debug": {
                "queries": queries,
                "reason": "No evidence found - defaulting to supported"
            }
        }
    
    verdict, confidence, top_evidence = compute_verdict(all_evidence, is_nepal_claim)
    
    # Format evidence for output
    formatted_evidence = []
    for e in top_evidence:
        formatted_evidence.append({
            "source": e.get("source", "Unknown"),
            "title": e.get("title", ""),
            "url": e.get("url", ""),
            "snippet": e.get("snippet", "")[:250],
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
            "top_evidence_count": len(formatted_evidence),
            "is_nepal_claim": is_nepal_claim
        }
    }
    
    if DEBUG:
        print("\n" + "=" * 70)
        print("FINAL RESULT")
        print("=" * 70)
        print(f"Verdict: {verdict.upper()}")
        print(f"Confidence: {confidence:.1%}")
        print(f"Total evidence: {len(all_evidence)}")
        if is_nepal_claim:
            nepal_sources_count = sum(1 for e in formatted_evidence 
                                     if any(ns in e['source'].lower() 
                                           for ns in ["kathmandu", "himalayan", "nepal", "republica"]))
            print(f"🇳🇵 Nepal sources used: {nepal_sources_count}")
        print(f"\nTop 5 Evidence:")
        for i, e in enumerate(formatted_evidence[:5], 1):
            print(f"  {i}. [{e['stance'].upper()}] {e['source']}")
            print(f"     Score: {e['score']:.3f} | {e['title'][:60]}...")
        print("=" * 70)
    
    return result

# ------------------ Main ------------------
if __name__ == "__main__":
    # Test with Nepal-related and general claims
    test_claims = [
        ("Nepal's GDP growth rate increased last year", True),
        ("Kathmandu Post reports new education policy", True),
        ("Mount Everest is in Nepal", True),
        ("Vaccines cause autism", False),
        ("COVID-19 vaccines contain microchips", False),
    ]
    
    print("\n" + "=" * 70)
    print("TESTING VERIFICATION SYSTEM - NEPAL ENHANCED")
    print("=" * 70)
    
    for claim, expected_true in test_claims:
        is_nepal = is_nepal_related(claim)
        print(f"\n\nTesting: {claim}")
        print(f"Expected: {'ACCURATE' if expected_true else 'FAKE'}")
        if is_nepal:
            print("🇳🇵 Nepal-related claim")
        print("-" * 70)
        
        result = verify_article(claim, "", debug=False)
        
        verdict = result["verdict"]
        confidence = result["confidence"]
        
        print(f"Result: {verdict.upper()} ({confidence:.1%} confidence)")
        
        is_correct = (expected_true and verdict == "supported") or \
                     (not expected_true and verdict == "refuted") or \
                     (verdict == "inconclusive")
        
        print(f"Status: {'✓ CORRECT' if is_correct else '✗ INCORRECT'}")
