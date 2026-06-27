from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import sqlite3
import json
import httpx
import random
import base64
import urllib.parse
from datetime import datetime

try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False

app = FastAPI(title="Stock Topic Finder API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DB_PATH = "stock_topics.db"

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY,
            value TEXT
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS saved_topics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            topic TEXT NOT NULL,
            type TEXT,
            demand TEXT,
            competition TEXT,
            trend_percent REAL,
            opportunity_score INTEGER,
            keyword TEXT,
            saved_at TEXT DEFAULT (datetime('now'))
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS search_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            keyword TEXT NOT NULL,
            results_count INTEGER,
            searched_at TEXT DEFAULT (datetime('now'))
        )
    """)
    conn.commit()
    conn.close()

init_db()

# Models
class ApiKeyRequest(BaseModel):
    api_key: str

class SearchRequest(BaseModel):
    keyword: str
    topic_type: Optional[str] = "all"
    category: Optional[str] = "all"
    country: Optional[str] = "worldwide"
    time_range: Optional[str] = "past_24_hours"
    max_results: Optional[int] = 20
    exclude_topics: Optional[List[str]] = None

class SaveTopicRequest(BaseModel):
    topic: str
    type: str
    demand: str
    competition: str
    trend_percent: float
    opportunity_score: int
    keyword: str

class GenerateIdeasRequest(BaseModel):
    topic: str
    exclude: Optional[List[str]] = None
    per_category: Optional[int] = 6

class MarketAnalyzeRequest(BaseModel):
    market: str
    keyword: str
    niche_count: Optional[int] = 5
    keywords_per_niche: Optional[int] = 16

# Settings
@app.get("/api/settings")
def get_settings():
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT key, value FROM settings")
    rows = c.fetchall()
    conn.close()
    settings = {r["key"]: r["value"] for r in rows}
    api_key = settings.get("groq_api_key", "")
    masked = ""
    if api_key:
        masked = api_key[:4] + "*" * (len(api_key) - 4) if len(api_key) > 4 else "****"
    return {"has_api_key": bool(api_key), "masked_key": masked}

@app.post("/api/settings/api-key")
def save_api_key(req: ApiKeyRequest):
    conn = get_db()
    c = conn.cursor()
    c.execute("INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)", ("groq_api_key", req.api_key))
    conn.commit()
    conn.close()
    return {"success": True, "message": "API key saved successfully"}

@app.delete("/api/settings/api-key")
def delete_api_key():
    conn = get_db()
    c = conn.cursor()
    c.execute("DELETE FROM settings WHERE key = 'groq_api_key'")
    conn.commit()
    conn.close()
    return {"success": True}

# Helpers
def get_groq_api_key():
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT value FROM settings WHERE key = 'groq_api_key'")
    row = c.fetchone()
    conn.close()
    return row["value"] if row else None

def compute_opportunity_score(demand, trend_pct, competition):
    demand_w = {"High": 40, "Medium": 25, "Low": 10}.get(demand, 20)
    trend_w = min(max(int(trend_pct / 2), -15), 30)
    comp_w = {"Low": 0, "Medium": -10, "High": -25}.get(competition, -10)
    score = demand_w + trend_w + 40 + comp_w
    return max(0, min(100, score))

def clean_json_content(content):
    content = content.strip()
    if content.startswith("```"):
        content = content.split("```")[1]
        if content.startswith("json"):
            content = content[4:]
    return content.strip()

def score_topics(raw_topics):
    """Takes raw topic dicts from the AI (or mock generator) and attaches id + opportunity_score."""
    result = []
    for i, t in enumerate(raw_topics):
        trend_pct = float(t.get("trend_percent", random.uniform(-10, 30)))
        demand = t.get("demand", "Medium")
        competition = t.get("competition", "Medium")
        score = compute_opportunity_score(demand, trend_pct, competition)
        result.append({
            "id": i + 1,
            "topic": t["topic"],
            "type": t.get("type", "Sub Topic"),
            "demand": demand,
            "competition": competition,
            "trend_percent": round(trend_pct, 1),
            "opportunity_score": score,
        })
    return result

def build_exclusion_text(exclude):
    if not exclude:
        return ""
    sample = exclude[:40]
    return (
        "\nDo NOT repeat any of these already-suggested topics — generate fresh, different ones:\n- "
        + "\n- ".join(sample)
    )

TIME_RANGE_LABELS = {
    "past_hour": "past 1 hour",
    "past_4_hours": "past 4 hours",
    "past_24_hours": "past 24 hours",
    "past_week": "past 7 days",
    "past_month": "past 30 days",
    "past_3_months": "past 3 months",
    "past_12_months": "past 12 months",
    "past_5_years": "past 5 years",
    "2004_present": "2004 to present",
}

COUNTRY_LABELS = {
    "worldwide": "worldwide", "us": "United States", "gb": "United Kingdom",
    "bd": "Bangladesh", "in": "India", "de": "Germany", "fr": "France",
    "ca": "Canada", "au": "Australia", "br": "Brazil", "jp": "Japan",
    "kr": "South Korea", "id": "Indonesia", "pk": "Pakistan", "sa": "Saudi Arabia",
    "ae": "UAE", "ru": "Russia", "tr": "Turkey", "mx": "Mexico",
    "it": "Italy", "es": "Spain", "nl": "Netherlands", "pl": "Poland",
    "ng": "Nigeria", "za": "South Africa",
}

async def generate_topics_with_groq(keyword, api_key, count=20, exclude=None, time_range="past_24_hours", country="worldwide"):
    main_count = max(1, round(count * 0.25))
    sub_count = max(0, count - main_count)
    exclusion_text = build_exclusion_text(exclude)
    time_label = TIME_RANGE_LABELS.get(time_range, "past 24 hours")
    country_label = COUNTRY_LABELS.get(country, "worldwide")

    prompt = f"""Generate exactly {count} Adobe Stock content topics for the keyword "{keyword}".
Focus on trending topics in {country_label} over the {time_label}.
Mix of Main Topics ({main_count}) and Sub Topics ({sub_count}).
For each topic return JSON with these fields:
- topic: specific descriptive topic name
- type: "Main Topic" or "Sub Topic"
- demand: "High", "Medium", or "Low" (based on {time_label} trend in {country_label})
- competition: "Low", "Medium", or "High"
- trend_percent: number between -15 and 45 (reflect actual {time_label} trend momentum)
{exclusion_text}

Return ONLY a valid JSON array, no explanation, no markdown, no backticks."""

    async with httpx.AsyncClient(timeout=45) as client:
        resp = await client.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": "meta-llama/llama-4-scout-17b-16e-instruct",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.8,
                "max_tokens": 2800,
            }
        )

    if resp.status_code != 200:
        raise Exception(f"Groq error: {resp.text}")

    content = clean_json_content(resp.json()["choices"][0]["message"]["content"])
    topics = json.loads(content)
    return score_topics(topics)

async def generate_topics_from_image_with_groq(image_b64, mime_type, api_key, count=20, exclude=None):
    main_count = max(1, round(count * 0.25))
    sub_count = max(0, count - main_count)
    exclusion_text = build_exclusion_text(exclude)

    prompt = f"""Look closely at this image — its subject, style, colors, and mood.
Then generate exactly {count} Adobe Stock content topic ideas inspired by what's in the image, useful for a stock contributor creating similar icon sets, illustrations, or graphics around this theme.
Mix of Main Topics ({main_count}) and Sub Topics ({sub_count}).
For each topic return these fields:
- topic: specific descriptive topic name
- type: "Main Topic" or "Sub Topic"
- demand: "High", "Medium", or "Low"
- competition: "Low", "Medium", or "High"
- trend_percent: number between -15 and 45
{exclusion_text}

Return ONLY valid JSON in this exact shape, no explanation, no markdown, no backticks:
{{"theme": "short 3-6 word description of the image", "topics": [{{"topic": "...", "type": "...", "demand": "...", "competition": "...", "trend_percent": 0}}]}}"""

    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": "meta-llama/llama-4-scout-17b-16e-instruct",
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{image_b64}"}}
                    ]
                }],
                "temperature": 0.8,
                "max_tokens": 2800,
            }
        )

    if resp.status_code != 200:
        raise Exception(f"Groq error: {resp.text}")

    content = clean_json_content(resp.json()["choices"][0]["message"]["content"])
    data = json.loads(content)
    theme = data.get("theme", "Image topics")
    topics = score_topics(data.get("topics", []))
    return theme, topics

MOCK_TEMPLATES = [
    ("{kw} Modern Design", "Main Topic", "High", "Low", 28),
    ("{kw} Business Professional", "Sub Topic", "High", "Medium", 12),
    ("{kw} Flat Icon Set", "Sub Topic", "Medium", "Low", 18),
    ("{kw} Colorful Illustration", "Main Topic", "Medium", "Medium", 22),
    ("{kw} Minimal Logo", "Sub Topic", "High", "High", -5),
    ("{kw} Background Pattern", "Main Topic", "Medium", "Low", 15),
    ("{kw} Social Media Template", "Sub Topic", "Medium", "High", -8),
    ("{kw} Infographic Elements", "Sub Topic", "Medium", "Medium", 5),
    ("{kw} Eco Sustainable", "Sub Topic", "Medium", "Low", 30),
    ("{kw} Vintage Retro Style", "Sub Topic", "Low", "Low", 10),
    ("{kw} 3D Render Concept", "Main Topic", "High", "Medium", 35),
    ("{kw} Hand Drawn Sketch", "Sub Topic", "Low", "Low", 8),
    ("{kw} Line Art Outline", "Sub Topic", "Medium", "Low", 14),
    ("{kw} Watercolor Texture", "Sub Topic", "Low", "Low", 6),
    ("{kw} Corporate Banner", "Sub Topic", "Medium", "High", -3),
    ("{kw} Gradient Abstract", "Main Topic", "Medium", "Medium", 20),
    ("{kw} Cartoon Character", "Sub Topic", "High", "Medium", 25),
    ("{kw} Seamless Pattern", "Sub Topic", "Medium", "Low", 17),
    ("{kw} Isometric Illustration", "Main Topic", "High", "Medium", 32),
    ("{kw} Doodle Set", "Sub Topic", "Low", "Low", 4),
    ("{kw} Neon Glow Style", "Sub Topic", "Low", "Medium", 19),
    ("{kw} Paper Cut Style", "Sub Topic", "Low", "Low", 9),
    ("{kw} Festival Celebration", "Main Topic", "High", "Medium", 27),
    ("{kw} Minimal Outline Icon", "Sub Topic", "Medium", "Low", 13),
    ("{kw} Geometric Shapes", "Sub Topic", "Medium", "Medium", 11),
    ("{kw} Children Education Theme", "Sub Topic", "High", "Medium", 21),
    ("{kw} Vector Mascot", "Main Topic", "Medium", "Medium", 16),
    ("{kw} Dark Mode UI Element", "Sub Topic", "Medium", "High", 7),
    ("{kw} Sustainable Packaging", "Sub Topic", "High", "Low", 29),
    ("{kw} Retro 80s Style", "Sub Topic", "Low", "Low", 3),
]

def generate_mock_topics(keyword, count=20, exclude=None):
    exclude = set(exclude or [])
    pool = []
    for tmpl, typ, dem, comp, trend in MOCK_TEMPLATES:
        title = tmpl.replace("{kw}", keyword.title())
        if title not in exclude:
            pool.append((title, typ, dem, comp, trend))
    random.shuffle(pool)

    # If we still don't have enough (e.g. repeated "load more" clicks), generate numbered variants
    extra_round = 2
    while len(pool) < count:
        for tmpl, typ, dem, comp, trend in MOCK_TEMPLATES:
            title = f"{tmpl.replace('{kw}', keyword.title())} Vol {extra_round}"
            if title not in exclude:
                pool.append((title, typ, dem, comp, trend))
        extra_round += 1
        if extra_round > 10:
            break

    raw = [
        {"topic": p[0], "type": p[1], "demand": p[2], "competition": p[3], "trend_percent": p[4]}
        for p in pool[:count]
    ]
    return score_topics(raw)

# ---------------------------------------------------------------------------
# Generate Ideas (categorized, de-duplicated) — used by the "Generate Ideas"
# panel in Topic Finder. Ideas are grouped into Photo / Vector / Icon /
# Illustration / Background buckets and de-duplicated so results don't all
# look like the same idea with a different keyword swapped in.
# ---------------------------------------------------------------------------

IDEA_CATEGORIES = [
    ("Photo", [
        "photography", "photo close up", "isolated on white background", "top view flat lay",
        "studio shot", "lifestyle photo", "professional photo", "high quality photo",
        "macro photography", "editorial photo", "outdoor photo", "black and white photo",
        "with copy space", "candid moment", "aerial view",
    ]),
    ("Vector", [
        "vector illustration", "flat design vector", "hand drawn vector", "watercolor vector",
        "cartoon vector", "seamless pattern vector", "clipart bundle", "sticker set vector",
        "infographic vector", "logo template vector", "line art vector", "isometric vector",
    ]),
    ("Icon", [
        "icon set", "flat icon", "outline icon", "filled icon", "line icon", "gradient icon",
        "colored icon", "3d icon", "glyph icon", "duotone icon", "badge icon", "minimal icon",
    ]),
    ("Illustration", [
        "flat illustration", "isometric illustration", "doodle illustration", "minimal illustration",
        "children illustration", "abstract illustration", "concept illustration", "character illustration",
    ]),
    ("Background", [
        "background pattern", "gradient background", "abstract background", "texture background",
        "banner background", "wallpaper design",
    ]),
]

def generate_mock_categorized_ideas(topic, exclude=None, per_category=6):
    exclude_norm = {e.strip().lower() for e in (exclude or [])}
    categories_out = []
    for cat_name, modifiers in IDEA_CATEGORIES:
        mods = modifiers.copy()
        random.shuffle(mods)
        ideas = []
        seen_local = set()
        round_n = 1
        while len(ideas) < per_category and round_n <= 4:
            for m in mods:
                if len(ideas) >= per_category:
                    break
                suffix = m if round_n == 1 else f"{m} (variation {round_n})"
                title = f"{topic.strip().title()} {suffix}".strip()
                key = title.lower()
                if key in exclude_norm or key in seen_local:
                    continue
                seen_local.add(key)
                trend = round(random.uniform(-10, 40), 1)
                demand = random.choice(["High", "High", "Medium", "Medium", "Medium", "Low"])
                competition = random.choice(["Low", "Low", "Medium", "Medium", "High"])
                score = compute_opportunity_score(demand, trend, competition)
                ideas.append({
                    "topic": title, "category": cat_name, "type": "Sub Topic",
                    "demand": demand, "competition": competition,
                    "trend_percent": trend, "opportunity_score": score,
                })
            round_n += 1
        categories_out.append({"name": cat_name, "ideas": ideas})
    return categories_out

async def generate_categorized_ideas_with_groq(topic, api_key, exclude=None, per_category=6):
    exclusion_text = build_exclusion_text(exclude)
    prompt = f"""You are a stock content strategist. For the Adobe Stock topic "{topic}", generate fresh content ideas a contributor could shoot or design, grouped into these categories: Photo, Vector, Icon, Illustration, Background.

For EACH category generate exactly {per_category} distinct ideas. Each idea must be meaningfully different from the others in that category and across categories — vary the angle, composition, style, mood, or use-case. Do NOT just repeat the same descriptor with minor wording changes, and do not produce near-duplicate ideas.

For each idea return JSON fields:
- topic: specific descriptive topic name (include "{topic}" plus the distinguishing detail)
- demand: "High", "Medium", or "Low"
- competition: "Low", "Medium", or "High"
- trend_percent: number between -15 and 45
{exclusion_text}

Return ONLY valid JSON, no explanation, no markdown, no backticks, in this exact shape:
{{"categories": [{{"name": "Photo", "ideas": [{{"topic": "...", "demand": "...", "competition": "...", "trend_percent": 0}}]}}]}}"""

    async with httpx.AsyncClient(timeout=45) as client:
        resp = await client.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": "meta-llama/llama-4-scout-17b-16e-instruct",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.85,
                "max_tokens": 3200,
            }
        )

    if resp.status_code != 200:
        raise Exception(f"Groq error: {resp.text}")

    content = clean_json_content(resp.json()["choices"][0]["message"]["content"])
    data = json.loads(content)
    categories_out = []
    for cat in data.get("categories", []):
        ideas = []
        for t in cat.get("ideas", []):
            trend_pct = float(t.get("trend_percent", random.uniform(-10, 30)))
            demand = t.get("demand", "Medium")
            competition = t.get("competition", "Medium")
            score = compute_opportunity_score(demand, trend_pct, competition)
            ideas.append({
                "topic": t["topic"], "category": cat.get("name", "Other"), "type": "Sub Topic",
                "demand": demand, "competition": competition,
                "trend_percent": round(trend_pct, 1), "opportunity_score": score,
            })
        categories_out.append({"name": cat.get("name", "Other"), "ideas": ideas})
    return categories_out

@app.post("/api/generate-ideas")
async def generate_ideas(req: GenerateIdeasRequest):
    topic = req.topic.strip()
    if not topic:
        raise HTTPException(status_code=400, detail="Topic is required.")
    per_cat = max(2, min(req.per_category or 6, 10))
    api_key = get_groq_api_key()
    if api_key:
        try:
            categories = await generate_categorized_ideas_with_groq(topic, api_key, exclude=req.exclude, per_category=per_cat)
        except Exception:
            categories = generate_mock_categorized_ideas(topic, exclude=req.exclude, per_category=per_cat)
    else:
        categories = generate_mock_categorized_ideas(topic, exclude=req.exclude, per_category=per_cat)

    # Defensive de-dupe across the whole response + sequential ids
    seen = set()
    next_id = 1
    total = 0
    for cat in categories:
        deduped = []
        for idea in cat["ideas"]:
            key = idea["topic"].strip().lower()
            if key in seen:
                continue
            seen.add(key)
            idea["id"] = next_id
            next_id += 1
            deduped.append(idea)
        cat["ideas"] = deduped
        total += len(deduped)

    categories = [c for c in categories if c["ideas"]]
    return {"topic": topic, "total": total, "categories": categories}

# ---------------------------------------------------------------------------
# Market Analyzer — pick a stock marketplace (Adobe, Shutterstock, iStock,
# Getty, Freepik, Pond5), point it at a keyword, and pull profitable niches
# with long-tail keywords. Tries a real server-side fetch of the live search
# page for signal; falls back gracefully (AI- or template-based) if the site
# blocks automated requests, since these marketplaces can change/limit access
# at any time.
# ---------------------------------------------------------------------------

MARKET_CONFIG = {
    "adobe": {"name": "Adobe Stock", "url": "https://stock.adobe.com/search?k={kw}&search_type=usertyped"},
    "shutterstock": {"name": "Shutterstock", "url": "https://www.shutterstock.com/search/{kw}"},
    "istock": {"name": "iStock", "url": "https://www.istockphoto.com/search/2/image?phrase={kw}"},
    "getty": {"name": "Getty Images", "url": "https://www.gettyimages.com/photos/{kw}"},
    "freepik": {"name": "Freepik", "url": "https://www.freepik.com/search?format=search&query={kw}"},
    "pond5": {"name": "Pond5", "url": "https://www.pond5.com/search?kw={kw}"},
}

def build_market_url(market, keyword):
    cfg = MARKET_CONFIG.get(market)
    if not cfg:
        return None, None
    if market == "shutterstock":
        kw_part = urllib.parse.quote(keyword.strip().replace(" ", "-"))
    else:
        kw_part = urllib.parse.quote_plus(keyword.strip())
    return cfg["url"].format(kw=kw_part), cfg["name"]

def extract_preview_images(soup, base_url, limit=12):
    """Pull likely content-image URLs (thumbnails) out of a fetched search
    results page, resolving relative/lazy-loaded URLs to absolute ones."""
    images = []
    seen = set()

    def add(raw_url):
        if not raw_url or len(images) >= limit:
            return
        raw_url = raw_url.strip()
        if not raw_url or raw_url.startswith("data:"):
            return
        abs_url = urllib.parse.urljoin(base_url, raw_url)
        if abs_url not in seen:
            seen.add(abs_url)
            images.append(abs_url)

    def first_from_srcset(srcset):
        if not srcset:
            return None
        first = srcset.split(",")[0].strip()
        return first.split(" ")[0] if first else None

    for img in soup.find_all("img"):
        if len(images) >= limit:
            break
        candidate = (
            img.get("src") or img.get("data-src") or img.get("data-lazy-src")
            or img.get("data-lazy") or img.get("data-original")
            or first_from_srcset(img.get("srcset") or img.get("data-srcset"))
        )
        add(candidate)

    if len(images) < limit:
        for source in soup.find_all("source"):
            if len(images) >= limit:
                break
            add(first_from_srcset(source.get("srcset") or source.get("data-srcset")))

    return images[:limit]

async def fetch_adobe_stock_images(keyword, limit=12):
    """Fetch real thumbnail images from Adobe Stock.
    Strategy 1: Adobe Stock internal REST API (most reliable, returns JSON).
    Strategy 2: Scrape search page for ftcdn.net CDN image URLs.
    Returns (image_urls, signals, success)."""
    import json as json_mod

    # --- Strategy 1: Adobe Stock internal REST API ---
    try:
        api_url = "https://stock.adobe.com/Rest/Media/1/Search/Files"
        params = {
            "locale": "en_US",
            "search_parameters[words]": keyword,
            "search_parameters[limit]": limit,
            "search_parameters[offset]": 0,
            "search_parameters[filters][content_type:photo]": 1,
            "search_parameters[filters][content_type:illustration]": 1,
            "search_parameters[filters][content_type:vector]": 1,
            "result_columns[]": ["thumbnail_500_url", "thumbnail_url", "title", "id", "keywords"],
        }
        headers = {
            "X-Product": "MainSite",
            "X-API-Key": "stock1",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                          "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            "Referer": "https://stock.adobe.com/",
            "Accept": "application/json",
            "Accept-Language": "en-US,en;q=0.9",
        }
        async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
            resp = await client.get(api_url, params=params, headers=headers)
        if resp.status_code == 200:
            data = resp.json()
            files = data.get("files", [])
            images = []
            signals = []
            for f in files:
                thumb = (f.get("thumbnail_500_url") or f.get("thumbnail_url")
                         or f.get("thumbnail_240_url") or f.get("thumbnail_160_url"))
                if thumb:
                    images.append(thumb)
                title = f.get("title", "")
                if title:
                    signals.append(title)
                for kw in (f.get("keywords") or []):
                    kw_name = kw.get("name", "") if isinstance(kw, dict) else str(kw)
                    if kw_name:
                        signals.append(kw_name)
            if images:
                return images[:limit], list(dict.fromkeys(signals))[:100], True
    except Exception:
        pass

    # --- Strategy 2: Scrape search page for CDN image URLs ---
    if BS4_AVAILABLE:
        try:
            search_url = (
                f"https://stock.adobe.com/search?k={urllib.parse.quote_plus(keyword)}"
                "&search_type=usertyped"
            )
            hdrs2 = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                              "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
                "Referer": "https://stock.adobe.com/",
            }
            async with httpx.AsyncClient(timeout=12, follow_redirects=True, headers=hdrs2) as client:
                resp2 = await client.get(search_url)
            if resp2.status_code == 200:
                soup = BeautifulSoup(resp2.text, "html.parser")
                images = []
                seen = set()
                # Try JSON-LD first
                for script in soup.find_all("script", type="application/ld+json"):
                    try:
                        ld = json_mod.loads(script.string or "")
                        items = ld if isinstance(ld, list) else [ld]
                        for item in items:
                            img_obj = item.get("image") or item.get("thumbnail")
                            url_val = img_obj if isinstance(img_obj, str) else (img_obj or {}).get("url", "")
                            if url_val and url_val not in seen:
                                seen.add(url_val)
                                images.append(url_val)
                    except Exception:
                        pass
                # Grab ftcdn.net CDN thumbnails from img tags
                for img in soup.find_all("img"):
                    for attr in ("src", "data-src", "data-lazy-src", "data-original"):
                        src = img.get(attr, "")
                        if ("ftcdn.net" in src or "adobe.com" in src) and src not in seen:
                            seen.add(src)
                            images.append(src)
                if images:
                    return images[:limit], [], True
        except Exception:
            pass

    return [], [], False


async def fetch_page_data(url):
    """Best-effort fetch of a live marketplace search page to pull lightweight
    text signals (title, meta tags, image alt text, short link text that looks
    like related-search suggestions) AND a handful of real preview image URLs
    so the UI can show what's already on the marketplace for this keyword.
    Returns (signals, preview_images, fetched_ok)."""

    # --- Special fast path for Adobe Stock: use their internal JSON API ---
    if "stock.adobe.com" in url:
        try:
            kw_raw = urllib.parse.urlparse(url)
            qs = urllib.parse.parse_qs(kw_raw.query)
            keyword = (qs.get("k") or qs.get("words") or [""])[0]
            if keyword:
                images, signals, ok = await fetch_adobe_stock_images(keyword, limit=12)
                if ok and images:
                    return signals, images, True
        except Exception:
            pass

    if not BS4_AVAILABLE:
        return [], [], False
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9",
    }
    try:
        async with httpx.AsyncClient(timeout=10, follow_redirects=True, headers=headers) as client:
            resp = await client.get(url)
        if resp.status_code != 200:
            return [], [], False
        soup = BeautifulSoup(resp.text, "html.parser")
        raw_signals = []
        if soup.title and soup.title.string:
            raw_signals.append(soup.title.string)
        for meta_name in ("description", "keywords"):
            tag = soup.find("meta", attrs={"name": meta_name})
            if tag and tag.get("content"):
                raw_signals.append(tag["content"])
        alt_count = 0
        for img in soup.find_all("img"):
            alt = (img.get("alt") or "").strip()
            if alt and len(alt) > 2 and alt_count < 60:
                raw_signals.append(alt)
                alt_count += 1
        link_count = 0
        for a in soup.find_all("a"):
            text = (a.get_text() or "").strip()
            if text and 2 <= len(text.split()) <= 6 and link_count < 40:
                raw_signals.append(text)
                link_count += 1
        seen = set()
        cleaned = []
        for s in raw_signals:
            s = " ".join(str(s).split())
            key = s.lower()
            if s and 0 < len(s) < 120 and key not in seen:
                seen.add(key)
                cleaned.append(s)
        preview_images = extract_preview_images(soup, url, limit=12)
        return cleaned[:150], preview_images, True
    except Exception:
        return [], [], False

async def generate_niches_with_groq(market_name, keyword, signals, api_key, count=5, kw_per_niche=16):
    signals_text = ""
    if signals:
        signals_text = (
            f"\nReal signals extracted from the live {market_name} search results page for this "
            "keyword (page title, meta tags, image alt text, suggested searches) — ground your "
            "niches in these where relevant:\n- " + "\n- ".join(signals[:120])
        )

    prompt = f"""You are a stock marketplace research analyst. A contributor searched "{keyword}" on {market_name}.{signals_text}

Identify exactly {count} distinct, profitable content niches related to "{keyword}" that a stock contributor could target on {market_name}. For each niche provide:
- name: short specific niche name (2-5 words)
- demand: "High", "Medium", or "Low"
- rationale: one short sentence on why this niche is profitable
- keywords: a list of exactly {kw_per_niche} long-tail search keyword phrases real buyers would type, specific to this niche (lowercase, no hashtags, no duplicates)

Return ONLY valid JSON, no explanation, no markdown, no backticks, in this exact shape:
{{"niches": [{{"name": "...", "demand": "...", "rationale": "...", "keywords": ["...", "..."]}}]}}"""

    async with httpx.AsyncClient(timeout=50) as client:
        resp = await client.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": "meta-llama/llama-4-scout-17b-16e-instruct",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7,
                "max_tokens": 3500,
            }
        )

    if resp.status_code != 200:
        raise Exception(f"Groq error: {resp.text}")

    content = clean_json_content(resp.json()["choices"][0]["message"]["content"])
    data = json.loads(content)
    return data.get("niches", [])

def generate_mock_niches(keyword, count=5, kw_per_niche=16):
    kw = keyword.strip().lower()
    base_angles = [
        ("{kw} concept", "High", f"High buyer demand for clean, conceptual visuals of {kw}."),
        ("{kw} at home", "High", f"Lifestyle and at-home imagery for {kw} performs consistently well."),
        ("{kw} for business", "Medium", f"Corporate and business-themed {kw} content is steadily requested."),
        ("{kw} flat lay & top view", "Medium", f"Flat lay compositions of {kw} are popular for social and editorial use."),
        ("{kw} icon & vector set", "Medium", f"Designers frequently search for simple icon/vector sets around {kw}."),
        ("{kw} background & texture", "Low", f"Background and texture variants of {kw} have lower competition."),
        ("{kw} black and white", "Low", f"Monochrome takes on {kw} are a smaller but distinct niche."),
    ]
    random.shuffle(base_angles)
    suffixes = [
        "closeup shot", "isolated on white background", "top view flat lay", "in natural light",
        "with copy space", "minimal style", "studio photography", "hand holding", "top down view",
        "aerial view", "macro detail", "with text space", "in a modern office", "outdoors",
        "on wooden table", "with soft shadow", "in pastel colors", "high resolution",
        "editorial style", "candid moment",
    ]
    niches = []
    for tmpl, demand, rationale in base_angles[:count]:
        name = tmpl.format(kw=keyword.strip().title())
        shuffled = suffixes.copy()
        random.shuffle(shuffled)
        keywords, seen = [], set()
        for s in shuffled:
            phrase = f"{kw} {s}"
            if phrase not in seen:
                seen.add(phrase)
                keywords.append(phrase)
            if len(keywords) >= kw_per_niche:
                break
        niches.append({"name": name, "demand": demand, "rationale": rationale, "keywords": keywords})
    return niches

@app.post("/api/market-analyze")
async def market_analyze(req: MarketAnalyzeRequest):
    keyword = req.keyword.strip()
    if not keyword:
        raise HTTPException(status_code=400, detail="Keyword is required.")

    page_url, market_name = build_market_url(req.market, keyword)
    if not page_url:
        raise HTTPException(status_code=400, detail="Unsupported market.")

    signals, preview_images, fetched_live = await fetch_page_data(page_url)

    count = max(1, min(req.niche_count or 5, 8))
    kw_per_niche = max(5, min(req.keywords_per_niche or 16, 25))

    api_key = get_groq_api_key()
    used_ai = False
    if api_key:
        try:
            niches = await generate_niches_with_groq(market_name, keyword, signals, api_key, count=count, kw_per_niche=kw_per_niche)
            used_ai = True
        except Exception:
            niches = generate_mock_niches(keyword, count=count, kw_per_niche=kw_per_niche)
    else:
        niches = generate_mock_niches(keyword, count=count, kw_per_niche=kw_per_niche)

    clean_niches = []
    for i, n in enumerate(niches):
        kws, seen_kw = [], set()
        for k in (n.get("keywords") or []):
            k = " ".join(str(k).split()).lower()
            if k and k not in seen_kw:
                seen_kw.add(k)
                kws.append(k)
        clean_niches.append({
            "id": i + 1,
            "name": n.get("name", f"{keyword} niche {i + 1}"),
            "demand": n.get("demand", "Medium"),
            "rationale": n.get("rationale", ""),
            "keywords": kws,
        })

    conn = get_db()
    c = conn.cursor()
    c.execute(
        "INSERT INTO search_history (keyword, results_count) VALUES (?, ?)",
        (f"[{market_name}] {keyword}", sum(len(n["keywords"]) for n in clean_niches)),
    )
    conn.commit()
    conn.close()

    return {
        "market": req.market,
        "market_name": market_name,
        "keyword": keyword,
        "page_url": page_url,
        "fetched_live": fetched_live,
        "preview_images": preview_images,
        "used_ai": used_ai,
        "niches": clean_niches,
    }

@app.post("/api/search")
async def search_topics(req: SearchRequest):
    count = max(1, min(req.max_results or 20, 40))
    api_key = get_groq_api_key()
    if api_key:
        try:
            topics = await generate_topics_with_groq(req.keyword, api_key, count=count, exclude=req.exclude_topics, time_range=req.time_range or "past_24_hours", country=req.country or "worldwide")
        except Exception:
            topics = generate_mock_topics(req.keyword, count=count, exclude=req.exclude_topics)
    else:
        topics = generate_mock_topics(req.keyword, count=count, exclude=req.exclude_topics)

    if req.topic_type and req.topic_type != "all":
        label = "Main Topic" if req.topic_type == "main" else "Sub Topic"
        topics = [t for t in topics if t["type"] == label]

    conn = get_db()
    c = conn.cursor()
    c.execute("INSERT INTO search_history (keyword, results_count) VALUES (?, ?)", (req.keyword, len(topics)))
    conn.commit()
    conn.close()

    return {"keyword": req.keyword, "total": len(topics), "topics": topics}

@app.post("/api/analyze-image")
async def analyze_image(
    file: UploadFile = File(...),
    max_results: int = Form(20),
    exclude_topics: Optional[str] = Form(None),
):
    api_key = get_groq_api_key()
    if not api_key:
        raise HTTPException(status_code=400, detail="Add a Groq API key in API Settings to use image analysis.")

    contents = await file.read()
    if len(contents) > 8 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Image too large. Please use an image under 8MB.")

    mime_type = file.content_type or "image/png"
    image_b64 = base64.b64encode(contents).decode("utf-8")

    try:
        exclude = json.loads(exclude_topics) if exclude_topics else []
    except Exception:
        exclude = []

    count = max(1, min(max_results or 20, 40))

    try:
        theme, topics = await generate_topics_from_image_with_groq(
            image_b64, mime_type, api_key, count=count, exclude=exclude
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Image analysis failed: {str(e)}")

    conn = get_db()
    c = conn.cursor()
    c.execute("INSERT INTO search_history (keyword, results_count) VALUES (?, ?)", (f"[Image] {theme}", len(topics)))
    conn.commit()
    conn.close()

    return {"keyword": theme, "total": len(topics), "topics": topics, "source": "image"}

@app.get("/api/saved-topics")
def get_saved_topics():
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT * FROM saved_topics ORDER BY saved_at DESC")
    rows = c.fetchall()
    conn.close()
    return {"topics": [dict(r) for r in rows]}

@app.post("/api/saved-topics")
def save_topic(req: SaveTopicRequest):
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT id FROM saved_topics WHERE topic = ? AND keyword = ?", (req.topic, req.keyword))
    if c.fetchone():
        conn.close()
        raise HTTPException(status_code=409, detail="Topic already saved")
    c.execute("""
        INSERT INTO saved_topics (topic, type, demand, competition, trend_percent, opportunity_score, keyword)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (req.topic, req.type, req.demand, req.competition, req.trend_percent, req.opportunity_score, req.keyword))
    conn.commit()
    new_id = c.lastrowid
    conn.close()
    return {"success": True, "id": new_id}

@app.delete("/api/saved-topics/{topic_id}")
def delete_saved_topic(topic_id: int):
    conn = get_db()
    c = conn.cursor()
    c.execute("DELETE FROM saved_topics WHERE id = ?", (topic_id,))
    conn.commit()
    conn.close()
    return {"success": True}

@app.get("/api/history")
def get_history():
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT * FROM search_history ORDER BY searched_at DESC LIMIT 50")
    rows = c.fetchall()
    conn.close()
    return {"history": [dict(r) for r in rows]}

@app.delete("/api/history")
def clear_history():
    conn = get_db()
    c = conn.cursor()
    c.execute("DELETE FROM search_history")
    conn.commit()
    conn.close()
    return {"success": True}

@app.get("/api/health")
def health():
    return {"status": "ok"}
