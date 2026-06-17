from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import sqlite3
import json
import httpx
import random
import base64
from datetime import datetime

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
    time_range: Optional[str] = "past_12_months"
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

async def generate_topics_with_groq(keyword, api_key, count=20, exclude=None):
    main_count = max(1, round(count * 0.25))
    sub_count = max(0, count - main_count)
    exclusion_text = build_exclusion_text(exclude)

    prompt = f"""Generate exactly {count} Adobe Stock content topics for the keyword "{keyword}".
Mix of Main Topics ({main_count}) and Sub Topics ({sub_count}).
For each topic return JSON with these fields:
- topic: specific descriptive topic name
- type: "Main Topic" or "Sub Topic"
- demand: "High", "Medium", or "Low"
- competition: "Low", "Medium", or "High"
- trend_percent: number between -15 and 45
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

@app.post("/api/search")
async def search_topics(req: SearchRequest):
    count = max(1, min(req.max_results or 20, 40))
    api_key = get_groq_api_key()
    if api_key:
        try:
            topics = await generate_topics_with_groq(req.keyword, api_key, count=count, exclude=req.exclude_topics)
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
